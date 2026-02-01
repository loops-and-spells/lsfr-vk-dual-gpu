#include "context.hpp"
#include "config/config.hpp"
#include "common/exception.hpp"
#include "extract/extract.hpp"
#include "extract/trans.hpp"
#include "utils/utils.hpp"
#include "hooks.hpp"
#include "layer.hpp"

#include <vulkan/vulkan_core.h>
#include <lsfg_3_1.hpp>
#include <lsfg_3_1p.hpp>

#include <filesystem>
#include <exception>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <array>

LsContext::LsContext(const Hooks::DeviceInfo& info, VkSwapchainKHR swapchain,
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages)
        : swapchain(swapchain), swapchainImages(swapchainImages),
          extent(extent) {
    // get updated configuration
    auto& conf = Config::activeConf;
    if (!conf.config_file.empty()
            && (
                    !std::filesystem::exists(conf.config_file)
                  || conf.timestamp != std::filesystem::last_write_time(conf.config_file)
            )) {
        std::cerr << "lsfg-vk: Rereading configuration, as it is no longer valid.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // reread configuration
        const std::string file = Utils::getConfigFile();
        const auto name = Utils::getProcessName();
        try {
            Config::updateConfig(file);
            conf = Config::getConfig(name);
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: Failed to update configuration, continuing using old:\n";
            std::cerr << "- " << e.what() << '\n';
        }

        LSFG_3_1P::finalize();
        LSFG_3_1::finalize();

        // print config
        std::cerr << "lsfg-vk: Reloaded configuration for " << name.second << ":\n";
        if (!conf.dll.empty()) std::cerr << "  Using DLL from: " << conf.dll << '\n';
        std::cerr << "  Multiplier: " << conf.multiplier << '\n';
        std::cerr << "  Flow Scale: " << conf.flowScale << '\n';
        std::cerr << "  Performance Mode: " << (conf.performance ? "Enabled" : "Disabled") << '\n';
        std::cerr << "  HDR Mode: " << (conf.hdr ? "Enabled" : "Disabled") << '\n';
        if (conf.gpu_secondary.has_value())
            std::cerr << "  Dual-GPU: Secondary GPU at PCI " << conf.gpu_secondary.value() << '\n';
        if (conf.e_present != 2) std::cerr << "  ! Present Mode: " << conf.e_present << '\n';

        if (conf.multiplier <= 1) return;
    }
    // we could take the format from the swapchain,
    // but honestly this is safer.
    const VkFormat format = conf.hdr
        ? VK_FORMAT_R8G8B8A8_UNORM
        : VK_FORMAT_R16G16B16A16_SFLOAT;

    // prepare textures for lsfg
    std::array<int, 2> fds{};
    this->frame_0 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(0));
    this->frame_1 = Mini::Image(info.device, info.physicalDevice,
        extent, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        &fds.at(1));

    std::vector<int> outFds(conf.multiplier - 1);
    for (size_t i = 0; i < (conf.multiplier - 1); ++i)
        this->out_n.emplace_back(info.device, info.physicalDevice,
            extent, format,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
            &outFds.at(i));

    // initialize lsfg
    auto* lsfgInitialize = LSFG_3_1::initialize;
    auto* lsfgCreateContext = LSFG_3_1::createContext;
    auto* lsfgDeleteContext = LSFG_3_1::deleteContext;
    if (conf.performance) {
        lsfgInitialize = LSFG_3_1P::initialize;
        lsfgCreateContext = LSFG_3_1P::createContext;
        lsfgDeleteContext = LSFG_3_1P::deleteContext;
    }

    setenv("DISABLE_LSFG", "1", 1); // NOLINT

    // Determine which GPU to use for frame generation
    uint64_t targetDeviceUUID = Utils::getDeviceUUID(info.physicalDevice);
    std::cerr << "lsfg-vk: Primary GPU UUID: 0x" << std::hex << targetDeviceUUID << std::dec << '\n';

    // Check if we need staging mode (dual-GPU with different devices)
    bool needsStaging = false;
    if (conf.gpu_secondary.has_value()) {
        std::cerr << "lsfg-vk: Looking for secondary GPU at PCI " << conf.gpu_secondary.value() << '\n';
        auto secondaryUUID = Utils::findDeviceByPci(conf.gpu_secondary.value());
        if (secondaryUUID.has_value()) {
            std::cerr << "lsfg-vk: Secondary GPU UUID: 0x" << std::hex << secondaryUUID.value() << std::dec << '\n';
            targetDeviceUUID = secondaryUUID.value();
            needsStaging = true; // Cross-device requires staging
            std::cerr << "lsfg-vk: DUAL-GPU ACTIVE - Using STAGING MODE for cross-device transfer\n";
        } else {
            std::cerr << "lsfg-vk: Warning: Secondary GPU not found at PCI " << conf.gpu_secondary.value() << ", using primary GPU\n";
        }
    } else {
        std::cerr << "lsfg-vk: No secondary GPU configured, using primary GPU for frame generation\n";
    }

    this->stagingMode = needsStaging;
    this->framegenScale = conf.framegen_scale;
    this->framegenUpscale = conf.framegen_upscale;
    this->framegenDebug = conf.framegen_debug;
    std::cerr << "lsfg-vk: Final target device UUID for LSFG: 0x" << std::hex << targetDeviceUUID << std::dec << '\n';
    std::cerr << "lsfg-vk: Staging mode: " << (stagingMode ? "ENABLED" : "disabled") << '\n';

    // Calculate scaled extent for framegen (only applies in staging mode)
    this->scaledExtent = extent;
    std::cerr << "lsfg-vk: DEBUG: stagingMode=" << stagingMode << ", framegen_scale=" << conf.framegen_scale
              << ", framegen_debug=" << conf.framegen_debug << ", condition=" << (stagingMode && conf.framegen_scale < 1.0F) << '\n';
    // Also write to file for easier debugging
    {
        std::ofstream logfile("/tmp/lsfg-vk-debug.log", std::ios::app);
        logfile << "DEBUG: stagingMode=" << stagingMode << ", framegen_scale=" << conf.framegen_scale
                << ", framegen_debug=" << conf.framegen_debug << ", framegen_upscale=" << conf.framegen_upscale
                << ", condition=" << (stagingMode && conf.framegen_scale < 1.0F) << '\n';
    }
    if (stagingMode && conf.framegen_scale < 1.0F) {
        this->scaledExtent.width = static_cast<uint32_t>(extent.width * conf.framegen_scale);
        this->scaledExtent.height = static_cast<uint32_t>(extent.height * conf.framegen_scale);
        // Ensure dimensions are at least 64 and divisible by 2
        this->scaledExtent.width = std::max(64U, (this->scaledExtent.width / 2) * 2);
        this->scaledExtent.height = std::max(64U, (this->scaledExtent.height / 2) * 2);
        std::cerr << "lsfg-vk: Scaled framegen resolution: " << scaledExtent.width << "x" << scaledExtent.height
                  << " (scale=" << conf.framegen_scale << ")\n";
    }

    // Finalize any existing LSFG instance first (singleton may have been initialized with wrong device)
    auto* lsfgFinalize = LSFG_3_1::finalize;
    if (conf.performance) {
        lsfgFinalize = LSFG_3_1P::finalize;
    }
    lsfgFinalize();

    lsfgInitialize(
        targetDeviceUUID,
        conf.hdr, 1.0F / conf.flowScale, conf.multiplier - 1,
        [](const std::string& name) {
            auto dxbc = Extract::getShader(name);
            auto spirv = Extract::translateShader(dxbc);
            return spirv;
        }
    );

    if (stagingMode) {
        // Create staging buffers at SCALED resolution (smaller = faster PCIe transfer)
        this->inStaging_0 = Mini::StagingBuffer(info.device, info.physicalDevice, scaledExtent, format);
        this->inStaging_1 = Mini::StagingBuffer(info.device, info.physicalDevice, scaledExtent, format);
        for (size_t i = 0; i < (conf.multiplier - 1); ++i)
            this->outStagingN.emplace_back(info.device, info.physicalDevice, scaledExtent, format);

        // Create scaled intermediate images on primary GPU (for down/upsampling)
        if (conf.framegen_scale < 1.0F) {
            this->scaledInput_0 = Mini::Image(info.device, info.physicalDevice, scaledExtent, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            this->scaledInput_1 = Mini::Image(info.device, info.physicalDevice, scaledExtent, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            for (size_t i = 0; i < (conf.multiplier - 1); ++i)
                this->scaledOutputN.emplace_back(info.device, info.physicalDevice, scaledExtent, format,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            std::cerr << "lsfg-vk: Created " << (2 + scaledOutputN.size()) << " scaled intermediate images\n";
        }

        // Create context in staging mode at SCALED resolution
        auto* lsfgCreateContextStaging = LSFG_3_1::createContextStaging;
        auto* lsfgGetStagingPointers = LSFG_3_1::getStagingPointers;
        // Note: LSFG_3_1P doesn't have staging mode yet, fall back to LSFG_3_1
        // if (conf.performance) { ... }

        this->lsfgCtxId = std::shared_ptr<int32_t>(
            new int32_t(lsfgCreateContextStaging(scaledExtent, format)),
            [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
                lsfgDeleteContext(*id);
            }
        );

        // Get LSFG's staging pointers
        lsfgGetStagingPointers(*this->lsfgCtxId, &this->lsfgInPtr0, &this->lsfgInPtr1, this->lsfgOutPtrs);
        std::cerr << "lsfg-vk: LSFG staging pointers: in0=" << lsfgInPtr0
                  << ", in1=" << lsfgInPtr1 << ", outCount=" << lsfgOutPtrs.size() << '\n';
    } else {
        // Standard FD mode
        this->lsfgCtxId = std::shared_ptr<int32_t>(
            new int32_t(lsfgCreateContext(fds.at(0), fds.at(1), outFds, extent, format)),
            [lsfgDeleteContext = lsfgDeleteContext](const int32_t* id) {
                lsfgDeleteContext(*id);
            }
        );
    }

    unsetenv("DISABLE_LSFG"); // NOLINT

    // prepare render passes
    this->cmdPool = Mini::CommandPool(info.device, info.queue.first);
    for (size_t i = 0; i < 8; i++) {
        auto& pass = this->passInfos.at(i);
        pass.renderSemaphores.resize(conf.multiplier - 1);
        pass.acquireSemaphores.resize(conf.multiplier - 1);
        pass.postCopyBufs.resize(conf.multiplier - 1);
        pass.postCopySemaphores.resize(conf.multiplier - 1);
        pass.prevPostCopySemaphores.resize(conf.multiplier - 1);
    }
}

VkResult LsContext::present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    if (this->stagingMode) {
        return presentStagingMode(info, pNext, queue, gameRenderSemaphores, presentIdx);
    }
    return presentFdMode(info, pNext, queue, gameRenderSemaphores, presentIdx);
}

VkResult LsContext::presentFdMode(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

    // 1. copy swapchain image to frame_0/frame_1
    int preCopySemaphoreFd{};
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device, &preCopySemaphoreFd);
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    Utils::copyImage(pass.preCopyBuf.handle(),
        this->swapchainImages.at(presentIdx),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        this->extent.width, this->extent.height,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        true, false);

    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());
    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() });

    // 2. render intermediary frames
    std::vector<int> renderSemaphoreFds(conf.multiplier - 1);
    for (size_t i = 0; i < (conf.multiplier - 1); ++i)
        pass.renderSemaphores.at(i) = Mini::Semaphore(info.device, &renderSemaphoreFds.at(i));

    if (conf.performance)
        LSFG_3_1P::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);
    else
        LSFG_3_1::presentContext(*this->lsfgCtxId,
            preCopySemaphoreFd,
            renderSemaphoreFds);

    for (size_t i = 0; i < (conf.multiplier - 1); i++) {
        // 3. acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        // 4. copy output image to swapchain image
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        Utils::copyImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(),
            this->swapchainImages.at(imageIdx),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            false, true);

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle(),
              pass.renderSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        // 5. present swapchain image
        std::vector<VkSemaphore> waitSemaphores{ pass.postCopySemaphores.at(i).handle() };
        if (i != 0) waitSemaphores.emplace_back(pass.prevPostCopySemaphores.at(i - 1).handle());

        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr, // only set on first present
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    }

    // 6. present actual next frame
    VkSemaphore lastPrevPostCopySemaphore =
        pass.prevPostCopySemaphores.at(conf.multiplier - 1 - 1).handle();
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &lastPrevPostCopySemaphore,
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &presentIdx,
    };
    auto res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present swapchain image");

    this->frameIdx++;
    return res;
}

LsContext::~LsContext() {
    if (this->stagingMode && this->hasPreviousOutput) {
        try {
            LSFG_3_1::waitStagingOutput(*this->lsfgCtxId);
        } catch (...) {}
    }
}

VkResult LsContext::presentStagingMode(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

    // ---- Step 1: Submit swapchain → staging copy (non-blocking GPU submit) ----
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device);
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    auto& currentStaging = this->frameIdx % 2 == 0 ? this->inStaging_0 : this->inStaging_1;

    if (this->framegenScale < 1.0F) {
        // SCALED MODE: swapchain → scaledInput (downsample) → staging
        auto& scaledInput = this->frameIdx % 2 == 0 ? this->scaledInput_0 : this->scaledInput_1;

        Utils::blitImage(pass.preCopyBuf.handle(),
            this->swapchainImages.at(presentIdx), this->extent, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            scaledInput.handle(), this->scaledExtent,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_FILTER_LINEAR);

        // Transition swapchain image back to PRESENT_SRC_KHR for deferred present
        const VkImageMemoryBarrier presentBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = this->swapchainImages.at(presentIdx),
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };
        Layer::ovkCmdPipelineBarrier(pass.preCopyBuf.handle(),
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
            0, nullptr, 0, nullptr, 1, &presentBarrier);

        currentStaging.copyFromImage(pass.preCopyBuf.handle(),
            scaledInput.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    } else {
        // FULL-RES MODE: swapchain → frame_0/frame_1 → staging
        Utils::copyImage(pass.preCopyBuf.handle(),
            this->swapchainImages.at(presentIdx),
            this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            true, false);

        currentStaging.copyFromImage(pass.preCopyBuf.handle(),
            this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    pass.preCopyBuf.end();

    // Create fence for input copy completion
    this->inputCopyFence = Mini::Fence(info.device);

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());

    pass.preCopyBuf.submit(info.queue.second, this->inputCopyFence.handle(),
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() });

    // ---- Frame 0: synchronous startup ----
    if (!this->hasPreviousOutput) {
        // Wait for input copy to finish
        (void)this->inputCopyFence.wait(info.device);

        // memcpy staging → LSFG input
        void* lsfgInPtr = this->frameIdx % 2 == 0 ? this->lsfgInPtr0 : this->lsfgInPtr1;
        if (lsfgInPtr && currentStaging.data())
            std::memcpy(lsfgInPtr, currentStaging.data(), currentStaging.size());

        // Submit async LSFG generation (non-blocking)
        LSFG_3_1::submitStagingFrame(*this->lsfgCtxId);

        // Present the original frame
        VkSemaphore preCopySem = pass.preCopySemaphores.at(0).handle();
        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = pNext,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &preCopySem,
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &presentIdx,
        };
        auto res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");

        this->hasPreviousOutput = true;
        this->prevPresentIdx = presentIdx;
        this->frameIdx++;
        return res;
    }

    // ---- Frame 1+: pipelined steady state ----
    // Step 2: Wait for previous LSFG generation to complete
    LSFG_3_1::waitStagingOutput(*this->lsfgCtxId);

    // Step 3: memcpy LSFG output → layer outStaging buffers
    for (size_t i = 0; i < (conf.multiplier - 1) && i < this->lsfgOutPtrs.size(); i++) {
        if (this->lsfgOutPtrs.at(i) && this->outStagingN.at(i).data()) {
            std::memcpy(this->outStagingN.at(i).data(), this->lsfgOutPtrs.at(i),
                this->outStagingN.at(i).size());
        }
    }

    // Step 4: Present generated frames
    for (size_t i = 0; i < (conf.multiplier - 1) && i < this->lsfgOutPtrs.size(); i++) {
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        if (this->framegenScale < 1.0F) {
            auto& scaledOutput = this->scaledOutputN.at(i);

            this->outStagingN.at(i).copyToImage(pass.postCopyBufs.at(i).handle(),
                scaledOutput.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            if (this->framegenDebug) {
                const VkImageMemoryBarrier clearBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .image = this->swapchainImages.at(imageIdx),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .levelCount = 1,
                        .layerCount = 1
                    }
                };
                Layer::ovkCmdPipelineBarrier(pass.postCopyBufs.at(i).handle(),
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    0, nullptr, 0, nullptr, 1, &clearBarrier);

                const VkClearColorValue magenta = { .float32 = {1.0f, 0.0f, 1.0f, 1.0f} };
                const VkImageSubresourceRange range = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1
                };
                Layer::ovkCmdClearColorImage(pass.postCopyBufs.at(i).handle(),
                    this->swapchainImages.at(imageIdx), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    &magenta, 1, &range);
            }

            VkFilter filter = this->framegenUpscale ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            VkExtent2D dstExtent = this->extent;
            if (this->framegenDebug) {
                dstExtent.width = this->extent.width > 40 ? this->extent.width - 40 : this->extent.width;
                dstExtent.height = this->extent.height > 40 ? this->extent.height - 40 : this->extent.height;
            }
            Utils::blitImage(pass.postCopyBufs.at(i).handle(),
                scaledOutput.handle(), this->scaledExtent, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                this->swapchainImages.at(imageIdx), dstExtent,
                this->framegenDebug ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                filter);
        } else {
            this->outStagingN.at(i).copyToImage(pass.postCopyBufs.at(i).handle(),
                this->out_n.at(i).handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            Utils::copyImage(pass.postCopyBufs.at(i).handle(),
                this->out_n.at(i).handle(),
                this->swapchainImages.at(imageIdx),
                this->extent.width, this->extent.height,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                false, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        std::vector<VkSemaphore> waitSemaphores{ pass.postCopySemaphores.at(i).handle() };
        if (i != 0) waitSemaphores.emplace_back(pass.prevPostCopySemaphores.at(i - 1).handle());

        const VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = i == 0 ? pNext : nullptr,
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .swapchainCount = 1,
            .pSwapchains = &this->swapchain,
            .pImageIndices = &imageIdx,
        };
        res = Layer::ovkQueuePresentKHR(queue, &presentInfo);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to present swapchain image");
    }

    // Step 5: Present the previous frame's original
    VkSemaphore lastPrevPostCopySemaphore =
        pass.prevPostCopySemaphores.at(conf.multiplier - 1 - 1).handle();
    uint32_t prevIdx = this->prevPresentIdx;
    const VkPresentInfoKHR origPresentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &lastPrevPostCopySemaphore,
        .swapchainCount = 1,
        .pSwapchains = &this->swapchain,
        .pImageIndices = &prevIdx,
    };
    auto res = Layer::ovkQueuePresentKHR(queue, &origPresentInfo);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw LSFG::vulkan_error(res, "Failed to present original swapchain image");

    // Step 6: Wait for input copy fence (likely already done since step 2 blocked)
    (void)this->inputCopyFence.wait(info.device);

    // Step 7: memcpy inStaging → LSFG input
    void* lsfgInPtr = this->frameIdx % 2 == 0 ? this->lsfgInPtr0 : this->lsfgInPtr1;
    if (lsfgInPtr && currentStaging.data())
        std::memcpy(lsfgInPtr, currentStaging.data(), currentStaging.size());

    // Step 8: Submit async LSFG generation (non-blocking)
    LSFG_3_1::submitStagingFrame(*this->lsfgCtxId);

    // Step 9: Store state for next frame
    this->prevPresentIdx = presentIdx;
    this->frameIdx++;
    return res;
}
