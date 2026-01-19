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
    std::cerr << "lsfg-vk: Final target device UUID for LSFG: 0x" << std::hex << targetDeviceUUID << std::dec << '\n';
    std::cerr << "lsfg-vk: Staging mode: " << (stagingMode ? "ENABLED" : "disabled") << '\n';

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
        // Create staging buffers on primary GPU (layer side)
        this->inStaging_0 = Mini::StagingBuffer(info.device, info.physicalDevice, extent, format);
        this->inStaging_1 = Mini::StagingBuffer(info.device, info.physicalDevice, extent, format);
        for (size_t i = 0; i < (conf.multiplier - 1); ++i)
            this->outStagingN.emplace_back(info.device, info.physicalDevice, extent, format);

        // Create context in staging mode
        auto* lsfgCreateContextStaging = LSFG_3_1::createContextStaging;
        auto* lsfgGetStagingPointers = LSFG_3_1::getStagingPointers;
        // Note: LSFG_3_1P doesn't have staging mode yet, fall back to LSFG_3_1
        // if (conf.performance) { ... }

        this->lsfgCtxId = std::shared_ptr<int32_t>(
            new int32_t(lsfgCreateContextStaging(extent, format)),
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

VkResult LsContext::presentStagingMode(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx) {
    static auto lastFpsTime = std::chrono::steady_clock::now();
    static uint64_t fpsFrameCount = 0;
    static double totalWait1 = 0, totalMemcpy1 = 0, totalLsfg = 0, totalMemcpy2 = 0, totalPresent = 0;
    fpsFrameCount++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsTime).count();
    if (elapsed >= 2000) {
        double fps = fpsFrameCount * 1000.0 / elapsed;
        std::cerr << "lsfg-vk [STAGING]: " << fps << " FPS | wait1=" << (totalWait1/fpsFrameCount)
                  << "ms memcpy1=" << (totalMemcpy1/fpsFrameCount) << "ms lsfg=" << (totalLsfg/fpsFrameCount)
                  << "ms memcpy2=" << (totalMemcpy2/fpsFrameCount) << "ms present=" << (totalPresent/fpsFrameCount) << "ms\n";
        fpsFrameCount = 0;
        totalWait1 = totalMemcpy1 = totalLsfg = totalMemcpy2 = totalPresent = 0;
        lastFpsTime = now;
    }
    auto t0 = std::chrono::steady_clock::now();

    const auto& conf = Config::activeConf;
    auto& pass = this->passInfos.at(this->frameIdx % 8);

    // 1. copy swapchain image to frame_0/frame_1, then to staging buffer (on primary GPU)
    pass.preCopySemaphores.at(0) = Mini::Semaphore(info.device);
    pass.preCopySemaphores.at(1) = Mini::Semaphore(info.device);
    pass.preCopyBuf = Mini::CommandBuffer(info.device, this->cmdPool);
    pass.preCopyBuf.begin();

    Utils::copyImage(pass.preCopyBuf.handle(),
        this->swapchainImages.at(presentIdx),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        this->extent.width, this->extent.height,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        true, false);

    // Copy to staging buffer for cross-device transfer
    // After Utils::copyImage, the dest image is in TRANSFER_DST_OPTIMAL layout
    auto& currentStaging = this->frameIdx % 2 == 0 ? this->inStaging_0 : this->inStaging_1;
    currentStaging.copyFromImage(pass.preCopyBuf.handle(),
        this->frameIdx % 2 == 0 ? this->frame_0.handle() : this->frame_1.handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    pass.preCopyBuf.end();

    std::vector<VkSemaphore> gameRenderSemaphores2 = gameRenderSemaphores;
    if (this->frameIdx > 0)
        gameRenderSemaphores2.emplace_back(this->passInfos.at((this->frameIdx - 1) % 8)
            .preCopySemaphores.at(1).handle());

    pass.preCopyBuf.submit(info.queue.second,
        gameRenderSemaphores2,
        { pass.preCopySemaphores.at(0).handle(),
          pass.preCopySemaphores.at(1).handle() });

    // Wait for all GPU work to complete before CPU memcpy
    // This is a synchronization point - not optimal but ensures correctness
    Layer::ovkQueueWaitIdle(info.queue.second);
    auto t1 = std::chrono::steady_clock::now();
    totalWait1 += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Copy from layer staging to LSFG staging (CPU memcpy)
    // Use streaming stores for better performance with GPU memory
    void* lsfgInPtr = this->frameIdx % 2 == 0 ? this->lsfgInPtr0 : this->lsfgInPtr1;
    if (lsfgInPtr && currentStaging.data()) {
        const size_t size = currentStaging.size();
        const auto* src = static_cast<const char*>(currentStaging.data());
        auto* dst = static_cast<char*>(lsfgInPtr);
        // Use regular memcpy - streaming stores don't help much for reads from GPU memory
        std::memcpy(dst, src, size);
    }
    auto t2 = std::chrono::steady_clock::now();
    totalMemcpy1 += std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 3. Run frame generation on secondary GPU
    // Note: presentContextStaging is synchronous
    LSFG_3_1::presentContextStaging(*this->lsfgCtxId);
    auto t3 = std::chrono::steady_clock::now();
    totalLsfg += std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 4. Copy generated frames from LSFG staging to layer staging, then to output images
    auto t4 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < (conf.multiplier - 1) && i < this->lsfgOutPtrs.size(); i++) {
        // CPU memcpy from LSFG staging to our staging
        if (this->lsfgOutPtrs.at(i) && this->outStagingN.at(i).data()) {
            std::memcpy(this->outStagingN.at(i).data(), this->lsfgOutPtrs.at(i),
                this->outStagingN.at(i).size());
        }
    }
    auto t5 = std::chrono::steady_clock::now();
    totalMemcpy2 += std::chrono::duration<double, std::milli>(t5 - t4).count();

    for (size_t i = 0; i < (conf.multiplier - 1) && i < this->lsfgOutPtrs.size(); i++) {
        // Acquire next swapchain image
        pass.acquireSemaphores.at(i) = Mini::Semaphore(info.device);
        uint32_t imageIdx{};
        auto res = Layer::ovkAcquireNextImageKHR(info.device, this->swapchain, UINT64_MAX,
            pass.acquireSemaphores.at(i).handle(), VK_NULL_HANDLE, &imageIdx);
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw LSFG::vulkan_error(res, "Failed to acquire next swapchain image");

        // Copy staging → out_n → swapchain (need blit for format conversion)
        pass.postCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.prevPostCopySemaphores.at(i) = Mini::Semaphore(info.device);
        pass.postCopyBufs.at(i) = Mini::CommandBuffer(info.device, this->cmdPool);
        pass.postCopyBufs.at(i).begin();

        // Step 1: Copy staging buffer to out_n image (same 16-bit format)
        // Use TRANSFER_SRC_OPTIMAL as final layout for the subsequent blit
        this->outStagingN.at(i).copyToImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        // Step 2: Blit out_n to swapchain (format conversion 16-bit → 8-bit)
        Utils::copyImage(pass.postCopyBufs.at(i).handle(),
            this->out_n.at(i).handle(),
            this->swapchainImages.at(imageIdx),
            this->extent.width, this->extent.height,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            false, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        pass.postCopyBufs.at(i).end();
        pass.postCopyBufs.at(i).submit(info.queue.second,
            { pass.acquireSemaphores.at(i).handle() },
            { pass.postCopySemaphores.at(i).handle(),
              pass.prevPostCopySemaphores.at(i).handle() });

        // Present swapchain image
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

    // Present the original frame
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

    auto t6 = std::chrono::steady_clock::now();
    totalPresent += std::chrono::duration<double, std::milli>(t6 - t5).count();

    this->frameIdx++;
    return res;
}
