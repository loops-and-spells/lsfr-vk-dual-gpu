#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "v3_1/context.hpp"
#include "common/utils.hpp"
#include "common/exception.hpp"

#include <iostream>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <optional>
#include <cstdint>
#include <cstring>

using namespace LSFG_3_1;

void Context::initShaders(Vulkan& vk, const std::vector<int>& outN, VkFormat format, bool stagingMode) {
    // prepare render data
    for (size_t i = 0; i < 8; i++) {
        auto& data = this->data.at(i);
        data.internalSemaphores.resize(vk.generationCount);
        data.outSemaphores.resize(vk.generationCount);
        data.completionFences.resize(vk.generationCount);
        data.cmdBuffers2.resize(vk.generationCount);
    }

    // create shader chains
    this->mipmaps = Shaders::Mipmaps(vk, this->inImg_0, this->inImg_1);
    for (size_t i = 0; i < 7; i++)
        this->alpha.at(i) = Shaders::Alpha(vk, this->mipmaps.getOutImages().at(i));
    this->beta = Shaders::Beta(vk, this->alpha.at(0).getOutImages());
    for (size_t i = 0; i < 7; i++) {
        this->gamma.at(i) = Shaders::Gamma(vk,
            this->alpha.at(6 - i).getOutImages(),
            this->beta.getOutImages().at(std::min<size_t>(6 - i, 5)),
            (i == 0) ? std::nullopt : std::make_optional(this->gamma.at(i - 1).getOutImage()));
        if (i < 4) continue;

        this->delta.at(i - 4) = Shaders::Delta(vk,
            this->alpha.at(6 - i).getOutImages(),
            this->beta.getOutImages().at(6 - i),
            (i == 4) ? std::nullopt : std::make_optional(this->gamma.at(i - 1).getOutImage()),
            (i == 4) ? std::nullopt : std::make_optional(this->delta.at(i - 5).getOutImage1()),
            (i == 4) ? std::nullopt : std::make_optional(this->delta.at(i - 5).getOutImage2()));
    }
    this->generate = Shaders::Generate(vk,
        this->inImg_0, this->inImg_1,
        this->gamma.at(6).getOutImage(),
        this->delta.at(2).getOutImage1(),
        this->delta.at(2).getOutImage2(),
        outN, format, stagingMode);
}

Context::Context(Vulkan& vk,
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format)
        : stagingMode(false) {
    // import input images
    this->inImg_0 = Core::Image(vk.device, extent, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, in0);
    this->inImg_1 = Core::Image(vk.device, extent, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, in1);

    initShaders(vk, outN, format);
}

// Staging mode constructor
Context::Context(Vulkan& vk, VkExtent2D extent, VkFormat format)
        : stagingMode(true) {
    std::cerr << "lsfg-framegen: Creating context in STAGING MODE\n";

    // Create local images (no FD import)
    this->inImg_0 = Core::Image(vk.device, extent, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    this->inImg_1 = Core::Image(vk.device, extent, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Create staging buffers for input
    this->inStaging_0 = Core::StagingBuffer(vk.device, extent, format);
    this->inStaging_1 = Core::StagingBuffer(vk.device, extent, format);

    // Create staging buffers for output (one per generated frame)
    for (size_t i = 0; i < vk.generationCount; ++i) {
        this->outStagingN.emplace_back(vk.device, extent, format);
    }

    std::cerr << "lsfg-framegen: Created " << vk.generationCount << " output staging buffers\n";

    // Initialize shaders with empty outN (staging mode doesn't use FD export)
    // Pass stagingMode=true so output images get TRANSFER_SRC_BIT for copying to staging
    std::vector<int> emptyOutN(vk.generationCount, -1);
    initShaders(vk, emptyOutN, format, true);
}

void Context::getStagingPointers(void** in0, void** in1, std::vector<void*>& outPtrs) const {
    if (!stagingMode) {
        throw LSFG::vulkan_error(VK_ERROR_UNKNOWN, "getStagingPointers called on non-staging context");
    }
    if (in0) *in0 = inStaging_0.data();
    if (in1) *in1 = inStaging_1.data();
    outPtrs.clear();
    for (const auto& staging : outStagingN) {
        outPtrs.push_back(staging.data());
    }
}

void Context::present(Vulkan& vk,
        int inSem, const std::vector<int>& outSem) {
    auto& data = this->data.at(this->frameIdx % 8);

    // 3. wait for completion of previous frame in this slot
    if (data.shouldWait)
        for (auto& fence : data.completionFences)
            if (!fence.wait(vk.device, UINT64_MAX))
                throw LSFG::vulkan_error(VK_TIMEOUT, "Fence wait timed out");
    data.shouldWait = true;

    // 1. create mipmaps and process input image
    if (inSem >= 0) data.inSemaphore = Core::Semaphore(vk.device, inSem);
    for (size_t i = 0; i < vk.generationCount; i++)
        data.internalSemaphores.at(i) = Core::Semaphore(vk.device);

    data.cmdBuffer1 = Core::CommandBuffer(vk.device, vk.commandPool);
    data.cmdBuffer1.begin();

    this->mipmaps.Dispatch(data.cmdBuffer1, this->frameIdx);
    for (size_t i = 0; i < 7; i++)
        this->alpha.at(6 - i).Dispatch(data.cmdBuffer1, this->frameIdx);
    this->beta.Dispatch(data.cmdBuffer1, this->frameIdx);

    data.cmdBuffer1.end();
    std::vector<Core::Semaphore> waits = { data.inSemaphore };
    if (inSem < 0) waits.clear();
    data.cmdBuffer1.submit(vk.device.getComputeQueue(), std::nullopt,
        waits, std::nullopt,
        data.internalSemaphores, std::nullopt);

    // 2. generate intermediary frames
    for (size_t pass = 0; pass < vk.generationCount; pass++) {
        auto& internalSemaphore = data.internalSemaphores.at(pass);
        auto& outSemaphore = data.outSemaphores.at(pass);
        if (inSem >= 0) outSemaphore = Core::Semaphore(vk.device, outSem.empty() ? -1 : outSem.at(pass));
        auto& completionFence = data.completionFences.at(pass);
        completionFence = Core::Fence(vk.device);

        auto& buf2 = data.cmdBuffers2.at(pass);
        buf2 = Core::CommandBuffer(vk.device, vk.commandPool);
        buf2.begin();

        for (size_t i = 0; i < 7; i++) {
            this->gamma.at(i).Dispatch(buf2, this->frameIdx, pass);
            if (i >= 4)
                this->delta.at(i - 4).Dispatch(buf2, this->frameIdx, pass);
        }
        this->generate.Dispatch(buf2, this->frameIdx, pass);

        buf2.end();
        std::vector<Core::Semaphore> signals = { outSemaphore };
        if (inSem < 0) signals.clear();
        buf2.submit(vk.device.getComputeQueue(), completionFence,
            { internalSemaphore }, std::nullopt,
            signals, std::nullopt);
    }

    this->frameIdx++;
}

void Context::presentStaging(Vulkan& vk) {
    if (!stagingMode) {
        throw LSFG::vulkan_error(VK_ERROR_UNKNOWN, "presentStaging called on non-staging context");
    }

    auto& renderData = this->data.at(this->frameIdx % 8);

    // Wait for previous frame in this slot
    if (renderData.shouldWait) {
        if (renderData.stagingFence.handle() != VK_NULL_HANDLE) {
            if (!renderData.stagingFence.wait(vk.device, UINT64_MAX))
                throw LSFG::vulkan_error(VK_TIMEOUT, "Staging fence wait timed out");
        }
        for (auto& fence : renderData.completionFences)
            if (fence.handle() != VK_NULL_HANDLE && !fence.wait(vk.device, UINT64_MAX))
                throw LSFG::vulkan_error(VK_TIMEOUT, "Fence wait timed out");
    }
    renderData.shouldWait = true;

    // 1. Copy from staging buffers to input images
    renderData.stagingInCmd = Core::CommandBuffer(vk.device, vk.commandPool);
    renderData.stagingInCmd.begin();

    // Copy to the appropriate input image based on frame index
    if (this->frameIdx % 2 == 0) {
        inStaging_0.copyToImage(renderData.stagingInCmd.handle(),
            inImg_0.handle(), inImg_0.getLayout());
        inImg_0.setLayout(VK_IMAGE_LAYOUT_GENERAL);
    } else {
        inStaging_1.copyToImage(renderData.stagingInCmd.handle(),
            inImg_1.handle(), inImg_1.getLayout());
        inImg_1.setLayout(VK_IMAGE_LAYOUT_GENERAL);
    }

    renderData.stagingInCmd.end();

    // Submit staging copy and wait for it
    Core::Fence stagingInFence(vk.device);
    renderData.stagingInCmd.submit(vk.device.getComputeQueue(), stagingInFence,
        {}, std::nullopt, {}, std::nullopt);
    if (!stagingInFence.wait(vk.device, UINT64_MAX))
        throw LSFG::vulkan_error(VK_TIMEOUT, "Staging input copy fence wait timed out");

    // 2. Run the frame generation (same as regular present but without semaphores)
    for (size_t i = 0; i < vk.generationCount; i++)
        renderData.internalSemaphores.at(i) = Core::Semaphore(vk.device);

    renderData.cmdBuffer1 = Core::CommandBuffer(vk.device, vk.commandPool);
    renderData.cmdBuffer1.begin();

    this->mipmaps.Dispatch(renderData.cmdBuffer1, this->frameIdx);
    for (size_t i = 0; i < 7; i++)
        this->alpha.at(6 - i).Dispatch(renderData.cmdBuffer1, this->frameIdx);
    this->beta.Dispatch(renderData.cmdBuffer1, this->frameIdx);

    renderData.cmdBuffer1.end();
    renderData.cmdBuffer1.submit(vk.device.getComputeQueue(), std::nullopt,
        {}, std::nullopt,
        renderData.internalSemaphores, std::nullopt);

    // Generate intermediary frames
    for (size_t pass = 0; pass < vk.generationCount; pass++) {
        auto& internalSemaphore = renderData.internalSemaphores.at(pass);
        auto& completionFence = renderData.completionFences.at(pass);
        completionFence = Core::Fence(vk.device);

        auto& buf2 = renderData.cmdBuffers2.at(pass);
        buf2 = Core::CommandBuffer(vk.device, vk.commandPool);
        buf2.begin();

        for (size_t i = 0; i < 7; i++) {
            this->gamma.at(i).Dispatch(buf2, this->frameIdx, pass);
            if (i >= 4)
                this->delta.at(i - 4).Dispatch(buf2, this->frameIdx, pass);
        }
        this->generate.Dispatch(buf2, this->frameIdx, pass);

        buf2.end();
        buf2.submit(vk.device.getComputeQueue(), completionFence,
            { internalSemaphore }, std::nullopt,
            {}, std::nullopt);
    }

    // Wait for all generation to complete
    for (auto& fence : renderData.completionFences)
        if (!fence.wait(vk.device, UINT64_MAX))
            throw LSFG::vulkan_error(VK_TIMEOUT, "Generation fence wait timed out");

    // 3. Copy output images to staging buffers
    renderData.stagingOutCmd = Core::CommandBuffer(vk.device, vk.commandPool);
    renderData.stagingOutCmd.begin();

    auto& outImages = this->generate.getOutImages();
    for (size_t i = 0; i < vk.generationCount && i < outStagingN.size(); ++i) {
        outStagingN.at(i).copyFromImage(renderData.stagingOutCmd.handle(),
            outImages.at(i).handle(), outImages.at(i).getLayout());
    }

    renderData.stagingOutCmd.end();

    renderData.stagingFence = Core::Fence(vk.device);
    renderData.stagingOutCmd.submit(vk.device.getComputeQueue(), renderData.stagingFence,
        {}, std::nullopt, {}, std::nullopt);

    // Wait for output copy to complete
    if (!renderData.stagingFence.wait(vk.device, UINT64_MAX))
        throw LSFG::vulkan_error(VK_TIMEOUT, "Staging output copy fence wait timed out");

    this->frameIdx++;
}
