#pragma once

#include "hooks.hpp"
#include "mini/commandbuffer.hpp"
#include "mini/commandpool.hpp"
#include "mini/fence.hpp"
#include "mini/image.hpp"
#include "mini/semaphore.hpp"
#include "mini/staging.hpp"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

///
/// This class is the frame generation context. There should be one instance per swapchain.
///
class LsContext {
public:
    ///
    /// Create the swapchain context.
    ///
    /// @param info The device information to use.
    /// @param swapchain The Vulkan swapchain to use.
    /// @param extent The extent of the swapchain images.
    /// @param swapchainImages The swapchain images to use.
    ///
    /// @throws LSFG::vulkan_error if any Vulkan call fails.
    ///
    LsContext(const Hooks::DeviceInfo& info, VkSwapchainKHR swapchain,
        VkExtent2D extent, const std::vector<VkImage>& swapchainImages);

    ///
    /// Custom present logic.
    ///
    /// @param info The device information to use.
    /// @param pNext Unknown pointer set in the present info structure.
    /// @param queue The Vulkan queue to present the frame on.
    /// @param gameRenderSemaphores The semaphores to wait on before presenting.
    /// @param presentIdx The index of the swapchain image to present.
    /// @return The result of the Vulkan present operation, which can be VK_SUCCESS or VK_SUBOPTIMAL_KHR.
    ///
    /// @throws LSFG::vulkan_error if any Vulkan call fails.
    ///
    VkResult present(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx);

    // Non-copyable, trivially moveable and destructible
    LsContext(const LsContext&) = delete;
    LsContext& operator=(const LsContext&) = delete;
    LsContext(LsContext&&) = default;
    LsContext& operator=(LsContext&&) = default;
    ~LsContext();
private:
    VkResult presentFdMode(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx);
    VkResult presentStagingMode(const Hooks::DeviceInfo& info, const void* pNext, VkQueue queue,
        const std::vector<VkSemaphore>& gameRenderSemaphores, uint32_t presentIdx);

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    VkExtent2D extent;
    VkExtent2D scaledExtent; // scaled resolution for framegen (may equal extent if scale=1.0)
    float framegenScale{1.0F}; // framegen scale factor
    bool framegenUpscale{true}; // whether to upscale output back to full resolution
    bool framegenDebug{false}; // debug mode: add colored border to generated frames

    std::shared_ptr<int32_t> lsfgCtxId; // lsfg context id
    Mini::Image frame_0, frame_1; // frames shared with lsfg. write to frame_0 when fc % 2 == 0
    std::vector<Mini::Image> out_n; // output images shared with lsfg, indexed by framegen id

    // Staging mode support for cross-device operation
    bool stagingMode{false};
    Mini::StagingBuffer inStaging_0, inStaging_1; // layer-side staging for input images
    std::vector<Mini::StagingBuffer> outStagingN; // layer-side staging for output images
    void* lsfgInPtr0{nullptr}; // LSFG-side staging pointers
    void* lsfgInPtr1{nullptr};
    std::vector<void*> lsfgOutPtrs;

    // Scaled intermediate images for down/upsampling (only used when scale < 1.0)
    Mini::Image scaledInput_0, scaledInput_1; // downsampled input images on primary GPU
    std::vector<Mini::Image> scaledOutputN; // for upsampling output on primary GPU

    Mini::CommandPool cmdPool;
    uint64_t frameIdx{0};

    // Async pipeline state for staging mode
    bool hasPreviousOutput{false};
    uint32_t prevPresentIdx{0};
    Mini::Fence inputCopyFence;

    struct RenderPassInfo {
        Mini::CommandBuffer preCopyBuf; // copy from swapchain image to frame_0/frame_1
        std::array<Mini::Semaphore, 2> preCopySemaphores; // signal when preCopyBuf is done

        std::vector<Mini::Semaphore> renderSemaphores; // signal when lsfg is done with frame n

        std::vector<Mini::Semaphore> acquireSemaphores; // signal for swapchain image n

        std::vector<Mini::CommandBuffer> postCopyBufs; // copy from out_n to swapchain image
        std::vector<Mini::Semaphore> postCopySemaphores; // signal when postCopyBuf is done
        std::vector<Mini::Semaphore> prevPostCopySemaphores; // signal for previous postCopyBuf
    }; // data for a single render pass
    std::array<RenderPassInfo, 8> passInfos; // allocate 8 because why not
};
