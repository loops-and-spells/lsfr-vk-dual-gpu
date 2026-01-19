#pragma once

#include <vulkan/vulkan_core.h>

#include <functional>
#include <cstdint>
#include <string>
#include <vector>

namespace LSFG_3_1 {

    ///
    /// Initialize the LSFG library.
    ///
    /// @param deviceUUID The UUID of the Vulkan device to use.
    /// @param isHdr Whether the images are in HDR format.
    /// @param flowScale Internal flow scale factor.
    /// @param generationCount Number of frames to generate.
    /// @param loader Function to load shader source code by name.
    ///
    /// @throws LSFG::vulkan_error if Vulkan objects fail to initialize.
    ///
    __attribute__((visibility("default")))
    void initialize(uint64_t deviceUUID,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader);

    ///
    /// Create a new LSFG context on a swapchain.
    ///
    /// @param in0 File descriptor for the first input image.
    /// @param in1 File descriptor for the second input image.
    /// @param outN File descriptor for each output image. This defines the LSFG level.
    /// @param extent The size of the images
    /// @param format The format of the images.
    /// @return A unique identifier for the created context.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be created.
    ///
    __attribute__((visibility("default")))
    int32_t createContext(
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format);

    ///
    /// Present a context.
    ///
    /// @param id Unique identifier of the context to present.
    /// @param inSem Semaphore to wait on before starting the generation.
    /// @param outSem Semaphores to signal once each output image is ready.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be presented.
    ///
    __attribute__((visibility("default")))
    void presentContext(int32_t id, int inSem, const std::vector<int>& outSem);

    ///
    /// Delete an LSFG context.
    ///
    /// @param id Unique identifier of the context to delete.
    ///
    __attribute__((visibility("default")))
    void deleteContext(int32_t id);

    ///
    /// Deinitialize the LSFG library.
    ///
    __attribute__((visibility("default")))
    void finalize();

    // ==================== STAGING MODE API ====================
    // For cross-device operation where FD sharing isn't supported

    ///
    /// Create a new LSFG context using staging mode (no FD import).
    /// Input/output is done via CPU memory pointers.
    ///
    /// @param extent The size of the images
    /// @param format The format of the images.
    /// @return A unique identifier for the created context.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be created.
    ///
    __attribute__((visibility("default")))
    int32_t createContextStaging(VkExtent2D extent, VkFormat format);

    ///
    /// Get the staging buffer pointers for a context.
    ///
    /// @param id Unique identifier of the context.
    /// @param inPtr0 Output: pointer to source image 0 staging buffer
    /// @param inPtr1 Output: pointer to source image 1 staging buffer
    /// @param outPtrs Output: pointers to output image staging buffers
    ///
    __attribute__((visibility("default")))
    void getStagingPointers(int32_t id, void** inPtr0, void** inPtr1, std::vector<void*>& outPtrs);

    ///
    /// Present a context in staging mode.
    /// Before calling, copy source data to inPtr0/inPtr1 staging buffers.
    /// After calling, generated frames are in outPtrs staging buffers.
    ///
    /// @param id Unique identifier of the context to present.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be presented.
    ///
    __attribute__((visibility("default")))
    void presentContextStaging(int32_t id);

}
