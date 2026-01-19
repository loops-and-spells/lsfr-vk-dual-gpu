#pragma once

#include <vulkan/vulkan_core.h>

#include <memory>
#include <cstdint>

namespace Mini {

    ///
    /// Host-visible staging buffer for cross-device data transfer.
    ///
    class StagingBuffer {
    public:
        StagingBuffer() = default;
        StagingBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
            VkExtent2D extent, VkFormat format);

        /// Copy image contents to staging buffer
        void copyFromImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout);

        /// Copy staging buffer contents to image
        /// @param finalLayout Layout to transition image to after copy (default: PRESENT_SRC_KHR)
        void copyToImage(VkCommandBuffer cmd, VkImage image,
            VkImageLayout finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        /// Get mapped memory pointer for CPU access
        [[nodiscard]] void* data() const { return mappedData; }

        /// Get buffer size in bytes
        [[nodiscard]] VkDeviceSize size() const { return bufferSize; }

        /// Get buffer handle
        [[nodiscard]] VkBuffer handle() const { return *buffer; }

        [[nodiscard]] VkExtent2D getExtent() const { return extent; }
        [[nodiscard]] VkFormat getFormat() const { return format; }

    private:
        std::shared_ptr<VkBuffer> buffer;
        std::shared_ptr<VkDeviceMemory> memory;
        void* mappedData = nullptr;
        VkDeviceSize bufferSize = 0;
        VkExtent2D extent{};
        VkFormat format{};
    };

}
