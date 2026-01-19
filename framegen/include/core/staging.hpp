#pragma once

#include "core/device.hpp"

#include <vulkan/vulkan_core.h>

#include <memory>

namespace LSFG::Core {

    ///
    /// Host-visible staging buffer for CPU-GPU data transfer.
    ///
    class StagingBuffer {
    public:
        StagingBuffer() noexcept = default;

        ///
        /// Create a staging buffer.
        ///
        /// @param device Vulkan device
        /// @param extent Extent of the images to transfer
        /// @param format Format of the images (for size calculation)
        ///
        /// @throws LSFG::vulkan_error if creation fails.
        ///
        StagingBuffer(const Core::Device& device, VkExtent2D extent, VkFormat format);

        /// Copy image contents to staging buffer
        void copyFromImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout);

        /// Copy staging buffer contents to image
        void copyToImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout);

        /// Get mapped memory pointer for CPU access
        [[nodiscard]] void* data() const { return mappedData; }

        /// Get buffer size in bytes
        [[nodiscard]] VkDeviceSize size() const { return bufferSize; }

        /// Get buffer handle
        [[nodiscard]] VkBuffer handle() const { return *buffer; }

        [[nodiscard]] VkExtent2D getExtent() const { return extent; }
        [[nodiscard]] VkFormat getFormat() const { return format; }

        /// Trivially copyable, moveable and destructible
        StagingBuffer(const StagingBuffer&) noexcept = default;
        StagingBuffer& operator=(const StagingBuffer&) noexcept = default;
        StagingBuffer(StagingBuffer&&) noexcept = default;
        StagingBuffer& operator=(StagingBuffer&&) noexcept = default;
        ~StagingBuffer() = default;

    private:
        std::shared_ptr<VkBuffer> buffer;
        std::shared_ptr<VkDeviceMemory> memory;
        void* mappedData = nullptr;
        VkDeviceSize bufferSize = 0;
        VkExtent2D extent{};
        VkFormat format{};
    };

}
