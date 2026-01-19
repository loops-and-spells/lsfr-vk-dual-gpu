#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "core/staging.hpp"
#include "core/device.hpp"
#include "common/exception.hpp"

#include <iostream>
#include <memory>
#include <cstdint>
#include <optional>

using namespace LSFG::Core;

StagingBuffer::StagingBuffer(const Core::Device& device, VkExtent2D extent, VkFormat format)
        : extent(extent), format(format) {
    // Calculate buffer size based on format
    uint32_t bytesPerPixel = 4;
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT) bytesPerPixel = 8;
    this->bufferSize = extent.width * extent.height * bytesPerPixel;

    // Create staging buffer
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = this->bufferSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    VkBuffer bufferHandle{};
    auto res = vkCreateBuffer(device.handle(), &bufferInfo, nullptr, &bufferHandle);
    if (res != VK_SUCCESS || bufferHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Failed to create staging buffer");

    // Find host-visible memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(device.getPhysicalDevice(), &memProps);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device.handle(), bufferHandle, &memReqs);

    // Prefer HOST_CACHED for faster CPU reads, fall back to non-cached
    std::optional<uint32_t> memType{};
    std::optional<uint32_t> fallbackMemType{};
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
                memType.emplace(i);
                break;
            }
            if (!fallbackMemType.has_value())
                fallbackMemType.emplace(i);
        }
    }
    if (!memType.has_value()) memType = fallbackMemType;
    if (!memType.has_value())
        throw LSFG::vulkan_error(VK_ERROR_UNKNOWN, "Unable to find host-visible memory type");

    // Allocate memory
    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType.value()
    };
    VkDeviceMemory memoryHandle{};
    res = vkAllocateMemory(device.handle(), &allocInfo, nullptr, &memoryHandle);
    if (res != VK_SUCCESS || memoryHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Failed to allocate staging memory");

    res = vkBindBufferMemory(device.handle(), bufferHandle, memoryHandle, 0);
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Failed to bind staging buffer memory");

    // Map memory
    void* mappedPtr = nullptr;
    res = vkMapMemory(device.handle(), memoryHandle, 0, memReqs.size, 0, &mappedPtr);
    if (res != VK_SUCCESS || mappedPtr == nullptr)
        throw LSFG::vulkan_error(res, "Failed to map staging memory");

    this->mappedData = mappedPtr;

    // Store in shared_ptr for cleanup
    this->buffer = std::shared_ptr<VkBuffer>(
        new VkBuffer(bufferHandle),
        [dev = device.handle()](VkBuffer* buf) {
            vkDestroyBuffer(dev, *buf, nullptr);
        }
    );
    this->memory = std::shared_ptr<VkDeviceMemory>(
        new VkDeviceMemory(memoryHandle),
        [dev = device.handle(), mapped = mappedPtr](VkDeviceMemory* mem) {
            vkUnmapMemory(dev, *mem);
            vkFreeMemory(dev, *mem, nullptr);
        }
    );

    std::cerr << "lsfg-framegen: Created staging buffer " << bufferSize << " bytes, mapped at " << mappedPtr << '\n';
}

void StagingBuffer::copyFromImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout) {
    // Transition image to transfer src
    const VkImageMemoryBarrier srcBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = currentLayout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &srcBarrier);

    // Copy image to buffer
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1}
    };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        *buffer, 1, &region);

    // Transition back
    const VkImageMemoryBarrier dstBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = currentLayout,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr, 0, nullptr, 1, &dstBarrier);
}

void StagingBuffer::copyToImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout) {
    // Transition image to transfer dst
    const VkImageMemoryBarrier dstBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = currentLayout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &dstBarrier);

    // Copy buffer to image
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1}
    };
    vkCmdCopyBufferToImage(cmd, *buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);

    // Transition to general/storage layout
    const VkImageMemoryBarrier finalBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &finalBarrier);
}
