#include "mini/fence.hpp"
#include "common/exception.hpp"
#include "layer.hpp"

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

using namespace Mini;

Fence::Fence(VkDevice device) {
    const VkFenceCreateInfo desc{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    VkFence fenceHandle{};
    auto res = Layer::ovkCreateFence(device, &desc, nullptr, &fenceHandle);
    if (res != VK_SUCCESS || fenceHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Unable to create fence");

    this->fence = std::shared_ptr<VkFence>(
        new VkFence(fenceHandle),
        [dev = device](VkFence* fenceHandle) {
            Layer::ovkDestroyFence(dev, *fenceHandle, nullptr);
        }
    );
}

bool Fence::wait(VkDevice device, uint64_t timeout) const {
    VkFence fenceHandle = this->handle();
    auto res = Layer::ovkWaitForFences(device, 1, &fenceHandle, VK_TRUE, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
        throw LSFG::vulkan_error(res, "Unable to wait for fence");
    return res == VK_SUCCESS;
}

void Fence::reset(VkDevice device) const {
    VkFence fenceHandle = this->handle();
    auto res = Layer::ovkResetFences(device, 1, &fenceHandle);
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Unable to reset fence");
}
