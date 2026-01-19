#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "core/device.hpp"
#include "core/instance.hpp"
#include "common/exception.hpp"

#include <iostream>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using namespace LSFG::Core;

const std::vector<const char*> requiredExtensions = {
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_robustness2",
};

Device::Device(const Instance& instance, uint64_t deviceUUID) {
    // get all physical devices
    uint32_t deviceCount{};
    auto res = vkEnumeratePhysicalDevices(instance.handle(), &deviceCount, nullptr);
    if (res != VK_SUCCESS || deviceCount == 0)
        throw LSFG::vulkan_error(res, "Failed to enumerate physical devices");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    res = vkEnumeratePhysicalDevices(instance.handle(), &deviceCount, devices.data());
    if (res != VK_SUCCESS)
        throw LSFG::vulkan_error(res, "Failed to get physical devices");

    // deviceUUID format: either vendorID:deviceID (legacy) or PCI-encoded
    // High 16 bits: if 0xFFFF, it's PCI-encoded: bus in bits 8-15, device in bits 0-7
    // Otherwise it's vendorID << 32 | deviceID
    const bool isPciEncoded = ((deviceUUID >> 48) & 0xFFFF) == 0xFFFF;
    uint32_t targetPciBus = 0, targetPciDevice = 0;
    if (isPciEncoded) {
        targetPciBus = (deviceUUID >> 8) & 0xFF;
        targetPciDevice = deviceUUID & 0xFF;
        std::cerr << "lsfg-framegen: PCI-encoded device selection - looking for bus="
                  << targetPciBus << " device=" << targetPciDevice << '\n';
    } else {
        std::cerr << "lsfg-framegen: Legacy device selection - UUID=0x"
                  << std::hex << deviceUUID << std::dec << '\n';
    }

    // get device by uuid or PCI bus
    std::optional<VkPhysicalDevice> physicalDevice;
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        if (isPciEncoded) {
            // Match by PCI bus info
            VkPhysicalDevicePCIBusInfoPropertiesEXT pciInfo{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT
            };
            VkPhysicalDeviceProperties2 props2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &pciInfo
            };
            vkGetPhysicalDeviceProperties2(device, &props2);

            std::cerr << "lsfg-framegen: Checking device " << props2.properties.deviceName
                      << " at PCI bus=" << pciInfo.pciBus << " device=" << pciInfo.pciDevice << '\n';

            if (pciInfo.pciBus == targetPciBus && pciInfo.pciDevice == targetPciDevice) {
                std::cerr << "lsfg-framegen: MATCHED! Using " << props2.properties.deviceName << '\n';
                physicalDevice = device;
                break;
            }
        } else {
            // Legacy: match by vendorID:deviceID
            const uint64_t uuid =
                static_cast<uint64_t>(properties.vendorID) << 32 | properties.deviceID;
            if (deviceUUID == uuid || deviceUUID == 0x1463ABAC) {
                physicalDevice = device;
                break;
            }
        }
    }
    if (!physicalDevice)
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED,
            "Could not find physical device with UUID/PCI");

    // find queue family indices
    uint32_t familyCount{};
    vkGetPhysicalDeviceQueueFamilyProperties(*physicalDevice, &familyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(*physicalDevice, &familyCount, queueFamilies.data());

    std::optional<uint32_t> computeFamilyIdx;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            computeFamilyIdx = i;
    }
    if (!computeFamilyIdx)
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "No compute queue family found");

    // create logical device
    const float queuePriority{1.0F}; // highest priority
    VkPhysicalDeviceRobustness2FeaturesEXT robustness2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
        .nullDescriptor = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &robustness2,
        .synchronization2 = VK_TRUE
    };
    const VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &features13,
        .timelineSemaphore = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE
    };
    const VkDeviceQueueCreateInfo computeQueueDesc{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = *computeFamilyIdx,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };
    const VkDeviceCreateInfo deviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &computeQueueDesc,
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data()
    };
    VkDevice deviceHandle{};
    res = vkCreateDevice(*physicalDevice, &deviceCreateInfo, nullptr, &deviceHandle);
    if (res != VK_SUCCESS | deviceHandle == VK_NULL_HANDLE)
        throw LSFG::vulkan_error(res, "Failed to create logical device");

    volkLoadDevice(deviceHandle);

    // get compute queue
    VkQueue queueHandle{};
    vkGetDeviceQueue(deviceHandle, *computeFamilyIdx, 0, &queueHandle);

    // store in shared ptr
    this->computeQueue = queueHandle;
    this->computeFamilyIdx = *computeFamilyIdx;
    this->physicalDevice = *physicalDevice;
    this->device = std::shared_ptr<VkDevice>(
        new VkDevice(deviceHandle),
        [](VkDevice* device) {
            vkDestroyDevice(*device, nullptr);
        }
    );
}
