#pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace Mini {

    ///
    /// C++ wrapper class for a Vulkan fence.
    ///
    /// This class manages the lifetime of a Vulkan fence.
    ///
    class Fence {
    public:
        Fence() noexcept = default;

        ///
        /// Create the fence.
        ///
        /// @param device Vulkan device
        ///
        /// @throws LSFG::vulkan_error if object creation fails.
        ///
        Fence(VkDevice device);

        ///
        /// Wait for the fence to be signaled.
        ///
        /// @param device Vulkan device
        /// @param timeout Timeout in nanoseconds, or UINT64_MAX for no timeout.
        /// @return true if the fence was signaled, false on timeout.
        ///
        /// @throws LSFG::vulkan_error if waiting fails.
        ///
        [[nodiscard]] bool wait(VkDevice device, uint64_t timeout = UINT64_MAX) const;

        ///
        /// Reset the fence to unsignaled state.
        ///
        /// @param device Vulkan device
        ///
        /// @throws LSFG::vulkan_error if resetting fails.
        ///
        void reset(VkDevice device) const;

        /// Get the Vulkan handle.
        [[nodiscard]] VkFence handle() const {
            return fence ? *fence : VK_NULL_HANDLE;
        }

        // Trivially copyable, moveable and destructible
        Fence(const Fence&) noexcept = default;
        Fence& operator=(const Fence&) noexcept = default;
        Fence(Fence&&) noexcept = default;
        Fence& operator=(Fence&&) noexcept = default;
        ~Fence() = default;
    private:
        std::shared_ptr<VkFence> fence;
    };

}
