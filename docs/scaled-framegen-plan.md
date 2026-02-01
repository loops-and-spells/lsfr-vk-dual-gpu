# Scaled Dual-GPU Frame Generation

## Problem

Current dual-GPU staging transfers full-resolution frames via PCIe, causing significant overhead:
- 4K @ 16-bit: ~88MB per frame
- memcpy time: ~4.4ms each direction (~9ms round trip)
- Total overhead makes dual-GPU slower than single-GPU for high-FPS games

## Solution

Run frame generation at reduced resolution and use the primary GPU for up/downscaling:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Primary GPU (Game)                            │
│  Game Frame (4K) ──► Downsample ──► Staging Buffer (1080p)      │
└────────────────────────────┬────────────────────────────────────┘
                             │ PCIe (~22MB instead of 88MB)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Secondary GPU (Frame Generation)                 │
│  Import (1080p) ──► Framegen ──► Export (1080p)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ PCIe (~22MB)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Primary GPU (Present)                         │
│  Staging Buffer (1080p) ──► Upsample ──► Swapchain (4K)         │
└─────────────────────────────────────────────────────────────────┘
```

## Expected Performance Gains

| Resolution | Frame Size | Transfer Time | Round Trip |
|------------|------------|---------------|------------|
| 4K (current) | 88MB | 4.4ms | 9ms |
| 1080p (0.5x) | 22MB | 1.1ms | 2.2ms |
| 720p (0.33x) | 10MB | 0.5ms | 1ms |

Potential improvement: **7-8ms saved per frame** at 0.5x scale

## Configuration

```toml
[[game]]
exe = "GameThread"
multiplier = 3
gpu = "225:0.0"
gpu_secondary = "17:0.0"
framegen_scale = 0.5  # NEW: Run framegen at 50% resolution
```

## Implementation

### Phase 1: Config Changes

**File: `include/config/config.hpp`**
```cpp
struct GameConf {
    // ... existing fields ...
    std::optional<float> framegen_scale;  // 0.25 to 1.0, default 1.0
};
```

**File: `src/config/config.cpp`**
- Parse `framegen_scale` from TOML
- Validate range (0.25 to 1.0)

### Phase 2: Layer-Side Scaling

**File: `src/context.cpp`**

1. Calculate scaled extent:
```cpp
VkExtent2D scaledExtent = {
    static_cast<uint32_t>(extent.width * conf.framegen_scale.value_or(1.0f)),
    static_cast<uint32_t>(extent.height * conf.framegen_scale.value_or(1.0f))
};
```

2. Create staging buffers at scaled resolution:
```cpp
// Staging buffers at reduced resolution
this->inStaging_0 = Mini::StagingBuffer(device, physicalDevice, scaledExtent, format);
this->inStaging_1 = Mini::StagingBuffer(device, physicalDevice, scaledExtent, format);
```

3. Add downsample before staging copy:
```cpp
void LsContext::downsampleToStaging(VkCommandBuffer cmd, VkImage src,
                                     StagingBuffer& staging, VkExtent2D srcExtent) {
    // Create temp image at scaled resolution if needed, or blit directly
    // Use VK_FILTER_LINEAR for quality downsampling
    VkImageBlit region{...};
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   tempScaledImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_LINEAR);
    // Then copy to staging buffer
    staging.copyFromImage(cmd, tempScaledImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}
```

4. Add upsample after staging copy back:
```cpp
void LsContext::upsampleFromStaging(VkCommandBuffer cmd, StagingBuffer& staging,
                                     VkImage dst, VkExtent2D dstExtent) {
    // Copy from staging to temp scaled image
    staging.copyToImage(cmd, tempScaledImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    // Blit upsample to full resolution
    VkImageBlit region{...};
    vkCmdBlitImage(cmd, tempScaledImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_LINEAR);
}
```

### Phase 3: Framegen-Side Changes

**File: `framegen/v3.1_src/context.cpp`**

1. Create context at scaled resolution (already passed via extent parameter)
2. Staging buffers created at scaled resolution
3. No other changes needed - framegen works at whatever resolution it's given

### Phase 4: Intermediate Images

Need temporary images for blit operations:

**Layer side (primary GPU):**
- `scaledInput[2]` - For downsampled input frames before staging copy
- `scaledOutput[N]` - For upsampling output frames after staging copy

**Framegen side (secondary GPU):**
- Already creates images at the extent it's given - no changes needed

### Files to Modify

| File | Changes |
|------|---------|
| `include/config/config.hpp` | Add `framegen_scale` field |
| `src/config/config.cpp` | Parse `framegen_scale` from TOML |
| `include/context.hpp` | Add scaled images, scaledExtent member |
| `src/context.cpp` | Implement down/upsample, create scaled staging |
| `include/mini/image.hpp` | (possibly) Add blit helper |

### New Helper Functions

```cpp
// In utils.hpp/cpp
namespace Utils {
    void blitImage(VkCommandBuffer cmd,
                   VkImage src, VkExtent2D srcExtent, VkImageLayout srcLayout,
                   VkImage dst, VkExtent2D dstExtent, VkImageLayout dstLayout,
                   VkFilter filter);
}
```

## Testing Plan

1. **vkcube at 4K with 0.5x scale**
   - Verify visual quality acceptable
   - Measure memcpy times (should be ~1ms vs ~4ms)
   - Measure total FPS improvement

2. **Cyberpunk at 4K with 0.5x scale**
   - Compare FPS: single-GPU vs dual-GPU-scaled
   - Visual quality assessment

3. **Different scale factors**
   - Test 0.25, 0.5, 0.75, 1.0
   - Find optimal quality/performance tradeoff

## Rollback Plan

If `framegen_scale` is not set or set to 1.0, behavior is identical to current implementation (no scaling).

## Future Enhancements

1. **Auto-scale**: Automatically determine optimal scale based on resolution and PCIe bandwidth
2. **Temporal upscaling**: Use FSR/DLSS-style temporal upscaling instead of simple bilinear
3. **Selective scaling**: Only scale when dual-GPU, not for single-GPU mode
