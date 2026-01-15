# Apple Silicon Optimization Notes

## Unified Memory Architecture

On Apple Silicon (M1/M2/M3/M4), CPU and GPU share the same physical memory. There is no PCIe bus or discrete GPU memory - both processors access a unified memory pool.

### Current Implementation

The current wgpu backend uses `queue.write_buffer()` to "upload" positions to the GPU:

```rust
self.queue.write_buffer(
    &self.all_positions,
    offset as u64,
    bytemuck::cast_slice(&data),
);
```

On Apple Silicon, this is essentially a `memcpy` to the same memory space (no DMA transfer). However, there is still overhead from:

1. **API overhead**: wgpu validation, command buffer recording
2. **Synchronization barriers**: Ensuring GPU sees updated data before compute dispatch
3. **The memcpy itself**: Still O(data size) even without transfer

### Potential Optimizations

#### 1. Mapped Buffers (wgpu)

Use `MAP_WRITE` buffers for direct CPU access:

```rust
// Create buffer mapped at creation
let buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("positions"),
    size: total_size,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE,
    mapped_at_creation: true,
});

// Write directly to GPU-visible memory
{
    let mut view = buffer.slice(..).get_mapped_range_mut();
    view[offset..offset + size].copy_from_slice(&data);
}
buffer.unmap();
```

**Limitation**: wgpu doesn't support persistently mapped buffers. Must unmap before GPU use, then remap for next write.

#### 2. Native Metal Backend

For maximum Apple Silicon performance, use Metal directly:

```rust
use metal::*;

// Create shared buffer (CPU + GPU visible)
let buffer = device.new_buffer_with_data(
    data.as_ptr() as *const _,
    size,
    MTLResourceOptions::StorageModeShared,
);

// Get direct pointer - no "upload" needed
let ptr = buffer.contents() as *mut f32;
unsafe {
    std::ptr::copy_nonoverlapping(new_data.as_ptr(), ptr.add(offset), count);
}
// GPU immediately sees the changes (with appropriate barriers)
```

**Benefits**:
- Zero-copy updates via direct pointer access
- No wgpu abstraction overhead
- Can use Metal-specific features (argument buffers, indirect command buffers)

**Drawbacks**:
- macOS only (no cross-platform)
- More complex API
- Manual synchronization required

#### 3. Hybrid Approach

Keep wgpu for portability but add Metal fast-path:

```rust
#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(not(target_os = "macos"))]
mod wgpu_backend;
```

### Memory Layout Considerations

Apple Silicon benefits from:
- **16-byte alignment**: GPU likes vec4 (16-byte) aligned data
- **Page-aligned buffers**: For large allocations (> 16 KB)
- **Contiguous access patterns**: Better cache utilization

Current position buffer layout is already optimal:
```
[mol0_site0, mol0_site1, ..., mol1_site0, mol1_site1, ...]
     ↑ vec4 (16 bytes each)
```

### Tested: Partial Position Uploads

**Result: No benefit on Apple Silicon**

We tested uploading only the moved molecule's positions (11 KB) instead of all positions (4.5 MB for 400 molecules):

| Molecules | Full Upload | Partial Upload | Change |
|-----------|-------------|----------------|--------|
| 100 | 127.6 steps/s | 100.0 steps/s | **-22%** |
| 200 | 68.8 steps/s | 64.8 steps/s | **-6%** |
| 400 | 32.7 steps/s | 41.2 steps/s | +26% |

**Conclusion**: On unified memory, the memcpy is so fast that the overhead from partial upload logic (offset calculations, smaller non-contiguous writes) outweighs the benefit. This optimization would help on discrete GPUs with PCIe transfers but is counterproductive on Apple Silicon.

### Estimated Impact of Other Optimizations

| Optimization | Effort | Speedup Estimate |
|--------------|--------|------------------|
| Mapped buffers | Medium | ~10-20% (removes API overhead) |
| Native Metal | High | ~20-50% (zero-copy + Metal features) |

### Benchmarking Notes

To measure actual transfer overhead on Apple Silicon:

```bash
# Profile with Instruments
xcrun xctrace record --template 'Metal System Trace' --launch ./target/release/mc_simulator

# Or with metal-cpp timing
MTL_DEBUG_LAYER=1 ./target/release/mc_simulator
```

### References

- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Optimizing for Apple Silicon](https://developer.apple.com/documentation/apple-silicon)
- [wgpu Metal backend](https://github.com/gfx-rs/wgpu/tree/trunk/wgpu-hal/src/metal)
