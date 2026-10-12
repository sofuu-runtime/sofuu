// src/metal/metal_engine.h — C FFI bridge for Apple Silicon Metal GPU engine.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns true if running on macOS with an active Apple Silicon GPU.
bool sofuu_metal_available(void);

// Initialize Metal device, command queue, and compute pipeline states.
// Returns an opaque context pointer, or NULL on failure.
void* sofuu_metal_init(void);

// Clean up Metal context and release all GPU resources.
void sofuu_metal_free(void* ctx);

// Wrap page-aligned host memory (e.g. mmap) in a zero-copy MTLBuffer.
void* sofuu_metal_buffer_create_zerocopy(void* ctx, const void* ptr, size_t size_bytes);
void sofuu_metal_buffer_free(void* buf);

// Execute BF16 Matrix-Vector multiplication on Apple Silicon GPU.
// y = W * x
void sofuu_metal_gemv_bf16(
    void* ctx,
    void* w_buf,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

// Execute FP16 Matrix-Vector multiplication on Apple Silicon GPU.
void sofuu_metal_gemv_f16(
    void* ctx,
    void* w_buf,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

// Execute FP32 Matrix-Vector multiplication on Apple Silicon GPU.
void sofuu_metal_gemv_f32(
    void* ctx,
    void* w_buf,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

// Execute RMSNorm on GPU.
void sofuu_metal_rmsnorm(
    void* ctx,
    const float* x,
    void* w_buf,
    size_t w_offset,
    float* y,
    size_t dim,
    float eps
);

// Execute SwiGLU in-place on GPU: g = silu(g) * u.
void sofuu_metal_swiglu(
    void* ctx,
    float* g,
    const float* u,
    size_t size
);

// Execute depthwise Conv1d (kernel_size=3) for Liquid Foundation Models.
void sofuu_metal_conv1d_3(
    void* ctx,
    void* conv_w_buf,
    size_t conv_w_offset,
    const float* state,
    float* out,
    size_t channels
);

// Execute MLX 4-bit affine GEMV on GPU: y = dequant(W, scales, biases) * x
void sofuu_metal_gemv_mlx4(
    void* ctx,
    void* w_buf,
    size_t w_offset,
    void* scales_buf,
    size_t scales_offset,
    void* biases_buf,
    size_t biases_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    size_t group_size
);

#ifdef __cplusplus
}
#endif
