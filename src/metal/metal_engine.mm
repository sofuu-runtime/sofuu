#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_engine.h"
#include <cstring>
#include <iostream>

#if defined(__APPLE__)

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso_gemv_bf16;
    id<MTLComputePipelineState> pso_gemv_f16;
    id<MTLComputePipelineState> pso_gemv_f32;
    id<MTLComputePipelineState> pso_gemv_mlx4;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_conv1d_3;

    id<MTLBuffer> x_buf;
    id<MTLBuffer> y_buf;
    size_t x_buf_cap;
    size_t y_buf_cap;
};

static const char* METAL_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv_bf16(
    device const bfloat *w          [[buffer(0)]],
    device const float  *x          [[buffer(1)]],
    device float        *y          [[buffer(2)]],
    constant uint       &cols       [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    device const bfloat *row_w = w + row * cols;

    uint i = lane * 4;
    for (; i + 4 <= cols; i += 32 * 4) {
        sum += float(row_w[i + 0]) * x[i + 0];
        sum += float(row_w[i + 1]) * x[i + 1];
        sum += float(row_w[i + 2]) * x[i + 2];
        sum += float(row_w[i + 3]) * x[i + 3];
    }
    for (; i < cols; ++i) {
        sum += float(row_w[i]) * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f16(
    device const half  *w           [[buffer(0)]],
    device const float *x           [[buffer(1)]],
    device float       *y           [[buffer(2)]],
    constant uint      &cols        [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    device const half *row_w = w + row * cols;

    uint i = lane * 4;
    for (; i + 4 <= cols; i += 32 * 4) {
        sum += float(row_w[i + 0]) * x[i + 0];
        sum += float(row_w[i + 1]) * x[i + 1];
        sum += float(row_w[i + 2]) * x[i + 2];
        sum += float(row_w[i + 3]) * x[i + 3];
    }
    for (; i < cols; ++i) {
        sum += float(row_w[i]) * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f32(
    device const float *w           [[buffer(0)]],
    device const float *x           [[buffer(1)]],
    device float       *y           [[buffer(2)]],
    constant uint      &cols        [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    device const float *row_w = w + row * cols;

    uint i = lane * 4;
    for (; i + 4 <= cols; i += 32 * 4) {
        sum += row_w[i + 0] * x[i + 0];
        sum += row_w[i + 1] * x[i + 1];
        sum += row_w[i + 2] * x[i + 2];
        sum += row_w[i + 3] * x[i + 3];
    }
    for (; i < cols; ++i) {
        sum += row_w[i] * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void rmsnorm(
    device const float  *x          [[buffer(0)]],
    device const bfloat *w          [[buffer(1)]],
    device float        *y          [[buffer(2)]],
    constant uint       &dim        [[buffer(3)]],
    constant float      &eps        [[buffer(4)]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float val = x[i];
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);

    for (uint i = lane; i < dim; i += 32) {
        y[i] = x[i] * rms * float(w[i]);
    }
}

kernel void swiglu(
    device float       *g           [[buffer(0)]],
    device const float *u           [[buffer(1)]],
    uint id                         [[thread_position_in_grid]]
) {
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

kernel void conv1d_3(
    device const bfloat *conv_w     [[buffer(0)]],
    device const float  *state      [[buffer(1)]],
    device float        *out        [[buffer(2)]],
    uint c                          [[thread_position_in_grid]]
) {
    device const bfloat *cw = conv_w + c * 3;
    device const float  *st = state + c * 3;
    float sum = float(cw[0]) * st[0] + float(cw[1]) * st[1] + float(cw[2]) * st[2];
    out[c] = sum;
}

kernel void gemv_mlx4_affine(
    device const uint32_t *w        [[buffer(0)]],
    device const bfloat   *scales   [[buffer(1)]],
    device const bfloat   *biases   [[buffer(2)]],
    device const float    *x        [[buffer(3)]],
    device float          *y        [[buffer(4)]],
    constant uint         &cols     [[buffer(5)]],
    constant uint         &group_sz [[buffer(6)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint u32_per_row = cols / 8;
    device const uint32_t *row_w = w + row * u32_per_row;
    uint groups_per_row = cols / group_sz;
    device const bfloat *row_scales = scales + row * groups_per_row;
    device const bfloat *row_biases = biases + row * groups_per_row;

    for (uint u = lane; u < u32_per_row; u += 32) {
        uint32_t packed = row_w[u];
        uint col_base = u * 8;
        uint grp = col_base / group_sz;
        float scale = float(row_scales[grp]);
        float bias  = float(row_biases[grp]);

        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint val_4b = (packed >> (i * 4)) & 0x0F;
            float weight = float(val_4b) * scale + bias;
            sum += weight * x[col_base + i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}
)";

bool sofuu_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil) {
            return true;
        }
        return false;
    }
}

void* sofuu_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) return NULL;

        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (q == nil) return NULL;

        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:METAL_KERNEL_SOURCE];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&error];
        if (lib == nil) {
            NSLog(@"Sofuu Metal shader compile error: %@", error);
            return NULL;
        }

        auto load_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (fn == nil) return nil;
            NSError* pso_err = nil;
            return [dev newComputePipelineStateWithFunction:fn error:&pso_err];
        };

        MetalContext* ctx = new MetalContext();
        ctx->device = dev;
        ctx->queue = q;
        ctx->pso_gemv_bf16 = load_pso(@"gemv_bf16");
        ctx->pso_gemv_f16  = load_pso(@"gemv_f16");
        ctx->pso_gemv_f32  = load_pso(@"gemv_f32");
        ctx->pso_gemv_mlx4 = load_pso(@"gemv_mlx4_affine");
        ctx->pso_rmsnorm   = load_pso(@"rmsnorm");
        ctx->pso_swiglu    = load_pso(@"swiglu");
        ctx->pso_conv1d_3  = load_pso(@"conv1d_3");

        // Initial 64KB activation buffers
        ctx->x_buf_cap = 65536 * sizeof(float);
        ctx->y_buf_cap = 65536 * sizeof(float);
        ctx->x_buf = [dev newBufferWithLength:ctx->x_buf_cap options:MTLResourceStorageModeShared];
        ctx->y_buf = [dev newBufferWithLength:ctx->y_buf_cap options:MTLResourceStorageModeShared];

        return (void*)ctx;
    }
}

void sofuu_metal_free(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    delete ctx;
}

void* sofuu_metal_buffer_create_zerocopy(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                           length:size_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        return (__bridge_retained void*)buf;
    }
}

void sofuu_metal_buffer_free(void* buf_ptr) {
    if (!buf_ptr) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_ptr;
        (void)buf;
    }
}

static inline void ensure_buffers(MetalContext* ctx, size_t x_bytes, size_t y_bytes) {
    if (x_bytes > ctx->x_buf_cap) {
        ctx->x_buf_cap = x_bytes * 2;
        ctx->x_buf = [ctx->device newBufferWithLength:ctx->x_buf_cap options:MTLResourceStorageModeShared];
    }
    if (y_bytes > ctx->y_buf_cap) {
        ctx->y_buf_cap = y_bytes * 2;
        ctx->y_buf = [ctx->device newBufferWithLength:ctx->y_buf_cap options:MTLResourceStorageModeShared];
    }
}

void sofuu_metal_gemv_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_gemv_bf16];
        [enc setBuffer:w_buf offset:w_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_f16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_gemv_f16];
        [enc setBuffer:w_buf offset:w_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_f32(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_gemv_f32];
        [enc setBuffer:w_buf offset:w_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_rmsnorm(
    void* ctx_ptr,
    const float* x,
    void* w_buf_ptr,
    size_t w_offset,
    float* y,
    size_t dim,
    float eps
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || dim == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t bytes = dim * sizeof(float);
        ensure_buffers(ctx, bytes, bytes);
        std::memcpy(ctx->x_buf.contents, x, bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_rmsnorm];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:w_offset atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint d = (uint)dim;
        [enc setBytes:&d length:sizeof(uint) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(1, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, bytes);
    }
}

void sofuu_metal_swiglu(
    void* ctx_ptr,
    float* g,
    const float* u,
    size_t size
) {
    if (!ctx_ptr || !g || !u || size == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;

    @autoreleasepool {
        size_t bytes = size * sizeof(float);
        ensure_buffers(ctx, bytes, bytes);
        std::memcpy(ctx->x_buf.contents, g, bytes);
        std::memcpy(ctx->y_buf.contents, u, bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_swiglu];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:1];

        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(g, ctx->x_buf.contents, bytes);
    }
}

void sofuu_metal_conv1d_3(
    void* ctx_ptr,
    void* conv_w_buf_ptr,
    size_t conv_w_offset,
    const float* state,
    float* out,
    size_t channels
) {
    if (!ctx_ptr || !conv_w_buf_ptr || !state || !out || channels == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> conv_w_buf = (__bridge id<MTLBuffer>)conv_w_buf_ptr;

    @autoreleasepool {
        size_t state_bytes = channels * 3 * sizeof(float);
        size_t out_bytes = channels * sizeof(float);
        ensure_buffers(ctx, state_bytes, out_bytes);

        std::memcpy(ctx->x_buf.contents, state, state_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_conv1d_3];
        [enc setBuffer:conv_w_buf offset:conv_w_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];

        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroups = MTLSizeMake((channels + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(out, ctx->y_buf.contents, out_bytes);
    }
}

void sofuu_metal_gemv_mlx4(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    void* scales_buf_ptr,
    size_t scales_offset,
    void* biases_buf_ptr,
    size_t biases_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    size_t group_size
) {
    if (!ctx_ptr || !w_buf_ptr || !scales_buf_ptr || !biases_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;
    id<MTLBuffer> scales_buf = (__bridge id<MTLBuffer>)scales_buf_ptr;
    id<MTLBuffer> biases_buf = (__bridge id<MTLBuffer>)biases_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_gemv_mlx4];
        [enc setBuffer:w_buf offset:w_offset atIndex:0];
        [enc setBuffer:scales_buf offset:scales_offset atIndex:1];
        [enc setBuffer:biases_buf offset:biases_offset atIndex:2];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:3];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:4];
        uint c = (uint)cols;
        uint g = (uint)group_size;
        [enc setBytes:&c length:sizeof(uint) atIndex:5];
        [enc setBytes:&g length:sizeof(uint) atIndex:6];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

#else

// Stubs for non-macOS platforms (Linux, Android, Windows)
bool sofuu_metal_available(void) { return false; }
void* sofuu_metal_init(void) { return NULL; }
void sofuu_metal_free(void* ctx) {}
void* sofuu_metal_buffer_create_zerocopy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void sofuu_metal_buffer_free(void* buf) {}
void sofuu_metal_gemv_bf16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f32(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_rmsnorm(void* ctx, const float* x, void* w_buf, size_t w_offset, float* y, size_t dim, float eps) {}
void sofuu_metal_swiglu(void* ctx, float* g, const float* u, size_t size) {}
void sofuu_metal_conv1d_3(void* ctx, void* conv_w_buf, size_t conv_w_offset, const float* state, float* out, size_t channels) {}
void sofuu_metal_gemv_mlx4(void* ctx, void* w_buf, size_t w_offset, void* scales_buf, size_t scales_offset, void* biases_buf, size_t biases_offset, const float* x, float* y, size_t rows, size_t cols, size_t group_size) {}

#endif
