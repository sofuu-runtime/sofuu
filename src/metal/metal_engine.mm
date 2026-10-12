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

    id<MTLCommandBuffer> pending_cmd;
    size_t pending_y_bytes;
    float* pending_y_dst;
};

static const char* METAL_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

static inline uint read_u32(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 3) == 0) {
        return *(device const uint*)p;
    } else {
        return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    }
}

static inline bfloat read_bf16(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 1) == 0) {
        return *(device const bfloat*)p;
    } else {
        ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
        return as_type<bfloat>(u);
    }
}

static inline half read_f16(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 1) == 0) {
        return *(device const half*)p;
    } else {
        ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
        return as_type<half>(u);
    }
}

kernel void gemv_bf16(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 2;

    if (((uintptr_t(w_base) + w_rem) & 1) == 0) {
        device const bfloat *row_w = (device const bfloat*)(w_base + row_byte);
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
    } else {
        for (uint i = lane; i < cols; i += 32) {
            sum += float(read_bf16(w_base, row_byte + i * 2)) * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f16(
    device const uchar *w_base          [[buffer(0)]],
    device const float *x               [[buffer(1)]],
    device float       *y               [[buffer(2)]],
    constant uint      &cols            [[buffer(3)]],
    constant uint      &w_rem           [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 2;

    if (((uintptr_t(w_base) + w_rem) & 1) == 0) {
        device const half *row_w = (device const half*)(w_base + row_byte);
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
    } else {
        for (uint i = lane; i < cols; i += 32) {
            sum += float(read_f16(w_base, row_byte + i * 2)) * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f32(
    device const uchar *w_base          [[buffer(0)]],
    device const float *x               [[buffer(1)]],
    device float       *y               [[buffer(2)]],
    constant uint      &cols            [[buffer(3)]],
    constant uint      &w_rem           [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 4;

    if (((uintptr_t(w_base) + w_rem) & 3) == 0) {
        device const float *row_w = (device const float*)(w_base + row_byte);
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
    } else {
        for (uint i = lane; i < cols; i += 32) {
            float val = as_type<float>(read_u32(w_base, row_byte + i * 4));
            sum += val * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void rmsnorm(
    device const float  *x              [[buffer(0)]],
    device const uchar  *w_base         [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &dim            [[buffer(3)]],
    constant float      &eps            [[buffer(4)]],
    constant uint       &w_rem          [[buffer(5)]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float val = x[i];
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);

    for (uint i = lane; i < dim; i += 32) {
        y[i] = x[i] * rms * float(read_bf16(w_base, w_rem + i * 2));
    }
}

kernel void swiglu(
    device float       *g               [[buffer(0)]],
    device const float *u               [[buffer(1)]],
    uint id                             [[thread_position_in_grid]]
) {
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

kernel void conv1d_3(
    device const uchar *conv_w_base     [[buffer(0)]],
    device const float *state           [[buffer(1)]],
    device float       *out             [[buffer(2)]],
    constant uint      &w_rem           [[buffer(3)]],
    uint c                              [[thread_position_in_grid]]
) {
    uint base = w_rem + c * 3 * 2;
    float w0 = float(read_bf16(conv_w_base, base + 0));
    float w1 = float(read_bf16(conv_w_base, base + 2));
    float w2 = float(read_bf16(conv_w_base, base + 4));

    device const float *st = state + c * 3;
    float sum = w0 * st[0] + w1 * st[1] + w2 * st[2];
    out[c] = sum;
}

kernel void gemv_mlx4_affine(
    device const uchar  *w_base         [[buffer(0)]],
    device const uchar  *scales_base    [[buffer(1)]],
    device const uchar  *biases_base    [[buffer(2)]],
    device const float  *x              [[buffer(3)]],
    device float        *y              [[buffer(4)]],
    constant uint       &cols           [[buffer(5)]],
    constant uint       &group_size     [[buffer(6)]],
    constant uint4      &rems           [[buffer(7)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint u32_cols = cols / 8;
    uint n_groups = cols / group_size;

    uint qw_row_byte = rems.x + row * (u32_cols * 4);
    uint s_row_byte  = rems.y + row * (n_groups * 2);
    uint b_row_byte  = rems.z + row * (n_groups * 2);

    for (uint col_u32 = lane; col_u32 < u32_cols; col_u32 += 32) {
        uint val = read_u32(w_base, qw_row_byte + col_u32 * 4);
        uint c_base = col_u32 * 8;
        uint g_idx = c_base / group_size;
        float s = float(read_bf16(scales_base, s_row_byte + g_idx * 2));
        float b = float(read_bf16(biases_base, b_row_byte + g_idx * 2));

        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint nibble = (val >> (i * 4)) & 0x0F;
            float w_val = float(nibble) * s + b;
            sum += w_val * x[c_base + i];
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
        ctx->pending_cmd = nil;
        ctx->pending_y_bytes = 0;
        ctx->pending_y_dst = nullptr;

        return (void*)ctx;
    }
}

void sofuu_metal_sync(void* ctx_ptr);

void sofuu_metal_free(void* ctx_ptr) {
    if (!ctx_ptr) return;
    sofuu_metal_sync(ctx_ptr);
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

void* sofuu_metal_buffer_create(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        if (((uintptr_t)ptr % 4096) == 0) {
            id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                               length:size_bytes
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
            if (buf != nil) {
                return (__bridge_retained void*)buf;
            }
        }
        id<MTLBuffer> buf = [ctx->device newBufferWithBytes:ptr
                                                     length:size_bytes
                                                    options:MTLResourceStorageModeShared];
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

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_bf16];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_qkv_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t q_offset,
    size_t k_offset,
    size_t v_offset,
    const float* x,
    float* q_out,
    float* k_out,
    float* v_out,
    size_t q_rows,
    size_t k_rows,
    size_t v_rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t q_bytes = q_rows * sizeof(float);
        size_t k_bytes = k_rows * sizeof(float);
        size_t v_bytes = v_rows * sizeof(float);

        size_t q_slot = (q_bytes + 255) & ~255ULL;
        size_t k_slot = (k_bytes + 255) & ~255ULL;
        size_t total_y_bytes = q_slot + k_slot + v_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        // 1. Q projection
        if (q_out && q_rows > 0) {
            size_t aligned_q = q_offset & ~255ULL;
            uint rem_q = (uint)(q_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_q atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_q length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(q_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 2. K projection
        if (k_out && k_rows > 0) {
            size_t aligned_k = k_offset & ~255ULL;
            uint rem_k = (uint)(k_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_k atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_k length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(k_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 3. V projection
        if (v_out && v_rows > 0) {
            size_t aligned_v = v_offset & ~255ULL;
            uint rem_v = (uint)(v_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_v atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:(q_slot + k_slot) atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_v length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(v_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (q_out && q_rows > 0) {
            std::memcpy(q_out, y_ptr, q_bytes);
        }
        if (k_out && k_rows > 0) {
            std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        }
        if (v_out && v_rows > 0) {
            std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
        }
    }
}

void sofuu_metal_gemv_gate_up_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t gate_offset,
    size_t up_offset,
    const float* x,
    float* gate_out,
    float* up_out,
    size_t inter_rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || cols == 0 || inter_rows == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t row_bytes = inter_rows * sizeof(float);
        size_t slot = (row_bytes + 255) & ~255ULL;
        size_t total_y_bytes = slot + row_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        // 1. Gate projection
        if (gate_out) {
            size_t aligned_g = gate_offset & ~255ULL;
            uint rem_g = (uint)(gate_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_g atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_g length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 2. Up projection
        if (up_out) {
            size_t aligned_u = up_offset & ~255ULL;
            uint rem_u = (uint)(up_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_u atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_u length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (gate_out) {
            std::memcpy(gate_out, y_ptr, row_bytes);
        }
        if (up_out) {
            std::memcpy(up_out, y_ptr + slot, row_bytes);
        }
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

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_f16];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

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

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_f32];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

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

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_rmsnorm];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint d = (uint)dim;
        [enc setBytes:&d length:sizeof(uint) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:5];

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

        size_t aligned_offset = conv_w_offset & ~255ULL;
        uint w_rem = (uint)(conv_w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_conv1d_3];
        [enc setBuffer:conv_w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:3];

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

        size_t w_aligned = w_offset & ~255ULL;
        size_t scales_aligned = scales_offset & ~255ULL;
        size_t biases_aligned = biases_offset & ~255ULL;
        uint32_t rems[4] = {
            (uint32_t)(w_offset & 255ULL),
            (uint32_t)(scales_offset & 255ULL),
            (uint32_t)(biases_offset & 255ULL),
            0
        };

        [enc setComputePipelineState:ctx->pso_gemv_mlx4];
        [enc setBuffer:w_buf offset:w_aligned atIndex:0];
        [enc setBuffer:scales_buf offset:scales_aligned atIndex:1];
        [enc setBuffer:biases_buf offset:biases_aligned atIndex:2];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:3];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:4];
        uint c = (uint)cols;
        uint g = (uint)group_size;
        [enc setBytes:&c length:sizeof(uint) atIndex:5];
        [enc setBytes:&g length:sizeof(uint) atIndex:6];
        [enc setBytes:rems length:sizeof(rems) atIndex:7];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_qkv_multi(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset, size_t q_rows,
    void* k_buf_ptr, size_t k_offset, size_t k_rows,
    void* v_buf_ptr, size_t v_offset, size_t v_rows,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || cols == 0 || !x) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> q_buf = q_buf_ptr ? (__bridge id<MTLBuffer>)q_buf_ptr : nil;
    id<MTLBuffer> k_buf = k_buf_ptr ? (__bridge id<MTLBuffer>)k_buf_ptr : q_buf;
    id<MTLBuffer> v_buf = v_buf_ptr ? (__bridge id<MTLBuffer>)v_buf_ptr : q_buf;
    if (!q_buf && !k_buf && !v_buf) return;

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t q_bytes = q_rows * sizeof(float);
        size_t k_bytes = k_rows * sizeof(float);
        size_t v_bytes = v_rows * sizeof(float);

        size_t q_slot = (q_bytes + 255) & ~255ULL;
        size_t k_slot = (k_bytes + 255) & ~255ULL;
        size_t total_y_bytes = q_slot + k_slot + v_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        if (q_out && q_rows > 0 && q_buf != nil) {
            size_t aligned_q = q_offset & ~255ULL;
            uint rem_q = (uint)(q_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:q_buf offset:aligned_q atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_q length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(q_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (k_out && k_rows > 0 && k_buf != nil) {
            size_t aligned_k = k_offset & ~255ULL;
            uint rem_k = (uint)(k_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:k_buf offset:aligned_k atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_k length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(k_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (v_out && v_rows > 0 && v_buf != nil) {
            size_t aligned_v = v_offset & ~255ULL;
            uint rem_v = (uint)(v_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:v_buf offset:aligned_v atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot + k_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_v length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(v_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (q_out && q_rows > 0) std::memcpy(q_out, y_ptr, q_bytes);
        if (k_out && k_rows > 0) std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        if (v_out && v_rows > 0) std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
    }
}

void sofuu_metal_gemv_gate_up_multi(
    void* ctx_ptr,
    void* gate_buf_ptr, size_t gate_offset,
    void* up_buf_ptr, size_t up_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t inter_rows,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || cols == 0 || !x || inter_rows == 0) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> g_buf = gate_buf_ptr ? (__bridge id<MTLBuffer>)gate_buf_ptr : nil;
    id<MTLBuffer> u_buf = up_buf_ptr ? (__bridge id<MTLBuffer>)up_buf_ptr : g_buf;
    if (!g_buf && !u_buf) return;

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t row_bytes = inter_rows * sizeof(float);
        size_t slot = (row_bytes + 255) & ~255ULL;
        size_t total_y_bytes = slot + row_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        if (gate_out && g_buf != nil) {
            size_t aligned_g = gate_offset & ~255ULL;
            uint rem_g = (uint)(gate_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:g_buf offset:aligned_g atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_g length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (up_out && u_buf != nil) {
            size_t aligned_u = up_offset & ~255ULL;
            uint rem_u = (uint)(up_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:u_buf offset:aligned_u atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_u length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (gate_out) std::memcpy(gate_out, y_ptr, row_bytes);
        if (up_out) std::memcpy(up_out, y_ptr + slot, row_bytes);
    }
}

void sofuu_metal_sync(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->pending_cmd != nil) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = ctx->pending_cmd;
            [cmd waitUntilCompleted];
            if (ctx->pending_y_dst && ctx->pending_y_bytes > 0) {
                std::memcpy(ctx->pending_y_dst, ctx->y_buf.contents, ctx->pending_y_bytes);
            }
            ctx->pending_cmd = nil;
            ctx->pending_y_dst = nullptr;
            ctx->pending_y_bytes = 0;
            CFRelease((__bridge CFTypeRef)cmd);
        }
    }
}

void sofuu_metal_gemv_async(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;

    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;
    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        ctx->pending_cmd = (__bridge id<MTLCommandBuffer>)(void*)CFBridgingRetain(cmd);
        ctx->pending_y_dst = y;
        ctx->pending_y_bytes = y_bytes;
    }
}

#else

// Stubs for non-macOS platforms (Linux, Android, Windows)
bool sofuu_metal_available(void) { return false; }
void* sofuu_metal_init(void) { return NULL; }
void sofuu_metal_free(void* ctx) {}
void* sofuu_metal_buffer_create_zerocopy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void* sofuu_metal_buffer_create(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void sofuu_metal_buffer_free(void* buf) {}
void sofuu_metal_gemv_bf16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f32(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_rmsnorm(void* ctx, const float* x, void* w_buf, size_t w_offset, float* y, size_t dim, float eps) {}
void sofuu_metal_swiglu(void* ctx, float* g, const float* u, size_t size) {}
void sofuu_metal_conv1d_3(void* ctx, void* conv_w_buf, size_t conv_w_offset, const float* state, float* out, size_t channels) {}
void sofuu_metal_gemv_mlx4(void* ctx, void* w_buf, size_t w_offset, void* scales_buf, size_t scales_offset, void* biases_buf, size_t biases_offset, const float* x, float* y, size_t rows, size_t cols, size_t group_size) {}
void sofuu_metal_gemv_qkv_multi(void* ctx, void* q_buf, size_t q_offset, size_t q_rows, void* k_buf, size_t k_offset, size_t k_rows, void* v_buf, size_t v_offset, size_t v_rows, const float* x, float* q_out, float* k_out, float* v_out, size_t cols, int dtype) {}
void sofuu_metal_gemv_gate_up_multi(void* ctx, void* gate_buf, size_t gate_offset, void* up_buf, size_t up_offset, const float* x, float* gate_out, float* up_out, size_t inter_rows, size_t cols, int dtype) {}
void sofuu_metal_gemv_async(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols, int dtype) {}
void sofuu_metal_sync(void* ctx) {}

#endif

