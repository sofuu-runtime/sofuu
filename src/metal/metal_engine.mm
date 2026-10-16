#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_engine.h"
#include <cstring>
#include <iostream>
#include <unistd.h>

#if defined(__APPLE__)

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso_gemv_bf16;
    id<MTLComputePipelineState> pso_gemv_f16;
    id<MTLComputePipelineState> pso_gemv_f32;
    id<MTLComputePipelineState> pso_gemv_mlx4_f16;
    id<MTLComputePipelineState> pso_gemv_mlx4_bf16;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_conv1d_3;

    id<MTLBuffer> x_buf;
    id<MTLBuffer> y_buf;
    size_t x_buf_cap;
    size_t y_buf_cap;

    id<MTLCommandBuffer> pending_cmd;
    id<MTLComputeCommandEncoder> batch_enc;
    size_t pending_y_bytes;
    float* pending_y_dst;
};

// The shader source lives in src/metal/kernels.metal; build.rs generates it
// into this header so the RUNTIME-compiled path and the shipped
// default.metallib (scripts/dist/ios.sh) can never disagree about which
// kernels exist. See the note in crates/sofuu-ffi/build.rs.
#include "kernels_metal.h"

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

        // Two ways to obtain the compute pipelines, tried in this order:
        //
        //   1. `default.metallib` in the app bundle — kernels.metal compiled
        //      at BUILD time. Xcode compiles the .metal files in the app
        //      target into the bundle's default library; the iOS framework
        //      ships one in Resources/ as well (scripts/dist/ios.sh). No
        //      runtime shader compilation, and a shader that fails to build
        //      is caught by the build, not by a user's first prompt.
        //   2. The same source compiled at RUNTIME. This is the only path
        //      the macOS CLI has (it ships no bundle) and the fallback
        //      everywhere else.
        //
        // A bundled library is accepted only when EVERY pipeline the engine
        // loads is present: a stale or partial metallib must degrade to (2),
        // never to a nil pipeline state at inference time.
        //
        // iOS floor note: the bf16 kernels use `bfloat`, which the Metal
        // language gained in MSL 3.1 (iOS 17). On an older OS the bundled
        // library will not load AND the runtime compiler does not know the
        // type, so init returns NULL and the engine stays on the CPU path —
        // the designed degradation, logged below.
        NSArray<NSString*>* required = @[
            @"gemv_bf16", @"gemv_f16", @"gemv_f32",
            @"gemv_mlx4_affine_f16", @"gemv_mlx4_affine_bf16",
            @"rmsnorm", @"swiglu", @"conv1d_3"
        ];

        id<MTLLibrary> lib = nil;
        id<MTLLibrary> bundled = [dev newDefaultLibrary];
        if (bundled != nil) {
            BOOL complete = YES;
            for (NSString* name in required) {
                if ([bundled newFunctionWithName:name] == nil) {
                    complete = NO;
                    break;
                }
            }
            if (complete) lib = bundled;
        }

        NSError* error = nil;
        if (lib == nil) {
            NSString* src = [NSString stringWithUTF8String:METAL_KERNEL_SOURCE];
            lib = [dev newLibraryWithSource:src options:nil error:&error];
            if (lib == nil) {
                NSLog(@"Sofuu Metal: no usable shader library — bundled "
                      @"default.metallib missing or incomplete, and runtime "
                      @"compile failed (%@). bf16 kernels need MSL 3.1 / "
                      @"iOS 17+; running on CPU.", error);
                return NULL;
            }
            NSLog(@"Sofuu Metal: kernels compiled at runtime (no complete "
                  @"default.metallib in the bundle)");
        } else {
            NSLog(@"Sofuu Metal: using the bundled default.metallib "
                  @"(precompiled kernels)");
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
        ctx->pso_gemv_mlx4_f16  = load_pso(@"gemv_mlx4_affine_f16");
        ctx->pso_gemv_mlx4_bf16 = load_pso(@"gemv_mlx4_affine_bf16");
        ctx->pso_rmsnorm   = load_pso(@"rmsnorm");
        ctx->pso_swiglu    = load_pso(@"swiglu");
        ctx->pso_conv1d_3  = load_pso(@"conv1d_3");

        // Initial 64KB activation buffers
        ctx->x_buf_cap = 65536 * sizeof(float);
        ctx->y_buf_cap = 65536 * sizeof(float);
        ctx->x_buf = [dev newBufferWithLength:ctx->x_buf_cap options:MTLResourceStorageModeShared];
        ctx->y_buf = [dev newBufferWithLength:ctx->y_buf_cap options:MTLResourceStorageModeShared];
        ctx->pending_cmd = nil;
        ctx->batch_enc = nil;
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
#if TARGET_OS_SIMULATOR
    /* A no-copy buffer here exports the memory over XPC shared memory, and
     * libxpc aborts the process (_xpc_api_misuse) instead of failing when
     * the range cannot be exported as one mapping — which is the case for
     * the allocator memory callers pass in. See the measured note in
     * sofuu_metal_buffer_create. Returning NULL sends the caller there,
     * where the driver's copying path runs instead. */
    return NULL;
#else
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        if (page_size == 0) page_size = 16384;
        if (((uintptr_t)ptr % page_size) != 0) return NULL;
        size_t aligned_len = ((size_bytes + page_size - 1) / page_size) * page_size;
        id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                           length:aligned_len
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        return (__bridge_retained void*)buf;
    }
#endif
}

void* sofuu_metal_buffer_create(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
#if TARGET_OS_SIMULATOR
        /* MTLSimDriver exports every buffer's storage over XPC shared memory.
         * Its no-copy path hands our pointer straight to xpc_shmem_create,
         * and libxpc TRAPS (_xpc_api_misuse — it aborts the process rather
         * than returning an error) unless [ptr, ptr+length) lies inside one
         * exportable mapping. Memory from the system allocator does not
         * qualify: measured in a simulator probe, a 2 MiB malloc returns a
         * page-aligned pointer whose enclosing vm region reports as a single
         * 16 KiB page, and the export aborts. Every Rust Vec the engine
         * copies from is such an allocation, so the whole no-copy path is
         * unusable here.
         *
         * Its copying path is safe because the driver mmaps its own
         * MAP_SHARED|MAP_ANON region for the storage and exports that
         * instead — verified up to 32 MiB, including lengths that are not
         * page multiples. The simulator therefore gets a copied buffer: GPU
         * math still runs, and the only thing lost is the zero-copy
         * shortcut MTLSimDriver cannot honour for allocator memory.
         * (Reached from sofuu_metal_buffer_create_zerocopy too — it returns
         * NULL under TARGET_OS_SIMULATOR and the Rust wrapper falls back
         * here.) */
        return (__bridge_retained void*)[ctx->device newBufferWithBytes:ptr
                                                                 length:size_bytes
                                                                options:MTLResourceStorageModeShared];
#else
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        if (page_size == 0) page_size = 16384;
        if (((uintptr_t)ptr % page_size) == 0) {
            size_t aligned_len = ((size_bytes + page_size - 1) / page_size) * page_size;
            id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                               length:aligned_len
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
#endif
    }
}


void sofuu_metal_buffer_free(void* buf_ptr) {
    if (!buf_ptr) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_ptr;
        (void)buf;
    }
}

/* waitUntilCompleted returns whether or not the GPU accepted the work:
 * an unsupported instruction, an out-of-range access or a failed allocation
 * leaves status == Error and the output buffer untouched. Without this check
 * the callers read that untouched buffer back and return zeros as if they
 * were results — silent wrong answers instead of a loud failure. (Measured:
 * a bf16 kernel on the iOS simulator fails this way with
 * MTLCommandBufferErrorInternal, while f16/f32 kernels complete.) */
static inline void wait_and_report(id<MTLCommandBuffer> cmd, const char* kernel) {
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
        NSLog(@"Sofuu Metal: %s command buffer failed: %@", kernel, cmd.error);
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
        wait_and_report(cmd, "gemv_bf16");

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
        wait_and_report(cmd, "gemv_qkv_bf16");

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
        wait_and_report(cmd, "gemv_gate_up_bf16");

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
        wait_and_report(cmd, "gemv_f16");

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
        wait_and_report(cmd, "gemv_f32");

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
        wait_and_report(cmd, "rmsnorm");

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
        wait_and_report(cmd, "swiglu");

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
        wait_and_report(cmd, "conv1d_3");

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
    size_t group_size,
    int is_bf16
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

        id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;
        [enc setComputePipelineState:pso];
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
        wait_and_report(cmd, is_bf16 ? "gemv_mlx4_bf16" : "gemv_mlx4_f16");

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

static inline void dispatch_mlx4_gemv(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso,
    id<MTLBuffer> w_buf, size_t w_offset,
    id<MTLBuffer> scales_buf, size_t scales_offset,
    id<MTLBuffer> biases_buf, size_t biases_offset,
    id<MTLBuffer> x_buf,
    id<MTLBuffer> y_buf, size_t y_offset,
    size_t rows, size_t cols, size_t group_size
) {
    size_t w_aligned = w_offset & ~255ULL;
    size_t scales_aligned = scales_offset & ~255ULL;
    size_t biases_aligned = biases_offset & ~255ULL;
    uint32_t rems[4] = {
        (uint32_t)(w_offset & 255ULL),
        (uint32_t)(scales_offset & 255ULL),
        (uint32_t)(biases_offset & 255ULL),
        0
    };
    uint c = (uint)cols;
    uint g = (uint)group_size;

    [enc setComputePipelineState:pso];
    [enc setBuffer:w_buf offset:w_aligned atIndex:0];
    [enc setBuffer:scales_buf offset:scales_aligned atIndex:1];
    [enc setBuffer:biases_buf offset:biases_aligned atIndex:2];
    [enc setBuffer:x_buf offset:0 atIndex:3];
    [enc setBuffer:y_buf offset:y_offset atIndex:4];
    [enc setBytes:&c length:sizeof(uint) atIndex:5];
    [enc setBytes:&g length:sizeof(uint) atIndex:6];
    [enc setBytes:rems length:sizeof(rems) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_gemv_qkv_mlx4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset, size_t q_rows,
    void* q_scales_ptr, size_t q_scales_offset,
    void* q_biases_ptr, size_t q_biases_offset,
    void* k_buf_ptr, size_t k_offset, size_t k_rows,
    void* k_scales_ptr, size_t k_scales_offset,
    void* k_biases_ptr, size_t k_biases_offset,
    void* v_buf_ptr, size_t v_offset, size_t v_rows,
    void* v_scales_ptr, size_t v_scales_offset,
    void* v_biases_ptr, size_t v_biases_offset,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr || !q_buf_ptr || !k_buf_ptr || !v_buf_ptr || !q_scales_ptr || !k_scales_ptr || !v_scales_ptr ||
        !q_biases_ptr || !k_biases_ptr || !v_biases_ptr || !x || !q_out || !k_out || !v_out ||
        !q_rows || !k_rows || !v_rows || !cols) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;
    pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

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
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)q_buf_ptr, q_offset,
            (__bridge id<MTLBuffer>)q_scales_ptr, q_scales_offset,
            (__bridge id<MTLBuffer>)q_biases_ptr, q_biases_offset,
            ctx->x_buf, ctx->y_buf, 0, q_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)k_buf_ptr, k_offset,
            (__bridge id<MTLBuffer>)k_scales_ptr, k_scales_offset,
            (__bridge id<MTLBuffer>)k_biases_ptr, k_biases_offset,
            ctx->x_buf, ctx->y_buf, q_slot, k_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)v_buf_ptr, v_offset,
            (__bridge id<MTLBuffer>)v_scales_ptr, v_scales_offset,
            (__bridge id<MTLBuffer>)v_biases_ptr, v_biases_offset,
            ctx->x_buf, ctx->y_buf, q_slot + k_slot, v_rows, cols, group_size);
        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, is_bf16 ? "gemv_qkv_mlx4_bf16" : "gemv_qkv_mlx4_f16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        std::memcpy(q_out, y_ptr, q_bytes);
        std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
    }
}

void sofuu_metal_gemv_gate_up_mlx4(
    void* ctx_ptr,
    void* gate_buf_ptr, size_t gate_offset, size_t gate_rows,
    void* gate_scales_ptr, size_t gate_scales_offset,
    void* gate_biases_ptr, size_t gate_biases_offset,
    void* up_buf_ptr, size_t up_offset, size_t up_rows,
    void* up_scales_ptr, size_t up_scales_offset,
    void* up_biases_ptr, size_t up_biases_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr || !gate_buf_ptr || !up_buf_ptr || !gate_scales_ptr || !up_scales_ptr ||
        !gate_biases_ptr || !up_biases_ptr || !x || !gate_out || !up_out ||
        !gate_rows || !up_rows || !cols) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t gate_bytes = gate_rows * sizeof(float);
        size_t up_bytes = up_rows * sizeof(float);
        size_t gate_slot = (gate_bytes + 255) & ~255ULL;
        size_t total_y_bytes = gate_slot + up_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);
        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)gate_buf_ptr, gate_offset,
            (__bridge id<MTLBuffer>)gate_scales_ptr, gate_scales_offset,
            (__bridge id<MTLBuffer>)gate_biases_ptr, gate_biases_offset,
            ctx->x_buf, ctx->y_buf, 0, gate_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)up_buf_ptr, up_offset,
            (__bridge id<MTLBuffer>)up_scales_ptr, up_scales_offset,
            (__bridge id<MTLBuffer>)up_biases_ptr, up_biases_offset,
            ctx->x_buf, ctx->y_buf, gate_slot, up_rows, cols, group_size);
        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, is_bf16 ? "gemv_gate_up_mlx4_bf16" : "gemv_gate_up_mlx4_f16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        std::memcpy(gate_out, y_ptr, gate_bytes);
        std::memcpy(up_out, y_ptr + gate_slot, up_bytes);
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
        wait_and_report(cmd, "gemv_qkv_multi (f16/f32 by dtype)");

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
        wait_and_report(cmd, "gemv_gate_up_multi (f16/f32 by dtype)");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (gate_out) std::memcpy(gate_out, y_ptr, row_bytes);
        if (up_out) std::memcpy(up_out, y_ptr + slot, row_bytes);
    }
}

void sofuu_metal_sync(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc != nil) {
        [ctx->batch_enc endEncoding];
        CFRelease((__bridge CFTypeRef)ctx->batch_enc);
        ctx->batch_enc = nil;
        if (ctx->pending_cmd != nil) [ctx->pending_cmd commit];
    }
    if (ctx->pending_cmd != nil) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = ctx->pending_cmd;
            wait_and_report(cmd, "async gemv (deferred)");
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

// ── G0: real multi-op batching ─────────────────────────────────────────────
// batch_open retains ONE uncommitted command buffer with a LIVE encoder;
// every batch_gemv appends an op reading caller-owned MTLBuffers (so ops can
// chain y→x with zero CPU staging); batch_end ends the encoder, commits once,
// waits. One GPU submission per batch, independent of op count.
int sofuu_metal_batch_open(void* ctx_ptr) {
    fprintf(stderr, "[G0BATCH-LIB-V3] open\n");
    if (!ctx_ptr) return -1;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    sofuu_metal_sync(ctx_ptr);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        ctx->batch_enc = (__bridge id<MTLComputeCommandEncoder>)(void*)CFBridgingRetain([cmd computeCommandEncoder]);
        ctx->pending_cmd = (__bridge id<MTLCommandBuffer>)(void*)CFBridgingRetain(cmd);
        ctx->pending_y_dst = nullptr;
        ctx->pending_y_bytes = 0;
        if (ctx->batch_enc == nil) return -1;
    }
    return 0;
}

void sofuu_metal_batch_gemv(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, int dtype
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    // Cross-dispatch buffer visibility needs an explicit barrier; ops chain
    // y->x inside ONE encoder (one commit per batch).
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_end(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) { sofuu_metal_sync(ctx_ptr); return; }
    @autoreleasepool {
        [ctx->batch_enc endEncoding];
        CFRelease((__bridge CFTypeRef)ctx->batch_enc);
        ctx->batch_enc = nil;
        id<MTLCommandBuffer> cmd = ctx->pending_cmd;
        if (cmd != nil) {
            [cmd commit];
            wait_and_report(cmd, "batch gemv");
            ctx->pending_cmd = nil;
            CFRelease((__bridge CFTypeRef)cmd);
        }
    }
}

#else

// Stubs for non-macOS platforms (Linux, Android, Windows)
bool sofuu_metal_available(void) { return false; }
int sofuu_metal_batch_open(void* ctx) { (void)ctx; return -1; }
void sofuu_metal_batch_gemv(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, int d) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)d; }
void sofuu_metal_batch_end(void* ctx) { (void)ctx; }
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
void sofuu_metal_gemv_mlx4(void* ctx, void* w_buf, size_t w_offset, void* scales_buf, size_t scales_offset, void* biases_buf, size_t biases_offset, const float* x, float* y, size_t rows, size_t cols, size_t group_size, int is_bf16) {}
void sofuu_metal_gemv_qkv_multi(void* ctx, void* q_buf, size_t q_offset, size_t q_rows, void* k_buf, size_t k_offset, size_t k_rows, void* v_buf, size_t v_offset, size_t v_rows, const float* x, float* q_out, float* k_out, float* v_out, size_t cols, int dtype) {}
void sofuu_metal_gemv_gate_up_multi(void* ctx, void* gate_buf, size_t gate_offset, void* up_buf, size_t up_offset, const float* x, float* gate_out, float* up_out, size_t inter_rows, size_t cols, int dtype) {}
void sofuu_metal_gemv_async(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols, int dtype) {}
void sofuu_metal_sync(void* ctx) {}
void sofuu_metal_gemv_qkv_mlx4(void* ctx, void* q_buf, size_t q_offset, size_t q_rows, void* q_scales_buf, size_t q_scales_offset, void* q_biases_buf, size_t q_biases_offset, void* k_buf, size_t k_offset, size_t k_rows, void* k_scales_buf, size_t k_scales_offset, void* k_biases_buf, size_t k_biases_offset, void* v_buf, size_t v_offset, size_t v_rows, void* v_scales_buf, size_t v_scales_offset, void* v_biases_buf, size_t v_biases_offset, const float* x, float* q_out, float* k_out, float* v_out, size_t cols, size_t group_size, int is_bf16) {}
void sofuu_metal_gemv_gate_up_mlx4(void* ctx, void* gate_buf, size_t gate_offset, size_t gate_rows, void* gate_scales_buf, size_t gate_scales_offset, void* gate_biases_buf, size_t gate_biases_offset, void* up_buf, size_t up_offset, size_t up_rows, void* up_scales_buf, size_t up_scales_offset, void* up_biases_buf, size_t up_biases_offset, const float* x, float* gate_out, float* up_out, size_t cols, size_t group_size, int is_bf16) {}

#endif

