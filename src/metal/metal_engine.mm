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
    // GPU1: in-batch glue kernels (whole-decode-step batches).
    id<MTLComputePipelineState> pso_add_inplace;
    id<MTLComputePipelineState> pso_lfm2_conv_gate;
    id<MTLComputePipelineState> pso_lfm2_kv_write;
    id<MTLComputePipelineState> pso_lfm2_attn;
    // G2: GGUF dequant-dot GEMV kernels.
    id<MTLComputePipelineState> pso_gemv_q8_0;
    id<MTLComputePipelineState> pso_gemv_q4_0;
    id<MTLComputePipelineState> pso_gemv_q4_k;
    id<MTLComputePipelineState> pso_gemv_q6_k;
    id<MTLComputePipelineState> pso_gemv_q4_k_b4;

    // GPU2.1: one-commit-per-layer llama kernels.
    id<MTLComputePipelineState> pso_rmsnorm_f32w;
    id<MTLComputePipelineState> pso_rmsnorm_f32w_b4;
    id<MTLComputePipelineState> pso_llama_kv_write;
    id<MTLComputePipelineState> pso_llama_attn;
    id<MTLComputePipelineState> pso_head_rmsnorm;
    id<MTLComputePipelineState> pso_llama_attn_part;
    id<MTLComputePipelineState> pso_llama_attn_fin;
    id<MTLComputePipelineState> pso_llama_kv_write_b4;
    id<MTLComputePipelineState> pso_head_rmsnorm_b4;
    id<MTLComputePipelineState> pso_llama_attn_part_b4;
    id<MTLComputePipelineState> pso_llama_attn_fin_b4;
    id<MTLComputePipelineState> pso_llama_attn_1c_b4;
    id<MTLComputePipelineState> pso_gemv_q4_k_b4c2r;
    id<MTLComputePipelineState> pso_gemm_q4_k;
    id<MTLComputePipelineState> pso_gemm_q4_k_multi;
    id<MTLComputePipelineState> pso_gemm_q4_k_acc;
    id<MTLComputePipelineState> pso_gpu_q4_k_b4multi;
    id<MTLComputePipelineState> pso_gpu_q6_k_b4multi;
    id<MTLComputePipelineState> pso_gpu_q8_0_b4multi;
    bool pso_gemm_q4_k_live;
    bool pso_gemm_q4_k_multi_live;
    bool pso_gemm_q4_k_acc_live;
    bool pso_gpu_q4_k_b4multi_live;
    bool pso_gpu_q6_k_b4multi_live;
    bool pso_gpu_q8_0_b4multi_live;

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
            @"rmsnorm", @"swiglu", @"conv1d_3",
            @"add_inplace", @"lfm2_conv_gate", @"lfm2_kv_write", @"lfm2_attn",
            @"gemv_q8_0", @"gemv_q4_0", @"gemv_q4_k", @"gemv_q6_k",
            @"rmsnorm_f32w", @"llama_kv_write", @"llama_attn",
            @"head_rmsnorm", @"llama_attn_part", @"llama_attn_fin",
            @"gemv_q4_k_b4"
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

        // env-gated q4_K kernel class: coalesced-lane vs shipped scalar
        BOOL q4k_coalesced = NO;
        auto load_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];            if (fn == nil) return nil;
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
        ctx->pso_add_inplace     = load_pso(@"add_inplace");
        ctx->pso_lfm2_conv_gate  = load_pso(@"lfm2_conv_gate");
        ctx->pso_lfm2_kv_write   = load_pso(@"lfm2_kv_write");
        ctx->pso_lfm2_attn       = load_pso(@"lfm2_attn");
        ctx->pso_gemv_q8_0 = load_pso(@"gemv_q8_0");
        ctx->pso_gemv_q4_0 = load_pso(@"gemv_q4_0");
        ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k");
        // SOFUU_Q4K_COALESCED=1 flips BOTH q4_K PSOs to the coalesced-lane
        // kernels — one kernel class per engine instance so serial, spec and
        // b4 GEMVs all share the same f32 summation order (text parity holds
        // internally; the coalesced class has its own CPU-oracle parity gate).
        const char* q4k_class = getenv("SOFUU_Q4K_COALESCED");
        int coalesced_level = (q4k_class && strcmp(q4k_class, "0") != 0) ? atoi(q4k_class) : 0;
        q4k_coalesced = coalesced_level > 0;
        ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
        ctx->pso_rmsnorm_f32w  = load_pso(@"rmsnorm_f32w");
        ctx->pso_rmsnorm_f32w_b4 = load_pso(@"rmsnorm_f32w_b4");
        ctx->pso_llama_kv_write = load_pso(@"llama_kv_write");
        ctx->pso_llama_attn     = load_pso(@"llama_attn");
        ctx->pso_head_rmsnorm    = load_pso(@"head_rmsnorm");
        ctx->pso_llama_attn_part = load_pso(@"llama_attn_part");
        ctx->pso_llama_attn_fin  = load_pso(@"llama_attn_fin");
        ctx->pso_llama_kv_write_b4   = load_pso(@"llama_kv_write_b4");
        ctx->pso_head_rmsnorm_b4     = load_pso(@"head_rmsnorm_b4");
        ctx->pso_llama_attn_part_b4  = load_pso(@"llama_attn_part_b4");
        ctx->pso_llama_attn_fin_b4   = load_pso(@"llama_attn_fin_b4");
        ctx->pso_llama_attn_1c_b4    = load_pso(@"llama_attn_1c_b4");
        ctx->pso_gemv_q4_k_b4c2r     = load_pso(@"gemv_q4_k_b4c2r");
        // class 1 = byte-lane coalesced; class 2 = 16B-span vec4 coalesced
        // (the faster class: b4c2 100 GB/s vs 69, head q6_K 90.5 vs the
        // v1 q6_K 15.5). BOTH the single-row (serial client) and batched
        // (spec chain) q4_K PSOs flip together so every GEMV in a run
        // shares one summation order.
        if (coalesced_level == 2) {
            // level 2 = BEST PER SHAPE (2x2 matrix receipt: byte-lane b4c
            // 107/50 GB/s compact/5G vs vec4 46/37 — but the vec4 WINS the
            // single-row shape as the 12.4 t/s serial receipt showed):
            // batched pass GEMVs = byte-lane b4c, single-row = vec4 c2.
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k_c2");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4c");
        } else if (coalesced_level == 1) {
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k_c");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4c");
        }
        if (coalesced_level > 0) {
            // the q6_K class rides the same env: the ~514MB lm_head runs on
            // the scalar q6_K at 15.6 GB/s (32.9ms) vs 92 GB/s (5.5ms)
            // coalesced
            ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k_c");
            if (ctx->pso_gemv_q6_k == nil) {
                // bundled metallib predates it: degrade to the shipped
                // q6_K class rather than a silent no-op GEMV
                ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
            }
        }
        if (ctx->pso_gemv_q4_k == nil || ctx->pso_gemv_q4_k_b4 == nil) {
            // coalesced class unavailable (e.g. a complete bundled metallib
            // predating it): degrade to the shipped class, never a silent
            // no-op GEMV
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4");
            if (coalesced_level > 0 && ctx->pso_gemv_q6_k == nil) {
                ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
            }
        }
        if (ctx->pso_gemv_q6_k == nil) {
            ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
        }
        // prefill GEMM class (P1 of llama.cpp parity): if the bundled
        // metallib predates the kernel the live flag stays false and the
        // GEMM prefill route never engages — visible unavailable, never a
        // silent no-op GEMM
        ctx->pso_gemm_q4_k = load_pso(@"gemm_q4_k");
        ctx->pso_gemm_q4_k_live = ctx->pso_gemm_q4_k != nil;
        ctx->pso_gemm_q4_k_multi = load_pso(@"gemm_q4_k_multi");
        ctx->pso_gemm_q4_k_multi_live = ctx->pso_gemm_q4_k_multi != nil;
        ctx->pso_gemm_q4_k_acc = load_pso(@"gemm_q4_k_acc");
        ctx->pso_gemm_q4_k_acc_live = ctx->pso_gemm_q4_k_acc != nil;
        ctx->pso_gpu_q4_k_b4multi = load_pso(@"gemv_q4_k_b4multi");
        ctx->pso_gpu_q4_k_b4multi_live = ctx->pso_gpu_q4_k_b4multi != nil;
        ctx->pso_gpu_q6_k_b4multi = load_pso(@"gemv_q6_k_b4multi");
        ctx->pso_gpu_q6_k_b4multi_live = ctx->pso_gpu_q6_k_b4multi != nil;
        ctx->pso_gpu_q8_0_b4multi = load_pso(@"gemv_q8_0_b4multi");
        ctx->pso_gpu_q8_0_b4multi_live = ctx->pso_gpu_q8_0_b4multi != nil;

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


void* sofuu_metal_buffer_create_copy(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        // Always a driver-managed copy (never wraps caller memory): the GPU
        // decode clients use this to test whether device-owned placement
        // streams weights faster than a wrapped host mmap.
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
    // SOFUU_GPU_TIMING=1: device-side execution time of every command
    // buffer (GPUStartTime/GPUEndTime) — the GPU-busy vs CPU-wall split
    // that attributes the spec-pass overhead without an Instruments trace
    static int timing = -1;
    if (timing < 0) { const char* t = getenv("SOFUU_GPU_TIMING"); timing = (t && atoi(t) != 0) ? 1 : 0; }
    if (timing) {
        fprintf(stderr, "SOFUU_CMD_TIMING kernel=%s gpu_us=%.0f\n",
                kernel, (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6);
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
        uint sw_n = (uint)size;
        [enc setBytes:&sw_n length:sizeof(uint) atIndex:2];

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
        (uint32_t)rows
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

// G0: first-op x staging — CPU writes to a no-copy slab between GPU
// submissions are not guaranteed visible, so the BATCH'S FIRST op takes x
// from CPU through the private internal buffer (memcpy at call time, reads
// are correct at the op's execution); chained ops use GPU-written buffers.
void sofuu_metal_batch_gemv_staged(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    const float* x,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, int dtype
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x || !y_buf_ptr || rows == 0 || cols == 0) return;
    size_t x_bytes = cols * sizeof(float);
    ensure_buffers(ctx, x_bytes, rows * sizeof(float));
    std::memcpy(ctx->x_buf.contents, x, x_bytes);

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:0];
    [ctx->batch_enc setBuffer:ctx->x_buf offset:0 atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// P1: prefill GEMM op (q4_K first). x is [B x cols], y is [B x rows]
// (row-major), one commit rides the live batch like every other op.
// Returns 1 dispatched, 0 unavailable (PSO missing / bad args) — the Rust
// side gates the GPU prefill route on a probe, never on silent zeros.
int sofuu_metal_batch_gguf_gemm(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || B == 0) return 0;
    if (!ctx->pso_gemm_q4_k_live) return 0;
    if (ty != 12) return 0; // q4_K only for now; q8_0/q6_K follow the receipt
    id<MTLComputePipelineState> pso = ctx->pso_gemm_q4_k;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint subtiles = (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P1 op-diet: fused multi-section GEMM (qkv -> n_sec=3, gate_up -> n_sec=2).
// secs is n_sec (<=4, host-padded to 4) {off, rem, rows_end} entries where
// off is the 256-aligned w_buf byte offset and rows_end the prefix row sum.
// x is [B x cols], y is [B x rows_total]. Returns 1 dispatched.
int sofuu_metal_batch_gguf_gemm_multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows_total, size_t cols, size_t B,
    const uint32_t* secs /* [4][3] padded */, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || B == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gemm_q4_k_multi_live) return 0;
    if (ty != 12) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows_total;
    uint ns = (uint)n_sec;
    uint sblob[20]; // 4 sections x {off, rem, rows_end, y_off, y_stride}
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_multi];
    // sections carry u32 DELTAS from w_base_offset (the first section's
    // 256-aligned start) — absolute offsets overflow u32 on 5GB files
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2]; // sections carry absolute slab offsets
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:6];
    // NOTE n_sec is redundant with the padded blob but kept for clarity of
    // the trace; the kernel scans all 4 padded entries (empty spans never
    // match). Dispatch grid: (rows_total, subtiles).
    uint subtiles = (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P1 wave 2: resid-add epilogue GEMM (o/down absorb their residual add —
// one writer per element). y[t*rows + row] = dot(W[row], X[t]) +
// res[t*res_stride + row]; y and res usually alias the x slab (y bound at
// the x region offset, res at the same region start). Returns 1 dispatched.
int sofuu_metal_batch_gguf_gemm_acc(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* res_buf_ptr, size_t res_offset, size_t res_stride,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !res_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || B == 0) return 0;
    if (!ctx->pso_gemm_q4_k_acc_live) return 0;
    if (ty != 12) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rs = (uint)res_stride;
    uint rn = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_acc];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)res_buf_ptr offset:res_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:3];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:8];
    uint subtiles = (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P2 decode diet: multi-section q4_K b4 GEMV (see kernels.metal). All four
// token rows computed per threadgroup; sections write their own y regions.
// (external "C": this op sits past the file's extern-c region)
extern "C" int sofuu_metal_batch_gguf_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs /* [4][5] padded */, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q4_k_b4multi_live) return 0;
    if (ty != 12) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q4_k_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P1 completion: multi-section q6_K b4 GEMV (same table as the q4_K one).
extern "C" int sofuu_metal_batch_gguf6_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q6_k_b4multi_live) return 0;
    if (ty != 14) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q6_k_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P2: multi-section q8_0 b4 GEMV (1B fleet; MiniCPM's q8_0 tensors).
extern "C" int sofuu_metal_batch_gguf8_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q8_0_b4multi_live) return 0;
    if (ty != 8) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q8_0_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// G1: batched mlx4 GEMV — bindings mirror sofuu_metal_gemv_mlx4 exactly
// (0=w 1=s 2=b 3=x 4=y, 5=cols 6=group 7=rems[4]) but with caller MTLBuffers
// for x/y so batch ops chain activations GPU-side.
void sofuu_metal_batch_mlx4(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* s_buf_ptr, size_t s_offset,
    void* b_buf_ptr, size_t b_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !s_buf_ptr || !b_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

    size_t w_aligned = w_offset & ~255ULL;
    size_t scales_aligned = s_offset & ~255ULL;
    size_t biases_aligned = b_offset & ~255ULL;
    uint32_t rems[4] = {
        (uint32_t)(w_offset & 255ULL),
        (uint32_t)(s_offset & 255ULL),
        (uint32_t)(b_offset & 255ULL),
        (uint32_t)rows
    };
    uint c = (uint)cols;
    uint g = (uint)group_size;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)s_buf_ptr offset:scales_aligned atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)b_buf_ptr offset:biases_aligned atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:4];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&g length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:rems length:sizeof(rems) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── GPU1: in-batch glue ops (whole-decode-step batches) ────────────────────
// Same contract as batch_gemv: caller-owned MTLBuffers + byte offsets so ops
// chain slot-to-slot inside ONE encoder; a memory barrier before each op
// orders the previous dispatch's writes. The CPU touches the slab only
// before batch_open / after batch_end.

void sofuu_metal_batch_add(
    void* ctx_ptr,
    void* acc_buf_ptr, size_t acc_offset,
    void* add_buf_ptr, size_t add_offset,
    size_t n
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !acc_buf_ptr || !add_buf_ptr || n == 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint un = (uint)n;
    [ctx->batch_enc setComputePipelineState:ctx->pso_add_inplace];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)add_buf_ptr offset:add_offset atIndex:1];
    [ctx->batch_enc setBytes:&un length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sofuu_metal_batch_swiglu(
    void* ctx_ptr,
    void* g_buf_ptr, size_t g_offset,
    void* u_buf_ptr, size_t u_offset,
    size_t n
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !g_buf_ptr || !u_buf_ptr || n == 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    uint un = (uint)n;
    [ctx->batch_enc setComputePipelineState:ctx->pso_swiglu];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)g_buf_ptr offset:g_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)u_buf_ptr offset:u_offset atIndex:1];
    [ctx->batch_enc setBytes:&un length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Slot-based RMSNorm (w is bf16 in the model buffer) reusing the per-op
// `rmsnorm` kernel with caller offsets.
void sofuu_metal_batch_rmsnorm(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// LFM2 gated short conv: y = C * (s0*w0 + s1*w1 + (B*x)*w2), state shifts.
void sofuu_metal_batch_conv_gate(
    void* ctx_ptr,
    void* bcx_buf_ptr, size_t bcx_offset,
    void* conv_w_buf_ptr, size_t conv_w_offset,
    void* state_buf_ptr, size_t state_offset,
    void* out_buf_ptr, size_t out_offset,
    size_t dim
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !bcx_buf_ptr || !conv_w_buf_ptr || !state_buf_ptr || !out_buf_ptr || dim == 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = conv_w_offset & ~255ULL;
    uint w_rem = (uint)(conv_w_offset & 255ULL);
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_conv_gate];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)bcx_buf_ptr offset:bcx_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)conv_w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)state_buf_ptr offset:state_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:3];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((dim + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Head-RMSNorm + split-half RoPE on k (+plain copy of v), then append at
// position `pos` of this layer's KV-cache slice. kc/vc offsets delimit the
// LAYER slice; the kernel indexes kv*max_seq + pos inside it.
void sofuu_metal_batch_kv_write(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !kln_buf_ptr) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = kln_offset & ~255ULL;
    uint w_rem = (uint)(kln_offset & 255ULL);
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_kv_write];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kln_buf_ptr offset:aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:10];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:11];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GQA decode attention: head-norm + RoPE on q, softmax(q·K)V against the
// cache slice written by batch_kv_write earlier in this same batch.
void sofuu_metal_batch_attn(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !qln_buf_ptr || !out_buf_ptr) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = qln_offset & ~255ULL;
    uint w_rem = (uint)(qln_offset & 255ULL);
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_attn];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_buf_ptr offset:aligned_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3: per-head f32 RMSNorm (q-norm) on a [heads x head_dim] slot.
void sofuu_metal_batch_head_rmsnorm(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    uint32_t head_dim, uint32_t n_heads,
    uint32_t has_norm, uint32_t pos, float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_buf_ptr ? (w_offset & ~255ULL) : 0;
    uint w_rem = w_buf_ptr ? (uint)(w_offset & 255ULL) / 4 : 0;
    uint hd = head_dim;
    uint nh = n_heads;
    uint hn = has_norm;
    uint rm = rope_mode;
    [ctx->batch_enc setComputePipelineState:ctx->pso_head_rmsnorm];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(w_buf_ptr ? w_buf_ptr : x_buf_ptr) offset:aligned atIndex:1];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:5];
    [ctx->batch_enc setBytes:&hn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:8];
    [ctx->batch_enc setBytes:&rm length:sizeof(uint) atIndex:9];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3 attention, stage 1: chunked softmax+V partials to the meta buffer.
void sofuu_metal_batch_llama_attn_part(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* meta_buf_ptr, size_t meta_offset,
    uint32_t has_norm_q_unused,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) { (void)has_norm_q_unused; return; }
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = qln_buf_ptr ? (qln_offset & ~255ULL) : 0;
    uint w_rem = qln_buf_ptr ? (uint)(qln_offset & 255ULL) / 4 : 0;
    uint hn = has_norm_q_unused;   // kept for ABI symmetry; norm lives in head_rmsnorm now
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_part];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(qln_buf_ptr ? qln_buf_ptr : q_buf_ptr)
                      offset: aligned atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&hn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    uint n_chunks = (pos + 1 + 127) / 128;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads * n_chunks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3 attention, stage 2: rescale chunk partials into head_out.
void sofuu_metal_batch_llama_attn_fin(
    void* ctx_ptr,
    void* meta_buf_ptr, size_t meta_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t pos,
    uint32_t n_heads, uint32_t head_dim
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_fin];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── MULTI-ROW (B≤4 chain) attention ops: positions come from a u32 buffer ──
// grid = rows × heads(× chunks), row = threadgroup / heads; contiguous
// [B][...] row slots; meta regions stride meta_stride per row.

// FUSED 1-chunk attention (pos<128): part+fin in one dispatch, no meta.
// Returns 1 when dispatched, 0 when the PSO is unavailable (caller falls
// back to the part+fin pair).
int sofuu_metal_batch_llama_attn_1c_b4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr) return 0;
    if (ctx->pso_llama_attn_1c_b4 == nil) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_1c_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:8];
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// B4 q4_K GEMV with the resid-add epilogue (o/down projections): returns 1
// when dispatched, 0 when unavailable (caller falls back to gemv + add).
int sofuu_metal_batch_gguf_b4r(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    void* acc_buf_ptr, size_t acc_offset,
    size_t rows, size_t cols, size_t acc_dim
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || !acc_buf_ptr) return 0;
    if (ctx->pso_gemv_q4_k_b4c2r == nil) return 0;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint r = (uint)rows;
    int8_t acc_dim_i = (int8_t)acc_dim; (void)acc_dim_i;
    uint acc_dim_u = (uint)acc_dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_b4c2r];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:6];
    [ctx->batch_enc setBytes:&acc_dim_u length:sizeof(uint) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

void sofuu_metal_batch_llama_kv_write_b4(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t has_norm,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr) return;
    if (ctx->pso_llama_kv_write_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = kln_buf_ptr ? (kln_offset & ~255ULL) : 0;
    uint w_rem = kln_buf_ptr ? (uint)(kln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_kv_write_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(kln_buf_ptr ? kln_buf_ptr : k_buf_ptr)
                      offset: aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:13];
    uint grid = rows * n_kv_heads;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_head_rmsnorm_b4(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads,
    uint32_t has_norm,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr) return;
    if (ctx->pso_head_rmsnorm_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = w_buf_ptr ? (w_offset & ~255ULL) : 0;
    uint w_rem = w_buf_ptr ? (uint)(w_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_head_rmsnorm_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(w_buf_ptr ? w_buf_ptr : x_buf_ptr)
                      offset: aligned_offset atIndex:1];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:7];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:8];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:9];
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_llama_attn_part_b4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* meta_buf_ptr, size_t meta_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    uint32_t n_chunks_max, uint32_t meta_stride
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !meta_buf_ptr) return;
    if (ctx->pso_llama_attn_part_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_part_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:5];
    // buffer 3 (q_ln) unused in the b4 flow: bind q as a placeholder
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:3];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:11];
    uint grid = rows * n_heads * n_chunks_max;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_llama_attn_fin_b4(
    void* ctx_ptr,
    void* meta_buf_ptr, size_t meta_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t n_heads, uint32_t head_dim,
    uint32_t n_chunks_max, uint32_t meta_stride
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !meta_buf_ptr || !out_buf_ptr) return;
    if (ctx->pso_llama_attn_fin_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_fin_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:2];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:6];
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU2.1: RMSNorm with F32 weights (GGUF norm tensors). Weight bound at a
// 256-byte-aligned offset with the element remainder passed as a constant.
void sofuu_metal_batch_rmsnorm_f32w(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL) / 4;   // element remainder (f32)
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm_f32w];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// B-row RMSNorm (spec chain): rows contiguous [B][dim], grid = B threadgroups
void sofuu_metal_batch_rmsnorm_f32w_b4(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps, size_t rows
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0 || rows == 0) return;
    if (ctx->pso_rmsnorm_f32w_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL) / 4;
    uint d = (uint)dim;
    uint r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm_f32w_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// Per-kv-head: optional f32 head-RMSNorm, split-half (mode 0) or interleaved
// (mode 1) RoPE on k, append k(roped)/v(raw) at `pos` of this layer's slice.
void sofuu_metal_batch_llama_kv_write(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t has_norm,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = kln_buf_ptr ? (kln_offset & ~255ULL) : 0;
    uint w_rem = kln_buf_ptr ? (uint)(kln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_kv_write];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(kln_buf_ptr ? kln_buf_ptr : k_buf_ptr)
                      offset: aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:13];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// Per-q-head GQA decode attention with optional f32 q-norm + RoPE, softmax
// over the full cache prefix, weighted V into head_out.
void sofuu_metal_batch_llama_attn(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t has_norm,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned_offset = qln_buf_ptr ? (qln_offset & ~255ULL) : 0;
    uint w_rem = qln_buf_ptr ? (uint)(qln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(qln_buf_ptr ? qln_buf_ptr : q_buf_ptr)
                      offset: aligned_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3: batched (B=4) q4_K GEMV — token-major x[4][cols] -> y[4][rows].
void sofuu_metal_batch_gguf_b4(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    if (ctx->pso_gemv_q4_k_b4 == nil) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// G2: GGUF dequant-dot GEMV, one op in the live batch. ty is the GGML type
// id (2 = q4_0, 8 = q8_0, 12 = q4_K, 14 = q6_K). Bindings: 0=w 1=x 2=y
// 3=cols 4=w_rem.
void sofuu_metal_batch_gguf(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 8:  pso = ctx->pso_gemv_q8_0; break;
        case 2:  pso = ctx->pso_gemv_q4_0; break;
        case 12: pso = ctx->pso_gemv_q4_k; break;
        case 14: pso = ctx->pso_gemv_q6_k; break;
        default: return;
    }
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── G0: real multi-op batching ─────────────────────────────────────────────
// batch_open retains ONE uncommitted command buffer with a LIVE encoder;
// every batch_gemv appends an op reading caller-owned MTLBuffers (so ops can
// chain y→x with zero CPU staging); batch_end ends the encoder, commits once,
// waits. One GPU submission per batch, independent of op count.
// GPU-frame capture state (env-gated, shared between batch_open/batch_end;
// single-threaded engine — no sync needed)
static NSString *g_cap_path = nil;
static int g_cap_skip = -1, g_cap_n = 1, g_cap_cycles = 0;
static bool g_cap_active = false;

int sofuu_metal_batch_open(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    sofuu_metal_sync(ctx_ptr);
    @autoreleasepool {
        // Env-gated GPU-frame capture for the interleaved-vs-standalone GEMV
        // attribute pass: SOFUU_METAL_CAPTURE=<out.gputrace>,
        // SOFUU_METAL_CAPTURE_SKIP=<cycles> and SOFUU_METAL_CAPTURE_N=<n>
        // (default 1) — skip the process's warmup cycles, then record n
        // batch_open..batch_end command buffers of the steady-state pass.
        if (g_cap_skip < 0) {
            const char* p = getenv("SOFUU_METAL_CAPTURE");
            const char* s = getenv("SOFUU_METAL_CAPTURE_SKIP");
            const char* n = getenv("SOFUU_METAL_CAPTURE_N");
            g_cap_n = n ? atoi(n) : 1;
            if (g_cap_n < 1) g_cap_n = 1;
            g_cap_skip = s ? atoi(s) : 0;
            g_cap_path = p ? [NSString stringWithUTF8String:p] : nil;
        }
        if (g_cap_path != nil && !g_cap_active && g_cap_cycles >= g_cap_skip) {
            MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
            if ([mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
                MTLCaptureDescriptor *d = [MTLCaptureDescriptor new];
                d.captureObject = ctx->device;
                d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:g_cap_path];
                NSError *err = nil;
                if ([mgr startCaptureWithDescriptor:d error:&err]) {
                    g_cap_active = true;
                    fprintf(stderr, "[metal] gpu capture STARTED (cycle %d, %s)\n",
                            g_cap_cycles, g_cap_path.UTF8String);
                } else {
                    fprintf(stderr, "[metal] gpu capture FAILED: %s\n",
                            err.localizedDescription.UTF8String ?: "(nil)");
                    g_cap_path = nil;
                }
            } else {
                fprintf(stderr, "[metal] gpu capture unsupported (no GPUTraceDocument) — needs Xcode runtime support\n");
                g_cap_path = nil;
            }
        }
        g_cap_cycles++;

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
        // paired with the startCapture in batch_open: a captured cycle ends
        // exactly at its batch_end so the trace holds whole passes
        if (g_cap_active) {
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            g_cap_active = false;
            fprintf(stderr, "[metal] gpu capture STOPPED (cycle %d)\n",
                    g_cap_cycles - 1);
        }
    }
}

#else

// Stubs for non-macOS platforms (Linux, Android, Windows)
bool sofuu_metal_available(void) { return false; }
int sofuu_metal_batch_open(void* ctx) { (void)ctx; return -1; }
void sofuu_metal_batch_gemv(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, int d) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)d; }
void sofuu_metal_batch_end(void* ctx) { (void)ctx; }
void sofuu_metal_batch_mlx4(void* ctx, void* w, size_t wo, void* s, size_t so, void* b, size_t bo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t g, int bf) { (void)ctx;(void)w;(void)wo;(void)s;(void)so;(void)b;(void)bo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)g;(void)bf; }
void sofuu_metal_batch_gemv_staged(void* ctx, void* w, size_t wo, const float* x, void* y, size_t yo, size_t r, size_t c, int d) { (void)ctx;(void)w;(void)wo;(void)x;(void)y;(void)yo;(void)r;(void)c;(void)d; }
void sofuu_metal_batch_add(void* ctx, void* acc, size_t ao, void* add, size_t ado, size_t n) { (void)ctx;(void)acc;(void)ao;(void)add;(void)ado;(void)n; }
void sofuu_metal_batch_swiglu(void* ctx, void* g, size_t go, void* u, size_t uo, size_t n) { (void)ctx;(void)g;(void)go;(void)u;(void)uo;(void)n; }
void sofuu_metal_batch_rmsnorm(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps; }
void sofuu_metal_batch_conv_gate(void* ctx, void* bcx, size_t bco, void* cw, size_t cwo, void* st, size_t sto, void* out, size_t oo, size_t dim) { (void)ctx;(void)bcx;(void)bco;(void)cw;(void)cwo;(void)st;(void)sto;(void)out;(void)oo;(void)dim; }
void sofuu_metal_batch_kv_write(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t p, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)p;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb; }
void sofuu_metal_batch_attn(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* out, size_t oo, uint32_t p, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)out;(void)oo;(void)p;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb; }
void sofuu_metal_batch_rmsnorm_f32w(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps; }
void sofuu_metal_batch_rmsnorm_f32w_b4(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps, size_t rows) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps;(void)rows; }
void sofuu_metal_batch_llama_kv_write(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t hn, uint32_t p, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)hn;(void)p;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_llama_attn(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* out, size_t oo, uint32_t hn, uint32_t p, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)out;(void)oo;(void)hn;(void)p;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_head_rmsnorm(void* ctx, void* x, size_t xo, void* w, size_t wo, uint32_t hd, uint32_t nh, uint32_t hn, uint32_t p, float e, float rb, uint32_t rm) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)hd;(void)nh;(void)hn;(void)p;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_llama_attn_part(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* meta, size_t mo, uint32_t hn, uint32_t p, uint32_t hd2, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)meta;(void)mo;(void)hn;(void)p;(void)hd2;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_llama_attn_fin(void* ctx, void* meta, size_t mo, void* out, size_t oo, uint32_t p, uint32_t nh, uint32_t hd) { (void)ctx;(void)meta;(void)mo;(void)out;(void)oo;(void)p;(void)nh;(void)hd; }
void sofuu_metal_batch_llama_kv_write_b4(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t hn, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)hn;(void)pb;(void)po;(void)rows;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_head_rmsnorm_b4(void* ctx, void* x, size_t xo, void* w, size_t wo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t hn, float e, float rb, uint32_t rm) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)hn;(void)e;(void)rb;(void)rm; }
void sofuu_metal_batch_llama_attn_part_b4(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* meta, size_t mo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t ncm, uint32_t mstr) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)meta;(void)mo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)ncm;(void)mstr; }
void sofuu_metal_batch_llama_attn_fin_b4(void* ctx, void* meta, size_t mo, void* out, size_t oo, void* pb, size_t po, uint32_t rows, uint32_t nh, uint32_t hd, uint32_t ncm, uint32_t mstr) { (void)ctx;(void)meta;(void)mo;(void)out;(void)oo;(void)pb;(void)po;(void)rows;(void)nh;(void)hd;(void)ncm;(void)mstr; }
int sofuu_metal_batch_llama_attn_1c_b4(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* out, size_t oo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)out;(void)oo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms; return 0; }
int sofuu_metal_batch_gguf_b4r(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, void* acc, size_t ao, size_t r, size_t c, size_t ad) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)acc;(void)ao;(void)r;(void)c;(void)ad; return 0; }
void sofuu_metal_batch_gguf(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c; }
void sofuu_metal_batch_gguf_b4(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c; }
int sofuu_metal_batch_gguf_gemm(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_gemm_multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t yo, size_t rt, size_t c, size_t B, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)yo;(void)rt;(void)c;(void)B;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf_gemm_acc(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* res, size_t ro, size_t rs, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)res;(void)ro;(void)rs;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf6_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf8_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
void* sofuu_metal_init(void) { return NULL; }
void sofuu_metal_free(void* ctx) {}
void* sofuu_metal_buffer_create_zerocopy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void* sofuu_metal_buffer_create(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void* sofuu_metal_buffer_create_copy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
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

