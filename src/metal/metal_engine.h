// src/metal/metal_engine.h — C FFI bridge for Apple Silicon Metal GPU engine.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// 8x4-tile MMQ mirror (SOFUU_GPU_TILE_MMQ=1): act-quantize [B x cols] f32 rows
// into per-256 TileQ8K blocks (d + qs[256]), stride 260B, then the q4_K gemm
// that mirrors ggml_gemm_q4_K_8x4_q8_K_p's integer/f32 order bit-for-bit.
int sofuu_metal_batch_quantize_q8_k(void* ctx, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t cols, size_t B);
int sofuu_metal_batch_gemm_q4_k_mmq(void* ctx, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gemm_q6_k_mmq(void* ctx, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
// Per-32 a76 act-quant decode mirrors (SOFUU_DECODE_MMQ=1). The quantize
// op covers nb_total = B*cols/32 blocks of the f32 rows in ONE dispatch;
// the gemv mirrors consume the Q8Ext blocks (stride 40B: d, sum_x, qs[32]).
int sofuu_metal_batch_quantize_q8_k_pos(void* ctx, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t cols, size_t B);
int sofuu_metal_batch_quantize_q8_ext(void* ctx, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t nb_total);
int sofuu_metal_batch_gemv_q4_k_mmq32(void* ctx, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gemv_q6_k_mmq32(void* ctx, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);

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

// Execute Batched Q, K, V Matrix-Vector multiplications on Apple Silicon GPU
// in a single command buffer to eliminate per-projection dispatch latency.
void sofuu_metal_gemv_qkv_bf16(
    void* ctx,
    void* w_buf,
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
);

// Execute Batched Gate and Up Matrix-Vector multiplications for SwiGLU/GeGLU
// in a single command buffer.
void sofuu_metal_gemv_gate_up_bf16(
    void* ctx,
    void* w_buf,
    size_t gate_offset,
    size_t up_offset,
    const float* x,
    float* gate_out,
    float* up_out,
    size_t inter_rows,
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
    size_t group_size,
    int is_bf16
);

// Execute Batched Q, K, V MLX 4-bit affine GEMV in one command buffer.
void sofuu_metal_gemv_qkv_mlx4(
    void* ctx,
    void* q_buf, size_t q_offset, size_t q_rows,
    void* q_scales_buf, size_t q_scales_offset,
    void* q_biases_buf, size_t q_biases_offset,
    void* k_buf, size_t k_offset, size_t k_rows,
    void* k_scales_buf, size_t k_scales_offset,
    void* k_biases_buf, size_t k_biases_offset,
    void* v_buf, size_t v_offset, size_t v_rows,
    void* v_scales_buf, size_t v_scales_offset,
    void* v_biases_buf, size_t v_biases_offset,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols, size_t group_size, int is_bf16
);

// Execute Batched Gate and Up MLX 4-bit affine GEMV in one command buffer.
void sofuu_metal_gemv_gate_up_mlx4(
    void* ctx,
    void* gate_buf, size_t gate_offset, size_t gate_rows,
    void* gate_scales_buf, size_t gate_scales_offset,
    void* gate_biases_buf, size_t gate_biases_offset,
    void* up_buf, size_t up_offset, size_t up_rows,
    void* up_scales_buf, size_t up_scales_offset,
    void* up_biases_buf, size_t up_biases_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t cols, size_t group_size, int is_bf16
);

// Create a Metal buffer from pointer (zero-copy if page-aligned, copy fallback if not).
void* sofuu_metal_buffer_create(void* ctx, const void* ptr, size_t size_bytes);
// Always a driver-managed COPY of the bytes (never wraps caller memory).
void* sofuu_metal_buffer_create_copy(void* ctx, const void* ptr, size_t size_bytes);

// Execute Batched Q, K, V Matrix-Vector multiplications on Apple Silicon GPU
// supporting separate buffers and multiple data types (0 = BF16, 1 = F16, 2 = F32).
void sofuu_metal_gemv_qkv_multi(
    void* ctx,
    void* q_buf, size_t q_offset, size_t q_rows,
    void* k_buf, size_t k_offset, size_t k_rows,
    void* v_buf, size_t v_offset, size_t v_rows,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols,
    int dtype
);

// Execute Batched Gate and Up Matrix-Vector multiplications for SwiGLU/GeGLU
// supporting separate buffers and multiple data types (0 = BF16, 1 = F16, 2 = F32).
void sofuu_metal_gemv_gate_up_multi(
    void* ctx,
    void* gate_buf, size_t gate_offset,
    void* up_buf, size_t up_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t inter_rows,
    size_t cols,
    int dtype
);

// Asynchronously dispatch GEMV on GPU without blocking calling CPU thread.
void sofuu_metal_gemv_async(
    void* ctx,
    void* w_buf,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    int dtype
);

// Wait for any pending asynchronous Metal GPU dispatch to complete.
void sofuu_metal_sync(void* ctx);
int sofuu_metal_batch_open(void* ctx);
void sofuu_metal_batch_gemv(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, int dtype);
int sofuu_metal_batch_end(void* ctx);
void sofuu_metal_batch_gemv_staged(void* ctx, void* w_buf, size_t w_offset, const float* x, void* y_buf, size_t y_offset, size_t rows, size_t cols, int dtype);
void sofuu_metal_batch_mlx4(void* ctx, void* w_buf, size_t w_offset, void* s_buf, size_t s_offset, void* b_buf, size_t b_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t group_size, int is_bf16);

// GPU1: in-batch glue ops — whole-decode-step batches (lfm2 client first).
// All offsets are byte offsets into caller-owned MTLBuffers; ops chain
// slot-to-slot inside one encoder, one commit per batch (see infer/metal.rs).
int sofuu_metal_batch_add(void* ctx, void* acc_buf, size_t acc_offset, void* add_buf, size_t add_offset, size_t n);
int sofuu_metal_batch_swiglu(void* ctx, void* g_buf, size_t g_offset, void* u_buf, size_t u_offset, size_t n);
void sofuu_metal_batch_rmsnorm(void* ctx, void* x_buf, size_t x_offset, void* w_buf, size_t w_offset, void* y_buf, size_t y_offset, size_t dim, float eps);
void sofuu_metal_batch_conv_gate(void* ctx, void* bcx_buf, size_t bcx_offset, void* conv_w_buf, size_t conv_w_offset, void* state_buf, size_t state_offset, void* out_buf, size_t out_offset, size_t dim);
void sofuu_metal_batch_kv_write(void* ctx, void* k_buf, size_t k_offset, void* v_buf, size_t v_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* kln_buf, size_t kln_offset, uint32_t pos, uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base);
void sofuu_metal_batch_attn(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* qln_buf, size_t qln_offset, void* out_buf, size_t out_offset, uint32_t pos, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base);
int sofuu_metal_batch_rmsnorm_f32w(void* ctx, void* x_buf, size_t x_offset, void* w_buf, size_t w_offset, void* y_buf, size_t y_offset, size_t dim, float eps);
void sofuu_metal_batch_rmsnorm_f32w_b4(void* ctx, void* x_buf, size_t x_offset, void* w_buf, size_t w_offset, void* y_buf, size_t y_offset, size_t dim, float eps, size_t rows);
int sofuu_metal_batch_llama_kv_write(void* ctx, void* k_buf, size_t k_offset, void* v_buf, size_t v_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* kln_buf, size_t kln_offset, uint32_t has_norm, uint32_t pos, uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base, uint32_t rope_mode);
void sofuu_metal_batch_llama_attn(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* qln_buf, size_t qln_offset, void* out_buf, size_t out_offset, uint32_t has_norm, uint32_t pos, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base, uint32_t rope_mode);
int sofuu_metal_batch_head_rmsnorm(void* ctx, void* x_buf, size_t x_offset, void* w_buf, size_t w_offset, uint32_t head_dim, uint32_t n_heads, uint32_t has_norm, uint32_t pos, float eps, float rope_base, uint32_t rope_mode);
int sofuu_metal_batch_llama_attn_part(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* qln_buf, size_t qln_offset, void* meta_buf, size_t meta_offset, uint32_t has_norm, uint32_t pos, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base, uint32_t rope_mode);
int sofuu_metal_batch_llama_attn_fin(void* ctx, void* meta_buf, size_t meta_offset, void* out_buf, size_t out_offset, uint32_t pos, uint32_t n_heads, uint32_t head_dim);
void sofuu_metal_batch_llama_kv_write_b4(void* ctx, void* k_buf, size_t k_offset, void* v_buf, size_t v_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* kln_buf, size_t kln_offset, uint32_t has_norm, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq, float eps, float rope_base, uint32_t rope_mode, void* rope_tab, size_t rope_tab_offset, uint32_t has_table);
void sofuu_metal_batch_head_rmsnorm_b4(void* ctx, void* x_buf, size_t x_offset, void* w_buf, size_t w_offset, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t has_norm, float eps, float rope_base, uint32_t rope_mode, void* rope_tab, size_t rope_tab_offset, uint32_t has_table);
void sofuu_metal_batch_llama_attn_part_b4(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* meta_buf, size_t meta_offset, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, uint32_t n_chunks_max, uint32_t meta_stride);
void sofuu_metal_batch_llama_attn_fin_b4(void* ctx, void* meta_buf, size_t meta_offset, void* out_buf, size_t out_offset, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t n_heads, uint32_t head_dim, uint32_t n_chunks_max, uint32_t meta_stride);
int sofuu_metal_batch_llama_attn_1c_b4(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_layer_offset, void* vc_buf, size_t vc_layer_offset, void* out_buf, size_t out_offset, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, void* rope_tab, size_t rope_tab_offset, uint32_t has_table);
int sofuu_metal_batch_gguf_b4r(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, void* acc_buf, size_t acc_offset, size_t rows, size_t cols, size_t acc_dim);
// G2: GGUF dequant-dot GEMV in the live batch. ty: 2=q4_0 8=q8_0 12=q4_K 14=q6_K.
int sofuu_metal_batch_gguf(void* ctx, uint32_t ty, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gguf_mma8(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t rows, size_t cols, size_t B);
void sofuu_metal_batch_gguf_b4(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gguf_gemm(void* ctx, uint32_t ty, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_peak_fma16(void* ctx, void* out_buf, size_t n_blocks);
int sofuu_metal_peak_fma(void* ctx, void* out_buf, size_t n_blocks);
int sofuu_metal_peak_mma(void* ctx, void* out_buf, size_t n_blocks);
int sofuu_metal_batch_gguf_gemm_mma64(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemm_tiled_r88(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemm_tiled_wide_multi(void* ctx, uint32_t ty, void* w_buf, size_t w_base_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows_total, size_t cols, size_t B, const uint32_t* secs, size_t n_sec);
int sofuu_metal_batch_gguf_gemm_tiled_wide(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemm_mma(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_llama_attn_flash_b4(void* ctx, void* q_buf, size_t q_offset, void* kc_buf, size_t kc_offset, void* vc_buf, size_t vc_offset, void* out_buf, size_t out_offset, void* pos_buf, size_t pos_offset, uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq, uint32_t has_table);
int sofuu_metal_batch_gguf_gemm_tiled(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemv_q4k_v4(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gguf_gemv_q4k_v3(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gguf_gemv_q4k_v2(void* ctx, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows, size_t cols);
int sofuu_metal_batch_gguf_gemm_mma64_acc(void* ctx, uint32_t ty, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* res_buf, size_t res_offset, size_t res_stride, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemm_mma64_multi(void* ctx, uint32_t ty, void* w_buf, size_t w_base_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows_total, size_t cols, size_t B, const uint32_t* secs, size_t n_sec);
int sofuu_metal_batch_gguf_gemm_tiled_multi(void* ctx, uint32_t ty, void* w_buf, size_t w_base_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows_total, size_t cols, size_t B, const uint32_t* secs, size_t n_sec);
int sofuu_metal_batch_gguf_gemm_tiled_acc(void* ctx, uint32_t ty, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* res_buf, size_t res_offset, size_t res_stride, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);
int sofuu_metal_batch_gguf_gemm_multi(void* ctx, uint32_t ty, void* w_buf, size_t w_base_offset, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t rows_total, size_t cols, size_t B, const uint32_t* secs, size_t n_sec);
int sofuu_metal_batch_gguf_gemm_acc(void* ctx, uint32_t ty, void* w_buf, size_t w_offset, void* x_buf, size_t x_offset, void* res_buf, size_t res_offset, size_t res_stride, void* y_buf, size_t y_offset, size_t rows, size_t cols, size_t B);

#ifdef __cplusplus
}
#endif

