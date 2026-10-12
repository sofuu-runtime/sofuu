#ifndef SOFUU_INFER_KERNELS_H
#define SOFUU_INFER_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sofuu Inference Engine C++ Microkernels
 *
 * Hardware-tuned SIMD vector routines (ARM NEON / x86 AVX2 / FMA)
 * for maximum CPU throughput in quantized LLM matrix-vector operations.
 */

/* Multi-threaded CPU parallelization control */
void sofuu_set_threads(int n);
int sofuu_get_threads(void);

/* Dequantization routines for K-Quants and basic quants */
void sofuu_dequantize_row_q4_0(const void *src, float *dst, size_t k);
void sofuu_dequantize_row_q8_0(const void *src, float *dst, size_t k);
void sofuu_dequantize_row_q4_k(const void *src, float *dst, size_t k);
void sofuu_dequantize_row_q6_k(const void *src, float *dst, size_t k);

/* Dot product of a single Q4_0 / Q8_0 / Q4_K / Q6_K row with float32 activation vector x. */
float sofuu_vec_dot_q4_0(const void *weight_bytes, const float *x, size_t n_blocks);
float sofuu_vec_dot_q8_0(const void *weight_bytes, const float *x, size_t n_blocks);
float sofuu_vec_dot_q4_k(const void *weight_bytes, const float *x, size_t n_blocks);
float sofuu_vec_dot_q6_k(const void *weight_bytes, const float *x, size_t n_blocks);

/* Matrix-vector multiplication: out = W * x
 * Where W has shape [rows, cols] (in row-major layout).
 * Fully multi-core parallelized across available CPU cores. */
void sofuu_gemv_q4_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_q8_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_q4_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_q6_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_f32(const float *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_f16(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_bf16(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols);
void sofuu_gemv_mlx4(const void *w_bytes, const void *scales_bytes, const void *biases_bytes, const float *x, float *out, size_t rows, size_t cols, size_t group_size, int is_bf16);

// Batched MLX4 GEMV: runs num_mats matrices against one activation vector.
void sofuu_gemv_mlx4_multi(
    const void *const *w_bytes,
    const void *const *scales_bytes,
    const void *const *biases_bytes,
    const float *x,
    float *const *outs,
    const size_t *rows,
    size_t num_mats,
    size_t cols,
    size_t group_size,
    int is_bf16
);

/* Matrix-matrix multiplication (batched prefill): out = X * W^T
 * Where W has shape [rows, cols] (in row-major layout),
 * X has shape [batch_size, cols] (row-major),
 * and out has shape [batch_size, rows] (row-major).
 * Fully multi-core parallelized across CPU cores. */
void sofuu_gemm_q4_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);
void sofuu_gemm_q8_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);
void sofuu_gemm_q4_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);
void sofuu_gemm_q6_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);
void sofuu_gemm_f32(const float *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);
void sofuu_gemm_f16(const uint16_t *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size);

/* Root Mean Square Layer Normalization (RMSNorm):
 * out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i] */
void sofuu_rmsnorm(float *out, const float *x, const float *weight, size_t n, float eps);

/* Rotary Position Embedding (RoPE) applied to query or key head vectors.
 * n_heads: number of attention heads
 * head_dim: dimension of each head (e.g. 64, 128)
 * pos: sequence token index
 * freq_base: base frequency (typically 10000.0 or 500000.0 for Llama 3 / Qwen) */
void sofuu_rope(float *vec, size_t n_heads, size_t head_dim, size_t pos, float freq_base);

/* Fused SwiGLU feed-forward activation:
 * out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i] */
void sofuu_swiglu(float *out, const float *gate, const float *up, size_t n);

/* In-place numerically stable Softmax:
 * scores[i] = exp(scores[i] - max) / sum(exp(scores - max)) */
void sofuu_softmax(float *scores, size_t n);

/* Hardware SIMD vector dot product (NEON / AVX2): returns sum(a[i] * b[i]) */
float sofuu_vec_dot_f32(const float *a, const float *b, size_t n);

/* 4-way SIMD vector dot product: calculates dot(w, x0..3) in a single pass */
void sofuu_vec_dot_f32_x4(const float *w, const float *x0, const float *x1, const float *x2, const float *x3, size_t n,
                          float *out0, float *out1, float *out2, float *out3);

/* 8-way SIMD vector dot product: calculates dot(w, x0..7) in a single pass */
void sofuu_vec_dot_f32_x8(const float *w,
                          const float *x0, const float *x1, const float *x2, const float *x3,
                          const float *x4, const float *x5, const float *x6, const float *x7,
                          size_t n,
                          float *out0, float *out1, float *out2, float *out3,
                          float *out4, float *out5, float *out6, float *out7);

/* Hardware SIMD multiply-accumulate (NEON / AVX2): dst[i] += src[i] * scalar */
void sofuu_vec_mad_f32(float *dst, const float *src, float scalar, size_t n);

/* Shift RoPE position of key vectors by delta_pos (can be negative for left shift / compaction).
 * Rotates each 2D sub-vector by (delta_pos * freq).
 * Exactly equivalent to embedding the token at (original_pos + delta_pos). */
void sofuu_rope_shift(float *vec, size_t n_heads, size_t head_dim, int64_t delta_pos, float freq_base);

/* Compact the KV cache across all layers in place:
 * Retains sink_tokens at [0 .. sink_tokens), drops drop_tokens from the middle,
 * shifts surviving recent tokens [sink_tokens + drop_tokens .. current_seq_len) left to start at sink_tokens,
 * and shifts RoPE positions of the moved K vectors by -(int64_t)drop_tokens. */
void sofuu_kv_compact(
    float *k_cache,
    float *v_cache,
    size_t n_layers,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t current_seq_len,
    size_t sink_tokens,
    size_t drop_tokens,
    float freq_base
);

/* Multi-core parallel causal self-attention for batched prefill and autoregressive decode:
 * Computes Q * K^T, softmax scaling, and V weighted sum across all heads in parallel. */
void sofuu_batched_attention(
    const float *q_batch,
    const float *k_cache,
    const float *v_cache,
    float *head_out_batch,
    size_t b_size,
    size_t start_pos,
    size_t layer_idx,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len
);

/* Multi-core parallel causal self-attention with optional Sliding Window Attention (SWA) and Attention Sinks.
 * If sliding_window == 0, evaluates standard full causal attention up to seq_len.
 * If sliding_window > 0, restricts attention to the first sink_tokens and the most recent sliding_window tokens. */
void sofuu_batched_attention_ext(
    const float *q_batch,
    const float *k_cache,
    const float *v_cache,
    float *head_out_batch,
    size_t b_size,
    size_t start_pos,
    size_t layer_idx,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t sliding_window,
    size_t sink_tokens
);

#ifdef __cplusplus
}
#endif

#endif /* SOFUU_INFER_KERNELS_H */
