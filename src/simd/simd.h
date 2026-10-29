#pragma once
#include <stddef.h>

/*
 * SIMD vector math for Sofuu — auto-dispatches to NEON (arm64) or AVX2 (x86_64).
 *
 * All functions operate on float32 arrays of length `n`.
 * No alignment requirements — unaligned loads are handled internally.
 */

/* Dot product: sum of element-wise products */
float sofuu_dot_f32(const float *a, const float *b, size_t n);

/* L2 (Euclidean) distance */
float sofuu_l2_f32 (const float *a, const float *b, size_t n);

/* Cosine similarity: dot(a,b) / (||a|| * ||b||), range [-1, 1]
 *
 * R-27: the fast path accumulates in float32 and is unchanged for ordinary
 * magnitudes. When an f32 product or running sum overflows (|a_i| > 1.8e19,
 * or n*x^2 > FLT_MAX e.g. 768 lanes of 1e18) the accumulators become
 * non-finite — an exact overflow signal — and the kernel transparently
 * recomputes scaled in f64, so the result stays finite (previously NaN).
 * A norm that underflowed to exactly 0 is repaired the same way. Zero
 * vectors still return 0.0f and NaN/Inf operands still return NaN.
 * No extra per-call or per-lane cost: the checks are four compares after
 * the loop, never inside it. */
float sofuu_cosine_f32(const float *a, const float *b, size_t n);
