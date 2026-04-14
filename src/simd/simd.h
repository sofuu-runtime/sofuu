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

/* Cosine similarity: dot(a,b) / (||a|| * ||b||), range [-1, 1] */
float sofuu_cosine_f32(const float *a, const float *b, size_t n);
