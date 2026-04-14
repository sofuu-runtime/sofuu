/*
 * src/simd/neon.c — ARM NEON float32 vector ops (arm64)
 *
 * Compiled only on arm64 targets via Makefile arch detection.
 * Uses 128-bit NEON registers (4 x float32 per lane).
 */
#include "simd.h"
#include <math.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>

float sofuu_dot_f32(const float *a, const float *b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    /* horizontal sum of accumulator */
    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    /* tail elements */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

float sofuu_l2_f32(const float *a, const float *b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va   = vld1q_f32(a + i);
        float32x4_t vb   = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, diff, diff);
    }
    float32x2_t lo   = vget_low_f32(acc);
    float32x2_t hi   = vget_high_f32(acc);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    for (; i < n; i++) { float d = a[i] - b[i]; result += d * d; }
    return sqrtf(result);
}

float sofuu_cosine_f32(const float *a, const float *b, size_t n) {
    float dot  = sofuu_dot_f32(a, b, n);
    float norma = sofuu_dot_f32(a, a, n);
    float normb = sofuu_dot_f32(b, b, n);
    float denom = sqrtf(norma) * sqrtf(normb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

#else
/* ── Scalar fallback when NEON not available ── */

float sofuu_dot_f32(const float *a, const float *b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

float sofuu_l2_f32(const float *a, const float *b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) { float d = a[i] - b[i]; s += d * d; }
    return sqrtf(s);
}

float sofuu_cosine_f32(const float *a, const float *b, size_t n) {
    float dot = sofuu_dot_f32(a, b, n);
    float na  = sofuu_dot_f32(a, a, n);
    float nb  = sofuu_dot_f32(b, b, n);
    float den = sqrtf(na) * sqrtf(nb);
    if (den < 1e-10f) return 0.0f;
    return dot / den;
}
#endif
