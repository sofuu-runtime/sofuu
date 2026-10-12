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
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    /* Unroll 16 floats per iteration to saturate all 4 NEON FMA execution units */
    for (; i + 16 <= n; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);

        acc0 = vmlaq_f32(acc0, va0, vb0);
        acc1 = vmlaq_f32(acc1, va1, vb1);
        acc2 = vmlaq_f32(acc2, va2, vb2);
        acc3 = vmlaq_f32(acc3, va3, vb3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vmlaq_f32(acc0, va, vb);
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float32x2_t lo = vget_low_f32(acc0);
    float32x2_t hi = vget_high_f32(acc0);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

float sofuu_l2_f32(const float *a, const float *b, size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t diff0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t diff1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc0 = vmlaq_f32(acc0, diff0, diff0);
        acc1 = vmlaq_f32(acc1, diff1, diff1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t diff = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vmlaq_f32(acc0, diff, diff);
    }
    acc0 = vaddq_f32(acc0, acc1);
    float32x2_t lo = vget_low_f32(acc0);
    float32x2_t hi = vget_high_f32(acc0);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    for (; i < n; i++) { float d = a[i] - b[i]; result += d * d; }
    return sqrtf(result);
}

/* Fused single-pass cosine similarity: computes dot(a,b), norm(a), norm(b) in ONE memory traversal */
float sofuu_cosine_f32(const float *a, const float *b, size_t n) {
    float32x4_t acc_ab0 = vdupq_n_f32(0.0f);
    float32x4_t acc_ab1 = vdupq_n_f32(0.0f);
    float32x4_t acc_aa0 = vdupq_n_f32(0.0f);
    float32x4_t acc_aa1 = vdupq_n_f32(0.0f);
    float32x4_t acc_bb0 = vdupq_n_f32(0.0f);
    float32x4_t acc_bb1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);

        acc_ab0 = vmlaq_f32(acc_ab0, va0, vb0);
        acc_ab1 = vmlaq_f32(acc_ab1, va1, vb1);

        acc_aa0 = vmlaq_f32(acc_aa0, va0, va0);
        acc_aa1 = vmlaq_f32(acc_aa1, va1, va1);

        acc_bb0 = vmlaq_f32(acc_bb0, vb0, vb0);
        acc_bb1 = vmlaq_f32(acc_bb1, vb1, vb1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc_ab0 = vmlaq_f32(acc_ab0, va, vb);
        acc_aa0 = vmlaq_f32(acc_aa0, va, va);
        acc_bb0 = vmlaq_f32(acc_bb0, vb, vb);
    }

    acc_ab0 = vaddq_f32(acc_ab0, acc_ab1);
    acc_aa0 = vaddq_f32(acc_aa0, acc_aa1);
    acc_bb0 = vaddq_f32(acc_bb0, acc_bb1);

    float32x2_t lo_ab = vget_low_f32(acc_ab0), hi_ab = vget_high_f32(acc_ab0);
    float dot = vget_lane_f32(vpadd_f32(vadd_f32(lo_ab, hi_ab), vadd_f32(lo_ab, hi_ab)), 0);

    float32x2_t lo_aa = vget_low_f32(acc_aa0), hi_aa = vget_high_f32(acc_aa0);
    float norma = vget_lane_f32(vpadd_f32(vadd_f32(lo_aa, hi_aa), vadd_f32(lo_aa, hi_aa)), 0);

    float32x2_t lo_bb = vget_low_f32(acc_bb0), hi_bb = vget_high_f32(acc_bb0);
    float normb = vget_lane_f32(vpadd_f32(vadd_f32(lo_bb, hi_bb), vadd_f32(lo_bb, hi_bb)), 0);

    for (; i < n; i++) {
        dot += a[i] * b[i];
        norma += a[i] * a[i];
        normb += b[i] * b[i];
    }
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
    float dot = 0.0f;
    float na  = 0.0f;
    float nb  = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float den = sqrtf(na) * sqrtf(nb);
    if (den < 1e-10f) return 0.0f;
    return dot / den;
}
#endif
