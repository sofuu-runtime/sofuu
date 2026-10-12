/*
 * src/simd/avx.c — x86 AVX2 float32 vector ops
 *
 * Compiled only on x86_64 targets. Uses 256-bit YMM registers (8 x float32).
 * Falls back to scalar if AVX2 not detected at compile time.
 */
#include "simd.h"
#include <math.h>

#if defined(__x86_64__) && defined(__AVX2__)
#include <immintrin.h>

static inline float hsum256_ps(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

float sofuu_dot_f32(const float *a, const float *b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    float result = hsum256_ps(acc0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

float sofuu_l2_f32(const float *a, const float *b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 diff0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 diff1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    float result = hsum256_ps(acc0);
    for (; i < n; i++) { float d = a[i] - b[i]; result += d * d; }
    return sqrtf(result);
}

/* Fused single-pass cosine similarity: computes dot(a,b), norm(a), norm(b) in ONE memory traversal */
float sofuu_cosine_f32(const float *a, const float *b, size_t n) {
    __m256 acc_ab0 = _mm256_setzero_ps();
    __m256 acc_ab1 = _mm256_setzero_ps();
    __m256 acc_aa0 = _mm256_setzero_ps();
    __m256 acc_aa1 = _mm256_setzero_ps();
    __m256 acc_bb0 = _mm256_setzero_ps();
    __m256 acc_bb1 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);

        acc_ab0 = _mm256_fmadd_ps(va0, vb0, acc_ab0);
        acc_ab1 = _mm256_fmadd_ps(va1, vb1, acc_ab1);

        acc_aa0 = _mm256_fmadd_ps(va0, va0, acc_aa0);
        acc_aa1 = _mm256_fmadd_ps(va1, va1, acc_aa1);

        acc_bb0 = _mm256_fmadd_ps(vb0, vb0, acc_bb0);
        acc_bb1 = _mm256_fmadd_ps(vb1, vb1, acc_bb1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc_ab0 = _mm256_fmadd_ps(va, vb, acc_ab0);
        acc_aa0 = _mm256_fmadd_ps(va, va, acc_aa0);
        acc_bb0 = _mm256_fmadd_ps(vb, vb, acc_bb0);
    }

    acc_ab0 = _mm256_add_ps(acc_ab0, acc_ab1);
    acc_aa0 = _mm256_add_ps(acc_aa0, acc_aa1);
    acc_bb0 = _mm256_add_ps(acc_bb0, acc_bb1);

    float dot   = hsum256_ps(acc_ab0);
    float norma = hsum256_ps(acc_aa0);
    float normb = hsum256_ps(acc_bb0);

    for (; i < n; i++) {
        dot   += a[i] * b[i];
        norma += a[i] * a[i];
        normb += b[i] * b[i];
    }
    float denom = sqrtf(norma) * sqrtf(normb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

#else
/* ── Scalar fallback ── */

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
