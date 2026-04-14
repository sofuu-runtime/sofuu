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

float sofuu_dot_f32(const float *a, const float *b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    /* horizontal reduction */
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

float sofuu_l2_f32(const float *a, const float *b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
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
    float dot = sofuu_dot_f32(a, b, n);
    float na  = sofuu_dot_f32(a, a, n);
    float nb  = sofuu_dot_f32(b, b, n);
    float den = sqrtf(na) * sqrtf(nb);
    if (den < 1e-10f) return 0.0f;
    return dot / den;
}
#endif
