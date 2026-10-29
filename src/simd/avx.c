/*
 * src/simd/avx.c — x86 AVX2 float32 vector ops
 *
 * Compiled only on x86_64 targets. Uses 256-bit YMM registers (8 x float32).
 * Falls back to scalar if AVX2 not detected at compile time.
 */
#include "simd.h"
#include <math.h>

/* ── R-27: extreme-magnitude escape hatch ──────────────────────────────
 * Identical to the NEON kernel's (src/simd/neon.c) so both arches answer
 * the same way. f32 cosine accumulation overflows once |a_i| > 1.8e19
 * (x*x > FLT_MAX) or once the sum of squares passes 3.4e38 (768 components
 * of 1e18 = 7.7e38), after which dot/denom is Inf/Inf = NaN or a silent
 * 0.0. An overflowed f32 sum can never return to a finite value, so a
 * non-finite accumulator is an exact detector; only then is the whole
 * computation redone in f64 after scaling each vector by its max |component|
 * (cosine is scale-invariant). Ordinary magnitudes keep the branch-free
 * accumulate loops and bit-identical results. */
static float sofuu_cosine_scaled_f32(const float *a, const float *b, size_t n) {
    double sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x = fabs((double)a[i]);
        double y = fabs((double)b[i]);
        if (!isfinite(x) || !isfinite(y)) return NAN;
        if (x > sa) sa = x;
        if (y > sb) sb = y;
    }
    if (sa == 0.0 || sb == 0.0) return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x = (double)a[i] / sa;
        double y = (double)b[i] / sb;
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    double den = sqrt(na) * sqrt(nb);
    if (!(den > 1e-10)) return 0.0f; /* also catches den == NaN */
    double r = dot / den;
    if (r > 1.0) r = 1.0;
    else if (r < -1.0) r = -1.0;
    return (float)r;
}

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
    /* R-27: overflow (Inf/NaN) or an underflowed-to-zero norm cannot be
     * answered from these f32 sums — redo scaled in f64. Four compares after
     * the loop, never inside it. */
    if (!isfinite(dot) || !isfinite(norma) || !isfinite(normb) ||
        !isfinite(denom) || norma == 0.0f || normb == 0.0f)
        return sofuu_cosine_scaled_f32(a, b, n);
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
    /* R-27: see sofuu_cosine_scaled_f32 — same overflow/underflow escape. */
    if (!isfinite(dot) || !isfinite(na) || !isfinite(nb) || !isfinite(den) ||
        na == 0.0f || nb == 0.0f)
        return sofuu_cosine_scaled_f32(a, b, n);
    if (den < 1e-10f) return 0.0f;
    return dot / den;
}
#endif
