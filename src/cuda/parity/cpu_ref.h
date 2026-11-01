// src/cuda/parity/cpu_ref.h — independent scalar references for the CUDA
// kernel parity harnesses.
//
// These are intentionally written from the GGML format description and the
// documented Sofuu conventions (src/simd/infer_kernels.cpp), *not* by calling
// the kernels' own helpers. The host harness compares kernels.cu's host build
// against this file; the GPU harness compares the actual launches against it.
//
// Accumulation is done in double so the reference is not itself an
// fp32 summation-order artifact.

#ifndef SOFUU_CUDA_CPU_REF_H
#define SOFUU_CUDA_CPU_REF_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace sofuu_ref {

#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)

// Exact IEEE binary16 -> float, written independently of kernels.cu.
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            // Normalise the subnormal.
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant ? 0x400000u : 0u);
    } else {
        bits = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = (uint8_t)(q[j] & 63u);
        *m = (uint8_t)(q[j + 4] & 63u);
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

struct Dot {
    double value;  // high-precision reference dot
    double l1;     // sum |dequantised weight * x|
};

inline Dot dot_q8_0(const uint8_t *w, const float *x, int cols) {
    const BlockQ8_0 *blocks = reinterpret_cast<const BlockQ8_0 *>(w);
    int nb = cols / 32;
    double sum = 0.0, l1 = 0.0;
    for (int b = 0; b < nb; ++b) {
        double d = f16_to_f32(blocks[b].d);
        const float *xb = x + (size_t)b * 32;
        for (int j = 0; j < 32; ++j) {
            double wv = d * (double)blocks[b].qs[j];
            sum += wv * xb[j];
            l1 += std::fabs(wv * xb[j]);
        }
    }
    return {sum, l1};
}

inline Dot dot_q4_k(const uint8_t *w, const float *x, int cols) {
    const BlockQ4_K *blocks = reinterpret_cast<const BlockQ4_K *>(w);
    int nb = cols / 256;
    double sum = 0.0, l1 = 0.0;
    for (int b = 0; b < nb; ++b) {
        double d = f16_to_f32(blocks[b].d);
        double dmin = f16_to_f32(blocks[b].dmin);
        const uint8_t *q = blocks[b].qs;
        const float *xb = x + (size_t)b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, blocks[b].scales, &sc, &mn);
            double d1 = d * (double)sc, m1 = dmin * (double)mn;
            get_scale_min_k4(is + 1, blocks[b].scales, &sc, &mn);
            double d2 = d * (double)sc, m2 = dmin * (double)mn;
            for (int l = 0; l < 32; ++l) {
                double w1 = d1 * (double)(q[l] & 0x0fu) - m1;
                double w2 = d2 * (double)(q[l] >> 4) - m2;
                sum += w1 * xb[l] + w2 * xb[l + 32];
                l1 += std::fabs(w1 * xb[l]) + std::fabs(w2 * xb[l + 32]);
            }
            q += 32;
            xb += 64;
            is += 2;
        }
    }
    return {sum, l1};
}

inline Dot dot_q6_k(const uint8_t *w, const float *x, int cols) {
    const BlockQ6_K *blocks = reinterpret_cast<const BlockQ6_K *>(w);
    int nb = cols / 256;
    double sum = 0.0, l1 = 0.0;
    for (int b = 0; b < nb; ++b) {
        double d = f16_to_f32(blocks[b].d);
        const uint8_t *ql = blocks[b].ql;
        const uint8_t *qh = blocks[b].qh;
        const int8_t *sc = blocks[b].scales;
        const float *xb = x + (size_t)b * 256;
        for (int n = 0; n < 2; ++n) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l + 0] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                double w1 = d * (double)sc[is + 0] * (double)q1;
                double w2 = d * (double)sc[is + 2] * (double)q2;
                double w3 = d * (double)sc[is + 4] * (double)q3;
                double w4 = d * (double)sc[is + 6] * (double)q4;
                sum += w1 * xb[l + 0] + w2 * xb[l + 32] + w3 * xb[l + 64] +
                       w4 * xb[l + 96];
                l1 += std::fabs(w1 * xb[l + 0]) + std::fabs(w2 * xb[l + 32]) +
                      std::fabs(w3 * xb[l + 64]) + std::fabs(w4 * xb[l + 96]);
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return {sum, l1};
}

inline double dot_f32(const float *w, const float *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)w[i] * (double)x[i];
    return sum;
}

inline void rmsnorm_ref(const float *x, const float *w, float *y, int n,
                        float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    double scale = 1.0 / std::sqrt(ss / (double)n + (double)eps);
    for (int i = 0; i < n; ++i)
        y[i] = (float)((double)x[i] * scale * (double)w[i]);
}

inline void swiglu_ref(const float *g, const float *u, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        double s = 1.0 / (1.0 + std::exp(-(double)g[i]));
        y[i] = (float)((double)g[i] * s * (double)u[i]);
    }
}

inline void rope_ref(float *x, const float *cos_tab, const float *sin_tab,
                     int n_heads, int head_dim, int n_rot, int interleaved) {
    int half = n_rot >> 1;
    for (int h = 0; h < n_heads; ++h) {
        float *base = x + (size_t)h * (size_t)head_dim;
        for (int i = 0; i < half; ++i) {
            int i0 = interleaved ? (2 * i) : i;
            int i1 = interleaved ? (2 * i + 1) : (i + half);
            float c = cos_tab[i];
            float s = sin_tab[i];
            float x0 = base[i0];
            float x1 = base[i1];
            base[i0] = x0 * c - x1 * s;
            base[i1] = x0 * s + x1 * c;
        }
    }
}

inline void kv_write_ref(float *k_cache, float *v_cache, const float *k,
                         const float *v, int pos, int n_kv_heads, int head_dim,
                         int cap) {
    for (int h = 0; h < n_kv_heads; ++h) {
        size_t slot = ((size_t)h * (size_t)cap + (size_t)pos) * (size_t)head_dim;
        const float *krow = k + (size_t)h * (size_t)head_dim;
        const float *vrow = v + (size_t)h * (size_t)head_dim;
        for (int d = 0; d < head_dim; ++d) {
            k_cache[slot + (size_t)d] = krow[d];
            v_cache[slot + (size_t)d] = vrow[d];
        }
    }
}

// Deterministic fixture generation shared by both harnesses.
inline uint32_t lcg_next(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

inline float lcg_float(uint32_t &state, float lo, float hi) {
    uint32_t v = lcg_next(state);
    float unit = (float)(v >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * unit;
}

// finite f16 bit patterns used for block headers (never NaN/Inf).
static const uint16_t F16_0625 = 0x2c00u;  //  0.0625
static const uint16_t F16_2_POW_M10 = 0x1400u;  // 2^-10

} // namespace sofuu_ref

#endif // SOFUU_CUDA_CPU_REF_H
