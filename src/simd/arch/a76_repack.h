// src/simd/arch/a76_repack.h — Packed-tile GEMM/GEMV kernels for CPU prefill/decode.
//
// Portions ported from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors)
// ggml/src/ggml-cpu/repack.cpp *_generic kernels and repack helpers, adapted to the
// Sofuu kernel layer (pointer-based repack, renamed types, identical numerics).
#ifndef SOFUU_ARCH_A76_REPACK_H
#define SOFUU_ARCH_A76_REPACK_H

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef QK_K
#define QK_K 256
#endif
#define QK8_0 32
#define K_SCALE_SIZE 12
#define UNUSED(x) (void)(x)
#define GGML_RESTRICT __restrict

static inline float ggml_cpu_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h >> 15) & 1;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) { uint32_t bits=sign<<31; float v; memcpy(&v,&bits,4); return v; }
        uint32_t e=113,m=mant; while((m&0x400)==0){m<<=1;e--;}
        uint32_t bits=(sign<<31)|(e<<23)|((m&0x3ff)<<13); float v; memcpy(&v,&bits,4); return v;
    }
    if (exp == 31) { uint32_t bits=(sign<<31)|(0xff<<23)|(mant<<13); float v; memcpy(&v,&bits,4); return v; }
    uint32_t bits=((sign<<31)|((exp+127-15)<<23)|(mant<<13)); float v; memcpy(&v,&bits,4); return v;
}
#define GGML_CPU_FP16_TO_FP32(x) ggml_cpu_f16_to_f32(x)

struct tile_block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
};
struct tile_block_q6_K {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};
struct tile_block_q8_0 {
    uint16_t d;
    int8_t   qs[QK8_0];
};
struct tile_block_q4_Kx8 {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
};
// T1: q4_K repack-time pre-decoded sub-block scales, as a SEPARATE region
// (192 B per stock 1152 B block, +16.7% weight bytes) rather than appended
// to the tile. Measured on Pi 5 (2026-10-15): appending to the tile block
// cost the decode GEMV -10% (the 192-byte holes break its contiguous
// stream) while prefill stayed flat; two regions keep the gemv on an
// unbroken stock layout and give the compute-bound GEMM the section as a
// second stream. sb_scale holds the col scales as f16 (ints 0..63 — exact)
// per (sb, half) in col order; sb_min holds the raw widen bytes the decode
// obtains from its u32 lanes (kernel does the same vmovl_u8 the decode
// did). All f32 scale products stay < 2^24 (acc lane sums <= ~2^16, scales
// <= 63, pair sums < 2^23), so the f32-multiply apply chain is
// bit-identical to the stock int-multiply chain.
struct tile_ss_q4_Kx8s {
    __fp16  sb_scale[(QK_K / 64) * 2 * 8];  // 64 f16, group g = sb*2+half
    uint8_t sb_min[(QK_K / 64) * 2 * 8];    // 64 u8, same order
};
struct tile_block_q6_Kx8 {
    uint16_t d[8];
    int8_t scales[QK_K / 16 * 8];
    uint8_t ql[QK_K / 2 * 8];
    uint8_t qh[QK_K / 4 * 8];
};
struct tile_block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[QK8_0 * 4];
};
struct tile_block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
struct tile_block_q8_Kx4 {
    float d[4];
    int8_t qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
};

// 8-activation-row q8_K tile for the wider-batch q4_K prefill GEMM
// (SOFUU_GEMM_BATCH_TILE=8, see the kernel section further down). Two 4-row
// row groups in one block, with the per-(sb, half, kg) 16-byte row groups of
// the upper 4 rows stored directly after the lower 4 — one 32-byte load then
// fetches all 8 activation rows of a k group. Same bytes, same int8 values,
// same d/bsums as two tile_block_q8_Kx4 blocks; bsums are stored as the
// 32-k-half sums the kernels consume ([sb][row][half]).
struct tile_block_q8_Kx8 {
    float   d[8];
    int8_t  qs[QK_K * 8];
    int16_t bsums[QK_K / 4 * 2];
};
static_assert(sizeof(tile_block_q8_Kx8) == 2 * sizeof(tile_block_q8_Kx4),
              "8-row act tile must not change the scratch unit");

using tile_block_q4_K   = tile_block_q4_K;
using tile_block_q6_K   = tile_block_q6_K;
using tile_block_q8_0   = tile_block_q8_0;
using tile_block_q8_K   = tile_block_q8_K;

void ggml_gemv_q8_0_4x4_q8_0_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc);

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int32_t t; memcpy(&t, &val, sizeof(int32_t));
    return (t & 0x007fffff) - 0x00400000;
}

static inline uint16_t ggml_cpu_fp32_to_fp16(float val) {
    uint32_t x; memcpy(&x, &val, 4);
    uint32_t sign=(x>>31)&1;
    int32_t e=((x>>23)&0xff)-127+15;
    uint32_t m=(x>>13)&0x3ff;
    if (e <= 0) {
        if (e < -10) return (uint16_t)(sign<<15);
        m = ((x & 0x7fffff) | 0x800000) >> (1 - e + 13);
        if ((x & 0x7fffff) & ((1 << (1 - e + 13)) - 1)) m = (m + 1) & 0x3ff;
        return (uint16_t)((sign<<15) | m);
    }
    if (e >= 31) return (uint16_t)((sign<<15) | 0x7c00);
    if ((x & 0xffff) > 0x8000) m = m + 1;
    return (uint16_t)((sign<<15) | ((uint32_t)e<<10) | m);
}
#define GGML_CPU_FP32_TO_FP16(x) ggml_cpu_fp32_to_fp16(x)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


void ggml_quantize_mat_q8_0_4x8_generic(const float * __restrict x, void * __restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    tile_block_q8_0x4 * __restrict y = (tile_block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x8_generic(const float * __restrict x, void * __restrict vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    tile_block_q8_Kx4 * __restrict y = (tile_block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * __restrict      s,
                                                 size_t                     bs,
                                                 const void * __restrict vx,
                                                 const void * __restrict vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;
    const int     q8_half_stride    = 512;
    const int     q8_low_high_step  = 256;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];

    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_Kx4 * a_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q6_Kx8 * b_ptr = (const tile_block_q6_Kx8 *) vx + (x * nb);

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    const int base_h = base_l + 64;

                    const int scale_idx_l = base_l / 16;
                    const int scale_idx_h = base_h / 16;

                    const int qh_shift_l = ((base_l % 128) / 32) * 2;
                    const int qh_shift_h = ((base_h % 128) / 32) * 2;

                    const int qh_half_l = (base_l / 128) * 32;
                    const int qh_half_h = (base_h / 128) * 32;

                    const int q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            int sumi_l = 0;
                            int sumi_h = 0;

                            for (int i = 0; i < blocklen; i++) {
                                const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                const int qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                const int qh_chunk_l = qh_idx_l / blocklen;
                                const int qh_pos_l   = qh_idx_l % blocklen;
                                const int qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                const int hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                const int qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                const int qh_chunk_h = qh_idx_h / blocklen;
                                const int qh_pos_h   = qh_idx_h % blocklen;
                                const int qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                const int hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                const int q_l = ((hi_2_l << 4) | l_4) - 32;
                                const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                                const int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                const int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_low_high_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <int M, int N>
static void ggml_gemv_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * __restrict      s,
                                                 size_t                     bs,
                                                 const void * __restrict vx,
                                                 const void * __restrict vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];

    const tile_block_q8_K * a_ptr = (const tile_block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q6_Kx8 * b_ptr = (const tile_block_q6_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                const int base_h = base_l + 64;

                const int scale_idx_l = base_l / 16;
                const int scale_idx_h = base_h / 16;

                const int qh_shift_l = ((base_l % 128) / 32) * 2;
                const int qh_shift_h = ((base_h % 128) / 32) * 2;

                const int qh_half_l = (base_l / 128) * 32;
                const int qh_half_h = (base_h / 128) * 32;

                for (int j = 0; j < ncols_interleaved; j++) {
                    const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    int sumi_l = 0;
                    int sumi_h = 0;

                    for (int i = 0; i < blocklen; i++) {
                        const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        const int qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        const int qh_chunk_l  = qh_idx_l / blocklen;
                        const int qh_pos_l    = qh_idx_l % blocklen;
                        const int qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        const int hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        const int qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        const int qh_chunk_h  = qh_idx_h / blocklen;
                        const int qh_pos_h    = qh_idx_h % blocklen;
                        const int qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        const int hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        const int q_l = ((hi_2_l << 4) | l_4) - 32;
                        const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                        const int8_t a_l = a_ptr[l].qs[base_l + i];
                        const int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const tile_block_q8_K * a_ptr = (const tile_block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q4_Kx8 * b_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q8_0_4x8_q8_0_generic(int                        n,
                                     float * __restrict      s,
                                     size_t                     bs,
                                     const void * __restrict vx,
                                     const void * __restrict vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const tile_block_q8_0 * a_ptr = (const tile_block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_Kx4 * a_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q4_Kx8 * b_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x8_q8_0_generic(int                        n,
                                     float * __restrict      s,
                                     size_t                     bs,
                                     const void * __restrict vx,
                                     const void * __restrict vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_0x4 * a_ptr = (const tile_block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}
static tile_block_q4_Kx8 make_block_q4_Kx8(tile_block_q4_K * in, unsigned int blck_size_interleave) {
    tile_block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    return out;
}

static tile_block_q6_Kx8 make_block_q6_Kx8(tile_block_q6_K * in, unsigned int blck_size_interleave) {
    tile_block_q6_Kx8  out;
    constexpr int n_blocks = 8;  // Kx8
    for (int i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    const int end_ls = QK_K * 4 / blck_size_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end_ls; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_size_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_size_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    const int end_hs = end_ls / 2;
    for (int i = 0; i < end_hs; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_size_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr int n_scales = QK_K / 16;

    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    return out;
}

static tile_block_q8_0x4 make_block_q8_0x4(tile_block_q8_0 * in, unsigned int blck_size_interleave) {
    tile_block_q8_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 4 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 4;
        int src_offset = (i / 4) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    return out;
}


// ── Repack: raw GGUF row-major blocks → interleaved tile layout ──────
static inline int sofuu_repack_q4k_8bl(uint8_t * dst, const uint8_t * src, int64_t nrow, int64_t nblocks, unsigned interleave = 4) {
    if (nrow % 8 != 0 || nblocks % 8 != 0 /* cols % 2048 */ ) { /* cols handled by caller gate */ }
    tile_block_q4_Kx8 * d = (tile_block_q4_Kx8 *) dst;
    const tile_block_q4_K * s = (const tile_block_q4_K *) src;
    tile_block_q4_K tmp[8];
    for (int64_t b = 0; b < nrow; b += 8) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q4_Kx8(tmp, interleave);
        }
        s += 8 * nblocks;
    }
    return 0;
}
static inline int sofuu_repack_q6k_8bl(uint8_t * dst, const uint8_t * src, int64_t nrow, int64_t nblocks, unsigned interleave = 4) {
    tile_block_q6_Kx8 * d = (tile_block_q6_Kx8 *) dst;
    const tile_block_q6_K * s = (const tile_block_q6_K *) src;
    tile_block_q6_K tmp[8];
    for (int64_t b = 0; b < nrow; b += 8) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q6_Kx8(tmp, interleave);
        }
        s += 8 * nblocks;
    }
    return 0;
}
static inline int sofuu_repack_q8_0_4bl(uint8_t * dst, const uint8_t * src, int64_t nrow, int64_t nblocks, unsigned interleave = 4) {
    tile_block_q8_0x4 * d = (tile_block_q8_0x4 *) dst;
    const tile_block_q8_0 * s = (const tile_block_q8_0 *) src;
    tile_block_q8_0 tmp[4];
    for (int64_t b = 0; b < nrow; b += 4) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 4; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q8_0x4(tmp, interleave);
        }
        s += 4 * nblocks;
    }
    return 0;
}

// ── Ranged repacks for parallel warm: rows [r8_begin, r8_end) of 8-row
// groups land at the matching dst block range; a full-matrix call is the
// concatenation of all ranges, so the thread-pool split is bit-identical. ──
static inline int sofuu_repack_q4k_8bl_range(uint8_t * dst, const uint8_t * src, int64_t nblocks, int64_t r8_begin, int64_t r8_end, unsigned interleave = 4) {
    tile_block_q4_Kx8 * d = (tile_block_q4_Kx8 *) dst + r8_begin * nblocks;
    const tile_block_q4_K * s = (const tile_block_q4_K *) src + r8_begin * 8 * nblocks;
    tile_block_q4_K tmp[8];
    for (int64_t b = r8_begin; b < r8_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q4_Kx8(tmp, interleave);
        }
        s += 8 * nblocks;
    }
    return 0;
}
static inline int sofuu_repack_q6k_8bl_range(uint8_t * dst, const uint8_t * src, int64_t nblocks, int64_t r8_begin, int64_t r8_end, unsigned interleave = 4) {
    tile_block_q6_Kx8 * d = (tile_block_q6_Kx8 *) dst + r8_begin * nblocks;
    const tile_block_q6_K * s = (const tile_block_q6_K *) src + r8_begin * 8 * nblocks;
    tile_block_q6_K tmp[8];
    for (int64_t b = r8_begin; b < r8_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q6_Kx8(tmp, interleave);
        }
        s += 8 * nblocks;
    }
    return 0;
}
static inline int sofuu_repack_q8_0_4bl_range(uint8_t * dst, const uint8_t * src, int64_t nblocks, int64_t r4_begin, int64_t r4_end, unsigned interleave = 4) {
    tile_block_q8_0x4 * d = (tile_block_q8_0x4 *) dst + r4_begin * nblocks;
    const tile_block_q8_0 * s = (const tile_block_q8_0 *) src + r4_begin * 4 * nblocks;
    tile_block_q8_0 tmp[4];
    for (int64_t b = r4_begin; b < r4_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 4; i++) tmp[i] = s[x + i * nblocks];
            *d++ = make_block_q8_0x4(tmp, interleave);
        }
        s += 4 * nblocks;
    }
    return 0;
}

// Bulk row q8_K quantizer (decode GEMV path): 256-elem blocks + bsums.
static inline void sofuu_quantize_row_q8_K(const float * __restrict x, tile_block_q8_K * __restrict y, int64_t cols) {
    const int64_t nb = cols / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float * xb = x + i * QK_K;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) amax = MAX(amax, fabsf(xb[j]));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;
        y[i].d = d;
        for (int g = 0; g < 16; ++g) y[i].bsums[g] = 0;
        for (int j = 0; j < QK_K; ++j) {
            const int v = nearest_int(xb[j] * id);
            y[i].qs[j] = (int8_t)v;
            y[i].bsums[j / 16] += (int16_t)v;
        }
    }
}

void ggml_quantize_mat_q8_0_4x4_generic(const float * __restrict x, void * __restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    tile_block_q8_0x4 * __restrict y = (tile_block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

// NEON activation quantizer for the 4-row q8_K tile — bit-identical to
// ggml_quantize_mat_q8_K_4x4_generic below. The generic was the
// single-threaded serial prologue of every q4_K/q6_K prefill GEMM (measured
// 22-37% of the path at batch 64 on Pi 5, ~10 scalar ops + div/mod per
// element). Design notes for exactness:
//  - amax via vmaxvq (max is exact/associative), then max's SIGN recovered
//    with an early-break scan for the first |x| == amax — identical to the
//    generic's strict `>` update (first attainer wins on ties).
//  - iscale/d use the same scalar expressions (4 divisions per block).
//  - Rounding: nearest_int(f) = bit pattern of (f + 12582912.0f); lanes go
//    f*iscale -> +magic -> vsub magic in int32 -> narrow-low bytes, which is
//    that same bit trick without the scalar float<->int bottleneck (values
//    |q| <= 127 so every narrowing is exact).
//  - bsums: all four j-runs in a (t, row) step share one slot; int16 adds
//    are exact (|sums| <= 16*127), so lane order cannot change results.
//  - Rewrite of the per-element index math: for j = 16t + 4s + u (t<64,
//    s<4, u<4): src_id = s, src_offset = 4t + u (a contiguous vld1q), and
//    bsums-index = 4s + (t>>4)*16 + ((t>>2)&3).
static inline void ggml_quantize_mat_q8_K_4x4_neon(const float * __restrict x, void * __restrict vy, int64_t k) {
#if defined(__ARM_NEON)
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    tile_block_q8_Kx4 * __restrict y = (tile_block_q8_Kx4 *) vy;
    const float32x4_t f32_magic = vdupq_n_f32(12582912.0f);
    const int32x4_t i32_magic = vreinterpretq_s32_f32(f32_magic); // 0x4B400000 bit pattern

    for (int i = 0; i < nb; i++) {
        float iscale[4];
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            const float *row = x + row_iter * k + i * QK_K;
            float32x4_t va = vabsq_f32(vld1q_f32(row));
            for (int j = 4; j < QK_K; j += 4) {
                va = vmaxq_f32(va, vabsq_f32(vld1q_f32(row + j)));
            }
            const float amax = vmaxvq_f32(va);

            float max = 0.0f;
            if (amax > 0.0f) {
                for (int j = 0; j < QK_K; j++) {
                    if (fabsf(row[j]) == amax) { max = row[j]; break; }
                }
            }
            iscale[row_iter] = amax ? -127.f / max : 0;
            y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
        }

        memset(y[i].bsums, 0, sizeof(y[i].bsums));

        for (int t = 0; t < QK_K / 4; t++) {
            int16x4_t qs16[4];
            int16_t badd[4];
            for (int s = 0; s < 4; s++) {
                const float32x4_t q = vmulq_n_f32(vld1q_f32(x + s * k + i * QK_K + 4 * t), iscale[s]);
                const int32x4_t val = vsubq_s32(vreinterpretq_s32_f32(vaddq_f32(q, f32_magic)), i32_magic);
                qs16[s] = vmovn_s32(val);
                badd[s] = (int16_t) vaddvq_s32(val);
                const int idx = 4 * s + (t >> 4) * 16 + ((t >> 2) & 3);
                y[i].bsums[idx] += badd[s];
            }
            vst1_s8(y[i].qs + 16 * t, vmovn_s16(vcombine_s16(qs16[0], qs16[1])));
            vst1_s8(y[i].qs + 16 * t + 8, vmovn_s16(vcombine_s16(qs16[2], qs16[3])));
        }
    }
#else
    ggml_quantize_mat_q8_K_4x4_generic(x, vy, k);
#endif
}

void ggml_quantize_mat_q8_K_4x4_generic(const float * __restrict x, void * __restrict vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    tile_block_q8_Kx4 * __restrict y = (tile_block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}



void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const tile_block_q8_K * a_ptr = (const tile_block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q4_Kx8 * b_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_Kx4 * a_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q4_Kx8 * b_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    tile_block_q8_0x4 * GGML_RESTRICT y = (tile_block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

void ggml_gemv_q8_0_4x4_q8_0(int                        n,
                             float * __restrict      s,
                             size_t                     bs,
                             const void * __restrict vx,
                             const void * __restrict vy,
                             int                        nr,
                             int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const tile_block_q8_0 * a_ptr = (const tile_block_q8_0 *) vy;
        float32x4_t        acc   = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16x4_t b_low  = vld1q_s8_x4((const int8_t *) b_ptr->qs);
            int8x16x4_t b_high = vld1q_s8_x4((const int8_t *) b_ptr->qs + 64);
            float16x4_t bd     = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16x2_t a  = vld1q_s8_x2(a_ptr->qs);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret = vdupq_n_s32(0);

            ret = vdotq_laneq_s32(ret, b_low.val[0], a.val[0], 0);
            ret = vdotq_laneq_s32(ret, b_low.val[1], a.val[0], 1);
            ret = vdotq_laneq_s32(ret, b_low.val[2], a.val[0], 2);
            ret = vdotq_laneq_s32(ret, b_low.val[3], a.val[0], 3);

            ret = vdotq_laneq_s32(ret, b_high.val[0], a.val[1], 0);
            ret = vdotq_laneq_s32(ret, b_high.val[1], a.val[1], 1);
            ret = vdotq_laneq_s32(ret, b_high.val[2], a.val[1], 2);
            ret = vdotq_laneq_s32(ret, b_high.val[3], a.val[1], 3);

            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;

#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x4_q8_0(int                        n,
                             float * __restrict      s,
                             size_t                     bs,
                             const void * __restrict vx,
                             const void * __restrict vy,
                             int                        nr,
                             int                        nc,
                             const float * in_acc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_0x4 * a_ptr = (const tile_block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx + (x * nb);

            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = vdupq_n_f32(0);
            }

            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *) a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *) b_ptr[l].d));

                int32x4_t sumi_0 = vdupq_n_s32(0);
                int32x4_t sumi_1 = vdupq_n_s32(0);
                int32x4_t sumi_2 = vdupq_n_s32(0);
                int32x4_t sumi_3 = vdupq_n_s32(0);

                for (int k_group = 0; k_group < 8; k_group += 4) {
                    int8x16x4_t a = vld1q_s8_x4(a_ptr[l].qs + 16 * k_group);
                    int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * k_group);

                    for (int k = 0; k < 4; k++) {
                        sumi_0 = vdotq_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }

                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }

            for (int m = 0; m < 4; m++) {
                const int offset = (y * 4 + m) * bs + x * 4;
                if (in_acc) {
                    vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), sumf[m]));
                } else {
                    vst1q_f32(s + offset, sumf[m]);
                }
            }
        }
    }
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

// 2x activation-panel variant: identical tile_block_q8_0x4 layout and the
// identical per-block fold schedule as ggml_gemm_q8_0_4x4_q8_0 — vx is ONE
// 4-row weight group (the dispatch keeps passing it per row group), vy is
// the activation scratch. Differences: each weight-tile row and weight-scale
// convert is shared across TWO consecutive 4-row activation groups (8 acts)
// per block, halving b-side loads and b_d converts per MAC. Fold order per
// output is unchanged -> bit-identical to the 4x4 kernel. nr % 8 == 0.
void ggml_gemm_q8_0_8ax4_q8_0(int                        n,
                              float * __restrict      s,
                              size_t                     bs,
                              const void * __restrict vx,
                              const void * __restrict vy,
                              int                        nr,
                              int                        nc,
                              const float * in_acc) {
    const int nb = n / QK8_0;
    assert(n % QK8_0 == 0);
    assert(nr % 8 == 0);
    (void) nc;

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx;
    for (int y2 = 0; y2 < nr / 8; y2++) {
        const tile_block_q8_0x4 * a0_ptr = (const tile_block_q8_0x4 *) vy + ((size_t)(2 * y2) * nb);
        const tile_block_q8_0x4 * a1_ptr = (const tile_block_q8_0x4 *) vy + ((size_t)(2 * y2 + 1) * nb);

        float32x4_t sumf0[4];
        float32x4_t sumf1[4];
        for (int m = 0; m < 4; m++) {
            sumf0[m] = vdupq_n_f32(0);
            sumf1[m] = vdupq_n_f32(0);
        }

        for (int l = 0; l < nb; l++) {
            float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *) b_ptr[l].d));

            int32x4_t s00 = vdupq_n_s32(0), s01 = vdupq_n_s32(0), s02 = vdupq_n_s32(0), s03 = vdupq_n_s32(0);
            for (int kg = 0; kg < 8; kg += 4) {
                int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * kg);
                int8x16x4_t a = vld1q_s8_x4(a0_ptr[l].qs + 16 * kg);
                for (int k = 0; k < 4; k++) {
                    s00 = vdotq_laneq_s32(s00, b.val[k], a.val[k], 0);
                    s01 = vdotq_laneq_s32(s01, b.val[k], a.val[k], 1);
                    s02 = vdotq_laneq_s32(s02, b.val[k], a.val[k], 2);
                    s03 = vdotq_laneq_s32(s03, b.val[k], a.val[k], 3);
                }
            }
            float32x4_t a_d0 = vcvt_f32_f16(vld1_f16((const float16_t *) a0_ptr[l].d));
            sumf0[0] = vmlaq_f32(sumf0[0], vmulq_laneq_f32(b_d, a_d0, 0), vcvtq_f32_s32(s00));
            sumf0[1] = vmlaq_f32(sumf0[1], vmulq_laneq_f32(b_d, a_d0, 1), vcvtq_f32_s32(s01));
            sumf0[2] = vmlaq_f32(sumf0[2], vmulq_laneq_f32(b_d, a_d0, 2), vcvtq_f32_s32(s02));
            sumf0[3] = vmlaq_f32(sumf0[3], vmulq_laneq_f32(b_d, a_d0, 3), vcvtq_f32_s32(s03));

            int32x4_t s10 = vdupq_n_s32(0), s11 = vdupq_n_s32(0), s12 = vdupq_n_s32(0), s13 = vdupq_n_s32(0);
            for (int kg = 0; kg < 8; kg += 4) {
                int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * kg);
                int8x16x4_t a = vld1q_s8_x4(a1_ptr[l].qs + 16 * kg);
                for (int k = 0; k < 4; k++) {
                    s10 = vdotq_laneq_s32(s10, b.val[k], a.val[k], 0);
                    s11 = vdotq_laneq_s32(s11, b.val[k], a.val[k], 1);
                    s12 = vdotq_laneq_s32(s12, b.val[k], a.val[k], 2);
                    s13 = vdotq_laneq_s32(s13, b.val[k], a.val[k], 3);
                }
            }
            float32x4_t a_d1 = vcvt_f32_f16(vld1_f16((const float16_t *) a1_ptr[l].d));
            sumf1[0] = vmlaq_f32(sumf1[0], vmulq_laneq_f32(b_d, a_d1, 0), vcvtq_f32_s32(s10));
            sumf1[1] = vmlaq_f32(sumf1[1], vmulq_laneq_f32(b_d, a_d1, 1), vcvtq_f32_s32(s11));
            sumf1[2] = vmlaq_f32(sumf1[2], vmulq_laneq_f32(b_d, a_d1, 2), vcvtq_f32_s32(s12));
            sumf1[3] = vmlaq_f32(sumf1[3], vmulq_laneq_f32(b_d, a_d1, 3), vcvtq_f32_s32(s13));
        }

        for (int m = 0; m < 4; m++) {
            const int off0 = ((2 * y2) * 4 + m) * (int)bs;
            if (in_acc) {
                vst1q_f32(s + off0, vaddq_f32(vld1q_f32(s + off0), sumf0[m]));
            } else {
                vst1q_f32(s + off0, sumf0[m]);
            }
            const int off1 = ((2 * y2 + 1) * 4 + m) * (int)bs;
            if (in_acc) {
                vst1q_f32(s + off1, vaddq_f32(vld1q_f32(s + off1), sumf1[m]));
            } else {
                vst1q_f32(s + off1, sumf1[m]);
            }
        }
    }
    return;
#endif
    (void) n; (void) s; (void) bs; (void) vx; (void) vy; (void) nr; (void) nc; (void) in_acc;
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
void ggml_gemv_q8_0_4x4_q8_0_generic(int                        n,
                                     float * __restrict      s,
                                     size_t                     bs,
                                     const void * __restrict vx,
                                     const void * __restrict vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const tile_block_q8_0 * a_ptr = (const tile_block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemm_q8_0_4x4_q8_0_generic(int                        n,
                                     float * __restrict      s,
                                     size_t                     bs,
                                     const void * __restrict vx,
                                     const void * __restrict vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const tile_block_q8_0x4 * a_ptr = (const tile_block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q8_0x4 * b_ptr = (const tile_block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Helper for decoding scales and mins of Q4_K/Q6_K block formats (8-col repack)
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static inline void decode_q_Kx8_6bit_scales(const uint8_t * scales_in, int16x8_t * out_mins, int8_t * out_scales) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;
    constexpr uint8_t  scales_size = 12;

    uint32_t sm[3];
    memcpy(sm, scales_in, scales_size);

    const uint32_t   mins_0_3 = sm[1] & kmask1;
    const uint32_t   mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const uint32x2_t mins_u32 = { mins_0_3, mins_4_7 };

    *out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins_u32)));

    uint32_t scales_u32[2];
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    memcpy(out_scales, scales_u32, 8);
}
#endif

// T1 fill: derive the pre-decoded scale section from an already interleaved
// stock tile block. Runs the stock 12-byte-group decode once per (subblock,
// half) so the stored bytes are what the stock kernels' hot loop computes —
// correctness by construction.
static inline void fill_q4k_s_block(tile_ss_q4_Kx8s * __restrict out, const tile_block_q4_Kx8 * __restrict in) {
    for (int g = 0; g < (QK_K / 64) * 2; g++) {
        int8_t    aux[8];
        int16x8_t mins;
        decode_q_Kx8_6bit_scales(&in->scales[g * 12], &mins, aux);
        __fp16 * sp = out->sb_scale + g * 8;
        for (int r = 0; r < 8; r++) {
            // GGML_CPU_FP32_TO_FP16 returns the f16 BIT PATTERN as uint16_t —
            // must memcpy it into the __fp16 slot (a plain assignment would
            // value-convert the pattern instead of reinterpreting it).
            uint16_t bits = ggml_cpu_fp32_to_fp16((float)aux[r]);
            memcpy(&sp[r], &bits, 2);
        }
        vst1_u8(out->sb_min + g * 8, vqmovn_u16(vreinterpretq_u16_s16(mins)));
    }
}

void ggml_gemm_q4_K_8x4_q8_K_s(int n, float * __restrict s, size_t bs, const void * __restrict vx,
                               const void * __restrict ss, const void * __restrict vy, int nr, int nc,
                               const float * in_acc);
void ggml_gemv_q4_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc);
void ggml_gemv_q6_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc);
void ggml_gemv_q8_0_4x4_q8_0(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc);
void ggml_gemm_q8_0_8ax4_q8_0(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc);

// Pair + 8-act panel: gate and up weight groups (vx1, vx2) share the SAME
// quantized activations (vy), processed two 4-row groups at a time (8 acts).
// Fold schedule per output matches ggml_gemm_q8_0_4x4_q8_0 bit for bit.
// nr (activation count) % 8 == 0 required.
void ggml_gemm_q8_0_8ax4_pair_q8_0(int                        n,
                                   float * __restrict      s1,
                                   float * __restrict      s2,
                                   size_t                     bs,
                                   const void * __restrict vx1,
                                   const void * __restrict vx2,
                                   const void * __restrict vy,
                                   int                        nr,
                                   int                        nc) {
    const int nb = n / QK8_0;
    assert(n % QK8_0 == 0);
    assert(nr % 8 == 0);
    (void) nc;

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const tile_block_q8_0x4 * u_ptr = (const tile_block_q8_0x4 *) vx1;
    const tile_block_q8_0x4 * v_ptr = (const tile_block_q8_0x4 *) vx2;
    for (int y2 = 0; y2 < nr / 8; y2++) {
        const tile_block_q8_0x4 * a0_ptr = (const tile_block_q8_0x4 *) vy + ((size_t)(2 * y2) * nb);
        const tile_block_q8_0x4 * a1_ptr = (const tile_block_q8_0x4 *) vy + ((size_t)(2 * y2 + 1) * nb);

        float32x4_t f00[4], f01[4], f10[4], f11[4];
        for (int m = 0; m < 4; m++) {
            f00[m] = vdupq_n_f32(0); f01[m] = vdupq_n_f32(0);
            f10[m] = vdupq_n_f32(0); f11[m] = vdupq_n_f32(0);
        }

        for (int l = 0; l < nb; l++) {
            float32x4_t u_d = vcvt_f32_f16(vld1_f16((const float16_t *) u_ptr[l].d));
            float32x4_t v_d = vcvt_f32_f16(vld1_f16((const float16_t *) v_ptr[l].d));

            int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0), a02 = vdupq_n_s32(0), a03 = vdupq_n_s32(0);
            int32x4_t b00 = vdupq_n_s32(0), b01 = vdupq_n_s32(0), b02 = vdupq_n_s32(0), b03 = vdupq_n_s32(0);
            for (int kg = 0; kg < 8; kg += 4) {
                int8x16x4_t u = vld1q_s8_x4(u_ptr[l].qs + 16 * kg);
                int8x16x4_t v = vld1q_s8_x4(v_ptr[l].qs + 16 * kg);
                int8x16x4_t x0 = vld1q_s8_x4(a0_ptr[l].qs + 16 * kg);
                for (int k = 0; k < 4; k++) {
                    a00 = vdotq_laneq_s32(a00, u.val[k], x0.val[k], 0);
                    a01 = vdotq_laneq_s32(a01, u.val[k], x0.val[k], 1);
                    a02 = vdotq_laneq_s32(a02, u.val[k], x0.val[k], 2);
                    a03 = vdotq_laneq_s32(a03, u.val[k], x0.val[k], 3);
                    b00 = vdotq_laneq_s32(b00, v.val[k], x0.val[k], 0);
                    b01 = vdotq_laneq_s32(b01, v.val[k], x0.val[k], 1);
                    b02 = vdotq_laneq_s32(b02, v.val[k], x0.val[k], 2);
                    b03 = vdotq_laneq_s32(b03, v.val[k], x0.val[k], 3);
                }
            }
            float32x4_t d0 = vcvt_f32_f16(vld1_f16((const float16_t *) a0_ptr[l].d));
            // scale factor for output (w-row j, act m) = u_d[j] * d0[m] — the
            // original's per-act-lane broadcast form, kept bit-identical.
            f00[0] = vmlaq_f32(f00[0], vmulq_laneq_f32(u_d, d0, 0), vcvtq_f32_s32(a00));
            f00[1] = vmlaq_f32(f00[1], vmulq_laneq_f32(u_d, d0, 1), vcvtq_f32_s32(a01));
            f00[2] = vmlaq_f32(f00[2], vmulq_laneq_f32(u_d, d0, 2), vcvtq_f32_s32(a02));
            f00[3] = vmlaq_f32(f00[3], vmulq_laneq_f32(u_d, d0, 3), vcvtq_f32_s32(a03));
            f01[0] = vmlaq_f32(f01[0], vmulq_laneq_f32(v_d, d0, 0), vcvtq_f32_s32(b00));
            f01[1] = vmlaq_f32(f01[1], vmulq_laneq_f32(v_d, d0, 1), vcvtq_f32_s32(b01));
            f01[2] = vmlaq_f32(f01[2], vmulq_laneq_f32(v_d, d0, 2), vcvtq_f32_s32(b02));
            f01[3] = vmlaq_f32(f01[3], vmulq_laneq_f32(v_d, d0, 3), vcvtq_f32_s32(b03));

            int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0), a12 = vdupq_n_s32(0), a13 = vdupq_n_s32(0);
            int32x4_t b10 = vdupq_n_s32(0), b11 = vdupq_n_s32(0), b12 = vdupq_n_s32(0), b13 = vdupq_n_s32(0);
            for (int kg = 0; kg < 8; kg += 4) {
                int8x16x4_t u = vld1q_s8_x4(u_ptr[l].qs + 16 * kg);
                int8x16x4_t v = vld1q_s8_x4(v_ptr[l].qs + 16 * kg);
                int8x16x4_t x1 = vld1q_s8_x4(a1_ptr[l].qs + 16 * kg);
                for (int k = 0; k < 4; k++) {
                    a10 = vdotq_laneq_s32(a10, u.val[k], x1.val[k], 0);
                    a11 = vdotq_laneq_s32(a11, u.val[k], x1.val[k], 1);
                    a12 = vdotq_laneq_s32(a12, u.val[k], x1.val[k], 2);
                    a13 = vdotq_laneq_s32(a13, u.val[k], x1.val[k], 3);
                    b10 = vdotq_laneq_s32(b10, v.val[k], x1.val[k], 0);
                    b11 = vdotq_laneq_s32(b11, v.val[k], x1.val[k], 1);
                    b12 = vdotq_laneq_s32(b12, v.val[k], x1.val[k], 2);
                    b13 = vdotq_laneq_s32(b13, v.val[k], x1.val[k], 3);
                }
            }
            float32x4_t d1 = vcvt_f32_f16(vld1_f16((const float16_t *) a1_ptr[l].d));
            f10[0] = vmlaq_f32(f10[0], vmulq_laneq_f32(u_d, d1, 0), vcvtq_f32_s32(a10));
            f10[1] = vmlaq_f32(f10[1], vmulq_laneq_f32(u_d, d1, 1), vcvtq_f32_s32(a11));
            f10[2] = vmlaq_f32(f10[2], vmulq_laneq_f32(u_d, d1, 2), vcvtq_f32_s32(a12));
            f10[3] = vmlaq_f32(f10[3], vmulq_laneq_f32(u_d, d1, 3), vcvtq_f32_s32(a13));
            f11[0] = vmlaq_f32(f11[0], vmulq_laneq_f32(v_d, d1, 0), vcvtq_f32_s32(b10));
            f11[1] = vmlaq_f32(f11[1], vmulq_laneq_f32(v_d, d1, 1), vcvtq_f32_s32(b11));
            f11[2] = vmlaq_f32(f11[2], vmulq_laneq_f32(v_d, d1, 2), vcvtq_f32_s32(b12));
            f11[3] = vmlaq_f32(f11[3], vmulq_laneq_f32(v_d, d1, 3), vcvtq_f32_s32(b13));
        }

        for (int m = 0; m < 4; m++) {
            const int o0 = ((2 * y2) * 4 + m) * (int)bs;
            vst1q_f32(s1 + o0, f00[m]);
            vst1q_f32(s2 + o0, f01[m]);
            const int o1 = ((2 * y2 + 1) * 4 + m) * (int)bs;
            vst1q_f32(s1 + o1, f10[m]);
            vst1q_f32(s2 + o1, f11[m]);
        }
    }
    return;
#endif
    (void) n; (void) s1; (void) s2; (void) bs; (void) vx1; (void) vx2; (void) vy; (void) nr; (void) nc;
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s1, bs, vx1, vy, nr, nc);
}
void ggml_gemm_q8_0_8ax4_pair_q8_0(int n, float * __restrict s1, float * __restrict s2, size_t bs, const void * __restrict vx1, const void * __restrict vx2, const void * __restrict vy, int nr, int nc);

void ggml_gemv_q4_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    col_groups = ncols_interleaved / 4; // 0123 and 4567
    const uint8x16_t m4b        = vdupq_n_u8(0x0f);

    // 1x8 tile = 2 x 4
    float32x4_t acc_f32[col_groups];

    const tile_block_q8_K * __restrict q8_ptr = (const tile_block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q4_Kx8 * __restrict q4_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);

        for (int i = 0; i < col_groups; i++) {
            acc_f32[i] = vdupq_n_f32(0);
        }

        for (int b = 0; b < nb; b++) {
            float32x4_t q4_d_0        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));      // d0 d1 d2 d3
            float32x4_t q4_d_1        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));  // d4 d5 d6 d7
            float32x4_t q8_d          = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0123 = vmulq_f32(q4_d_0, q8_d);
            float32x4_t sb_scale_4567 = vmulq_f32(q4_d_1, q8_d);
            float32x4_t q4_dmin_0     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));      // dmin 0..3
            float32x4_t q4_dmin_1     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));  // dmin 4..7
            float32x4_t sb_min_0123   = vmulq_f32(q4_dmin_0, q8_d);
            float32x4_t sb_min_4567   = vmulq_f32(q4_dmin_1, q8_d);

            // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
            int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
            int32x4_t acc_lo[col_groups];
            int32x4_t acc_hi[col_groups];

            // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int16_t         bsums_arr[8];
            vst1q_s16(bsums_arr, bsums);
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int i = 0; i < col_groups; i++) {
                    acc_lo[i] = vdupq_n_s32(0);
                    acc_hi[i] = vdupq_n_s32(0);
                }
                // Need scales for the low and high nibbles
                // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                int16x8_t q4sb_mins[2];
                int16x8_t q4sb_scales[2];
                for (int i = 0; i < 2; i++) {
                    int8_t    aux_q4sb[8];
                    const int offset = sb * 24 + i * 12;
                    decode_q_Kx8_6bit_scales(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                    q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                }

                int8x16_t q8_qs[64 / 16];
                for (int i = 0; i < 64 / 16; i++) {
                    q8_qs[i] = vld1q_s8(q8_ptr[b].qs + sb * 64 + i * 16);
                }

                for (int c = 0; c < col_groups; c++) {
                    uint8x16_t q4_cols[8];
                    for (int i = 0; i < 8; i++) {
                        q4_cols[i] = vld1q_u8(q4_ptr[b].qs + sb * QK_K + i * 32 + 16 * c);
                    }

                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)), q8_qs[0], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)), q8_qs[0], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)), q8_qs[0], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)), q8_qs[0], 3);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)), q8_qs[1], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)), q8_qs[1], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)), q8_qs[1], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)), q8_qs[1], 3);

                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)), q8_qs[2], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)), q8_qs[2], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)), q8_qs[2], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)), q8_qs[2], 3);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)), q8_qs[3], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)), q8_qs[3], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)), q8_qs[3], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)), q8_qs[3], 3);
                }

                // Scales
                // row c0123 blk0 and blk1
                const int16x4_t   sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                const int16x4_t   sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                const float32x4_t sumf_0123  = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[0]),
                                                                       vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[0])));
                acc_f32[0]                   = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);
                // row c4567 blk0 and blk1
                const int16x4_t   sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                const int16x4_t   sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                const float32x4_t sumf_4567  = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[1]),
                                                                       vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[1])));
                acc_f32[1]                   = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                // Bias Correction
                const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb + 0]);
                const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
            }  // for sb

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
        }  // for b

        int base = x * ncols_interleaved;
        vst1q_f32(s + base, acc_f32[0]);
        vst1q_f32(s + base + 4, acc_f32[1]);
    }  // for x
    return;
#endif  // #if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x4_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    q8_k_blocklen = 4;
    constexpr int    acc_size  = 2 * 4;  // 2 row pairs × 4 col pairs
    const uint8x16_t m4b       = vdupq_n_u8(0x0f);

    // 8 accumulators: 2 row pairs × 4 col pairs
    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q4_Kx8 * __restrict q4_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                // d4 0 1 2 3, 4 5 6 7
                float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                // d8 0 1 2 3
                float32x4_t q8_d_0123    = vld1q_f32(q8_ptr[b].d);
                // mins
                float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                // Precomputation of scales and mins
                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];
                float32x4_t sbd_min_0123[q8_k_blocklen];
                float32x4_t sbd_min_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 0);
                sbd_min_0123[0]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 0);
                sbd_min_4567[0]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 0);

                sbd_scale_0123[1] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 1);
                sbd_min_0123[1]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 1);
                sbd_min_4567[1]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 1);

                sbd_scale_0123[2] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 2);
                sbd_min_0123[2]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 2);
                sbd_min_4567[2]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 2);

                sbd_scale_0123[3] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 3);
                sbd_min_0123[3]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 3);
                sbd_min_4567[3]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 3);

                // Precomputation of bsums, each vpaddq calcs all the bsums for each row
                const int16x8_t bsums[q8_k_blocklen] = {
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[QK_K / 64][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                // interleaved bias_acc: [0]->r0 0123, [1]->r1 0123, .., [4]->r0 4567, [5]->r1 4567 ..
                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    // Int accumulators for qs vecdot (4 row x 2 col quartets)
                    int32x4_t acc_lo[acc_size];
                    int32x4_t acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    // Need scales for the low and high nibbles
                    // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                    int16x8_t q4sb_scales[2];
                    int16x8_t q4sb_mins[2];
                    for (int i = 0; i < 2; i++) {
                        int8_t    aux_q4sb[8];
                        const int offset = sb * 24 + i * 12;
                        decode_q_Kx8_6bit_scales(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                        q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                    }

                    constexpr int reads_per_sb = 8;  // 8 * 16 bytes each => 32 qs * 4 rows
                    for (int k = 0; k < reads_per_sb; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k + 128);

                        // 0..3 & 32..35
                        const uint8x16_t q4_0123 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k);
                        const uint8x16_t q4_4567 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k + 16);

                        const int8x16_t q4_0123_lo = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));
                        const int8x16_t q4_0123_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));

                        acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);  //  0..3  r0 c0123
                        acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);  //  0..3  r1 c0123
                        acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);  //  0..3  r2 c0123
                        acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);  //  0..3  r3 c0123

                        acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);  // 32..35 r0 c0123
                        acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);  // 32..35 r1 c0123
                        acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);  // 32..35 r2 c0123
                        acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);  // 32..35 r3 c0123

                        const int8x16_t q4_4567_lo = vreinterpretq_s8_u8(vandq_u8(q4_4567, m4b));
                        const int8x16_t q4_4567_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_4567, 4));

                        acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q4_4567_lo, q8_blk0, 0);  //  0..3  r0 c4567
                        acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q4_4567_lo, q8_blk0, 1);  //  0..3  r1 c4567
                        acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q4_4567_lo, q8_blk0, 2);  //  0..3  r2 c4567
                        acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q4_4567_lo, q8_blk0, 3);  //  0..3  r3 c4567

                        acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q4_4567_hi, q8_blk1, 0);  // 32..35 r0 c4567
                        acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q4_4567_hi, q8_blk1, 1);  // 32..35 r1 c4567
                        acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q4_4567_hi, q8_blk1, 2);  // 32..35 r2 c4567
                        acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q4_4567_hi, q8_blk1, 3);  // 32..35 r3 c4567
                    }

                    // Scale and bias application
                    // acc is stored interleaved to match output layout
                    const int16x4_t sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                    const int16x4_t sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                    const int16x4_t sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                    const int16x4_t sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                    for (int row = 0; row < q8_k_blocklen; row++) {
                        // Bias correction
                        // row c0123 blk0 and blk1
                        const float32x4_t sumf_0123 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[row]),
                                                    vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[row])));
                        acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        // row c4567 blk0 and blk1
                        const float32x4_t sumf_4567 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[row + 4]),
                                                    vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[row + 4])));
                        acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        // Bias
                        const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][row * 2]);
                        const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);

                        // row c0123 blk0 and blk1
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));

                        // row c4567 blk0 and blk1
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
                    }
                }  // for sb

                for (int row = 0; row < q8_k_blocklen; row++) {
                    acc_f32[2 * row] = vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] =
                        vmlsq_f32(acc_f32[2 * row + 1], vcvtq_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * ncols_interleaved + j * 4;
                    int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// ── T1: pre-decoded-scale GEMM (stock tile + separate scale section) ──
// Same reduction order and accumulator layout as the stock kernel; the
// 6-bit group unpack is replaced by direct f16/byte scale loads from the
// ss region (structure comment above). Scale products stay < 2^24 and pair
// sums < 2^23, so the f32 apply chain is bit-identical to the stock
// vmulq_s32 chain.

void ggml_gemm_q4_K_8x4_q8_K_s(int n, float * __restrict s, size_t bs, const void * __restrict vx,
                               const void * __restrict ss, const void * __restrict vy, int nr, int nc,
                               const float * in_acc) {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int q8_k_blocklen = 4;
    constexpr int acc_size      = 2 * 4;
    const uint8x16_t m4b        = vdupq_n_u8(0x0f);

    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict    q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8 * __restrict  q4_ptr = (const tile_block_q4_Kx8 *) vx + ((size_t)x * nb);
            const tile_ss_q4_Kx8s * __restrict    ss_ptr = (const tile_ss_q4_Kx8s *) ss + ((size_t)x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                float32x4_t q8_d_0123    = vld1q_f32(q8_ptr[b].d);
                float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];
                float32x4_t sbd_min_0123[q8_k_blocklen];
                float32x4_t sbd_min_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 0);
                sbd_min_0123[0]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 0);
                sbd_min_4567[0]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 0);

                sbd_scale_0123[1] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 1);
                sbd_min_0123[1]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 1);
                sbd_min_4567[1]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 1);

                sbd_scale_0123[2] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 2);
                sbd_min_0123[2]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 2);
                sbd_min_4567[2]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 2);

                sbd_scale_0123[3] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 3);
                sbd_min_0123[3]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 3);
                sbd_min_4567[3]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 3);

                const int16x8_t bsums[q8_k_blocklen] = {
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[QK_K / 64][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int32x4_t acc_lo[acc_size];
                    int32x4_t acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    const __fp16 * sbp = ss_ptr[b].sb_scale + (size_t)sb * 16;
                    const float32x4_t sc_h0_0123 = vcvt_f32_f16(vld1_f16(sbp));
                    const float32x4_t sc_h0_4567 = vcvt_f32_f16(vld1_f16(sbp + 4));
                    const float32x4_t sc_h1_0123 = vcvt_f32_f16(vld1_f16(sbp + 8));
                    const float32x4_t sc_h1_4567 = vcvt_f32_f16(vld1_f16(sbp + 12));
                    const int16x8_t   mins_h0    = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ss_ptr[b].sb_min + (size_t)sb * 16)));
                    const int16x8_t   mins_h1    = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ss_ptr[b].sb_min + (size_t)sb * 16 + 8)));

                    constexpr int reads_per_sb = 8;
                    for (int k = 0; k < reads_per_sb; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k + 128);

                        const uint8x16_t q4_0123 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k);
                        const uint8x16_t q4_4567 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k + 16);

                        const int8x16_t q4_0123_lo = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));
                        const int8x16_t q4_0123_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));

                        acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);
                        acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);
                        acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);
                        acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);

                        acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);
                        acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);
                        acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);
                        acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);

                        const int8x16_t q4_4567_lo = vreinterpretq_s8_u8(vandq_u8(q4_4567, m4b));
                        const int8x16_t q4_4567_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_4567, 4));

                        acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q4_4567_lo, q8_blk0, 0);
                        acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q4_4567_lo, q8_blk0, 1);
                        acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q4_4567_lo, q8_blk0, 2);
                        acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q4_4567_lo, q8_blk0, 3);

                        acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q4_4567_hi, q8_blk1, 0);
                        acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q4_4567_hi, q8_blk1, 1);
                        acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q4_4567_hi, q8_blk1, 2);
                        acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q4_4567_hi, q8_blk1, 3);
                    }

                    for (int row = 0; row < q8_k_blocklen; row++) {
                        const float32x4_t sumf_0123 =
                            vaddq_f32(vmulq_f32(sc_h0_0123, vcvtq_f32_s32(acc_lo[row])),
                                      vmulq_f32(sc_h1_0123, vcvtq_f32_s32(acc_hi[row])));
                        acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        const float32x4_t sumf_4567 =
                            vaddq_f32(vmulq_f32(sc_h0_4567, vcvtq_f32_s32(acc_lo[row + 4])),
                                      vmulq_f32(sc_h1_4567, vcvtq_f32_s32(acc_hi[row + 4])));
                        acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][row * 2]);
                        const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);

                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_lo, vget_low_s16(mins_h0));
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_hi, vget_low_s16(mins_h1));

                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, vget_high_s16(mins_h0));
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, vget_high_s16(mins_h1));
                    }
                }  // for sb

                for (int row = 0; row < q8_k_blocklen; row++) {
                    acc_f32[2 * row] = vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] =
                        vmlsq_f32(acc_f32[2 * row + 1], vcvtq_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * 8 + j * 4;
                    int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// Decoded-scale sidecar for the q4_K 8-row tile blocks, filled at repack
// time from the same 12-byte groups the hot kernels decode per call today.
// sb_scales holds the 6-bit sub-block scales as f32 (one value per row per
// 32-element half); sb_mins keeps the decoded int16 mins so the int bias-
// accumulation path is untouched. With these, the hot loop only needs f32
// scale loads — no byte-shuffle unpack, no int16 widening.
typedef struct tile_sc_q4_Kx8 {
    float   sb_scales[QK_K / 64 * 2][8];
    int16_t sb_mins[QK_K / 64 * 2][8];
} tile_sc_q4_Kx8;  // 8 (sb,half) groups x (32 + 16) bytes = 384 bytes

// Scale-product bound: vdot lanes sum <= 8 vdots x 4 pairs x 15 x 127 per
// 16-element group (q4 nibbles are unsigned 0..15), scale <= 63 => every
// int32 product < 2^24 and its f32 twin is exact, so the f32 scale loads
// below reproduce the int16-widen + vmulq_s32 chain bit-for-bit.
void ggml_gemv_q4_K_8x4_q8_K_sc(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const void * __restrict vsc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nc % 8 == 0);
    UNUSED(nb);
    UNUSED(bs);
    UNUSED(nr);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    col_groups = 2;  // rows 0123 and 4567
    const uint8x16_t m4b        = vdupq_n_u8(0x0f);

    float32x4_t acc_f32[col_groups];
    const tile_block_q8_K * __restrict q8_ptr = (const tile_block_q8_K *) vy;

    for (int x = 0; x < nc / 8; x++) {
        const tile_block_q4_Kx8 * __restrict q4_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);
        const tile_sc_q4_Kx8   * __restrict sc_ptr  = (const tile_sc_q4_Kx8 *) vsc   + (x * nb);

        for (int i = 0; i < col_groups; i++) {
            acc_f32[i] = vdupq_n_f32(0);
        }

        for (int b = 0; b < nb; b++) {
            float32x4_t q4_d_0        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
            float32x4_t q4_d_1        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
            float32x4_t q8_d          = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0123 = vmulq_f32(q4_d_0, q8_d);
            float32x4_t sb_scale_4567 = vmulq_f32(q4_d_1, q8_d);
            float32x4_t q4_dmin_0     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
            float32x4_t q4_dmin_1     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));
            float32x4_t sb_min_0123   = vmulq_f32(q4_dmin_0, q8_d);
            float32x4_t sb_min_4567   = vmulq_f32(q4_dmin_1, q8_d);

            int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
            int32x4_t acc_lo[col_groups];
            int32x4_t acc_hi[col_groups];

            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int16_t         bsums_arr[8];
            vst1q_s16(bsums_arr, bsums);
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int i = 0; i < col_groups; i++) {
                    acc_lo[i] = vdupq_n_s32(0);
                    acc_hi[i] = vdupq_n_s32(0);
                }
                const float32x4_t sc_h0_0123 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 0]);
                const float32x4_t sc_h0_4567 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 0] + 4);
                const float32x4_t sc_h1_0123 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 1]);
                const float32x4_t sc_h1_4567 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 1] + 4);
                const int16x8_t   mins_h0    = vld1q_s16(sc_ptr[b].sb_mins[sb * 2 + 0]);
                const int16x8_t   mins_h1    = vld1q_s16(sc_ptr[b].sb_mins[sb * 2 + 1]);

                int8x16_t q8_qs[64 / 16];
                for (int i = 0; i < 64 / 16; i++) {
                    q8_qs[i] = vld1q_s8(q8_ptr[b].qs + sb * 64 + i * 16);
                }

                for (int c = 0; c < col_groups; c++) {
                    uint8x16_t q4_cols[8];
                    for (int i = 0; i < 8; i++) {
                        q4_cols[i] = vld1q_u8(q4_ptr[b].qs + sb * QK_K + i * 32 + 16 * c);
                    }

                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)), q8_qs[0], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)), q8_qs[0], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)), q8_qs[0], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)), q8_qs[0], 3);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)), q8_qs[1], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)), q8_qs[1], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)), q8_qs[1], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)), q8_qs[1], 3);

                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)), q8_qs[2], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)), q8_qs[2], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)), q8_qs[2], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)), q8_qs[2], 3);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)), q8_qs[3], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)), q8_qs[3], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)), q8_qs[3], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)), q8_qs[3], 3);
                }

                const float32x4_t sumf_0123 = vaddq_f32(vmulq_f32(sc_h0_0123, vcvtq_f32_s32(acc_lo[0])),
                                                        vmulq_f32(sc_h1_0123, vcvtq_f32_s32(acc_hi[0])));
                acc_f32[0]                  = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);
                const float32x4_t sumf_4567 = vaddq_f32(vmulq_f32(sc_h0_4567, vcvtq_f32_s32(acc_lo[1])),
                                                        vmulq_f32(sc_h1_4567, vcvtq_f32_s32(acc_hi[1])));
                acc_f32[1]                  = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb + 0]);
                const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(mins_h0));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(mins_h1));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(mins_h0));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(mins_h1));
            }  // for sb

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
        }  // for b

        int base = x * 8;
        vst1q_f32(s + base, acc_f32[0]);
        vst1q_f32(s + base + 4, acc_f32[1]);
    }  // for x
    return;
#endif
    ggml_gemv_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x4_q8_K_sc(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const void * __restrict vsc, const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    q8_k_blocklen = 4;
    constexpr int    acc_size      = 2 * 4;
    const uint8x16_t m4b           = vdupq_n_u8(0x0f);

    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8 * __restrict q4_ptr = (const tile_block_q4_Kx8 *) vx + (x * nb);
            const tile_sc_q4_Kx8   * __restrict sc_ptr  = (const tile_sc_q4_Kx8 *) vsc   + (x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                float32x4_t q8_d_0123    = vld1q_f32(q8_ptr[b].d);
                float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];
                float32x4_t sbd_min_0123[q8_k_blocklen];
                float32x4_t sbd_min_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 0);
                sbd_min_0123[0]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 0);
                sbd_min_4567[0]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 0);

                sbd_scale_0123[1] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 1);
                sbd_min_0123[1]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 1);
                sbd_min_4567[1]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 1);

                sbd_scale_0123[2] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 2);
                sbd_min_0123[2]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 2);
                sbd_min_4567[2]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 2);

                sbd_scale_0123[3] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 3);
                sbd_min_0123[3]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 3);
                sbd_min_4567[3]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 3);

                const int16x8_t bsums[q8_k_blocklen] = {
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[QK_K / 64][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int32x4_t acc_lo[acc_size];
                    int32x4_t acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    const float32x4_t sc_h0_0123 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 0]);
                    const float32x4_t sc_h0_4567 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 0] + 4);
                    const float32x4_t sc_h1_0123 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 1]);
                    const float32x4_t sc_h1_4567 = vld1q_f32(sc_ptr[b].sb_scales[sb * 2 + 1] + 4);
                    const int16x8_t   mins_h0    = vld1q_s16(sc_ptr[b].sb_mins[sb * 2 + 0]);
                    const int16x8_t   mins_h1    = vld1q_s16(sc_ptr[b].sb_mins[sb * 2 + 1]);

                    constexpr int reads_per_sb = 8;
                    for (int k = 0; k < reads_per_sb; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k + 128);

                        const uint8x16_t q4_0123 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k);
                        const uint8x16_t q4_4567 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k + 16);

                        const int8x16_t q4_0123_lo = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));
                        const int8x16_t q4_0123_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));

                        acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);
                        acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);
                        acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);
                        acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);

                        acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);
                        acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);
                        acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);
                        acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);

                        const int8x16_t q4_4567_lo = vreinterpretq_s8_u8(vandq_u8(q4_4567, m4b));
                        const int8x16_t q4_4567_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_4567, 4));

                        acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q4_4567_lo, q8_blk0, 0);
                        acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q4_4567_lo, q8_blk0, 1);
                        acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q4_4567_lo, q8_blk0, 2);
                        acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q4_4567_lo, q8_blk0, 3);

                        acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q4_4567_hi, q8_blk1, 0);
                        acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q4_4567_hi, q8_blk1, 1);
                        acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q4_4567_hi, q8_blk1, 2);
                        acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q4_4567_hi, q8_blk1, 3);
                    }

                    for (int row = 0; row < q8_k_blocklen; row++) {
                        const float32x4_t sumf_0123 = vaddq_f32(vmulq_f32(sc_h0_0123, vcvtq_f32_s32(acc_lo[row])),
                                                                vmulq_f32(sc_h1_0123, vcvtq_f32_s32(acc_hi[row])));
                        acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        const float32x4_t sumf_4567 = vaddq_f32(vmulq_f32(sc_h0_4567, vcvtq_f32_s32(acc_lo[row + 4])),
                                                                vmulq_f32(sc_h1_4567, vcvtq_f32_s32(acc_hi[row + 4])));
                        acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][row * 2]);
                        const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);

                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_lo, vget_low_s16(mins_h0));
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_hi, vget_low_s16(mins_h1));

                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, vget_high_s16(mins_h0));
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, vget_high_s16(mins_h1));
                    }
                }  // for sb

                for (int row = 0; row < q8_k_blocklen; row++) {
                    acc_f32[2 * row] = vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] =
                        vmlsq_f32(acc_f32[2 * row + 1], vcvtq_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * 8 + j * 4;
                    int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q6_K_8x4_q8_K(int                        n,
                             float * __restrict      s,
                             size_t                     bs,
                             const void * __restrict vx,
                             const void * __restrict vy,
                             int                        nr,
                             int                        nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    col_groups = ncols_interleaved / 4;
    const uint8x16_t m4b        = vdupq_n_u8(0x0f);
    const uint8x16_t mask_lo    = vdupq_n_u8(0x03);
    const uint8x16_t mask_hi    = vdupq_n_u8(0x30);

    // 1x8 tile = 2 x 4
    float32x4_t acc_f32[2];

    const tile_block_q8_K * __restrict q8_ptr = (const tile_block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const tile_block_q6_Kx8 * __restrict q6_ptr = (const tile_block_q6_Kx8 *) vx + (x * nb);

        for (int i = 0; i < col_groups; i++) {
            acc_f32[i] = vdupq_n_f32(0);
        }

        for (int b = 0; b < nb; b++) {
            float32x4_t q6_d_0     = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d));      // d0 d1 d2 d3
            float32x4_t q6_d_1     = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d + 4));  // d4 d5 d6 d7
            float32x4_t q8_d       = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0 = vmulq_f32(q6_d_0, q8_d);
            float32x4_t sb_scale_1 = vmulq_f32(q6_d_1, q8_d);

            int32x4_t acc[col_groups];
            for (int i = 0; i < col_groups; i++) {
                acc[i] = vdupq_n_s32(0);
            }

            // Load all 16 scales once and widen to int16 (Q6_K has 16 scales per block)
            // Reused for bias and dequantization later
            int16_t q6_scales[16 * 8];
            for (int i = 0; i < 16; i++) {
                int16x8_t scales = vmovl_s8(vld1_s8(q6_ptr[b].scales + i * 8));
                vst1q_s16(q6_scales + i * 8, scales);
            }

            // Compute bias per column using q8 bsums and preloaded scales to skip the -32 shift
            int32x4_t bias_lo = vdupq_n_s32(0);
            int32x4_t bias_hi = vdupq_n_s32(0);

            // Load bsums in chunks of 4 to process with vectorized operations
            for (int i = 0; i < 16; i += 4) {
                int16x4_t bsums_vec   = vld1_s16(q8_ptr[b].bsums + i);
                int16x4_t scales_lo_0 = vld1_s16(q6_scales + (i + 0) * 8);
                int16x4_t scales_hi_0 = vld1_s16(q6_scales + (i + 0) * 8 + 4);
                int16x4_t scales_lo_1 = vld1_s16(q6_scales + (i + 1) * 8);
                int16x4_t scales_hi_1 = vld1_s16(q6_scales + (i + 1) * 8 + 4);
                int16x4_t scales_lo_2 = vld1_s16(q6_scales + (i + 2) * 8);
                int16x4_t scales_hi_2 = vld1_s16(q6_scales + (i + 2) * 8 + 4);
                int16x4_t scales_lo_3 = vld1_s16(q6_scales + (i + 3) * 8);
                int16x4_t scales_hi_3 = vld1_s16(q6_scales + (i + 3) * 8 + 4);

                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_0, bsums_vec, 0);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_0, bsums_vec, 0);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_1, bsums_vec, 1);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_1, bsums_vec, 1);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_2, bsums_vec, 2);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_2, bsums_vec, 2);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_3, bsums_vec, 3);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_3, bsums_vec, 3);
            }
            bias_lo = vshlq_n_s32(bias_lo, 5);
            bias_hi = vshlq_n_s32(bias_hi, 5);

            // Process two 128-value halves per superblock
            for (int half = 0; half < 2; half++) {
                const uint8_t * ql_base = q6_ptr[b].ql + half * 512;
                const uint8_t * qh_base = q6_ptr[b].qh + half * 256;

                // A subblock (sb) is a set of weights that share the scale
                // Since q6_K scales are per 16 elements
                // num sbs -> 256 elements / (16 elements/scale * 2 elements/byte * 2 halves)
                for (int sb = 0; sb < QK_K / 64; sb++) {
                    const int8_t * q8_base_l = q8_ptr[b].qs + half * 128 + sb * 16;
                    const int8_t * q8_base_h = q8_base_l + 64;

                    // Load and duplicate q8 values (each register covers four interleaved columns of q6)
                    int8x16_t q8_l[4];
                    int8x16_t q8_h[4];
                    for (int i = 0; i < 4; i++) {
                        q8_l[i] = (int8x16_t) vld1q_dup_s32((const int32_t *) (q8_base_l + i * 4));
                        q8_h[i] = (int8x16_t) vld1q_dup_s32((const int32_t *) (q8_base_h + i * 4));
                    }

                    const int ql_off_base = sb * QK_K / 2;
                    const int qh_off_base = ql_off_base & 255;  // wraps after 256 bytes

                    // Load 4 vectors at once (64 bytes each for ql_0, ql_1, qh_0, qh_1)
                    uint8x16x4_t q6_ql_0 = vld1q_u8_x4(ql_base + ql_off_base);
                    uint8x16x4_t q6_ql_1 = vld1q_u8_x4(ql_base + ql_off_base + 64);
                    uint8x16x4_t q6_qh_0 = vld1q_u8_x4(qh_base + qh_off_base);
                    uint8x16x4_t q6_qh_1 = vld1q_u8_x4(qh_base + qh_off_base + 64);

                    // Adjust qh for subblocks 2 and 3 (shift right by 2)
                    if (sb > 1) {
                        q6_qh_0.val[0] = vshrq_n_u8(q6_qh_0.val[0], 2);
                        q6_qh_0.val[1] = vshrq_n_u8(q6_qh_0.val[1], 2);
                        q6_qh_0.val[2] = vshrq_n_u8(q6_qh_0.val[2], 2);
                        q6_qh_0.val[3] = vshrq_n_u8(q6_qh_0.val[3], 2);
                        q6_qh_1.val[0] = vshrq_n_u8(q6_qh_1.val[0], 2);
                        q6_qh_1.val[1] = vshrq_n_u8(q6_qh_1.val[1], 2);
                        q6_qh_1.val[2] = vshrq_n_u8(q6_qh_1.val[2], 2);
                        q6_qh_1.val[3] = vshrq_n_u8(q6_qh_1.val[3], 2);
                    }

                    const uint8x16_t q6_ql[8] = { q6_ql_0.val[0], q6_ql_0.val[1], q6_ql_0.val[2], q6_ql_0.val[3],
                                                  q6_ql_1.val[0], q6_ql_1.val[1], q6_ql_1.val[2], q6_ql_1.val[3] };
                    const uint8x16_t q6_qh[8] = { q6_qh_0.val[0], q6_qh_0.val[1], q6_qh_0.val[2], q6_qh_0.val[3],
                                                  q6_qh_1.val[0], q6_qh_1.val[1], q6_qh_1.val[2], q6_qh_1.val[3] };

                    // Process column groups (0-3, 4-7)
                    for (int g = 0; g < col_groups; g++) {
                        int32x4_t sb_acc_l = vdupq_n_s32(0);
                        int32x4_t sb_acc_h = vdupq_n_s32(0);

                        for (int chunk = 0; chunk < 4; chunk++) {
                            const int idx = chunk * 2 + g;

                            const uint8x16_t q6_qs_l = q6_ql[idx];
                            const uint8x16_t q6_qs_h = q6_qh[idx];

                            // Extract high 2 bits for upper nibble reconstruction
                            const uint8x16_t q6_qs_hh = vandq_u8(q6_qs_h, mask_hi);

                            // q6 = (low4 | high2<<4), without -32 bias (handled via bsums)
                            const int8x16_t q6_l =
                                vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(q6_qs_l, m4b), vandq_u8(q6_qs_h, mask_lo), 4));
                            const int8x16_t q6_h = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_qs_l, 4), q6_qs_hh));

                            sb_acc_l = vdotq_s32(sb_acc_l, q6_l, q8_l[chunk]);
                            sb_acc_h = vdotq_s32(sb_acc_h, q6_h, q8_h[chunk]);
                        }

                        const int scale_idx_l = half * 8 + sb;
                        const int scale_idx_h = half * 8 + sb + 4;

                        const int32x4_t scale_vec_l = vmovl_s16(vld1_s16(q6_scales + scale_idx_l * 8 + g * 4));
                        const int32x4_t scale_vec_h = vmovl_s16(vld1_s16(q6_scales + scale_idx_h * 8 + g * 4));

                        acc[g] = vmlaq_s32(acc[g], sb_acc_l, scale_vec_l);
                        acc[g] = vmlaq_s32(acc[g], sb_acc_h, scale_vec_h);
                    }
                }
            }  // for half

            // Bias correction
            acc[0] = vsubq_s32(acc[0], bias_lo);
            acc[1] = vsubq_s32(acc[1], bias_hi);

            // Apply superblock scale (no mins for q6_K)
            // acc[g] has [c0, c1, c2, c3]
            float32x4_t w_0123 = vmulq_f32(vcvtq_f32_s32(acc[0]), sb_scale_0);
            float32x4_t w_4567 = vmulq_f32(vcvtq_f32_s32(acc[1]), sb_scale_1);

            acc_f32[0] = vaddq_f32(acc_f32[0], w_0123);
            acc_f32[1] = vaddq_f32(acc_f32[1], w_4567);
        }  // for b

        int base = x * ncols_interleaved;
        vst1q_f32(s + base, acc_f32[0]);
        vst1q_f32(s + base + 4, acc_f32[1]);
    }  // for x
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q6_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x4_q8_K(int                        n,
                             float * __restrict      s,
                             size_t                     bs,
                             const void * __restrict vx,
                             const void * __restrict vy,
                             int                        nr,
                             int                        nc,
                             const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    q8_k_blocklen = 4;
    constexpr int    col_groups    = ncols_interleaved / 4;
    constexpr int    acc_size      = q8_k_blocklen * col_groups;  // 4 rows, 2 column groups
    const uint8x16_t m4b           = vdupq_n_u8(0x0f);
    const uint8x16_t mask_lo       = vdupq_n_u8(0x03);
    const uint8x16_t mask_hi       = vdupq_n_u8(0x30);
    const int8x16_t  m32s          = vdupq_n_s8(32);

    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const tile_block_q6_Kx8 * __restrict q6_ptr = (const tile_block_q6_Kx8 *) vx + (x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                float32x4_t q6_d_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d));
                float32x4_t q6_d_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q6_ptr[b].d + 4));
                float32x4_t q8_d_0123 = vld1q_f32(q8_ptr[b].d);

                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q6_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q6_d_4567, q8_d_0123, 0);
                sbd_scale_0123[1] = vmulq_laneq_f32(q6_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q6_d_4567, q8_d_0123, 1);
                sbd_scale_0123[2] = vmulq_laneq_f32(q6_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q6_d_4567, q8_d_0123, 2);
                sbd_scale_0123[3] = vmulq_laneq_f32(q6_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q6_d_4567, q8_d_0123, 3);

                int32x4_t acc_s32[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    acc_s32[i] = vdupq_n_s32(0);
                }

                int16_t q6_scales[8 * 16];
                for (int i = 0; i < 16; i++) {
                    int16x8_t scales = vmovl_s8(vld1_s8(q6_ptr[b].scales + i * 8));
                    vst1q_s16(q6_scales + i * 8, scales);
                }

                for (int half = 0; half < 2; half++) {
                    const uint8_t * ql_base = q6_ptr[b].ql + half * 512;
                    const uint8_t * qh_base = q6_ptr[b].qh + half * 256;

                    for (int sb = 0; sb < QK_K / 64; sb++) {
                        int32x4_t acc_lo[acc_size];
                        int32x4_t acc_hi[acc_size];
                        for (int i = 0; i < acc_size; i++) {
                            acc_lo[i] = vdupq_n_s32(0);
                            acc_hi[i] = vdupq_n_s32(0);
                        }

                        const int8_t * q8_base_l = q8_ptr[b].qs + half * 512 + sb * 64;
                        const int8_t * q8_base_h = q8_ptr[b].qs + half * 512 + 256 + sb * 64;

                        // 4 rows * 16 elements per scale
                        // 4 reads of 16 bytes each
                        constexpr int reads_per_sb = 4;
                        int8x16_t     q8_l[reads_per_sb];
                        int8x16_t     q8_h[reads_per_sb];
                        for (int k = 0; k < reads_per_sb; k++) {
                            q8_l[k] = vld1q_s8(q8_base_l + 16 * k);
                            q8_h[k] = vld1q_s8(q8_base_h + 16 * k);
                        }

                        const int ql_off_base = sb * QK_K / 2;
                        const int qh_off_base = ql_off_base & 255;

                        uint8x16_t q6_ql_0123[reads_per_sb];
                        uint8x16_t q6_ql_4567[reads_per_sb];
                        uint8x16_t q6_qh_0123[reads_per_sb];
                        uint8x16_t q6_qh_4567[reads_per_sb];

                        for (int k = 0; k < reads_per_sb; k++) {
                            q6_ql_0123[k] = vld1q_u8(ql_base + ql_off_base + k * 32);
                            q6_ql_4567[k] = vld1q_u8(ql_base + ql_off_base + k * 32 + 16);
                            q6_qh_0123[k] = vld1q_u8(qh_base + qh_off_base + k * 32);
                            q6_qh_4567[k] = vld1q_u8(qh_base + qh_off_base + k * 32 + 16);
                        }

                        if (sb > 1) {
                            for (int k = 0; k < reads_per_sb; k++) {
                                q6_qh_0123[k] = vshrq_n_u8(q6_qh_0123[k], 2);
                                q6_qh_4567[k] = vshrq_n_u8(q6_qh_4567[k], 2);
                            }
                        }

                        for (int k = 0; k < reads_per_sb; k++) {
                            // q = (ql | qh) - 32
                            const uint8x16_t hbit_lo_0123 = vandq_u8(q6_qh_0123[k], mask_lo);
                            const uint8x16_t hbit_hi_0123 = vandq_u8(q6_qh_0123[k], mask_hi);
                            const uint8x16_t hbit_lo_4567 = vandq_u8(q6_qh_4567[k], mask_lo);
                            const uint8x16_t hbit_hi_4567 = vandq_u8(q6_qh_4567[k], mask_hi);

                            const int8x16_t q6_0123_lo = vsubq_s8(
                                vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(q6_ql_0123[k], m4b), hbit_lo_0123, 4)), m32s);
                            const int8x16_t q6_0123_hi = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_0123[k], 4), hbit_hi_0123)), m32s);

                            acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q6_0123_lo, q8_l[k], 0);  //  0..3  r0 c0123
                            acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q6_0123_lo, q8_l[k], 1);  //  0..3  r1 c0123
                            acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q6_0123_lo, q8_l[k], 2);  //  0..3  r2 c0123
                            acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q6_0123_lo, q8_l[k], 3);  //  0..3  r3 c0123

                            acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q6_0123_hi, q8_h[k], 0);  // 64..67 r0 c0123
                            acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q6_0123_hi, q8_h[k], 1);  // 64..67 r1 c0123
                            acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q6_0123_hi, q8_h[k], 2);  // 64..67 r2 c0123
                            acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q6_0123_hi, q8_h[k], 3);  // 64..67 r3 c0123

                            const int8x16_t q6_4567_lo = vsubq_s8(
                                vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(q6_ql_4567[k], m4b), hbit_lo_4567, 4)), m32s);
                            const int8x16_t q6_4567_hi = vsubq_s8(
                                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_4567[k], 4), hbit_hi_4567)), m32s);

                            acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q6_4567_lo, q8_l[k], 0);  //  0..3  r0 c4567
                            acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q6_4567_lo, q8_l[k], 1);  //  0..3  r1 c4567
                            acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q6_4567_lo, q8_l[k], 2);  //  0..3  r2 c4567
                            acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q6_4567_lo, q8_l[k], 3);  //  0..3  r3 c4567

                            acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q6_4567_hi, q8_h[k], 0);  // 64..67 r0 c4567
                            acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q6_4567_hi, q8_h[k], 1);  // 64..67 r1 c4567
                            acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q6_4567_hi, q8_h[k], 2);  // 64..67 r2 c4567
                            acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q6_4567_hi, q8_h[k], 3);  // 64..67 r3 c4567
                        }

                        // Scale and bias
                        const int scale_idx_l = half * 8 + sb;
                        const int scale_idx_h = half * 8 + sb + 4;

                        for (int g = 0; g < col_groups; g++) {
                            const int16x4_t scales_l16  = vld1_s16(q6_scales + scale_idx_l * 8 + g * 4);
                            const int16x4_t scales_h16  = vld1_s16(q6_scales + scale_idx_h * 8 + g * 4);
                            const int32x4_t scale_vec_l = vmovl_s16(scales_l16);
                            const int32x4_t scale_vec_h = vmovl_s16(scales_h16);
                            const int       acc_offset  = g * q8_k_blocklen;

                            for (int row = 0; row < q8_k_blocklen; row++) {
                                const int idx = row * 2 + g;
                                acc_s32[idx]  = vmlaq_s32(acc_s32[idx], acc_lo[acc_offset + row], scale_vec_l);
                                acc_s32[idx]  = vmlaq_s32(acc_s32[idx], acc_hi[acc_offset + row], scale_vec_h);
                            }
                        }
                    }
                }

                // Finally we apply the superblock scales
                for (int row = 0; row < q8_k_blocklen; row++) {
                    const int       idx0     = 2 * row;
                    const int       idx1     = 2 * row + 1;
                    const int32x4_t acc_0123 = acc_s32[idx0];
                    const int32x4_t acc_4567 = acc_s32[idx1];

                    acc_f32[idx0] = vmlaq_f32(acc_f32[idx0], vcvtq_f32_s32(acc_0123), sbd_scale_0123[row]);
                    acc_f32[idx1] = vmlaq_f32(acc_f32[idx1], vcvtq_f32_s32(acc_4567), sbd_scale_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * ncols_interleaved + j * 4;
                    int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemm_q6_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// ══════════ Nibble-plane q4_K prefill GEMM ══════════
// The packed q4_K 8x4 GEMM performs 4 aux NEON ops per k-iter (two vand + two
// vshr nibble unpacks) that steal vdot-pipe slots on Cortex-A76: measured
// 121 GMAC/s packed-q4_K vs 185 q8_0. At warm time we expand both nibbles to
// one int8 byte each (two planes, qs_lo/qs_hi) so the hot loop is pure vdot.
// The int math is unchanged — each plane byte holds exactly the nibble the
// packed kernel materialized with vand/vshr, element order preserved — so
// results are bit-identical. GEMV stays on the packed tile: decode is
// DRAM-bound and the plane tile is +89% bytes (2176 vs 1152 per 8-row block).

struct tile_block_q4_Kx8p {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[96];
    uint8_t  qs_lo[1024];  // low nibbles: per (sb,k) 32B = rows 0..3 then 4..7
    uint8_t  qs_hi[1024];  // high nibbles, same (sb,k) layout
};

static inline int sofuu_repack_q4k_8bl_planes_range(uint8_t * dst, const uint8_t * src, int64_t nblocks, int64_t r8_begin, int64_t r8_end, unsigned interleave = 4) {
    tile_block_q4_Kx8p * d = (tile_block_q4_Kx8p *) dst + r8_begin * nblocks;
    const tile_block_q4_K * s = (const tile_block_q4_K *) src + r8_begin * 8 * nblocks;
    tile_block_q4_K tmp[8];
    for (int64_t b = r8_begin; b < r8_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            tile_block_q4_Kx8 blk = make_block_q4_Kx8(tmp, interleave);
            memcpy(d->d, blk.d, sizeof(d->d));
            memcpy(d->dmin, blk.dmin, sizeof(d->dmin));
            memcpy(d->scales, blk.scales, sizeof(d->scales));
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int k = 0; k < 8; k++) {
                    const uint8_t * p = blk.qs + sb * QK_K + 32 * k;
                    uint8_t * lo = d->qs_lo + (sb * 8 + k) * 32;
                    uint8_t * hi = d->qs_hi + (sb * 8 + k) * 32;
                    for (int j = 0; j < 16; j++) {
                        lo[j]      = (uint8_t)(p[j] & 0x0F);        // rows 0..3 lo
                        hi[j]      = (uint8_t)(p[j] >> 4);          // rows 0..3 hi
                        lo[j + 16] = (uint8_t)(p[16 + j] & 0x0F);   // rows 4..7 lo
                        hi[j + 16] = (uint8_t)(p[16 + j] >> 4);     // rows 4..7 hi
                    }
                }
            }
            d++;
        }
        s += 8 * nblocks;
    }
    return 0;
}

static inline int sofuu_repack_q4k_8bl_planes(uint8_t * dst, const uint8_t * src, int64_t nrow, int64_t nblocks, unsigned interleave = 4) {
    return sofuu_repack_q4k_8bl_planes_range(dst, src, nblocks, 0, nrow / 8, interleave);
}

void ggml_gemm_q4_K_8x4_q8_K_p(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    UNUSED(nb);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int q8_k_blocklen = 4;
    constexpr int acc_size  = 2 * 4;  // 2 row pairs × 4 col pairs

    // 8 accumulators: 2 row pairs × 4 col pairs
    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8p * __restrict q4_ptr = (const tile_block_q4_Kx8p *) vx + (x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                float32x4_t q8_d_0123    = vld1q_f32(q8_ptr[b].d);
                float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                // Precomputation of scales and mins
                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];
                float32x4_t sbd_min_0123[q8_k_blocklen];
                float32x4_t sbd_min_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 0);
                sbd_min_0123[0]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 0);
                sbd_min_4567[0]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 0);

                sbd_scale_0123[1] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 1);
                sbd_min_0123[1]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 1);
                sbd_min_4567[1]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 1);

                sbd_scale_0123[2] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 2);
                sbd_min_0123[2]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 2);
                sbd_min_4567[2]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 2);

                sbd_scale_0123[3] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 3);
                sbd_min_0123[3]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 3);
                sbd_min_4567[3]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 3);

                // Precomputation of bsums, each vpaddq calcs all the bsums for each row
                const int16x8_t bsums[q8_k_blocklen] = {
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[QK_K / 64][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                // interleaved bias_acc: [0]->r0 0123, [1]->r1 0123, .., [4]->r0 4567, [5]->r1 4567 ..
                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    // Int accumulators for qs vecdot (4 row x 2 col quartets)
                    int32x4_t acc_lo[acc_size];
                    int32x4_t acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    // Need scales for the low and high nibbles
                    // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                    int16x8_t q4sb_mins[2];
                    int16x8_t q4sb_scales[2];
                    for (int i = 0; i < 2; i++) {
                        int8_t    aux_q4sb[8];
                        const int offset = sb * 24 + i * 12;
                        decode_q_Kx8_6bit_scales(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                        q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                    }

                    constexpr int reads_per_sb = 8;
                    for (int k = 0; k < reads_per_sb; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k + 128);

                        // Pre-expanded nibble planes; identical int8 vectors to
                        // the packed kernel's vand/vshr outputs.
                        const uint32_t poff = (uint32_t)((sb * 8 + k) * 32);
                        const int8x16_t q4_0123_lo = vld1q_s8((const int8_t *)(q4_ptr[b].qs_lo + poff));
                        const int8x16_t q4_0123_hi = vld1q_s8((const int8_t *)(q4_ptr[b].qs_hi + poff));
                        const int8x16_t q4_4567_lo = vld1q_s8((const int8_t *)(q4_ptr[b].qs_lo + poff + 16));
                        const int8x16_t q4_4567_hi = vld1q_s8((const int8_t *)(q4_ptr[b].qs_hi + poff + 16));

                        acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);
                        acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);
                        acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);
                        acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);

                        acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);
                        acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);
                        acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);
                        acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);

                        acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q4_4567_lo, q8_blk0, 0);
                        acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q4_4567_lo, q8_blk0, 1);
                        acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q4_4567_lo, q8_blk0, 2);
                        acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q4_4567_lo, q8_blk0, 3);

                        acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q4_4567_hi, q8_blk1, 0);
                        acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q4_4567_hi, q8_blk1, 1);
                        acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q4_4567_hi, q8_blk1, 2);
                        acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q4_4567_hi, q8_blk1, 3);
                    }

                    // Scale and bias application — identical to the packed kernel
                    const int16x4_t sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                    const int16x4_t sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                    const int16x4_t sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                    const int16x4_t sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                    for (int row = 0; row < q8_k_blocklen; row++) {
                        const float32x4_t sumf_0123 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[row]),
                                                    vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[row])));
                        acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        const float32x4_t sumf_4567 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[row + 4]),
                                                    vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[row + 4])));
                        acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][row * 2]);
                        const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);

                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));

                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
                    }
                }  // for sb

                for (int row = 0; row < q8_k_blocklen; row++) {
                    acc_f32[2 * row] = vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] =
                        vmlsq_f32(acc_f32[2 * row + 1], vcvtq_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * 8 + j * 4;
                    int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// ══════════ Apply-phase q4_K 8x4 GEMM (SOFUU_Q4K_APPLY2) ══════════
// Same int8 operands, same int32 dot order and the same per-output arithmetic
// as the nibble-plane kernel ggml_gemm_q4_K_8x4_q8_K_p, with the scale/bias
// apply's instruction count cut down. It is bit-identical to it.
//
// What was removed (arm64 instruction census of the shipped build, per 64-k
// sub-block of the 8x4 tile, i.e. per 128 sdot: packed 311, plane (_p) 303,
// this kernel 244):
//   * 36 instructions of per-(sub-block, half) 6-bit scale/min decode (two
//     packed-word loads, ~16 and/orr bit extracts, 8 fmov/mov.s lane builds,
//     6 sshll widenings). Those values are fixed per weight row and block, so
//     they are decoded once at repack time into int32 sub-block scales and
//     int16 mins, already in the 8-weight-row lane order the apply wants
//     (2 ldr + 2 ldr per sub-block, no bit twiddling).
//   * the load address arithmetic: the two nibble planes are re-laid so a
//     (sub-block, k-group, weight-row half) pair is one 32 B chunk (lo plane
//     in [0..15], hi in [16..31]) instead of two 16 B loads 1024 B apart, and
//     the bsums come from the q8 registers by lane instead of a stack array
//     read back with ld1r.
//   * 8 registers of sbd_scale/sbd_min hoisting: they are rebuilt per
//     sub-block, which keeps the 16 accumulators + 8 f32 accumulators in
//     registers instead of spilling ~16 vectors per sub-block.
//
// Byte layout per 8-weight-row block: 16+16 d/dmin (f16, as in _p) + 256
// int32 scales + 128 int16 mins + 2048 nibble planes = 2464 B, versus 2176
// for the plane tile and 1344 for the default packed + T1-sidecar stream.
//
// Measured (M2 Pro, 4 pooled threads, sofuu_bench_gemm_plane = interleaved
// variants in one run, batch 128, 21 iters, min GMAC/s):
//     shape              packed   T1(_s)  planes  apply2
//     1024x3072           435.8    416.8   494.8   551.1
//     3072x1024           432.9    413.5   490.3   544.0
//     1024x1024           431.7    412.5   488.8   545.4
// i.e. +32% over the default T1 path and +11% over the plane tile, and the
// outputs match the packed kernel float for float (0/393216 differ).
//
// End to end on a 1454-token prefill of a 0.6B Q4_K_M model this is much
// smaller (interleaved, medians of the summed per-chunk gemm time: 1651 ms
// default, 1590 ms plane tile, 1543 ms apply2 = -6.5%), because streaming the
// bigger tile dominates: the kernel only reaches ~60 GB/s of the ~118 GB/s
// this machine streams with 4 threads, so the extra bytes of a 2464 B block
// eat most of the instruction win. Shrinking the block (packed nibbles +
// int8 scales, ~1440 B) or doubling the activation tile is the next lever,
// not more instruction cuts.
//
// Nothing about the int8 operands or the int32 dot order changes, and the
// per-output arithmetic keeps the _p kernel's exact sequence
//   sumf = f32( sc_lo*acc_lo + sc_hi*acc_hi )        (int32, exact)
//   acc  = vfma(acc, d*q8_d, sumf)
//   acc  = vmls(acc, f32(sum(b_min*bsums)), dmin*q8_d)
// so the output is bit-identical (int32 mul+mla is exact, hence equal to the
// _p kernel's mul+mul+add; verified live by sofuu_bench_gemm_plane and by the
// end-to-end logits gate).
struct tile_block_q4_Kx8q {
    uint16_t d[8];
    uint16_t dmin[8];
    int32_t  sc[8][8];         // [(sb*2+half)][row]: 6-bit scale, decoded
    int16_t  mn[8][8];         // [(sb*2+half)][row]: 6-bit min, decoded
    uint8_t  pl[4][8][2][32];  // [sb][k-group][weight-row half][0..15]=lo, [16..31]=hi
};

// Build the apply-phase tile from the stock packed 8-row tile. The scale/min
// values come from the same decode_q_Kx8_6bit_scales the hot kernels call, so
// they are bit-equal to what those kernels would compute; the plane bytes are
// the same nibbles the _p repack writes (identical int8 operands).
static inline void sofuu_fill_q4k_apply2_block(tile_block_q4_Kx8q * __restrict out,
                                               const tile_block_q4_Kx8 * __restrict in) {
    memcpy(out->d, in->d, sizeof(out->d));
    memcpy(out->dmin, in->dmin, sizeof(out->dmin));
    for (int g = 0; g < (QK_K / 64) * 2; g++) {          // g = sb*2 + half
        int8_t    aux[8];
        int16x8_t mins;
        decode_q_Kx8_6bit_scales(&in->scales[g * 12], &mins, aux);
        for (int r = 0; r < 8; r++) out->sc[g][r] = (int32_t) aux[r];
        vst1q_s16(out->mn[g], mins);
    }
    for (int sb = 0; sb < QK_K / 64; sb++) {
        for (int kg = 0; kg < 8; kg++) {
            // 32 B hold rows 0..3 then rows 4..7, four k values per row; byte
            // j carries both nibbles, so lo/hi come from the same byte.
            const uint8_t * q = in->qs + sb * QK_K + 32 * kg;
            for (int hw = 0; hw < 2; hw++) {
                for (int j = 0; j < 16; j++) {
                    out->pl[sb][kg][hw][j]      = (uint8_t) (q[16 * hw + j] & 0x0F);
                    out->pl[sb][kg][hw][j + 16] = (uint8_t) (q[16 * hw + j] >> 4);
                }
            }
        }
    }
}

// Written as a whole-tile pass so the warm path can split it by 8-row group
// (the repack is a pure per-group shuffle, same as the other q4_K tiles).
static inline int sofuu_repack_q4k_8bl_apply2_range(uint8_t * dst, const uint8_t * src,
                                                    int64_t nblocks, int64_t r8_begin,
                                                    int64_t r8_end, unsigned interleave = 4) {
    tile_block_q4_Kx8q * d = (tile_block_q4_Kx8q *) dst + r8_begin * nblocks;
    const tile_block_q4_K * s = (const tile_block_q4_K *) src + r8_begin * 8 * nblocks;
    tile_block_q4_K tmp[8];
    for (int64_t b = r8_begin; b < r8_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            tile_block_q4_Kx8 blk = make_block_q4_Kx8(tmp, interleave);
            sofuu_fill_q4k_apply2_block(d, &blk);
            d++;
        }
        s += 8 * nblocks;
    }
    return 0;
}

static inline int sofuu_repack_q4k_8bl_apply2(uint8_t * dst, const uint8_t * src, int64_t nrow,
                                              int64_t nblocks, unsigned interleave = 4) {
    return sofuu_repack_q4k_8bl_apply2_range(dst, src, nblocks, 0, nrow / 8, interleave);
}

// Shape/gate for the variant: the kernel keeps the 8x4 tile, so the caller
// gates are the same as the _p kernel's.
void ggml_gemm_q4_K_8x4_q8_K_p2(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc) {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);
    UNUSED(nb);

    constexpr int q8_k_blocklen = 4;
    constexpr int acc_size      = 2 * 4;   // 2 weight-row quads x 4 activation rows
    constexpr int nsub          = QK_K / 64;

    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4 * __restrict q8_ptr = (const tile_block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8q * __restrict q4_ptr = (const tile_block_q4_Kx8q *) vx + ((size_t) x * nb);

            for (int i = 0; i < acc_size; i++) acc_f32[i] = vdupq_n_f32(0);

            for (int b = 0; b < nb; b++) {
                const float16x8_t d8    = vld1q_f16((const __fp16 *) q4_ptr[b].d);
                const float16x8_t dm8   = vld1q_f16((const __fp16 *) q4_ptr[b].dmin);
                const float32x4_t q4_d_0123 = vcvt_f32_f16(vget_low_f16(d8));
                const float32x4_t q4_d_4567 = vcvt_f32_f16(vget_high_f16(d8));
                const float32x4_t q4_m_0123 = vcvt_f32_f16(vget_low_f16(dm8));
                const float32x4_t q4_m_4567 = vcvt_f32_f16(vget_high_f16(dm8));
                const float32x4_t q8_d0 = vld1q_f32(q8_ptr[b].d);

                // The per-(activation row, weight-row quad) scale products
                // (the _p kernel's sbd_scale) are rebuilt in the apply instead
                // of being hoisted: 8 extra vmul per sub-block buys back the 8
                // registers that otherwise spill the accumulators.

                // Per-activation-row bsums (8 int16 = 4 sub-blocks x 2 halves),
                // kept in registers and read with lane dups instead of being
                // stored to a stack array and reloaded with ld1r.
                int16x8_t bs[q8_k_blocklen];
                for (int r = 0; r < q8_k_blocklen; r++) {
                    bs[r] = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * r),
                                       vld1q_s16(q8_ptr[b].bsums + 16 * r + 8));
                }

                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) bias_acc[i] = vdupq_n_s32(0);

                for (int sb = 0; sb < nsub; sb++) {
                    int32x4_t acc_lo[acc_size], acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    // Pre-decoded scales/mins: 2 + 2 loads, no bit extraction.
                    const int32x4x2_t scl = vld1q_s32_x2(q4_ptr[b].sc[sb * 2]);
                    const int32x4x2_t sch = vld1q_s32_x2(q4_ptr[b].sc[sb * 2 + 1]);
                    const int16x8_t   mnl = vld1q_s16(q4_ptr[b].mn[sb * 2]);
                    const int16x8_t   mnh = vld1q_s16(q4_ptr[b].mn[sb * 2 + 1]);

                    const int8_t * __restrict q8lo = q8_ptr[b].qs + sb * 256;
                    // Plane bytes are walked linearly (32 B per weight-row half,
                    // lo then hi) so the compiler can use post-increment loads
                    // instead of materialising 16 addresses per sub-block.
                    const int8_t * __restrict plp = (const int8_t *) q4_ptr[b].pl[sb][0][0];

                    for (int k = 0; k < 8; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8lo + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8lo + 16 * k + 128);
                        for (int hw = 0; hw < 2; hw++) {
                            // lo plane (4 rows) then hi plane (4 rows); two
                            // plain loads so both use the base register with an
                            // immediate offset (an ld1 pair cannot).
                            const int8x16_t w_lo = vld1q_s8(plp);
                            const int8x16_t w_hi = vld1q_s8(plp + 16);
                            plp += 32;
                            const int b0 = 4 * hw;
                            acc_lo[b0 + 0] = vdotq_laneq_s32(acc_lo[b0 + 0], w_lo, q8_blk0, 0);
                            acc_lo[b0 + 1] = vdotq_laneq_s32(acc_lo[b0 + 1], w_lo, q8_blk0, 1);
                            acc_lo[b0 + 2] = vdotq_laneq_s32(acc_lo[b0 + 2], w_lo, q8_blk0, 2);
                            acc_lo[b0 + 3] = vdotq_laneq_s32(acc_lo[b0 + 3], w_lo, q8_blk0, 3);
                            acc_hi[b0 + 0] = vdotq_laneq_s32(acc_hi[b0 + 0], w_hi, q8_blk1, 0);
                            acc_hi[b0 + 1] = vdotq_laneq_s32(acc_hi[b0 + 1], w_hi, q8_blk1, 1);
                            acc_hi[b0 + 2] = vdotq_laneq_s32(acc_hi[b0 + 2], w_hi, q8_blk1, 2);
                            acc_hi[b0 + 3] = vdotq_laneq_s32(acc_hi[b0 + 3], w_hi, q8_blk1, 3);
                        }
                    }

                    // Scale apply + bias accumulation, same op order as _p
                    // (int32 scale products are exact; vmla == mul+mul+add).
                    // Unrolled over the 4 activation rows because the bsums
                    // lane dups need constant lane indices.
#define SOFUU_Q4K_APPLY2_ROW(R)                                                                 \
    do {                                                                                        \
        const int16x4_t b_lo = vdup_n_s16(vgetq_lane_s16(bs[sb], 2 * (R)));                       \
        const int16x4_t b_hi = vdup_n_s16(vgetq_lane_s16(bs[sb], 2 * (R) + 1));                   \
        const float32x4_t sumf_0123 = vcvtq_f32_s32(                                             \
            vmlaq_s32(vmulq_s32(scl.val[0], acc_lo[(R)]), sch.val[0], acc_hi[(R)]));             \
        acc_f32[2 * (R)] = vfmaq_f32(acc_f32[2 * (R)],                                           \
            vmulq_laneq_f32(q4_d_0123, q8_d0, (R)), sumf_0123);                                  \
        const float32x4_t sumf_4567 = vcvtq_f32_s32(                                             \
            vmlaq_s32(vmulq_s32(scl.val[1], acc_lo[(R) + 4]), sch.val[1], acc_hi[(R) + 4]));     \
        acc_f32[2 * (R) + 1] = vfmaq_f32(acc_f32[2 * (R) + 1],                                   \
            vmulq_laneq_f32(q4_d_4567, q8_d0, (R)), sumf_4567);                                  \
        bias_acc[2 * (R)]     = vmlal_s16(bias_acc[2 * (R)], b_lo, vget_low_s16(mnl));            \
        bias_acc[2 * (R)]     = vmlal_s16(bias_acc[2 * (R)], b_hi, vget_low_s16(mnh));            \
        bias_acc[2 * (R) + 1] = vmlal_s16(bias_acc[2 * (R) + 1], b_lo, vget_high_s16(mnl));       \
        bias_acc[2 * (R) + 1] = vmlal_s16(bias_acc[2 * (R) + 1], b_hi, vget_high_s16(mnh));       \
    } while (0)
                    SOFUU_Q4K_APPLY2_ROW(0);
                    SOFUU_Q4K_APPLY2_ROW(1);
                    SOFUU_Q4K_APPLY2_ROW(2);
                    SOFUU_Q4K_APPLY2_ROW(3);
#undef SOFUU_Q4K_APPLY2_ROW
                }  // for sb

                // Bias step, same op order as _p: f32(exact int32 sum) times
                // (dmin * q8_d), subtracted once per 256-element block.
                const float32x4_t sm0 = vmulq_laneq_f32(q4_m_0123, q8_d0, 0);
                const float32x4_t sm1 = vmulq_laneq_f32(q4_m_4567, q8_d0, 0);
                const float32x4_t sm2 = vmulq_laneq_f32(q4_m_0123, q8_d0, 1);
                const float32x4_t sm3 = vmulq_laneq_f32(q4_m_4567, q8_d0, 1);
                const float32x4_t sm4 = vmulq_laneq_f32(q4_m_0123, q8_d0, 2);
                const float32x4_t sm5 = vmulq_laneq_f32(q4_m_4567, q8_d0, 2);
                const float32x4_t sm6 = vmulq_laneq_f32(q4_m_0123, q8_d0, 3);
                const float32x4_t sm7 = vmulq_laneq_f32(q4_m_4567, q8_d0, 3);
                acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sm0);
                acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sm1);
                acc_f32[2] = vmlsq_f32(acc_f32[2], vcvtq_f32_s32(bias_acc[2]), sm2);
                acc_f32[3] = vmlsq_f32(acc_f32[3], vcvtq_f32_s32(bias_acc[3]), sm3);
                acc_f32[4] = vmlsq_f32(acc_f32[4], vcvtq_f32_s32(bias_acc[4]), sm4);
                acc_f32[5] = vmlsq_f32(acc_f32[5], vcvtq_f32_s32(bias_acc[5]), sm5);
                acc_f32[6] = vmlsq_f32(acc_f32[6], vcvtq_f32_s32(bias_acc[6]), sm6);
                acc_f32[7] = vmlsq_f32(acc_f32[7], vcvtq_f32_s32(bias_acc[7]), sm7);
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                const int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    const size_t offset = (size_t) row * bs + (size_t) x * 8 + j * 4;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[2 * i + j]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[2 * i + j]);
                    }
                }
            }
        }  // for x
    }  // for y
    return;
#endif
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// ══════════ Wider batch tile: 8 activation rows per weight load ══════════
// The 8x4 tile loads each 8-weight-row block once per 4 activation rows, so a
// prefill over N activation rows streams the whole weight tile set N/4 times
// (0.6B model: 212 MB of tile + sidecar bytes -> 6.8 GB per 128-token chunk),
// and the obvious lever is the activation dimension of the tile: 8 activation
// rows per weight load halves it.
//
// Measured caveat, see gemm_batch8_enabled() in infer_kernels.cpp: at 4 threads
// the 8x4 kernel already runs at ~420 GMAC/s, ~98% of the issue rate this code
// reaches over cache-resident weights, and only ~58 of the ~85 GB/s the machine
// sustains, so halving the stream does not shorten the GEMM — interleaved pairs
// measure parity (569 vs 571 tk/s end to end). This is a bandwidth reduction,
// not a speedup. The sections below document what it took to keep it
// bit-identical; they are dead code unless SOFUU_GEMM_BATCH_TILE=8.
//
// Register budget is what caps the tile shape. One int32x4 accumulator holds a
// (4 weight rows x 1 activation row) int32 dot sum, and the low/high nibble
// halves need separate accumulators (different 6-bit sub-block scales), so a
// tile needs 2 * (weight_rows/4) * activation_rows int32x4 accumulators: 8x4
// already uses 16, and a full 8x8 tile would need 32 — the whole register file,
// before the 8 f32 accumulators. This kernel therefore keeps 4 weight rows per
// pass (16 int32x4 accumulators, the same count as 8x4) and walks the two
// 4-row halves of the SAME 8-row weight block. Bytes per output element are
// unchanged by that split (one 2176 B plane block feeds both halves and stays
// in L1 between them: nb * 2176 B), while the activation rows per weight load
// double.
//
// The bias term (weight mins x activation bsums) is accumulated after the dot
// loops instead of inside them so its 8 int32x4 accumulators do not compete
// with the 16 dot accumulators; int32 sums are exact and order-independent, so
// the value is the one the 8x4 kernel forms inside its sub-block loop.
//
// Bit-identity: every int32 accumulator sums exactly the same int8 products as
// the 8x4 kernel, and the per-output-element scale/bias apply below is the same
// op sequence on the same f32 values with 8 activation lanes instead of 4, so
// no output bit changes. OPT-IN via SOFUU_GEMM_BATCH_TILE=8 (see
// gemm_batch8_enabled in infer_kernels.cpp); default OFF. Verified offline by
// memcmp against both 8x4 kernels over 9 shapes, with and without in_acc.
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

// 8-row activation quantizer: identical per-row amax/iscale/rounding/bsums math
// to ggml_quantize_mat_q8_K_4x4_neon (see its design notes above), stored into
// the interleaved 8-row layout. bsums are summed straight into the 32-k halves
// the kernels consume with int16 adds only (|sum| <= 32*127), which is the same
// value the 4-row path derives inside the kernel with vpaddq_s16.
static inline void ggml_quantize_mat_q8_K_8rows_neon(const float * __restrict x, void * __restrict vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    tile_block_q8_Kx8 * __restrict y = (tile_block_q8_Kx8 *) vy;
    const float32x4_t f32_magic = vdupq_n_f32(12582912.0f);
    const int32x4_t i32_magic = vreinterpretq_s32_f32(f32_magic); // 0x4B400000 bit pattern

    for (int i = 0; i < nb; i++) {
        float iscale[8];
        for (int row_iter = 0; row_iter < 8; row_iter++) {
            const float *row = x + row_iter * k + i * QK_K;
            float32x4_t va = vabsq_f32(vld1q_f32(row));
            for (int j = 4; j < QK_K; j += 4) {
                va = vmaxq_f32(va, vabsq_f32(vld1q_f32(row + j)));
            }
            const float amax = vmaxvq_f32(va);

            float max = 0.0f;
            if (amax > 0.0f) {
                for (int j = 0; j < QK_K; j++) {
                    if (fabsf(row[j]) == amax) { max = row[j]; break; }
                }
            }
            iscale[row_iter] = amax ? -127.f / max : 0;
            y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
        }

        memset(y[i].bsums, 0, sizeof(y[i].bsums));

        for (int t = 0; t < QK_K / 4; t++) {
            int16x4_t qs16[8];
            for (int s = 0; s < 8; s++) {
                const float32x4_t q = vmulq_n_f32(vld1q_f32(x + s * k + i * QK_K + 4 * t), iscale[s]);
                const int32x4_t val = vsubq_s32(vreinterpretq_s32_f32(vaddq_f32(q, f32_magic)), i32_magic);
                qs16[s] = vmovn_s32(val);
                // bsums slot: sub-block (t>>4), row s, 32-k half ((t>>2)&3)>>1 —
                // both 16-k groups of a half accumulate into the same int16 slot.
                y[i].bsums[16 * (t >> 4) + 2 * s + (((t >> 2) & 3) >> 1)] += (int16_t) vaddvq_s32(val);
            }
            vst1q_s8(y[i].qs + 32 * t,
                     vcombine_s8(vmovn_s16(vcombine_s16(qs16[0], qs16[1])),
                                 vmovn_s16(vcombine_s16(qs16[2], qs16[3]))));
            vst1q_s8(y[i].qs + 32 * t + 16,
                     vcombine_s8(vmovn_s16(vcombine_s16(qs16[4], qs16[5])),
                                 vmovn_s16(vcombine_s16(qs16[6], qs16[7]))));
        }
    }
}

// Shared body for both weight-tile flavours: PLANE = pre-expanded nibble-plane
// tile (tile_block_q4_Kx8p, self-contained scales), !PLANE = stock packed tile
// with the T1 pre-decoded scale section (tile_ss_q4_Kx8s).
template <bool PLANE>
static void ggml_gemm_q4_K_8x8_q8_K_impl(int n, float * __restrict s, size_t bs, const void * __restrict vx,
                                         const void * __restrict ss, const void * __restrict vy, int nr, int nc,
                                         const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nr % 8 == 0);
    assert(nc % 8 == 0);
    UNUSED(nb);

    const uint8x16_t m4b = vdupq_n_u8(0x0f);

    for (int y = 0; y < nr / 8; y++) {
        const tile_block_q8_Kx8 * __restrict q8_ptr = (const tile_block_q8_Kx8 *) vy + (size_t) y * nb;

        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8p * __restrict q4p = (const tile_block_q4_Kx8p *) vx + (size_t) x * nb;
            const tile_block_q4_Kx8  * __restrict q4s = (const tile_block_q4_Kx8 *)  vx + (size_t) x * nb;
            const tile_ss_q4_Kx8s    * __restrict ssp = (const tile_ss_q4_Kx8s *)    ss + (size_t) x * nb;
            UNUSED(q4p); UNUSED(q4s); UNUSED(ssp);

            for (int hw = 0; hw < 2; hw++) {  // 4-weight-row half of the 8-row block
                float32x4_t acc_f32[8];
                for (int r = 0; r < 8; r++) acc_f32[r] = vdupq_n_f32(0);

                for (int b = 0; b < nb; b++) {
                    const uint16_t *dptr    = nullptr;
                    const uint16_t *dminptr = nullptr;
                    if constexpr (PLANE) { dptr = q4p[b].d;    dminptr = q4p[b].dmin; }
                    else                 { dptr = q4s[b].d;    dminptr = q4s[b].dmin; }

                    const float32x4_t q4_d    = vcvt_f32_f16(vld1_f16((const __fp16 *) dptr + 4 * hw));
                    const float32x4_t q4_dmin = vcvt_f32_f16(vld1_f16((const __fp16 *) dminptr + 4 * hw));
                    const float32x4_t q8_d0   = vld1q_f32(q8_ptr[b].d);
                    const float32x4_t q8_d1   = vld1q_f32(q8_ptr[b].d + 4);

                    for (int sb = 0; sb < QK_K / 64; sb++) {
                        // Int accumulators for the qs vecdot (4 weight rows x 8 act rows)
                        int32x4_t acc_lo[8], acc_hi[8];
                        for (int r = 0; r < 8; r++) {
                            acc_lo[r] = vdupq_n_s32(0);
                            acc_hi[r] = vdupq_n_s32(0);
                        }

                        // Per-(sub-block, 32-k half) scales/mins for these 4 weight rows
                        // (8-lane decode split by the weight-row half, as in 8x4).
                        int16x4_t isc_lo = vdup_n_s16(0), isc_hi = vdup_n_s16(0);
                        float32x4_t fsc_lo = vdupq_n_f32(0), fsc_hi = vdupq_n_f32(0);
                        int16x4_t mins_lo = vdup_n_s16(0), mins_hi = vdup_n_s16(0);
                        if constexpr (PLANE) {
                            int8_t    aux[2][8];
                            int16x8_t mins[2];
                            decode_q_Kx8_6bit_scales(&q4p[b].scales[sb * 24], &mins[0], aux[0]);
                            decode_q_Kx8_6bit_scales(&q4p[b].scales[sb * 24 + 12], &mins[1], aux[1]);
                            const int16x8_t sc8_lo = vmovl_s8(vld1_s8(aux[0]));
                            const int16x8_t sc8_hi = vmovl_s8(vld1_s8(aux[1]));
                            isc_lo  = hw ? vget_high_s16(sc8_lo) : vget_low_s16(sc8_lo);
                            isc_hi  = hw ? vget_high_s16(sc8_hi) : vget_low_s16(sc8_hi);
                            mins_lo = hw ? vget_high_s16(mins[0]) : vget_low_s16(mins[0]);
                            mins_hi = hw ? vget_high_s16(mins[1]) : vget_low_s16(mins[1]);
                        } else {
                            const __fp16 * sbp = ssp[b].sb_scale + (size_t) sb * 16 + 4 * hw;
                            fsc_lo  = vcvt_f32_f16(vld1_f16(sbp));
                            fsc_hi  = vcvt_f32_f16(vld1_f16(sbp + 8));
                            const int16x8_t m8_lo = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ssp[b].sb_min + (size_t) sb * 16)));
                            const int16x8_t m8_hi = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ssp[b].sb_min + (size_t) sb * 16 + 8)));
                            mins_lo = hw ? vget_high_s16(m8_lo) : vget_low_s16(m8_lo);
                            mins_hi = hw ? vget_high_s16(m8_hi) : vget_low_s16(m8_hi);
                        }

                        const int8_t * __restrict q8lo = q8_ptr[b].qs + (size_t) sb * 512;
                        const int8_t * __restrict q8hi = q8lo + 256;

                        constexpr int reads_per_sb = 8;
                        for (int kg = 0; kg < reads_per_sb; kg++) {
                            // 32 B = both 4-activation-row groups of this k group.
                            const int8x16x2_t a_lo = vld1q_s8_x2(q8lo + 32 * kg);
                            const int8x16x2_t a_hi = vld1q_s8_x2(q8hi + 32 * kg);

                            int8x16_t w_lo, w_hi;
                            if constexpr (PLANE) {
                                // Pre-expanded nibble planes: identical int8 vectors
                                // to the packed tile's vand/vshr outputs.
                                const uint32_t poff = (uint32_t) ((sb * 8 + kg) * 32 + 16 * hw);
                                w_lo = vld1q_s8((const int8_t *) (q4p[b].qs_lo + poff));
                                w_hi = vld1q_s8((const int8_t *) (q4p[b].qs_hi + poff));
                            } else {
                                const uint8x16_t p = vld1q_u8(q4s[b].qs + sb * QK_K + 32 * kg + 16 * hw);
                                w_lo = vreinterpretq_s8_u8(vandq_u8(p, m4b));
                                w_hi = vreinterpretq_s8_u8(vshrq_n_u8(p, 4));
                            }

                            acc_lo[0] = vdotq_laneq_s32(acc_lo[0], w_lo, a_lo.val[0], 0);
                            acc_lo[1] = vdotq_laneq_s32(acc_lo[1], w_lo, a_lo.val[0], 1);
                            acc_lo[2] = vdotq_laneq_s32(acc_lo[2], w_lo, a_lo.val[0], 2);
                            acc_lo[3] = vdotq_laneq_s32(acc_lo[3], w_lo, a_lo.val[0], 3);
                            acc_lo[4] = vdotq_laneq_s32(acc_lo[4], w_lo, a_lo.val[1], 0);
                            acc_lo[5] = vdotq_laneq_s32(acc_lo[5], w_lo, a_lo.val[1], 1);
                            acc_lo[6] = vdotq_laneq_s32(acc_lo[6], w_lo, a_lo.val[1], 2);
                            acc_lo[7] = vdotq_laneq_s32(acc_lo[7], w_lo, a_lo.val[1], 3);

                            acc_hi[0] = vdotq_laneq_s32(acc_hi[0], w_hi, a_hi.val[0], 0);
                            acc_hi[1] = vdotq_laneq_s32(acc_hi[1], w_hi, a_hi.val[0], 1);
                            acc_hi[2] = vdotq_laneq_s32(acc_hi[2], w_hi, a_hi.val[0], 2);
                            acc_hi[3] = vdotq_laneq_s32(acc_hi[3], w_hi, a_hi.val[0], 3);
                            acc_hi[4] = vdotq_laneq_s32(acc_hi[4], w_hi, a_hi.val[1], 0);
                            acc_hi[5] = vdotq_laneq_s32(acc_hi[5], w_hi, a_hi.val[1], 1);
                            acc_hi[6] = vdotq_laneq_s32(acc_hi[6], w_hi, a_hi.val[1], 2);
                            acc_hi[7] = vdotq_laneq_s32(acc_hi[7], w_hi, a_hi.val[1], 3);
                        }

                        // Scale apply — same op sequence as the matching 8x4 kernel.
                        // Unrolled because the vmul/vdot lane forms need constant
                        // lane indices (activation row R uses lane R & 3 of its
                        // 4-row q8 d vector).
#define SOFUU_Q4K_8X8_SCALE_ROW(R, Q8D)                                                          \
    do {                                                                                         \
        float32x4_t sumf;                                                                        \
        if constexpr (PLANE) {                                                                    \
            sumf = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(isc_lo), acc_lo[R]),              \
                                           vmulq_s32(vmovl_s16(isc_hi), acc_hi[R])));            \
        } else {                                                                                 \
            sumf = vaddq_f32(vmulq_f32(fsc_lo, vcvtq_f32_s32(acc_lo[R])),                        \
                             vmulq_f32(fsc_hi, vcvtq_f32_s32(acc_hi[R])));                       \
        }                                                                                        \
        acc_f32[R] = vfmaq_f32(acc_f32[R], vmulq_laneq_f32(q4_d, Q8D, (R) & 3), sumf);          \
    } while (0)
                        SOFUU_Q4K_8X8_SCALE_ROW(0, q8_d0);
                        SOFUU_Q4K_8X8_SCALE_ROW(1, q8_d0);
                        SOFUU_Q4K_8X8_SCALE_ROW(2, q8_d0);
                        SOFUU_Q4K_8X8_SCALE_ROW(3, q8_d0);
                        SOFUU_Q4K_8X8_SCALE_ROW(4, q8_d1);
                        SOFUU_Q4K_8X8_SCALE_ROW(5, q8_d1);
                        SOFUU_Q4K_8X8_SCALE_ROW(6, q8_d1);
                        SOFUU_Q4K_8X8_SCALE_ROW(7, q8_d1);
#undef SOFUU_Q4K_8X8_SCALE_ROW
                    }  // for sb

                    // Bias term (exact int32 accumulation of mins x bsums), then
                    // the same vmlsq the 8x4 kernel runs at the end of each block.
                    int32x4_t bias_acc[8];
                    for (int r = 0; r < 8; r++) bias_acc[r] = vdupq_n_s32(0);
                    for (int sb = 0; sb < QK_K / 64; sb++) {
                        int16x4_t mins_lo, mins_hi;
                        if constexpr (PLANE) {
                            int8_t    aux[8];
                            int16x8_t mins;
                            decode_q_Kx8_6bit_scales(&q4p[b].scales[sb * 24], &mins, aux);
                            mins_lo = hw ? vget_high_s16(mins) : vget_low_s16(mins);
                            decode_q_Kx8_6bit_scales(&q4p[b].scales[sb * 24 + 12], &mins, aux);
                            mins_hi = hw ? vget_high_s16(mins) : vget_low_s16(mins);
                        } else {
                            const int16x8_t m8_lo = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ssp[b].sb_min + (size_t) sb * 16)));
                            const int16x8_t m8_hi = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ssp[b].sb_min + (size_t) sb * 16 + 8)));
                            mins_lo = hw ? vget_high_s16(m8_lo) : vget_low_s16(m8_lo);
                            mins_hi = hw ? vget_high_s16(m8_hi) : vget_low_s16(m8_hi);
                        }
                        for (int r = 0; r < 8; r++) {
                            const int16x4_t bsums_lo = vdup_n_s16(q8_ptr[b].bsums[16 * sb + 2 * r]);
                            const int16x4_t bsums_hi = vdup_n_s16(q8_ptr[b].bsums[16 * sb + 2 * r + 1]);
                            bias_acc[r] = vmlal_s16(bias_acc[r], bsums_lo, mins_lo);
                            bias_acc[r] = vmlal_s16(bias_acc[r], bsums_hi, mins_hi);
                        }
                    }
#define SOFUU_Q4K_8X8_MIN_ROW(R, Q8D) \
    acc_f32[R] = vmlsq_f32(acc_f32[R], vcvtq_f32_s32(bias_acc[R]), vmulq_laneq_f32(q4_dmin, Q8D, (R) & 3))
                    SOFUU_Q4K_8X8_MIN_ROW(0, q8_d0);
                    SOFUU_Q4K_8X8_MIN_ROW(1, q8_d0);
                    SOFUU_Q4K_8X8_MIN_ROW(2, q8_d0);
                    SOFUU_Q4K_8X8_MIN_ROW(3, q8_d0);
                    SOFUU_Q4K_8X8_MIN_ROW(4, q8_d1);
                    SOFUU_Q4K_8X8_MIN_ROW(5, q8_d1);
                    SOFUU_Q4K_8X8_MIN_ROW(6, q8_d1);
                    SOFUU_Q4K_8X8_MIN_ROW(7, q8_d1);
#undef SOFUU_Q4K_8X8_MIN_ROW
                }  // for b

                for (int r = 0; r < 8; r++) {
                    const size_t offset = (size_t) (y * 8 + r) * bs + (size_t) x * 8 + 4 * hw;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), acc_f32[r]));
                    } else {
                        vst1q_f32(s + offset, acc_f32[r]);
                    }
                }
            }  // for hw
        }  // for x
    }  // for y
}

// Nibble-plane weight tile (tile_block_q4_Kx8p), 8 weight rows x 8 act rows.
void ggml_gemm_q4_K_8x8_q8_K_p(int n, float * __restrict s, size_t bs, const void * __restrict vx,
                               const void * __restrict vy, int nr, int nc, const float * in_acc) {
    ggml_gemm_q4_K_8x8_q8_K_impl<true>(n, s, bs, vx, nullptr, vy, nr, nc, in_acc);
}

// Stock packed weight tile (tile_block_q4_Kx8) + T1 scale section, 8x8.
void ggml_gemm_q4_K_8x8_q8_K_s(int n, float * __restrict s, size_t bs, const void * __restrict vx,
                               const void * __restrict ss, const void * __restrict vy, int nr, int nc,
                               const float * in_acc) {
    ggml_gemm_q4_K_8x8_q8_K_impl<false>(n, s, bs, vx, ss, vy, nr, nc, in_acc);
}

#endif  // __aarch64__ && __ARM_NEON && __ARM_FEATURE_DOTPROD

// ══════════ i8mm (FEAT_I8MM / SMMLA) q4_K prefill GEMM ══════════
// SMMLA (vmmlaq_s32) computes a 2-wt x 2-act block of 8-element int8 dots per
// instruction (32 MACs) from operands laid out as two 8-byte rows:
//   w[16] = [wt row 2p, k0..k7][wt row 2p+1, k0..k7]
//   a[16] = [act row 2q, k0..k7][act row 2q+1, k0..k7]
//   lanes = (w2p,a2q) (w2p,a2q+1) (w2p+1,a2q) (w2p+1,a2q+1)
// so both tiles need an 8-k-grouped layout: the SDOT plane tile groups 4 k per
// byte lane and the q8_Kx4 tile groups 4 k per row, hence the two new tile
// types below. The int32 partial sums are exact and order-independent, so the
// i8mm kernel is bit-identical to the SDOT plane kernel provided every int32
// product / f32 fma per output element is unchanged — the scale apply is the
// same elementwise op sequence, only the lanes are permuted (gated by
// sofuu_test_gemm_q4_K_i8mm in infer_kernels.cpp).
//
// Weight tile: same d/dmin/scales header as tile_block_q4_Kx8p; plane byte x of
// the 16B group (sb, kg, p) is the nibble of (row 2p+(x>>3), k = sb*64 +
// half*32 + kg*8 + (x&7)) — byte-identical values to the SDOT planes, regrouped.
struct tile_block_q4_Kx8i {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[96];
    uint8_t  qs_lo[1024];  // SMMLA form, 16B = rows 2p,2p+1 x 8 k
    uint8_t  qs_hi[1024];
};

// Activation tile: qs holds 16B = rows 2q,2q+1 x 8 k per (sb, kg); bsums holds
// the per-(sb, batch-pair, half) sums already broadcast to the SMMLA lane
// pattern, so the bias path needs no scalar shuffling in the hot loop.
struct tile_block_q8_Kx4i {
    float   d[4];
    int8_t  qs[1024];
    int16_t bsums[64];  // [((sb*2+q)*2+half)*4 + lane] = [b(2q),b(2q+1),b(2q),b(2q+1)]
};

static inline int sofuu_repack_q4k_8bl_i8mm_range(uint8_t * dst, const uint8_t * src, int64_t nblocks, int64_t r8_begin, int64_t r8_end) {
    tile_block_q4_Kx8i * d = (tile_block_q4_Kx8i *) dst + r8_begin * nblocks;
    const tile_block_q4_K * s = (const tile_block_q4_K *) src + r8_begin * 8 * nblocks;
    tile_block_q4_K tmp[8];
    for (int64_t b = r8_begin; b < r8_end; b++) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) tmp[i] = s[x + i * nblocks];
            // Reuse the stock repacker for d/dmin/scales (bit-identical 6-bit
            // group packing to every other q4_K tile).
            tile_block_q4_Kx8 blk = make_block_q4_Kx8(tmp, 4);
            memcpy(d->d, blk.d, sizeof(d->d));
            memcpy(d->dmin, blk.dmin, sizeof(d->dmin));
            memcpy(d->scales, blk.scales, sizeof(d->scales));
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int half = 0; half < 2; half++) {
                    for (int kg = 0; kg < 4; kg++) {
                        for (int p = 0; p < 4; p++) {
                            uint8_t * o = (half ? d->qs_hi : d->qs_lo) + (((sb * 4 + kg) * 4 + p) * 16);
                            for (int x = 0; x < 16; x++) {
                                const int r = 2 * p + (x >> 3);
                                // k = sb*64 + half*32 + kg*8 + (x&7)  ->  the q4_K
                                // row byte e = (k/64)*32 + (k%32), nibble by half.
                                const int e = sb * 32 + kg * 8 + (x & 7);
                                const uint8_t byte = tmp[r].qs[e];
                                o[x] = (uint8_t)(half ? (byte >> 4) : (byte & 0x0F));
                            }
                        }
                    }
                }
            }
            d++;
        }
        s += 8 * nblocks;
    }
    return 0;
}

static inline int sofuu_repack_q4k_8bl_i8mm(uint8_t * dst, const uint8_t * src, int64_t nrow, int64_t nblocks) {
    return sofuu_repack_q4k_8bl_i8mm_range(dst, src, nblocks, 0, nrow / 8);
}

// NEON activation quantizer for the SMMLA q8_K tile: the same per-256 block
// quantization as ggml_quantize_mat_q8_K_4x4_neon (identical d, identical int8
// values and bsums — only the store offsets differ), so the int8 operands are
// the same bytes the SDOT path uses.
static inline void ggml_quantize_mat_q8_K_4x4_i8mm_neon(const float * __restrict x, void * __restrict vy, int64_t k) {
#if defined(__ARM_NEON)
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    tile_block_q8_Kx4i * __restrict y = (tile_block_q8_Kx4i *) vy;
    const float32x4_t f32_magic = vdupq_n_f32(12582912.0f);
    const int32x4_t i32_magic = vreinterpretq_s32_f32(f32_magic);

    for (int i = 0; i < nb; i++) {
        float iscale[4];
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            const float *row = x + row_iter * k + i * QK_K;
            float32x4_t va = vabsq_f32(vld1q_f32(row));
            for (int j = 4; j < QK_K; j += 4) {
                va = vmaxq_f32(va, vabsq_f32(vld1q_f32(row + j)));
            }
            const float amax = vmaxvq_f32(va);

            float max = 0.0f;
            if (amax > 0.0f) {
                for (int j = 0; j < QK_K; j++) {
                    if (fabsf(row[j]) == amax) { max = row[j]; break; }
                }
            }
            iscale[row_iter] = amax ? -127.f / max : 0;
            y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
        }

        int16_t raw[QK_K / 4];
        memset(raw, 0, sizeof(raw));

        // 8-k group G: qs[G/8*256 + (G%8)*32 + row*8 + x] = row's k = G*8+x.
        // Rows are 8-byte strides apart inside the 16B SMMLA operand pair.
        for (int G = 0; G < QK_K / 8; G++) {
            int16x4_t qs16[4][2];
            for (int h = 0; h < 2; h++) {
                const int t = 2 * G + h;
                for (int s = 0; s < 4; s++) {
                    const float32x4_t q = vmulq_n_f32(vld1q_f32(x + s * k + i * QK_K + 4 * t), iscale[s]);
                    const int32x4_t val = vsubq_s32(vreinterpretq_s32_f32(vaddq_f32(q, f32_magic)), i32_magic);
                    qs16[s][h] = vmovn_s32(val);
                    const int idx = 4 * s + (t >> 4) * 16 + ((t >> 2) & 3);  // identical to the SDOT tile
                    raw[idx] = (int16_t)(raw[idx] + (int16_t)vaddvq_s32(val));
                }
            }
            uint8_t * o = (uint8_t *) (y[i].qs + ((G >> 3) * 256 + (G & 7) * 32));
            for (int s = 0; s < 4; s++)
                vst1_s8((int8_t *) (o + s * 8), vmovn_s16(vcombine_s16(qs16[s][0], qs16[s][1])));
        }
        // bsums: the SDOT kernels' bsums_arr[sb][l] = raw[sb*16+l] + raw[sb*16+l+8]
        // (a vpaddq_s16 of the two halves), indexed [2*batch_row + half]; store
        // that value pre-broadcast to the SMMLA accumulator lane pattern.
        for (int sb = 0; sb < QK_K / 64; sb++) {
            int16_t a8[8];
            // vpaddq_s16 is a PAIRWISE add over the concatenated halves:
            //   l < 4: raw[2l]     + raw[2l+1]        (row l, its two 16-groups)
            //   l >= 4: raw[8+2(l-4)] + raw[8+2(l-4)+1]  (rows 2,3)
            for (int l = 0; l < 4; l++) a8[l]     = (int16_t)(raw[sb * 16 + 2 * l] + raw[sb * 16 + 2 * l + 1]);
            for (int l = 4; l < 8; l++) a8[l]     = (int16_t)(raw[sb * 16 + 8 + 2 * (l - 4)] + raw[sb * 16 + 9 + 2 * (l - 4)]);
            for (int q = 0; q < 2; q++) {
                for (int half = 0; half < 2; half++) {
                    const int16_t v0 = a8[4 * q + half];
                    const int16_t v1 = a8[4 * q + 2 + half];
                    int16_t * o = y[i].bsums + ((sb * 2 + q) * 2 + half) * 4;
                    o[0] = v0; o[1] = v1; o[2] = v0; o[3] = v1;
                }
            }
        }
    }
#else
    (void) x; (void) vy; (void) k;
#endif
}

// ── i8mm q4_K 8x4 x q8_K GEMM ───────────────────────────────────────────────
// Tile shape: 8 weight rows x 4 batch rows per call (nc=8, nr%4==0), 16 int32
// accumulators (4 weight pairs x 2 batch pairs x lo/hi nibble plane) — the same
// footprint as the SDOT plane kernel, so the loop structure and the per-b/per-sb
// float apply are unchanged. Compiled with a per-function i8mm target attribute
// (this build's -march is armv8.2-a+fp16+dotprod, which does not define
// __ARM_FEATURE_I8MM); the dispatcher takes it only when the CPU implements
// FEAT_I8MM, otherwise the SDOT plane kernel below runs unchanged.
// Per-function target attribute: this build's -march is armv8.2-a+fp16+dotprod
// (crates/sofuu-ffi/build.rs FP16_MARCH), which does not define
// __ARM_FEATURE_I8MM, so the kernel body needs the feature added locally. Only
// clang accepts this spelling here, so builds without __ARM_FEATURE_I8MM *and*
// without clang compile the declaration only (SOFUU_HAVE_GEMM_I8MM stays
// undefined) and the dispatcher never selects the path.
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) &&     (defined(__ARM_FEATURE_I8MM) || defined(__clang__))
#define SOFUU_HAVE_GEMM_I8MM 1
#endif

#if defined(SOFUU_HAVE_GEMM_I8MM)
#if !defined(__ARM_FEATURE_I8MM)
__attribute__((target("i8mm")))
#endif
void ggml_gemm_q4_K_8x4_q8_K_i8mm(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    constexpr int q8_k_blocklen = 4;

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const tile_block_q8_Kx4i * __restrict q8_ptr = (const tile_block_q8_Kx4i *) vy + (y * nb);

        for (int x = 0; x < nc / 8; x++) {
            const tile_block_q4_Kx8i * __restrict q4_ptr = (const tile_block_q4_Kx8i *) vx + (x * nb);

            // acc[q][p] lanes: (w2p,a2q) (w2p,a2q+1) (w2p+1,a2q) (w2p+1,a2q+1)
            float32x4_t acc_f32[2][4];
            for (int q = 0; q < 2; q++)
                for (int p = 0; p < 4; p++) acc_f32[q][p] = vdupq_n_f32(0);

            for (int b = 0; b < nb; b++) {
                const float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                const float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                const float32x4_t q8_d         = vld1q_f32(q8_ptr[b].d);
                const float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                const float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                // Lane-shaped scale factors: dwp[p] = [dw2p,dw2p,dw2p+1,dw2p+1],
                // dq[q] = [dq2q,dq2q+1,dq2q,dq2q+1]; the products are the same
                // per-element f32 values the SDOT kernel forms with vmulq_laneq.
                float32x4_t dwp[4], dminp[4], dq[2];
                dwp[0]   = vzip1q_f32(q4_d_0123, q4_d_0123);
                dwp[1]   = vzip2q_f32(q4_d_0123, q4_d_0123);
                dwp[2]   = vzip1q_f32(q4_d_4567, q4_d_4567);
                dwp[3]   = vzip2q_f32(q4_d_4567, q4_d_4567);
                dminp[0] = vzip1q_f32(q4_dmin_0123, q4_dmin_0123);
                dminp[1] = vzip2q_f32(q4_dmin_0123, q4_dmin_0123);
                dminp[2] = vzip1q_f32(q4_dmin_4567, q4_dmin_4567);
                dminp[3] = vzip2q_f32(q4_dmin_4567, q4_dmin_4567);
                dq[0]    = vcombine_f32(vget_low_f32(q8_d), vget_low_f32(q8_d));
                dq[1]    = vcombine_f32(vget_high_f32(q8_d), vget_high_f32(q8_d));

                float32x4_t sbd_min[2][4];
                for (int q = 0; q < 2; q++)
                    for (int p = 0; p < 4; p++) sbd_min[q][p] = vmulq_f32(dminp[p], dq[q]);

                int32x4_t bias_acc[2][4];
                for (int q = 0; q < 2; q++)
                    for (int p = 0; p < 4; p++) bias_acc[q][p] = vdupq_n_s32(0);

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    int32x4_t acc_lo[2][4], acc_hi[2][4];
                    for (int q = 0; q < 2; q++)
                        for (int p = 0; p < 4; p++) {
                            acc_lo[q][p] = vdupq_n_s32(0);
                            acc_hi[q][p] = vdupq_n_s32(0);
                        }

                    const int8_t * __restrict ab = q8_ptr[b].qs + sb * 256;
                    const int8_t * __restrict wlo = (const int8_t *) (q4_ptr[b].qs_lo + sb * 256);
                    const int8_t * __restrict whi = (const int8_t *) (q4_ptr[b].qs_hi + sb * 256);

                    for (int kg = 0; kg < 4; kg++) {
                        const int8x16_t a_lo0 = vld1q_s8(ab + kg * 32);
                        const int8x16_t a_lo1 = vld1q_s8(ab + kg * 32 + 16);
                        const int8x16_t a_hi0 = vld1q_s8(ab + kg * 32 + 128);
                        const int8x16_t a_hi1 = vld1q_s8(ab + kg * 32 + 144);
                        const int8_t * w = wlo + kg * 64;
                        const int8_t * v = whi + kg * 64;
                        for (int p = 0; p < 4; p++) {
                            const int8x16_t w_lo = vld1q_s8(w + p * 16);
                            const int8x16_t w_hi = vld1q_s8(v + p * 16);
                            acc_lo[0][p] = vmmlaq_s32(acc_lo[0][p], w_lo, a_lo0);
                            acc_lo[1][p] = vmmlaq_s32(acc_lo[1][p], w_lo, a_lo1);
                            acc_hi[0][p] = vmmlaq_s32(acc_hi[0][p], w_hi, a_hi0);
                            acc_hi[1][p] = vmmlaq_s32(acc_hi[1][p], w_hi, a_hi1);
                        }
                    }

                    // 6-bit group decode + lane-shaped int16/int32 scales/mins
                    int16x8_t q4sb_mins[2];
                    int16x4_t scl[2][4];   // [half][p] int16 = [s2p,s2p,s2p+1,s2p+1]
                    int16x4_t mnb[2][4];
                    for (int h = 0; h < 2; h++) {
                        int8_t aux_q4sb[8];
                        decode_q_Kx8_6bit_scales(&q4_ptr[b].scales[sb * 24 + h * 12], &q4sb_mins[h], aux_q4sb);
                        const int8x8_t sv = vld1_s8(aux_q4sb);
                        const int16x8_t s16 = vmovl_s8(sv);
                        const int16x4_t sl = vzip1_s16(vget_low_s16(s16), vget_low_s16(s16));
                        const int16x4_t sh = vzip2_s16(vget_low_s16(s16), vget_low_s16(s16));
                        const int16x4_t tl = vzip1_s16(vget_high_s16(s16), vget_high_s16(s16));
                        const int16x4_t th = vzip2_s16(vget_high_s16(s16), vget_high_s16(s16));
                        scl[h][0] = sl; scl[h][1] = sh; scl[h][2] = tl; scl[h][3] = th;
                        const int16x4_t m0 = vget_low_s16(q4sb_mins[h]);
                        const int16x4_t m1 = vget_high_s16(q4sb_mins[h]);
                        mnb[h][0] = vzip1_s16(m0, m0);
                        mnb[h][1] = vzip2_s16(m0, m0);
                        mnb[h][2] = vzip1_s16(m1, m1);
                        mnb[h][3] = vzip2_s16(m1, m1);
                    }

                    for (int q = 0; q < 2; q++) {
                        const int16x4_t bs_lo = vld1_s16(q8_ptr[b].bsums + ((sb * 2 + q) * 2 + 0) * 4);
                        const int16x4_t bs_hi = vld1_s16(q8_ptr[b].bsums + ((sb * 2 + q) * 2 + 1) * 4);
                        for (int p = 0; p < 4; p++) {
                            const int32x4_t isum =
                                vaddq_s32(vmulq_s32(vmovl_s16(scl[0][p]), acc_lo[q][p]),
                                          vmulq_s32(vmovl_s16(scl[1][p]), acc_hi[q][p]));
                            const float32x4_t sumf = vcvtq_f32_s32(isum);
                            acc_f32[q][p] = vfmaq_f32(acc_f32[q][p], vmulq_f32(dwp[p], dq[q]), sumf);
                            bias_acc[q][p] = vmlal_s16(bias_acc[q][p], bs_lo, mnb[0][p]);
                            bias_acc[q][p] = vmlal_s16(bias_acc[q][p], bs_hi, mnb[1][p]);
                        }
                    }
                }  // for sb

                for (int q = 0; q < 2; q++)
                    for (int p = 0; p < 4; p++)
                        acc_f32[q][p] = vmlsq_f32(acc_f32[q][p], vcvtq_f32_s32(bias_acc[q][p]), sbd_min[q][p]);
            }  // for b

            // lanes -> (batch row, 4 weight cols): vuzp1 = even lanes (batch 2q),
            // vuzp2 = odd lanes (batch 2q+1).
            float32x4_t rowv[4][2];
            for (int q = 0; q < 2; q++) {
                rowv[2 * q][0]     = vuzp1q_f32(acc_f32[q][0], acc_f32[q][1]);
                rowv[2 * q][1]     = vuzp1q_f32(acc_f32[q][2], acc_f32[q][3]);
                rowv[2 * q + 1][0] = vuzp2q_f32(acc_f32[q][0], acc_f32[q][1]);
                rowv[2 * q + 1][1] = vuzp2q_f32(acc_f32[q][2], acc_f32[q][3]);
            }
            for (int i = 0; i < q8_k_blocklen; i++) {
                const int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    const int col    = x * 8 + j * 4;
                    const int offset = row * bs + col;
                    if (in_acc) {
                        vst1q_f32(s + offset, vaddq_f32(vld1q_f32(s + offset), rowv[i][j]));
                    } else {
                        vst1q_f32(s + offset, rowv[i][j]);
                    }
                }
            }
        }  // for x
    }  // for y
}
#else
void ggml_gemm_q4_K_8x4_q8_K_i8mm(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc, const float * in_acc);
#endif

#endif // SOFUU_ARCH_A76_REPACK_H
