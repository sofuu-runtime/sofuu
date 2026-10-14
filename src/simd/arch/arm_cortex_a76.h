#ifndef SOFUU_ARCH_ARM_CORTEX_A76_H
#define SOFUU_ARCH_ARM_CORTEX_A76_H

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

struct alignas(16) sofuu_q8_block_ext {
    float d;
    int16_t sum_x;
    int8_t qs[32];
};

static inline float sofuu_arch_f16_to_f32(uint16_t h) {
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

static inline uint16_t sofuu_arch_f32_to_f16(float val) {
    uint32_t x; memcpy(&x,&val,4);
    uint32_t sign=(x>>31)&1; int32_t exp=((x>>23)&0xff)-127+15; uint32_t mant=(x>>13)&0x3ff;
    if (exp<=0) return (uint16_t)(sign<<15);
    if (exp>=31) return (uint16_t)((sign<<15)|0x7c00);
    return (uint16_t)((sign<<15)|(exp<<10)|mant);
}

static inline float sofuu_arch_bf16_to_f32(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void sofuu_arch_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d=q[j]&63; *m=q[j+4]&63; }
    else { *d=(q[j+4]&0x0F)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j-0]>>6)<<4); }
}

static inline void sofuu_quantize_row_q8_a76(const float *x, sofuu_q8_block_ext *y, size_t n_blocks) {
    for (size_t b = 0; b < n_blocks; ++b) {
        const float *xb = x + b * 32;
        float32x4_t v_max = vdupq_n_f32(0.0f);
        for (int i = 0; i < 32; i += 4) {
            float32x4_t vx = vabsq_f32(vld1q_f32(xb + i));
            v_max = vmaxq_f32(v_max, vx);
        }
        float amax = vmaxvq_f32(v_max);
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        y[b].d = d;
        float32x4_t vid = vdupq_n_f32(id);
        int32x4_t sum_vec = vdupq_n_s32(0);
        for (int i = 0; i < 32; i += 16) {
            float32x4_t x0 = vmulq_f32(vld1q_f32(xb + i + 0), vid);
            float32x4_t x1 = vmulq_f32(vld1q_f32(xb + i + 4), vid);
            float32x4_t x2 = vmulq_f32(vld1q_f32(xb + i + 8), vid);
            float32x4_t x3 = vmulq_f32(vld1q_f32(xb + i + 12), vid);
            int32x4_t s0 = vcvtnq_s32_f32(x0);
            int32x4_t s1 = vcvtnq_s32_f32(x1);
            int32x4_t s2 = vcvtnq_s32_f32(x2);
            int32x4_t s3 = vcvtnq_s32_f32(x3);
            sum_vec = vaddq_s32(sum_vec, vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3)));
            int16x8_t p0 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
            int16x8_t p1 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
            int8x16_t qs = vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1));
            vst1q_s8(y[b].qs + i, qs);
        }
        y[b].sum_x = (int16_t)vaddvq_s32(sum_vec);
    }
}

static inline float sofuu_vec_dot_q8_0_dotprod_a76(const struct block_q8_0 *w, const sofuu_q8_block_ext *x, size_t n_blocks) {
    float total = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        int8x16_t w0 = vld1q_s8(w[b].qs + 0);
        int8x16_t w1 = vld1q_s8(w[b].qs + 16);
        int8x16_t x0 = vld1q_s8(x[b].qs + 0);
        int8x16_t x1 = vld1q_s8(x[b].qs + 16);
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, x0);
        acc = vdotq_s32(acc, w1, x1);
        total += (float)vaddvq_s32(acc) * (sofuu_arch_f16_to_f32(w[b].d) * x[b].d);
    }
    return total;
}

static inline float sofuu_vec_dot_q4_k_dotprod_a76(const struct block_q4_K *blocks, const sofuu_q8_block_ext *x_blocks, size_t n_blocks) {
    float total_sum = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const struct block_q4_K *blk = &blocks[b];
        float d = sofuu_arch_f16_to_f32(blk->d);
        float min = sofuu_arch_f16_to_f32(blk->dmin);
        const uint8_t *q = blk->qs;
        const sofuu_q8_block_ext *xb = x_blocks + b * 8;
        int is = 0; uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            sofuu_arch_get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc; float m1 = min * (float)m;
            sofuu_arch_get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc; float m2 = min * (float)m;
            uint8x16_t raw0 = vld1q_u8(q + 0);
            uint8x16_t raw1 = vld1q_u8(q + 16);
            int8x16_t q_lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, vdupq_n_u8(0x0f)));
            int8x16_t q_lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, vdupq_n_u8(0x0f)));
            int8x16_t x_lo0 = vld1q_s8(xb[0].qs + 0);
            int8x16_t x_lo1 = vld1q_s8(xb[0].qs + 16);
            int32x4_t acc_lo = vdotq_s32(vdupq_n_s32(0), q_lo0, x_lo0);
            acc_lo = vdotq_s32(acc_lo, q_lo1, x_lo1);
            float dx1 = xb[0].d;
            total_sum += (d1 * dx1) * (float)vaddvq_s32(acc_lo) - (m1 * dx1) * (float)xb[0].sum_x;
            int8x16_t q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            int8x16_t q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            int8x16_t x_hi0 = vld1q_s8(xb[1].qs + 0);
            int8x16_t x_hi1 = vld1q_s8(xb[1].qs + 16);
            int32x4_t acc_hi = vdotq_s32(vdupq_n_s32(0), q_hi0, x_hi0);
            acc_hi = vdotq_s32(acc_hi, q_hi1, x_hi1);
            float dx2 = xb[1].d;
            total_sum += (d2 * dx2) * (float)vaddvq_s32(acc_hi) - (m2 * dx2) * (float)xb[1].sum_x;
            q += 32; xb += 2; is += 2;
        }
    }
    return total_sum;
}

static inline float sofuu_vec_dot_mlx4_dotprod_a76(const uint8_t *w_row, const uint8_t *s_row, const uint8_t *b_row, const float *x, size_t cols, size_t group_size, const sofuu_q8_block_ext *x_blocks, bool is_bf16) {
    float total = 0.0f;
    const size_t groups = cols / group_size;
    const size_t u32_per_group = group_size / 8;
    const size_t q8_per_group = group_size / 32;
    const uint8x16_t mask_nibble = vdupq_n_u8(0x0f);

    for (size_t g = 0; g < groups; ++g) {
        uint16_t s_raw = 0;
        uint16_t b_raw = 0;
        memcpy(&s_raw, s_row + g * 2, 2);
        memcpy(&b_raw, b_row + g * 2, 2);
        const float scale = is_bf16 ? sofuu_arch_bf16_to_f32(s_raw) : sofuu_arch_f16_to_f32(s_raw);
        const float bias = is_bf16 ? sofuu_arch_bf16_to_f32(b_raw) : sofuu_arch_f16_to_f32(b_raw);
        const uint8_t *wg = w_row + g * u32_per_group * 4;
        const sofuu_q8_block_ext *xg = x_blocks + g * q8_per_group;

        for (size_t c = 0; c < u32_per_group; c += 4) {
            uint8x16_t raw = vld1q_u8(wg + c * 4);
            int8x16_t even = vreinterpretq_s8_u8(vandq_u8(raw, mask_nibble));
            int8x16_t odd = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
            int8x16_t q0 = vzip1q_s8(even, odd);
            int8x16_t q1 = vzip2q_s8(even, odd);
            const size_t block_idx = c / 4;
            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(xg[block_idx].qs + 0));
            acc = vdotq_s32(acc, q1, vld1q_s8(xg[block_idx].qs + 16));
            const int32_t block_dot = vaddvq_s32(acc);
            const int32_t block_sum_x = xg[block_idx].sum_x;
            total += (scale * (float)block_dot + bias * (float)block_sum_x) * xg[block_idx].d;
        }
    }
    return total;
}

static inline float sofuu_vec_dot_q6_k_dotprod_a76(const struct block_q6_K *blocks, const sofuu_q8_block_ext *x_blocks, size_t n_blocks) {
    float total_sum = 0.0f;
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t sub32 = vdupq_n_s8(32);
    for (size_t b = 0; b < n_blocks; ++b) {
        const struct block_q6_K *blk = &blocks[b];
        const float d = sofuu_arch_f16_to_f32(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t *sc = blk->scales;
        const sofuu_q8_block_ext *xb = x_blocks + b * 8;
        for (int half = 0; half < 2; ++half) {
            for (int part = 0; part < 2; ++part) {
                int l_base = part * 16;
                uint8x16_t ql_lo = vld1q_u8(ql + half * 64 + l_base);
                uint8x16_t ql_hi = vld1q_u8(ql + half * 64 + l_base + 32);
                uint8x16_t qh_v = vld1q_u8(qh + half * 32 + l_base);
                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_lo, vdupq_n_u8(0x0f)), vshlq_n_u8(vandq_u8(qh_v, mask2), 4))), sub32);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_hi, vdupq_n_u8(0x0f)), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), mask2), 4))), sub32);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_lo, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), mask2), 4))), sub32);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_hi, 4), vshlq_n_u8(vshrq_n_u8(qh_v, 6), 4))), sub32);
                const sofuu_q8_block_ext *xg = xb + half * 4;
                int32x4_t a1 = vdotq_s32(vdupq_n_s32(0), q1, vld1q_s8(xg[0].qs + l_base));
                int32x4_t a2 = vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(xg[1].qs + l_base));
                int32x4_t a3 = vdotq_s32(vdupq_n_s32(0), q3, vld1q_s8(xg[2].qs + l_base));
                int32x4_t a4 = vdotq_s32(vdupq_n_s32(0), q4, vld1q_s8(xg[3].qs + l_base));
                const int8_t *hs = sc + half * 8 + part;
                total_sum += (d * (float)hs[0] * xg[0].d) * (float)vaddvq_s32(a1);
                total_sum += (d * (float)hs[2] * xg[1].d) * (float)vaddvq_s32(a2);
                total_sum += (d * (float)hs[4] * xg[2].d) * (float)vaddvq_s32(a3);
                total_sum += (d * (float)hs[6] * xg[3].d) * (float)vaddvq_s32(a4);
            }
        }
    }
    return total_sum;
}

#endif
#endif
