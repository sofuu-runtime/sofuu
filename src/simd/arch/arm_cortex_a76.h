#ifndef SOFUU_ARCH_ARM_CORTEX_A76_H
#define SOFUU_ARCH_ARM_CORTEX_A76_H

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>

// ── ARMv8.2-A DotProd Hardware Acceleration for Cortex-A76 / Apple Silicon ──
// Process 4 integer dot products per clock cycle per NEON execution port.

struct alignas(16) sofuu_q8_block_ext {
    uint16_t d;          // fp16 scale
    int16_t  sum_x;      // sum of qs[0..31] for zero-overhead min subtraction
    int8_t   qs[32];     // 32 signed 8-bit quantized values
};

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
        y[b].d = f32_to_f16_val(d);

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

static inline float sofuu_vec_dot_q8_0_dotprod_a76(const block_q8_0 *w, const sofuu_q8_block_ext *x, size_t n_blocks) {
    float total = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        #if defined(__aarch64__)
        __builtin_prefetch(&w[b + 1], 0, 0);
        #endif
        int8x16_t w0 = vld1q_s8(w[b].qs + 0);
        int8x16_t w1 = vld1q_s8(w[b].qs + 16);
        int8x16_t x0 = vld1q_s8(x[b].qs + 0);
        int8x16_t x1 = vld1q_s8(x[b].qs + 16);

        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, x0);
        acc = vdotq_s32(acc, w1, x1);

        float dw = f16_to_f32_val(w[b].d);
        float dx = f16_to_f32_val(x[b].d);
        total += (float)vaddvq_s32(acc) * (dw * dx);
    }
    return total;
}

static inline float sofuu_vec_dot_q4_k_dotprod_a76(const block_q4_K *blocks, const sofuu_q8_block_ext *x_blocks, size_t n_blocks) {
    float total_sum = 0.0f;

    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_K *blk = &blocks[b];
        #if defined(__aarch64__)
        __builtin_prefetch(&blocks[b + 1], 0, 0);
        #endif
        float d = f16_to_f32_val(blk->d);
        float min = f16_to_f32_val(blk->dmin);
        const uint8_t *q = blk->qs;
        const sofuu_q8_block_ext *xb = x_blocks + b * 8; // 8 blocks of 32 = 256

        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = min * (float)m;
            get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = min * (float)m;

            // Group 1 (lower nibbles, 32 elements)
            uint8x16_t raw0 = vld1q_u8(q + 0);
            uint8x16_t raw1 = vld1q_u8(q + 16);

            int8x16_t q_lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, vdupq_n_u8(0x0f)));
            int8x16_t q_lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, vdupq_n_u8(0x0f)));

            int8x16_t x_lo0 = vld1q_s8(xb[0].qs + 0);
            int8x16_t x_lo1 = vld1q_s8(xb[0].qs + 16);

            int32x4_t acc_lo = vdotq_s32(vdupq_n_s32(0), q_lo0, x_lo0);
            acc_lo = vdotq_s32(acc_lo, q_lo1, x_lo1);

            float dx1 = f16_to_f32_val(xb[0].d);
            int32_t sum_lo = vaddvq_s32(acc_lo);
            total_sum += (d1 * dx1) * (float)sum_lo - (m1 * dx1) * (float)xb[0].sum_x;

            // Group 2 (upper nibbles, 32 elements)
            int8x16_t q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            int8x16_t q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));

            int8x16_t x_hi0 = vld1q_s8(xb[1].qs + 0);
            int8x16_t x_hi1 = vld1q_s8(xb[1].qs + 16);

            int32x4_t acc_hi = vdotq_s32(vdupq_n_s32(0), q_hi0, x_hi0);
            acc_hi = vdotq_s32(acc_hi, q_hi1, x_hi1);

            float dx2 = f16_to_f32_val(xb[1].d);
            int32_t sum_hi = vaddvq_s32(acc_hi);
            total_sum += (d2 * dx2) * (float)sum_hi - (m2 * dx2) * (float)xb[1].sum_x;

            q += 32;
            xb += 2;
            is += 2;
        }
    }
    return total_sum;
}

#endif // defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

#endif // SOFUU_ARCH_ARM_CORTEX_A76_H
