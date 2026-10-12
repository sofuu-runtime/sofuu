#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <arm_neon.h>

#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
};
#pragma pack(pop)

static inline float f16_to_f32_val(uint16_t h) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 fh;
    std::memcpy(&fh, &h, sizeof(h));
    return (float)fh;
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float val;
            std::memcpy(&val, &bits, 4);
            return val;
        }
        uint32_t e = 113;
        uint32_t m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e--;
        }
        uint32_t bits = (sign << 31) | (e << 23) | ((m & 0x3ff) << 13);
        float val;
        std::memcpy(&val, &bits, 4);
        return val;
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | (0xff << 23) | (mant << 13);
        float val;
        std::memcpy(&val, &bits, 4);
        return val;
    }
    uint32_t exp_f32 = exp + 127 - 15;
    uint32_t mant_f32 = mant << 13;
    uint32_t bits = (sign << 31) | (exp_f32 << 23) | mant_f32;
    float val;
    std::memcpy(&val, &bits, 4);
    return val;
#endif
}

static inline uint16_t f32_to_f16_val(float val) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 h = (__fp16)val;
    uint16_t r;
    std::memcpy(&r, &h, sizeof(r));
    return r;
#else
    uint32_t x;
    std::memcpy(&x, &val, 4);
    uint32_t sign = (x >> 31) & 1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
#endif
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// ── Reference Float SIMD (Stage 1.2) ──
float ref_vec_dot_q4_k(const block_q4_K *blocks, const float *x, size_t n_blocks) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        float min = f16_to_f32_val(blk->dmin);
        const uint8_t *q = blk->qs;
        const float *xb = x + b * 256;

        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = min * (float)m;
            get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = min * (float)m;

            float32x4_t vd1 = vdupq_n_f32(d1);
            float32x4_t vm1 = vdupq_n_f32(m1);
            float32x4_t vd2 = vdupq_n_f32(d2);
            float32x4_t vm2 = vdupq_n_f32(m2);

            float32x4_t dot_lo0 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo1 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo2 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo3 = vdupq_n_f32(0.0f);
            float32x4_t sum_x_lo = vdupq_n_f32(0.0f);

            float32x4_t dot_hi0 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi1 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi2 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi3 = vdupq_n_f32(0.0f);
            float32x4_t sum_x_hi = vdupq_n_f32(0.0f);

            for (int chunk = 0; chunk < 2; ++chunk) {
                uint8x16_t raw = vld1q_u8(q + chunk * 16);
                uint8x16_t lo_u8 = vandq_u8(raw, vdupq_n_u8(0x0f));
                uint8x16_t hi_u8 = vshrq_n_u8(raw, 4);

                uint16x8_t lo_16_0 = vmovl_u8(vget_low_u8(lo_u8));
                uint16x8_t lo_16_1 = vmovl_u8(vget_high_u8(lo_u8));

                float32x4_t f_lo_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_0)));
                float32x4_t f_lo_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_0)));
                float32x4_t f_lo_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_1)));
                float32x4_t f_lo_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_1)));

                float32x4_t x_lo_0 = vld1q_f32(xb + chunk * 16 + 0);
                float32x4_t x_lo_1 = vld1q_f32(xb + chunk * 16 + 4);
                float32x4_t x_lo_2 = vld1q_f32(xb + chunk * 16 + 8);
                float32x4_t x_lo_3 = vld1q_f32(xb + chunk * 16 + 12);

                dot_lo0 = vfmaq_f32(dot_lo0, f_lo_0, x_lo_0);
                dot_lo1 = vfmaq_f32(dot_lo1, f_lo_1, x_lo_1);
                dot_lo2 = vfmaq_f32(dot_lo2, f_lo_2, x_lo_2);
                dot_lo3 = vfmaq_f32(dot_lo3, f_lo_3, x_lo_3);

                sum_x_lo = vaddq_f32(sum_x_lo, vaddq_f32(vaddq_f32(x_lo_0, x_lo_1), vaddq_f32(x_lo_2, x_lo_3)));

                uint16x8_t hi_16_0 = vmovl_u8(vget_low_u8(hi_u8));
                uint16x8_t hi_16_1 = vmovl_u8(vget_high_u8(hi_u8));

                float32x4_t f_hi_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_0)));
                float32x4_t f_hi_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_0)));
                float32x4_t f_hi_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_1)));
                float32x4_t f_hi_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_1)));

                float32x4_t x_hi_0 = vld1q_f32(xb + 32 + chunk * 16 + 0);
                float32x4_t x_hi_1 = vld1q_f32(xb + 32 + chunk * 16 + 4);
                float32x4_t x_hi_2 = vld1q_f32(xb + 32 + chunk * 16 + 8);
                float32x4_t x_hi_3 = vld1q_f32(xb + 32 + chunk * 16 + 12);

                dot_hi0 = vfmaq_f32(dot_hi0, f_hi_0, x_hi_0);
                dot_hi1 = vfmaq_f32(dot_hi1, f_hi_1, x_hi_1);
                dot_hi2 = vfmaq_f32(dot_hi2, f_hi_2, x_hi_2);
                dot_hi3 = vfmaq_f32(dot_hi3, f_hi_3, x_hi_3);

                sum_x_hi = vaddq_f32(sum_x_hi, vaddq_f32(vaddq_f32(x_hi_0, x_hi_1), vaddq_f32(x_hi_2, x_hi_3)));
            }

            float32x4_t total_dot_lo = vaddq_f32(vaddq_f32(dot_lo0, dot_lo1), vaddq_f32(dot_lo2, dot_lo3));
            float32x4_t total_dot_hi = vaddq_f32(vaddq_f32(dot_hi0, dot_hi1), vaddq_f32(dot_hi2, dot_hi3));

            acc0 = vfmaq_f32(acc0, total_dot_lo, vd1);
            acc1 = vfmsq_f32(acc1, sum_x_lo, vm1);
            acc2 = vfmaq_f32(acc2, total_dot_hi, vd2);
            acc3 = vfmsq_f32(acc3, sum_x_hi, vm2);

            q += 32;
            xb += 64;
            is += 2;
        }
    }
    float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(total);
}

#include "../src/simd/arch/arm_cortex_a76.h"

int main() {
    std::cout << "Testing Kernel on A76...\n";
    constexpr size_t N_BLOCKS = 4; // 1024 elements (dim of Qwen 0.6B)
    std::vector<block_q4_K> blocks(N_BLOCKS);
    std::vector<float> x(N_BLOCKS * 256);

    // Initialize deterministic test data
    for (size_t b = 0; b < N_BLOCKS; ++b) {
        blocks[b].d = f32_to_f16_val(0.05f * (b + 1));
        blocks[b].dmin = f32_to_f16_val(0.02f * (b + 1));
        for (int i = 0; i < 12; ++i) blocks[b].scales[i] = (uint8_t)(10 + (i * 3) % 40);
        for (int i = 0; i < 128; ++i) blocks[b].qs[i] = (uint8_t)(i * 17);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(float(i) * 0.1f) * 1.5f;
    }

    float ref_val = ref_vec_dot_q4_k(blocks.data(), x.data(), N_BLOCKS);
    std::cout << "Reference float dot product: " << ref_val << "\n";

    // Now test DotProd
    alignas(16) sofuu_q8_block_ext x_q8[N_BLOCKS * 8];
    sofuu_quantize_row_q8_a76(x.data(), x_q8, N_BLOCKS * 8);

    float dp_val = sofuu_vec_dot_q4_k_dotprod_a76(blocks.data(), x_q8, N_BLOCKS);
    std::cout << "DotProd dot product:        " << dp_val << "\n";
    std::cout << "Relative error:             " << std::abs(dp_val - ref_val) / std::abs(ref_val) << "\n";

    // Benchmark 20,000 runs
    constexpr int ITERS = 20000;
    volatile float dummy = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        dummy += ref_vec_dot_q4_k(blocks.data(), x.data(), N_BLOCKS);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ref = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        dummy += sofuu_vec_dot_q4_k_dotprod_a76(blocks.data(), x_q8, N_BLOCKS);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double dt_dp = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Now test Q8_0
    std::cout << "\n--- Testing Q8_0 DotProd ---\n";
    constexpr size_t N_BLOCKS_Q8 = 32; // 1024 elements
    std::vector<block_q8_0> blocks_q8(N_BLOCKS_Q8);
    for (size_t b = 0; b < N_BLOCKS_Q8; ++b) {
        blocks_q8[b].d = f32_to_f16_val(0.01f * (b + 1));
        for (int i = 0; i < 32; ++i) blocks_q8[b].qs[i] = (int8_t)((i * 7) % 255 - 128);
    }

    float ref_q8 = 0.0f;
    for (size_t b = 0; b < N_BLOCKS_Q8; ++b) {
        float d = f16_to_f32_val(blocks_q8[b].d);
        const float *xb = x.data() + b * 32;
        float blk_sum = 0.0f;
        for (int j = 0; j < 32; ++j) blk_sum += ((float)blocks_q8[b].qs[j]) * xb[j];
        ref_q8 += blk_sum * d;
    }
    std::cout << "Reference float Q8_0 dot product: " << ref_q8 << "\n";

    float dp_q8 = sofuu_vec_dot_q8_0_dotprod_a76(blocks_q8.data(), x_q8, N_BLOCKS_Q8);
    std::cout << "DotProd Q8_0 dot product:        " << dp_q8 << "\n";
    std::cout << "Relative error Q8_0:             " << std::abs(dp_q8 - ref_q8) / std::abs(ref_q8) << "\n";

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        dummy += sofuu_vec_dot_q8_0_dotprod_a76(blocks_q8.data(), x_q8, N_BLOCKS_Q8);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double dt_q8 = std::chrono::duration<double, std::milli>(t5 - t4).count();
    std::cout << "DotProd Q8_0 time for " << ITERS << " calls: " << dt_q8 << " ms (" << (dt_q8 / ITERS * 1000.0) << " us/call)\n";

    std::cout << "Final dummy: " << dummy << "\n";

    return 0;
}
