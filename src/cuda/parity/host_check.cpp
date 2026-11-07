// src/cuda/parity/host_check.cpp — CPU algorithm-parity gate for the CUDA
// kernels (PLAN-GPU-CUDA C1).
//
// Compiles src/cuda/kernels.cu in SOFUU_CUDA_HOST mode and compares the exact
// per-value math the GPU kernels call against the independent references in
// cpu_ref.h. This runs on any machine (no NVIDIA driver needed) and catches
// block-layout / nibble-order / scale-expansion mistakes before a GPU pass.
//
// Build/run: bash scripts/cuda/check_kernels.sh

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cpu_ref.h"
#include "../kernels.cu" // SOFUU_CUDA_HOST is defined by the build script

namespace {

int g_failures = 0;

void report(const char *name, bool ok, double err, double bound,
            double reference) {
    std::printf("[%s] %-18s err=%.3e bound=%.3e ref=%.9g\n",
                ok ? "PASS" : "FAIL", name, err, bound, reference);
    if (!ok) ++g_failures;
}

void fill_bytes(uint8_t *p, size_t n, uint32_t &seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = (uint8_t)(sofuu_ref::lcg_next(seed) >> 24);
}

std::vector<float> random_vec(int n, uint32_t &seed, float lo, float hi) {
    std::vector<float> v((size_t)n);
    for (int i = 0; i < n; ++i) v[(size_t)i] = sofuu_ref::lcg_float(seed, lo, hi);
    return v;
}

void fill_q8_0(std::vector<uint8_t> &w, int rows, int cols, uint32_t &seed) {
    int nb = cols / 32;
    w.assign((size_t)rows * (size_t)nb * 34, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            sofuu_ref::BlockQ8_0 *blk = reinterpret_cast<sofuu_ref::BlockQ8_0 *>(
                w.data() + ((size_t)r * (size_t)nb + (size_t)b) * 34);
            blk->d = sofuu_ref::F16_0625;
            fill_bytes(reinterpret_cast<uint8_t *>(blk->qs), 32, seed);
        }
}

void fill_q4_k(std::vector<uint8_t> &w, int rows, int cols, uint32_t &seed) {
    int nb = cols / 256;
    w.assign((size_t)rows * (size_t)nb * 144, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            sofuu_ref::BlockQ4_K *blk = reinterpret_cast<sofuu_ref::BlockQ4_K *>(
                w.data() + ((size_t)r * (size_t)nb + (size_t)b) * 144);
            blk->d = sofuu_ref::F16_0625;
            blk->dmin = sofuu_ref::F16_2_POW_M10;
            fill_bytes(blk->scales, 12, seed);
            fill_bytes(blk->qs, 128, seed);
        }
}

void fill_q6_k(std::vector<uint8_t> &w, int rows, int cols, uint32_t &seed) {
    int nb = cols / 256;
    // Device layout: 224-byte blocks (see SOFUU_Q6_K_BYTES).
    w.assign((size_t)rows * (size_t)nb * sofuu_ref::Q6K_STRIDE, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            sofuu_ref::BlockQ6_K *blk = reinterpret_cast<sofuu_ref::BlockQ6_K *>(
                w.data() + ((size_t)r * (size_t)nb + (size_t)b) *
                               sofuu_ref::Q6K_STRIDE);
            blk->d = sofuu_ref::F16_0625;
            fill_bytes(blk->ql, 128, seed);
            fill_bytes(blk->qh, 64, seed);
            fill_bytes(reinterpret_cast<uint8_t *>(blk->scales), 16, seed);
        }
}

// The kernel returns fp32; compare against the high-precision reference with
// a summation-order budget relative to L1 (not to the result).
void check_dot(const char *name, float got, const sofuu_ref::Dot &want) {
    double err = std::fabs((double)got - want.value);
    double bound = 1e-5 * (want.l1 > 1.0 ? want.l1 : 1.0) + 1e-6;
    report(name, err <= bound, err, bound, want.value);
}

void check_vec(const char *name, const float *got, const float *want, int n,
               double tol, double abs_floor) {
    double err = 0.0, worst = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::fabs((double)got[i] - (double)want[i]);
        if (e > err) {
            err = e;
            worst = (double)want[i];
        }
    }
    double bound = tol * std::fabs(worst) + abs_floor;
    report(name, err <= bound, err, bound, worst);
}

void test_q8_0(uint32_t &seed) {
    const int cols = 1024, rows = 4;
    std::vector<uint8_t> w;
    fill_q8_0(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 32) * 34;
        float got = sofuu_host_q8_0_dot(row, x.data(), cols);
        check_dot("gemv_q8_0", got, sofuu_ref::dot_q8_0(row, x.data(), cols));
    }
}

void test_q4_k(uint32_t &seed) {
    const int cols = 2048, rows = 4;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 144;
        float got = sofuu_host_q4_k_dot(row, x.data(), cols);
        check_dot("gemv_q4_k", got, sofuu_ref::dot_q4_k(row, x.data(), cols));
    }
}

// q4_K against q8_1 activations: the dp4a algebra must reproduce the exact
// dot on the SAME quantized values (xd*xq). The reference dequantizes the
// activation from q8 and accumulates in double.
void test_q4_k_q8(uint32_t &seed) {
    const int cols = 2048, rows = 4;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<signed char> xq(cols);
    std::vector<float> xd(cols / 32);
    std::vector<int> xs(cols / 32);
    std::vector<int> xs16(cols / 16);
    sofuu_host_quantize_q8_1(x.data(), xq.data(), xd.data(), xs.data(),
                             xs16.data(), cols);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 144;
        float got = sofuu_host_q4_k_dot_q8(row, xq.data(), xd.data(), xs.data(), cols);
        check_dot("q4_k_x_q8", got,
                  sofuu_ref::dot_q4_k_q8(row, xq.data(), xd.data(), xs.data(), cols));
    }
}

void test_q6_k(uint32_t &seed) {
    const int cols = 2048, rows = 4;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row =
            w.data() + (size_t)r * (cols / 256) * sofuu_ref::Q6K_STRIDE;
        float got = sofuu_host_q6_k_dot(row, x.data(), cols);
        check_dot("gemv_q6_k", got, sofuu_ref::dot_q6_k(row, x.data(), cols));
    }
}

// q6_K against q8_1 activations: the same three-way check as q4_K, plus the
// per-16 sums the 6-bit bias correction depends on. The kernel enters the
// 6-bit values biased (+32) and removes 32 * sum(x) per 16-value half, so a
// half-sum that is off by one shifts the whole half's contribution.
void test_q6_k_q8(uint32_t &seed) {
    const int cols = 2048, rows = 4;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<signed char> xq(cols);
    std::vector<float> xd(cols / 32);
    std::vector<int> xs(cols / 32), xs16(cols / 16);
    sofuu_host_quantize_q8_1(x.data(), xq.data(), xd.data(), xs.data(),
                             xs16.data(), cols);
    int bad = 0;
    for (int b = 0; b < cols / 32; ++b)
        if (xs16[b * 2] + xs16[b * 2 + 1] != xs[b]) ++bad;
    if (bad) std::printf("      half-sum mismatch in %d blocks\n", bad);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row =
            w.data() + (size_t)r * (cols / 256) * sofuu_ref::Q6K_STRIDE;
        float got =
            sofuu_host_q6_k_dot_q8(row, xq.data(), xd.data(), xs16.data(), cols);
        check_dot("q6_k_x_q8", got,
                  sofuu_ref::dot_q6_k_q8(row, xq.data(), xd.data(), cols));
    }
}

void test_zero_scale(uint32_t &seed) {
    // A zero f16 scale must produce an exact zero, never NaN/Inf.
    const int cols = 512;
    const int rows = 1;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    for (int b = 0; b < cols / 256; ++b) {
        sofuu_ref::BlockQ4_K *blk =
            reinterpret_cast<sofuu_ref::BlockQ4_K *>(w.data() + (size_t)b * 144);
        blk->d = 0;
        blk->dmin = 0;
    }
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    float got = sofuu_host_q4_k_dot(w.data(), x.data(), cols);
    double bound = 1e-30;
    report("zero_scale_q4_k", std::fabs((double)got) <= bound,
           std::fabs((double)got), bound, 0.0);
}

void test_rmsnorm(uint32_t &seed) {
    const int n = 1024;
    std::vector<float> x = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> w = random_vec(n, seed, 0.5f, 1.5f);
    std::vector<float> got((size_t)n), want((size_t)n);
    sofuu_host_rmsnorm(x.data(), w.data(), got.data(), n, 1e-6f);
    sofuu_ref::rmsnorm_ref(x.data(), w.data(), want.data(), n, 1e-6f);
    check_vec("rmsnorm", got.data(), want.data(), n, 1e-6, 1e-6);
}

void test_swiglu(uint32_t &seed) {
    const int n = 4096;
    std::vector<float> g = random_vec(n, seed, -6.0f, 6.0f);
    std::vector<float> u = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> got((size_t)n), want((size_t)n);
    sofuu_host_swiglu(g.data(), u.data(), got.data(), n);
    sofuu_ref::swiglu_ref(g.data(), u.data(), want.data(), n);
    check_vec("swiglu", got.data(), want.data(), n, 1e-6, 1e-6);
}

void test_rope(uint32_t &seed) {
    const int n_heads = 4, head_dim = 64, n_rot = 32;
    const int n = n_heads * head_dim;
    std::vector<float> cos_t(16), sin_t(16);
    for (int i = 0; i < 16; ++i) {
        float a = 0.1f * (float)i;
        cos_t[(size_t)i] = std::cos(a);
        sin_t[(size_t)i] = std::sin(a);
    }
    for (int interleaved = 0; interleaved <= 1; ++interleaved) {
        std::vector<float> got = random_vec(n, seed, -1.0f, 1.0f);
        std::vector<float> want = got;
        sofuu_host_rope(got.data(), cos_t.data(), sin_t.data(), n_heads,
                        head_dim, n_rot, interleaved);
        sofuu_ref::rope_ref(want.data(), cos_t.data(), sin_t.data(), n_heads,
                            head_dim, n_rot, interleaved);
        check_vec(interleaved ? "rope_interleaved" : "rope_split", got.data(),
                  want.data(), n, 1e-6, 1e-6);
    }
}

void test_kv_write(uint32_t &seed) {
    const int n_kv = 8, hd = 64, cap = 16, pos = 3;
    const int cache_n = n_kv * cap * hd;
    std::vector<float> got_k((size_t)cache_n, -99.0f), got_v((size_t)cache_n, -99.0f);
    std::vector<float> want_k = got_k, want_v = got_v;
    std::vector<float> k = random_vec(n_kv * hd, seed, -1.0f, 1.0f);
    std::vector<float> v = random_vec(n_kv * hd, seed, -1.0f, 1.0f);
    int pos_dev = pos;
    sofuu_host_kv_write(got_k.data(), got_v.data(), k.data(), v.data(), &pos_dev,
                        n_kv, hd, cap);
    sofuu_ref::kv_write_ref(want_k.data(), want_v.data(), k.data(), v.data(),
                            pos, n_kv, hd, cap);
    double err = 0.0;
    for (int i = 0; i < cache_n; ++i) {
        err = std::fmax(err, std::fabs((double)got_k[(size_t)i] - want_k[(size_t)i]));
        err = std::fmax(err, std::fabs((double)got_v[(size_t)i] - want_v[(size_t)i]));
    }
    report("kv_write", err == 0.0, err, 0.0, 0.0);
}

void test_attn_decode(uint32_t &seed) {
    const int n_heads = 8, n_kv = 2, hd = 64, cap = 32, seq = 17;
    const int cache_n = n_kv * cap * hd;
    std::vector<float> q = random_vec(n_heads * hd, seed, -1.0f, 1.0f);
    std::vector<float> k = random_vec(cache_n, seed, -1.0f, 1.0f);
    std::vector<float> v = random_vec(cache_n, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)(n_heads * hd)), want((size_t)(n_heads * hd));
    float scale = 1.0f / std::sqrt((float)hd);
    int seq_dev = seq;
    sofuu_host_attn_decode(q.data(), k.data(), v.data(), got.data(), n_heads,
                           n_kv, hd, &seq_dev, cap, scale);
    sofuu_ref::attn_decode_ref(q.data(), k.data(), v.data(), want.data(),
                               n_heads, n_kv, hd, seq, cap, scale);
    check_vec("attn_decode", got.data(), want.data(), n_heads * hd, 2e-5, 2e-6);
}

void test_add(uint32_t &seed) {
    const int n = 4096;
    std::vector<float> a = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> b = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> got((size_t)n), want((size_t)n);
    sofuu_host_add(a.data(), b.data(), got.data(), n);
    sofuu_ref::add_ref(a.data(), b.data(), want.data(), n);
    check_vec("add", got.data(), want.data(), n, 0.0, 0.0);
}


// The GPU GEMV kernels were rewritten from one-thread-per-block to a
// coalesced lane mapping. These loops mirror that mapping exactly (including
// the warp stride over super-blocks) and compare against the independent
// scalar references, so a coverage/indexing bug is caught without a GPU.
void test_tiled_gemv(uint32_t &seed) {
    const int rows = 4;
    for (int cols : {2048, 8192}) {
        int nb = cols / 256;
        std::vector<uint8_t> w4, w6;
        fill_q4_k(w4, rows, cols, seed);
        fill_q6_k(w6, rows, cols, seed);
        std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
        for (int r = 0; r < rows; ++r) {
            float got4 = 0.0f, got6 = 0.0f;
            const uint8_t *row4 = w4.data() + (size_t)r * nb * 144;
            const uint8_t *row6 = w6.data() + (size_t)r * nb * sofuu_ref::Q6K_STRIDE;
            // One warp owns this row and walks every super-block; the block's
            // other seven warps are other rows and share x through L1.
            for (int lane = 0; lane < 32; ++lane) {
                int sub = lane >> 2, l0 = (lane & 3) << 3;
                for (int bi = 0; bi < nb; ++bi) {
                    const auto *b4 = reinterpret_cast<const sofuu_block_q4_K *>(
                        row4 + (size_t)bi * 144);
                    unsigned int sc, mn;
                    sofuu_q4k_scale_min(sub, b4->scales, &sc, &mn);
                    float d4 = sofuu_f16_to_f32(b4->d);
                    float dm = sofuu_f16_to_f32(b4->dmin);
                    const uint8_t *qs4 = b4->qs + (sub >> 1) * 32;
                    unsigned int hi = (unsigned int)(sub & 1);
                    float s4 = d4 * (float)sc, m4 = dm * (float)mn;
                    const float *xb4 = x.data() + (size_t)bi * 256 + sub * 32 + l0;
                    for (int k = 0; k < 8; ++k) {
                        unsigned int byte = qs4[l0 + k];
                        unsigned int q = hi ? (byte >> 4) : (byte & 0x0fu);
                        got4 += (s4 * (float)q - m4) * xb4[k];
                    }

                    const auto *b6 = reinterpret_cast<const sofuu_block_q6_K *>(
                        row6 + (size_t)bi * sofuu_ref::Q6K_STRIDE);
                    int n = sub >> 2, rr = sub & 3;
                    float d6 = sofuu_f16_to_f32(b6->d);
                    int sb = n * 8 + 2 * rr;
                    float sc0 = (float)b6->scales[sb];
                    float sc1 = (float)b6->scales[sb + 1];
                    int ql_base = n * 64 + ((rr == 1 || rr == 3) ? 32 : 0);
                    const float *xb6 = x.data() + (size_t)bi * 256 + sub * 32 + l0;
                    for (int k = 0; k < 8; ++k) {
                        int l = l0 + k;
                        unsigned int ql = b6->ql[ql_base + l];
                        unsigned int nib = (rr < 2) ? (ql & 0x0fu) : (ql >> 4);
                        unsigned int hi2 = ((unsigned int)b6->qh[n * 32 + l] >> (2 * rr)) & 3u;
                        int q = (int)(nib | (hi2 << 4)) - 32;
                        float sc = (l >= 16) ? sc1 : sc0;
                        got6 += d6 * sc * (float)q * xb6[k];
                    }
                }
            }
            check_dot(cols == 2048 ? "tiled_q4_k" : "tiled_q4_k_wide",
                      got4, sofuu_ref::dot_q4_k(row4, x.data(), cols));
            check_dot(cols == 2048 ? "tiled_q6_k" : "tiled_q6_k_wide",
                      got6, sofuu_ref::dot_q6_k(row6, x.data(), cols));
        }
    }

    for (int cols : {1024, 4096}) {
        int nb = cols / 32;
        std::vector<uint8_t> w;
        fill_q8_0(w, rows, cols, seed);
        std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
        for (int r = 0; r < rows; ++r) {
            float got = 0.0f;
            const uint8_t *row = w.data() + (size_t)r * nb * 34;
            for (int lane = 0; lane < 32; ++lane) {
                int lane_in_block = lane & 3;
                for (int bi = lane >> 2; bi < nb; bi += 8) {
                    const auto *b = reinterpret_cast<const sofuu_block_q8_0 *>(
                        row + (size_t)bi * 34);
                    float d = sofuu_f16_to_f32(b->d);
                    const float *xb =
                        x.data() + (size_t)bi * 32 + lane_in_block * 8;
                    for (int k = 0; k < 8; ++k)
                        got += d * (float)b->qs[lane_in_block * 8 + k] * xb[k];
                }
            }
            check_dot(cols == 1024 ? "tiled_q8_0" : "tiled_q8_0_wide",
                      got, sofuu_ref::dot_q8_0(row, x.data(), cols));
        }
    }
}


// The batched prefill kernels give one warp one (token, row) dot: lane l owns
// eight consecutive values at cg + l*8. These loops mirror that indexing for
// eight tokens at once, which is where a token/row/column mix-up would live.
void check_dots(const char *name, const std::vector<float> &got,
                const std::vector<sofuu_ref::Dot> &want) {
    double worst_ratio = 0.0, worst_err = 0.0, worst_bound = 0.0;
    double worst_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double err = std::fabs((double)got[i] - want[i].value);
        double bound = 1e-5 * (want[i].l1 > 1.0 ? want[i].l1 : 1.0) + 1e-6;
        double ratio = err / bound;
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_err = err;
            worst_bound = bound;
            worst_ref = want[i].value;
        }
    }
    report(name, worst_ratio <= 1.0, worst_err, worst_bound, worst_ref);
}

void test_b8_gemm(uint32_t &seed) {
    const int rows = 20, tokens = 8;
    for (int cols : {2048, 8192}) {
        int nb = cols / 256;
        std::vector<uint8_t> w4, w6;
        fill_q4_k(w4, rows, cols, seed);
        fill_q6_k(w6, rows, cols, seed);
        std::vector<float> x = random_vec(tokens * cols, seed, -1.0f, 1.0f);
        std::vector<float> got4((size_t)tokens * rows, 0.0f);
        std::vector<float> got6((size_t)tokens * rows, 0.0f);
        std::vector<sofuu_ref::Dot> want4, want6;
        want4.reserve((size_t)tokens * rows);
        want6.reserve((size_t)tokens * rows);
        for (int t = 0; t < tokens; ++t) {
            const float *xrow = x.data() + (size_t)t * cols;
            for (int row = 0; row < rows; ++row) {
                const uint8_t *row4 = w4.data() + (size_t)row * nb * 144;
                const uint8_t *row6 = w6.data() + (size_t)row * nb * sofuu_ref::Q6K_STRIDE;
                float a4 = 0.0f, a6 = 0.0f;
                for (int cg = 0; cg < cols; cg += 256) {
                    int bi = cg >> 8;
                    const auto *b4 = reinterpret_cast<const sofuu_block_q4_K *>(
                        row4 + (size_t)bi * 144);
                    const auto *b6 = reinterpret_cast<const sofuu_block_q6_K *>(
                        row6 + (size_t)bi * sofuu_ref::Q6K_STRIDE);
                    for (int lane = 0; lane < 32; ++lane) {
                        int sub = lane >> 2, l0 = (lane & 3) << 3;
                        unsigned int sc, mn;
                        sofuu_q4k_scale_min(sub, b4->scales, &sc, &mn);
                        float d4 = sofuu_f16_to_f32(b4->d);
                        float dm = sofuu_f16_to_f32(b4->dmin);
                        float s4 = d4 * (float)sc, m4 = dm * (float)mn;
                        const uint8_t *qs4 = b4->qs + (sub >> 1) * 32;
                        unsigned int hi = (unsigned int)(sub & 1);
                        int n6 = sub >> 2, rr = sub & 3;
                        float d6 = sofuu_f16_to_f32(b6->d);
                        int sb = n6 * 8 + 2 * rr;
                        float sc0 = (float)b6->scales[sb];
                        float sc1 = (float)b6->scales[sb + 1];
                        int ql_base = n6 * 64 + ((rr == 1 || rr == 3) ? 32 : 0);
                        for (int k = 0; k < 8; ++k) {
                            int col = cg + lane * 8 + k;
                            float xv = xrow[col];
                            unsigned int byte = qs4[l0 + k];
                            unsigned int q4v = hi ? (byte >> 4) : (byte & 0x0fu);
                            a4 += (s4 * (float)q4v - m4) * xv;
                            int l = l0 + k;
                            unsigned int ql = b6->ql[ql_base + l];
                            unsigned int nib = (rr < 2) ? (ql & 0x0fu) : (ql >> 4);
                            unsigned int hi2 =
                                ((unsigned int)b6->qh[n6 * 32 + l] >> (2 * rr)) & 3u;
                            int q6v = (int)(nib | (hi2 << 4)) - 32;
                            float sc6 = (l >= 16) ? sc1 : sc0;
                            a6 += d6 * sc6 * (float)q6v * xv;
                        }
                    }
                }
                got4[(size_t)t * rows + row] = a4;
                got6[(size_t)t * rows + row] = a6;
                want4.push_back(sofuu_ref::dot_q4_k(row4, xrow, cols));
                want6.push_back(sofuu_ref::dot_q6_k(row6, xrow, cols));
            }
        }
        check_dots(cols == 2048 ? "b8_q4_k" : "b8_q4_k_wide", got4, want4);
        check_dots(cols == 2048 ? "b8_q6_k" : "b8_q6_k_wide", got6, want6);
    }

    for (int cols : {1024, 4096}) {
        int nb = cols / 32;
        std::vector<uint8_t> w;
        fill_q8_0(w, rows, cols, seed);
        std::vector<float> x = random_vec(tokens * cols, seed, -1.0f, 1.0f);
        std::vector<float> got((size_t)tokens * rows, 0.0f);
        std::vector<sofuu_ref::Dot> want;
        want.reserve((size_t)tokens * rows);
        for (int t = 0; t < tokens; ++t) {
            const float *xrow = x.data() + (size_t)t * cols;
            for (int row = 0; row < rows; ++row) {
                const uint8_t *roww = w.data() + (size_t)row * nb * 34;
                float acc = 0.0f;
                for (int cg = 0; cg < cols; cg += 256) {
                    for (int lane = 0; lane < 32; ++lane) {
                        int bi = (cg >> 5) + (lane >> 2);
                        int lane_in_block = lane & 3;
                        const auto *b = reinterpret_cast<const sofuu_block_q8_0 *>(
                            roww + (size_t)bi * 34);
                        float d = sofuu_f16_to_f32(b->d);
                        for (int k = 0; k < 8; ++k) {
                            int col = cg + lane * 8 + k;
                            acc += d * (float)b->qs[lane_in_block * 8 + k] *
                                   xrow[col];
                        }
                    }
                }
                got[(size_t)t * rows + row] = acc;
                want.push_back(sofuu_ref::dot_q8_0(roww, xrow, cols));
            }
        }
        check_dots(cols == 1024 ? "b8_q8_0" : "b8_q8_0_wide", got, want);
    }

    // f32 has no quant blocks; the same warp/token map with a plain dot.
    {
        const int cols = 2048;
        std::vector<float> w = random_vec(rows * cols, seed, -1.0f, 1.0f);
        std::vector<float> x = random_vec(tokens * cols, seed, -1.0f, 1.0f);
        std::vector<float> got((size_t)tokens * rows, 0.0f);
        std::vector<sofuu_ref::Dot> want;
        want.reserve((size_t)tokens * rows);
        for (int t = 0; t < tokens; ++t) {
            const float *xrow = x.data() + (size_t)t * cols;
            for (int row = 0; row < rows; ++row) {
                const float *roww = w.data() + (size_t)row * cols;
                float acc = 0.0f;
                for (int cg = 0; cg < cols; cg += 256) {
                    for (int lane = 0; lane < 32; ++lane) {
                        for (int k = 0; k < 8; ++k) {
                            int col = cg + lane * 8 + k;
                            acc += roww[col] * xrow[col];
                        }
                    }
                }
                got[(size_t)t * rows + row] = acc;
                double sum = 0.0, l1 = 0.0;
                for (int c = 0; c < cols; ++c) {
                    double p = (double)roww[c] * xrow[c];
                    sum += p;
                    l1 += std::fabs(p);
                }
                want.push_back({sum, l1});
            }
        }
        check_dots("b8_f32", got, want);
    }
}

// Batched elementwise/attention indexing: each loop is the kernel's own
// (token, head) decomposition, checked against the single-row references.
void test_batch_paths(uint32_t &seed) {
    const int tokens = 8, n_heads = 8, n_kv = 2, hd = 64, cap = 32, start = 5;

    // RMSNorm over eight rows.
    {
        const int n = 512;
        std::vector<float> x = random_vec(tokens * n, seed, -2.0f, 2.0f);
        std::vector<float> w = random_vec(n, seed, 0.5f, 1.5f);
        std::vector<float> got((size_t)tokens * n), want((size_t)tokens * n);
        for (int t = 0; t < tokens; ++t) {
            sofuu_ref::rmsnorm_ref(x.data() + (size_t)t * n, w.data(),
                                   want.data() + (size_t)t * n, n, 1e-5f);
            // Mirror the kernel's float accumulation.
            const float *xr = x.data() + (size_t)t * n;
            float *yr = got.data() + (size_t)t * n;
            float ss = 0.0f;
            for (int i = 0; i < n; ++i) ss += xr[i] * xr[i];
            float scale = 1.0f / std::sqrt(ss / (float)n + 1e-5f);
            for (int i = 0; i < n; ++i) yr[i] = xr[i] * scale * w[i];
        }
        check_vec("rmsnorm_batch", got.data(), want.data(), tokens * n, 1e-6, 1e-6);
    }

    // RoPE over a table with one row per token.
    for (int interleaved = 0; interleaved < 2; ++interleaved) {
        const int n_rot = hd;
        std::vector<float> x = random_vec(tokens * n_heads * hd, seed, -1.0f, 1.0f);
        std::vector<float> table = random_vec(tokens * hd, seed, -1.0f, 1.0f);
        std::vector<float> got = x, want = x;
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                size_t off = ((size_t)t * n_heads + h) * hd;
                const float *ct = table.data() + (size_t)t * hd;
                sofuu_ref::rope_ref(want.data() + off, ct, ct + hd / 2, 1, hd,
                                    n_rot, interleaved);
                float *base = got.data() + off;
                for (int i = 0; i < n_rot / 2; ++i) {
                    int i0 = interleaved ? 2 * i : i;
                    int i1 = interleaved ? 2 * i + 1 : i + n_rot / 2;
                    float c = ct[i], s = ct[hd / 2 + i];
                    float x0 = base[i0], x1 = base[i1];
                    base[i0] = x0 * c - x1 * s;
                    base[i1] = x0 * s + x1 * c;
                }
            }
        }
        check_vec(interleaved ? "rope_batch_interleaved" : "rope_batch_split",
                  got.data(), want.data(), tokens * n_heads * hd, 1e-6, 1e-6);
    }

    // KV writes for a chunk.
    {
        std::vector<float> kc((size_t)n_kv * cap * hd, 0.0f);
        std::vector<float> vc((size_t)n_kv * cap * hd, 0.0f);
        std::vector<float> k = random_vec(tokens * n_kv * hd, seed, -1.0f, 1.0f);
        std::vector<float> v = random_vec(tokens * n_kv * hd, seed, -1.0f, 1.0f);
        std::vector<float> got_k = kc, got_v = vc;
        std::vector<float> want_k = kc, want_v = vc;
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < n_kv; ++h) {
                size_t slot = ((size_t)h * cap + start + t) * hd;
                const float *kr = k.data() + ((size_t)t * n_kv + h) * hd;
                const float *vr = v.data() + ((size_t)t * n_kv + h) * hd;
                std::memcpy(got_k.data() + slot, kr, hd * 4);
                std::memcpy(got_v.data() + slot, vr, hd * 4);
            }
            sofuu_ref::kv_write_ref(want_k.data(), want_v.data(),
                                    k.data() + (size_t)t * n_kv * hd,
                                    v.data() + (size_t)t * n_kv * hd, start + t,
                                    n_kv, hd, cap);
        }
        check_vec("kv_write_batch_k", got_k.data(), want_k.data(), n_kv * cap * hd,
                  0.0, 0.0);
        check_vec("kv_write_batch_v", got_v.data(), want_v.data(), n_kv * cap * hd,
                  0.0, 0.0);
    }

    // Causal chunk attention: token t sees rows 0..start+t.
    {
        std::vector<float> q = random_vec(tokens * n_heads * hd, seed, -1.0f, 1.0f);
        std::vector<float> kc = random_vec(n_kv * cap * hd, seed, -1.0f, 1.0f);
        std::vector<float> vc = random_vec(n_kv * cap * hd, seed, -1.0f, 1.0f);
        std::vector<float> got((size_t)tokens * n_heads * hd, 0.0f);
        std::vector<float> want((size_t)tokens * n_heads * hd, 0.0f);
        float scale = 1.0f / std::sqrt((float)hd);
        for (int t = 0; t < tokens; ++t) {
            sofuu_ref::attn_decode_ref(
                q.data() + (size_t)t * n_heads * hd, kc.data(), vc.data(),
                want.data() + (size_t)t * n_heads * hd, n_heads, n_kv, hd,
                start + t + 1, cap, scale);
            // The prefill kernel is the decode math with a token offset.
            int seq_dev = start + t + 1;
            sofuu_host_attn_decode(q.data() + (size_t)t * n_heads * hd,
                                   kc.data(), vc.data(),
                                   got.data() + (size_t)t * n_heads * hd,
                                   n_heads, n_kv, hd, &seq_dev, cap, scale);
        }
        check_vec("attn_prefill", got.data(), want.data(), tokens * n_heads * hd,
                  2e-5, 2e-6);
    }
}

// The q4_K kernel reads eight qs bytes as one little-endian uint64. Verify the
// shift/nibble extraction against the per-byte definition on the host.
void test_q4_vector_extract(uint32_t &seed) {
    std::vector<uint8_t> w;
    fill_q4_k(w, 1, 256, seed);
    const auto *b = reinterpret_cast<const sofuu_block_q4_K *>(w.data());
    int bad = 0;
    for (int sub = 0; sub < 8; ++sub) {
        for (int l0 = 0; l0 < 32; l0 += 8) {
            unsigned long long qv = 0;
            std::memcpy(&qv, b->qs + (sub >> 1) * 32 + l0, 8);
            for (int k = 0; k < 8; ++k) {
                unsigned int byte = (unsigned int)((qv >> (8 * k)) & 0xffu);
                unsigned int q = (sub & 1) ? (byte >> 4) : (byte & 0x0fu);
                unsigned int ref_byte = b->qs[(sub >> 1) * 32 + l0 + k];
                unsigned int ref = (sub & 1) ? (ref_byte >> 4) : (ref_byte & 0x0fu);
                if (q != ref) ++bad;
            }
        }
    }
    report("q4_vector_extract", bad == 0, (double)bad, 0.0, 0.0);
}

} // namespace

int main(void) {
    uint32_t seed = 0x5eed1234u;
    std::printf("Sofuu CUDA host kernel parity (SOFUU_CUDA_HOST)\n");
    test_q8_0(seed);
    test_q4_k(seed);
    test_q4_k_q8(seed);
    test_q6_k(seed);
    test_q6_k_q8(seed);
    test_zero_scale(seed);
    test_rmsnorm(seed);
    test_swiglu(seed);
    test_rope(seed);
    test_kv_write(seed);
    test_attn_decode(seed);
    test_add(seed);
    test_tiled_gemv(seed);
    test_q4_vector_extract(seed);
    test_b8_gemm(seed);
    test_batch_paths(seed);
    if (g_failures == 0) {
        std::printf("RESULT: PASS (all host kernel references match)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
