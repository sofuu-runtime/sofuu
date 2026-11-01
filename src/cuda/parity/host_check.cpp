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
    w.assign((size_t)rows * (size_t)nb * 210, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            sofuu_ref::BlockQ6_K *blk = reinterpret_cast<sofuu_ref::BlockQ6_K *>(
                w.data() + ((size_t)r * (size_t)nb + (size_t)b) * 210);
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

void test_q6_k(uint32_t &seed) {
    const int cols = 2048, rows = 4;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 210;
        float got = sofuu_host_q6_k_dot(row, x.data(), cols);
        check_dot("gemv_q6_k", got, sofuu_ref::dot_q6_k(row, x.data(), cols));
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
    sofuu_host_kv_write(got_k.data(), got_v.data(), k.data(), v.data(), pos,
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

} // namespace

int main(void) {
    uint32_t seed = 0x5eed1234u;
    std::printf("Sofuu CUDA host kernel parity (SOFUU_CUDA_HOST)\n");
    test_q8_0(seed);
    test_q4_k(seed);
    test_q6_k(seed);
    test_zero_scale(seed);
    test_rmsnorm(seed);
    test_swiglu(seed);
    test_rope(seed);
    test_kv_write(seed);
    if (g_failures == 0) {
        std::printf("RESULT: PASS (all host kernel references match)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
