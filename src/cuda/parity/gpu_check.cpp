// src/cuda/parity/gpu_check.cpp — on-GPU parity gate for the Sofuu CUDA
// kernels (PLAN-GPU-CUDA C1/C2). Runs on Kaggle's Tesla T4.
//
// No CUDA toolkit or libcudart is required: src/cuda/cuda_loader.c dlopens the
// driver, this file loads the checked-in src/cuda/kernels.ptx (sm_70, JITs to
// sm_75 on T4), and every result is compared against the independent scalar
// references in cpu_ref.h.
//
// Build/run: bash scripts/cuda/kaggle_gpu_check.sh
// Exit codes: 0 = all pass, 1 = parity failure, 2 = no CUDA driver/device.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../cuda_loader.h"
#include "cpu_ref.h"

namespace {

int g_failures = 0;
sofuu_cuda_driver *g_drv = nullptr;
sofuu_cuda_ctx *g_ctx = nullptr;

void fail_msg(const char *what) {
    std::fprintf(stderr, "[cuda] %s: %s\n", what, sofuu_cuda_last_error());
}

void report(const char *name, bool ok, double err, double bound,
            double reference, const char *extra = nullptr) {
    std::printf("[%s] %-18s err=%.3e tol=%.3e ref[0]=%.9g%s%s\n",
                ok ? "PASS" : "FAIL", name, err, bound, reference,
                extra ? " " : "", extra ? extra : "");
    if (!ok) ++g_failures;
}

bool alloc_upload(const void *host, size_t bytes, unsigned long long *out) {
    if (sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)bytes, out) != 0) {
        fail_msg("cuMemAlloc");
        return false;
    }
    if (sofuu_cuda_ctx_upload(g_ctx, *out, host, bytes) != 0) {
        fail_msg("cuMemcpyHtoD");
        return false;
    }
    return true;
}

// Allocate a device-only buffer (no host copy). The raw API returns 0 for
// success, so this small wrapper keeps call sites on the same bool convention
// as alloc_upload; `!sofuu_cuda_ctx_alloc(...)` inverted success and made six
// kernels report a bogus allocation failure on the T4 while allocation worked.
bool alloc_only(size_t bytes, unsigned long long *out) {
    if (sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)bytes, out) != 0) {
        fail_msg("cuMemAlloc");
        return false;
    }
    return true;
}

bool download(unsigned long long dev, void *host, size_t bytes) {
    if (sofuu_cuda_ctx_download(g_ctx, host, dev, bytes) != 0) {
        fail_msg("cuMemcpyDtoH");
        return false;
    }
    return true;
}

bool launch(void *fn, unsigned grid, unsigned block,
            std::vector<void *> params) {
    if (sofuu_cuda_ctx_launch(g_ctx, fn, grid, 1, 1, block, 1, 1, 0,
                              params.data()) != 0) {
        fail_msg("cuLaunchKernel");
        return false;
    }
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("cuStreamSynchronize");
        return false;
    }
    return true;
}

// Same launch without the per-call sync; the benchmark batches many launches
// and syncs once, so the number is kernel throughput rather than driver
// round-trip latency.
bool launch_async(void *fn, unsigned grid, unsigned block,
                  std::vector<void *> params) {
    if (sofuu_cuda_ctx_launch(g_ctx, fn, grid, 1, 1, block, 1, 1, 0,
                              params.data()) != 0) {
        fail_msg("cuLaunchKernel");
        return false;
    }
    return true;
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
    // Device layout: 224-byte blocks, so ql/qh/scales are 16-byte aligned.
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

bool get_function(const char *name, void **out) {
    if (sofuu_cuda_ctx_function(g_ctx, name, out) != 0) {
        fail_msg(name);
        return false;
    }
    return true;
}

// ── GEMV tests ─────────────────────────────────────────────────────────────
void test_gemv_q8_0(void *fn, uint32_t &seed, int cols) {
    const int rows = 8;
    std::vector<uint8_t> w;
    fill_q8_0(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)((rows + 7) / 8), 256, params) &&
              download(dy, got.data(), (size_t)rows * sizeof(float));
    double max_err = 0.0, bound = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 32) * 34;
        sofuu_ref::Dot want_d = sofuu_ref::dot_q8_0(row, x.data(), cols);
        want[(size_t)r] = (float)want_d.value;
        double e = std::fabs((double)got[(size_t)r] - want_d.value);
        double b = 1e-5 * (want_d.l1 > 1.0 ? want_d.l1 : 1.0) + 1e-6;
        if (e > max_err) max_err = e;
        if (b > bound) bound = b;
    }
    report(cols == 1024 ? "gemv_q8_0" : "gemv_q8_0_wide", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_q4_k(void *fn, uint32_t &seed, int cols) {
    const int rows = 8;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)((rows + 7) / 8), 256, params) &&
              download(dy, got.data(), (size_t)rows * sizeof(float));
    double max_err = 0.0, bound = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 144;
        sofuu_ref::Dot want_d = sofuu_ref::dot_q4_k(row, x.data(), cols);
        want[(size_t)r] = (float)want_d.value;
        double e = std::fabs((double)got[(size_t)r] - want_d.value);
        double b = 1e-5 * (want_d.l1 > 1.0 ? want_d.l1 : 1.0) + 1e-6;
        if (e > max_err) max_err = e;
        if (b > bound) bound = b;
    }
    report(cols == 2048 ? "gemv_q4_k" : "gemv_q4_k_wide", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_q6_k(void *fn, uint32_t &seed, int cols) {
    const int rows = 8;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)((rows + 7) / 8), 256, params) &&
              download(dy, got.data(), (size_t)rows * sizeof(float));
    double max_err = 0.0, bound = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row =
            w.data() + (size_t)r * (cols / 256) * sofuu_ref::Q6K_STRIDE;
        sofuu_ref::Dot want_d = sofuu_ref::dot_q6_k(row, x.data(), cols);
        want[(size_t)r] = (float)want_d.value;
        double e = std::fabs((double)got[(size_t)r] - want_d.value);
        double b = 1e-5 * (want_d.l1 > 1.0 ? want_d.l1 : 1.0) + 1e-6;
        if (e > max_err) max_err = e;
        if (b > bound) bound = b;
    }
    report(cols == 2048 ? "gemv_q6_k" : "gemv_q6_k_wide", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// q4_K / q6_K against q8_1 activations. Three things are checked:
//   1. the quantize kernel reproduces the host quantizer byte-for-byte
//      (q, per-32 scale, per-32 sum, per-16 sums),
//   2. the dp4a dot matches the exact dot on those same quantized values,
//   3. the quantized dot stays inside the full-precision tolerance.
void test_gemv_q4_k_q8(void *fn_quant, void *fn_gemv, uint32_t &seed) {
    const int rows = 8, cols = 2048;
    const int nblk = cols / 32;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<int8_t> xq(cols);
    std::vector<float> xd(nblk);
    std::vector<int32_t> xs(nblk), xs16(nblk * 2);
    sofuu_ref::quantize_q8_1(x.data(), xq.data(), xd.data(), xs.data(),
                             xs16.data(), cols);

    unsigned long long dw = 0, dx = 0, dq = 0, dd = 0, ds = 0, d16 = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)cols, &dq) ||
        !alloc_only((unsigned long long)nblk * 4, &dd) ||
        !alloc_only((unsigned long long)nblk * 4, &ds) ||
        !alloc_only((unsigned long long)nblk * 8, &d16) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    {
        int n = cols;
        std::vector<void *> params = {&dx, &dq, &dd, &ds, &d16, (void *)&n};
        if (!launch(fn_quant, (unsigned)((nblk + 7) / 8), 256, params)) return;
    }
    std::vector<int8_t> q_dev(cols);
    std::vector<float> d_dev(nblk);
    std::vector<int32_t> s_dev(nblk), s16_dev(nblk * 2);
    if (!download(dq, q_dev.data(), cols) ||
        !download(dd, d_dev.data(), (size_t)nblk * 4) ||
        !download(ds, s_dev.data(), (size_t)nblk * 4) ||
        !download(d16, s16_dev.data(), (size_t)nblk * 8)) {
        return;
    }
    int q_bad = 0, d_bad = 0, s_bad = 0, s16_bad = 0;
    for (int i = 0; i < cols; ++i)
        if (q_dev[(size_t)i] != xq[(size_t)i]) ++q_bad;
    for (int b = 0; b < nblk; ++b) {
        if (d_dev[(size_t)b] != xd[(size_t)b]) ++d_bad;
        if (s_dev[(size_t)b] != xs[(size_t)b]) ++s_bad;
        for (int h = 0; h < 2; ++h)
            if (s16_dev[(size_t)(b * 2 + h)] != xs16[(size_t)(b * 2 + h)])
                ++s16_bad;
    }

    std::vector<float> got((size_t)rows, 0.0f);
    {
        int r = rows, c = cols;
        std::vector<void *> params = {&dw, &dq, &dd, &ds, &dy, (void *)&r,
                                      (void *)&c};
        if (!launch(fn_gemv, (unsigned)((rows + 7) / 8), 256, params) ||
            !download(dy, got.data(), (size_t)rows * sizeof(float))) {
            return;
        }
    }
    double worst = 0.0, bound = 0.0, worst_ref = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 144;
        sofuu_ref::Dot want =
            sofuu_ref::dot_q4_k_q8(row, xq.data(), xd.data(), xs.data(), cols);
        double e = std::fabs((double)got[(size_t)r] - want.value);
        double b = 1e-5 * (want.l1 > 1.0 ? want.l1 : 1.0) + 1e-6;
        if (e > worst) { worst = e; worst_ref = want.value; }
        if (b > bound) bound = b;
    }
    bool ok = q_bad == 0 && d_bad == 0 && s_bad == 0 && s16_bad == 0 &&
              worst <= bound;
    report("gemv_q4_k_x_q8", ok, worst, bound, worst_ref);
    if (q_bad || d_bad || s_bad || s16_bad) {
        std::printf("      quantize mismatch: q=%d d=%d sum=%d sum16=%d\n",
                    q_bad, d_bad, s_bad, s16_bad);
    }
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dq);
    sofuu_cuda_ctx_free(g_ctx, dd);
    sofuu_cuda_ctx_free(g_ctx, ds);
    sofuu_cuda_ctx_free(g_ctx, d16);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_q6_k_q8(void *fn_quant, void *fn_gemv, uint32_t &seed) {
    const int rows = 8, cols = 2048;
    const int nblk = cols / 32;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<int8_t> xq(cols);
    std::vector<float> xd(nblk);
    std::vector<int32_t> xs(nblk), xs16(nblk * 2);
    sofuu_ref::quantize_q8_1(x.data(), xq.data(), xd.data(), xs.data(),
                             xs16.data(), cols);
    unsigned long long dw = 0, dx = 0, dq = 0, dd = 0, ds = 0, d16 = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)cols, &dq) ||
        !alloc_only((unsigned long long)nblk * 4, &dd) ||
        !alloc_only((unsigned long long)nblk * 4, &ds) ||
        !alloc_only((unsigned long long)nblk * 8, &d16) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    {
        int n = cols;
        std::vector<void *> params = {&dx, &dq, &dd, &ds, &d16, (void *)&n};
        if (!launch(fn_quant, (unsigned)((nblk + 7) / 8), 256, params)) return;
    }
    std::vector<float> got((size_t)rows, 0.0f);
    {
        int r = rows, c = cols;
        std::vector<void *> params = {&dw, &dq, &dd, &ds, &d16, &dy, (void *)&r,
                                      (void *)&c};
        if (!launch(fn_gemv, (unsigned)((rows + 7) / 8), 256, params) ||
            !download(dy, got.data(), (size_t)rows * sizeof(float))) {
            return;
        }
    }
    double worst = 0.0, bound = 0.0, worst_ref = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row =
            w.data() + (size_t)r * (cols / 256) * sofuu_ref::Q6K_STRIDE;
        sofuu_ref::Dot want =
            sofuu_ref::dot_q6_k_q8(row, xq.data(), xd.data(), cols);
        double e = std::fabs((double)got[(size_t)r] - want.value);
        double b = 1e-5 * (want.l1 > 1.0 ? want.l1 : 1.0) + 1e-6;
        if (e > worst) { worst = e; worst_ref = want.value; }
        if (b > bound) bound = b;
    }
    report("gemv_q6_k_x_q8", worst <= bound, worst, bound, worst_ref);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dq);
    sofuu_cuda_ctx_free(g_ctx, dd);
    sofuu_cuda_ctx_free(g_ctx, ds);
    sofuu_cuda_ctx_free(g_ctx, d16);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_f32(void *fn, uint32_t &seed) {
    const int rows = 8, cols = 1024;
    std::vector<float> w = random_vec(rows * cols, seed, -1.0f, 1.0f);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);
    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size() * sizeof(float), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)((rows + 7) / 8), 256, params) &&
              download(dy, got.data(), (size_t)rows * sizeof(float));
    double max_err = 0.0, bound = 0.0;
    for (int r = 0; r < rows; ++r) {
        const float *row = w.data() + (size_t)r * (size_t)cols;
        double ref = sofuu_ref::dot_f32(row, x.data(), cols);
        double l1 = 0.0;
        for (int i = 0; i < cols; ++i)
            l1 += std::fabs((double)row[i] * x[(size_t)i]);
        want[(size_t)r] = (float)ref;
        double e = std::fabs((double)got[(size_t)r] - ref);
        double b = 1e-5 * (l1 > 1.0 ? l1 : 1.0) + 1e-6;
        if (e > max_err) max_err = e;
        if (b > bound) bound = b;
    }
    report("gemv_f32", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// ── Batched prefill ───────────────────────────────────────────────────────
// The b8 kernels are one warp per (token, row) dot. Verify every pair, not
// just row 0, because a token/row mix-up would still produce plausible values.
void test_gemm_b8(void *fn, uint32_t &seed, int kind, int cols) {
    const int rows = 20, tokens = 8;
    std::vector<uint8_t> wbytes;
    std::vector<float> wf;
    const uint8_t *wdata = nullptr;
    size_t wsize = 0;
    if (kind == 0) {
        fill_q8_0(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else if (kind == 1) {
        fill_q4_k(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else if (kind == 2) {
        fill_q6_k(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else {
        wf = random_vec(rows * cols, seed, -1.0f, 1.0f);
        wdata = reinterpret_cast<const uint8_t *>(wf.data());
        wsize = wf.size() * sizeof(float);
    }
    std::vector<float> x = random_vec(tokens * cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)tokens * rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(wdata, wsize, &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)tokens * rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols,
                                  (void *)&tokens};
    bool ok = launch(fn, (unsigned)((tokens * rows + 7) / 8), 256, params) &&
              download(dy, got.data(), (size_t)tokens * rows * sizeof(float));

    double worst_ratio = 0.0, worst_err = 0.0, worst_bound = 0.0;
    double first_ref = 0.0;
    for (int t = 0; t < tokens; ++t) {
        const float *xrow = x.data() + (size_t)t * cols;
        for (int row = 0; row < rows; ++row) {
            const uint8_t *roww = wdata + (size_t)row *
                                             (size_t)(kind == 0 ? (cols / 32) * 34
                                                    : kind == 1 ? (cols / 256) * 144
                                                    : kind == 2 ? (cols / 256) * sofuu_ref::Q6K_STRIDE
                                                                : (size_t)cols * 4);
            sofuu_ref::Dot want;
            if (kind == 0) want = sofuu_ref::dot_q8_0(roww, xrow, cols);
            else if (kind == 1) want = sofuu_ref::dot_q4_k(roww, xrow, cols);
            else if (kind == 2) want = sofuu_ref::dot_q6_k(roww, xrow, cols);
            else {
                const float *rowf = wf.data() + (size_t)row * cols;
                double sum = 0.0, l1 = 0.0;
                for (int c = 0; c < cols; ++c) {
                    double p = (double)rowf[c] * xrow[c];
                    sum += p;
                    l1 += std::fabs(p);
                }
                want = {sum, l1};
            }
            double err = std::fabs((double)got[(size_t)t * rows + row] - want.value);
            double bound = 1e-5 * (want.l1 > 1.0 ? want.l1 : 1.0) + 1e-6;
            double ratio = err / bound;
            if (ratio > worst_ratio) {
                worst_ratio = ratio;
                worst_err = err;
                worst_bound = bound;
                first_ref = want.value;
            }
        }
    }
    const char *base = kind == 0 ? "b8_q8_0" : kind == 1 ? "b8_q4_k"
                       : kind == 2 ? "b8_q6_k" : "b8_f32";
    bool wide = cols >= 4096;
    char name[32];
    std::snprintf(name, sizeof(name), "%s%s", base, wide ? "_wide" : "");
    report(name, ok && worst_ratio <= 1.0, worst_err, worst_bound, first_ref);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_attn_prefill(void *fn, uint32_t &seed) {
    const int tokens = 8, n_heads = 8, n_kv = 2, hd = 64, cap = 32, start = 5;
    std::vector<float> q = random_vec(tokens * n_heads * hd, seed, -1.0f, 1.0f);
    std::vector<float> kc = random_vec(n_kv * cap * hd, seed, -1.0f, 1.0f);
    std::vector<float> vc = random_vec(n_kv * cap * hd, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)tokens * n_heads * hd, 0.0f);
    std::vector<float> want((size_t)tokens * n_heads * hd, 0.0f);
    float scale = 1.0f / std::sqrt((float)hd);

    unsigned long long dq = 0, dk = 0, dv = 0, dout = 0;
    if (!alloc_upload(q.data(), q.size() * sizeof(float), &dq) ||
        !alloc_upload(kc.data(), kc.size() * sizeof(float), &dk) ||
        !alloc_upload(vc.data(), vc.size() * sizeof(float), &dv) ||
        !alloc_only((unsigned long long)tokens * n_heads * hd * sizeof(float), &dout)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dq, &dk, &dv, &dout, (void *)&n_heads,
                                  (void *)&n_kv, (void *)&hd, (void *)&start,
                                  (void *)&tokens, (void *)&cap, &scale};
    bool ok = launch(fn, (unsigned)(tokens * n_heads), 256, params) &&
              download(dout, got.data(), (size_t)tokens * n_heads * hd * sizeof(float));
    for (int t = 0; t < tokens; ++t) {
        sofuu_ref::attn_decode_ref(
            q.data() + (size_t)t * n_heads * hd, kc.data(), vc.data(),
            want.data() + (size_t)t * n_heads * hd, n_heads, n_kv, hd,
            start + t + 1, cap, scale);
    }
    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::fmax(max_err, std::fabs((double)got[i] - want[i]));
        max_ref = std::fmax(max_ref, std::fabs((double)want[i]));
    }
    double bound = 2e-5 * (max_ref > 1.0 ? max_ref : 1.0) + 1e-6;
    report("attn_prefill", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dq);
    sofuu_cuda_ctx_free(g_ctx, dk);
    sofuu_cuda_ctx_free(g_ctx, dv);
    sofuu_cuda_ctx_free(g_ctx, dout);
}

void test_rmsnorm_batch(void *fn, uint32_t &seed) {
    const int rows = 8, n = 512;
    float eps = 1e-5f;
    std::vector<float> x = random_vec(rows * n, seed, -2.0f, 2.0f);
    std::vector<float> w = random_vec(n, seed, 0.5f, 1.5f);
    std::vector<float> got((size_t)rows * n, 0.0f), want((size_t)rows * n, 0.0f);
    for (int r = 0; r < rows; ++r)
        sofuu_ref::rmsnorm_ref(x.data() + (size_t)r * n, w.data(),
                               want.data() + (size_t)r * n, n, eps);
    unsigned long long dx = 0, dw = 0, dy = 0;
    if (!alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_upload(w.data(), w.size() * sizeof(float), &dw) ||
        !alloc_only((unsigned long long)rows * n * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dx, &dw, &dy, (void *)&n, &eps,
                                  (void *)&rows};
    bool ok = launch(fn, (unsigned)rows, 256, params) &&
              download(dy, got.data(), (size_t)rows * n * sizeof(float));
    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::fmax(max_err, std::fabs((double)got[i] - want[i]));
        max_ref = std::fmax(max_ref, std::fabs((double)want[i]));
    }
    double bound = 1e-4 * (max_ref > 1.0 ? max_ref : 1.0) + 1e-5;
    report("rmsnorm_batch", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// A kernel-only bandwidth receipt. The model's end-to-end tok/s mixes this
// with attention, lm_head and launch gaps; if this number is healthy but the
// model is slow, the next suspect is phase serialization, not the GEMV.
void bench_gemv(const char *name, void *fn, int kind, int rows, int cols,
                int iters, uint32_t &seed) {
    std::vector<uint8_t> wbytes;
    std::vector<float> wf;
    const uint8_t *wdata = nullptr;
    size_t wsize = 0;
    if (kind == 0) {
        fill_q8_0(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else if (kind == 1) {
        fill_q4_k(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else if (kind == 2) {
        fill_q6_k(wbytes, rows, cols, seed);
        wdata = wbytes.data();
        wsize = wbytes.size();
    } else {
        wf = random_vec(rows * cols, seed, -1.0f, 1.0f);
        wdata = reinterpret_cast<const uint8_t *>(wf.data());
        wsize = wf.size() * sizeof(float);
    }
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> y((size_t)rows, 0.0f);
    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(wdata, wsize, &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    unsigned grid = (unsigned)((rows + 7) / 8);
    for (int i = 0; i < 3; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("warmup sync");
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("bench sync");
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double row_bytes = kind == 0 ? (double)(cols / 32) * 34
                       : kind == 1 ? (double)(cols / 256) * 144
                       : kind == 2 ? (double)(cols / 256) * sofuu_ref::Q6K_STRIDE
                                   : (double)cols * 4;
    double bytes = (double)rows * row_bytes * (double)iters;
    std::printf("[bench] %-10s %dx%d  %.1f GB/s  %.3f ms/launch\n", name, rows,
                cols, bytes / sec / 1e9, sec * 1000.0 / (double)iters);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// q4_K / q6_K GEMV against q8_1 activations, timed with the weights already
// quantized. Without these rows the dp4a kernels are only measurable end to
// end, which is how the first version hid a 3x gap to the f32 row-dot.
void bench_gemv_q8(const char *name, void *fn, int kind, int rows, int cols,
                   int iters, uint32_t &seed) {
    std::vector<uint8_t> w;
    if (kind == 1) fill_q4_k(w, rows, cols, seed);
    else fill_q6_k(w, rows, cols, seed);
    const int nblk = cols / 32;
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<int8_t> xq(cols);
    std::vector<float> xd(nblk);
    std::vector<int32_t> xs(nblk), xs16(nblk * 2);
    sofuu_ref::quantize_q8_1(x.data(), xq.data(), xd.data(), xs.data(),
                             xs16.data(), cols);
    std::vector<float> y((size_t)rows, 0.0f);
    unsigned long long dw = 0, dq = 0, dd = 0, ds = 0, d16 = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(xq.data(), xq.size(), &dq) ||
        !alloc_upload(xd.data(), (size_t)nblk * 4, &dd) ||
        !alloc_upload(xs.data(), (size_t)nblk * 4, &ds) ||
        !alloc_upload(xs16.data(), (size_t)nblk * 8, &d16) ||
        !alloc_only((unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    // q4_K takes (w, xq, xd, xs, y, rows, cols) and q6_K adds xs16 before y --
    // one parameter list cannot serve both, and getting it wrong writes through
    // a misaligned argument (an illegal address, seen on the T4).
    std::vector<void *> params = kind == 2
        ? std::vector<void *>{&dw, &dq, &dd, &ds, &d16, &dy, (void *)&rows,
                              (void *)&cols}
        : std::vector<void *>{&dw, &dq, &dd, &ds, &dy, (void *)&rows,
                              (void *)&cols};
    unsigned grid = (unsigned)((rows + 7) / 8);
    for (int i = 0; i < 3; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("q8 warmup sync");
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("q8 bench sync");
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double row_bytes = kind == 1 ? (double)(cols / 256) * 144.0
                                 : (double)(cols / 256) * sofuu_ref::Q6K_STRIDE;
    double bytes = (double)rows * row_bytes * (double)iters;
    std::printf("[bench] %-10s %dx%d  %.1f GB/s  %.3f ms/launch\n", name, rows,
                cols, bytes / sec / 1e9, sec * 1000.0 / (double)iters);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dq);
    sofuu_cuda_ctx_free(g_ctx, dd);
    sofuu_cuda_ctx_free(g_ctx, ds);
    sofuu_cuda_ctx_free(g_ctx, d16);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// Empty-launch cost on the GPU timeline. A decode step issues ~290 kernels, so
// any fixed per-launch gap is multiplied by 290: this row decides whether the
// next step is more kernel work or CUDA graphs.
void bench_launch(void *fn, int n, int iters, const char *label) {
    uint32_t seed = 0x51ed1u;
    std::vector<float> x = random_vec(n, seed, -1.0f, 1.0f);
    std::vector<float> w(n, 1.0f);
    unsigned long long dx = 0, dw = 0, dy = 0;
    if (!alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_upload(w.data(), w.size() * sizeof(float), &dw) ||
        !alloc_only((unsigned long long)n * sizeof(float), &dy)) {
        fail_msg("launch bench alloc");
        ++g_failures;
        return;
    }
    int nn = n;
    float eps = 1e-5f;
    std::vector<void *> params = {&dx, &dw, &dy, &nn, &eps};
    unsigned grid = (unsigned)((n + 255) / 256);
    if (!launch_async(fn, grid, 256, params) || sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("launch bench warmup");
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!launch_async(fn, grid, 256, params)) return;
        // Drain periodically: a full launch queue blocks cuLaunchKernel, which
        // would time the driver's queue depth instead of the launch itself.
        if ((i & 63) == 63 && sofuu_cuda_ctx_sync(g_ctx) != 0) {
            fail_msg("launch bench sync");
            return;
        }
    }
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("launch bench sync");
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration<double>(t1 - t0).count() * 1e6 / (double)iters;
    std::printf("[bench] launch_%s  n=%d grid=%u  %.2f us/launch\n", label, n,
                grid, us);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// Raw device-memory control: c = a + b reads 8 B and writes 4 B per element.
// If this is also ~50 GB/s, the instance's memory subsystem is capped and the
// GEMV numbers cannot be blamed on kernel layout.
void bench_add(void *fn, int n, int iters, uint32_t &seed) {
    std::vector<float> a = random_vec(n, seed, -1.0f, 1.0f);
    std::vector<float> b = random_vec(n, seed, -1.0f, 1.0f);
    unsigned long long da = 0, db = 0, dy = 0;
    if (!alloc_upload(a.data(), a.size() * sizeof(float), &da) ||
        !alloc_upload(b.data(), b.size() * sizeof(float), &db) ||
        !alloc_only((unsigned long long)n * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&da, &db, &dy, (void *)&n};
    unsigned grid = (unsigned)((n + 255) / 256);
    for (int i = 0; i < 2; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("add warmup sync");
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        if (!launch_async(fn, grid, 256, params)) return;
    if (sofuu_cuda_ctx_sync(g_ctx) != 0) {
        fail_msg("add bench sync");
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double bytes = (double)n * 12.0 * (double)iters;
    std::printf("[bench] add_f32   %d floats  %.1f GB/s  %.3f ms/launch\n", n,
                bytes / sec / 1e9, sec * 1000.0 / (double)iters);
    sofuu_cuda_ctx_free(g_ctx, da);
    sofuu_cuda_ctx_free(g_ctx, db);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

// ── Elementwise tests ──────────────────────────────────────────────────────
void test_rmsnorm(void *fn, uint32_t &seed) {
    const int n = 1024;
    float eps = 1e-6f;
    std::vector<float> x = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> w = random_vec(n, seed, 0.5f, 1.5f);
    std::vector<float> got((size_t)n, 0.0f), want((size_t)n, 0.0f);
    sofuu_ref::rmsnorm_ref(x.data(), w.data(), want.data(), n, eps);
    unsigned long long dx = 0, dw = 0, dy = 0;
    if (!alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !alloc_upload(w.data(), w.size() * sizeof(float), &dw) ||
        !alloc_only((unsigned long long)n * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dx, &dw, &dy, (void *)&n, &eps};
    bool ok = launch(fn, 1, 256, params) &&
              download(dy, got.data(), (size_t)n * sizeof(float));
    double max_err = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i) {
        max_err = std::fmax(max_err, std::fabs((double)got[(size_t)i] - want[(size_t)i]));
        scale = std::fmax(scale, std::fabs((double)want[(size_t)i]));
    }
    double bound = 1e-4 * (scale > 1.0 ? scale : 1.0) + 1e-5;
    report("rmsnorm", ok && max_err <= bound, max_err, bound, (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_swiglu(void *fn, uint32_t &seed) {
    const int n = 4096;
    std::vector<float> g = random_vec(n, seed, -6.0f, 6.0f);
    std::vector<float> u = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> got((size_t)n, 0.0f), want((size_t)n, 0.0f);
    sofuu_ref::swiglu_ref(g.data(), u.data(), want.data(), n);
    unsigned long long dg = 0, du = 0, dy = 0;
    if (!alloc_upload(g.data(), g.size() * sizeof(float), &dg) ||
        !alloc_upload(u.data(), u.size() * sizeof(float), &du) ||
        !alloc_only((unsigned long long)n * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dg, &du, &dy, (void *)&n};
    bool ok = launch(fn, (unsigned)((n + 255) / 256), 256, params) &&
              download(dy, got.data(), (size_t)n * sizeof(float));
    double max_err = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i) {
        max_err = std::fmax(max_err, std::fabs((double)got[(size_t)i] - want[(size_t)i]));
        scale = std::fmax(scale, std::fabs((double)want[(size_t)i]));
    }
    double bound = 1e-5 * (scale > 1.0 ? scale : 1.0) + 1e-5;
    report("swiglu", ok && max_err <= bound, max_err, bound, (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dg);
    sofuu_cuda_ctx_free(g_ctx, du);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_rope(void *fn, uint32_t &seed) {
    const int n_heads = 8, head_dim = 128, n_rot = 64;
    const int n = n_heads * head_dim;
    std::vector<float> cos_t(32), sin_t(32);
    for (int i = 0; i < 32; ++i) {
        float a = 0.05f * (float)i;
        cos_t[(size_t)i] = std::cos(a);
        sin_t[(size_t)i] = std::sin(a);
    }
    for (int interleaved = 0; interleaved <= 1; ++interleaved) {
        std::vector<float> x = random_vec(n, seed, -1.0f, 1.0f);
        std::vector<float> want = x;
        sofuu_ref::rope_ref(want.data(), cos_t.data(), sin_t.data(), n_heads,
                            head_dim, n_rot, interleaved);
        unsigned long long dx = 0, dc = 0, ds = 0;
        if (!alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
            !alloc_upload(cos_t.data(), cos_t.size() * sizeof(float), &dc) ||
            !alloc_upload(sin_t.data(), sin_t.size() * sizeof(float), &ds)) {
            fail_msg("alloc");
            ++g_failures;
            return;
        }
        std::vector<void *> params = {&dx, &dc, &ds, (void *)&n_heads,
                                      (void *)&head_dim, (void *)&n_rot,
                                      (void *)&interleaved};
        bool ok = launch(fn, (unsigned)n_heads, 128, params) &&
                  download(dx, x.data(), (size_t)n * sizeof(float));
        double max_err = 0.0;
        for (int i = 0; i < n; ++i)
            max_err = std::fmax(max_err, std::fabs((double)x[(size_t)i] - want[(size_t)i]));
        report(interleaved ? "rope_interleaved" : "rope_split",
               ok && max_err <= 1e-6, max_err, 1e-6, (double)want[0]);
        sofuu_cuda_ctx_free(g_ctx, dx);
        sofuu_cuda_ctx_free(g_ctx, dc);
        sofuu_cuda_ctx_free(g_ctx, ds);
    }
}

void test_kv_write(void *fn, uint32_t &seed) {
    const int n_kv = 8, hd = 128, cap = 16, pos = 3;
    const int cache_n = n_kv * cap * hd;
    std::vector<float> got_k((size_t)cache_n, -99.0f), got_v((size_t)cache_n, -99.0f);
    std::vector<float> want_k = got_k, want_v = got_v;
    std::vector<float> k = random_vec(n_kv * hd, seed, -1.0f, 1.0f);
    std::vector<float> v = random_vec(n_kv * hd, seed, -1.0f, 1.0f);
    sofuu_ref::kv_write_ref(want_k.data(), want_v.data(), k.data(), v.data(),
                            pos, n_kv, hd, cap);
    unsigned long long dk = 0, dv = 0, dj = 0, dl = 0;
    if (!alloc_upload(got_k.data(), got_k.size() * sizeof(float), &dk) ||
        !alloc_upload(got_v.data(), got_v.size() * sizeof(float), &dv) ||
        !alloc_upload(k.data(), k.size() * sizeof(float), &dj) ||
        !alloc_upload(v.data(), v.size() * sizeof(float), &dl)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dk, &dv, &dj, &dl, (void *)&pos,
                                  (void *)&n_kv, (void *)&hd, (void *)&cap};
    bool ok = launch(fn, (unsigned)n_kv, 128, params) &&
              download(dk, got_k.data(), (size_t)cache_n * sizeof(float)) &&
              download(dv, got_v.data(), (size_t)cache_n * sizeof(float));
    double max_err = 0.0;
    for (int i = 0; i < cache_n; ++i) {
        max_err = std::fmax(max_err, std::fabs((double)got_k[(size_t)i] - want_k[(size_t)i]));
        max_err = std::fmax(max_err, std::fabs((double)got_v[(size_t)i] - want_v[(size_t)i]));
    }
    report("kv_write", ok && max_err == 0.0, max_err, 0.0, (double)want_k[0]);
    sofuu_cuda_ctx_free(g_ctx, dk);
    sofuu_cuda_ctx_free(g_ctx, dv);
    sofuu_cuda_ctx_free(g_ctx, dj);
    sofuu_cuda_ctx_free(g_ctx, dl);
}

void test_attn_decode(void *fn, uint32_t &seed) {
    const int n_heads = 8, n_kv = 2, hd = 64, cap = 32, seq = 17;
    const int cache_n = n_kv * cap * hd;
    std::vector<float> q = random_vec(n_heads * hd, seed, -1.0f, 1.0f);
    std::vector<float> k = random_vec(cache_n, seed, -1.0f, 1.0f);
    std::vector<float> v = random_vec(cache_n, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)(n_heads * hd), 0.0f);
    std::vector<float> want((size_t)(n_heads * hd), 0.0f);
    float scale = 1.0f / std::sqrt((float)hd);
    sofuu_ref::attn_decode_ref(q.data(), k.data(), v.data(), want.data(),
                               n_heads, n_kv, hd, seq, cap, scale);
    unsigned long long dq = 0, dk = 0, dv = 0, dout = 0;
    if (!alloc_upload(q.data(), q.size() * sizeof(float), &dq) ||
        !alloc_upload(k.data(), k.size() * sizeof(float), &dk) ||
        !alloc_upload(v.data(), v.size() * sizeof(float), &dv) ||
        !alloc_only((unsigned long long)got.size() * sizeof(float), &dout)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dq, &dk, &dv, &dout, (void *)&n_heads,
                                  (void *)&n_kv, (void *)&hd, (void *)&seq,
                                  (void *)&cap, (void *)&scale};
    bool ok = launch(fn, (unsigned)n_heads, 256, params) &&
              download(dout, got.data(), got.size() * sizeof(float));
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        max_err = std::fmax(max_err,
                            std::fabs((double)got[i] - (double)want[i]));
    report("attn_decode", ok && max_err <= 5e-5, max_err, 5e-5,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dq);
    sofuu_cuda_ctx_free(g_ctx, dk);
    sofuu_cuda_ctx_free(g_ctx, dv);
    sofuu_cuda_ctx_free(g_ctx, dout);
}

void test_add(void *fn, uint32_t &seed) {
    const int n = 4096;
    std::vector<float> a = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> b = random_vec(n, seed, -2.0f, 2.0f);
    std::vector<float> got((size_t)n, 0.0f), want((size_t)n, 0.0f);
    sofuu_ref::add_ref(a.data(), b.data(), want.data(), n);
    unsigned long long da = 0, db = 0, dy = 0;
    if (!alloc_upload(a.data(), a.size() * sizeof(float), &da) ||
        !alloc_upload(b.data(), b.size() * sizeof(float), &db) ||
        !alloc_only((unsigned long long)n * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&da, &db, &dy, (void *)&n};
    bool ok = launch(fn, (unsigned)((n + 255) / 256), 256, params) &&
              download(dy, got.data(), got.size() * sizeof(float));
    double max_err = 0.0;
    for (int i = 0; i < n; ++i)
        max_err = std::fmax(max_err,
                            std::fabs((double)got[(size_t)i] - (double)want[(size_t)i]));
    report("add", ok && max_err == 0.0, max_err, 0.0, (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, da);
    sofuu_cuda_ctx_free(g_ctx, db);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

std::string read_file(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char **argv) {
    const char *ptx_path = argc > 1 ? argv[1] : "src/cuda/kernels.ptx";
    std::printf("Sofuu CUDA GPU parity gate\n");
    std::printf("PTX: %s\n", ptx_path);

    g_drv = sofuu_cuda_driver_open(nullptr);
    if (!g_drv) {
        std::fprintf(stderr, "NO-DRIVER: %s\n", sofuu_cuda_last_error());
        std::fprintf(stderr,
                     "This host has no libcuda.so.1 / NVIDIA device. Run on a "
                     "GPU runtime (Kaggle T4 with Accelerator = GPU).\n");
        return 2;
    }
    int cc_major = 0, cc_minor = 0, integrated = 0, unified = 0;
    char name[256] = {0};
    unsigned long long vram = 0;
    if (sofuu_cuda_driver_device_info(g_drv, 0, name, sizeof(name), &cc_major,
                                      &cc_minor, &vram, &integrated,
                                      &unified) != 0) {
        fail_msg("device_info");
        return 2;
    }
    std::printf("device: %s cc%d.%d vram=%.2f GiB integrated=%d unified=%d "
                "driver=%d lib=%s\n",
                name, cc_major, cc_minor,
                (double)vram / (1024.0 * 1024.0 * 1024.0), integrated, unified,
                sofuu_cuda_driver_version(g_drv),
                sofuu_cuda_driver_path(g_drv));

    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) {
        std::fprintf(stderr, "FAIL: cannot read PTX at %s\n", ptx_path);
        return 1;
    }
    g_ctx = sofuu_cuda_ctx_create(g_drv, 0);
    if (!g_ctx) {
        fail_msg("ctx_create");
        return 2;
    }
    if (sofuu_cuda_ctx_load_ptx(g_ctx, ptx.c_str()) != 0) {
        fail_msg("cuModuleLoadData");
        return 1;
    }

    void *f_q8 = nullptr, *f_q4 = nullptr, *f_q6 = nullptr, *f_f32 = nullptr;
    void *f_quant = nullptr, *f_q4q8 = nullptr, *f_q6q8 = nullptr;
    void *f_rms = nullptr, *f_swi = nullptr, *f_rope = nullptr, *f_kv = nullptr;
    void *f_attn = nullptr, *f_add = nullptr;
    void *f_b8_q4 = nullptr, *f_b8_q6 = nullptr, *f_b8_q8 = nullptr;
    void *f_b8_f32 = nullptr, *f_rms_b = nullptr, *f_attn_p = nullptr;
    if (!get_function("sofuu_gemv_q8_0", &f_q8) ||
        !get_function("sofuu_gemv_q4_k", &f_q4) ||
        !get_function("sofuu_gemv_q6_k", &f_q6) ||
        !get_function("sofuu_gemv_f32", &f_f32) ||
        !get_function("sofuu_quantize_q8_1", &f_quant) ||
        !get_function("sofuu_gemv_q4_k_q8", &f_q4q8) ||
        !get_function("sofuu_gemv_q6_k_q8", &f_q6q8) ||
        !get_function("sofuu_rmsnorm", &f_rms) ||
        !get_function("sofuu_swiglu", &f_swi) ||
        !get_function("sofuu_rope_f32", &f_rope) ||
        !get_function("sofuu_kv_write_f32", &f_kv) ||
        !get_function("sofuu_attn_decode_f32", &f_attn) ||
        !get_function("sofuu_add_f32", &f_add) ||
        !get_function("sofuu_gemm_q4_k_b8", &f_b8_q4) ||
        !get_function("sofuu_gemm_q6_k_b8", &f_b8_q6) ||
        !get_function("sofuu_gemm_q8_0_b8", &f_b8_q8) ||
        !get_function("sofuu_gemm_f32_b8", &f_b8_f32) ||
        !get_function("sofuu_rmsnorm_batch", &f_rms_b) ||
        !get_function("sofuu_attn_prefill_f32", &f_attn_p)) {
        return 1;
    }

    uint32_t seed = 0x5eed1234u;
    for (int cols : {1024, 4096}) test_gemv_q8_0(f_q8, seed, cols);
    for (int cols : {2048, 8192}) {
        test_gemv_q4_k(f_q4, seed, cols);
        test_gemv_q6_k(f_q6, seed, cols);
    }
    test_gemv_q4_k_q8(f_quant, f_q4q8, seed);
    test_gemv_q6_k_q8(f_quant, f_q6q8, seed);
    test_gemv_f32(f_f32, seed);
    test_rmsnorm(f_rms, seed);
    bench_launch(f_rms, 2048, 4000, "tiny");
    bench_launch(f_rms, 1 << 20, 2000, "big");
    test_swiglu(f_swi, seed);
    test_rope(f_rope, seed);
    test_kv_write(f_kv, seed);
    test_attn_decode(f_attn, seed);
    test_add(f_add, seed);
    for (int cols : {1024, 4096}) test_gemm_b8(f_b8_q8, seed, 0, cols);
    for (int cols : {2048, 8192}) {
        test_gemm_b8(f_b8_q4, seed, 1, cols);
        test_gemm_b8(f_b8_q6, seed, 2, cols);
    }
    test_gemm_b8(f_b8_f32, seed, 3, 1024);
    test_rmsnorm_batch(f_rms_b, seed);
    test_attn_prefill(f_attn_p, seed);

    // The idle T4 sits at 300 MHz SM / 405 MHz memory; a 15 ms benchmark
    // measures the ramp, not the kernel. Hold a sustained 64M-float add for
    // ~0.6 s first, then benchmark at boost clocks.
    bench_add(f_add, 64 * 1024 * 1024, 200, seed);
    // Kernel-only throughput, using the model's 8192x2048 FFN shape.
    bench_gemv("gemv_q8_0", f_q8, 0, 8192, 2048, 200, seed);
    bench_gemv("gemv_q4_k", f_q4, 1, 8192, 2048, 400, seed);
    bench_gemv("gemv_q6_k", f_q6, 2, 8192, 2048, 300, seed);
    bench_gemv("gemv_f32", f_f32, 3, 8192, 2048, 80, seed);
    // The other two shapes that matter per token: q/o projections
    // (2048x2048) and lm_head (128256x2048, the single largest matrix).
    bench_gemv("qo_q4_k", f_q4, 1, 2048, 2048, 600, seed);
    bench_gemv("qo_q6_k", f_q6, 2, 2048, 2048, 600, seed);
    bench_gemv("lm_head_q4_k", f_q4, 1, 128256, 2048, 60, seed);
    bench_gemv("lm_head_q6_k", f_q6, 2, 128256, 2048, 60, seed);
    bench_gemv_q8("x_q8_1row", f_q4q8, 1, 8192, 2048, 400, seed);
    bench_gemv_q8("x_q8_1row_lm", f_q4q8, 1, 128256, 2048, 60, seed);
    bench_gemv_q8("q6k_x_q8", f_q6q8, 2, 8192, 2048, 400, seed);
    bench_gemv_q8("q6k_x_q8_lm", f_q6q8, 2, 128256, 2048, 60, seed);

    sofuu_cuda_ctx_destroy(g_ctx);
    sofuu_cuda_driver_close(g_drv);
    if (g_failures == 0) {
        std::printf("RESULT: PASS (all GPU kernels match the scalar references)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d kernel check(s) failed)\n", g_failures);
    return 1;
}
