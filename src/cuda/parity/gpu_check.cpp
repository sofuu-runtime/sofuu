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

bool get_function(const char *name, void **out) {
    if (sofuu_cuda_ctx_function(g_ctx, name, out) != 0) {
        fail_msg(name);
        return false;
    }
    return true;
}

// ── GEMV tests ─────────────────────────────────────────────────────────────
void test_gemv_q8_0(void *fn, uint32_t &seed) {
    const int rows = 8, cols = 1024;
    std::vector<uint8_t> w;
    fill_q8_0(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)rows, 256, params) &&
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
    report("gemv_q8_0", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_q4_k(void *fn, uint32_t &seed) {
    const int rows = 8, cols = 2048;
    std::vector<uint8_t> w;
    fill_q4_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)rows, 256, params) &&
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
    report("gemv_q4_k", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
    sofuu_cuda_ctx_free(g_ctx, dy);
}

void test_gemv_q6_k(void *fn, uint32_t &seed) {
    const int rows = 8, cols = 2048;
    std::vector<uint8_t> w;
    fill_q6_k(w, rows, cols, seed);
    std::vector<float> x = random_vec(cols, seed, -1.0f, 1.0f);
    std::vector<float> got((size_t)rows, 0.0f), want((size_t)rows, 0.0f);

    unsigned long long dw = 0, dx = 0, dy = 0;
    if (!alloc_upload(w.data(), w.size(), &dw) ||
        !alloc_upload(x.data(), x.size() * sizeof(float), &dx) ||
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)rows, 256, params) &&
              download(dy, got.data(), (size_t)rows * sizeof(float));
    double max_err = 0.0, bound = 0.0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = w.data() + (size_t)r * (cols / 256) * 210;
        sofuu_ref::Dot want_d = sofuu_ref::dot_q6_k(row, x.data(), cols);
        want[(size_t)r] = (float)want_d.value;
        double e = std::fabs((double)got[(size_t)r] - want_d.value);
        double b = 1e-5 * (want_d.l1 > 1.0 ? want_d.l1 : 1.0) + 1e-6;
        if (e > max_err) max_err = e;
        if (b > bound) bound = b;
    }
    report("gemv_q6_k", ok && max_err <= bound, max_err, bound,
           (double)want[0]);
    sofuu_cuda_ctx_free(g_ctx, dw);
    sofuu_cuda_ctx_free(g_ctx, dx);
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
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)rows * sizeof(float), &dy)) {
        fail_msg("alloc");
        ++g_failures;
        return;
    }
    std::vector<void *> params = {&dw, &dx, &dy, (void *)&rows, (void *)&cols};
    bool ok = launch(fn, (unsigned)rows, 256, params) &&
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
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)n * sizeof(float), &dy)) {
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
        !sofuu_cuda_ctx_alloc(g_ctx, (unsigned long long)n * sizeof(float), &dy)) {
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
    void *f_rms = nullptr, *f_swi = nullptr, *f_rope = nullptr, *f_kv = nullptr;
    if (!get_function("sofuu_gemv_q8_0", &f_q8) ||
        !get_function("sofuu_gemv_q4_k", &f_q4) ||
        !get_function("sofuu_gemv_q6_k", &f_q6) ||
        !get_function("sofuu_gemv_f32", &f_f32) ||
        !get_function("sofuu_rmsnorm", &f_rms) ||
        !get_function("sofuu_swiglu", &f_swi) ||
        !get_function("sofuu_rope_f32", &f_rope) ||
        !get_function("sofuu_kv_write_f32", &f_kv)) {
        return 1;
    }

    uint32_t seed = 0x5eed1234u;
    test_gemv_q8_0(f_q8, seed);
    test_gemv_q4_k(f_q4, seed);
    test_gemv_q6_k(f_q6, seed);
    test_gemv_f32(f_f32, seed);
    test_rmsnorm(f_rms, seed);
    test_swiglu(f_swi, seed);
    test_rope(f_rope, seed);
    test_kv_write(f_kv, seed);

    sofuu_cuda_ctx_destroy(g_ctx);
    sofuu_cuda_driver_close(g_drv);
    if (g_failures == 0) {
        std::printf("RESULT: PASS (all GPU kernels match the scalar references)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d kernel check(s) failed)\n", g_failures);
    return 1;
}
