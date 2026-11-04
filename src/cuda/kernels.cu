// src/cuda/kernels.cu — Sofuu native CUDA kernel fleet (PLAN-GPU-CUDA C1/C2).
//
// This file is deliberately self-contained: it uses no CUDA headers and no
// CUDA builtins. Thread/block identity, warp shuffles, barriers, rsqrt and exp
// are expressed as inline PTX so the same source can be built three ways:
//
//   1. clang NVPTX -> src/cuda/kernels.ptx (checked in, loaded by the driver)
//   2. nvcc        -> src/cuda/kernels.ptx (fallback generator, same entries)
//   3. host C++    -> -DSOFUU_CUDA_HOST, for the CPU algorithm-parity harness
//
// The host build exposes the same per-value math the kernels use, so
// `scripts/cuda/check_kernels.sh` can prove the dequant formulas on a machine
// with no NVIDIA hardware. The actual launches are proven on a GPU by
// src/cuda/parity/gpu_check.cpp (Kaggle T4 gate).

#include <stddef.h>

#if defined(SOFUU_CUDA_HOST)
#include <math.h>
#define SOFUU_DEVICE
#define SOFUU_GLOBAL
#define SOFUU_SHARED
#define SOFUU_INLINE inline
#define SOFUU_LAUNCH_BOUNDS(n)
#else
#define SOFUU_DEVICE __attribute__((device))
#define SOFUU_GLOBAL __attribute__((global))
#define SOFUU_SHARED __attribute__((shared))
#define SOFUU_INLINE __inline__ __attribute__((always_inline))
// Declare the block size each kernel is launched with, so the compiler budgets
// registers for it instead of guessing. Without this a future edit can push a
// kernel over the per-thread register limit and the launch fails at runtime
// with cudaErrorLaunchOutOfResources (701) on the one machine nobody tested.
// Emitted into the PTX as .maxntid, which is what makes it verifiable without
// a GPU: see the .maxntid assertion in domains/infer/cuda.rs.
#define SOFUU_LAUNCH_BOUNDS(n) __attribute__((launch_bounds(n)))
#endif

// ── GGML block layouts (packed, little-endian host) ────────────────────────
#pragma pack(push, 1)
typedef struct {
    unsigned short d;      // f16 scale
    signed char qs[32];    // 32 int8 values
} sofuu_block_q8_0;

typedef struct {
    unsigned short d;          // f16 super-block scale
    unsigned short dmin;       // f16 super-block min
    unsigned char scales[12];  // six 6-bit scales + six mins
    unsigned char qs[128];     // 256 nibbles: group g -> low/high halves
} sofuu_block_q4_K;

typedef struct {
    unsigned char ql[128];     // low 4 bits
    unsigned char qh[64];      // high 2 bits
    signed char scales[16];    // int8 block scales
    unsigned short d;          // f16 super-block scale
} sofuu_block_q6_K;
#pragma pack(pop)

#define SOFUU_QK_Q8_0 32
#define SOFUU_QK_K 256
#define SOFUU_Q4_K_BYTES 144
#define SOFUU_Q6_K_BYTES 210
#define SOFUU_Q8_0_BYTES 34
#define SOFUU_GEMV_THREADS 256

// ── Portable half -> float (exact for every finite binary16 value) ─────────
static SOFUU_DEVICE SOFUU_INLINE float sofuu_f16_to_f32(unsigned short h) {
    unsigned int sign = (unsigned int)(h >> 15) & 1u;
    unsigned int exp = (unsigned int)(h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int)h & 0x3ffu;
    float v;
    if (exp == 0) {
        // subnormal: mant * 2^-24 (exact in f32 for all 10-bit mantissas)
        v = (float)mant * 5.9604644775390625e-8f;
    } else if (exp == 31) {
        // inf / NaN. Use the canonical quiet NaN for any NaN payload.
        union { unsigned int u; float f; } c;
        c.u = mant ? 0x7fc00000u : 0x7f800000u;
        v = c.f;
    } else {
        union { unsigned int u; float f; } c;
        c.u = ((exp + 112u) << 23) | (mant << 13);
        v = c.f;
    }
    {
        union { unsigned int u; float f; } c;
        c.f = v;
        c.u |= sign << 31;
        return c.f;
    }
}

// ── q4_K scale/min expansion (bit-identical to src/simd/infer_kernels.cpp) ─
static SOFUU_DEVICE SOFUU_INLINE void sofuu_q4k_scale_min(
    int j, const unsigned char *q, unsigned int *sc, unsigned int *mn) {
    if (j < 4) {
        *sc = (unsigned int)(q[j] & 63u);
        *mn = (unsigned int)(q[j + 4] & 63u);
    } else {
        *sc = (unsigned int)(q[j + 4] & 0x0fu) |
              (((unsigned int)(q[j - 4] >> 6) & 3u) << 4);
        *mn = (unsigned int)(q[j + 4] >> 4) |
              (((unsigned int)(q[j] >> 6) & 3u) << 4);
    }
}

// One q4_K value: `sub` is the 32-value sub-block 0..7, `l` the index within it.
// Group g = sub/2 owns qs bytes [32g .. 32g+31]; the low nibble is sub 2g, the
// high nibble is sub 2g+1.
static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_value(
    const sofuu_block_q4_K *b, int sub, int l, float x) {
    unsigned int sc, mn;
    sofuu_q4k_scale_min(sub, b->scales, &sc, &mn);
    float d = sofuu_f16_to_f32(b->d);
    float dmin = sofuu_f16_to_f32(b->dmin);
    unsigned char byte = b->qs[(sub >> 1) * 32 + l];
    unsigned int q = (sub & 1) ? (unsigned int)(byte >> 4)
                               : (unsigned int)(byte & 0x0fu);
    float w = d * (float)sc * (float)q - dmin * (float)mn;
    return w * x;
}

// One q6_K value: `run` 0..7 is (half n = run/4, one of the four 32-value runs).
// run 0: ql[l] low, qh shift 0, scale slot n*8+part+0
// run 1: ql[l+32] low, qh shift 2, scale slot n*8+part+2
// run 2: ql[l] high, qh shift 4, scale slot n*8+part+4
// run 3: ql[l+32] high, qh shift 6, scale slot n*8+part+6
static SOFUU_DEVICE SOFUU_INLINE float sofuu_q6k_value(
    const sofuu_block_q6_K *b, int run, int l, float x) {
    int n = run >> 2;
    int r = run & 3;
    int part = l >> 4;
    int slot = n * 8 + part + 2 * r;
    unsigned int ql_byte =
        (unsigned int)b->ql[n * 64 + ((r == 1 || r == 3) ? (l + 32) : l)];
    unsigned int nib = (r < 2) ? (ql_byte & 0x0fu) : (ql_byte >> 4);
    unsigned int qh_byte = (unsigned int)b->qh[n * 32 + l];
    unsigned int hi2 = (qh_byte >> (2 * r)) & 3u;
    int q = (int)(nib | (hi2 << 4)) - 32;
    float d = sofuu_f16_to_f32(b->d);
    return d * (float)b->scales[slot] * (float)q * x;
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_q8_0_value(
    const sofuu_block_q8_0 *b, int l, float x) {
    return sofuu_f16_to_f32(b->d) * (float)b->qs[l] * x;
}

// ── Activation math ────────────────────────────────────────────────────────
static SOFUU_DEVICE SOFUU_INLINE float sofuu_rsqrt(float x) {
#if defined(SOFUU_CUDA_HOST)
    return 1.0f / sqrtf(x);
#else
    float r;
    __asm__ volatile("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
#endif
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_exp(float x) {
#if defined(SOFUU_CUDA_HOST)
    return expf(x);
#else
    float r;
    // exp(x) = 2^(x * log2(e)); ex2.approx is a hardware instruction.
    __asm__ volatile("ex2.approx.f32 %0, %1;"
                     : "=f"(r)
                     : "f"(x * 1.4426950408889634f));
    return r;
#endif
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_silu(float x) {
    return x / (1.0f + sofuu_exp(-x));
}

// ── Device primitives (identity, shuffle, barrier) ─────────────────────────
#if !defined(SOFUU_CUDA_HOST)
static SOFUU_DEVICE SOFUU_INLINE unsigned int sofuu_tid_x(void) {
    unsigned int v;
    __asm__ volatile("mov.u32 %0, %%tid.x;" : "=r"(v));
    return v;
}
static SOFUU_DEVICE SOFUU_INLINE unsigned int sofuu_ctaid_x(void) {
    unsigned int v;
    __asm__ volatile("mov.u32 %0, %%ctaid.x;" : "=r"(v));
    return v;
}
static SOFUU_DEVICE SOFUU_INLINE unsigned int sofuu_ntid_x(void) {
    unsigned int v;
    __asm__ volatile("mov.u32 %0, %%ntid.x;" : "=r"(v));
    return v;
}
static SOFUU_DEVICE SOFUU_INLINE float sofuu_shfl_down(float v, int delta,
                                                       unsigned int mask) {
    float r;
    __asm__ volatile("shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;"
                     : "=f"(r)
                     : "f"(v), "r"(delta), "r"(mask));
    return r;
}
static SOFUU_DEVICE SOFUU_INLINE void sofuu_bar_sync(void) {
    __asm__ volatile("bar.sync 0;");
}
#endif

// Full-block sum. Returns the row sum on thread 0 and 0 elsewhere.
// `smem` must hold at least 8 floats.
#if !defined(SOFUU_CUDA_HOST)
static SOFUU_DEVICE SOFUU_INLINE float sofuu_block_reduce(float v,
                                                          float *smem) {
    unsigned int tid = sofuu_tid_x();
    v += sofuu_shfl_down(v, 16, 0xffffffffu);
    v += sofuu_shfl_down(v, 8, 0xffffffffu);
    v += sofuu_shfl_down(v, 4, 0xffffffffu);
    v += sofuu_shfl_down(v, 2, 0xffffffffu);
    v += sofuu_shfl_down(v, 1, 0xffffffffu);
    if ((tid & 31u) == 0u) smem[tid >> 5] = v;
    sofuu_bar_sync();
    // Warp 0 reduces the eight warp sums. All 32 lanes participate (lanes 8-31
    // contribute zero) so the full membermask is well-defined: shfl.sync with
    // a source lane outside its mask is undefined behaviour.
    if (tid < 32u) {
        v = (tid < 8u) ? smem[tid] : 0.0f;
        v += sofuu_shfl_down(v, 16, 0xffffffffu);
        v += sofuu_shfl_down(v, 8, 0xffffffffu);
        v += sofuu_shfl_down(v, 4, 0xffffffffu);
        v += sofuu_shfl_down(v, 2, 0xffffffffu);
        v += sofuu_shfl_down(v, 1, 0xffffffffu);
        if (tid == 0u) return v;
    }
    return 0.0f;
}
#endif

// ── GEMV kernels ───────────────────────────────────────────────────────────
// q8_0: one thread owns one 32-value block.
#if !defined(SOFUU_CUDA_HOST)
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_gemv_q8_0(const unsigned char *w,
                                             const float *x, float *y,
                                             int rows, int cols) {
    int row = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    int nb = cols / SOFUU_QK_Q8_0;
    const unsigned char *roww =
        w + (unsigned long long)row * (unsigned long long)nb * SOFUU_Q8_0_BYTES;
    float acc = 0.0f;
    for (int bi = (int)tid; bi < nb; bi += (int)sofuu_ntid_x()) {
        const sofuu_block_q8_0 *b =
            (const sofuu_block_q8_0 *)(roww + (unsigned long long)bi * 34);
        const float *xb = x + (unsigned long long)bi * 32;
        for (int l = 0; l < 32; ++l) {
            acc += sofuu_q8_0_value(b, l, xb[l]);
        }
    }
    SOFUU_SHARED float smem[8];
    float total = sofuu_block_reduce(acc, smem);
    if (sofuu_tid_x() == 0u) y[row] = total;
}

// q4_K: eight threads own one 256-value super-block, one 32-value sub-block
// each; 256 threads cover 32 super-blocks per iteration.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_gemv_q4_k(const unsigned char *w,
                                             const float *x, float *y,
                                             int rows, int cols) {
    int row = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww =
        w + (unsigned long long)row * (unsigned long long)nb * SOFUU_Q4_K_BYTES;
    int sub = (int)(tid & 7u);
    int slot = (int)(tid >> 3);
    float acc = 0.0f;
    for (int bi = slot; bi < nb; bi += 32) {
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)bi * 144);
        const float *xb = x + (unsigned long long)bi * 256 + (unsigned long long)sub * 32;
        for (int l = 0; l < 32; ++l) {
            acc += sofuu_q4k_value(b, sub, l, xb[l]);
        }
    }
    SOFUU_SHARED float smem[8];
    float total = sofuu_block_reduce(acc, smem);
    if (sofuu_tid_x() == 0u) y[row] = total;
}

// q6_K: eight threads own one 256-value super-block, one 32-value run each.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_gemv_q6_k(const unsigned char *w,
                                             const float *x, float *y,
                                             int rows, int cols) {
    int row = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww =
        w + (unsigned long long)row * (unsigned long long)nb * SOFUU_Q6_K_BYTES;
    int run = (int)(tid & 7u);
    int slot = (int)(tid >> 3);
    float acc = 0.0f;
    for (int bi = slot; bi < nb; bi += 32) {
        const sofuu_block_q6_K *b =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)bi * 210);
        const float *xb = x + (unsigned long long)bi * 256 + (unsigned long long)run * 32;
        for (int l = 0; l < 32; ++l) {
            acc += sofuu_q6k_value(b, run, l, xb[l]);
        }
    }
    SOFUU_SHARED float smem[8];
    float total = sofuu_block_reduce(acc, smem);
    if (sofuu_tid_x() == 0u) y[row] = total;
}

// f32 reference GEMV (also the C1 correctness floor).
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_gemv_f32(const float *w, const float *x,
                                            float *y, int rows, int cols) {
    int row = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    const float *roww = w + (unsigned long long)row * (unsigned long long)cols;
    float acc = 0.0f;
    for (int i = (int)tid; i < cols; i += (int)sofuu_ntid_x()) {
        acc += roww[i] * x[i];
    }
    SOFUU_SHARED float smem[8];
    float total = sofuu_block_reduce(acc, smem);
    if (sofuu_tid_x() == 0u) y[row] = total;
}

// ── RMSNorm (one block per row) ────────────────────────────────────────────
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_rmsnorm(const float *x, const float *w,
                                           float *y, int n, float eps) {
    unsigned int tid = sofuu_tid_x();
    float ss = 0.0f;
    for (int i = (int)tid; i < n; i += (int)sofuu_ntid_x()) {
        ss += x[i] * x[i];
    }
    SOFUU_SHARED float red[8];
    SOFUU_SHARED float scale;
    float total = sofuu_block_reduce(ss, red);
    if (tid == 0u) scale = sofuu_rsqrt(total / (float)n + eps);
    sofuu_bar_sync();
    float s = scale;
    for (int i = (int)tid; i < n; i += (int)sofuu_ntid_x()) {
        y[i] = x[i] * s * w[i];
    }
}

// ── SwiGLU: y = silu(g) * u ────────────────────────────────────────────────
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256) void sofuu_swiglu(const float *g, const float *u,
                                          float *y, int n) {
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    int base = (int)sofuu_ctaid_x() * (int)stride;
    int i = base + (int)tid;
    if (i < n) y[i] = sofuu_silu(g[i]) * u[i];
}

// ── RoPE (in place), split-half and interleaved ────────────────────────────
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128) void sofuu_rope_f32(float *x, const float *cos_tab,
                                            const float *sin_tab, int n_heads,
                                            int head_dim, int n_rot,
                                            int interleaved) {
    int head = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    int half = n_rot >> 1;
    float *base = x + (unsigned long long)head * (unsigned long long)head_dim;
    for (int i = (int)tid; i < half; i += (int)stride) {
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

// ── Decode attention (f32, GQA, one block per query head) ─────────────────
//
// Two passes over the cached keys: the first finds the max score, the second
// accumulates exp(score-max)*V. That is one extra dot product per position and
// no per-step rescaling of the accumulator, which keeps the numerics a plain
// softmax instead of an online one. `seq_len` is the number of cached rows
// (pos+1); the kernel never reads past it.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_attn_decode_f32(const float *q, const float *k_cache,
                           const float *v_cache, float *out, int n_heads,
                           int n_kv_heads, int head_dim, int seq_len, int cap,
                           float scale) {
    int h = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    if (h >= n_heads || n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
        seq_len <= 0 || seq_len > cap || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int group = n_heads / n_kv_heads;
    int kvh = h / group;
    if (kvh >= n_kv_heads) return;

    const float *qv = q + (unsigned long long)h * (unsigned long long)head_dim;
    const float *kbase =
        k_cache + (unsigned long long)kvh * (unsigned long long)cap *
                      (unsigned long long)head_dim;
    const float *vbase =
        v_cache + (unsigned long long)kvh * (unsigned long long)cap *
                      (unsigned long long)head_dim;

    SOFUU_SHARED float red[8];
    SOFUU_SHARED float acc[256];
    SOFUU_SHARED float s_max;
    SOFUU_SHARED float s_w;
    SOFUU_SHARED float s_sum;

    if (tid == 0u) s_max = -3.402823466e38f;
    for (int d = (int)tid; d < head_dim; d += (int)stride) acc[d] = 0.0f;
    sofuu_bar_sync();

    // Pass 1: max score.
    for (int p = 0; p < seq_len; ++p) {
        float part = 0.0f;
        if ((int)tid < head_dim) {
            part = qv[tid] *
                   kbase[(unsigned long long)p * (unsigned long long)head_dim +
                         (unsigned long long)tid];
        }
        float sum = sofuu_block_reduce(part, red);
        if (tid == 0u) {
            float score = sum * scale;
            if (score > s_max) s_max = score;
        }
        // The reducer reuses `red`; a second barrier keeps the next iteration
        // from overwriting it while warp 0 is still reducing.
        sofuu_bar_sync();
    }

    // Pass 2: sum(exp(score-max) * V).
    if (tid == 0u) s_sum = 0.0f;
    sofuu_bar_sync();
    for (int p = 0; p < seq_len; ++p) {
        float part = 0.0f;
        if ((int)tid < head_dim) {
            part = qv[tid] *
                   kbase[(unsigned long long)p * (unsigned long long)head_dim +
                         (unsigned long long)tid];
        }
        float sum = sofuu_block_reduce(part, red);
        if (tid == 0u) {
            float w = sofuu_exp(sum * scale - s_max);
            s_sum += w;
            s_w = w;
        }
        sofuu_bar_sync();
        float w = s_w;
        for (int d = (int)tid; d < head_dim; d += (int)stride) {
            acc[d] += w * vbase[(unsigned long long)p *
                                    (unsigned long long)head_dim +
                                (unsigned long long)d];
        }
        sofuu_bar_sync();
    }

    if (tid == 0u) s_w = (s_sum > 0.0f) ? 1.0f / s_sum : 0.0f;
    sofuu_bar_sync();
    float inv = s_w;
    for (int d = (int)tid; d < head_dim; d += (int)stride) {
        out[(unsigned long long)h * (unsigned long long)head_dim +
            (unsigned long long)d] = acc[d] * inv;
    }
}

// ── Residual add: y = a + b (in-place when y == a) ─────────────────────────
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_add_f32(const float *a, const float *b, float *y, int n) {
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    int base = (int)sofuu_ctaid_x() * (int)stride;
    int i = base + (int)tid;
    if (i < n) y[i] = a[i] + b[i];
}

// ── KV cache write (f32), one block per KV head ────────────────────────────
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128) void sofuu_kv_write_f32(float *k_cache, float *v_cache,
                                                const float *k, const float *v,
                                                int pos, int n_kv_heads,
                                                int head_dim, int cap) {
    int head = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    // The host validates this too; a kernel that trusts its arguments is one
    // refactor away from writing past the end of the cache.
    if (pos < 0 || pos >= cap || head >= n_kv_heads) {
        return;
    }
    unsigned long long slot =
        ((unsigned long long)head * (unsigned long long)cap +
         (unsigned long long)pos) *
        (unsigned long long)head_dim;
    const float *krow = k + (unsigned long long)head * (unsigned long long)head_dim;
    const float *vrow = v + (unsigned long long)head * (unsigned long long)head_dim;
    for (int d = (int)tid; d < head_dim; d += (int)stride) {
        k_cache[slot + (unsigned long long)d] = krow[d];
        v_cache[slot + (unsigned long long)d] = vrow[d];
    }
}
#endif // !SOFUU_CUDA_HOST

// ── Host algorithm reference (SOFUU_CUDA_HOST only) ────────────────────────
// These loop the SAME per-value helpers the kernels call. The GPU harness
// compares the real launches against src/cuda/parity/cpu_ref.h instead.
#if defined(SOFUU_CUDA_HOST)
extern "C" float sofuu_host_q8_0_dot(const void *w, const float *x, int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 32;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const sofuu_block_q8_0 *b =
            (const sofuu_block_q8_0 *)(p + (unsigned long long)bi * 34);
        const float *xb = x + (unsigned long long)bi * 32;
        for (int l = 0; l < 32; ++l) total += sofuu_q8_0_value(b, l, xb[l]);
    }
    return total;
}

extern "C" float sofuu_host_q4_k_dot(const void *w, const float *x, int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 256;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(p + (unsigned long long)bi * 144);
        const float *xb = x + (unsigned long long)bi * 256;
        for (int sub = 0; sub < 8; ++sub)
            for (int l = 0; l < 32; ++l)
                total += sofuu_q4k_value(b, sub, l, xb[sub * 32 + l]);
    }
    return total;
}

extern "C" float sofuu_host_q6_k_dot(const void *w, const float *x, int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 256;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const sofuu_block_q6_K *b =
            (const sofuu_block_q6_K *)(p + (unsigned long long)bi * 210);
        const float *xb = x + (unsigned long long)bi * 256;
        for (int run = 0; run < 8; ++run)
            for (int l = 0; l < 32; ++l)
                total += sofuu_q6k_value(b, run, l, xb[run * 32 + l]);
    }
    return total;
}

extern "C" void sofuu_host_rmsnorm(const float *x, const float *w, float *y,
                                   int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float s = sofuu_rsqrt(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * s * w[i];
}

extern "C" void sofuu_host_swiglu(const float *g, const float *u, float *y,
                                  int n) {
    for (int i = 0; i < n; ++i) y[i] = sofuu_silu(g[i]) * u[i];
}

extern "C" void sofuu_host_rope(float *x, const float *cos_tab,
                                const float *sin_tab, int n_heads, int head_dim,
                                int n_rot, int interleaved) {
    int half = n_rot >> 1;
    for (int h = 0; h < n_heads; ++h) {
        float *base = x + (unsigned long long)h * (unsigned long long)head_dim;
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

extern "C" void sofuu_host_kv_write(float *k_cache, float *v_cache,
                                    const float *k, const float *v, int pos,
                                    int n_kv_heads, int head_dim, int cap) {
    for (int h = 0; h < n_kv_heads; ++h) {
        unsigned long long slot =
            ((unsigned long long)h * (unsigned long long)cap +
             (unsigned long long)pos) *
            (unsigned long long)head_dim;
        const float *krow = k + (unsigned long long)h * (unsigned long long)head_dim;
        const float *vrow = v + (unsigned long long)h * (unsigned long long)head_dim;
        for (int d = 0; d < head_dim; ++d) {
            k_cache[slot + (unsigned long long)d] = krow[d];
            v_cache[slot + (unsigned long long)d] = vrow[d];
        }
    }
}

extern "C" void sofuu_host_attn_decode(const float *q, const float *k_cache,
                                       const float *v_cache, float *out,
                                       int n_heads, int n_kv_heads, int head_dim,
                                       int seq_len, int cap, float scale) {
    if (n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 || seq_len <= 0 ||
        seq_len > cap || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int group = n_heads / n_kv_heads;
    for (int h = 0; h < n_heads; ++h) {
        int kvh = h / group;
        const float *qv = q + (unsigned long long)h * (unsigned long long)head_dim;
        const float *kb = k_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        const float *vb = v_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        double mx = -1e300;
        for (int p = 0; p < seq_len; ++p) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d)
                dot += (double)qv[d] * (double)kb[(unsigned long long)p *
                                                      (unsigned long long)head_dim +
                                                  (unsigned long long)d];
            double score = dot * (double)scale;
            if (score > mx) mx = score;
        }
        double acc[256];
        for (int d = 0; d < head_dim; ++d) acc[d] = 0.0;
        double sum = 0.0;
        for (int p = 0; p < seq_len; ++p) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d)
                dot += (double)qv[d] * (double)kb[(unsigned long long)p *
                                                      (unsigned long long)head_dim +
                                                  (unsigned long long)d];
            double w = exp(dot * (double)scale - mx);
            sum += w;
            for (int d = 0; d < head_dim; ++d)
                acc[d] += w * (double)vb[(unsigned long long)p *
                                              (unsigned long long)head_dim +
                                          (unsigned long long)d];
        }
        for (int d = 0; d < head_dim; ++d)
            out[(unsigned long long)h * (unsigned long long)head_dim +
                (unsigned long long)d] =
                (float)(sum > 0.0 ? acc[d] / sum : 0.0);
    }
}

extern "C" void sofuu_host_add(const float *a, const float *b, float *y,
                               int n) {
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
}
#endif // SOFUU_CUDA_HOST
