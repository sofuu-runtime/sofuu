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
#define SOFUU_LAUNCH_BOUNDS_MIN(n, m)
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
// Same, plus a minimum resident blocks-per-SM target. The driver JIT trades
// registers against that floor; a decode GEMV that looks memory-bound is often
// register-bound (too few resident warps to cover load latency), and this is
// what makes the intended occupancy explicit instead of incidental.
#define SOFUU_LAUNCH_BOUNDS_MIN(n, m) __attribute__((launch_bounds(n, m)))
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

// No CUDA headers here, so float4 is not declared. 16-byte alignment is what
// lets the compiler emit one vector load for four activations.
typedef struct __attribute__((aligned(16))) {
    float x, y, z, w;
} sofuu_float4;

// Four words, 16-byte aligned: the quantized-weight stream reads whole 128-bit
// sectors, so one lane load is one `ld.global.v4.u32` instead of four 32-bit
// loads or eight byte loads. A q4_K block is exactly 144 = 9 x 16 bytes, and
// its 128-byte qs run starts at block+16, so every chunk of every block is
// 16-byte aligned in the GGUF's native layout (no repacking needed).
typedef struct __attribute__((aligned(16))) {
    unsigned int w[4];
} sofuu_u128;

#define SOFUU_QK_Q8_0 32
#define SOFUU_QK_K 256
#define SOFUU_Q4_K_BYTES 144
// Device-side q6_K stride. The on-disk GGML block is 210 bytes; upload pads
// every block to 224 (14 x 16) so ql (+0), qh (+128), scales (+192) and the
// super-block scale (+208) are all reachable with 16-byte vector loads and the
// stride itself is 16-byte aligned. The first 210 bytes of a block are
// untouched, so the padding is the only layout difference.
#define SOFUU_Q6_K_BYTES 224
#define SOFUU_Q8_0_BYTES 34
#define SOFUU_GEMV_THREADS 256

// ── Portable half -> float (exact for every finite binary16 value) ─────────
#if !defined(SOFUU_CUDA_HOST)
static SOFUU_DEVICE SOFUU_INLINE float sofuu_f16_to_f32(unsigned short h) {
    // The hardware widening is one instruction; the software decode below is
    // ~8 instructions per scale and twice per q4_K block, which is a
    // measurable share of a decode row-dot. Exact for every binary16 value,
    // including subnormals and inf.
    float v;
    __asm__("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h));
    return v;
}
#else
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
#endif

// ── Quantized-activation dot primitives ────────────────────────────────────
// The weights are int4/int6 and the activations are int8 (q8_1: one scale and
// one integer sum per 32 values). dp4a accumulates four int8 products into an
// int32 in one instruction, which replaces the per-value convert + FMA chain.
// The host build emulates it so the CPU parity gate covers the same algebra.
static SOFUU_DEVICE SOFUU_INLINE int sofuu_dp4a(int a, int b, int c) {
#if defined(SOFUU_CUDA_HOST)
    int sum = c;
    for (int k = 0; k < 4; ++k) {
        int av = (int)(signed char)((a >> (8 * k)) & 0xff);
        int bv = (int)(signed char)((b >> (8 * k)) & 0xff);
        sum += av * bv;
    }
    return sum;
#else
    int r;
    __asm__("dp4a.s32.s32 %0, %1, %2, %3;"
            : "=r"(r)
            : "r"(a), "r"(b), "r"(c));
    return r;
#endif
}

// Round to nearest even, the same mode the activation quantizer relies on.
static SOFUU_DEVICE SOFUU_INLINE int sofuu_round_int(float v) {
#if defined(SOFUU_CUDA_HOST)
    return (int)lrintf(v);
#else
    int r;
    __asm__("cvt.rni.s32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
#endif
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_absf(float v) {
    return v < 0.0f ? -v : v;
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_maxf(float a, float b) {
    return a > b ? a : b;
}

static SOFUU_DEVICE SOFUU_INLINE sofuu_u128 sofuu_load_u128(const void *p) {
    return *reinterpret_cast<const sofuu_u128 *>(p);
}

// Sixteen int8 values dotted against sixteen int8 weights: two independent
// dp4a chains joined by an add, so the chain depth is two, not four.
static SOFUU_DEVICE SOFUU_INLINE int sofuu_dot16(const unsigned int w[4],
                                                 const unsigned int x[4]) {
    int a = sofuu_dp4a((int)w[0], (int)x[0], 0);
    a = sofuu_dp4a((int)w[1], (int)x[1], a);
    int b = sofuu_dp4a((int)w[2], (int)x[2], 0);
    b = sofuu_dp4a((int)w[3], (int)x[3], b);
    return a + b;
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
// Butterfly shuffle: every lane converges on the same reduction, which is what
// the q8_1 quantizer needs (each lane scales and rounds its own value).
static SOFUU_DEVICE SOFUU_INLINE float sofuu_shfl_xor_f(float v, int lanemask) {
    float r;
    __asm__ volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                     : "=f"(r)
                     : "f"(v), "r"(lanemask));
    return r;
}
static SOFUU_DEVICE SOFUU_INLINE int sofuu_shfl_xor_i(int v, int lanemask) {
    int r;
    __asm__ volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                     : "=r"(r)
                     : "r"(v), "r"(lanemask));
    return r;
}
static SOFUU_DEVICE SOFUU_INLINE float sofuu_warp_max_f(float v) {
    v = sofuu_maxf(v, sofuu_shfl_xor_f(v, 16));
    v = sofuu_maxf(v, sofuu_shfl_xor_f(v, 8));
    v = sofuu_maxf(v, sofuu_shfl_xor_f(v, 4));
    v = sofuu_maxf(v, sofuu_shfl_xor_f(v, 2));
    v = sofuu_maxf(v, sofuu_shfl_xor_f(v, 1));
    return v;
}
static SOFUU_DEVICE SOFUU_INLINE int sofuu_warp_sum_i(int v) {
    v += sofuu_shfl_xor_i(v, 16);
    v += sofuu_shfl_xor_i(v, 8);
    v += sofuu_shfl_xor_i(v, 4);
    v += sofuu_shfl_xor_i(v, 2);
    v += sofuu_shfl_xor_i(v, 1);
    return v;
}
// Sum within each 16-lane half. q6_K scales change every 16 values, so its
// bias correction needs the half sums, not just the 32-value total.
static SOFUU_DEVICE SOFUU_INLINE int sofuu_warp_sum16_i(int v) {
    v += sofuu_shfl_xor_i(v, 8);
    v += sofuu_shfl_xor_i(v, 4);
    v += sofuu_shfl_xor_i(v, 2);
    v += sofuu_shfl_xor_i(v, 1);
    return v;
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

static SOFUU_DEVICE SOFUU_INLINE void sofuu_load8(const float *p, float *o) {
    sofuu_float4 a = *reinterpret_cast<const sofuu_float4 *>(p);
    sofuu_float4 b = *reinterpret_cast<const sofuu_float4 *>(p + 4);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
    o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_dot8(
    const sofuu_block_q4_K *blk, int sub, int l0, const float *xr) {
    unsigned int sc, mn;
    sofuu_q4k_scale_min(sub, blk->scales, &sc, &mn);
    float d = sofuu_f16_to_f32(blk->d);
    float dmin = sofuu_f16_to_f32(blk->dmin);
    float s = d * (float)sc;
    float m = dmin * (float)mn;
    unsigned long long qv = *reinterpret_cast<const unsigned long long *>(
        blk->qs + (sub >> 1) * 32 + l0);
    unsigned int high = (unsigned int)(sub & 1);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        unsigned int q = high ? (unsigned int)((qv >> (8 * k + 4)) & 0x0fu)
                              : (unsigned int)((qv >> (8 * k)) & 0x0fu);
        acc += (s * (float)q - m) * xr[k];
    }
    return acc;
}

// q4_K x q8_1, one lane's 32 values held in a single 16-byte qs chunk. The
// low nibbles are 16 values of sub-block 2k, the high nibbles the same 16
// positions of sub-block 2k+1, so the two halves have their own scale/min pair
// and their own q8_1 scale and 32-value (here: 16-value) integer sum. The min
// term no longer needs a designated lane: every lane owns whole halves, so the
// corrections partition the row exactly once.
static SOFUU_DEVICE SOFUU_INLINE void sofuu_q4k_x_load(
    const signed char *xq, int b, int chunk, sofuu_u128 *xlo,
    sofuu_u128 *xhi) {
    const int k = chunk >> 1, h = chunk & 1;
    const unsigned long long base =
        (unsigned long long)b * 256 + (unsigned long long)(64 * k + 16 * h);
    *xlo = sofuu_load_u128(xq + base);
    *xhi = sofuu_load_u128(xq + base + 32);
}

static SOFUU_DEVICE SOFUU_INLINE void sofuu_q4k_w_load(
    const sofuu_block_q4_K *blk, int chunk, sofuu_u128 *q) {
    const int k = chunk >> 1, h = chunk & 1;
    *q = sofuu_load_u128(blk->qs + 32 * k + 16 * h);
}

static SOFUU_DEVICE SOFUU_INLINE void sofuu_q4k_chunk32_load(
    const sofuu_block_q4_K *blk, const signed char *xrow, int chunk,
    sofuu_u128 *q, sofuu_u128 *xlo, sofuu_u128 *xhi) {
    sofuu_q4k_w_load(blk, chunk, q);
    sofuu_q4k_x_load(xrow, 0, chunk, xlo, xhi);
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_chunk32_compute(
    const sofuu_u128 q, const sofuu_u128 xlo, const sofuu_u128 xhi,
    const sofuu_block_q4_K *blk, int chunk, float sx_lo, int isum_lo,
    float sx_hi, int isum_hi) {
    const int k = chunk >> 1, h = chunk & 1;
    const unsigned int M4 = 0x0f0f0f0fu;
    unsigned int qlo[4], qhi[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        qlo[t] = q.w[t] & M4;
        qhi[t] = (q.w[t] >> 4) & M4;
    }
    unsigned int sc_lo, mn_lo, sc_hi, mn_hi;
    sofuu_q4k_scale_min(2 * k, blk->scales, &sc_lo, &mn_lo);
    sofuu_q4k_scale_min(2 * k + 1, blk->scales, &sc_hi, &mn_hi);
    float d = sofuu_f16_to_f32(blk->d);
    float lo = (d * (float)sc_lo) * (float)sofuu_dot16(qlo, xlo.w);
    float hi = (d * (float)sc_hi) * (float)sofuu_dot16(qhi, xhi.w);
    // The min term needs the sum over all 32 values of a sub-block; a chunk
    // holds only one of its two 16-value halves, so the h == 0 lane of each
    // (block, k) applies it once for both sub-blocks and nothing double-counts.
    if (h == 0) {
        float dmin = sofuu_f16_to_f32(blk->dmin);
        lo -= (dmin * (float)mn_lo) * (float)isum_lo;
        hi -= (dmin * (float)mn_hi) * (float)isum_hi;
    }
    return sx_lo * lo + sx_hi * hi;
}

// Load-then-compute for callers that do not prefetch (the CPU mirror).
static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_chunk32_q8(
    const sofuu_block_q4_K *blk, const signed char *xrow, int chunk, int b,
    const float *xd, const int *xs) {
    sofuu_u128 q, xlo, xhi;
    sofuu_q4k_chunk32_load(blk, xrow, chunk, &q, &xlo, &xhi);
    const int ilo = b * 8 + 2 * (chunk >> 1);
    return sofuu_q4k_chunk32_compute(q, xlo, xhi, blk, chunk, xd[ilo], xs[ilo],
                                     xd[ilo + 1], xs[ilo + 1]);
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_q6k_dot8(
    const sofuu_block_q6_K *b, int sub, int l0, const float *xr) {    int n = sub >> 2;
    int r = sub & 3;
    float d = sofuu_f16_to_f32(b->d);
    int sbase = n * 8 + 2 * r;
    float s = (l0 >= 16) ? (float)b->scales[sbase + 1]
                         : (float)b->scales[sbase];
    float ds = d * s;
    int ql_base = n * 64 + ((r == 1 || r == 3) ? 32 : 0);
    const unsigned char *qh = b->qh + n * 32;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int l = l0 + k;
        unsigned int ql_byte = (unsigned int)b->ql[ql_base + l];
        unsigned int nib = (r < 2) ? (ql_byte & 0x0fu) : (ql_byte >> 4);
        unsigned int hi2 = ((unsigned int)qh[l] >> (2 * r)) & 3u;
        int q = (int)(nib | (hi2 << 4)) - 32;
        acc += ds * (float)q * xr[k];
    }
    return acc;
}

static SOFUU_DEVICE SOFUU_INLINE int sofuu_q6k_val_q8(unsigned int ql,
                                                      unsigned int qh, int high,
                                                      int sh) {
    unsigned int nib = high ? ((ql >> 4) & 0x0f0f0f0fu) : (ql & 0x0f0f0f0fu);
    unsigned int hi = (sh < 4) ? ((qh << (4 - sh)) & 0x30303030u)
                               : ((qh >> (sh - 4)) & 0x30303030u);
    return (int)(nib | hi);
}

// Sixteen values of one run: 16 ql bytes, 16 qh bytes, 16 activation bytes.
static SOFUU_DEVICE SOFUU_INLINE int sofuu_q6k_dot16_q8(
    const sofuu_u128 ql, const sofuu_u128 qh, const sofuu_u128 x, int high,
    int sh) {
    int a = sofuu_dp4a(sofuu_q6k_val_q8(ql.w[0], qh.w[0], high, sh), (int)x.w[0], 0);
    a = sofuu_dp4a(sofuu_q6k_val_q8(ql.w[1], qh.w[1], high, sh), (int)x.w[1], a);
    int b = sofuu_dp4a(sofuu_q6k_val_q8(ql.w[2], qh.w[2], high, sh), (int)x.w[2], 0);
    b = sofuu_dp4a(sofuu_q6k_val_q8(ql.w[3], qh.w[3], high, sh), (int)x.w[3], b);
    return a + b;
}

static SOFUU_DEVICE SOFUU_INLINE int sofuu_i8_at(const sofuu_u128 v, int s) {
    return (int)(signed char)((v.w[s >> 2] >> (8 * (s & 3))) & 0xffu);
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_q8_0_dot8(
    const sofuu_block_q8_0 *b, int lane_in_block, const float *xr) {
    float d = sofuu_f16_to_f32(b->d);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc += d * (float)b->qs[lane_in_block * 8 + k] * xr[k];
    }
    return acc;
}

static SOFUU_DEVICE SOFUU_INLINE float sofuu_f32_dot8(const float *row,
                                                      const float *xr) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) acc += row[k] * xr[k];
    return acc;
}

// ── GEMV kernels (decode) ──────────────────────────────────────────────────
// Decode streams weights, and the measured control (add_f32 at 263 GB/s) says
// the memory system is not the wall: the row-dot pipeline is. Two properties
// of the original layout cost throughput:
//
//   * a per-256-value shared activation tile needed two block barriers per
//     chunk, which lockstepped all eight row-warps and allowed only one
//     chunk's weight loads in flight per block;
//   * one 64-bit qs load per warp iteration left two sectors outstanding,
//     not enough memory-level parallelism to cover DRAM latency.
//
// The kernels below read the activation row directly (the block's eight warps
// read the same row, so it is an L1 hit after the first) and keep four
// independent super-block loads in flight. The per-value math and its
// accumulation order are unchanged: lane l still owns eight consecutive values
// per 256-value block, reduced with warp shuffles.
#if !defined(SOFUU_CUDA_HOST)
static SOFUU_DEVICE SOFUU_INLINE float sofuu_warp_sum(float v) {
    v += sofuu_shfl_down(v, 16, 0xffffffffu);
    v += sofuu_shfl_down(v, 8, 0xffffffffu);
    v += sofuu_shfl_down(v, 4, 0xffffffffu);
    v += sofuu_shfl_down(v, 2, 0xffffffffu);
    v += sofuu_shfl_down(v, 1, 0xffffffffu);
    return v;
}

// q8_0: 4 lanes per 32-value block, each owns 8 values. A warp covers eight
// blocks (256 values) per iteration, so a 1024-wide row is four iterations.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4) void sofuu_gemv_q8_0(const unsigned char *__restrict__ w,
                                             const float *__restrict__ x, float *__restrict__ y,
                                             int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_Q8_0;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q8_0_BYTES;
    int lane_in_block = lane & 3;
    int block_in_warp = lane >> 2;
    float acc = 0.0f;
    int cg = 0;
    for (; cg + 512 <= cols; cg += 512) {
        // Two 256-value chunks per iteration: two independent qs loads.
        float xa[8], xb[8];
        sofuu_load8(x + cg + lane * 8, xa);
        sofuu_load8(x + cg + 256 + lane * 8, xb);
        int bi0 = (cg >> 5) + block_in_warp;
        int bi1 = ((cg + 256) >> 5) + block_in_warp;
        const sofuu_block_q8_0 *b0 =
            (const sofuu_block_q8_0 *)(roww + (unsigned long long)bi0 * 34);
        const sofuu_block_q8_0 *b1 =
            (const sofuu_block_q8_0 *)(roww + (unsigned long long)bi1 * 34);
        acc += sofuu_q8_0_dot8(b0, lane_in_block, xa);
        acc += sofuu_q8_0_dot8(b1, lane_in_block, xb);
    }
    for (; cg < cols; cg += 256) {
        int col = cg + lane * 8;
        if (col < cols) {
            float xr[8];
            sofuu_load8(x + col, xr);
            int bi = (cg >> 5) + block_in_warp;
            const sofuu_block_q8_0 *b =
                (const sofuu_block_q8_0 *)(roww + (unsigned long long)bi * 34);
            acc += sofuu_q8_0_dot8(b, lane_in_block, xr);
        }
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
}

// q4_K: a warp covers one 256-value super-block per iteration; lane l owns the
// eight values of sub-block l/4 at offset (l%4)*8. Lanes 0-3 and 4-7 read the
// same 32 qs bytes (low/high nibble), which the coalescer merges.
//
// Two super-blocks in flight, not four: the T4 gate measured 64 registers
// (the four-block cap) with four, and both q4_K and q6_K pinned that cap.
// Halving the live activation set trades per-warp ILP for resident warps.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 6) void sofuu_gemv_q4_k(const unsigned char *__restrict__ w,
                                             const float *__restrict__ x, float *__restrict__ y,
                                             int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q4_K_BYTES;
    int sub = lane >> 2;
    int l0 = (lane & 3) << 3;
    float acc = 0.0f;
    int bi = 0;
    // qs is 16-byte aligned inside every 144-byte block, so each lane's
    // 8-byte load is naturally aligned; the scale/min bytes share one 32-byte
    // sector with d/dmin.
    for (; bi + 2 <= nb; bi += 2) {
        const sofuu_block_q4_K *b0 =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)(bi + 0) * 144);
        const sofuu_block_q4_K *b1 =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)(bi + 1) * 144);
        float xr0[8], xr1[8];
        sofuu_load8(x + (bi + 0) * 256 + lane * 8, xr0);
        sofuu_load8(x + (bi + 1) * 256 + lane * 8, xr1);
        acc += sofuu_q4k_dot8(b0, sub, l0, xr0);
        acc += sofuu_q4k_dot8(b1, sub, l0, xr1);
    }
    for (; bi < nb; ++bi) {
        float xr[8];
        sofuu_load8(x + bi * 256 + lane * 8, xr);
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)bi * 144);
        acc += sofuu_q4k_dot8(b, sub, l0, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
}

// ── q8_1 activation quantization ───────────────────────────────────────────
// One warp per 32-value block: amax -> d = amax/127, q = round(x/d), and the
// integer sum the q4_K min term needs. The butterfly reduction leaves every
// lane with the block max, so each lane rounds its own value without a
// broadcast.
// One 32-value block, one value per lane: amax -> d = amax/127, q = round(x/d),
// the 32-value sum and the two 16-value halves. Shared by the standalone
// quantizer and the fused kernels below, so the activation encoding has exactly
// one definition.
static SOFUU_DEVICE SOFUU_INLINE void sofuu_quantize_block32(
    const float *x, signed char *q, float *d, int *s, int *s16, int block,
    int lane) {
    int base = block * 32;
    float v = x[base + lane];
    float amax = sofuu_warp_max_f(sofuu_absf(v));
    float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int qi = sofuu_round_int(v * inv);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    q[base + lane] = (signed char)qi;
    int sum = sofuu_warp_sum_i(qi);
    int half = sofuu_warp_sum16_i(qi);
    if ((lane & 15u) == 0u) s16[block * 2 + (lane >> 4)] = half;
    if (lane == 0) {
        d[block] = scale;
        s[block] = sum;
    }
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_quantize_q8_1(const float *__restrict__ x, signed char *__restrict__ q,
                         float *__restrict__ d, int *__restrict__ s,
                         int *__restrict__ s16, int n) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int block = (int)sofuu_ctaid_x() * 8 + warp;
    if (block * 32 >= n) return;
    sofuu_quantize_block32(x, q, d, s, s16, block, lane);
}

// q4_K x q8_1: eight lanes cover one block (each holds a 16-byte qs chunk), so
// a warp spans four blocks per instruction and a 2048-column row is two
// iterations whose six 128-bit loads all issue before the first dp4a. That is
// the difference to the earlier 8-byte version: same arithmetic, 4x the weight
// bytes per load instruction and 6x the bytes in flight, which is what a decode
// row-dot needs to cover DRAM latency.
// One row's lane-partial accumulator over `nb` q4_K blocks; the caller finishes
// with a warp sum. The fused gate+up kernel below carries its own loop (it
// loads the activation chunk once for two matrices), so this one stays the
// shape the single-matrix kernel wants.
static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_row_q8(
    const unsigned char *__restrict__ roww, const signed char *__restrict__ xq,
    const float *__restrict__ xd, const int *__restrict__ xs, int nb, int chunk,
    int bgrp, int k) {
    float acc = 0.0f;
    int b = bgrp;
    // Pairs of four-block groups: both groups' weight and activation loads are
    // signed to this warp, issued back to back, then consumed.
    for (; b + 4 < nb; b += 8) {
        const sofuu_block_q4_K *p0 = (const sofuu_block_q4_K *)(
            roww + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        const sofuu_block_q4_K *p1 = (const sofuu_block_q4_K *)(
            roww + (unsigned long long)(b + 4) * SOFUU_Q4_K_BYTES);
        sofuu_u128 q0, l0, h0, q1, l1, h1;
        sofuu_q4k_chunk32_load(p0, xq + (unsigned long long)b * 256, chunk, &q0,
                               &l0, &h0);
        sofuu_q4k_chunk32_load(p1, xq + (unsigned long long)(b + 4) * 256, chunk,
                               &q1, &l1, &h1);
        int i0 = b * 8 + 2 * k, i1 = (b + 4) * 8 + 2 * k;
        acc += sofuu_q4k_chunk32_compute(q0, l0, h0, p0, chunk, xd[i0], xs[i0],
                                         xd[i0 + 1], xs[i0 + 1]);
        acc += sofuu_q4k_chunk32_compute(q1, l1, h1, p1, chunk, xd[i1], xs[i1],
                                         xd[i1 + 1], xs[i1 + 1]);
    }
    for (; b < nb; b += 4) {
        const sofuu_block_q4_K *p = (const sofuu_block_q4_K *)(
            roww + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        acc += sofuu_q4k_chunk32_q8(p, xq + (unsigned long long)b * 256, chunk,
                                   b, xd, xs);
    }
    return acc;
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemv_q4_k_q8(const unsigned char *__restrict__ w,
                        const signed char *__restrict__ xq,
                        const float *__restrict__ xd,
                        const int *__restrict__ xs,
                        float *__restrict__ y, int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q4_K_BYTES;
    const int chunk = lane & 7;  // 16-byte qs chunk inside a block
    const int bgrp = lane >> 3;  // which of four blocks this lane feeds
    float acc = sofuu_q4k_row_q8(roww, xq, xd, xs, nb, chunk, bgrp, chunk >> 1);
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
}

// gate_proj and up_proj share the normed row, the shape and (in every Q4_K_M
// model) the quant type, so a decode step launched them back to back over the
// same activation. That cost two dispatches and two dependent stalls for what
// is one 2x-longer weight stream: the mixed launch bench puts a single
// intervening kernel at ~8 us, which on a 9.4 MiB matrix is a fifth of its
// runtime. The activation chunk is loaded once and fed to both matrices, so the
// per-lane dp4a count doubles while the load count only grows by the weights.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemv_q4_k_q8_x2(const unsigned char *__restrict__ w0,
                           const unsigned char *__restrict__ w1,
                           const signed char *__restrict__ xq,
                           const float *__restrict__ xd,
                           const int *__restrict__ xs,
                           float *__restrict__ y0, float *__restrict__ y1,
                           int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_K;
    unsigned long long stride = (unsigned long long)nb * SOFUU_Q4_K_BYTES;
    const unsigned char *r0 = w0 + (unsigned long long)row * stride;
    const unsigned char *r1 = w1 + (unsigned long long)row * stride;
    const int chunk = lane & 7;
    const int bgrp = lane >> 3;
    const int k = chunk >> 1;
    float acc0 = 0.0f, acc1 = 0.0f;
    int b = bgrp;
    for (; b + 4 < nb; b += 8) {
        const sofuu_block_q4_K *a0 = (const sofuu_block_q4_K *)(
            r0 + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        const sofuu_block_q4_K *c0 = (const sofuu_block_q4_K *)(
            r1 + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        const sofuu_block_q4_K *a1 = (const sofuu_block_q4_K *)(
            r0 + (unsigned long long)(b + 4) * SOFUU_Q4_K_BYTES);
        const sofuu_block_q4_K *c1 = (const sofuu_block_q4_K *)(
            r1 + (unsigned long long)(b + 4) * SOFUU_Q4_K_BYTES);
        sofuu_u128 wa0, wc0, wa1, wc1, xlo0, xhi0, xlo1, xhi1;
        // Four weight loads and two activation loads signed to this warp before
        // the first dp4a; the activation pairs are then used twice each.
        sofuu_q4k_w_load(a0, chunk, &wa0);
        sofuu_q4k_w_load(c0, chunk, &wc0);
        sofuu_q4k_x_load(xq + (unsigned long long)b * 256, 0, chunk, &xlo0,
                         &xhi0);
        sofuu_q4k_w_load(a1, chunk, &wa1);
        sofuu_q4k_w_load(c1, chunk, &wc1);
        sofuu_q4k_x_load(xq + (unsigned long long)(b + 4) * 256, 0, chunk,
                         &xlo1, &xhi1);
        int i0 = b * 8 + 2 * k, i1 = (b + 4) * 8 + 2 * k;
        acc0 += sofuu_q4k_chunk32_compute(wa0, xlo0, xhi0, a0, chunk, xd[i0],
                                          xs[i0], xd[i0 + 1], xs[i0 + 1]);
        acc1 += sofuu_q4k_chunk32_compute(wc0, xlo0, xhi0, c0, chunk, xd[i0],
                                          xs[i0], xd[i0 + 1], xs[i0 + 1]);
        acc0 += sofuu_q4k_chunk32_compute(wa1, xlo1, xhi1, a1, chunk, xd[i1],
                                          xs[i1], xd[i1 + 1], xs[i1 + 1]);
        acc1 += sofuu_q4k_chunk32_compute(wc1, xlo1, xhi1, c1, chunk, xd[i1],
                                          xs[i1], xd[i1 + 1], xs[i1 + 1]);
    }
    for (; b < nb; b += 4) {
        const sofuu_block_q4_K *a = (const sofuu_block_q4_K *)(
            r0 + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        const sofuu_block_q4_K *c = (const sofuu_block_q4_K *)(
            r1 + (unsigned long long)b * SOFUU_Q4_K_BYTES);
        const signed char *xrow = xq + (unsigned long long)b * 256;
        acc0 += sofuu_q4k_chunk32_q8(a, xrow, chunk, b, xd, xs);
        acc1 += sofuu_q4k_chunk32_q8(c, xrow, chunk, b, xd, xs);
    }
    float t0 = sofuu_warp_sum(acc0);
    float t1 = sofuu_warp_sum(acc1);
    if (lane == 0) {
        y0[row] = t0;
        y1[row] = t1;
    }
}

// Four lanes per block, 64 values each, so a warp covers eight blocks and a
// 2048-column row is a single iteration of four 128-bit weight loads. Every
// lane covers runs (r, r+2) of one 128-value half, which share one 32-byte ql
// run and one 32-byte qh run: the low nibbles feed the first run, the high
// nibbles the second, and the qh bit pair is 2r and 2r+4.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_gemv_q6_k_q8(const unsigned char *__restrict__ w,
                        const signed char *__restrict__ xq,
                        const float *__restrict__ xd,
                        const int *__restrict__ xs,
                        const int *__restrict__ xs16,
                        float *__restrict__ y, int rows, int cols) {
    (void)xs;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q6_K_BYTES;
    const int chunk = lane & 3;  // 64 values of a block
    const int bgrp = lane >> 2;  // one of eight blocks per warp
    const int n = chunk >> 1, j = chunk & 1;
    const int sh_a = 2 * j, sh_b = sh_a + 4;
    const int slot = n * 8 + 2 * j;
    float acc = 0.0f;
    for (int b = bgrp; b < nb; b += 8) {
        const unsigned char *blk =
            roww + (unsigned long long)b * SOFUU_Q6_K_BYTES;
        sofuu_u128 ql0, ql1, qh0, qh1;
        ql0 = sofuu_load_u128(blk + 64 * n + 32 * j);
        ql1 = sofuu_load_u128(blk + 64 * n + 32 * j + 16);
        qh0 = sofuu_load_u128(blk + 128 + 32 * n);
        qh1 = sofuu_load_u128(blk + 128 + 32 * n + 16);
        const int ia = b * 8 + n * 4 + j;  // q8_1 block of the first run
        const int ib = ia + 2;             // second run, two runs later
        sofuu_u128 xa0, xa1, xb0, xb1;
        xa0 = sofuu_load_u128(xq + (unsigned long long)ia * 32);
        xa1 = sofuu_load_u128(xq + (unsigned long long)ia * 32 + 16);
        xb0 = sofuu_load_u128(xq + (unsigned long long)ib * 32);
        xb1 = sofuu_load_u128(xq + (unsigned long long)ib * 32 + 16);
        sofuu_u128 scv = sofuu_load_u128(blk + 192);
        float d = sofuu_f16_to_f32((unsigned short)(
            sofuu_load_u128(blk + 208).w[0] & 0xffffu));
        float ca_lo = xd[ia] * d * (float)sofuu_i8_at(scv, slot);
        float ca_hi = xd[ia] * d * (float)sofuu_i8_at(scv, slot + 1);
        float cb_lo = xd[ib] * d * (float)sofuu_i8_at(scv, slot + 4);
        float cb_hi = xd[ib] * d * (float)sofuu_i8_at(scv, slot + 5);
        int ka = sofuu_q6k_dot16_q8(ql0, qh0, xa0, 0, sh_a);
        int kb = sofuu_q6k_dot16_q8(ql1, qh1, xa1, 0, sh_a);
        int kc = sofuu_q6k_dot16_q8(ql0, qh0, xb0, 1, sh_b);
        int kd = sofuu_q6k_dot16_q8(ql1, qh1, xb1, 1, sh_b);
        // Six-bit values are stored biased (+32) so they stay positive for
        // dp4a; each 16-value half removes 32 * sum(x) before its scale.
        acc += ca_lo * (float)(ka - 32 * xs16[2 * ia]);
        acc += ca_hi * (float)(kb - 32 * xs16[2 * ia + 1]);
        acc += cb_lo * (float)(kc - 32 * xs16[2 * ib]);
        acc += cb_hi * (float)(kd - 32 * xs16[2 * ib + 1]);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
}

// q6_K: same warp/lane mapping as q4_K; ql at the run's 64-byte base, qh at
// n*32+l. The 210-byte block stride is not 8-byte aligned on odd blocks, so
// these stay scalar byte loads.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 6) void sofuu_gemv_q6_k(const unsigned char *__restrict__ w,
                                             const float *__restrict__ x, float *__restrict__ y,
                                             int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    int nb = cols / SOFUU_QK_K;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q6_K_BYTES;
    int sub = lane >> 2;
    int l0 = (lane & 3) << 3;
    float acc = 0.0f;
    int bi = 0;
    for (; bi + 2 <= nb; bi += 2) {
        const sofuu_block_q6_K *b0 =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)(bi + 0) * SOFUU_Q6_K_BYTES);
        const sofuu_block_q6_K *b1 =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)(bi + 1) * SOFUU_Q6_K_BYTES);
        float xr0[8], xr1[8];
        sofuu_load8(x + (bi + 0) * 256 + lane * 8, xr0);
        sofuu_load8(x + (bi + 1) * 256 + lane * 8, xr1);
        acc += sofuu_q6k_dot8(b0, sub, l0, xr0);
        acc += sofuu_q6k_dot8(b1, sub, l0, xr1);
    }
    for (; bi < nb; ++bi) {
        float xr[8];
        sofuu_load8(x + bi * 256 + lane * 8, xr);
        const sofuu_block_q6_K *b =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)bi * SOFUU_Q6_K_BYTES);
        acc += sofuu_q6k_dot8(b, sub, l0, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
}

// f32 reference GEMV (also the C1 correctness floor). A value is 4 B, so x and
// weights are the same size.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4) void sofuu_gemv_f32(const float *__restrict__ w, const float *__restrict__ x,
                                            float *__restrict__ y, int rows, int cols) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int row = (int)sofuu_ctaid_x() * 8 + warp;
    if (row >= rows) return;
    const float *roww = w + (unsigned long long)row * (unsigned long long)cols;
    float acc = 0.0f;
    int cg = 0;
    for (; cg + 512 <= cols; cg += 512) {
        float xa[8], xb[8];
        sofuu_load8(x + cg + lane * 8, xa);
        sofuu_load8(x + cg + 256 + lane * 8, xb);
        acc += sofuu_f32_dot8(roww + cg + lane * 8, xa);
        acc += sofuu_f32_dot8(roww + cg + 256 + lane * 8, xb);
    }
    for (; cg < cols; cg += 256) {
        int col = cg + lane * 8;
        if (col + 8 <= cols) {
            float xr[8];
            sofuu_load8(x + col, xr);
            acc += sofuu_f32_dot8(roww + col, xr);
        } else {
            for (int k = 0; k < 8 && col + k < cols; ++k) {
                acc += roww[col + k] * x[col + k];
            }
        }
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0) y[row] = total;
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

// ── Fused norm / swiglu + q8_1 quantization ────────────────────────────────
// The decode step normalizes a row and immediately quantizes it for the dp4a
// GEMV. As two kernels that is two launches, two global round trips and a
// dependent stall for 2048 values; one block now does the reduction, the write
// and (after a barrier) the quantization, with the same arithmetic.
//
// `sofuu_rmsnorm_add_q8` also swallows the residual add that always precedes
// the FFN norm, so the profile of a layer loses three launches there too.
static SOFUU_DEVICE SOFUU_INLINE void sofuu_quantize_row_blocks(
    const float *y, signed char *q, float *d, int *s, int *s16, int first,
    int count, int lane) {
    for (int j = 0; j < count; ++j) {
        sofuu_quantize_block32(y, q, d, s, s16, first + j, lane);
    }
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_rmsnorm_q8(const float *__restrict__ x, const float *__restrict__ w,
                      float *__restrict__ y, signed char *__restrict__ q,
                      float *__restrict__ d, int *__restrict__ s,
                      int *__restrict__ s16, int n, float eps) {
    unsigned int tid = sofuu_tid_x();
    unsigned int nt = sofuu_ntid_x();
    float ss = 0.0f;
    for (int i = (int)tid; i < n; i += (int)nt) ss += x[i] * x[i];
    SOFUU_SHARED float red[8];
    SOFUU_SHARED float scale;
    float total = sofuu_block_reduce(ss, red);
    if (tid == 0u) scale = sofuu_rsqrt(total / (float)n + eps);
    sofuu_bar_sync();
    float sc = scale;
    for (int i = (int)tid; i < n; i += (int)nt) y[i] = x[i] * sc * w[i];
    // The quantization reads values other warps just wrote.
    sofuu_bar_sync();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int per = n / 32 / 8;
    sofuu_quantize_row_blocks(y, q, d, s, s16, warp * per, per, lane);
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_rmsnorm_add_q8(const float *__restrict__ a,
                          const float *__restrict__ b, const float *__restrict__ w,
                          float *__restrict__ xres, float *__restrict__ y,
                          signed char *__restrict__ q, float *__restrict__ d,
                          int *__restrict__ s, int *__restrict__ s16, int n,
                          float eps) {
    unsigned int tid = sofuu_tid_x();
    unsigned int nt = sofuu_ntid_x();
    float ss = 0.0f;
    for (int i = (int)tid; i < n; i += (int)nt) {
        float v = a[i] + b[i];
        ss += v * v;
    }
    SOFUU_SHARED float red[8];
    SOFUU_SHARED float scale;
    float total = sofuu_block_reduce(ss, red);
    if (tid == 0u) scale = sofuu_rsqrt(total / (float)n + eps);
    sofuu_bar_sync();
    float sc = scale;
    for (int i = (int)tid; i < n; i += (int)nt) {
        float v = a[i] + b[i];
        xres[i] = v;
        y[i] = v * sc * w[i];
    }
    sofuu_bar_sync();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int per = n / 32 / 8;
    sofuu_quantize_row_blocks(y, q, d, s, s16, warp * per, per, lane);
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_swiglu_q8(const float *__restrict__ g, const float *__restrict__ u,
                     float *__restrict__ y, signed char *__restrict__ q,
                     float *__restrict__ d, int *__restrict__ s,
                     int *__restrict__ s16, int n) {
    unsigned int tid = sofuu_tid_x();
    int i = (int)sofuu_ctaid_x() * (int)sofuu_ntid_x() + (int)tid;
    if (i < n) y[i] = sofuu_silu(g[i]) * u[i];
    sofuu_bar_sync();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    // One block covers 8 q8 blocks, one per warp: every warp reads values the
    // block wrote, hence the barrier.
    sofuu_quantize_block32(y, q, d, s, s16, (int)sofuu_ctaid_x() * 8 + warp, lane);
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
                           int n_kv_heads, int head_dim, const int *seq_dev,
                           int cap, float scale, signed char *aq, float *ad,
                           int *as, int *as16, int do_quant) {
    int seq_len = *seq_dev;
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

    // The o_proj GEMV needs this row quantized to q8_1, which used to be the
    // very next launch: 2048 values round-tripping through DRAM behind a
    // dependent stall. A block owns exactly one head's values, so it can
    // quantize them itself once the barrier has published the writes --
    // `head_dim / 32` warps, each taking one 32-value block. Same per-block
    // arithmetic as the standalone quantizer, so the bytes match.
    if (do_quant != 0) {
        sofuu_bar_sync();
        int nb_head = head_dim >> 5;
        int warp = (int)(tid >> 5);
        int lane = (int)(tid & 31u);
        if (warp < nb_head) {
            sofuu_quantize_block32(out, aq, ad, as, as16,
                                   h * nb_head + warp, lane);
        }
    }
}

// ── Decode attention, split-K (FlashDecoding shape) ───────────────────────
//
// The block-per-head kernel above waits a full memory round trip per cached
// position: at 2402 tokens it measured 69 ms of a 73 ms token. The causes are
// structural, not arithmetic -- 32 warps existed for 40 SMs, each doing 2402
// dependent 256-byte reads twice, so neither the SM nor the warp had anything
// to overlap the latency with.
//
// This pair keeps the softmax exact but gives the problem the parallelism it
// always had:
//   * positions are cut into `n_splits` ranges, so the grid is
//     n_heads * n_splits warps (512 here instead of 32) and an SM holds enough
//     resident warps to cover DRAM latency;
//   * each warp loads four positions' keys AND values into registers before
//     consuming any of them, so one stall amortizes four dots;
//   * the per-split partials (max, sum, weighted V) are merged by a second
//     kernel with the split-softmax identity -- reweight each split by
//     exp(m_s - M) -- which is exact, not an approximation.
//
// `chunk = ceil(cap / n_splits)` is derived from the CACHE size, never from the
// sequence length, so a captured graph stays valid as the position advances:
// splits past seq_len write an empty partial and cost one block each.
//
// `npoh` (values per lane) is a template parameter because register arrays need
// compile-time indices; the host dispatch covers head_dim 32/64/128, which is
// every model in the tree, and the model falls back to the block kernel
// otherwise.
static SOFUU_DEVICE SOFUU_INLINE float sofuu_warp_sum_all(float v) {
    v += sofuu_shfl_xor_f(v, 16);
    v += sofuu_shfl_xor_f(v, 8);
    v += sofuu_shfl_xor_f(v, 4);
    v += sofuu_shfl_xor_f(v, 2);
    v += sofuu_shfl_xor_f(v, 1);
    return v;
}

// One warp computes the attention of one query head over one position range and
// writes (m, s, acc[head_dim]) for the merge.
//
// The range is passed in, not derived from blockIdx: decode gives every (head,
// split) a range over the shared cache, while prefill gives every (token, head,
// split) a range that ends at that token's own position. Same arithmetic, so
// both go through here.
template <int NPOS>
static SOFUU_DEVICE SOFUU_INLINE void sofuu_attn_split_range(
    const float *__restrict__ qv_row, const float *__restrict__ kb,
    const float *__restrict__ vb, float *__restrict__ dst, int cap, float scale,
    int p0, int p1, int lane) {
    const int hd = NPOS * 32;
    float qv[NPOS];
#pragma unroll
    for (int i = 0; i < NPOS; ++i) qv[i] = qv_row[(unsigned long long)lane + 32 * i];

    float m = -3.402823466e38f;
    float s = 0.0f;
    float acc[NPOS];
#pragma unroll
    for (int i = 0; i < NPOS; ++i) acc[i] = 0.0f;

    for (int p = p0; p < p1; p += 4) {
        int n = p1 - p;
        if (n > 4) n = 4;
        float k0[4][NPOS], v0[4][NPOS];
        // All four positions' key and value loads issue before the first dot:
        // that is the whole point of the batch, so do not interleave them.
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            if (b < n) {
                unsigned long long off =
                    (unsigned long long)(p + b) * (unsigned long long)hd +
                    (unsigned long long)lane;
#pragma unroll
                for (int i = 0; i < NPOS; ++i) {
                    k0[b][i] = kb[off + 32 * i];
                    v0[b][i] = vb[off + 32 * i];
                }
            }
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            if (b < n) {
                float d0 = 0.0f;
#pragma unroll
                for (int i = 0; i < NPOS; ++i) d0 += qv[i] * k0[b][i];
                float sc = sofuu_warp_sum_all(d0) * scale;
                float mn = m > sc ? m : sc;
                float corr = sofuu_exp(m - mn);
                float w = sofuu_exp(sc - mn);
                s = s * corr + w;
                m = mn;
#pragma unroll
                for (int i = 0; i < NPOS; ++i)
                    acc[i] = acc[i] * corr + w * v0[b][i];
            }
        }
    }

    if (lane == 0) {
        dst[0] = (p1 > p0) ? m : -3.402823466e38f;
        dst[1] = s;
    }
#pragma unroll
    for (int i = 0; i < NPOS; ++i) dst[2 + lane + 32 * i] = acc[i];
}

// Partial slot for (kv head, split, row) where row = t * n_heads + h. Decode
// has one row per head (n_tokens = 1), prefill one row per (token, head), so
// the same layout and the same combine kernel serve both.
static SOFUU_DEVICE SOFUU_INLINE float *sofuu_attn_part_slot(
    float *part, int kvh, int n_splits, int split, int rows, int row,
    int head_dim) {
    return part + ((((unsigned long long)kvh * n_splits + split) *
                        (unsigned long long)rows +
                    (unsigned long long)row) *
                   (unsigned long long)(2 + head_dim));
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(32)
void sofuu_attn_decode_split(const float *__restrict__ q,
                             const float *__restrict__ k_cache,
                             const float *__restrict__ v_cache,
                             float *__restrict__ part, int n_heads,
                             int n_kv_heads, int head_dim, int cap, float scale,
                             const int *seq_dev, int n_splits) {
    int seq_len = *seq_dev;
    int bx = (int)sofuu_ctaid_x();
    int lane = (int)(sofuu_tid_x() & 31u);
    if (n_kv_heads <= 0 || head_dim <= 0 || n_splits <= 0 ||
        (n_heads % n_kv_heads) != 0 || n_heads / n_kv_heads > 8 ||
        bx >= n_heads * n_splits) {
        return;
    }
    if (seq_len <= 0 || seq_len > cap) return;
    {
        int group = n_heads / n_kv_heads;
        int split = bx / n_heads;
        int h = bx - split * n_heads;
        int kvh = h / group;
        int chunk = (cap + n_splits - 1) / n_splits;
        int p0 = split * chunk;
        int p1 = p0 + chunk;
        if (p1 > seq_len) p1 = seq_len;
        unsigned long long cache_base =
            (unsigned long long)kvh * (unsigned long long)cap *
            (unsigned long long)head_dim;
        float *dst = sofuu_attn_part_slot(part, kvh, n_splits, split, n_heads, h,
                                          head_dim);
        if (head_dim == 32) {
            sofuu_attn_split_range<1>(
                q + (unsigned long long)h * 32u, k_cache + cache_base,
                v_cache + cache_base, dst, cap, scale, p0, p1, lane);
        } else if (head_dim == 64) {
            sofuu_attn_split_range<2>(
                q + (unsigned long long)h * 64u, k_cache + cache_base,
                v_cache + cache_base, dst, cap, scale, p0, p1, lane);
        } else if (head_dim == 128) {
            sofuu_attn_split_range<4>(
                q + (unsigned long long)h * 128u, k_cache + cache_base,
                v_cache + cache_base, dst, cap, scale, p0, p1, lane);
        }
    }
}

// Prefill: one warp per (token, head, position range). A prefill chunk used to
// run `sofuu_attn_prefill_f32`, which is the block-per-(token,head) kernel with
// two block barriers per position -- the same latency disease the decode split
// fixed, and at 2402 tokens it accounted for ~25 s of a 37.8 s prompt. Every
// token in the chunk has its own `seq_len` (start_pos + t + 1), so the ranges
// are clamped per block rather than against one shared length.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(32)
void sofuu_attn_prefill_split(const float *__restrict__ q,
                              const float *__restrict__ k_cache,
                              const float *__restrict__ v_cache,
                              float *__restrict__ part, int n_heads,
                              int n_kv_heads, int head_dim, int cap, float scale,
                              int start_pos, int n_tokens, int n_splits) {
    int bx = (int)sofuu_ctaid_x();
    int lane = (int)(sofuu_tid_x() & 31u);
    int rows = n_tokens * n_heads;
    if (n_kv_heads <= 0 || head_dim <= 0 || n_splits <= 0 || n_tokens <= 0 ||
        (n_heads % n_kv_heads) != 0 || n_heads / n_kv_heads > 8 ||
        bx >= rows * n_splits) {
        return;
    }
    int group = n_heads / n_kv_heads;
    int split = bx / rows;
    int row = bx - split * rows;
    int h = row % n_heads;
    int t = row / n_heads;
    int kvh = h / group;
    int seq_len = start_pos + t + 1;
    if (seq_len <= 0 || seq_len > cap) return;
    int chunk = (cap + n_splits - 1) / n_splits;
    int p0 = split * chunk;
    int p1 = p0 + chunk;
    if (p1 > seq_len) p1 = seq_len;
    unsigned long long cache_base = (unsigned long long)kvh *
                                    (unsigned long long)cap *
                                    (unsigned long long)head_dim;
    float *dst = sofuu_attn_part_slot(part, kvh, n_splits, split, rows, row,
                                      head_dim);
    const float *qv_row = q + (unsigned long long)row * (unsigned long long)head_dim;
    if (head_dim == 32) {
        sofuu_attn_split_range<1>(qv_row, k_cache + cache_base,
                                  v_cache + cache_base, dst, cap, scale, p0, p1,
                                  lane);
    } else if (head_dim == 64) {
        sofuu_attn_split_range<2>(qv_row, k_cache + cache_base,
                                  v_cache + cache_base, dst, cap, scale, p0, p1,
                                  lane);
    } else if (head_dim == 128) {
        sofuu_attn_split_range<4>(qv_row, k_cache + cache_base,
                                  v_cache + cache_base, dst, cap, scale, p0, p1,
                                  lane);
    }
}

// Merge the splits for one row and write it. A row is one query head in decode
// (n_tokens = 1) or one (token, head) pair in prefill, so `rows` is the grid
// size and `n_heads` is what maps a row back to its KV head. `do_quant` fuses
// the q8_1 encoding the o_proj/down GEMV needs, one 32-value block per warp, so
// the launch after this one has nothing to prepare.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_attn_combine(const float *__restrict__ part, float *__restrict__ out,
                        int rows, int n_heads, int n_kv_heads, int head_dim,
                        int n_splits, signed char *aq, float *ad, int *as,
                        int *as16, int do_quant) {
    int row = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    if (n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 || n_splits <= 0 ||
        n_heads <= 0 || row >= rows || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int h = row % n_heads;
    int group = n_heads / n_kv_heads;
    int kvh = h / group;
    unsigned long long sstride =
        (unsigned long long)rows * (unsigned long long)(2 + head_dim);
    const float *b0 = part + (((unsigned long long)kvh * n_splits *
                                   (unsigned long long)rows +
                               (unsigned long long)row) *
                              (unsigned long long)(2 + head_dim));
    if (tid < (unsigned int)head_dim) {
        float M = -3.402823466e38f;
        for (int s = 0; s < n_splits; ++s) {
            float ms = b0[s * sstride];
            if (ms > M) M = ms;
        }
        float num = 0.0f, den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float *r = b0 + s * sstride;
            // An empty split carries -inf; multiplying its zero weight by a
            // nan would poison the sum, so skip it by name.
            if (!(r[0] > -3.0e38f)) continue;
            float w = sofuu_exp(r[0] - M);
            den += r[1] * w;
            num += r[2 + tid] * w;
        }
        out[(unsigned long long)row * (unsigned long long)head_dim +
            (unsigned long long)tid] = den > 0.0f ? num / den : 0.0f;
    }
    if (do_quant != 0) {
        sofuu_bar_sync();
        int nb_head = head_dim >> 5;
        int warp = (int)(tid >> 5);
        int lane = (int)(tid & 31u);
        if (warp < nb_head) {
            sofuu_quantize_block32(out, aq, ad, as, as16, row * nb_head + warp,
                                   lane);
        }
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
// `pos_dev` points at the token's position in device memory, not a kernel
// argument: a CUDA graph captured once per model replays at a new position by
// updating that four-byte buffer, and a kernel parameter cannot change between
// launches of the same graph.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128) void sofuu_kv_write_f32(float *k_cache, float *v_cache,
                                                const float *k, const float *v,
                                                const int *pos_dev,
                                                int n_kv_heads, int head_dim,
                                                int cap) {
    int pos = *pos_dev;
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

// ── RoPE on q and k, then the KV write: one launch ────────────────────────
//
// A decode step rotated q (n_heads blocks), then rotated k (n_kv_heads
// blocks), then copied k and v into the cache (n_kv_heads blocks) -- three
// launches for ~3 KiB of data, each paying the dispatch and the drain the mixed
// bench prices at ~8 us. They are also strictly ordered: the cache must hold
// the *rotated* k. One kernel keeps that order with a block barrier, since a
// block owns one head's row, and merges the three grids:
//
//   block <  n_heads            -> rotate q head `block`
//   block >= n_heads            -> rotate k head `block - n_heads`, then write
//                                  that head's k and v rows to the cache
//
// The kv blocks pack two rows of head_dim floats, so the post-barrier copy is
// the same loop the standalone kv_write ran.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128)
void sofuu_rope_kv_f32(float *q, float *k, const float *v, float *k_cache,
                       float *v_cache, const float *cos_tab,
                       const float *sin_tab, const int *pos_dev, int n_heads,
                       int n_kv_heads, int head_dim, int n_rot, int interleaved,
                       int cap) {
    int blk = (int)sofuu_ctaid_x();
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    int half = n_rot >> 1;
    if (blk < n_heads) {
        float *base = q + (unsigned long long)blk * (unsigned long long)head_dim;
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
        return;
    }
    int kvh = blk - n_heads;
    int pos = *pos_dev;
    if (kvh >= n_kv_heads || pos < 0 || pos >= cap) return;
    float *kbase = k + (unsigned long long)kvh * (unsigned long long)head_dim;
    for (int i = (int)tid; i < half; i += (int)stride) {
        int i0 = interleaved ? (2 * i) : i;
        int i1 = interleaved ? (2 * i + 1) : (i + half);
        float c = cos_tab[i];
        float s = sin_tab[i];
        float x0 = kbase[i0];
        float x1 = kbase[i1];
        kbase[i0] = x0 * c - x1 * s;
        kbase[i1] = x0 * s + x1 * c;
    }
    // The copy below must see this block's own rotated values.
    sofuu_bar_sync();
    unsigned long long slot =
        ((unsigned long long)kvh * (unsigned long long)cap +
         (unsigned long long)pos) *
        (unsigned long long)head_dim;
    const float *vrow = v + (unsigned long long)kvh * (unsigned long long)head_dim;
    for (int d = (int)tid; d < head_dim; d += (int)stride) {
        k_cache[slot + (unsigned long long)d] = kbase[d];
        v_cache[slot + (unsigned long long)d] = vrow[d];
    }
}

// ── Batched prefill: one warp per (token, row) dot ─────────────────────────
//
// Prefill used to run the decode GEMV once per prompt token, which streams the
// whole weight file per token. These kernels instead give every warp ONE
// (token, row) dot and enumerate the token-major output across the grid:
//
//   idx   = blockIdx.x * 8 + warp
//   token = idx % n_tokens
//   row   = idx / n_tokens
//
// With n_tokens = 8 a block covers one row for all eight tokens, so the row's
// weight tile is fetched once into L1 and reused eight times. With n < 8 the
// block spans 8/n consecutive rows and every warp is still busy: the earlier
// mapping returned the idle warps and made each survivor walk `row_tile` rows
// serially, which made a chunk cost n_tokens decode steps and left the weight
// reuse with nothing to speed up.
//
// The row/lane mapping is the decode mapping (lane owns eight consecutive
// values), so the proven nibble/scale extraction is shared through the dot8
// helpers. There are no block barriers: each warp reduces its own row sum with
// shuffles and writes its token's output.

// ── Prefill q4_K, rows per warp on q8_1 activations ────────────────────────
//
// The (token, row) mapping above gives every warp its own row, which means
// every warp streams the token's whole activation row through L1: at 8192 rows
// a 6-token chunk moved ~67 MB of activations against 9.4 MB of weights, so the
// kernel measured ~14 GB/s of weight throughput and a chunk cost 37-56 ms where
// one pass over the weights is 3.3 ms.
//
// This variant fixes the ratio twice over:
//   * the activations are q8_1 (one byte per value, quantized ONCE per chunk by
//     the caller) instead of f32, and dp4a consumes them directly;
//   * one warp walks FOUR rows of the same matrix for the same token, so an
//     activation chunk load feeds four dots. Traffic per warp becomes 4 rows x
//     1152 B of weights against 2 KB of activations -- ~1.4x the weights, where
//     it was 7x.
// A block is still eight warps, each on its own (token, row-quad).
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemm_q4_k_b8_q8(const unsigned char *__restrict__ w,
                           const signed char *__restrict__ xq,
                           const float *__restrict__ xd,
                           const int *__restrict__ xs,
                           float *__restrict__ y, int rows, int cols,
                           int n_tokens) {
    if (n_tokens <= 0) return;
    const int RPW = 4;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int idx = (int)sofuu_ctaid_x() * 8 + warp;
    int token = idx % n_tokens;
    int rblk = idx / n_tokens;
    int row0 = rblk * RPW;
    if (row0 >= rows) return;
    int nb = cols / SOFUU_QK_K;
    const int chunk = lane & 7;
    const int bgrp = lane >> 3;
    const int k = chunk >> 1;
    const signed char *xq_t = xq + (unsigned long long)token *
                                       (unsigned long long)cols;
    // Per-token q8_1 side arrays: (cols/32) entries for d/s, laid out per token
    // the same way the row is.
    const int nb32 = cols / 32;
    const float *xd_t = xd + (unsigned long long)token * (unsigned long long)nb32;
    const int *xs_t = xs + (unsigned long long)token * (unsigned long long)nb32;

    const unsigned char *roww[4];
    int nrows = rows - row0;
    if (nrows > RPW) nrows = RPW;
#pragma unroll
    for (int r = 0; r < RPW; ++r)
        roww[r] = w + (unsigned long long)(row0 + r) *
                          (unsigned long long)nb * SOFUU_Q4_K_BYTES;

    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.0f;

    for (int b = bgrp; b < nb; b += 4) {
        sofuu_u128 xlo, xhi;
        sofuu_q4k_x_load(xq_t + (unsigned long long)b * 256, 0, chunk, &xlo,
                         &xhi);
        int ilo = b * 8 + 2 * k;
        float sx0 = xd_t[ilo], sx1 = xd_t[ilo + 1];
        int si0 = xs_t[ilo], si1 = xs_t[ilo + 1];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r >= nrows) break;
            const sofuu_block_q4_K *p = (const sofuu_block_q4_K *)(
                roww[r] + (unsigned long long)b * SOFUU_Q4_K_BYTES);
            sofuu_u128 wq;
            sofuu_q4k_w_load(p, chunk, &wq);
            acc[r] += sofuu_q4k_chunk32_compute(wq, xlo, xhi, p, chunk, sx0,
                                                si0, sx1, si1);
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows) break;
        float total = sofuu_warp_sum(acc[r]);
        if (lane == 0)
            y[(unsigned long long)token * (unsigned long long)rows +
              (unsigned long long)(row0 + r)] = total;
    }
}

// One block = eight warps = eight (token, row) dots.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemm_q4_k_b8(const unsigned char *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                        int rows, int cols, int n_tokens) {
    if (n_tokens <= 0) return;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int idx = (int)sofuu_ctaid_x() * 8 + warp;
    int token = idx % n_tokens;
    int row = idx / n_tokens;
    if (row >= rows) return;
    int sub = lane >> 2;
    int l0 = (lane & 3) << 3;
    int nb = cols / SOFUU_QK_K;
    const float *xrow = x + (unsigned long long)token * (unsigned long long)cols;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q4_K_BYTES;
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        float xr[8];
        sofuu_load8(xrow + bi * 256 + lane * 8, xr);
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)bi * 144);
        acc += sofuu_q4k_dot8(b, sub, l0, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0)
        y[(unsigned long long)token * (unsigned long long)rows +
          (unsigned long long)row] = total;
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemm_q6_k_b8(const unsigned char *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                        int rows, int cols, int n_tokens) {
    if (n_tokens <= 0) return;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int idx = (int)sofuu_ctaid_x() * 8 + warp;
    int token = idx % n_tokens;
    int row = idx / n_tokens;
    if (row >= rows) return;
    int sub = lane >> 2;
    int l0 = (lane & 3) << 3;
    int nb = cols / SOFUU_QK_K;
    const float *xrow = x + (unsigned long long)token * (unsigned long long)cols;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q6_K_BYTES;
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        float xr[8];
        sofuu_load8(xrow + bi * 256 + lane * 8, xr);
        const sofuu_block_q6_K *b =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)bi * SOFUU_Q6_K_BYTES);
        acc += sofuu_q6k_dot8(b, sub, l0, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0)
        y[(unsigned long long)token * (unsigned long long)rows +
          (unsigned long long)row] = total;
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemm_q8_0_b8(const unsigned char *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                        int rows, int cols, int n_tokens) {
    if (n_tokens <= 0) return;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int idx = (int)sofuu_ctaid_x() * 8 + warp;
    int token = idx % n_tokens;
    int row = idx / n_tokens;
    if (row >= rows) return;
    int block_in_warp = lane >> 2;
    int lane_in_block = lane & 3;
    int nb = cols / SOFUU_QK_Q8_0;
    const float *xrow = x + (unsigned long long)token * (unsigned long long)cols;
    const unsigned char *roww = w + (unsigned long long)row *
                                     (unsigned long long)nb * SOFUU_Q8_0_BYTES;
    float acc = 0.0f;
    for (int cg = 0; cg < cols; cg += 256) {
        float xr[8];
        sofuu_load8(xrow + cg + lane * 8, xr);
        int bi = (cg >> 5) + block_in_warp;
        const sofuu_block_q8_0 *b =
            (const sofuu_block_q8_0 *)(roww + (unsigned long long)bi * 34);
        acc += sofuu_q8_0_dot8(b, lane_in_block, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0)
        y[(unsigned long long)token * (unsigned long long)rows +
          (unsigned long long)row] = total;
}

extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 4)
void sofuu_gemm_f32_b8(const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y, int rows,
                       int cols, int n_tokens) {
    if (n_tokens <= 0) return;
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int idx = (int)sofuu_ctaid_x() * 8 + warp;
    int token = idx % n_tokens;
    int row = idx / n_tokens;
    if (row >= rows) return;
    const float *xrow = x + (unsigned long long)token * (unsigned long long)cols;
    const float *roww = w + (unsigned long long)row * (unsigned long long)cols;
    float acc = 0.0f;
    for (int cg = 0; cg < cols; cg += 256) {
        float xr[8];
        sofuu_load8(xrow + cg + lane * 8, xr);
        acc += sofuu_f32_dot8(roww + cg + lane * 8, xr);
    }
    float total = sofuu_warp_sum(acc);
    if (lane == 0)
        y[(unsigned long long)token * (unsigned long long)rows +
          (unsigned long long)row] = total;
}

// One block per token; same reduction as the single-row kernel.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_rmsnorm_batch(const float *x, const float *w, float *y, int n,
                         float eps, int rows) {
    int row = (int)sofuu_ctaid_x();
    if (row >= rows) return;
    const float *xr = x + (unsigned long long)row * (unsigned long long)n;
    float *yr = y + (unsigned long long)row * (unsigned long long)n;
    unsigned int tid = sofuu_tid_x();
    float ss = 0.0f;
    for (int i = (int)tid; i < n; i += (int)sofuu_ntid_x()) ss += xr[i] * xr[i];
    SOFUU_SHARED float red[8];
    SOFUU_SHARED float scale;
    float total = sofuu_block_reduce(ss, red);
    if (tid == 0u) scale = sofuu_rsqrt(total / (float)n + eps);
    sofuu_bar_sync();
    float s = scale;
    for (int i = (int)tid; i < n; i += (int)sofuu_ntid_x()) {
        yr[i] = xr[i] * s * w[i];
    }
}

// One block per (token, head); the table holds one [head_dim] row per token.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128)
void sofuu_rope_batch(float *x, const float *table, int n_heads, int head_dim,
                      int n_rot, int interleaved, int n_tokens) {
    int idx = (int)sofuu_ctaid_x();
    int t = idx / n_heads;
    int head = idx - t * n_heads;
    if (t >= n_tokens) return;
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    int half = n_rot >> 1;
    float *base = x + ((unsigned long long)t * (unsigned long long)n_heads +
                       (unsigned long long)head) *
                          (unsigned long long)head_dim;
    const float *cos_tab = table + (unsigned long long)t *
                                       (unsigned long long)head_dim;
    const float *sin_tab = cos_tab + (head_dim >> 1);
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

// One block per (token, KV head).
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(128)
void sofuu_kv_write_batch(float *k_cache, float *v_cache, const float *k,
                          const float *v, int start_pos, int n_tokens,
                          int n_kv_heads, int head_dim, int cap) {
    int idx = (int)sofuu_ctaid_x();
    int t = idx / n_kv_heads;
    int head = idx - t * n_kv_heads;
    int pos = start_pos + t;
    if (t >= n_tokens || head >= n_kv_heads || pos < 0 || pos >= cap) return;
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();
    unsigned long long slot =
        ((unsigned long long)head * (unsigned long long)cap +
         (unsigned long long)pos) *
        (unsigned long long)head_dim;
    const float *krow =
        k + ((unsigned long long)t * (unsigned long long)n_kv_heads +
             (unsigned long long)head) *
                (unsigned long long)head_dim;
    const float *vrow =
        v + ((unsigned long long)t * (unsigned long long)n_kv_heads +
             (unsigned long long)head) *
                (unsigned long long)head_dim;
    for (int d = (int)tid; d < head_dim; d += (int)stride) {
        k_cache[slot + (unsigned long long)d] = krow[d];
        v_cache[slot + (unsigned long long)d] = vrow[d];
    }
}

// Causal attention for a chunk: one block per (token, query head), attending
// over positions [0, start_pos+t]. The KV writes for the whole chunk happen
// before this launch, so tokens inside the chunk see each other.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_attn_prefill_f32(const float *q, const float *k_cache,
                            const float *v_cache, float *out, int n_heads,
                            int n_kv_heads, int head_dim, int start_pos,
                            int n_tokens, int cap, float scale) {
    int idx = (int)sofuu_ctaid_x();
    int t = idx / n_heads;
    int h = idx - t * n_heads;
    if (t >= n_tokens || h >= n_heads || n_kv_heads <= 0 || head_dim <= 0 ||
        head_dim > 256 || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int seq_len = start_pos + t + 1;
    if (seq_len <= 0 || seq_len > cap) return;
    int group = n_heads / n_kv_heads;
    int kvh = h / group;
    const float *qv = q + ((unsigned long long)t * (unsigned long long)n_heads +
                           (unsigned long long)h) *
                              (unsigned long long)head_dim;
    float *ov = out + ((unsigned long long)t * (unsigned long long)n_heads +
                       (unsigned long long)h) *
                          (unsigned long long)head_dim;
    const float *kbase =
        k_cache + (unsigned long long)kvh * (unsigned long long)cap *
                      (unsigned long long)head_dim;
    const float *vbase =
        v_cache + (unsigned long long)kvh * (unsigned long long)cap *
                      (unsigned long long)head_dim;
    unsigned int tid = sofuu_tid_x();
    unsigned int stride = sofuu_ntid_x();

    SOFUU_SHARED float red[8];
    SOFUU_SHARED float acc[256];
    SOFUU_SHARED float s_max;
    SOFUU_SHARED float s_w;
    SOFUU_SHARED float s_sum;

    if (tid == 0u) s_max = -3.402823466e38f;
    for (int d = (int)tid; d < head_dim; d += (int)stride) acc[d] = 0.0f;
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
            float score = sum * scale;
            if (score > s_max) s_max = score;
        }
        sofuu_bar_sync();
    }

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
        ov[d] = acc[d] * inv;
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

extern "C" void sofuu_host_quantize_q8_1(const float *x, signed char *q,
                                         float *d, int *s, int *s16, int n) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = sofuu_absf(x[b * 32 + i]);
            if (a > amax) amax = a;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0, lo = 0;
        for (int i = 0; i < 32; ++i) {
            int qi = sofuu_round_int(x[b * 32 + i] * inv);
            qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
            q[b * 32 + i] = (signed char)qi;
            sum += qi;
            if (i < 16) lo += qi;
        }
        d[b] = scale;
        s[b] = sum;
        s16[b * 2] = lo;
        s16[b * 2 + 1] = sum - lo;
    }
}

extern "C" float sofuu_host_q4_k_dot_q8(const void *w, const signed char *xq,
                                        const float *xd, const int *xs,
                                        int cols);

// Prefill rows-per-warp mirror: the mapping and the output indexing are the new
// part, so build it from the already-proven per-row dot rather than restating
// the nibble math.
extern "C" void sofuu_host_gemm_q4_k_b8_q8(const unsigned char *w,
                                           const signed char *xq,
                                           const float *xd, const int *xs,
                                           float *y, int rows, int cols,
                                           int n_tokens) {
    int nb32 = cols / 32;
    int rb = (cols / 256) * 144;
    for (int t = 0; t < n_tokens; ++t) {
        for (int r = 0; r < rows; ++r) {
            y[(unsigned long long)t * (unsigned long long)rows +
              (unsigned long long)r] =
                sofuu_host_q4_k_dot_q8(
                    w + (unsigned long long)r * (unsigned long long)rb,
                    xq + (unsigned long long)t * (unsigned long long)cols,
                    xd + (unsigned long long)t * (unsigned long long)nb32,
                    xs + (unsigned long long)t * (unsigned long long)nb32,
                    cols);
        }
    }
}

extern "C" float sofuu_host_q4_k_dot_q8(const void *w, const signed char *xq,
                                        const float *xd, const int *xs,
                                        int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 256;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(p + (unsigned long long)bi * 144);
        for (int chunk = 0; chunk < 8; ++chunk) {
            total += sofuu_q4k_chunk32_q8(b, xq + bi * 256, chunk, bi, xd, xs);
        }
    }
    return total;
}

extern "C" float sofuu_host_q6_k_dot_q8(const void *w, const signed char *xq,
                                        const float *xd, const int *xs16,
                                        int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 256;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const unsigned char *blk = p + (unsigned long long)bi * SOFUU_Q6_K_BYTES;
        for (int chunk = 0; chunk < 4; ++chunk) {
            const int n = chunk >> 1, j = chunk & 1;
            const int sh_a = 2 * j, sh_b = sh_a + 4;
            const int slot = n * 8 + 2 * j;
            sofuu_u128 ql0 = sofuu_load_u128(blk + 64 * n + 32 * j);
            sofuu_u128 ql1 = sofuu_load_u128(blk + 64 * n + 32 * j + 16);
            sofuu_u128 qh0 = sofuu_load_u128(blk + 128 + 32 * n);
            sofuu_u128 qh1 = sofuu_load_u128(blk + 128 + 32 * n + 16);
            const int ia = bi * 8 + n * 4 + j, ib = ia + 2;
            sofuu_u128 xa0 = sofuu_load_u128(xq + (unsigned long long)ia * 32);
            sofuu_u128 xa1 = sofuu_load_u128(xq + (unsigned long long)ia * 32 + 16);
            sofuu_u128 xb0 = sofuu_load_u128(xq + (unsigned long long)ib * 32);
            sofuu_u128 xb1 = sofuu_load_u128(xq + (unsigned long long)ib * 32 + 16);
            sofuu_u128 scv = sofuu_load_u128(blk + 192);
            float d = sofuu_f16_to_f32((unsigned short)(
                sofuu_load_u128(blk + 208).w[0] & 0xffffu));
            float ca_lo = xd[ia] * d * (float)sofuu_i8_at(scv, slot);
            float ca_hi = xd[ia] * d * (float)sofuu_i8_at(scv, slot + 1);
            float cb_lo = xd[ib] * d * (float)sofuu_i8_at(scv, slot + 4);
            float cb_hi = xd[ib] * d * (float)sofuu_i8_at(scv, slot + 5);
            total += ca_lo * (float)(sofuu_q6k_dot16_q8(ql0, qh0, xa0, 0, sh_a) -
                                     32 * xs16[2 * ia]);
            total += ca_hi * (float)(sofuu_q6k_dot16_q8(ql1, qh1, xa1, 0, sh_a) -
                                     32 * xs16[2 * ia + 1]);
            total += cb_lo * (float)(sofuu_q6k_dot16_q8(ql0, qh0, xb0, 1, sh_b) -
                                     32 * xs16[2 * ib]);
            total += cb_hi * (float)(sofuu_q6k_dot16_q8(ql1, qh1, xb1, 1, sh_b) -
                                     32 * xs16[2 * ib + 1]);
        }
    }
    return total;
}

extern "C" float sofuu_host_q6_k_dot(const void *w, const float *x, int cols) {
    const unsigned char *p = (const unsigned char *)w;
    int nb = cols / 256;
    float total = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const sofuu_block_q6_K *b =
            (const sofuu_block_q6_K *)(p + (unsigned long long)bi * SOFUU_Q6_K_BYTES);
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

extern "C" void sofuu_host_rmsnorm_q8(const float *x, const float *w, float *y,
                                      signed char *q, float *d, int *s,
                                      int *s16, int n, float eps) {
    sofuu_host_rmsnorm(x, w, y, n, eps);
    sofuu_host_quantize_q8_1(y, q, d, s, s16, n);
}

extern "C" void sofuu_host_rmsnorm_add_q8(const float *a, const float *b,
                                         const float *w, float *xres, float *y,
                                         signed char *q, float *d, int *s,
                                         int *s16, int n, float eps) {
    for (int i = 0; i < n; ++i) xres[i] = a[i] + b[i];
    sofuu_host_rmsnorm(xres, w, y, n, eps);
    sofuu_host_quantize_q8_1(y, q, d, s, s16, n);
}

extern "C" void sofuu_host_swiglu(const float *g, const float *u, float *y,
                                  int n) {
    for (int i = 0; i < n; ++i) y[i] = sofuu_silu(g[i]) * u[i];
}

extern "C" void sofuu_host_swiglu_q8(const float *g, const float *u, float *y,
                                     signed char *q, float *d, int *s, int *s16,
                                     int n) {
    sofuu_host_swiglu(g, u, y, n);
    sofuu_host_quantize_q8_1(y, q, d, s, s16, n);
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
                                    const float *k, const float *v,
                                    const int *pos_dev, int n_kv_heads,
                                    int head_dim, int cap) {
    int pos = *pos_dev;
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
                                       const int *seq_dev, int cap, float scale) {
    int seq_len = *seq_dev;
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

// Split-K decode attention, host reference. Partials are written in the same
// layout the device kernel uses:
//   part[((kvh * n_splits + split) * n_heads + h) * (2 + head_dim)]
//   = { m, sum, acc[head_dim] }
// Computed in double so the host check isolates the split/merge identity from
// the device's float rounding.
extern "C" void sofuu_host_attn_split(const float *q, const float *k_cache,
                                      const float *v_cache, float *part,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int cap, float scale,
                                      int seq_len, int n_splits) {
    if (n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 || n_splits <= 0 ||
        seq_len < 0 || seq_len > cap || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int group = n_heads / n_kv_heads;
    int chunk = (cap + n_splits - 1) / n_splits;
    for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
        const float *kb = k_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        const float *vb = v_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        for (int s = 0; s < n_splits; ++s) {
            int p0 = s * chunk;
            int p1 = p0 + chunk;
            if (p1 > seq_len) p1 = seq_len;
            for (int h = kvh * group; h < (kvh + 1) * group; ++h) {
                const float *qv =
                    q + (unsigned long long)h * (unsigned long long)head_dim;
                float *out_part =
                    part + ((((unsigned long long)kvh * n_splits + s) *
                                 (unsigned long long)n_heads +
                             (unsigned long long)h) *
                            (unsigned long long)(2 + head_dim));
                double acc[256];
                for (int d = 0; d < head_dim; ++d) acc[d] = 0.0;
                double m = -1e300, sum = 0.0;
                for (int p = p0; p < p1; ++p) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += (double)qv[d] *
                               (double)kb[(unsigned long long)p *
                                              (unsigned long long)head_dim +
                                          (unsigned long long)d];
                    double score = dot * (double)scale;
                    double mn = m > score ? m : score;
                    double corr = exp(m - mn);
                    double w = exp(score - mn);
                    sum = sum * corr + w;
                    m = mn;
                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = acc[d] * corr +
                                 w * (double)vb[(unsigned long long)p *
                                                    (unsigned long long)head_dim +
                                                (unsigned long long)d];
                }
                out_part[0] = (p1 > p0) ? (float)m : -3.402823466e38f;
                out_part[1] = (float)sum;
                for (int d = 0; d < head_dim; ++d)
                    out_part[2 + d] = (float)acc[d];
            }
        }
    }
}

// Prefill split-K, host reference: rows are (token, head) pairs and each row's
// range ends at that token's own position.
extern "C" void sofuu_host_attn_prefill_split(
    const float *q, const float *k_cache, const float *v_cache, float *part,
    int n_heads, int n_kv_heads, int head_dim, int cap, float scale,
    int start_pos, int n_tokens, int n_splits) {
    if (n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 || n_splits <= 0 ||
        n_tokens <= 0 || (n_heads % n_kv_heads) != 0) {
        return;
    }
    int group = n_heads / n_kv_heads;
    int rows = n_tokens * n_heads;
    int chunk = (cap + n_splits - 1) / n_splits;
    for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
        const float *kb = k_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        const float *vb = v_cache + (unsigned long long)kvh *
                                        (unsigned long long)cap *
                                        (unsigned long long)head_dim;
        for (int s = 0; s < n_splits; ++s) {
            for (int row = 0; row < rows; ++row) {
                int h = row % n_heads;
                if (h / group != kvh) continue;
                int t = row / n_heads;
                int seq_len = start_pos + t + 1;
                if (seq_len <= 0 || seq_len > cap) continue;
                int p0 = s * chunk;
                int p1 = p0 + chunk;
                if (p1 > seq_len) p1 = seq_len;
                const float *qv =
                    q + (unsigned long long)row * (unsigned long long)head_dim;
                float *out_part =
                    part + ((((unsigned long long)kvh * n_splits + s) *
                                 (unsigned long long)rows +
                             (unsigned long long)row) *
                            (unsigned long long)(2 + head_dim));
                double acc[256];
                for (int d = 0; d < head_dim; ++d) acc[d] = 0.0;
                double m = -1e300, sum = 0.0;
                for (int p = p0; p < p1; ++p) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += (double)qv[d] *
                               (double)kb[(unsigned long long)p *
                                              (unsigned long long)head_dim +
                                          (unsigned long long)d];
                    double score = dot * (double)scale;
                    double mn = m > score ? m : score;
                    double corr = exp(m - mn);
                    double w = exp(score - mn);
                    sum = sum * corr + w;
                    m = mn;
                    for (int d = 0; d < head_dim; ++d)
                        acc[d] = acc[d] * corr +
                                 w * (double)vb[(unsigned long long)p *
                                                    (unsigned long long)head_dim +
                                                (unsigned long long)d];
                }
                out_part[0] = (p1 > p0) ? (float)m : -3.402823466e38f;
                out_part[1] = (float)sum;
                for (int d = 0; d < head_dim; ++d)
                    out_part[2 + d] = (float)acc[d];
            }
        }
    }
}

extern "C" void sofuu_host_attn_combine(const float *part, float *out,
                                        int rows, int n_heads, int n_kv_heads,
                                        int head_dim, int n_splits) {
    if (n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 || n_splits <= 0 ||
        (n_heads % n_kv_heads) != 0) {
        return;
    }
    int group = n_heads / n_kv_heads;
    for (int row = 0; row < rows; ++row) {
        int h = row % n_heads;
        int kvh = h / group;
        unsigned long long sstride =
            (unsigned long long)rows * (unsigned long long)(2 + head_dim);
        const float *b0 =
            part + (((unsigned long long)kvh * n_splits *
                         (unsigned long long)rows +
                     (unsigned long long)row) *
                    (unsigned long long)(2 + head_dim));
        for (int d = 0; d < head_dim; ++d) {
            double M = -1e300;
            for (int s = 0; s < n_splits; ++s) {
                double ms = (double)b0[s * sstride];
                if (ms > M) M = ms;
            }
            double num = 0.0, den = 0.0;
            for (int s = 0; s < n_splits; ++s) {
                const float *r = b0 + s * sstride;
                if (!(r[0] > -3.0e38f)) continue;
                double w = exp((double)r[0] - M);
                den += (double)r[1] * w;
                num += (double)r[2 + d] * w;
            }
            out[(unsigned long long)row * (unsigned long long)head_dim +
                (unsigned long long)d] =
                (float)(den > 0.0 ? num / den : 0.0);
        }
    }
}

extern "C" void sofuu_host_add(const float *a, const float *b, float *y,
                               int n) {
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
}
#endif // SOFUU_CUDA_HOST
