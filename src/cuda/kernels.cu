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

#define SOFUU_QK_Q8_0 32
#define SOFUU_QK_K 256
#define SOFUU_Q4_K_BYTES 144
#define SOFUU_Q6_K_BYTES 210
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

// q4_K sub-block dot against q8_1 activations. `xrow` is the token's q8 row
// (int8), `sx`/`isum` are that q8 block's scale and integer sum. The min term
// is contributed by one lane per sub-block (l0 == 0) from the precomputed
// 32-value sum, so the four lanes of a sub-block do not each need it.
static SOFUU_DEVICE SOFUU_INLINE float sofuu_q4k_dot8_q8(
    const sofuu_block_q4_K *blk, const signed char *xrow, int sub, int l0,
    float sx, int isum) {
    unsigned int sc, mn;
    sofuu_q4k_scale_min(sub, blk->scales, &sc, &mn);
    float s = sofuu_f16_to_f32(blk->d) * (float)sc;
    float m = sofuu_f16_to_f32(blk->dmin) * (float)mn;
    unsigned long long qv = *reinterpret_cast<const unsigned long long *>(
        blk->qs + (sub >> 1) * 32 + l0);
    unsigned long long xv = *reinterpret_cast<const unsigned long long *>(
        xrow + sub * 32 + l0);
    const unsigned int mask = 0x0f0f0f0fu;
    unsigned int q0 = (unsigned int)qv, q1 = (unsigned int)(qv >> 32);
    unsigned int a0 = (sub & 1) ? ((q0 >> 4) & mask) : (q0 & mask);
    unsigned int a1 = (sub & 1) ? ((q1 >> 4) & mask) : (q1 & mask);
    unsigned int x0 = (unsigned int)xv, x1 = (unsigned int)(xv >> 32);
    int dot = sofuu_dp4a(a0, x0, sofuu_dp4a(a1, x1, 0));
    float part = sx * s * (float)dot;
    if (l0 == 0) part -= sx * m * (float)isum;
    return part;
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
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS(256)
void sofuu_quantize_q8_1(const float *__restrict__ x, signed char *__restrict__ q,
                         float *__restrict__ d, int *__restrict__ s, int n) {
    unsigned int tid = sofuu_tid_x();
    int warp = (int)(tid >> 5);
    int lane = (int)(tid & 31u);
    int block = (int)sofuu_ctaid_x() * 8 + warp;
    int base = block * 32;
    if (base >= n) return;
    float v = x[base + lane];
    float amax = sofuu_warp_max_f(sofuu_absf(v));
    float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int qi = sofuu_round_int(v * inv);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    q[base + lane] = (signed char)qi;
    int sum = sofuu_warp_sum_i(qi);
    if (lane == 0) {
        d[block] = scale;
        s[block] = sum;
    }
}

// q4_K x q8_1: the decode row/lane mapping with the per-value convert + FMA
// chain replaced by two dp4a per eight values. Two super-blocks stay in flight.
extern "C" SOFUU_GLOBAL SOFUU_LAUNCH_BOUNDS_MIN(256, 6)
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
    int sub = lane >> 2;
    int l0 = (lane & 3) << 3;
    float acc = 0.0f;
    int bi = 0;
    for (; bi + 2 <= nb; bi += 2) {
        const sofuu_block_q4_K *b0 =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)(bi + 0) * 144);
        const sofuu_block_q4_K *b1 =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)(bi + 1) * 144);
        int i0 = (bi + 0) * 8 + sub;
        int i1 = (bi + 1) * 8 + sub;
        acc += sofuu_q4k_dot8_q8(b0, xq + (bi + 0) * 256, sub, l0, xd[i0], xs[i0]);
        acc += sofuu_q4k_dot8_q8(b1, xq + (bi + 1) * 256, sub, l0, xd[i1], xs[i1]);
    }
    for (; bi < nb; ++bi) {
        const sofuu_block_q4_K *b =
            (const sofuu_block_q4_K *)(roww + (unsigned long long)bi * 144);
        int i0 = bi * 8 + sub;
        acc += sofuu_q4k_dot8_q8(b, xq + bi * 256, sub, l0, xd[i0], xs[i0]);
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
            (const sofuu_block_q6_K *)(roww + (unsigned long long)(bi + 0) * 210);
        const sofuu_block_q6_K *b1 =
            (const sofuu_block_q6_K *)(roww + (unsigned long long)(bi + 1) * 210);
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
            (const sofuu_block_q6_K *)(roww + (unsigned long long)bi * 210);
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
            (const sofuu_block_q6_K *)(roww + (unsigned long long)bi * 210);
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
                                         float *d, int *s, int n) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = sofuu_absf(x[b * 32 + i]);
            if (a > amax) amax = a;
        }
        float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            int qi = sofuu_round_int(x[b * 32 + i] * inv);
            qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
            q[b * 32 + i] = (signed char)qi;
            sum += qi;
        }
        d[b] = scale;
        s[b] = sum;
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
        for (int lane = 0; lane < 32; ++lane) {
            int sub = lane >> 2;
            int l0 = (lane & 3) << 3;
            int i0 = bi * 8 + sub;
            total += sofuu_q4k_dot8_q8(b, xq + bi * 256, sub, l0, xd[i0], xs[i0]);
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
