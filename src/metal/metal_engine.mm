#define GV2_ROWS_HOST 8
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_engine.h"
#include <cstring>
#include <chrono>
#include <cstdlib>
// AUDIT-2026-10-31 P1-8: inflate the embedded (gzipped) MSL source — see
// sofuu_metal_kernel_source below.
#include <zlib.h>

#include <iostream>
#include <unistd.h>

// ── GQA attention shape table ───────────────────────────────────────────────
// One entry per compile-time (R, NR, RG) instantiation of
// llama_attn_pre_row_gqa. R is the q-heads-per-kv-head ratio the kernel
// hard-codes, rg the rows per threadgroup and threads = (rg/nr)*32 after the
// kernel rows-per-warp. Entries 0..4 are the shipped 2:1 shapes and keep the
// SOFUU_ATTN_GQA=1..5 meaning they always had; entries 5.. are the other
// ratios the model fleet actually ships, so a 4:1 / 3:1 / 8:1 model can use
// the paired K/V stream instead of the per-head kernel.
//
// sofuu_attn_gqa_pick / sofuu_attn_gqa_pick_ex are the single selection rule
// and they are pure, so the shape table is testable without a GPU (see the
// Rust tests in infer::metal). The rule is availability-driven: an entry is
// selectable only when its own pipeline loaded AND reports
// maxTotalThreadsPerThreadgroup >= its threadgroup, and the preference order
// inside one ratio is the measured cost order (see k_gqa_pref_*).
#define SOFUU_GQA_N 12
struct SofuuGqaShape {
    const char *name;
    uint32_t ratio;    // n_heads / n_kv_heads the kernel R must equal
    uint32_t rg;       // rows per threadgroup
    uint32_t threads;  // threads per threadgroup (checked against the pipeline)
    uint32_t opt_in;   // 1 => only under SOFUU_ATTN_GQA_ALL=1 (unmeasured shape)
};
static const SofuuGqaShape k_sofuu_gqa[SOFUU_GQA_N] = {
    { "attn_gqa_r2_nr1_rg32",    2u, 32u, 1024u, 0u },  // 0  mode 1: pairing only,
                                                         //    2x the K/V traffic of nr2
    { "attn_gqa_r2_nr2_rg32",    2u, 32u,  512u, 0u },  // 1  mode 2 (default, measured
                                                        //    cheapest available form)
    { "attn_gqa_r2_nr2_rg16",    2u, 16u,  256u, 0u },  // 2  mode 3: same warps as 1,
                                                        //    more threadgroups
    { "attn_gqa_r2_nr4_rg32",    2u, 32u,  256u, 0u },  // 3  mode 4: 4 rows per warp
    { "attn_gqa_r2_nr1_rg32_pf", 2u, 32u, 1024u, 0u },  // 4  mode 5: prefetch twin of 0;
                                                        //    needs 1024 threads, and its
                                                        //    register use leaves only 896
                                                        //    on this GPU => nil here
    { "attn_gqa_r4_nr1_rg16",    4u, 16u,  512u, 0u },  // 5  4:1, 512
    { "attn_gqa_r4_nr2_rg16",    4u, 16u,  256u, 0u },  // 6  4:1, 256: fits a 384-thread ceiling
    { "attn_gqa_r4_nr4_rg32",    4u, 32u,  256u, 0u },  // 7  4:1, 4 rows per warp
    { "attn_gqa_r3_nr1_rg16",    3u, 16u,  512u, 0u },  // 8  3:1
    { "attn_gqa_r3_nr2_rg16",    3u, 16u,  256u, 0u },  // 9  3:1, 256
    { "attn_gqa_r8_nr1_rg8",     8u,  8u,  256u, 0u },  // 10 8:1
    { "attn_gqa_r1_nr2_rg32",    1u, 32u,  512u, 1u },  // 11 1:1 - no K/V sharing and
                                                        //    no 1:1 model in the fleet
                                                        //    to measure it on => opt-in
};
// ── the selection rule ────────────────────────────────────────────────
// avail bit i is set by sofuu_metal_init for exactly the shapes whose pipeline
// LOADED and whose maxTotalThreadsPerThreadgroup >= the threadgroup the shape
// would dispatch (the guard loop there). That is the only per-shape ceiling
// Metal gives us - it is a property of the compiled pipeline (register use),
// not of the device: on this GPU attn_gqa_r2_nr1_rg32 allows 1024 while its
// prefetch twin attn_gqa_r2_nr1_rg32_pf allows 896. So a 1024-thread form can
// only ever be in avail on a pipeline that reports >= 1024, and the rule below
// can never dispatch an oversized threadgroup - Metal SILENTLY DROPS one in
// release and the output reads back as stale attention.
//
// Within one ratio the preference list is ordered by MEASURED per-dispatch
// attention cost (cheapest first), not by table order: the shape that shares
// the K/V fetch across the most rows wins, and the pairing-only 1024-thread
// form (2x the K/V traffic of the nr2 form) is a fallback, not a default. A
// form whose pipeline is nil is SKIPPED, never turned into a per-head fallback
// while a slower sibling of the same ratio is live.
//
// The two "not selected" outcomes are different and must not be conflated
// (that conflation is what hid a nil pipeline behind "per-head"):
//   -1  the table has no form for this ratio (5:1, 6:1, ...): per-head is the
//       design, not a failure.
//   -2  the table HAS forms for this ratio and not one is usable: a real
//       configuration failure, reported loudly by the dispatcher.
// Cost order per ratio. The numbers are the WHOLE-PREFILL attention cost
// measured in situ with SOFUU_ATTN_DUP=1 minus =0 (adjacent runs, interleaved
// ABBA, 5 rounds, medians, M2 Pro, 1454-token prompt = 13 chunks of 128 rows,
// load 1.3-3.6; divide by chunks x layers for one dispatch):
//   2:1 (Qwen3-0.6B, 28 layers)  per-head 614 ms
//       entry 1 (nr2/rg32, 512) 431 | entry 2 (nr2/rg16, 256) 429
//       entry 0 (nr1/rg32, 1024) 487 | entry 3 (nr4/rg32, 256) 567
//   Entries 1 and 2 are a TIE (0.6%, inside the run-to-run spread) - both pack
//   2 rows per warp, which is what halves the K/V fetches; the shipped default
//   keeps entry 1 because it is the shape every earlier receipt was taken on.
//   Entry 4 (_pf) is nil on this GPU (needs 1024 threads, pipeline allows 896)
//   and has never been measurable, so it stays LAST: an unmeasured shape must
//   never outrank a measured one.
// The other ratios, same technique, 5 interleaved rounds, medians
// (load 1.3-3.4; the per-dispatch cost quoted is the whole-prefill attention
// cost / (chunks x layers), i.e. what one dispatch costs on a 128-row chunk):
//   3:1 (SmolLM2-360M, 32 layers x 13 chunks)  per-head 887 ms
//       entry 9 (nr2/rg16, 256) 650 | entry 8 (nr1/rg16, 512) 709
//       => nr2 wins here: 2 rows per warp halves the K/V fetches and R=3
//          still fits the register budget, so the pairing-only form is the
//          SLOWER one (it is only kept as a fallback for a narrow GPU).
//   4:1 (Llama-3.2-1B, 16 layers x 13 chunks)  per-head 665 ms
//       entry 5 (nr1/rg16, 512) 453 | entry 6 (nr2/rg16, 256) 476
//       entry 7 (nr4/rg32, 256) 1224 (register-blocked, 2.7x slower)
//       => 4:1 keeps nr1 first: R=4 heads per warp is already wide.
//   8:1 (MiniCPM5-2B) has one form; per-head vs it is in the ABBA receipt.
static const int k_gqa_pref_r1[] = { 11 };
static const int k_gqa_pref_r2[] = { 1, 2, 0, 3, 4 };
static const int k_gqa_pref_r3[] = { 9, 8 };
static const int k_gqa_pref_r4[] = { 5, 6, 7 };
static const int k_gqa_pref_r8[] = { 10 };
struct SofuuGqaPref { uint32_t ratio; const int *order; int n; };
static const SofuuGqaPref k_gqa_pref[] = {
    { 1u, k_gqa_pref_r1, 1 },
    { 2u, k_gqa_pref_r2, 5 },
    { 3u, k_gqa_pref_r3, 2 },
    { 4u, k_gqa_pref_r4, 3 },
    { 8u, k_gqa_pref_r8, 1 },
};
static const int k_gqa_pref_n = (int)(sizeof(k_gqa_pref) / sizeof(k_gqa_pref[0]));

static int sofuu_attn_gqa_pick_ex(uint32_t ratio, uint32_t avail, int allow_opt_in) {
    if (ratio < 1u || ratio > 64u) return -1;
    bool any_form = false;
    for (int i = 0; i < SOFUU_GQA_N; ++i) {
        if (k_sofuu_gqa[i].ratio == ratio) { any_form = true; break; }
    }
    const SofuuGqaPref *pref = nullptr;
    for (int i = 0; i < k_gqa_pref_n; ++i) {
        if (k_gqa_pref[i].ratio == ratio) { pref = &k_gqa_pref[i]; break; }
    }
    if (pref != nullptr) {
        for (int i = 0; i < pref->n; ++i) {
            const int idx = pref->order[i];
            if (idx < 0 || idx >= SOFUU_GQA_N) continue;
            const SofuuGqaShape &s = k_sofuu_gqa[idx];
            // A list entry whose compile-time R does not equal the ratio must
            // never be dispatched: R fixes how many q heads read one kv head,
            // so a mismatch silently writes another head's rows.
            if (s.ratio != ratio) continue;
            if (s.opt_in && !allow_opt_in) continue;
            if (!(avail & (1u << idx))) continue;
            return idx;
        }
    }
    // A ratio with table forms but no preference list is the same class of
    // failure as one whose forms are all dead: somebody added a shape without
    // the rule, and silently running per-head is how that ships.
    return any_form ? -2 : -1;
}

// Index / -1 / -2 as documented above; pick() keeps the historical
// index-or--1 contract for callers that do not care why nothing was chosen.
static int sofuu_attn_gqa_pick(uint32_t ratio, uint32_t avail, int allow_opt_in) {
    const int r = sofuu_attn_gqa_pick_ex(ratio, avail, allow_opt_in);
    return (r >= 0) ? r : -1;
}

// Public, GPU-free views of the table so the shape choice is testable from
// the Rust side: pick() is the shipped rule, pick_ex() also tells "no form for
// this ratio" (-1) from "forms exist, none usable" (-2), and the per-index
// accessors let a test assert the table invariants.
int sofuu_metal_attn_gqa_pick(uint32_t ratio, uint32_t avail_mask, int allow_opt_in) {
    return sofuu_attn_gqa_pick(ratio, avail_mask, allow_opt_in);
}
int sofuu_metal_attn_gqa_pick_ex(uint32_t ratio, uint32_t avail_mask, int allow_opt_in) {
    return sofuu_attn_gqa_pick_ex(ratio, avail_mask, allow_opt_in);
}
uint32_t sofuu_metal_attn_gqa_count(void) { return (uint32_t)SOFUU_GQA_N; }
uint32_t sofuu_metal_attn_gqa_threads(int idx) {
    if (idx < 0 || idx >= SOFUU_GQA_N) return 0u;
    return k_sofuu_gqa[idx].threads;
}
uint32_t sofuu_metal_attn_gqa_rg(int idx) {
    if (idx < 0 || idx >= SOFUU_GQA_N) return 0u;
    return k_sofuu_gqa[idx].rg;
}
uint32_t sofuu_metal_attn_gqa_ratio(int idx) {
    if (idx < 0 || idx >= SOFUU_GQA_N) return 0u;
    return k_sofuu_gqa[idx].ratio;
}
// Per-shape pipeline ceiling. Filled by sofuu_metal_init below (the same number
// the guard compares against); 0 means "no pipeline", which is what an opt-in
// or over-wide shape reports on a GPU that cannot host it.
static uint32_t g_sofuu_gqa_max_tg[SOFUU_GQA_N];
uint32_t sofuu_metal_attn_gqa_max_threads(int idx) {
    if (idx < 0 || idx >= SOFUU_GQA_N) return 0u;
    return g_sofuu_gqa_max_tg[idx];
}
// Diagnostic: the shape index the LAST chunk-attention dispatch selected, or -1
// when it went to the per-head kernel. The parity test uses it to prove a shape
// was really dispatched (batch_llama_attn_row also falls back to the per-head
// kernel, so a true return value alone would not). Never read on a model path.
static int g_sofuu_gqa_last = -1;
int sofuu_metal_attn_gqa_last(void) { return g_sofuu_gqa_last; }

#if defined(__APPLE__)

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso_gemv_bf16;
    id<MTLComputePipelineState> pso_gemv_f16;
    id<MTLComputePipelineState> pso_gemv_f32;
    id<MTLComputePipelineState> pso_gemv_mlx4_f16;
    id<MTLComputePipelineState> pso_gemv_mlx4_bf16;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_conv1d_3;
    // GPU1: in-batch glue kernels (whole-decode-step batches).
    id<MTLComputePipelineState> pso_add_inplace;
    // Qwen2/2.5 q/k/v bias epilogue over a [b][rows] slot (one dispatch).
    id<MTLComputePipelineState> pso_add_chan_bias4;
    // W2 CPU+GPU feature split: compact-range placement + two-owner residual.
    id<MTLComputePipelineState> pso_strided_copy_f32;
    id<MTLComputePipelineState> pso_resid_split_add_f32;
    id<MTLComputePipelineState> pso_lfm2_conv_gate;
    id<MTLComputePipelineState> pso_lfm2_kv_write;
    id<MTLComputePipelineState> pso_lfm2_attn;
    // G2: GGUF dequant-dot GEMV kernels.
    id<MTLComputePipelineState> pso_gemv_q8_0;
    id<MTLComputePipelineState> pso_gemv_q4_0;
    id<MTLComputePipelineState> pso_gemv_q4_k_c2r8;
    bool no_barrier = false;
    int attn_noop = 0;
    id<MTLComputePipelineState> pso_llama_attn_w4;
    id<MTLComputePipelineState> pso_llama_attn_w8;
    id<MTLComputePipelineState> pso_llama_attn_w16;
    id<MTLComputePipelineState> pso_llama_attn_w32;
    id<MTLComputePipelineState> pso_llama_attn_fast;
    // KV4 class (quantized KV cache): the block quantizer and the two attention
    // kernels that read the packed blocks plus the f32 residual ring.
    id<MTLComputePipelineState> pso_kv4_quant4;
    id<MTLComputePipelineState> pso_kv4_quant2;
    id<MTLComputePipelineState> pso_kv4_quant8;
    id<MTLComputePipelineState> pso_llama_attn_w32_kv8;
    id<MTLComputePipelineState> pso_llama_attn_pre_w32_kv8;
    id<MTLComputePipelineState> pso_llama_attn_w32_kv4;
    id<MTLComputePipelineState> pso_llama_attn_w32_kv2;
    id<MTLComputePipelineState> pso_llama_attn_pre_w32_kv4;
    id<MTLComputePipelineState> pso_llama_attn_pre_w32_kv2;
    // KV4 v2 (the dequant-cost rewrite of the two kernels above, see the block
    // comment in kernels.metal). Decode: one warp per key for all R q heads of a
    // kv head, per-block K scale/min vectors, V dequantized once per key, and
    // the key range split across threadgroups so the reuse does not cost
    // occupancy; pso_kv4_cmb merges the split partials. Prefill: KV4G_RG rows per
    // threadgroup, one warp per row. Indexed [bits 4|8][R 2|4]; nil means that
    // (bits, ratio) keeps the shipped kernel, which stays in the source.
    id<MTLComputePipelineState> pso_kv4g_dec[2][2];
    id<MTLComputePipelineState> pso_kv4g_pre[2][2];
    id<MTLComputePipelineState> pso_kv4_cmb;
    id<MTLBuffer> kv4_sk_scratch;
    size_t kv4_sk_cap;
    id<MTLComputePipelineState> pso_gemv_q4_k_flat;
    id<MTLComputePipelineState> pso_add_rmsnorm_f32w;
    id<MTLComputePipelineState> pso_gemv_q4_k_split1;
    id<MTLComputePipelineState> pso_gemv_q4_k_split2;
    id<MTLBuffer> gemv_partial;
    id<MTLComputePipelineState> pso_gemv_q4_k_secs;
    id<MTLComputePipelineState> pso_bw_probe_q4k;
    id<MTLComputePipelineState> pso_gemv_q4_k_v4;
    bool pso_gemv_q4_k_v4_live;
    id<MTLComputePipelineState> pso_gemv_q4_k_v3;
    bool pso_gemv_q4_k_v3_live;
    id<MTLComputePipelineState> pso_gemv_q4_k_v2;
    bool pso_gemv_q4_k_v2_live;
    id<MTLComputePipelineState> pso_gemv_q4_k_b1blk;
    id<MTLComputePipelineState> pso_gemv_q4_k_b1v2;
    id<MTLComputePipelineState> pso_gemv_q4_k;
    id<MTLComputePipelineState> pso_gemv_q6_k;
    id<MTLComputePipelineState> pso_gemv_q4_k_b4;
    id<MTLComputePipelineState> pso_gemm_q4_k_mma8;
    // Sectioned form of the same kernel: one dispatch for the projections that
    // share an activation row (q/k/v, gate/up). Same threadgroup geometry, so
    // it carries the same 128-thread maxTotalThreadsPerThreadgroup floor.
    id<MTLComputePipelineState> pso_gemm_q4_k_mma8_secs;
    // Same kernel with the residual add folded into the store (o, down).
    id<MTLComputePipelineState> pso_gemm_q4_k_mma8_acc;
    // fp16-input forms of the same tile (SOFUU_GPU_F16W=1): the weight k-slice
    // is read from a matrix materialised once at load instead of unpacked from
    // q4_K per tile. Same 128-thread geometry, same guard.
    id<MTLComputePipelineState> pso_gemm_f16_mma8;
    id<MTLComputePipelineState> pso_gemm_f16_mma8_secs;
    id<MTLComputePipelineState> pso_gemm_f16_mma8_acc;
    id<MTLComputePipelineState> pso_dequant_q4k_f16;

    // GPU2.1: one-commit-per-layer llama kernels.
    id<MTLComputePipelineState> pso_rmsnorm_f32w;
    id<MTLComputePipelineState> pso_rmsnorm_f32w_b4;
    id<MTLComputePipelineState> pso_llama_kv_write;
    id<MTLComputePipelineState> pso_llama_attn;
    id<MTLComputePipelineState> pso_head_rmsnorm;
    id<MTLComputePipelineState> pso_llama_attn_part;
    id<MTLComputePipelineState> pso_llama_attn_fin;
    id<MTLComputePipelineState> pso_llama_kv_write_b4;
    id<MTLComputePipelineState> pso_head_rmsnorm_b4;
    id<MTLComputePipelineState> pso_llama_attn_part_b4;
    id<MTLComputePipelineState> pso_llama_attn_fin_b4;
    id<MTLComputePipelineState> pso_llama_attn_pre_w32;
    // Whole-row chunk attention, PAR_ROWS rows per threadgroup (the 128-row
    // chain pass). Same 1024-thread/32-warp shape as pre_w32, so it carries the
    // same maxTotalThreadsPerThreadgroup guard in sofuu_metal_init: nil here
    // means "the b4 family owns attention", never a silent no-op dispatch.
    id<MTLComputePipelineState> pso_llama_attn_pre_row;
    // GQA-paired form of the same pass (kernels.metal llama_attn_pre_row_gqa):
    // one warp owns a row and EVERY q head that shares its kv head, so the K/V
    // pair is loaded once for R heads instead of once per head. Same 1024-thread
    // threadgroup and the same maxTotalThreadsPerThreadgroup guard; nil means
    // the per-head kernel above owns the chunk.
    // One entry per (heads-per-kv, prefetch, rows-per-threadgroup) shape of
    // llama_attn_pre_row_gqa; see the table in sofuu_metal_init.
    id<MTLComputePipelineState> pso_attn_gqa[SOFUU_GQA_N];
    // SOFUU_ATTN_GQA: unset = auto (the shape table picks the fastest LIVE
    // form for the model's ratio), 0 = per-head kernel, 1..5 = pin table entry
    // 0..4 (the pinned form is used when its pipeline is live, otherwise the
    // rule falls through to the fastest live sibling and says so).
    int attn_gqa_mode;
    int attn_gqa_pin;     // 1 when SOFUU_ATTN_GQA pinned a 2:1 form explicitly
    uint attn_gqa_rg[SOFUU_GQA_N];      // rows per threadgroup of each entry
    uint attn_gqa_threads[SOFUU_GQA_N]; // threads per threadgroup of each entry
    // The pipeline's own maxTotalThreadsPerThreadgroup per entry (0 when the
    // PSO is nil). Recorded even for an entry the guard nils, so the "why" line
    // can report the real ceiling instead of only naming the shape.
    uint attn_gqa_max_tg[SOFUU_GQA_N];
    // Ratios already reported as "forms exist, none usable" (bit = ratio); the
    // line is emitted once per ratio per engine, not once per layer chunk.
    uint64_t attn_gqa_reported;
    int attn_gqa_pin_reported;   // the "pinned shape unavailable" line, once
    uint64_t attn_gqa_traced;    // SOFUU_ATTN_GQA_TRACE: shapes already named
    // bit i set <=> pso_attn_gqa[i] survived the maxTotalThreadsPerThreadgroup
    // guard; sofuu_attn_gqa_pick turns it into a shape index per ratio.
    uint32_t attn_gqa_mask;
    int attn_gqa_all;     // SOFUU_ATTN_GQA_ALL=1 admits the opt-in shapes
    int attn_gqa_only;    // SOFUU_ATTN_GQA_ONLY=<idx>: pin the shape under test
    // Split-K (flash-decoding) twin of table entry 1 (r2/nr2/rg32): the same
    // 512-thread threadgroup, one key slice per threadgroup via grid z, so the
    // per-thread register footprint and the maxTotalThreadsPerThreadgroup guard
    // are the shipped ones. pso_attn_sk_combine is the one-warp reduction that
    // turns the S partials in attn_sk_scratch into the output row. Both nil =>
    // SOFUU_ATTN_SPLIT_K is ignored and the shipped kernel owns the chunk.
    id<MTLComputePipelineState> pso_attn_gqa_sk;
    id<MTLComputePipelineState> pso_attn_sk_combine;
    // Device scratch for the S partials: S*rows*n_heads*(head_dim+4) floats,
    // grown on demand (GPU-private: nothing on the host ever reads it).
    id<MTLBuffer> attn_sk_scratch;
    size_t attn_sk_cap;
    id<MTLComputePipelineState> pso_llama_attn_copy_b4;
    // Single source of truth for the llama_attn_pre_w32 whole-row fast path,
    // written by the part pass and read by the fin pass. They used to gate
    // independently (fin hardcoded 128 and skipped the head_dim / n_kv_heads
    // checks), so raising SOFUU_PREW_MAXPOS or a head_dim > 128 made them
    // disagree about who owns head_out - silent corruption (OpenQodex finding 1).
    int pre_window_active;
    id<MTLComputePipelineState> pso_llama_attn_1c_b4;
    id<MTLComputePipelineState> pso_gemv_q4_k_b4c2r;
    id<MTLComputePipelineState> pso_gemm_q4_k;
    id<MTLComputePipelineState> pso_quantize_q8_k;
    bool pso_quantize_q8_k_live;
    id<MTLComputePipelineState> pso_peak_fma16;
    id<MTLComputePipelineState> pso_peak_fma;
    id<MTLComputePipelineState> pso_peak_mma;
    id<MTLComputePipelineState> pso_gemm_q8_0_mma64;
    bool pso_gemm_q8_0_mma64_live;
    id<MTLComputePipelineState> pso_gemm_q6_k_mma64;
    bool pso_gemm_q6_k_mma64_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_mma64;
    bool pso_gemm_q4_k_mma64_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_tiled_r88;
    bool pso_gemm_q4_k_tiled_r88_live;
    bool pso_gemm_q4_k_tiled_db_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_tiled_wide_fused;
    bool pso_gemm_q4_k_tiled_wide_fused_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_tiled_wide;
    bool pso_gemm_q4_k_tiled_wide_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_mma;
    bool pso_gemm_q4_k_mma_live;
    id<MTLComputePipelineState> pso_attn_flash_b4;
    bool pso_attn_flash_b4_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_tiled;
    bool pso_gemm_q4_k_tiled_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_tiled_fused;
    bool pso_gemm_q4_k_tiled_fused_live;
    id<MTLComputePipelineState> pso_gemm_q6_k_tiled_fused;
    bool pso_gemm_q6_k_tiled_fused_live;
    id<MTLComputePipelineState> pso_gemm_q8_0_tiled_fused;
    bool pso_gemm_q8_0_tiled_fused_live;
    // q4_0's wide-chunk GEMM: same 32x64 tile as q8_0 (q4_0 is also one block
    // per 32 values), so the whole q4_0 fleet stops walking one GEMV per token.
    id<MTLComputePipelineState> pso_gemm_q4_0_tiled_fused;
    bool pso_gemm_q4_0_tiled_fused_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_mmq;
    bool pso_gemm_q4_k_mmq_live;
    id<MTLComputePipelineState> pso_gemm_q6_k_mmq;
    bool pso_gemm_q6_k_mmq_live;
    id<MTLComputePipelineState> pso_quantize_q8_k_pos;
    bool pso_quantize_q8_k_pos_live;
    id<MTLComputePipelineState> pso_quantize_q8_ext;
    bool pso_quantize_q8_ext_live;
    id<MTLComputePipelineState> pso_gemv_q4_k_mmq32;
    bool pso_gemv_q4_k_mmq32_live;
    id<MTLComputePipelineState> pso_gemv_q6_k_mmq32;
    bool pso_gemv_q6_k_mmq32_live;
    id<MTLComputePipelineState> pso_gemm_q4_k_multi;
    id<MTLComputePipelineState> pso_gemm_q4_k_acc;
    id<MTLComputePipelineState> pso_gemm_q8_0;
    id<MTLComputePipelineState> pso_gemm_q8_0_multi;
    id<MTLComputePipelineState> pso_gemm_q8_0_acc;
    id<MTLComputePipelineState> pso_gemm_q6_k;
    id<MTLComputePipelineState> pso_gemm_q6_k_multi;
    id<MTLComputePipelineState> pso_gemm_q6_k_acc;
    id<MTLComputePipelineState> pso_gemm_q8_0_t32;
    id<MTLComputePipelineState> pso_gemm_q6_k_t32;
    id<MTLComputePipelineState> pso_gpu_q4_k_b4multi;
    id<MTLComputePipelineState> pso_gpu_q6_k_b4multi;
    id<MTLComputePipelineState> pso_gpu_q8_0_b4multi;
    id<MTLComputePipelineState> pso_gpu_q8_0_b4r4;
    bool pso_gemm_q4_k_live;
    bool pso_gemm_q4_k_multi_live;
    bool pso_gemm_q4_k_acc_live;
    bool pso_gemm_q8_0_live;
    bool pso_gemm_q8_0_multi_live;
    bool pso_gemm_q8_0_acc_live;
    bool pso_gemm_q6_k_live;
    bool pso_gemm_q6_k_multi_live;
    bool pso_gemm_q6_k_acc_live;
    bool pso_gemm_q8_0_t32_live;
    bool pso_gemm_q6_k_t32_live;
    // SOFUU_GEMM_TILE=32 selects the lane-token tile class (weights read
    // once per threadgroup; the tile-8 class re-reads weights per 8 tokens,
    // 16x at B=128). Read once at context init.
    bool gemm_tile32;
    bool pso_gpu_q4_k_b4multi_live;
    bool pso_gpu_q6_k_b4multi_live;
    bool pso_gpu_q8_0_b4multi_live;
    bool pso_gpu_q8_0_b4r4_live;

    id<MTLBuffer> x_buf;
    id<MTLBuffer> y_buf;
    size_t x_buf_cap;
    size_t y_buf_cap;

    id<MTLCommandBuffer> pending_cmd;
    id<MTLComputeCommandEncoder> batch_enc;
    size_t pending_y_bytes;
    float* pending_y_dst;
    // W1 CPU+GPU row split: command buffers committed with
    // sofuu_metal_batch_commit_async are kept alive here until
    // sofuu_metal_batch_wait_all drains them. The queue is serial, so waiting
    // on the last one implies every earlier one completed.
    void* inflight[64];
    int inflight_n = 0;
};

// Batch barrier. SOFUU_NO_BARRIER=1 drops every inter-dispatch barrier: only valid
// with SOFUU_HAZARD=untracked, and only useful as an upper bound on what the
// barriers cost, because the decode chain has a real dependency at every step.
static inline void sofuu_batch_barrier(MetalContext* ctx) {
    if (ctx->no_barrier) return;
    // LEDGER #190: layer mode gets its barrier from sofuu_metal_batch_barrier_mark.
    static int layer_mode = -1;
    if (layer_mode < 0) {
        const char* m = getenv("SOFUU_BARRIER_MODE");
        layer_mode = (m && strcmp(m, "layer") == 0) ? 1 : 0;
    }
    if (layer_mode == 1) return;
    // LEDGER #189 probe: emit the buffer barrier only every Nth call, to measure how many
    // of the 168 per decode token are actually required. SOFUU_BARRIER_EVERY=1 is the
    // shipped behaviour.
    static int every = -2;
    static unsigned long bcnt = 0;
    if (every == -2) {
        const char* e = getenv("SOFUU_BARRIER_EVERY");
        every = e ? atoi(e) : 1;
        if (every < 1) every = 1;
    }
    if (every > 1 && (bcnt++ % (unsigned long)every) != 0) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// ── shared RoPE binding (kernels.metal sofuu_rope_apply) ────────────────────
// Every Metal rope site (decode single-token q/k, the batched rope+KV prefill
// pass, the speculative-tree chain pass) binds the SAME six slots at the tail
// of its argument list: table pointer, has_table, n_rot, mscale, factor
// pointer, n_pairs. Keeping the binding in one helper is what makes the sites
// impossible to drift apart - the parameterisation is written once.
//
// * tab_ptr/tab_off   host-built cos|sin rows (quant::rope_table_rows). The
//   kernels only read it when has_table != 0.
// * n_rot             dims rotated per head; 0 or > head_dim means the whole
//   head (the sentinel every other backend uses).
// * adj/n_pairs       per-pair inverse-frequency multipliers (= 1/ggml factor).
//   Only read on the on-device theta path (has_table == 0).
// A nil pointer is never bound: the fallback is a buffer every caller of that
// kernel already has bound, and the kernels guard every read behind the flags.
static inline void sofuu_bind_rope(
    MetalContext* ctx, int base,
    void* tab_ptr, size_t tab_off, uint32_t has_table,
    void* adj_ptr, size_t adj_off, uint32_t n_pairs,
    uint32_t head_dim, uint32_t n_rot, float mscale,
    void* fallback)
{
    uint nr = (n_rot == 0u || n_rot > head_dim) ? head_dim : n_rot;
    uint ht = has_table;
    float ms = mscale;
    uint np = n_pairs;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(tab_ptr ? tab_ptr : fallback)
                      offset:(tab_ptr ? tab_off : 0) atIndex:base];
    [ctx->batch_enc setBytes:&ht length:sizeof(uint) atIndex:base + 1];
    [ctx->batch_enc setBytes:&nr length:sizeof(uint) atIndex:base + 2];
    [ctx->batch_enc setBytes:&ms length:sizeof(float) atIndex:base + 3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(adj_ptr ? adj_ptr : fallback)
                      offset:(adj_ptr ? adj_off : 0) atIndex:base + 4];
    [ctx->batch_enc setBytes:&np length:sizeof(uint) atIndex:base + 5];
}

// The shader source lives in src/metal/kernels.metal; build.rs generates it
// into this header so the RUNTIME-compiled path and the shipped
// default.metallib (scripts/dist/ios.sh) can never disagree about which
// kernels exist. See the note in crates/sofuu-ffi/build.rs.
//
// AUDIT-2026-10-31 P1-8: the header carries the source gzipped, because the
// raw literal was 403,186 B of every shipped binary for text only this one
// fallback reads. sofuu_metal_kernel_source() inflates it back.
#include "kernels_metal.h"

// Inflate the embedded MSL source (SOFUU_METAL_MSL_GZ, generated). Called once
// per sofuu_metal_init, and only when there is no complete bundled
// default.metallib. Returns nil - with a log line, never a guess - if the
// payload does not inflate to exactly SOFUU_METAL_MSL_LEN bytes.
static NSString* sofuu_metal_kernel_source(void) {
    const unsigned long want = SOFUU_METAL_MSL_LEN;
    if (want == 0 || SOFUU_METAL_MSL_GZ_LEN < 2) return nil;
    // +1 so the buffer is always NUL-terminated for stringWithUTF8String.
    NSMutableData* out = [NSMutableData dataWithLength:(NSUInteger)(want + 1)];
    if (out == nil) return nil;
    z_stream zs;
    memset(&zs, 0, sizeof zs);
    zs.next_in = (Bytef*)SOFUU_METAL_MSL_GZ;
    zs.avail_in = (uInt)SOFUU_METAL_MSL_GZ_LEN;
    zs.next_out = (Bytef*)[out mutableBytes];
    zs.avail_out = (uInt)want;
    // 15 + 16: DEFLATE wrapped in the gzip header build.rs writes.
    if (inflateInit2(&zs, 15 + 16) != Z_OK) return nil;
    int rc = inflate(&zs, Z_FINISH);
    const unsigned long got = (unsigned long)zs.total_out;
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || got != want) {
        NSLog(@"Sofuu Metal: embedded kernel source did not inflate "
              @"(zlib rc=%d, %lu/%lu bytes) - rebuild so kernels_metal.h "
              @"matches src/metal/kernels.metal",
              rc, got, want);
        return nil;
    }
    return [NSString stringWithUTF8String:(const char*)[out bytes]];
}


static int g_gemv_b1_override = -1;
static double g_last_gpu_us = 0.0;

// Decode (b=1) q4_K GEMV kernel choice (SOFUU_GEMV_B1):
//   6 (default) = gemv_q4_k_b1v2   — the shipped decode kernel
//   4 or 1      = gemv_q4_k_b1blk  — the kernel it replaced, kept as the A/B
//                                    reference (bit-identical output, measured
//                                    ~5% slower on the real token GEMV stream)
//   0           = the older per-row gemv_q4_k_c2 path
// Arms are read once per process; sofuu_metal_set_gemv_b1() lets the GEMV probe
// switch them inside one process (SOFUU_GEMV_BENCH_AB).
static inline int sofuu_gemv_b1_on(void) {
    static int on = -1;
    if (on < 0) {
        const char* v = getenv("SOFUU_GEMV_B1");
        on = v ? atoi(v) : 6;
    }
    return on;
}

// A/B instrumentation: lets the GEMV probe switch the decode q4_K kernel inside
// one process, so the two arms are measured against the same machine state.
extern "C" void sofuu_metal_set_gemv_b1(int v) { g_gemv_b1_override = v; }
// Device-side (GPUStartTime/GPUEndTime) duration of the last committed batch.
extern "C" double sofuu_metal_last_gpu_us(void) { return g_last_gpu_us; }
static inline int sofuu_gemv_b1_arm(void) {
    return g_gemv_b1_override >= 0 ? g_gemv_b1_override : sofuu_gemv_b1_on();
}

// Decode attention warps per query head: 32 (llama_attn_w32) by default, 16 (the
// shipped llama_attn_w16) with SOFUU_ATTN_W32=0 for an A/B on one binary.
static inline int sofuu_attn_w32_on(void) {
    static int on = -1;
    if (on < 0) {
        const char* v = getenv("SOFUU_ATTN_W32");
        on = v ? atoi(v) : 1;
    }
    return on;
}

// Prefill chunk attention: llama_attn_pre_w32 computes whole rows with an online
// softmax over the K/V cache instead of llama_attn_part_b4's one-warp-per-
// (row, head, key-chunk) three-pass sweep. SOFUU_PRE_ATTN=0 restores the old
// pair.
//
// SOFUU_PRE_ATTN_RPT (1..4) is accepted and forwarded to the kernel but no
// longer changes anything: the shipped kernel is one threadgroup per (row, head)
// and takes no row grouping, so the value only reaches the kernel's (void)rpt.
// It stays bound because the kernel signature is shared with the earlier
// row-grouped variants; a row-grouped kernel (R rows sharing one K/V load)
// needs ~93 registers/thread here, which drops the pipeline to 704 threads per
// threadgroup and makes the 1024-thread dispatch silently vanish - see the
// register-pressure note in sofuu_metal_init before trying it again.
static inline int sofuu_pre_attn_on(void) {
    static int on = -1;
    if (on < 0) {
        const char* v = getenv("SOFUU_PRE_ATTN");
        on = v ? atoi(v) : 1;
    }
    return on;
}

// Highest max-position a chunk may contain for the whole-row fast path to run.
// Default 0xFFFFFFFF = no bound, i.e. the fast path runs for every position in
// the prompt (validated 39..1454 tokens; the per-token-count deltas are in the
// gate comment in sofuu_metal_batch_llama_attn_part_b4). It also doubles as the
// rollback knob: SOFUU_PREW_MAXPOS=128 restores the old one-128-key-window
// behaviour and hands every longer chunk back to the shipped part/fin pair.
static inline uint32_t sofuu_prew_maxpos(void) {
    static long mp = -2;
    if (mp < 0) {
        const char* v = getenv("SOFUU_PREW_MAXPOS");
        mp = v ? atol(v) : 0xFFFFFFFFL;
    }
    return (uint32_t)mp;
}

static inline uint sofuu_pre_attn_rpt(void) {
    static long rpt = -1;
    if (rpt < 0) {
        const char* v = getenv("SOFUU_PRE_ATTN_RPT");
        rpt = v ? atol(v) : 4;
        if (rpt < 1) rpt = 1;
        if (rpt > 4) rpt = 4;
    }
    return (uint)rpt;
}


extern "C" void sofuu_metal_batch_barrier_mark(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return;
    if (ctx->no_barrier) return;
    [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

bool sofuu_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil) {
            return true;
        }
        return false;
    }
}

void* sofuu_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) return NULL;

        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (q == nil) return NULL;

        // Two ways to obtain the compute pipelines, tried in this order:
        //
        //   1. `default.metallib` in the app bundle — kernels.metal compiled
        //      at BUILD time. Xcode compiles the .metal files in the app
        //      target into the bundle's default library; the iOS framework
        //      ships one in Resources/ as well (scripts/dist/ios.sh). No
        //      runtime shader compilation, and a shader that fails to build
        //      is caught by the build, not by a user's first prompt.
        //   2. The same source compiled at RUNTIME. This is the only path
        //      the macOS CLI has (it ships no bundle) and the fallback
        //      everywhere else.
        //
        // A bundled library is accepted only when EVERY pipeline the engine
        // loads is present: a stale or partial metallib must degrade to (2),
        // never to a nil pipeline state at inference time.
        //
        // iOS floor note: the bf16 kernels use `bfloat`, which the Metal
        // language gained in MSL 3.1 (iOS 17). On an older OS the bundled
        // library will not load AND the runtime compiler does not know the
        // type, so init returns NULL and the engine stays on the CPU path —
        // the designed degradation, logged below.
        NSArray<NSString*>* required = @[
            @"gemv_bf16", @"gemv_f16", @"gemv_f32",
            @"gemv_mlx4_affine_f16", @"gemv_mlx4_affine_bf16",
            @"rmsnorm", @"swiglu", @"conv1d_3",
            @"add_inplace", @"add_chan_bias4", @"lfm2_conv_gate", @"lfm2_kv_write", @"lfm2_attn",
            @"gemv_q8_0", @"gemv_q4_0", @"gemv_q4_k", @"gemv_q6_k",
            @"rmsnorm_f32w", @"llama_kv_write", @"llama_attn",
            @"head_rmsnorm", @"llama_attn_part", @"llama_attn_fin",
            @"gemv_q4_k_b4"
        ];

        id<MTLLibrary> lib = nil;
        id<MTLLibrary> bundled = [dev newDefaultLibrary];
        if (bundled != nil) {
            BOOL complete = YES;
            for (NSString* name in required) {
                if ([bundled newFunctionWithName:name] == nil) {
                    complete = NO;
                    break;
                }
            }
            if (complete) lib = bundled;
        }

        NSError* error = nil;
        if (lib == nil) {
            // AUDIT-2026-10-31 P1-8: the source is embedded gzipped and
            // inflated here — the cold path that already compiles it.
            NSString* src = sofuu_metal_kernel_source();
            if (src == nil) {
                NSLog(@"Sofuu Metal: no usable shader library — bundled "
                      @"default.metallib missing or incomplete, and the "
                      @"embedded kernel source is unreadable. Running on CPU.");
                return NULL;
            }
            lib = [dev newLibraryWithSource:src options:nil error:&error];
            if (lib == nil) {
                NSLog(@"Sofuu Metal: no usable shader library — bundled "
                      @"default.metallib missing or incomplete, and runtime "
                      @"compile failed (%@). bf16 kernels need MSL 3.1 / "
                      @"iOS 17+; running on CPU.", error);
                return NULL;
            }
            NSLog(@"Sofuu Metal: kernels compiled at runtime (no complete "
                  @"default.metallib in the bundle)");
        } else {
            NSLog(@"Sofuu Metal: using the bundled default.metallib "
                  @"(precompiled kernels)");
        }

        // env-gated q4_K kernel class: coalesced-lane vs shipped scalar
        BOOL q4k_coalesced = NO;
        auto load_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];            if (fn == nil) return nil;
            NSError* pso_err = nil;
            return [dev newComputePipelineStateWithFunction:fn error:&pso_err];
        };

        MetalContext* ctx = new MetalContext();
        ctx->device = dev;
        ctx->queue = q;
        ctx->pso_gemv_bf16 = load_pso(@"gemv_bf16");
        ctx->pso_gemv_f16  = load_pso(@"gemv_f16");
        ctx->pso_gemv_f32  = load_pso(@"gemv_f32");
        ctx->pso_gemm_q4_k_mma8 = load_pso(@"gemm_q4_k_mma8");
        // Same maxTotalThreadsPerThreadgroup discipline as llama_attn_pre_w32
        // (see the 1024-thread guard below): the mma8 GEMM dispatches 128
        // threads (4 simdgroups) per threadgroup and Metal DROPS an oversized
        // threadgroup silently in release builds - the output would read back
        // as untouched scratch (all-zero heads), not as an error. Check the
        // limit at init and disable the pipeline rather than dispatching it;
        // sofuu_metal_batch_gguf_mma8 returns 0 on a nil PSO and every caller
        // falls back to the per-tensor path.
        if (ctx->pso_gemm_q4_k_mma8 != nil &&
            ctx->pso_gemm_q4_k_mma8.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q4_k_mma8 needs 128 threads/group, pipeline allows %lu - mma8 disabled\n",
                    (unsigned long)ctx->pso_gemm_q4_k_mma8.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q4_k_mma8 = nil;
        }
        ctx->pso_gemm_q4_k_mma8_secs = load_pso(@"gemm_q4_k_mma8_secs");
        if (ctx->pso_gemm_q4_k_mma8_secs != nil &&
            ctx->pso_gemm_q4_k_mma8_secs.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q4_k_mma8_secs needs 128 threads/group, pipeline allows %lu - sectioned mma8 disabled\n",
                    (unsigned long)ctx->pso_gemm_q4_k_mma8_secs.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q4_k_mma8_secs = nil;
        }
        ctx->pso_gemm_f16_mma8 = load_pso(@"gemm_f16_mma8");
        if (ctx->pso_gemm_f16_mma8 != nil &&
            ctx->pso_gemm_f16_mma8.maxTotalThreadsPerThreadgroup < 128ul) {
            NSLog(@"[metal] gemm_f16_mma8 needs 128 threads/group, pipeline allows %lu - fp16 chain GEMM disabled",
                  (unsigned long)ctx->pso_gemm_f16_mma8.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_f16_mma8 = nil;
        }
        ctx->pso_gemm_f16_mma8_secs = load_pso(@"gemm_f16_mma8_secs");
        if (ctx->pso_gemm_f16_mma8_secs != nil &&
            ctx->pso_gemm_f16_mma8_secs.maxTotalThreadsPerThreadgroup < 128ul) {
            NSLog(@"[metal] gemm_f16_mma8_secs needs 128 threads/group, pipeline allows %lu - fp16 sectioned GEMM disabled",
                  (unsigned long)ctx->pso_gemm_f16_mma8_secs.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_f16_mma8_secs = nil;
        }
        ctx->pso_gemm_f16_mma8_acc = load_pso(@"gemm_f16_mma8_acc");
        if (ctx->pso_gemm_f16_mma8_acc != nil &&
            ctx->pso_gemm_f16_mma8_acc.maxTotalThreadsPerThreadgroup < 128ul) {
            NSLog(@"[metal] gemm_f16_mma8_acc needs 128 threads/group, pipeline allows %lu - fp16 residual GEMM disabled",
                  (unsigned long)ctx->pso_gemm_f16_mma8_acc.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_f16_mma8_acc = nil;
        }
        // 128-thread flat grid, same floor: a >max dispatch is dropped silently
        // in release, which here would leave the arena half-written.
        ctx->pso_dequant_q4k_f16 = load_pso(@"dequant_q4k_f16");
        if (ctx->pso_dequant_q4k_f16 != nil &&
            ctx->pso_dequant_q4k_f16.maxTotalThreadsPerThreadgroup < 128ul) {
            NSLog(@"[metal] dequant_q4k_f16 needs 128 threads/group, pipeline allows %lu - fp16 weight staging disabled",
                  (unsigned long)ctx->pso_dequant_q4k_f16.maxTotalThreadsPerThreadgroup);
            ctx->pso_dequant_q4k_f16 = nil;
        }
        ctx->pso_gemm_q4_k_mma8_acc = load_pso(@"gemm_q4_k_mma8_acc");
        if (ctx->pso_gemm_q4_k_mma8_acc != nil &&
            ctx->pso_gemm_q4_k_mma8_acc.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q4_k_mma8_acc needs 128 threads/group, pipeline allows %lu - mma8 acc epilogue disabled\n",
                    (unsigned long)ctx->pso_gemm_q4_k_mma8_acc.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q4_k_mma8_acc = nil;
        }
        ctx->pso_gemv_mlx4_f16  = load_pso(@"gemv_mlx4_affine_f16");
        ctx->pso_gemv_mlx4_bf16 = load_pso(@"gemv_mlx4_affine_bf16");
        ctx->pso_rmsnorm   = load_pso(@"rmsnorm");
        ctx->pso_swiglu    = load_pso(@"swiglu");
        ctx->pso_conv1d_3  = load_pso(@"conv1d_3");
        ctx->pso_add_inplace     = load_pso(@"add_inplace");
        ctx->pso_add_chan_bias4  = load_pso(@"add_chan_bias4");
        ctx->pso_strided_copy_f32    = load_pso(@"strided_copy_f32");
        ctx->pso_resid_split_add_f32 = load_pso(@"resid_split_add_f32");
        ctx->pso_lfm2_conv_gate  = load_pso(@"lfm2_conv_gate");
        ctx->pso_lfm2_kv_write   = load_pso(@"lfm2_kv_write");
        ctx->pso_lfm2_attn       = load_pso(@"lfm2_attn");
        ctx->pso_gemv_q8_0 = load_pso(@"gemv_q8_0");
        ctx->pso_gemv_q4_0 = load_pso(@"gemv_q4_0");
        ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k");
        ctx->pso_gemv_q4_k_b1blk = load_pso(@"gemv_q4_k_b1blk");
        ctx->pso_gemv_q4_k_b1v2 = load_pso(@"gemv_q4_k_b1v2");
        // SOFUU_Q4K_COALESCED=1 flips BOTH q4_K PSOs to the coalesced-lane
        // kernels — one kernel class per engine instance so serial, spec and
        // b4 GEMVs all share the same f32 summation order (text parity holds
        // internally; the coalesced class has its own CPU-oracle parity gate).
        // Arm A (plan): class 2 is now the DEFAULT. It is the faster decode
        // GEMV mapping (b4c2 100 GB/s vs 69 for class 1, head q6_K 90.5 vs
        // 15.5) and it was gate-clean on the whole fleet, so the knob became
        // an opt-OUT: SOFUU_Q4K_COALESCED=0 restores the pre-arm-A mapping,
        // =1 pins class 1, =2 pins class 2. Unset means 2.
        const char* q4k_class = getenv("SOFUU_Q4K_COALESCED");
        int coalesced_level = 2;
        if (q4k_class != NULL) {
            coalesced_level = (strcmp(q4k_class, "0") != 0) ? atoi(q4k_class) : 0;
        }
        q4k_coalesced = coalesced_level > 0;
        ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
        ctx->pso_rmsnorm_f32w  = load_pso(@"rmsnorm_f32w");
        ctx->pso_rmsnorm_f32w_b4 = load_pso(@"rmsnorm_f32w_b4");
        ctx->pso_llama_kv_write = load_pso(@"llama_kv_write");
        ctx->pso_llama_attn     = load_pso(@"llama_attn");
        ctx->pso_head_rmsnorm    = load_pso(@"head_rmsnorm");
        ctx->pso_llama_attn_part = load_pso(@"llama_attn_part");
        ctx->pso_llama_attn_fin  = load_pso(@"llama_attn_fin");
        ctx->pso_llama_kv_write_b4   = load_pso(@"llama_kv_write_b4");
        ctx->pso_head_rmsnorm_b4     = load_pso(@"head_rmsnorm_b4");
        ctx->pso_llama_attn_part_b4  = load_pso(@"llama_attn_part_b4");
        ctx->pso_llama_attn_fin_b4   = load_pso(@"llama_attn_fin_b4");
        ctx->pso_llama_attn_pre_w32  = load_pso(@"llama_attn_pre_w32");
        // The whole-row fast path dispatches 1024 threads per threadgroup (32
        // warps) and keeps its per-warp softmax state in registers. A kernel
        // whose register footprint pushes maxTotalThreadsPerThreadgroup below
        // 1024 does NOT fail loudly: the oversized threadgroup is dropped, the
        // meta scratch keeps whatever it held, and the result reads as "one head
        // correct, the rest zero". Disable the path here - the gate in
        // sofuu_metal_batch_llama_attn_part_b4 then runs the shipped pair -
        // rather than dispatching a threadgroup the pipeline cannot host.
        if (ctx->pso_llama_attn_pre_w32 != nil &&
            ctx->pso_llama_attn_pre_w32.maxTotalThreadsPerThreadgroup < 1024ul) {
            fprintf(stderr,
                    "[metal] llama_attn_pre_w32 needs 1024 threads/group, pipeline allows %lu - fast path disabled\n",
                    (unsigned long)ctx->pso_llama_attn_pre_w32.maxTotalThreadsPerThreadgroup);
            ctx->pso_llama_attn_pre_w32 = nil;
        }
        ctx->pso_llama_attn_pre_row = load_pso(@"llama_attn_pre_row");
        // Same register-budget discipline as pso_llama_attn_pre_w32 above: the
        // kernel is 32 warps (1024 threads) and Metal DROPS an oversized
        // threadgroup without an error, so the output buffer keeps whatever the
        // previous layer left there. Nil the PSO out and let the caller run the
        // b4 family rather than dispatching into the void.
        if (ctx->pso_llama_attn_pre_row != nil &&
            ctx->pso_llama_attn_pre_row.maxTotalThreadsPerThreadgroup < 1024ul) {
            fprintf(stderr,
                    "[metal] llama_attn_pre_row needs 1024 threads/group, pipeline allows %lu - row fast path disabled\n",
                    (unsigned long)ctx->pso_llama_attn_pre_row.maxTotalThreadsPerThreadgroup);
            ctx->pso_llama_attn_pre_row = nil;
        }

        // KV4 (quantized-KV) class: the block quantizer and the two attention
        // kernels that read the packed blocks + the f32 residual ring. Loaded
        // unconditionally - --kv-dtype=4 selects this class, so a GEMV tuning
        // level must not gate it. Same 1024-thread discipline as
        // llama_attn_pre_w32 above: Metal DROPS an oversized threadgroup
        // without an error, so a pipeline whose register budget falls below
        // 1024 threads is nilled here and reported by
        // sofuu_metal_kv4_support() instead of being dispatched into the void.
        ctx->pso_kv4_quant4 = load_pso(@"kv4_quant_b4");
        ctx->pso_kv4_quant2 = load_pso(@"kv4_quant_b2");
        ctx->pso_kv4_quant8 = load_pso(@"kv4_quant_b8");
        ctx->pso_llama_attn_w32_kv8 = load_pso(@"llama_attn_w32_kv8");
        ctx->pso_llama_attn_pre_w32_kv8 = load_pso(@"llama_attn_pre_w32_kv8");
        ctx->pso_llama_attn_w32_kv4 = load_pso(@"llama_attn_w32_kv4");
        ctx->pso_llama_attn_w32_kv2 = load_pso(@"llama_attn_w32_kv2");
        ctx->pso_llama_attn_pre_w32_kv4 = load_pso(@"llama_attn_pre_w32_kv4");
        ctx->pso_llama_attn_pre_w32_kv2 = load_pso(@"llama_attn_pre_w32_kv2");
        if (ctx->pso_llama_attn_w32_kv4 != nil &&
            ctx->pso_llama_attn_w32_kv4.maxTotalThreadsPerThreadgroup < 1024ul) {
            fprintf(stderr,
                    "[metal] llama_attn_w32_kv4 needs 1024 threads/group, pipeline allows %lu - 4-bit KV decode disabled\n",
                    (unsigned long)ctx->pso_llama_attn_w32_kv4.maxTotalThreadsPerThreadgroup);
            ctx->pso_llama_attn_w32_kv4 = nil;
            ctx->pso_llama_attn_w32_kv2 = nil;
        }
        if (ctx->pso_llama_attn_pre_w32_kv4 != nil &&
            ctx->pso_llama_attn_pre_w32_kv4.maxTotalThreadsPerThreadgroup < 1024ul) {
            fprintf(stderr,
                    "[metal] llama_attn_pre_w32_kv4 needs 1024 threads/group, pipeline allows %lu - 4-bit KV prefill disabled\n",
                    (unsigned long)ctx->pso_llama_attn_pre_w32_kv4.maxTotalThreadsPerThreadgroup);
            ctx->pso_llama_attn_pre_w32_kv4 = nil;
            ctx->pso_llama_attn_pre_w32_kv2 = nil;
        }
        // KV4 v2 pipelines. Same discipline as above: they dispatch 1024
        // threads, and Metal drops an oversized threadgroup silently, so a
        // pipeline whose register budget fell below 1024 is nilled here and the
        // caller falls back to the shipped kv4 kernel for that (bits, ratio).
        {
            static const char* k_kv4g_dec[2][2] = {
                { "attn_kv4g_b4_r2", "attn_kv4g_b4_r4" },
                { "attn_kv4g_b8_r2", "attn_kv4g_b8_r4" },
            };
            static const char* k_kv4g_pre[2][2] = {
                { "attn_pre_row_kv4_b4_r2", "attn_pre_row_kv4_b4_r4" },
                { "attn_pre_row_kv4_b8_r2", "attn_pre_row_kv4_b8_r4" },
            };
            for (int bi = 0; bi < 2; ++bi) {
                for (int ri = 0; ri < 2; ++ri) {
                    // v2 kernels take their threadgroup size from the dispatch
                    // (sofuu_kv4g_threads), so the only hard floor is 8 warps:
                    // below that a threadgroup cannot cover head_dim channels
                    // with one warp and the slice bookkeeping degrades.
                    id<MTLComputePipelineState> pd = load_pso(@(k_kv4g_dec[bi][ri]));
                    if (pd != nil && pd.maxTotalThreadsPerThreadgroup < 256ul) {
                        fprintf(stderr,
                                "[metal] %s needs >=256 threads/group, pipeline allows %lu - KV4 v2 decode disabled for that shape\n",
                                k_kv4g_dec[bi][ri], (unsigned long)pd.maxTotalThreadsPerThreadgroup);
                        pd = nil;
                    }
                    ctx->pso_kv4g_dec[bi][ri] = pd;
                    id<MTLComputePipelineState> pp = load_pso(@(k_kv4g_pre[bi][ri]));
                    if (pp != nil && pp.maxTotalThreadsPerThreadgroup < 256ul) {
                        fprintf(stderr,
                                "[metal] %s needs >=256 threads/group, pipeline allows %lu - KV4 v2 prefill disabled for that shape\n",
                                k_kv4g_pre[bi][ri], (unsigned long)pp.maxTotalThreadsPerThreadgroup);
                        pp = nil;
                    }
                    ctx->pso_kv4g_pre[bi][ri] = pp;
                    if (pd != nil || pp != nil) {
                        fprintf(stderr, "[metal] KV4 v2 %u-bit R=%u: %lu threads/group\n",
                                (bi == 1) ? 8u : 4u, (ri == 1) ? 4u : 2u,
                                (unsigned long)(pd != nil ? pd.maxTotalThreadsPerThreadgroup
                                                          : pp.maxTotalThreadsPerThreadgroup));
                    }
                }
            }
            ctx->pso_kv4_cmb = load_pso(@"llama_attn_kv4_cmb");
        }
        // GQA-paired whole-row chunk attention: one table, one guard loop.
        // Index -> (kernel, rows per threadgroup). Entry 0 is the shape this
        // model wants (2 q heads per kv head, 32 rows = 1024 threads); entries
        // 1..3 keep the paired loads but shrink the row group so more
        // threadgroups are independently schedulable; entry 4 is the
        // prefetched K/V form; entry 5 is the 4:1 GQA shape (16 rows = 512
        // threads, the 1024 shape exceeds its register budget and cannot run).
        {
            // rg = rows per threadgroup, threads = (rg/nr)*32 after the kernel's
            // rows-per-warp. Entry 0 is the 2:1 GQA pairing at the per-head
            // kernel's own shape; entries 2..4 also pack rows into each warp,
            // which is what cuts the K/V bytes per (row, head, key).
            // The table itself is file-scope (k_sofuu_gqa) so the same
            // entries feed the pure selector; this loop only loads and guards
            // them. Every entry carries its own thread count, and a pipeline
            // that cannot host it is nil'd here: an oversized threadgroup is
            // silently DROPPED by Metal in release, which reads back as dead
            // attention, so the guard is the only thing between a new shape and
            // a wrong answer.
            ctx->attn_gqa_mask = 0u;
            ctx->attn_gqa_reported = 0u;
            ctx->attn_gqa_pin_reported = 0;
            ctx->attn_gqa_traced = 0u;
            ctx->attn_gqa_all = getenv("SOFUU_ATTN_GQA_ALL") != nullptr;
            // SOFUU_ATTN_GQA_ONLY=<idx> keeps exactly one shape selectable. It
            // exists so a shape the auto-rule would not pick on THIS GPU (e.g.
            // the 256-thread 4:1 form on a box that allows 512) can still be
            // dispatched and checked against the per-head kernel - a shape that
            // has never been run must never be selectable by the default rule.
            int only_idx = -1;
            if (const char *only_env = getenv("SOFUU_ATTN_GQA_ONLY")) {
                only_idx = atoi(only_env);
                if (only_idx < 0 || only_idx >= SOFUU_GQA_N) only_idx = -1;
            }
            ctx->attn_gqa_only = only_idx;
            for (int gi = 0; gi < SOFUU_GQA_N; ++gi) {
                ctx->attn_gqa_rg[gi] = k_sofuu_gqa[gi].rg;
                ctx->attn_gqa_threads[gi] = k_sofuu_gqa[gi].threads;
                ctx->attn_gqa_max_tg[gi] = 0u;
                g_sofuu_gqa_max_tg[gi] = 0u;
                if (k_sofuu_gqa[gi].opt_in && !ctx->attn_gqa_all) {
                    ctx->pso_attn_gqa[gi] = nil;
                    continue;
                }
                id<MTLComputePipelineState> p = load_pso(@(k_sofuu_gqa[gi].name));
                if (p != nil) {
                    // Record the ceiling even when it rejects the shape: the
                    // dispatcher's "why" line quotes this number instead of
                    // leaving a bare "disabled" with no cause attached.
                    ctx->attn_gqa_max_tg[gi] = (uint)p.maxTotalThreadsPerThreadgroup;
                    g_sofuu_gqa_max_tg[gi] = ctx->attn_gqa_max_tg[gi];
                }
                if (p != nil && p.maxTotalThreadsPerThreadgroup < (NSUInteger)k_sofuu_gqa[gi].threads) {
                    fprintf(stderr,
                            "[metal] %s needs %u threads/group, pipeline allows %lu - GQA shape disabled\n",
                            k_sofuu_gqa[gi].name, k_sofuu_gqa[gi].threads,
                            (unsigned long)p.maxTotalThreadsPerThreadgroup);
                    p = nil;
                }
                ctx->pso_attn_gqa[gi] = p;
                if (p != nil && (only_idx < 0 || only_idx == gi)) {
                    ctx->attn_gqa_mask |= (1u << gi);
                }
            }
        }
        // Split-K (flash-decoding) chunk attention: the shipped r2/nr2/rg32
        // threadgroup with the key range cut into grid z slices, and the
        // one-warp combine that reduces the partials. Same silent-drop rule as
        // the rest of this file: a pipeline that cannot host its threadgroup
        // is nil'd here rather than dispatched into the void.
        ctx->pso_attn_gqa_sk     = load_pso(@"attn_gqa_sk_r2_nr2_rg32");
        ctx->pso_attn_sk_combine = load_pso(@"llama_attn_sk_combine");
        if (ctx->pso_attn_gqa_sk != nil &&
            ctx->pso_attn_gqa_sk.maxTotalThreadsPerThreadgroup < 512ul) {
            fprintf(stderr,
                    "[metal] attn_gqa_sk_r2_nr2_rg32 needs 512 threads/group, pipeline allows %lu - split-K disabled\n",
                    (unsigned long)ctx->pso_attn_gqa_sk.maxTotalThreadsPerThreadgroup);
            ctx->pso_attn_gqa_sk = nil;
        }
        if (ctx->pso_attn_sk_combine != nil &&
            ctx->pso_attn_sk_combine.maxTotalThreadsPerThreadgroup < 32ul) {
            ctx->pso_attn_sk_combine = nil;
        }
        if (ctx->pso_attn_gqa_sk == nil || ctx->pso_attn_sk_combine == nil) {
            ctx->pso_attn_gqa_sk = nil;
            ctx->pso_attn_sk_combine = nil;
        }
        // SOFUU_ATTN_GQA selects the shape: unset = auto (the shape table
        // picks the fastest LIVE form for the model's ratio), 0 = the per-head
        // llama_attn_pre_row, 1 = r2/nr1/rg32 (pairing only), 2 = r2/nr2/rg32,
        // 3 = r2/nr2/rg16, 4 = r2/nr4/rg32, 5 = r2/nr1/rg32 with the prefetched
        // K/V. A PIN is a request, not an override: a shape whose pipeline is
        // nil falls through to the fastest live sibling of the same ratio (and
        // says so on stderr), and only reaches the per-head kernel when the
        // ratio has no usable form at all.
        ctx->attn_gqa_mode = -1;   // -1 = auto
        ctx->attn_gqa_pin = 0;
        if (const char *gqa_env = getenv("SOFUU_ATTN_GQA")) {
            int m = atoi(gqa_env);
            if (m >= 0 && m <= 5) {
                ctx->attn_gqa_mode = m;
                ctx->attn_gqa_pin = (m >= 1);
            }
        }
        ctx->pso_llama_attn_copy_b4  = load_pso(@"llama_attn_copy_b4");
        ctx->pso_llama_attn_1c_b4    = load_pso(@"llama_attn_1c_b4");
        ctx->pso_gemv_q4_k_b4c2r     = load_pso(@"gemv_q4_k_b4c2r");
        // class 1 = byte-lane coalesced; class 2 = 16B-span vec4 coalesced
        // (the faster class: b4c2 100 GB/s vs 69, head q6_K 90.5 vs the
        // v1 q6_K 15.5). BOTH the single-row (serial client) and batched
        // (spec chain) q4_K PSOs flip together so every GEMV in a run
        // shares one summation order.
        if (coalesced_level == 2) {
            // level 2 = BEST PER SHAPE (2x2 matrix receipt: byte-lane b4c
            // 107/50 GB/s compact/5G vs vec4 46/37 — but the vec4 WINS the
            // single-row shape as the 12.4 t/s serial receipt showed):
            // batched pass GEMVs = byte-lane b4c, single-row = vec4 c2.
            ctx->pso_gemv_q4_k_c2r8 = load_pso(@"gemv_q4_k_c2r8");
            ctx->pso_llama_attn_w4 = load_pso(@"llama_attn_w4");
            ctx->pso_llama_attn_w8 = load_pso(@"llama_attn_w8");
            ctx->pso_llama_attn_w16 = load_pso(@"llama_attn_w16");
            ctx->pso_llama_attn_w32 = load_pso(@"llama_attn_w32");
            ctx->pso_llama_attn_fast = load_pso(@"llama_attn_fast");
            ctx->pso_gemv_q4_k_flat = load_pso(@"gemv_q4_k_flat");
            ctx->pso_add_rmsnorm_f32w = load_pso(@"add_rmsnorm_f32w");
            ctx->pso_gemv_q4_k_split1 = load_pso(@"gemv_q4_k_split1");
            ctx->pso_gemv_q4_k_split2 = load_pso(@"gemv_q4_k_split2");
            ctx->pso_gemv_q4_k_secs = load_pso(@"gemv_q4_k_secs");
            ctx->pso_bw_probe_q4k = load_pso(@"bw_probe_q4k");
            ctx->pso_gemv_q4_k_v4 = load_pso(@"gemv_q4_k_v4");
            ctx->pso_gemv_q4_k_v4_live = ctx->pso_gemv_q4_k_v4 != nil;
            ctx->pso_gemv_q4_k_v3 = load_pso(@"gemv_q4_k_v3");
            ctx->pso_gemv_q4_k_v3_live = ctx->pso_gemv_q4_k_v3 != nil;
            ctx->pso_gemv_q4_k_v2 = load_pso(@"gemv_q4_k_v2");
            ctx->pso_gemv_q4_k_v2_live = ctx->pso_gemv_q4_k_v2 != nil;
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k_c2");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4c");
        } else if (coalesced_level == 1) {
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k_c");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4c");
        }
        if (coalesced_level > 0) {
            // the q6_K class rides the same env: the ~514MB lm_head runs on
            // the scalar q6_K at 15.6 GB/s (32.9ms) vs 92 GB/s (5.5ms)
            // coalesced
            ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k_c");
            if (ctx->pso_gemv_q6_k == nil) {
                // bundled metallib predates it: degrade to the shipped
                // q6_K class rather than a silent no-op GEMV
                ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
            }
        }
        if (ctx->pso_gemv_q4_k == nil || ctx->pso_gemv_q4_k_b4 == nil) {
            // coalesced class unavailable (e.g. a complete bundled metallib
            // predating it): degrade to the shipped class, never a silent
            // no-op GEMV
            ctx->pso_gemv_q4_k = load_pso(@"gemv_q4_k");
            ctx->pso_gemv_q4_k_b4 = load_pso(@"gemv_q4_k_b4");
            if (coalesced_level > 0 && ctx->pso_gemv_q6_k == nil) {
                ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
            }
        }
        if (ctx->pso_gemv_q6_k == nil) {
            ctx->pso_gemv_q6_k = load_pso(@"gemv_q6_k");
        }
        // prefill GEMM class (P1 of llama.cpp parity): if the bundled
        // metallib predates the kernel the live flag stays false and the
        // GEMM prefill route never engages — visible unavailable, never a
        // silent no-op GEMM
        ctx->pso_gemm_q4_k = load_pso(@"gemm_q4_k");
        ctx->pso_gemm_q4_k_live = ctx->pso_gemm_q4_k != nil;
        ctx->pso_gemm_q4_k_multi = load_pso(@"gemm_q4_k_multi");
        ctx->pso_gemm_q4_k_multi_live = ctx->pso_gemm_q4_k_multi != nil;
        ctx->pso_gemm_q4_k_acc = load_pso(@"gemm_q4_k_acc");
        ctx->pso_gemm_q4_k_acc_live = ctx->pso_gemm_q4_k_acc != nil;
        ctx->pso_gemm_q8_0 = load_pso(@"gemm_q8_0");
        ctx->pso_gemm_q8_0_live = ctx->pso_gemm_q8_0 != nil;
        ctx->pso_gemm_q8_0_multi = load_pso(@"gemm_q8_0_multi");
        ctx->pso_gemm_q8_0_multi_live = ctx->pso_gemm_q8_0_multi != nil;
        ctx->pso_gemm_q8_0_acc = load_pso(@"gemm_q8_0_acc");
        ctx->pso_gemm_q8_0_acc_live = ctx->pso_gemm_q8_0_acc != nil;
        ctx->pso_gemm_q6_k = load_pso(@"gemm_q6_k");
        ctx->pso_gemm_q6_k_live = ctx->pso_gemm_q6_k != nil;
        ctx->pso_gemm_q6_k_multi = load_pso(@"gemm_q6_k_multi");
        ctx->pso_gemm_q6_k_multi_live = ctx->pso_gemm_q6_k_multi != nil;
        ctx->pso_gemm_q6_k_acc = load_pso(@"gemm_q6_k_acc");
        ctx->pso_gemm_q6_k_acc_live = ctx->pso_gemm_q6_k_acc != nil;
        ctx->pso_gemm_q8_0_t32 = load_pso(@"gemm_q8_0_t32");
        ctx->pso_gemm_q8_0_t32_live = ctx->pso_gemm_q8_0_t32 != nil;
        ctx->pso_gemm_q6_k_t32 = load_pso(@"gemm_q6_k_t32");
        ctx->pso_gemm_q6_k_t32_live = ctx->pso_gemm_q6_k_t32 != nil;
        ctx->pso_quantize_q8_k = load_pso(@"quantize_q8_k");
        ctx->pso_quantize_q8_k_live = ctx->pso_quantize_q8_k != nil;
        ctx->pso_peak_fma16 = load_pso(@"peak_fma16");
        ctx->pso_peak_fma = load_pso(@"peak_fma");
        ctx->pso_peak_mma = load_pso(@"peak_mma");
        ctx->pso_gemm_q8_0_mma64 = load_pso(@"gemm_q8_0_mma64");
        ctx->pso_gemm_q6_k_mma64 = load_pso(@"gemm_q6_k_mma64");
        ctx->pso_gemm_q4_k_mma64 = load_pso(@"gemm_q4_k_mma64");
        // Same maxTotalThreadsPerThreadgroup discipline as gemm_q4_k_mma8
        // above: the 64x64 matrix-pipe GEMMs dispatch 128 threads (4
        // simdgroups) per threadgroup, and an oversized threadgroup is DROPPED
        // silently in release builds - the y slab would read back as stale
        // scratch instead of failing. Nil the PSO out so the typed entry
        // points (the chain's q6_K attn_v/ffn_down batching) return 0 and the
        // caller keeps its per-row fallback.
        if (ctx->pso_gemm_q6_k_mma64 != nil &&
            ctx->pso_gemm_q6_k_mma64.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q6_k_mma64 needs 128 threads/group, pipeline allows %lu - q6_K mma64 disabled\n",
                    (unsigned long)ctx->pso_gemm_q6_k_mma64.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q6_k_mma64 = nil;
        }
        if (ctx->pso_gemm_q8_0_mma64 != nil &&
            ctx->pso_gemm_q8_0_mma64.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q8_0_mma64 needs 128 threads/group, pipeline allows %lu - q8_0 mma64 disabled\n",
                    (unsigned long)ctx->pso_gemm_q8_0_mma64.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q8_0_mma64 = nil;
        }
        ctx->pso_gemm_q8_0_mma64_live = ctx->pso_gemm_q8_0_mma64 != nil;
        ctx->pso_gemm_q6_k_mma64_live = ctx->pso_gemm_q6_k_mma64 != nil;
        ctx->pso_gemm_q4_k_mma64_live = ctx->pso_gemm_q4_k_mma64 != nil;
        ctx->pso_gemm_q4_k_tiled_r88 = load_pso(@"gemm_q4_k_tiled_r88");
        ctx->pso_gemm_q4_k_tiled_r88_live = ctx->pso_gemm_q4_k_tiled_r88 != nil;
        ctx->pso_gemm_q4_k_tiled_wide_fused = load_pso(@"gemm_q4_k_tiled_wide_fused");
        ctx->pso_gemm_q4_k_tiled_wide_fused_live = ctx->pso_gemm_q4_k_tiled_wide_fused != nil;
        ctx->pso_gemm_q4_k_tiled_wide = load_pso(@"gemm_q4_k_tiled_wide");
        ctx->pso_gemm_q4_k_tiled_wide_live = ctx->pso_gemm_q4_k_tiled_wide != nil;
        ctx->pso_gemm_q4_k_mma = load_pso(@"gemm_q4_k_mma");
        ctx->pso_gemm_q4_k_mma_live = ctx->pso_gemm_q4_k_mma != nil;
        ctx->pso_attn_flash_b4 = load_pso(@"llama_attn_flash_b4");
        ctx->pso_attn_flash_b4_live = ctx->pso_attn_flash_b4 != nil;
        ctx->pso_gemm_q4_k_tiled = load_pso(@"gemm_q4_k_tiled");
        ctx->pso_gemm_q4_k_tiled_live = ctx->pso_gemm_q4_k_tiled != nil;
        ctx->pso_gemm_q4_k_tiled_fused = load_pso(@"gemm_q4_k_tiled_fused");
        ctx->pso_gemm_q4_k_tiled_fused_live = ctx->pso_gemm_q4_k_tiled_fused != nil;
        ctx->pso_gemm_q6_k_tiled_fused = load_pso(@"gemm_q6_k_tiled_fused");
        // Same 128-thread discipline as the mma64 PSOs above: the 32x64 tile
        // dispatches 128 threads and an oversized threadgroup is dropped
        // silently, leaving y stale. Nil it so the typed entry points return 0
        // and the chain keeps its per-row q6_K fallback.
        if (ctx->pso_gemm_q6_k_tiled_fused != nil &&
            ctx->pso_gemm_q6_k_tiled_fused.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q6_k_tiled_fused needs 128 threads/group, pipeline allows %lu - q6_K tiled disabled\n",
                    (unsigned long)ctx->pso_gemm_q6_k_tiled_fused.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q6_k_tiled_fused = nil;
        }
        ctx->pso_gemm_q6_k_tiled_fused_live = ctx->pso_gemm_q6_k_tiled_fused != nil;
        ctx->pso_gemm_q8_0_tiled_fused = load_pso(@"gemm_q8_0_tiled_fused");
        ctx->pso_gemm_q8_0_tiled_fused_live = ctx->pso_gemm_q8_0_tiled_fused != nil;
        ctx->pso_gemm_q4_0_tiled_fused = load_pso(@"gemm_q4_0_tiled_fused");
        // The 32x64 tile dispatches 128 threads. Metal DROPS an over-wide
        // threadgroup silently, which would leave y stale, so a pipeline that
        // cannot take 128 threads is nilled here and the typed entry points
        // return 0 (the chain then keeps its per-row q4_0 GEMV fallback).
        if (ctx->pso_gemm_q4_0_tiled_fused != nil &&
            ctx->pso_gemm_q4_0_tiled_fused.maxTotalThreadsPerThreadgroup < 128ul) {
            fprintf(stderr,
                    "[metal] gemm_q4_0_tiled_fused needs 128 threads/group, pipeline allows %lu - q4_0 tiled disabled\n",
                    (unsigned long)ctx->pso_gemm_q4_0_tiled_fused.maxTotalThreadsPerThreadgroup);
            ctx->pso_gemm_q4_0_tiled_fused = nil;
        }
        ctx->pso_gemm_q4_0_tiled_fused_live = ctx->pso_gemm_q4_0_tiled_fused != nil;
        ctx->pso_gemm_q4_k_mmq = load_pso(@"gemm_q4_k_mmq");
        ctx->pso_gemm_q4_k_mmq_live = ctx->pso_gemm_q4_k_mmq != nil;
        ctx->pso_gemm_q6_k_mmq = load_pso(@"gemm_q6_k_mmq");
        ctx->pso_gemm_q6_k_mmq_live = ctx->pso_gemm_q6_k_mmq != nil;
        ctx->pso_quantize_q8_k_pos = load_pso(@"quantize_q8_k_pos");
        ctx->pso_quantize_q8_k_pos_live = ctx->pso_quantize_q8_k_pos != nil;
        ctx->pso_quantize_q8_ext = load_pso(@"quantize_q8_ext");
        ctx->pso_quantize_q8_ext_live = ctx->pso_quantize_q8_ext != nil;
        ctx->pso_gemv_q4_k_mmq32 = load_pso(@"gemv_q4_k_mmq32");
        ctx->pso_gemv_q4_k_mmq32_live = ctx->pso_gemv_q4_k_mmq32 != nil;
        ctx->pso_gemv_q6_k_mmq32 = load_pso(@"gemv_q6_k_mmq32");
        ctx->pso_gemv_q6_k_mmq32_live = ctx->pso_gemv_q6_k_mmq32 != nil;
        const char* gemm_tile_env = std::getenv("SOFUU_GEMM_TILE");
        ctx->gemm_tile32 = gemm_tile_env && strcmp(gemm_tile_env, "32") == 0;
        ctx->pso_gpu_q4_k_b4multi = load_pso(@"gemv_q4_k_b4multi");
        ctx->pso_gpu_q4_k_b4multi_live = ctx->pso_gpu_q4_k_b4multi != nil;
        ctx->pso_gpu_q6_k_b4multi = load_pso(@"gemv_q6_k_b4multi");
        ctx->pso_gpu_q6_k_b4multi_live = ctx->pso_gpu_q6_k_b4multi != nil;
        ctx->pso_gpu_q8_0_b4multi = load_pso(@"gemv_q8_0_b4multi");
        // P2: q8_0 coalesced single-row class — flips the SERIAL decode PSO
        // the same way q4_K/q6_K ride SOFUU_Q4K_COALESCED (nil-degrade to v1)
        id<MTLComputePipelineState> q8c2 = load_pso(@"gemv_q8_0_c2");
        // Per-type coalesced flip (phase-4 deciding A/B, calm 1.4-3.5):
        // coalesced=2 wins on q4/q6 models (0.6B 61.4 vs 31.8, MiniCPM 22.0
        // vs 9.1, 8B 5.6 vs 2.8) but REGRESSES the all-q8_0 1B (33.2 vs
        // 50.3 serial) — so the q8_0 serial GEMV keeps the shipped class
        // unless the caller asks explicitly: coalesced=3 = "q8 c2 too".
        if (q8c2 != nil && coalesced_level >= 3) {
            ctx->pso_gemv_q8_0 = q8c2;
        }
        ctx->pso_gpu_q8_0_b4multi_live = ctx->pso_gpu_q8_0_b4multi != nil;
        ctx->pso_gpu_q8_0_b4r4 = load_pso(@"gemv_q8_0_b4r4");
        ctx->pso_gpu_q8_0_b4r4_live = ctx->pso_gpu_q8_0_b4r4 != nil;

        // Initial 64KB activation buffers
        ctx->x_buf_cap = 65536 * sizeof(float);
        ctx->y_buf_cap = 65536 * sizeof(float);
        ctx->x_buf = [dev newBufferWithLength:ctx->x_buf_cap options:MTLResourceStorageModeShared];
        ctx->y_buf = [dev newBufferWithLength:ctx->y_buf_cap options:MTLResourceStorageModeShared];
        ctx->pending_cmd = nil;
        ctx->batch_enc = nil;
        ctx->pending_y_bytes = 0;
        ctx->pending_y_dst = nullptr;

        return (void*)ctx;
    }
}

void sofuu_metal_sync(void* ctx_ptr);
// W1 row split: drains the committed-but-unwaited buffers (defined below).
int sofuu_metal_batch_wait_all(void* ctx_ptr);

void sofuu_metal_free(void* ctx_ptr) {
    if (!ctx_ptr) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    delete ctx;
}

void* sofuu_metal_buffer_create_zerocopy(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
#if TARGET_OS_SIMULATOR
    /* A no-copy buffer here exports the memory over XPC shared memory, and
     * libxpc aborts the process (_xpc_api_misuse) instead of failing when
     * the range cannot be exported as one mapping — which is the case for
     * the allocator memory callers pass in. See the measured note in
     * sofuu_metal_buffer_create. Returning NULL sends the caller there,
     * where the driver's copying path runs instead. */
    return NULL;
#else
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        if (page_size == 0) page_size = 16384;
        if (((uintptr_t)ptr % page_size) != 0) return NULL;
        size_t aligned_len = ((size_bytes + page_size - 1) / page_size) * page_size;
        // Every decode op reads and writes the same activation buffer, so with the
        // default (tracked) hazard mode the driver serialises the batch on that
        // buffer no matter what barriers we issue - which is why removing our
        // explicit barriers never changed a timing. SOFUU_HAZARD=untracked takes
        // the driver out of the dependency chain and leaves our barriers as the
        // only ordering, which is what the batch model assumes.
        MTLResourceOptions opts = MTLResourceStorageModeShared;
        if (const char* nb = std::getenv("SOFUU_NO_BARRIER")) { ctx->no_barrier = (nb[0] == '1'); }
        if (const char* an = std::getenv("SOFUU_ATTN_NOOP")) { ctx->attn_noop = atoi(an); }
        if (std::getenv("SOFUU_HAZARD") && std::strcmp(std::getenv("SOFUU_HAZARD"), "untracked") == 0) {
            opts |= MTLHazardTrackingModeUntracked;
        }
        id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                           length:aligned_len
                                                          options:opts
                                                      deallocator:nil];
        return (__bridge_retained void*)buf;
    }
#endif
}

void* sofuu_metal_buffer_create(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
#if TARGET_OS_SIMULATOR
        /* MTLSimDriver exports every buffer's storage over XPC shared memory.
         * Its no-copy path hands our pointer straight to xpc_shmem_create,
         * and libxpc TRAPS (_xpc_api_misuse — it aborts the process rather
         * than returning an error) unless [ptr, ptr+length) lies inside one
         * exportable mapping. Memory from the system allocator does not
         * qualify: measured in a simulator probe, a 2 MiB malloc returns a
         * page-aligned pointer whose enclosing vm region reports as a single
         * 16 KiB page, and the export aborts. Every Rust Vec the engine
         * copies from is such an allocation, so the whole no-copy path is
         * unusable here.
         *
         * Its copying path is safe because the driver mmaps its own
         * MAP_SHARED|MAP_ANON region for the storage and exports that
         * instead — verified up to 32 MiB, including lengths that are not
         * page multiples. The simulator therefore gets a copied buffer: GPU
         * math still runs, and the only thing lost is the zero-copy
         * shortcut MTLSimDriver cannot honour for allocator memory.
         * (Reached from sofuu_metal_buffer_create_zerocopy too — it returns
         * NULL under TARGET_OS_SIMULATOR and the Rust wrapper falls back
         * here.) */
        return (__bridge_retained void*)[ctx->device newBufferWithBytes:ptr
                                                                 length:size_bytes
                                                                options:MTLResourceStorageModeShared];
#else
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        if (page_size == 0) page_size = 16384;
        if (((uintptr_t)ptr % page_size) == 0) {
            size_t aligned_len = ((size_bytes + page_size - 1) / page_size) * page_size;
            id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                               length:aligned_len
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
            if (buf != nil) {
                return (__bridge_retained void*)buf;
            }
        }
        id<MTLBuffer> buf = [ctx->device newBufferWithBytes:ptr
                                                     length:size_bytes
                                                    options:MTLResourceStorageModeShared];
        return (__bridge_retained void*)buf;
#endif
    }
}


/* Driver-owned buffer of `size_bytes`, contents undefined. The fp16 weight
 * arena (SOFUU_GPU_F16W) needs ~0.8 GB that no host slice exists for: the
 * weights are dequantised by the GPU straight out of the mapped file, so a
 * copying buffer built from a caller slice would double the peak RSS for no
 * reason. Shared storage keeps the host able to inspect the result (the
 * bit-identity receipt) and is what every other buffer in the engine uses.
 * Returns NULL on allocation failure, which the caller treats as "fall back to
 * the q4_K path" — a 0.9 GB request on a full device is an expected miss. */
extern "C" void* sofuu_metal_buffer_alloc(void* ctx_ptr, size_t size_bytes) {
    if (!ctx_ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        if (page_size == 0) page_size = 16384;
        size_t aligned_len = ((size_bytes + page_size - 1) / page_size) * page_size;
        id<MTLBuffer> buf = [ctx->device newBufferWithLength:aligned_len
                                                    options:MTLResourceStorageModeShared];
        if (buf == nil) return NULL;
        return (__bridge_retained void*)buf;
    }
}

void* sofuu_metal_buffer_create_copy(void* ctx_ptr, const void* ptr, size_t size_bytes) {
    if (!ctx_ptr || !ptr || size_bytes == 0) return NULL;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    @autoreleasepool {
        // Always a driver-managed copy (never wraps caller memory): the GPU
        // decode clients use this to test whether device-owned placement
        // streams weights faster than a wrapped host mmap.
        id<MTLBuffer> buf = [ctx->device newBufferWithBytes:ptr
                                                     length:size_bytes
                                                    options:MTLResourceStorageModeShared];
        return (__bridge_retained void*)buf;
    }
}

void sofuu_metal_buffer_free(void* buf_ptr) {
    if (!buf_ptr) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_ptr;
        (void)buf;
    }
}

/* waitUntilCompleted returns whether or not the GPU accepted the work:
 * an unsupported instruction, an out-of-range access or a failed allocation
 * leaves status == Error and the output buffer untouched. Without this check
 * the callers read that untouched buffer back and return zeros as if they
 * were results — silent wrong answers instead of a loud failure. (Measured:
 * a bf16 kernel on the iOS simulator fails this way with
 * MTLCommandBufferErrorInternal, while f16/f32 kernels complete.) */
static inline bool wait_and_report(id<MTLCommandBuffer> cmd, const char* kernel) {
    [cmd waitUntilCompleted];
    // LEDGER #154 probe: count waits per process (SOFUU_COMMIT_TRACE=1).
    { static long g_waits = 0; g_waits++; if (getenv("SOFUU_COMMIT_TRACE") && (g_waits % 250 == 0)) fprintf(stderr, "[commit] waits=%ld kern=%s\n", g_waits, kernel); }
    if (cmd.status == MTLCommandBufferStatusError) {
        NSLog(@"Sofuu Metal: %s command buffer failed: %@", kernel, cmd.error);
        return false;
    }
    // SOFUU_GPU_TIMING=1: device-side execution time of every command
    // buffer (GPUStartTime/GPUEndTime) — the GPU-busy vs CPU-wall split
    // that attributes the spec-pass overhead without an Instruments trace
    static int timing = -1;
    if (timing < 0) { const char* t = getenv("SOFUU_GPU_TIMING"); timing = (t && atoi(t) != 0) ? 1 : 0; }
    g_last_gpu_us = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    if (timing) {
        fprintf(stderr, "SOFUU_CMD_TIMING kernel=%s gpu_us=%.0f\n",
                kernel, g_last_gpu_us);
    }
    return true;
}

static inline void ensure_buffers(MetalContext* ctx, size_t x_bytes, size_t y_bytes) {
    if (x_bytes > ctx->x_buf_cap) {
        ctx->x_buf_cap = x_bytes * 2;
        ctx->x_buf = [ctx->device newBufferWithLength:ctx->x_buf_cap options:MTLResourceStorageModeShared];
    }
    if (y_bytes > ctx->y_buf_cap) {
        ctx->y_buf_cap = y_bytes * 2;
        ctx->y_buf = [ctx->device newBufferWithLength:ctx->y_buf_cap options:MTLResourceStorageModeShared];
    }
}

void sofuu_metal_gemv_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_bf16];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "gemv_bf16");

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_qkv_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t q_offset,
    size_t k_offset,
    size_t v_offset,
    const float* x,
    float* q_out,
    float* k_out,
    float* v_out,
    size_t q_rows,
    size_t k_rows,
    size_t v_rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t q_bytes = q_rows * sizeof(float);
        size_t k_bytes = k_rows * sizeof(float);
        size_t v_bytes = v_rows * sizeof(float);

        size_t q_slot = (q_bytes + 255) & ~255ULL;
        size_t k_slot = (k_bytes + 255) & ~255ULL;
        size_t total_y_bytes = q_slot + k_slot + v_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        // 1. Q projection
        if (q_out && q_rows > 0) {
            size_t aligned_q = q_offset & ~255ULL;
            uint rem_q = (uint)(q_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_q atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_q length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(q_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 2. K projection
        if (k_out && k_rows > 0) {
            size_t aligned_k = k_offset & ~255ULL;
            uint rem_k = (uint)(k_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_k atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_k length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(k_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 3. V projection
        if (v_out && v_rows > 0) {
            size_t aligned_v = v_offset & ~255ULL;
            uint rem_v = (uint)(v_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_v atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:(q_slot + k_slot) atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_v length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(v_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, "gemv_qkv_bf16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (q_out && q_rows > 0) {
            std::memcpy(q_out, y_ptr, q_bytes);
        }
        if (k_out && k_rows > 0) {
            std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        }
        if (v_out && v_rows > 0) {
            std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
        }
    }
}

void sofuu_metal_gemv_gate_up_bf16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t gate_offset,
    size_t up_offset,
    const float* x,
    float* gate_out,
    float* up_out,
    size_t inter_rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || cols == 0 || inter_rows == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t row_bytes = inter_rows * sizeof(float);
        size_t slot = (row_bytes + 255) & ~255ULL;
        size_t total_y_bytes = slot + row_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        // 1. Gate projection
        if (gate_out) {
            size_t aligned_g = gate_offset & ~255ULL;
            uint rem_g = (uint)(gate_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_g atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_g length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        // 2. Up projection
        if (up_out) {
            size_t aligned_u = up_offset & ~255ULL;
            uint rem_u = (uint)(up_offset & 255ULL);
            [enc setComputePipelineState:ctx->pso_gemv_bf16];
            [enc setBuffer:w_buf offset:aligned_u atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_u length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, "gemv_gate_up_bf16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (gate_out) {
            std::memcpy(gate_out, y_ptr, row_bytes);
        }
        if (up_out) {
            std::memcpy(up_out, y_ptr + slot, row_bytes);
        }
    }
}

void sofuu_metal_gemv_f16(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_f16];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "gemv_f16");

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_gemv_f32(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_gemv_f32];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "gemv_f32");

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

void sofuu_metal_rmsnorm(
    void* ctx_ptr,
    const float* x,
    void* w_buf_ptr,
    size_t w_offset,
    float* y,
    size_t dim,
    float eps
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || dim == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;

    @autoreleasepool {
        size_t bytes = dim * sizeof(float);
        ensure_buffers(ctx, bytes, bytes);
        std::memcpy(ctx->x_buf.contents, x, bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_rmsnorm];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint d = (uint)dim;
        [enc setBytes:&d length:sizeof(uint) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:5];

        MTLSize threadgroups = MTLSizeMake(1, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "rmsnorm");

        std::memcpy(y, ctx->y_buf.contents, bytes);
    }
}

void sofuu_metal_swiglu(
    void* ctx_ptr,
    float* g,
    const float* u,
    size_t size
) {
    if (!ctx_ptr || !g || !u || size == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;

    @autoreleasepool {
        size_t bytes = size * sizeof(float);
        ensure_buffers(ctx, bytes, bytes);
        std::memcpy(ctx->x_buf.contents, g, bytes);
        std::memcpy(ctx->y_buf.contents, u, bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pso_swiglu];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:1];
        uint sw_n = (uint)size;
        [enc setBytes:&sw_n length:sizeof(uint) atIndex:2];

        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "swiglu");

        std::memcpy(g, ctx->x_buf.contents, bytes);
    }
}

void sofuu_metal_conv1d_3(
    void* ctx_ptr,
    void* conv_w_buf_ptr,
    size_t conv_w_offset,
    const float* state,
    float* out,
    size_t channels
) {
    if (!ctx_ptr || !conv_w_buf_ptr || !state || !out || channels == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> conv_w_buf = (__bridge id<MTLBuffer>)conv_w_buf_ptr;

    @autoreleasepool {
        size_t state_bytes = channels * 3 * sizeof(float);
        size_t out_bytes = channels * sizeof(float);
        ensure_buffers(ctx, state_bytes, out_bytes);

        std::memcpy(ctx->x_buf.contents, state, state_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = conv_w_offset & ~255ULL;
        uint w_rem = (uint)(conv_w_offset & 255ULL);

        [enc setComputePipelineState:ctx->pso_conv1d_3];
        [enc setBuffer:conv_w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:3];

        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroups = MTLSizeMake((channels + 255) / 256, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, "conv1d_3");

        std::memcpy(out, ctx->y_buf.contents, out_bytes);
    }
}

void sofuu_metal_gemv_mlx4(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    void* scales_buf_ptr,
    size_t scales_offset,
    void* biases_buf_ptr,
    size_t biases_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    size_t group_size,
    int is_bf16
) {
    if (!ctx_ptr || !w_buf_ptr || !scales_buf_ptr || !biases_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;
    id<MTLBuffer> scales_buf = (__bridge id<MTLBuffer>)scales_buf_ptr;
    id<MTLBuffer> biases_buf = (__bridge id<MTLBuffer>)biases_buf_ptr;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t w_aligned = w_offset & ~255ULL;
        size_t scales_aligned = scales_offset & ~255ULL;
        size_t biases_aligned = biases_offset & ~255ULL;
        uint32_t rems[4] = {
            (uint32_t)(w_offset & 255ULL),
            (uint32_t)(scales_offset & 255ULL),
            (uint32_t)(biases_offset & 255ULL),
            0
        };

        id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:w_aligned atIndex:0];
        [enc setBuffer:scales_buf offset:scales_aligned atIndex:1];
        [enc setBuffer:biases_buf offset:biases_aligned atIndex:2];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:3];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:4];
        uint c = (uint)cols;
        uint g = (uint)group_size;
        [enc setBytes:&c length:sizeof(uint) atIndex:5];
        [enc setBytes:&g length:sizeof(uint) atIndex:6];
        [enc setBytes:rems length:sizeof(rems) atIndex:7];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        wait_and_report(cmd, is_bf16 ? "gemv_mlx4_bf16" : "gemv_mlx4_f16");

        std::memcpy(y, ctx->y_buf.contents, y_bytes);
    }
}

static inline void dispatch_mlx4_gemv(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso,
    id<MTLBuffer> w_buf, size_t w_offset,
    id<MTLBuffer> scales_buf, size_t scales_offset,
    id<MTLBuffer> biases_buf, size_t biases_offset,
    id<MTLBuffer> x_buf,
    id<MTLBuffer> y_buf, size_t y_offset,
    size_t rows, size_t cols, size_t group_size
) {
    size_t w_aligned = w_offset & ~255ULL;
    size_t scales_aligned = scales_offset & ~255ULL;
    size_t biases_aligned = biases_offset & ~255ULL;
    uint32_t rems[4] = {
        (uint32_t)(w_offset & 255ULL),
        (uint32_t)(scales_offset & 255ULL),
        (uint32_t)(biases_offset & 255ULL),
        (uint32_t)rows
    };
    uint c = (uint)cols;
    uint g = (uint)group_size;

    [enc setComputePipelineState:pso];
    [enc setBuffer:w_buf offset:w_aligned atIndex:0];
    [enc setBuffer:scales_buf offset:scales_aligned atIndex:1];
    [enc setBuffer:biases_buf offset:biases_aligned atIndex:2];
    [enc setBuffer:x_buf offset:0 atIndex:3];
    [enc setBuffer:y_buf offset:y_offset atIndex:4];
    [enc setBytes:&c length:sizeof(uint) atIndex:5];
    [enc setBytes:&g length:sizeof(uint) atIndex:6];
    [enc setBytes:rems length:sizeof(rems) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_gemv_qkv_mlx4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset, size_t q_rows,
    void* q_scales_ptr, size_t q_scales_offset,
    void* q_biases_ptr, size_t q_biases_offset,
    void* k_buf_ptr, size_t k_offset, size_t k_rows,
    void* k_scales_ptr, size_t k_scales_offset,
    void* k_biases_ptr, size_t k_biases_offset,
    void* v_buf_ptr, size_t v_offset, size_t v_rows,
    void* v_scales_ptr, size_t v_scales_offset,
    void* v_biases_ptr, size_t v_biases_offset,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr || !q_buf_ptr || !k_buf_ptr || !v_buf_ptr || !q_scales_ptr || !k_scales_ptr || !v_scales_ptr ||
        !q_biases_ptr || !k_biases_ptr || !v_biases_ptr || !x || !q_out || !k_out || !v_out ||
        !q_rows || !k_rows || !v_rows || !cols) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;
    pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t q_bytes = q_rows * sizeof(float);
        size_t k_bytes = k_rows * sizeof(float);
        size_t v_bytes = v_rows * sizeof(float);
        size_t q_slot = (q_bytes + 255) & ~255ULL;
        size_t k_slot = (k_bytes + 255) & ~255ULL;
        size_t total_y_bytes = q_slot + k_slot + v_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);
        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)q_buf_ptr, q_offset,
            (__bridge id<MTLBuffer>)q_scales_ptr, q_scales_offset,
            (__bridge id<MTLBuffer>)q_biases_ptr, q_biases_offset,
            ctx->x_buf, ctx->y_buf, 0, q_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)k_buf_ptr, k_offset,
            (__bridge id<MTLBuffer>)k_scales_ptr, k_scales_offset,
            (__bridge id<MTLBuffer>)k_biases_ptr, k_biases_offset,
            ctx->x_buf, ctx->y_buf, q_slot, k_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)v_buf_ptr, v_offset,
            (__bridge id<MTLBuffer>)v_scales_ptr, v_scales_offset,
            (__bridge id<MTLBuffer>)v_biases_ptr, v_biases_offset,
            ctx->x_buf, ctx->y_buf, q_slot + k_slot, v_rows, cols, group_size);
        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, is_bf16 ? "gemv_qkv_mlx4_bf16" : "gemv_qkv_mlx4_f16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        std::memcpy(q_out, y_ptr, q_bytes);
        std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
    }
}

void sofuu_metal_gemv_gate_up_mlx4(
    void* ctx_ptr,
    void* gate_buf_ptr, size_t gate_offset, size_t gate_rows,
    void* gate_scales_ptr, size_t gate_scales_offset,
    void* gate_biases_ptr, size_t gate_biases_offset,
    void* up_buf_ptr, size_t up_offset, size_t up_rows,
    void* up_scales_ptr, size_t up_scales_offset,
    void* up_biases_ptr, size_t up_biases_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr || !gate_buf_ptr || !up_buf_ptr || !gate_scales_ptr || !up_scales_ptr ||
        !gate_biases_ptr || !up_biases_ptr || !x || !gate_out || !up_out ||
        !gate_rows || !up_rows || !cols) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t gate_bytes = gate_rows * sizeof(float);
        size_t up_bytes = up_rows * sizeof(float);
        size_t gate_slot = (gate_bytes + 255) & ~255ULL;
        size_t total_y_bytes = gate_slot + up_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);
        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)gate_buf_ptr, gate_offset,
            (__bridge id<MTLBuffer>)gate_scales_ptr, gate_scales_offset,
            (__bridge id<MTLBuffer>)gate_biases_ptr, gate_biases_offset,
            ctx->x_buf, ctx->y_buf, 0, gate_rows, cols, group_size);
        dispatch_mlx4_gemv(enc, pso,
            (__bridge id<MTLBuffer>)up_buf_ptr, up_offset,
            (__bridge id<MTLBuffer>)up_scales_ptr, up_scales_offset,
            (__bridge id<MTLBuffer>)up_biases_ptr, up_biases_offset,
            ctx->x_buf, ctx->y_buf, gate_slot, up_rows, cols, group_size);
        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, is_bf16 ? "gemv_gate_up_mlx4_bf16" : "gemv_gate_up_mlx4_f16");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        std::memcpy(gate_out, y_ptr, gate_bytes);
        std::memcpy(up_out, y_ptr + gate_slot, up_bytes);
    }
}

void sofuu_metal_gemv_qkv_multi(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset, size_t q_rows,
    void* k_buf_ptr, size_t k_offset, size_t k_rows,
    void* v_buf_ptr, size_t v_offset, size_t v_rows,
    const float* x,
    float* q_out, float* k_out, float* v_out,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || cols == 0 || !x) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> q_buf = q_buf_ptr ? (__bridge id<MTLBuffer>)q_buf_ptr : nil;
    id<MTLBuffer> k_buf = k_buf_ptr ? (__bridge id<MTLBuffer>)k_buf_ptr : q_buf;
    id<MTLBuffer> v_buf = v_buf_ptr ? (__bridge id<MTLBuffer>)v_buf_ptr : q_buf;
    if (!q_buf && !k_buf && !v_buf) return;

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t q_bytes = q_rows * sizeof(float);
        size_t k_bytes = k_rows * sizeof(float);
        size_t v_bytes = v_rows * sizeof(float);

        size_t q_slot = (q_bytes + 255) & ~255ULL;
        size_t k_slot = (k_bytes + 255) & ~255ULL;
        size_t total_y_bytes = q_slot + k_slot + v_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        if (q_out && q_rows > 0 && q_buf != nil) {
            size_t aligned_q = q_offset & ~255ULL;
            uint rem_q = (uint)(q_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:q_buf offset:aligned_q atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_q length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(q_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (k_out && k_rows > 0 && k_buf != nil) {
            size_t aligned_k = k_offset & ~255ULL;
            uint rem_k = (uint)(k_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:k_buf offset:aligned_k atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_k length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(k_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (v_out && v_rows > 0 && v_buf != nil) {
            size_t aligned_v = v_offset & ~255ULL;
            uint rem_v = (uint)(v_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:v_buf offset:aligned_v atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:q_slot + k_slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_v length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(v_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, "gemv_qkv_multi (f16/f32 by dtype)");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (q_out && q_rows > 0) std::memcpy(q_out, y_ptr, q_bytes);
        if (k_out && k_rows > 0) std::memcpy(k_out, y_ptr + q_slot, k_bytes);
        if (v_out && v_rows > 0) std::memcpy(v_out, y_ptr + q_slot + k_slot, v_bytes);
    }
}

void sofuu_metal_gemv_gate_up_multi(
    void* ctx_ptr,
    void* gate_buf_ptr, size_t gate_offset,
    void* up_buf_ptr, size_t up_offset,
    const float* x,
    float* gate_out, float* up_out,
    size_t inter_rows,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || cols == 0 || !x || inter_rows == 0) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLBuffer> g_buf = gate_buf_ptr ? (__bridge id<MTLBuffer>)gate_buf_ptr : nil;
    id<MTLBuffer> u_buf = up_buf_ptr ? (__bridge id<MTLBuffer>)up_buf_ptr : g_buf;
    if (!g_buf && !u_buf) return;

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t row_bytes = inter_rows * sizeof(float);
        size_t slot = (row_bytes + 255) & ~255ULL;
        size_t total_y_bytes = slot + row_bytes;
        ensure_buffers(ctx, x_bytes, total_y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        uint c = (uint)cols;
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);

        if (gate_out && g_buf != nil) {
            size_t aligned_g = gate_offset & ~255ULL;
            uint rem_g = (uint)(gate_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:g_buf offset:aligned_g atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_g length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        if (up_out && u_buf != nil) {
            size_t aligned_u = up_offset & ~255ULL;
            uint rem_u = (uint)(up_offset & 255ULL);
            [enc setComputePipelineState:pso];
            [enc setBuffer:u_buf offset:aligned_u atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->y_buf offset:slot atIndex:2];
            [enc setBytes:&c length:sizeof(uint) atIndex:3];
            [enc setBytes:&rem_u length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(inter_rows, 1, 1) threadsPerThreadgroup:threadsPerGroup];
        }

        [enc endEncoding];
        [cmd commit];
        wait_and_report(cmd, "gemv_gate_up_multi (f16/f32 by dtype)");

        const char* y_ptr = (const char*)ctx->y_buf.contents;
        if (gate_out) std::memcpy(gate_out, y_ptr, row_bytes);
        if (up_out) std::memcpy(up_out, y_ptr + slot, row_bytes);
    }
}

void sofuu_metal_sync(void* ctx_ptr) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    // W1: a previous split pass may have left committed-but-unwaited
    // buffers; sync means "everything the GPU was given is done".
    sofuu_metal_batch_wait_all(ctx_ptr);
    if (ctx->batch_enc != nil) {
        [ctx->batch_enc endEncoding];
        CFRelease((__bridge CFTypeRef)ctx->batch_enc);
        ctx->batch_enc = nil;
        if (ctx->pending_cmd != nil) [ctx->pending_cmd commit];
    }
    if (ctx->pending_cmd != nil) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = ctx->pending_cmd;
            wait_and_report(cmd, "async gemv (deferred)");
            if (ctx->pending_y_dst && ctx->pending_y_bytes > 0) {
                std::memcpy(ctx->pending_y_dst, ctx->y_buf.contents, ctx->pending_y_bytes);
            }
            ctx->pending_cmd = nil;
            ctx->pending_y_dst = nullptr;
            ctx->pending_y_bytes = 0;
            CFRelease((__bridge CFTypeRef)cmd);
        }
    }
}

void sofuu_metal_gemv_async(
    void* ctx_ptr,
    void* w_buf_ptr,
    size_t w_offset,
    const float* x,
    float* y,
    size_t rows,
    size_t cols,
    int dtype
) {
    if (!ctx_ptr || !w_buf_ptr || !x || !y || rows == 0 || cols == 0) return;
    sofuu_metal_sync(ctx_ptr);
    MetalContext* ctx = (MetalContext*)ctx_ptr;

    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_ptr;
    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    @autoreleasepool {
        size_t x_bytes = cols * sizeof(float);
        size_t y_bytes = rows * sizeof(float);
        ensure_buffers(ctx, x_bytes, y_bytes);

        std::memcpy(ctx->x_buf.contents, x, x_bytes);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t aligned_offset = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);

        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:aligned_offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        uint c = (uint)cols;
        [enc setBytes:&c length:sizeof(uint) atIndex:3];
        [enc setBytes:&w_rem length:sizeof(uint) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];

        [cmd commit];
        ctx->pending_cmd = (__bridge id<MTLCommandBuffer>)(void*)CFBridgingRetain(cmd);
        ctx->pending_y_dst = y;
        ctx->pending_y_bytes = y_bytes;
    }
}

// G0: first-op x staging — CPU writes to a no-copy slab between GPU
// submissions are not guaranteed visible, so the BATCH'S FIRST op takes x
// from CPU through the private internal buffer (memcpy at call time, reads
// are correct at the op's execution); chained ops use GPU-written buffers.
void sofuu_metal_batch_gemv_staged(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    const float* x,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, int dtype
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x || !y_buf_ptr || rows == 0 || cols == 0) return;
    size_t x_bytes = cols * sizeof(float);
    ensure_buffers(ctx, x_bytes, rows * sizeof(float));
    std::memcpy(ctx->x_buf.contents, x, x_bytes);

    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:0];
    [ctx->batch_enc setBuffer:ctx->x_buf offset:0 atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// P1: prefill GEMM op (q4_K first). x is [B x cols], y is [B x rows]
// (row-major), one commit rides the live batch like every other op.
// Returns 1 dispatched, 0 unavailable (PSO missing / bad args) — the Rust
// side gates the GPU prefill route on a probe, never on silent zeros.
// 8x4-tile MMQ mirror: act-quantize to per-256 TileQ8K blocks (stride 260B).

int sofuu_metal_batch_quantize_q8_k(
    void* ctx_ptr,
    void* x_buf, size_t x_offset,
    void* y_buf, size_t y_offset,
    size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf || !y_buf || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_quantize_q8_k_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_quantize_q8_k];
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:1];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, cols / 256, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

// q4_K gemm over TileQ8K activations, bit-mirroring ggml_gemm_q4_K_8x4_q8_K_p.
int sofuu_metal_batch_gemm_q4_k_mmq(
    void* ctx_ptr,
    void* w_buf, size_t w_offset,
    void* xq_buf, size_t xq_offset,
    void* y_buf, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf || !xq_buf || !y_buf || rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_mmq_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mmq];
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)xq_buf offset:xq_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, (NSUInteger)B, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

// Per-32 act-quant to Q8Ext blocks (stride 40B); nb_total = B*cols/32.
int sofuu_metal_batch_quantize_q8_k_pos(
    void* ctx_ptr, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset,
    size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf || !y_buf || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_quantize_q8_k_pos_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_quantize_q8_k_pos];
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:1];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, cols / 256, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

int sofuu_metal_batch_quantize_q8_ext(
    void* ctx_ptr, void* x_buf, size_t x_offset, void* y_buf, size_t y_offset, size_t nb_total
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf || !y_buf || nb_total == 0) return 0;
    if (!ctx->pso_quantize_q8_ext_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_quantize_q8_ext];
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:1];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nb_total, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

static int gemv_mmq32_dispatch(
    MetalContext* ctx, id<MTLComputePipelineState> pso,
    void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset,
    void* y_buf, size_t y_offset, size_t rows, size_t cols
) {
    if (ctx->batch_enc == nil || !w_buf || !xq_buf || !y_buf || rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    [ctx->batch_enc setComputePipelineState:pso];
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)xq_buf offset:xq_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

int sofuu_metal_batch_gemv_q4_k_mmq32(
    void* ctx_ptr, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset,
    void* y_buf, size_t y_offset, size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (!ctx->pso_gemv_q4_k_mmq32_live) return 0;
    return gemv_mmq32_dispatch(ctx, ctx->pso_gemv_q4_k_mmq32, w_buf, w_offset, xq_buf, xq_offset, y_buf, y_offset, rows, cols);
}

int sofuu_metal_batch_gemv_q6_k_mmq32(
    void* ctx_ptr, void* w_buf, size_t w_offset, void* xq_buf, size_t xq_offset,
    void* y_buf, size_t y_offset, size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (!ctx->pso_gemv_q6_k_mmq32_live) return 0;
    return gemv_mmq32_dispatch(ctx, ctx->pso_gemv_q6_k_mmq32, w_buf, w_offset, xq_buf, xq_offset, y_buf, y_offset, rows, cols);
}

// q6_K MMQ mirror over TileQ8K activations (bit-twin of ggml_gemm_q6_K_8x4_q8_K).
int sofuu_metal_batch_gemm_q6_k_mmq(
    void* ctx_ptr,
    void* w_buf, size_t w_offset,
    void* xq_buf, size_t xq_offset,
    void* y_buf, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf || !xq_buf || !y_buf || rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q6_k_mmq_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q6_k_mmq];
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)xq_buf offset:xq_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, (NSUInteger)B, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    return 1;
}

int sofuu_metal_batch_gguf_gemm(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || B == 0) return 0;
    if (!ctx->pso_gemm_q4_k_live) return 0;
    // Phase 3: the real GEMM class covers all three GEMV-eligible quant
    // types (8 = q8_0, 12 = q4_K, 14 = q6_K); unsupported ty falls back to
    // the caller's per-row loop, visible-unavailable, never a silent no-op.
    bool tile32 = false;
    switch (ty) {
        case 8:
            if (ctx->gemm_tile32 && ctx->pso_gemm_q8_0_t32_live) { tile32 = true; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q8_0_t32]; }
            else { if (!ctx->pso_gemm_q8_0_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q8_0]; }
            break;
        case 12: [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k]; break;
        case 14:
            if (ctx->gemm_tile32 && ctx->pso_gemm_q6_k_t32_live) { tile32 = true; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q6_k_t32]; }
            else { if (!ctx->pso_gemm_q6_k_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q6_k]; }
            break;
        default: return 0;
    }
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint subtiles = tile32 ? (Bn + 31) / 32 : (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P5 arm-b: flash-style batched prefill attention (same ABI as the 1c
// variant plus the row count; grid = (ceil(rows/4), n_heads), 128 threads).
int sofuu_metal_batch_llama_attn_flash_b4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_offset,
    void* vc_buf_ptr, size_t vc_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t has_table
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr || !pos_buf_ptr) return 0;
    if (rows == 0 || head_dim == 0 || head_dim > 256 || n_heads == 0 || n_kv_heads == 0 || max_seq == 0) return 0;
    if (n_heads % n_kv_heads != 0) return 0;
    if (!ctx->pso_attn_flash_b4_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_attn_flash_b4];
    uint hd = head_dim;
    uint nh = n_heads;
    uint nkv = n_kv_heads;
    uint ms = max_seq;
    uint ht = has_table;
    uint rn = rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
    [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:9];
    [ctx->batch_enc setBytes:&ht length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:11];
    uint r_tiles = (rows + 3) / 4;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, n_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// P4 arm-d: threadgroup-tiled q4_K GEMM (32-row x 64-token tile, weights and
// activations staged in threadgroup memory). Same contract as
// sofuu_metal_batch_gguf_gemm for ty=12; returns 0 when the pipeline is
// unavailable so the caller keeps the shipped class.
int sofuu_metal_batch_gguf_gemm_tiled(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_tiled_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_tiled];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint r_tiles = (uint)((rows + 31) / 32);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
int sofuu_metal_batch_gguf_gemm_tiled_r88(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_tiled_r88_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_tiled_r88];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint r_tiles = (uint)((rows + 127) / 128);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
int sofuu_metal_batch_gguf_gemm_mma64(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_mma64_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mma64];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint r_tiles = (uint)((rows + 63) / 64);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// arm-E: 64-row tiled variant (x staged once per 64 rows instead of 32).
int sofuu_metal_batch_gguf_gemm_tiled_wide(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_tiled_wide_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_tiled_wide];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint r_tiles = (uint)((rows + 63) / 64);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// P5 arm-c: simdgroup-matrix GEMM (fp16 MMA units). Same ABI as the plain
// tiled entry so the two classes can be compared on identical buffers.
int sofuu_metal_batch_gguf_gemm_mma(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemm_q4_k_mma_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mma];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    uint r_tiles = (uint)((rows + 31) / 32);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// P4 arm-d: tiled fused GEMM — same section-table ABI as
// sofuu_metal_batch_gguf_gemm_multi, but the 32x64 threadgroup tile streams
// each weight row once per 64 tokens instead of once per 8. res_stride = 0
// keeps the pure multi behaviour; the _acc entry synthesizes a one-section
// table with res_stride != 0 for the o/down resid epilogue.
// Section-fused MMA: the kernel is single-tensor, so sections are dispatched
// back to back inside the caller's commit. Sections are independent (they write
// disjoint y regions), and the fused benefit is x reuse in L2 plus one commit.
// x-staged decode GEMV: 8 rows per 256-thread threadgroup, x in threadgroup mem.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_v4(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || cols > 4096) return 0;
    if (!ctx->pso_gemv_q4_k_v4_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_v4];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    uint groups = (uint)((rows + 7) / 8);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// split-K decode GEMV: one row per 128-thread threadgroup, one warp per block.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_v3(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemv_q4_k_v3_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_v3];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// decode GEMV v2: 8 rows per 256-thread threadgroup, one warp per row.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_v2(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    if (!ctx->pso_gemv_q4_k_v2_live) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_v2];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    uint groups = (uint)((rows + GV2_ROWS_HOST - 1) / GV2_ROWS_HOST);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// Matrix-pipe GEMM that ADDS into the residual (the o/down projections).
// res_stride must equal rows: the kernel indexes res with the same (gt*rows+grow)
// mapping it writes y with, which is what the fused-acc route already requires.
// Per-type column granularity of the batched GEMM kernels. MUST match
// MetalEngine::gemm_cols_floor in crates/sofuu-core/src/infer/metal.rs:
// q8_0 walks 32-value blocks, q4_K/q6_K a 256-value super-block. Every one
// of those kernels truncates to whole blocks, so a shape below the floor is
// a silent partial dot product - refuse it here as well as on the Rust side
// (defence in depth: the FFI is callable directly).
static inline bool sf_gemm_cols_ok(uint32_t ty, size_t cols) {
    if (cols == 0) return false;
    switch (ty) {
        case 2:  return (cols % 32u) == 0;   // q4_0: 18-byte block, 32 values
        case 8:  return (cols % 32u) == 0;
        case 12: return (cols % 256u) == 0;
        case 14: return (cols % 256u) == 0;
        default: return false;
    }
}
int sofuu_metal_batch_gguf_gemm_mma64_acc(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* res_buf_ptr, size_t res_offset, size_t res_stride,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || !res_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || B < 16) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    if (res_stride != rows) return 0;
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 12: pso = ctx->pso_gemm_q4_k_mma64_live ? ctx->pso_gemm_q4_k_mma64 : nil; break;
        case 14: pso = ctx->pso_gemm_q6_k_mma64_live ? ctx->pso_gemm_q6_k_mma64 : nil; break;
        case 8:  pso = ctx->pso_gemm_q8_0_mma64_live ? ctx->pso_gemm_q8_0_mma64 : nil; break;
        default: return 0;
    }
    if (pso == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    size_t aligned = w_offset & ~255ULL;
    uint rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    uint one_acc = 1;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)res_buf_ptr offset:res_offset atIndex:7];
    [ctx->batch_enc setBytes:&one_acc length:sizeof(uint) atIndex:8];
    uint r_tiles = (uint)((rows + 63) / 64);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

int sofuu_metal_batch_gguf_gemm_mma64_multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows_total, size_t cols, size_t B,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows_total == 0 || cols == 0 || B == 0) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    if (!secs || n_sec == 0 || n_sec > 4) return 0;
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 12: pso = ctx->pso_gemm_q4_k_mma64_live ? ctx->pso_gemm_q4_k_mma64 : nil; break;
        case 14: pso = ctx->pso_gemm_q6_k_mma64_live ? ctx->pso_gemm_q6_k_mma64 : nil; break;
        case 8:  pso = ctx->pso_gemm_q8_0_mma64_live ? ctx->pso_gemm_q8_0_mma64 : nil; break;
        default: return 0;
    }
    if (pso == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint prev_end = 0;
    uint zero_acc = 0;
    for (size_t s = 0; s < n_sec; ++s) {
        const uint32_t* sec = secs + s * 5;
        uint32_t off = sec[0];
        uint32_t rem = sec[1];
        uint32_t rows_end = sec[2];
        uint32_t y_off = sec[3];
        // padded slots carry rows_end == 0; they must never be interpreted as
        // a span (rows_end < prev_end), the same rule the fused kernels use
        if (rows_end <= prev_end) continue;
        uint32_t rows_sec = rows_end - prev_end;
        prev_end = rows_end;
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr
                           offset:(w_base_offset + off) atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:(y_offset + y_off) atIndex:2];
        [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
        [ctx->batch_enc setBytes:&rem length:sizeof(uint) atIndex:4];
        [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
        [ctx->batch_enc setBytes:&rows_sec length:sizeof(uint) atIndex:6];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:7];
        [ctx->batch_enc setBytes:&zero_acc length:sizeof(uint) atIndex:8];
        uint r_tiles = (rows_sec + 63) / 64;
        uint t_tiles = (B + 63) / 64;
        [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    return 1;
}

int sofuu_metal_batch_gguf_gemm_tiled_multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows_total, size_t cols, size_t B,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows_total == 0 || cols == 0 || B == 0) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    if (!secs || n_sec == 0 || n_sec > 4) return 0;
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 12: pso = ctx->pso_gemm_q4_k_tiled_fused; break;
        case 14: pso = ctx->pso_gemm_q6_k_tiled_fused; break;
        case 8:  pso = ctx->pso_gemm_q8_0_tiled_fused; break;
        case 2:  pso = ctx->pso_gemm_q4_0_tiled_fused_live ? ctx->pso_gemm_q4_0_tiled_fused : nil; break;
        default: return 0;
    }
    if (pso == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows_total;
    uint rs = 0;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:3];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:8];
    uint r_tiles = (uint)((rows_total + 31) / 32);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}
extern "C" int sofuu_metal_batch_gguf_gemm_tiled_wide_multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows_total, size_t cols, size_t B,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows_total == 0 || cols == 0 || B == 0 || cols % 256 != 0) return 0;
    if (!secs || n_sec == 0 || n_sec > 4) return 0;
    // the wide tile exists for q4_K only (its win is the q4_K gate/up shape)
    if (ty != 12) return 0;
    id<MTLComputePipelineState> pso = ctx->pso_gemm_q4_k_tiled_wide_fused;
    if (pso == nil || !ctx->pso_gemm_q4_k_tiled_wide_fused_live) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows_total;
    uint rs = 0;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:3];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:8];
    uint r_tiles = (uint)((rows_total + 63) / 64);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

int sofuu_metal_batch_gguf_gemm_tiled_acc(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* res_buf_ptr, size_t res_offset, size_t res_stride,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || !res_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || B == 0 || res_stride == 0) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 12: pso = ctx->pso_gemm_q4_k_tiled_fused; break;
        case 14: pso = ctx->pso_gemm_q6_k_tiled_fused; break;
        case 8:  pso = ctx->pso_gemm_q8_0_tiled_fused; break;
        case 2:  pso = ctx->pso_gemm_q4_0_tiled_fused_live ? ctx->pso_gemm_q4_0_tiled_fused : nil; break;
        default: return 0;
    }
    if (pso == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    size_t aligned = w_offset & ~255ULL;
    uint rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows;
    uint rs = (uint)res_stride;
    // The caller's section offsets are already RELATIVE to the lowest section
    // base, so the buffer is bound at that base and the table holds 0. Storing
    // the absolute base in the table truncates it to uint32 and silently reads
    // the wrong weights for any tensor past 4 GiB (the 8B model is 5.02 GB).
    uint sblob[20] = { 0u, rem, (uint)rows, 0u, (uint)rows, 0u, 0u, 0u,
                       (uint)rows, 0u, 0u, 0u, (uint)rows, 0u, 0u, 0u,
                       (uint)rows, 0u, 0u, 0u };
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)res_buf_ptr offset:res_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:3];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:8];
    uint r_tiles = (uint)((rows + 31) / 32);
    uint t_tiles = (uint)((B + 63) / 64);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r_tiles, t_tiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// P1 op-diet: fused multi-section GEMM (qkv -> n_sec=3, gate_up -> n_sec=2).
// secs is n_sec (<=4, host-padded to 4) {off, rem, rows_end} entries where
// off is the 256-aligned w_buf byte offset and rows_end the prefix row sum.
// x is [B x cols], y is [B x rows_total]. Returns 1 dispatched.
int sofuu_metal_batch_gguf_gemm_multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows_total, size_t cols, size_t B,
    const uint32_t* secs /* [4][3] padded */, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || B == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gemm_q4_k_multi_live) return 0;
    switch (ty) {
        case 8:  if (!ctx->pso_gemm_q8_0_multi_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q8_0_multi]; break;
        case 12: [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_multi]; break;
        case 14: if (!ctx->pso_gemm_q6_k_multi_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q6_k_multi]; break;
        default: return 0;
    }
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rn = (uint)rows_total;
    uint ns = (uint)n_sec;
    uint sblob[20]; // 4 sections x {off, rem, rows_end, y_off, y_stride}
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    // sections carry u32 DELTAS from w_base_offset (the first section's
    // 256-aligned start) — absolute offsets overflow u32 on 5GB files
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2]; // sections carry absolute slab offsets
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:6];
    // NOTE n_sec is redundant with the padded blob but kept for clarity of
    // the trace; the kernel scans all 4 padded entries (empty spans never
    // match). Dispatch grid: (rows_total, subtiles).
    uint subtiles = (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P1 wave 2: resid-add epilogue GEMM (o/down absorb their residual add —
// one writer per element). y[t*rows + row] = dot(W[row], X[t]) +
// res[t*res_stride + row]; y and res usually alias the x slab (y bound at
// the x region offset, res at the same region start). Returns 1 dispatched.
int sofuu_metal_batch_gguf_gemm_acc(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* res_buf_ptr, size_t res_offset, size_t res_stride,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !res_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || B == 0) return 0;
    if (!ctx->pso_gemm_q4_k_acc_live) return 0;
    switch (ty) {
        case 8:  if (!ctx->pso_gemm_q8_0_acc_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q8_0_acc]; break;
        case 12: [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_acc]; break;
        case 14: if (!ctx->pso_gemm_q6_k_acc_live) return 0; [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q6_k_acc]; break;
        default: return 0;
    }
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint Bn = (uint)B;
    uint rs = (uint)res_stride;
    uint rn = (uint)rows;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)res_buf_ptr offset:res_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:3];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&Bn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:8];
    uint subtiles = (Bn + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, subtiles, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P2 decode diet: multi-section q4_K b4 GEMV (see kernels.metal). All four
// token rows computed per threadgroup; sections write their own y regions.
// (external "C": this op sits past the file's extern-c region)
extern "C" int sofuu_metal_batch_gguf_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs /* [4][5] padded */, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q4_k_b4multi_live) return 0;
    if (ty != 12) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q4_k_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// mma8 with the residual add in the epilogue: o and down project onto the
// residual width, so the separate batch_add (and its dispatch + its y-slot
// round trip) folds into the store. rows must equal acc_dim; the caller checks
// the shape and falls back to mma8 + batch_add otherwise.
int sofuu_metal_batch_gguf_mma8_acc(void* ctx_ptr, void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset, void* acc_buf_ptr, size_t acc_offset,
    size_t rows, size_t cols, size_t B) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !acc_buf_ptr) return 0;
    if (ctx->pso_gemm_q4_k_mma8_acc == nil) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || B == 0) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mma8_acc];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    uint bn = (uint)B;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:6];
    // 64x64 tile: one threadgroup per (64 rows, 64 tokens). Still 128 threads,
    // so the maxTotalThreadsPerThreadgroup >= 128 guard above is the right one.
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (B + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// Sectioned mma8: ONE dispatch for a group of q4_K projections that share x and
// cols and write disjoint y slots (q/k/v off the normed input, gate/up off the
// ffn-normed one). secs is [n][4]: {w_off - w_base (256-aligned), w_rem, rows,
// y_off in BYTES}; w_base is the smallest aligned weight offset of the group, so
// the per-section deltas stay 32-bit on a >4 GiB model. Geometry is chosen from
// the WIDEST section; narrower sections return early from their spare tiles.
extern "C" int sofuu_metal_batch_gguf_mma8_secs(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_base,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t cols, size_t B,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (ctx->pso_gemm_q4_k_mma8_secs == nil) return 0;
    if (cols == 0 || cols % 256 != 0 || B == 0) return 0;
    if (!secs || n_sec == 0 || n_sec > 4) return 0;
    uint32_t max_rows = 0;
    for (size_t s = 0; s < n_sec; ++s) {
        if (secs[s * 4 + 2] > max_rows) max_rows = secs[s * 4 + 2];
    }
    if (max_rows == 0) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mma8_secs];
    uint c = (uint)cols;
    uint bn = (uint)B;
    unsigned long long wb = (unsigned long long)w_base;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:secs length:4 * sizeof(uint32_t) * n_sec atIndex:4];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&wb length:sizeof(unsigned long long) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((max_rows + 63u) / 64u, (B + 63u) / 64u, n_sec)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

int sofuu_metal_batch_gguf_mma8(void* ctx_ptr, void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset, void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || B == 0) return 0;
    if (ctx->pso_gemm_q4_k_mma8 == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_q4_k_mma8];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    uint bn = (uint)B;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:6];
    // 64x64 tile (see gemm_q4_k_mma8): grid = (rows/64, tokens/64), 128 threads.
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (B + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// ── fp16-input chain GEMM forms (SOFUU_GPU_F16W=1) ───────────────────────
// Same tile, grid and 128-thread threadgroup as the q4_K forms above; the only
// difference is that buffer 0 is the materialised fp16 arena and the weight
// address is a plain half offset (no 256-byte remainder trick: arena slots are
// 2 KiB aligned). ntok < 16 never comes here — the diversion is gated on the
// token count so the decode GEMV class keeps its q4_K stream.
extern "C" int sofuu_metal_batch_gguf_f16_mma8(void* ctx_ptr, void* w16_ptr, size_t w16_offset,
    void* x_buf_ptr, size_t x_offset, void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t B) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w16_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (ctx->pso_gemm_f16_mma8 == nil) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || B == 0) return 0;
    if (cols > 0xFFFFFFFFull || rows > 0xFFFFFFFFull || B > 0xFFFFFFFFull) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_f16_mma8];
    uint c = (uint)cols;
    uint rn = (uint)rows;
    uint bn = (uint)B;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w16_ptr offset:w16_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (B + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

extern "C" int sofuu_metal_batch_gguf_f16_mma8_acc(
    void* ctx_ptr, void* w16_ptr, size_t w16_offset,
    void* x_buf_ptr, size_t x_offset, void* acc_buf_ptr, size_t acc_offset,
    size_t rows, size_t cols, size_t B) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w16_ptr || !x_buf_ptr || !acc_buf_ptr) return 0;
    if (ctx->pso_gemm_f16_mma8_acc == nil) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0 || B == 0) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_f16_mma8_acc];
    uint c = (uint)cols;
    uint rn = (uint)rows;
    uint bn = (uint)B;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w16_ptr offset:w16_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (B + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// secs: [n][4] = {arena byte offset, unused, rows, y_off bytes} — the fp16
// twin of the q4_K section table, so one dispatch still covers q/k/v and
// gate/up.
extern "C" int sofuu_metal_batch_gguf_f16_mma8_secs(
    void* ctx_ptr,
    void* w16_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t cols, size_t B,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w16_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    if (ctx->pso_gemm_f16_mma8_secs == nil) return 0;
    if (cols == 0 || cols % 256 != 0 || B == 0) return 0;
    if (!secs || n_sec == 0 || n_sec > 4) return 0;
    uint32_t max_rows = 0;
    for (size_t s = 0; s < n_sec; ++s) {
        if (secs[s * 4 + 2] > max_rows) max_rows = secs[s * 4 + 2];
    }
    if (max_rows == 0) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemm_f16_mma8_secs];
    uint c = (uint)cols;
    uint bn = (uint)B;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w16_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:secs length:4 * sizeof(uint32_t) * n_sec atIndex:4];
    [ctx->batch_enc setBytes:&bn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((max_rows + 63u) / 64u, (B + 63u) / 64u, n_sec)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// One-shot q4_K → fp16 for ONE tensor: source = the mapped weights (bound at
// w_offset & ~255 with the remainder passed as w_rem), destination = the
// arena slot at out_offset bytes. Encoded into the live batch; the caller
// commits once for the whole materialisation.
extern "C" int sofuu_metal_dequant_q4k_f16(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* out_buf_ptr, size_t out_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->pso_dequant_q4k_f16 == nil) return 0;
    if (!w_buf_ptr || !out_buf_ptr) return 0;
    if (rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    if (out_offset % 2048 != 0) return 0;
    if (ctx->batch_enc == nil) return 0;      // the caller owns the batch
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_dequant_q4k_f16];
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint rn = (uint)rows;
    uint oh = (uint)(out_offset / 2);
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:0 atIndex:1];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&oh length:sizeof(uint) atIndex:5];
    size_t total = rows * (cols / 16);
    size_t tgs = (total + 127) / 128;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// P1 completion: multi-section q6_K b4 GEMV (same table as the q4_K one).
extern "C" int sofuu_metal_batch_gguf6_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q6_k_b4multi_live) return 0;
    if (ty != 14) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q6_k_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P2: multi-section q8_0 b4 GEMV (1B fleet; MiniCPM's q8_0 tensors).
extern "C" int sofuu_metal_batch_gguf8_b4multi(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q8_0_b4multi_live) return 0;
    if (ty != 8) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q8_0_b4multi];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows_total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// P2 rate wave: q8_0 with 4 weight-rows per threadgroup (x traffic / 4).
extern "C" int sofuu_metal_batch_gguf8_b4r4(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_base_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t rows_total, size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows_total == 0 || cols == 0 || !secs || n_sec == 0 || n_sec > 4) return 0;
    if (!ctx->pso_gpu_q8_0_b4r4_live) return 0;
    if (ty != 8) return 0;
    if (!sf_gemm_cols_ok(ty, cols)) return 0;
    sofuu_batch_barrier(ctx);
    uint c = (uint)cols;
    uint rn = (uint)rows_total;
    uint sblob[20];
    for (int i = 0; i < 20; ++i) sblob[i] = secs[i];
    [ctx->batch_enc setComputePipelineState:ctx->pso_gpu_q8_0_b4r4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_base_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:sblob length:sizeof(sblob) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows_total + 3) / 4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// G1: batched mlx4 GEMV — bindings mirror sofuu_metal_gemv_mlx4 exactly
// (0=w 1=s 2=b 3=x 4=y, 5=cols 6=group 7=rems[4]) but with caller MTLBuffers
// for x/y so batch ops chain activations GPU-side.
void sofuu_metal_batch_mlx4(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* s_buf_ptr, size_t s_offset,
    void* b_buf_ptr, size_t b_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, size_t group_size, int is_bf16
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !s_buf_ptr || !b_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    id<MTLComputePipelineState> pso = (is_bf16 != 0) ? ctx->pso_gemv_mlx4_bf16 : ctx->pso_gemv_mlx4_f16;

    size_t w_aligned = w_offset & ~255ULL;
    size_t scales_aligned = s_offset & ~255ULL;
    size_t biases_aligned = b_offset & ~255ULL;
    uint32_t rems[4] = {
        (uint32_t)(w_offset & 255ULL),
        (uint32_t)(s_offset & 255ULL),
        (uint32_t)(b_offset & 255ULL),
        (uint32_t)rows
    };
    uint c = (uint)cols;
    uint g = (uint)group_size;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:w_aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)s_buf_ptr offset:scales_aligned atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)b_buf_ptr offset:biases_aligned atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:4];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&g length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:rems length:sizeof(rems) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── GPU1: in-batch glue ops (whole-decode-step batches) ────────────────────
// Same contract as batch_gemv: caller-owned MTLBuffers + byte offsets so ops
// chain slot-to-slot inside ONE encoder; a memory barrier before each op
// orders the previous dispatch's writes. The CPU touches the slab only
// before batch_open / after batch_end.

// Metal dispatchThreadgroups caps each grid dimension at 1024. Prefill
// chunks of 128 rows × n_heads(×n_chunks) exceed that in one dimension —
// the launch silently truncates (rows past the cap never ran: the
// chunk-b=128 text bisect). The split keeps flat threadgroup semantics;
// kernels linearize tg.y*1024+tg.x, bit-identical when the grid fits a
// single x row (every decode/spec route).
static MTLSize sf_grid2d(uint total_groups) {
    return MTLSizeMake(total_groups > 1024u ? 1024u : total_groups,
                       (total_groups + 1023u) / 1024u, 1);
}

int sofuu_metal_batch_add(
    void* ctx_ptr,
    void* acc_buf_ptr, size_t acc_offset,
    void* add_buf_ptr, size_t add_offset,
    size_t n
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !acc_buf_ptr || !add_buf_ptr || n == 0) return 0;
    sofuu_batch_barrier(ctx);
    uint un = (uint)n;
    [ctx->batch_enc setComputePipelineState:ctx->pso_add_inplace];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)add_buf_ptr offset:add_offset atIndex:1];
    [ctx->batch_enc setBytes:&un length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((uint)((n + 255) / 256))
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// Qwen2/2.5 attention bias: y[t][j] += bias[j] for t in 0..b over up to four
// sections. Every offset is an f32 ELEMENT index into the bound buffer. One
// dispatch for the layer's whole q/k/v bias; returns 0 (caller refuses the
// commit) when the pipeline is missing, so a missing kernel degrades to "no
// bias support" instead of a silently unbiased projection.
int sofuu_metal_batch_add_chan_bias(
    void* ctx_ptr,
    void* y_buf_ptr,
    void* bias_buf_ptr,
    const uint32_t* secs,
    size_t nsec,
    size_t b
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !y_buf_ptr || !bias_buf_ptr || secs == NULL) return 0;
    if (ctx->pso_add_chan_bias4 == nil || nsec == 0 || nsec > 4 || b == 0) return 0;
    size_t total = 0;
    for (size_t s = 0; s < nsec; ++s) total += (size_t)secs[s * 4 + 2];
    total *= b;
    if (total == 0) return 1;
    sofuu_batch_barrier(ctx);
    uint unsec = (uint)nsec;
    uint ub = (uint)b;
    [ctx->batch_enc setComputePipelineState:ctx->pso_add_chan_bias4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)bias_buf_ptr offset:0 atIndex:1];
    [ctx->batch_enc setBytes:secs length:sizeof(uint32_t) * 4 * nsec atIndex:2];
    [ctx->batch_enc setBytes:&unsec length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&ub length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((uint)((total + 255) / 256))
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int sofuu_metal_batch_swiglu(
    void* ctx_ptr,
    void* g_buf_ptr, size_t g_offset,
    void* u_buf_ptr, size_t u_offset,
    size_t n
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !g_buf_ptr || !u_buf_ptr || n == 0) return 0;
    sofuu_batch_barrier(ctx);
    uint un = (uint)n;
    [ctx->batch_enc setComputePipelineState:ctx->pso_swiglu];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)g_buf_ptr offset:g_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)u_buf_ptr offset:u_offset atIndex:1];
    [ctx->batch_enc setBytes:&un length:sizeof(uint) atIndex:2];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((uint)((n + 255) / 256))
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// ── W2 CPU+GPU feature split glue ─────────────────────────────────────────
// strided_copy_f32: place a compact [rows][n] block into a column slice of the
// canonical [rows][dst_stride] layout. Offsets AND strides are f32 ELEMENT
// counts into the bound buffers (the kernels index a device float*).
int sofuu_metal_batch_strided_copy_f32(
    void* ctx_ptr,
    void* dst_buf_ptr, size_t dst_offset,
    size_t dst_stride,
    void* src_buf_ptr, size_t src_offset,
    size_t src_stride,
    size_t n, size_t rows
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !dst_buf_ptr || !src_buf_ptr || n == 0 || rows == 0) return 0;
    if (ctx->pso_strided_copy_f32 == nil) return 0;
    sofuu_batch_barrier(ctx);
    uint doff = (uint)dst_offset, dstr = (uint)dst_stride;
    uint soff = (uint)src_offset, sstr = (uint)src_stride;
    uint un = (uint)n, rn = (uint)rows;
    size_t total = n * rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_strided_copy_f32];
    // off/stride are ELEMENT offsets relative to the (byte) buffer bindings.
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)dst_buf_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBytes:&doff length:sizeof(uint) atIndex:1];
    [ctx->batch_enc setBytes:&dstr length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)src_buf_ptr offset:0 atIndex:3];
    [ctx->batch_enc setBytes:&soff length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&sstr length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&un length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((uint)((total + 255) / 256))
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// resid_split_add_f32: y += (j < n_a ? pa : pb) over rows x cols.
int sofuu_metal_batch_resid_split_add_f32(
    void* ctx_ptr,
    void* y_buf_ptr, size_t y_offset,
    void* pa_buf_ptr, size_t pa_offset,
    size_t n_a,
    void* pb_buf_ptr, size_t pb_offset,
    size_t n_b,
    size_t cols, size_t rows
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !y_buf_ptr || !pa_buf_ptr || !pb_buf_ptr) return 0;
    if (ctx->pso_resid_split_add_f32 == nil) return 0;
    if (cols == 0 || rows == 0 || n_a + n_b != cols) return 0;
    sofuu_batch_barrier(ctx);
    // Buffers bind at their region (byte offsets = element offsets * 4), so the
    // kernel's y/pa/pb are already the three blocks and no offset arithmetic is
    // needed inside the shader.
    size_t yb = y_offset * sizeof(float);
    size_t pab = pa_offset * sizeof(float);
    size_t pbb = pb_offset * sizeof(float);
    uint na = (uint)n_a, nb = (uint)n_b, cn = (uint)cols, rn = (uint)rows;
    size_t total = cols * rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_resid_split_add_f32];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:yb atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pa_buf_ptr offset:pab atIndex:1];
    [ctx->batch_enc setBytes:&na length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pb_buf_ptr offset:pbb atIndex:3];
    [ctx->batch_enc setBytes:&nb length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&cn length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((uint)((total + 255) / 256))
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

// Slot-based RMSNorm (w is bf16 in the model buffer) reusing the per-op
// `rmsnorm` kernel with caller offsets.
void sofuu_metal_batch_rmsnorm(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// LFM2 gated short conv: y = C * (s0*w0 + s1*w1 + (B*x)*w2), state shifts.
void sofuu_metal_batch_conv_gate(
    void* ctx_ptr,
    void* bcx_buf_ptr, size_t bcx_offset,
    void* conv_w_buf_ptr, size_t conv_w_offset,
    void* state_buf_ptr, size_t state_offset,
    void* out_buf_ptr, size_t out_offset,
    size_t dim
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !bcx_buf_ptr || !conv_w_buf_ptr || !state_buf_ptr || !out_buf_ptr || dim == 0) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = conv_w_offset & ~255ULL;
    uint w_rem = (uint)(conv_w_offset & 255ULL);
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_conv_gate];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)bcx_buf_ptr offset:bcx_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)conv_w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)state_buf_ptr offset:state_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:3];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((dim + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Head-RMSNorm + split-half RoPE on k (+plain copy of v), then append at
// position `pos` of this layer's KV-cache slice. kc/vc offsets delimit the
// LAYER slice; the kernel indexes kv*max_seq + pos inside it.
void sofuu_metal_batch_kv_write(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !kln_buf_ptr) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = kln_offset & ~255ULL;
    uint w_rem = (uint)(kln_offset & 255ULL);
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_kv_write];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kln_buf_ptr offset:aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:10];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:11];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GQA decode attention: head-norm + RoPE on q, softmax(q·K)V against the
// cache slice written by batch_kv_write earlier in this same batch.
void sofuu_metal_batch_attn(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !qln_buf_ptr || !out_buf_ptr) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = qln_offset & ~255ULL;
    uint w_rem = (uint)(qln_offset & 255ULL);
    [ctx->batch_enc setComputePipelineState:ctx->pso_lfm2_attn];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_buf_ptr offset:aligned_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3: per-head f32 RMSNorm (q-norm) on a [heads x head_dim] slot.
int sofuu_metal_batch_head_rmsnorm(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    uint32_t head_dim, uint32_t n_heads,
    uint32_t has_norm, uint32_t pos, float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned = w_buf_ptr ? (w_offset & ~255ULL) : 0;
    uint w_rem = w_buf_ptr ? (uint)(w_offset & 255ULL) / 4 : 0;
    uint hd = head_dim;
    uint nh = n_heads;
    uint hn = has_norm;
    uint rm = rope_mode;
    [ctx->batch_enc setComputePipelineState:ctx->pso_head_rmsnorm];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(w_buf_ptr ? w_buf_ptr : x_buf_ptr) offset:aligned atIndex:1];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:5];
    [ctx->batch_enc setBytes:&hn length:sizeof(uint) atIndex:6];
    if (ctx->attn_noop == 1 || ctx->attn_noop == 4) { pos = 0xFFFFFFFFu; }
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:8];
    [ctx->batch_enc setBytes:&rm length:sizeof(uint) atIndex:9];
    sofuu_bind_rope(ctx, 10, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, x_buf_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// GPU3 attention, stage 1: chunked softmax+V partials to the meta buffer.
int sofuu_metal_batch_llama_attn_part(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* meta_buf_ptr, size_t meta_offset,
    uint32_t has_norm_q_unused,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) { (void)has_norm_q_unused; return 0; }
    sofuu_batch_barrier(ctx);
    size_t aligned = qln_buf_ptr ? (qln_offset & ~255ULL) : 0;
    uint w_rem = qln_buf_ptr ? (uint)(qln_offset & 255ULL) / 4 : 0;
    uint hn = has_norm_q_unused;   // kept for ABI symmetry; norm lives in head_rmsnorm now
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_part];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(qln_buf_ptr ? qln_buf_ptr : q_buf_ptr)
                      offset: aligned atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&hn length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    uint n_chunks = (pos + 1 + 127) / 128;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads * n_chunks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// GPU3 attention, stage 2: rescale chunk partials into head_out.
int sofuu_metal_batch_llama_attn_fin(
    void* ctx_ptr,
    void* meta_buf_ptr, size_t meta_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t pos,
    uint32_t n_heads, uint32_t head_dim
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_fin];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// ── MULTI-ROW (B≤4 chain) attention ops: positions come from a u32 buffer ──
// grid = rows × heads(× chunks), row = threadgroup / heads; contiguous
// [B][...] row slots; meta regions stride meta_stride per row.

// FUSED 1-chunk attention (pos<128): part+fin in one dispatch, no meta.
// Returns 1 when dispatched, 0 when the PSO is unavailable (caller falls
// back to the part+fin pair).

// ===========================================================================
// KV4 (quantized KV cache, KIVI layout) - batch entry points.
// ===========================================================================

// Can this GPU + this model geometry run the KV4 class? 0 = no (a one-line
// reason is printed once), 1 = yes. The geometry gate is exactly what the two
// kernels accept: head_dim a multiple of 32 and <= 128 (one warp covers a head
// with <= 4 channels per lane), n_heads divisible by n_kv_heads (GQA ratio), a
// block/ring layout that fits (head_dim/32 groups, 32-token blocks) and BOTH
// pipelines live at their 1024-thread shape. Anything else must NOT be
// dispatched: the caller keeps the f32 KV cache instead (loudly).
int sofuu_metal_kv4_support(void* ctx_ptr, uint32_t head_dim, uint32_t n_heads,
                            uint32_t n_kv_heads, uint32_t bits) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->pso_kv4_quant4 == nil || ctx->pso_llama_attn_w32_kv4 == nil ||
        ctx->pso_llama_attn_pre_w32_kv4 == nil) {
        fprintf(stderr, "[kv4] unavailable: a KV4 pipeline is missing "
                        "(quant=%d decode_attn=%d prefill_attn=%d)\n",
                ctx->pso_kv4_quant4 != nil, ctx->pso_llama_attn_w32_kv4 != nil,
                ctx->pso_llama_attn_pre_w32_kv4 != nil);
        return 0;
    }
    if ((bits != 4u && bits != 2u && bits != 8u) ||
        (bits == 2u && (ctx->pso_kv4_quant2 == nil || ctx->pso_llama_attn_w32_kv2 == nil ||
                        ctx->pso_llama_attn_pre_w32_kv2 == nil)) ||
        (bits == 8u && (ctx->pso_kv4_quant8 == nil || ctx->pso_llama_attn_w32_kv8 == nil ||
                        ctx->pso_llama_attn_pre_w32_kv8 == nil))) {
        fprintf(stderr, "[kv4] unavailable: %u-bit pipelines missing\n", bits);
        return 0;
    }
    if (head_dim == 0u || (head_dim % 32u) != 0u || head_dim > 128u) {
        fprintf(stderr, "[kv4] unavailable: head_dim %u is not a multiple of 32 in (0,128]\n", head_dim);
        return 0;
    }
    if (n_kv_heads == 0u || n_heads == 0u || (n_heads % n_kv_heads) != 0u) {
        fprintf(stderr, "[kv4] unavailable: n_heads %u / n_kv_heads %u is not an integer GQA ratio\n",
                n_heads, n_kv_heads);
        return 0;
    }
    return 1;
}

// Quantize nblocks of the 32-token blocks that have just rolled out of the f32
// residual ring, for ONE layer (all kv heads). Buffers are this layer's slices.
int sofuu_metal_batch_kv4_quant(
    void* ctx_ptr,
    void* k_ring_ptr, size_t k_ring_offset,
    void* v_ring_ptr, size_t v_ring_offset,
    void* qk_ptr, size_t qk_offset,
    void* qks_ptr, size_t qks_offset,
    void* qkm_ptr, size_t qkm_offset,
    void* qv_ptr, size_t qv_offset,
    void* qvs_ptr, size_t qvs_offset,
    void* qvm_ptr, size_t qvm_offset,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t ring, uint32_t nblk,
    uint32_t block0, uint32_t nblocks, uint32_t bits
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || nblocks == 0u || ring == 0u) return 0;
    if (!k_ring_ptr || !v_ring_ptr || !qk_ptr || !qks_ptr || !qkm_ptr ||
        !qv_ptr || !qvs_ptr || !qvm_ptr) return 0;
    id<MTLComputePipelineState> pso = (bits == 2u)   ? ctx->pso_kv4_quant2
                                    : (bits == 8u) ? ctx->pso_kv4_quant8
                                                   : ctx->pso_kv4_quant4;
    if (pso == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_ring_ptr offset:k_ring_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_ring_ptr offset:v_ring_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qk_ptr offset:qk_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qks_ptr offset:qks_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qkm_ptr offset:qkm_offset atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qv_ptr offset:qv_offset atIndex:5];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvs_ptr offset:qvs_offset atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvm_ptr offset:qvm_offset atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&ring length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&nblk length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&block0 length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&nblocks length:sizeof(uint) atIndex:12];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(nblocks, n_kv_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// Decode attention over (quantized blocks + f32 residual ring). Same shape and
// bindings as sofuu_metal_batch_llama_attn_w32_out for buffers 0..20.
int sofuu_metal_batch_llama_attn_w32_kv4(
    void* ctx_ptr,
    void* q_ptr, size_t q_offset,
    void* k_ring_ptr, size_t k_ring_offset,
    void* v_ring_ptr, size_t v_ring_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs,
    void* qk_ptr, size_t qk_offset,
    void* qks_ptr, size_t qks_offset,
    void* qkm_ptr, size_t qkm_offset,
    void* qv_ptr, size_t qv_offset,
    void* qvs_ptr, size_t qvs_offset,
    void* qvm_ptr, size_t qvm_offset,
    uint32_t nblk, uint32_t ring, uint32_t bits
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (bits == 2u)   ? ctx->pso_llama_attn_w32_kv2
                                    : (bits == 8u) ? ctx->pso_llama_attn_w32_kv8
                                                   : ctx->pso_llama_attn_w32_kv4;
    if (ctx->batch_enc == nil || pso == nil) return 0;
    if (!q_ptr || !k_ring_ptr || !v_ring_ptr || !out_ptr) return 0;
    if (!qk_ptr || !qks_ptr || !qkm_ptr || !qv_ptr || !qvs_ptr || !qvm_ptr) return 0;
    if (pos >= 4096u || head_dim > 128u || n_kv_heads == 0u || (n_heads % n_kv_heads) != 0u) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = qln_ptr ? (qln_offset & ~255ULL) : 0;
    uint w_rem = qln_ptr ? (uint)(qln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_ring_ptr offset:k_ring_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_ring_ptr offset:v_ring_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(qln_ptr ? qln_ptr : q_ptr)
                      offset: aligned_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    if (ctx->attn_noop == 1 || ctx->attn_noop == 2) { pos = 0xFFFFFFFFu; }
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_ptr);
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qk_ptr offset:qk_offset atIndex:21];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qks_ptr offset:qks_offset atIndex:22];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qkm_ptr offset:qkm_offset atIndex:23];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qv_ptr offset:qv_offset atIndex:24];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvs_ptr offset:qvs_offset atIndex:25];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvm_ptr offset:qvm_offset atIndex:26];
    [ctx->batch_enc setBytes:&nblk length:sizeof(uint) atIndex:27];
    [ctx->batch_enc setBytes:&ring length:sizeof(uint) atIndex:28];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return 1;
}

// Prefill (whole 128-row chunk) attention over the same two K/V sources: one
// threadgroup per (row, head), 32 warps splitting the key range.
int sofuu_metal_batch_llama_attn_pre_w32_kv4(
    void* ctx_ptr,
    void* q_ptr, size_t q_offset,
    void* k_ring_ptr, size_t k_ring_offset,
    void* v_ring_ptr, size_t v_ring_offset,
    void* out_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t out_stride,
    void* qk_ptr, size_t qk_offset,
    void* qks_ptr, size_t qks_offset,
    void* qkm_ptr, size_t qkm_offset,
    void* qv_ptr, size_t qv_offset,
    void* qvs_ptr, size_t qvs_offset,
    void* qvm_ptr, size_t qvm_offset,
    uint32_t nblk, uint32_t ring, uint32_t bits
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLComputePipelineState> pso = (bits == 2u)   ? ctx->pso_llama_attn_pre_w32_kv2
                                    : (bits == 8u) ? ctx->pso_llama_attn_pre_w32_kv8
                                                   : ctx->pso_llama_attn_pre_w32_kv4;
    if (ctx->batch_enc == nil || pso == nil) return 0;
    if (!q_ptr || !k_ring_ptr || !v_ring_ptr || !out_ptr || !pos_buf_ptr) return 0;
    if (!qk_ptr || !qks_ptr || !qkm_ptr || !qv_ptr || !qvs_ptr || !qvm_ptr) return 0;
    if (rows == 0u || head_dim == 0u || head_dim > 128u || n_kv_heads == 0u ||
        (n_heads % n_kv_heads) != 0u) return 0;
    sofuu_batch_barrier(ctx);
    uint rpt = 1u, n_chunks_max = 1u;
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_ring_ptr offset:k_ring_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_ring_ptr offset:v_ring_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:5];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&out_stride length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&rows length:sizeof(uint) atIndex:12];
    [ctx->batch_enc setBytes:&rpt length:sizeof(uint) atIndex:13];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qk_ptr offset:qk_offset atIndex:14];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qks_ptr offset:qks_offset atIndex:15];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qkm_ptr offset:qkm_offset atIndex:16];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qv_ptr offset:qv_offset atIndex:17];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvs_ptr offset:qvs_offset atIndex:18];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvm_ptr offset:qvm_offset atIndex:19];
    [ctx->batch_enc setBytes:&nblk length:sizeof(uint) atIndex:20];
    [ctx->batch_enc setBytes:&ring length:sizeof(uint) atIndex:21];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(rows * n_heads)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return 1;
}

// ── KV4 v2 dispatch (see the KV4 v2 block comment in kernels.metal) ────────
// Number of key-range slices for one decode step. Default 2: n_kv_heads (8 on
// the models this was built for) x 2 = 16 threadgroups, which is exactly the
// threadgroup count the f32 decode arm runs and the count at which that arm's
// own note says resident warps stop being the limit. SOFUU_KV4_SPLITS=1 makes
// the merge the identity (the shipped single-threadgroup arithmetic); 4 is a
// legal sweep point (32 threadgroups).
static inline uint32_t sofuu_kv4_splits(void) {
    static int s = -2;
    if (s == -2) {
        s = 2;
        if (const char* e = getenv("SOFUU_KV4_SPLITS")) {
            int v = atoi(e);
            s = (v < 1) ? 1 : (v > 8 ? 8 : v);
        }
    }
    return (uint32_t)s;
}

// SOFUU_ATTN_DUP: how many EXTRA times to re-issue one kv4 attention dispatch.
// Both v2 kernels are idempotent (they only overwrite their own outputs and no
// kernel here mutates a live input), so the total-time delta between N=0 and
// N=1 is the marginal cost of ONE attention dispatch measured in situ, with the
// real grid, cache state and neighbouring dispatches. That is the kernel cost
// receipt the kv4 arms are reported against - the same instrument the f32 chunk
// path already carries.
static inline uint32_t sofuu_kv4_dup(void) {
    static int d = -2;
    if (d == -2) {
        d = 0;
        if (const char* e = getenv("SOFUU_ATTN_DUP")) {
            int v = atoi(e);
            d = (v > 0) ? (v > 8 ? 8 : v) : 0;
        }
    }
    return (uint32_t)d;
}

// Split-K partials: nsplit * n_heads records of (head_dim + 2) floats, each
// record start 4-float aligned. Private storage: only the split and merge
// kernels touch it.
static inline int sofuu_kv4_scratch(MetalContext* ctx, size_t bytes) {
    if (ctx->kv4_sk_scratch != nil && ctx->kv4_sk_cap >= bytes) return 1;
    id<MTLBuffer> b = [ctx->device newBufferWithLength:bytes
                                              options:MTLResourceStorageModePrivate];
    if (b == nil) {
        fprintf(stderr, "[kv4] could not allocate %zu B of split-K scratch - shipped kv4 kernel\n", bytes);
        return 0;
    }
    ctx->kv4_sk_scratch = b;
    ctx->kv4_sk_cap = bytes;
    return 1;
}

// HOW BIG A THREADGROUP the v2 kernels get. They keep R heads of online-softmax
// state per thread, so the register budget - not the 1024-thread ceiling - is
// what binds: measured ~102 regs/thread at R=2 and ~146 at R=4, i.e. 640 and
// 448 threads. The size is therefore read off the live pipeline: the largest
// multiple of 32 that fits, capped at 512 (16 warps, the KV4G_WMAX threadgroup
// memory bound) and floored at 256. Returns 0 when even that does not fit, which
// hands the shape back to the shipped kernel.
static inline uint32_t sofuu_kv4g_threads(id<MTLComputePipelineState> pso) {
    if (pso == nil) return 0;
    NSUInteger cap = pso.maxTotalThreadsPerThreadgroup;
    if (cap > 512ul) cap = 512ul;
    uint32_t nt = (uint32_t)((cap / 32ul) * 32ul);
    if (nt < 256u) return 0;
    return nt;
}

static inline int sofuu_kv4g_ratio_index(uint32_t n_heads, uint32_t n_kv_heads) {
    if (n_kv_heads == 0u || (n_heads % n_kv_heads) != 0u) return -1;
    uint32_t r = n_heads / n_kv_heads;
    if (r == 2u) return 0;
    if (r == 4u) return 1;
    return -1;
}

int sofuu_metal_batch_llama_attn_w32_kv4g(
    void* ctx_ptr,
    void* q_ptr, size_t q_offset,
    void* k_ring_ptr, size_t k_ring_offset,
    void* v_ring_ptr, size_t v_ring_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs,
    void* qk_ptr, size_t qk_offset,
    void* qks_ptr, size_t qks_offset,
    void* qkm_ptr, size_t qkm_offset,
    void* qv_ptr, size_t qv_offset,
    void* qvs_ptr, size_t qvs_offset,
    void* qvm_ptr, size_t qvm_offset,
    uint32_t nblk, uint32_t ring, uint32_t bits
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return 0;
    if (bits != 4u && bits != 8u) return 0;
    if ((ring >> 31u) != 0u) return 0;   // f16-K: the shipped kernel owns it
    int ri = sofuu_kv4g_ratio_index(n_heads, n_kv_heads);
    if (ri < 0) return 0;
    id<MTLComputePipelineState> pso = ctx->pso_kv4g_dec[(bits == 8u) ? 1 : 0][ri];
    if (pso == nil || ctx->pso_kv4_cmb == nil) return 0;
    if (!q_ptr || !k_ring_ptr || !v_ring_ptr || !out_ptr) return 0;
    if (!qk_ptr || !qks_ptr || !qkm_ptr || !qv_ptr || !qvs_ptr || !qvm_ptr) return 0;
    if (pos >= 4096u || head_dim == 0u || head_dim > 128u || (head_dim % 32u) != 0u) return 0;
    uint32_t nt = sofuu_kv4g_threads(pso);
    if (nt == 0u) return 0;
    uint32_t nwarp = nt / 32u;
    uint32_t nsplit = sofuu_kv4_splits();
    uint32_t seq = pos + 1u;
    while (nsplit > 1u && (seq / nsplit) < 32u) nsplit--;
    while (nsplit > 1u && (size_t)n_kv_heads * nsplit > 256u) nsplit--;
    uint32_t rec_stride = ((head_dim + 2u + 3u) / 4u) * 4u;
    size_t need = (size_t)nsplit * (size_t)n_heads * (size_t)rec_stride * sizeof(float);
    uint32_t rec_arg = (rec_stride & 0xFFFFu) | (nwarp << 16);
    if (!sofuu_kv4_scratch(ctx, need)) return 0;
    if (ctx->attn_noop == 1 || ctx->attn_noop == 2) { pos = 0xFFFFFFFFu; }
    sofuu_batch_barrier(ctx);
    uint32_t hd = head_dim, nh = n_heads, nkv = n_kv_heads, ms = max_seq;
    uint32_t nblk_v = nblk, ring_v = ring, ns = nsplit, rs = rec_arg;
    size_t aligned_offset = qln_ptr ? (qln_offset & ~255ULL) : 0;
    uint32_t w_rem = qln_ptr ? (uint32_t)(qln_offset & 255ULL) / 4u : 0u;
    uint32_t pos_v = pos, hn = has_norm;
    uint32_t dup = sofuu_kv4_dup();
    for (uint32_t rep = 0; rep <= dup; ++rep) {
        [ctx->batch_enc setComputePipelineState:pso];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_ring_ptr offset:k_ring_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_ring_ptr offset:v_ring_offset atIndex:2];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(qln_ptr ? qln_ptr : q_ptr)
                          offset: aligned_offset atIndex:3];
        [ctx->batch_enc setBuffer:ctx->kv4_sk_scratch offset:0 atIndex:4];
        [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
        [ctx->batch_enc setBytes:&hn length:sizeof(uint) atIndex:6];
        [ctx->batch_enc setBytes:&pos_v length:sizeof(uint) atIndex:7];
        [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:8];
        [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:9];
        [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:10];
        [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:11];
        [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
        [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
        [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
        sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                        rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                        head_dim, rope_n_rot, rope_mscale, q_ptr);
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qk_ptr offset:qk_offset atIndex:21];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qks_ptr offset:qks_offset atIndex:22];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qkm_ptr offset:qkm_offset atIndex:23];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qv_ptr offset:qv_offset atIndex:24];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvs_ptr offset:qvs_offset atIndex:25];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvm_ptr offset:qvm_offset atIndex:26];
        [ctx->batch_enc setBytes:&nblk_v length:sizeof(uint) atIndex:27];
        [ctx->batch_enc setBytes:&ring_v length:sizeof(uint) atIndex:28];
        [ctx->batch_enc setBytes:&ns length:sizeof(uint) atIndex:29];
        [ctx->batch_enc setBytes:&rs length:sizeof(uint) atIndex:30];
        [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(nkv * ns, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
        if (rep < dup) { sofuu_batch_barrier(ctx); }
    }
    if (pos == 0xFFFFFFFFu) return 1;   // ablation: the output is not written
    if (ctx->no_barrier == 0) { [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    [ctx->batch_enc setComputePipelineState:ctx->pso_kv4_cmb];
    [ctx->batch_enc setBuffer:ctx->kv4_sk_scratch offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:1];
    [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:3];
    uint32_t rs_plain = rec_stride;
    [ctx->batch_enc setBytes:&ns length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&rs_plain length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d((n_heads + 31u) / 32u)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return 1;
}

// Prefill over the same two K/V sources: KV4G_RG (32) rows of one kv head per
// threadgroup, one warp per row, all R q heads of that row per key.
int sofuu_metal_batch_llama_attn_pre_row_kv4(
    void* ctx_ptr,
    void* q_ptr, size_t q_offset,
    void* k_ring_ptr, size_t k_ring_offset,
    void* v_ring_ptr, size_t v_ring_offset,
    void* out_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t out_stride,
    void* qk_ptr, size_t qk_offset,
    void* qks_ptr, size_t qks_offset,
    void* qkm_ptr, size_t qkm_offset,
    void* qv_ptr, size_t qv_offset,
    void* qvs_ptr, size_t qvs_offset,
    void* qvm_ptr, size_t qvm_offset,
    uint32_t nblk, uint32_t ring, uint32_t bits
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return 0;
    if (bits != 4u && bits != 8u) return 0;
    if ((ring >> 31u) != 0u) return 0;   // f16-K: the shipped kernel owns it
    int ri = sofuu_kv4g_ratio_index(n_heads, n_kv_heads);
    if (ri < 0) return 0;
    id<MTLComputePipelineState> pso = ctx->pso_kv4g_pre[(bits == 8u) ? 1 : 0][ri];
    if (pso == nil) return 0;
    if (!q_ptr || !k_ring_ptr || !v_ring_ptr || !out_ptr || !pos_buf_ptr) return 0;
    if (!qk_ptr || !qks_ptr || !qkm_ptr || !qv_ptr || !qvs_ptr || !qvm_ptr) return 0;
    if (rows == 0u || head_dim == 0u || head_dim > 128u || (head_dim % 32u) != 0u) return 0;
    uint32_t nt = sofuu_kv4g_threads(pso);
    if (nt == 0u) return 0;
    uint32_t nwarp = nt / 32u;
    sofuu_batch_barrier(ctx);
    uint32_t hd = head_dim, nh = n_heads, nkv = n_kv_heads, ms = max_seq;
    uint32_t os = out_stride, rn = rows, nblk_v = nblk, ring_v = ring;
    uint32_t nw = nwarp;
    uint32_t dup = sofuu_kv4_dup();
    for (uint32_t rep = 0; rep <= dup; ++rep) {
        [ctx->batch_enc setComputePipelineState:pso];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_ring_ptr offset:k_ring_offset atIndex:2];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_ring_ptr offset:v_ring_offset atIndex:3];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
        [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:5];
        [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:6];
        [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:7];
        [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:8];
        [ctx->batch_enc setBytes:&os length:sizeof(uint) atIndex:9];
        [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:10];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qk_ptr offset:qk_offset atIndex:11];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qks_ptr offset:qks_offset atIndex:12];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qkm_ptr offset:qkm_offset atIndex:13];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qv_ptr offset:qv_offset atIndex:14];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvs_ptr offset:qvs_offset atIndex:15];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qvm_ptr offset:qvm_offset atIndex:16];
        [ctx->batch_enc setBytes:&nblk_v length:sizeof(uint) atIndex:17];
        [ctx->batch_enc setBytes:&ring_v length:sizeof(uint) atIndex:18];
        [ctx->batch_enc setBytes:&nw length:sizeof(uint) atIndex:19];
        [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, (rows + nwarp - 1u) / nwarp, 1)
                        threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
        if (rep < dup) { sofuu_batch_barrier(ctx); }
    }
    return 1;
}

int sofuu_metal_batch_llama_attn_1c_b4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr) return 0;
    if (ctx->pso_llama_attn_1c_b4 == nil) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_1c_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:8];
    // 1c consumes ALREADY-roped q (head_rmsnorm ran): table bound inert
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(rope_tab_ptr ? rope_tab_ptr : q_buf_ptr)
                      offset: (rope_tab_ptr ? rope_tab_offset : 0) atIndex:9];
    uint ht = has_table;
    [ctx->batch_enc setBytes:&ht length:sizeof(uint) atIndex:10];
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(grid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// B4 q4_K GEMV with the resid-add epilogue (o/down projections): returns 1
// when dispatched, 0 when unavailable (caller falls back to gemv + add).
int sofuu_metal_batch_gguf_b4r(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    void* acc_buf_ptr, size_t acc_offset,
    size_t rows, size_t cols, size_t acc_dim
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || !acc_buf_ptr) return 0;
    if (ctx->pso_gemv_q4_k_b4c2r == nil) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint r = (uint)rows;
    int8_t acc_dim_i = (int8_t)acc_dim; (void)acc_dim_i;
    uint acc_dim_u = (uint)acc_dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_b4c2r];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)acc_buf_ptr offset:acc_offset atIndex:6];
    [ctx->batch_enc setBytes:&acc_dim_u length:sizeof(uint) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

void sofuu_metal_batch_llama_kv_write_b4(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t has_norm,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs,
    uint32_t ring
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr) return;
    if (ctx->pso_llama_kv_write_b4 == nil) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = kln_buf_ptr ? (kln_offset & ~255ULL) : 0;
    uint w_rem = kln_buf_ptr ? (uint)(kln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_kv_write_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(kln_buf_ptr ? kln_buf_ptr : k_buf_ptr)
                      offset: aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:13];
    sofuu_bind_rope(ctx, 14, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, pos_buf_ptr);
    // KV4 ring mode: 0 = the shipped absolute (pos) row addressing, else the
    // f32 residual ring is `ring` rows per head and rows land at pos % ring.
    [ctx->batch_enc setBytes:&ring length:sizeof(uint) atIndex:20];
    uint grid = rows * n_kv_heads;
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(grid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_head_rmsnorm_b4(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads,
    uint32_t has_norm,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr) return;
    if (ctx->pso_head_rmsnorm_b4 == nil) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = w_buf_ptr ? (w_offset & ~255ULL) : 0;
    uint w_rem = w_buf_ptr ? (uint)(w_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_head_rmsnorm_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)(w_buf_ptr ? w_buf_ptr : x_buf_ptr)
                      offset: aligned_offset atIndex:1];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:2];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:7];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:8];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:9];
    sofuu_bind_rope(ctx, 10, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, x_buf_ptr);
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(grid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void sofuu_metal_batch_llama_attn_part_b4(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* meta_buf_ptr, size_t meta_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    uint32_t n_chunks_max, uint32_t meta_stride
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !meta_buf_ptr) return;
    if (ctx->pso_llama_attn_part_b4 == nil) return;
    sofuu_batch_barrier(ctx);
    // VALIDITY GATE for llama_attn_pre_w32, the whole-row fast attention. The
    // gate used to stop at one 128-key window because the kernel that shipped
    // then was only right below that (it wrote dead rows above it: max|delta|
    // 9.11/9.02/8.96/8.83 at 130/136/150/200 tokens, one head correct and the
    // other 15 zero - "attention contributed nothing"). The kernel in the tree
    // now is the per-(row,head) transplant, which has none of that machinery,
    // and it is correct at every position measured, so the gate opens for the
    // whole prompt. SOFUU_PREW_MAXPOS still bounds it and is the rollback knob
    // to the shipped part_b4+fin_b4 pair (SOFUU_PREW_MAXPOS=128 restores the
    // old one-window behaviour).
    //
    // Evidence (M2 Pro 16-core GPU, Qwen3-0.6B i1-Q4_K_M, route
    // SOFUU_GPU_PREFILL=1, CPU default prefill as the reference; max|delta| on
    // prefill_logits[..8], all three numbers from the same prompt prefix):
    //   tokens  fast vs CPU  pair vs CPU  fast vs pair
    //       39     0.341        0.341        0     (below one window the fused
    //      148     0.130        0.130        3e-6  1c kernel runs, both settings
    //      176     0.403        0.403        8e-6  agree bit for bit)
    //      286     0.218        0.218        6e-6
    //      637     0.222        0.222        3e-6
    //     1454     0.281        0.281        3e-6
    // The fast path IS the pair numerically (<=8e-6 apart); the 0.13-0.40 band
    // against the CPU is the bf16 rounding of this route, which is why 176 reads
    // 0.403 for both - parity with the pair, not a new tolerance (0.281 at 1454
    // is the established number).
    // Second sweep, one-token-per-word prompt, gate wide open, tokens
    // 126/127/128/129/130/136/150/200 -> 0.427/0.385/0.407/0.333/0.288/0.282/
    // 0.327/0.387, i.e. the same band straight through the 128-key boundary.
    //
    // positions is a zero-copy buffer, so the bound is read on the host here
    // instead of adding a kernel argument.
    int pre_window_ok = 0;
    if (sofuu_pre_attn_on() != 0 && ctx->pso_llama_attn_pre_w32 != nil && rows > 0u && pos_buf_ptr) {
        id<MTLBuffer> pb = (__bridge id<MTLBuffer>)pos_buf_ptr;
        if (pb.contents != NULL) {
            const uint32_t* pp = (const uint32_t*)((const char*)pb.contents + pos_offset);
            uint32_t maxpos = sofuu_prew_maxpos();
            pre_window_ok = (pp[rows - 1u] < maxpos) ? 1 : 0;
        }
    }
    // Decide ONCE and publish it for the fin pass (see the field comment).
    const int fast_ok = pre_window_ok != 0
        && head_dim <= 128u && n_kv_heads != 0u && n_heads % n_kv_heads == 0u && rows > 0u;
    ctx->pre_window_active = fast_ok;
    if (fast_ok) {
        uint rpt = sofuu_pre_attn_rpt();
        // One threadgroup per (row, head): the kernel takes a whole row, so rpt
        // does not shape the grid (it stays bound for ABI stability).
        uint groups = rows * n_heads;
        uint r_n = rows;
        [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_pre_w32];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:3];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:4];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:5];
        [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:6];
        [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:7];
        [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
        [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
        [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:10];
        [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:11];
        [ctx->batch_enc setBytes:&r_n length:sizeof(uint) atIndex:12];
        [ctx->batch_enc setBytes:&rpt length:sizeof(uint) atIndex:13];
        [ctx->batch_enc dispatchThreadgroups:sf_grid2d(groups)
                        threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        return;
    }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_part_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:5];
    // buffer 3 (q_ln) unused in the b4 flow: bind q as a placeholder
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:3];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:11];
    uint grid = rows * n_heads * n_chunks_max;
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(grid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── whole-row chunk attention for the 128-row chain pass ────────────────────
// Two kernels, one contract: the whole row's softmax is computed in ONE
// threadgroup per (row, head) or per (PAR_ROWS rows, head), the result lands
// directly in the caller's [row][head][head_dim] slab, and the function returns
// 0 (no dispatch at all) when the pipeline cannot host the threadgroup or the
// geometry is outside what the kernel computes. Returning 0 is what keeps the
// b4 family (flash -> 1c -> part/fin) as the fallback: a return of 1 means the
// dispatch was enqueued and attention is DONE for this layer, so the caller must
// skip the b4 family entirely.
//
// The gate is the same single source of truth the part_b4 fast path uses
// (sofuu_pre_attn_on + sofuu_prew_maxpos): SOFUU_PRE_ATTN=0 or a
// SOFUU_PREW_MAXPOS bound below the chunk's last position hands the chunk back
// to the b4 family rather than running a kernel with an invalid window.
static inline int sofuu_chain_attn_gate(MetalContext* ctx, void* pos_buf_ptr, size_t pos_offset,
                                        uint32_t rows, uint32_t head_dim, uint32_t n_heads,
                                        uint32_t n_kv_heads, uint32_t max_seq, uint32_t out_stride) {
    if (ctx->batch_enc == nil || rows == 0u || max_seq == 0u) return 0;
    if (head_dim == 0u || head_dim > 128u || (head_dim & 3u) != 0u) return 0;
    if (n_heads == 0u || n_kv_heads == 0u || n_heads % n_kv_heads != 0u) return 0;
    if (out_stride < n_heads * head_dim) return 0;
    if (sofuu_pre_attn_on() == 0) return 0;
    if (pos_buf_ptr == NULL) return 0;
    id<MTLBuffer> pb = (__bridge id<MTLBuffer>)pos_buf_ptr;
    if (pb.contents == NULL) return 0;
    const uint32_t* pp = (const uint32_t*)((const char*)pb.contents + pos_offset);
    // zero-copy positions: the host bound (not a kernel argument) is the same
    // read the part_b4 gate does.
    if (pp[rows - 1u] >= sofuu_prew_maxpos()) return 0;
    // llama_attn_pre_row bounds the whole row GROUP by its last row's position
    // (that is what keeps the 32 warps on the same key index), so it needs the
    // chunk's positions to be non-decreasing. The chain pass writes
    // start_pos + t and already refuses anything else, but a caller that ever
    // hands this a permuted position vector must get the b4 family (which bounds
    // every row independently) rather than a silently truncated window.
    for (uint32_t i = 1u; i < rows; ++i) {
        if (pp[i] < pp[i - 1u]) return 0;
    }
    return 1;
}

// ── "why is this shape not usable" text ─────────────────────────────────────
// Both "not usable" lines quote the REAL reason (the pipeline's own reported
// ceiling, or a nil pipeline, or the opt-in gate) instead of a bare "disabled":
// a shape that is nil because its threadgroup does not fit is a different
// problem from one that never compiled, and conflating them is what let the
// per-head fallback hide a dead shape table.
static void sofuu_gqa_why(MetalContext* ctx, int idx, char* out, size_t n) {
    if (out == nullptr || n == 0) return;
    out[0] = '\0';
    if (idx < 0 || idx >= SOFUU_GQA_N) { snprintf(out, n, "no such shape"); return; }
    const SofuuGqaShape &s = k_sofuu_gqa[idx];
    if (s.opt_in && !ctx->attn_gqa_all) {
        snprintf(out, n, "opt-in shape, SOFUU_ATTN_GQA_ALL=0");
    } else if (ctx->pso_attn_gqa[idx] == nil && ctx->attn_gqa_max_tg[idx] == 0u) {
        snprintf(out, n, "pipeline nil (function missing or failed to compile)");
    } else if (ctx->attn_gqa_max_tg[idx] < s.threads) {
        snprintf(out, n, "needs %u threads/group, pipeline allows %u",
                 s.threads, ctx->attn_gqa_max_tg[idx]);
    } else if (ctx->attn_gqa_only >= 0 && ctx->attn_gqa_only != idx) {
        snprintf(out, n, "live, but SOFUU_ATTN_GQA_ONLY=%d pins another form",
                 ctx->attn_gqa_only);
    } else {
        snprintf(out, n, "live but not selected");
    }
}
static void sofuu_gqa_why_ratio(MetalContext* ctx, uint32_t ratio, char* out, size_t n) {
    if (out == nullptr || n == 0) return;
    out[0] = '\0';
    size_t used = 0;
    int nf = 0;
    for (int i = 0; i < SOFUU_GQA_N; ++i) {
        if (k_sofuu_gqa[i].ratio != ratio) continue;
        ++nf;
        char why[160];
        char piece[240];
        sofuu_gqa_why(ctx, i, why, sizeof(why));
        snprintf(piece, sizeof(piece), "%s%s: %s", (nf > 1) ? "; " : "",
                 k_sofuu_gqa[i].name, why);
        const size_t len = strlen(piece);
        if (used + len + 1u < n) {
            memcpy(out + used, piece, len + 1u);
            used += len;
        }
    }
    if (nf == 0) snprintf(out, n, "the table has no form for this ratio");
    else if (used + 4u < n) memcpy(out + used, " ...", 5u);
}

// PAR_ROWS rows x 1 head per threadgroup, one warp per row (llama_attn_pre_row).
// Grid = (n_heads, ceil(rows/PAR_ROWS)); both dimensions are far below Metal's
// 1024 cap for any chunk width the chain pass allows (b <= 128). Head on x keeps
// the two heads that share a kv head adjacent in launch order, so they read the
// same K/V lines.
int sofuu_metal_batch_llama_attn_row(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t out_stride
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    {
        bool any = (ctx->pso_llama_attn_pre_row != nil) || (ctx->attn_gqa_mask != 0u);
        if (!any) return 0;
    }
    if (!q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr || !pos_buf_ptr) return 0;
    if (!sofuu_chain_attn_gate(ctx, pos_buf_ptr, pos_offset, rows, head_dim, n_heads,
                               n_kv_heads, max_seq, out_stride)) return 0;
    sofuu_batch_barrier(ctx);
    uint hd = head_dim, nh = n_heads, nkv = n_kv_heads, ms = max_seq;
    uint stride = out_stride, rn = rows;
    // GQA pairing: the n_heads/n_kv_heads q heads of one kv head read the same
    // K/V, so the paired kernel (kernels.metal llama_attn_pre_row_gqa) loads it
    // once for the whole group: grid x becomes the KV head count and each warp
    // computes R heads of NR rows. The row group and the threadgroup width come
    // from the shape table (rg rows, (rg/nr)*32 threads); each shape's own
    // maxTotalThreadsPerThreadgroup was checked in sofuu_metal_init, so this
    // dispatch can never exceed it.
    id<MTLComputePipelineState> pso = ctx->pso_llama_attn_pre_row;
    uint grid_x = n_heads;
    uint rg = 32u;
    uint nthr = 1024u;
    uint ratio = (nkv > 0u) ? (nh / nkv) : 0u;
    g_sofuu_gqa_last = -1;   // reset per dispatch: the diagnostic must not
                             // report a previous chunk's shape
    if (ctx->attn_gqa_mode != 0 && ratio >= 1u && nh % nkv == 0u) {
        // The shape table owns the choice. A ratio the table has no form for
        // (5:1, ...) leaves the per-head kernel in charge - the kernel's
        // compile-time R must equal the ratio, because R fixes how many q heads
        // read one kv head and a mismatch silently writes another head's rows.
        //
        // SOFUU_ATTN_GQA (2:1 pins, the historical 1..5 = table entry 0..4) and
        // SOFUU_ATTN_GQA_ONLY (one shape under test) are REQUESTS: whenever the
        // requested pipeline is not live - most importantly the _pf form, which
        // needs 1024 threads and is nil on any pipeline that reports less - the
        // rule falls through to the fastest live SIBLING of the same ratio and
        // says so once. It never turns an unavailable shape into a silent
        // per-head dispatch while a paired form of that ratio is available.
        int idx = -1;
        const bool pin_applies = ctx->attn_gqa_pin && ratio == 2u;
        if (pin_applies) {
            const int want = ctx->attn_gqa_mode - 1;
            const bool want_live = want >= 0 && want < SOFUU_GQA_N
                && k_sofuu_gqa[want].ratio == ratio
                && ctx->pso_attn_gqa[want] != nil
                && (ctx->attn_gqa_mask & (1u << want)) != 0u;
            if (want_live) {
                idx = want;
            } else {
                idx = sofuu_attn_gqa_pick_ex(ratio, ctx->attn_gqa_mask, ctx->attn_gqa_all);
                if (!ctx->attn_gqa_pin_reported) {
                    ctx->attn_gqa_pin_reported = 1;
                    char why[160];
                    sofuu_gqa_why(ctx, want, why, sizeof(why));
                    if (idx >= 0) {
                        fprintf(stderr,
                                "[metal] SOFUU_ATTN_GQA=%d pins %s, which is not usable (%s) - using %s instead\n",
                                ctx->attn_gqa_mode,
                                (want >= 0 && want < SOFUU_GQA_N) ? k_sofuu_gqa[want].name : "?",
                                why, k_sofuu_gqa[idx].name);
                    } else {
                        fprintf(stderr,
                                "[metal] SOFUU_ATTN_GQA=%d pins %s, which is not usable (%s) and no other paired shape of ratio %u is either - per-head kernel owns the chunk\n",
                                ctx->attn_gqa_mode,
                                (want >= 0 && want < SOFUU_GQA_N) ? k_sofuu_gqa[want].name : "?",
                                why, ratio);
                    }
                }
            }
        } else if (ctx->attn_gqa_only >= 0) {
            // The mask already holds only the pinned index, so pick() is the
            // availability check: a pinned-but-nil shape is reported below.
            idx = sofuu_attn_gqa_pick_ex(ratio, ctx->attn_gqa_mask, ctx->attn_gqa_all);
        } else {
            idx = sofuu_attn_gqa_pick_ex(ratio, ctx->attn_gqa_mask, ctx->attn_gqa_all);
        }
        if (idx == -2) {
            // LOUD: the table HAS forms for this ratio and not one is usable.
            // Silently running the per-head kernel here is what hid a nil
            // pipeline for the whole shape table; say it once per ratio.
            if (ratio < 64u && (ctx->attn_gqa_reported & (1ull << ratio)) == 0u) {
                ctx->attn_gqa_reported |= (1ull << ratio);
                char why[1024];
                sofuu_gqa_why_ratio(ctx, ratio, why, sizeof(why));
                fprintf(stderr,
                        "[metal] attn GQA ratio %u: none of the table's forms is usable (%s) - per-head kernel owns the chunk\n",
                        ratio, why);
            }
            idx = -1;
        }
        // The last gate before dispatch: a live pipeline whose OWN reported
        // maxTotalThreadsPerThreadgroup covers the threadgroup. Metal drops an
        // oversized threadgroup without an error and the output reads back as
        // stale attention, so this must never be skipped.
        if (idx >= 0 && idx < SOFUU_GQA_N && ctx->pso_attn_gqa[idx] != nil
            && k_sofuu_gqa[idx].ratio == ratio
            && ctx->attn_gqa_max_tg[idx] >= k_sofuu_gqa[idx].threads) {
            g_sofuu_gqa_last = idx;
            pso = ctx->pso_attn_gqa[idx];
            grid_x = nkv;
            rg = ctx->attn_gqa_rg[idx];
            nthr = ctx->attn_gqa_threads[idx];
            // SOFUU_ATTN_GQA_TRACE=1 names every (ratio, shape) pair the model
            // actually runs, once each. This is the only external receipt that
            // a paired shape dispatched: the LOGITS cannot show it, because the
            // paired kernel is bit-identical to the per-head kernel by
            // construction (same reduction order, only the K/V loads hoisted).
            static int trace = -1;
            if (trace < 0) trace = getenv("SOFUU_ATTN_GQA_TRACE") ? 1 : 0;
            if (trace && (ctx->attn_gqa_traced & (1ull << idx)) == 0u) {
                ctx->attn_gqa_traced |= (1ull << idx);
                fprintf(stderr,
                        "[metal] attn GQA ratio %u -> %s (rg=%u, %u threads/group, pipeline allows %u)\n",
                        ratio, k_sofuu_gqa[idx].name, rg, nthr,
                        ctx->attn_gqa_max_tg[idx]);
            }
        }
    }
    if (pso == nil) return 0;
    uint row_groups = (rows + rg - 1u) / rg;
    // ── split-K (flash-decoding) selection ──────────────────────────────────
    // SOFUU_ATTN_SPLIT_K=S cuts the key range into up to S slices, one
    // threadgroup each (grid z), when the chunk is long enough to pay for the
    // partial scratch and the combine pass. 0/unset keeps the shipped kernel
    // bit-for-bit (the default until this arm is proven faster); 1 routes the
    // shipped shape through the split path with a single slice, which the
    // combine reproduces EXACTLY (w = exp(0) = 1, one term) and is therefore
    // the bit-identity receipt for the whole split plumbing.
    //
    // The slice index rides grid z so the threadgroup keeps the shipped
    // 512-thread shape: the hardware 1024-thread limit is never approached and
    // each entry's own maxTotalThreadsPerThreadgroup was checked at init.
    static int sk_max_env = -2;
    if (sk_max_env == -2) {
        sk_max_env = 0;
        if (const char *e = getenv("SOFUU_ATTN_SPLIT_K")) {
            int m = atoi(e);
            if (m < 0) m = 0;
            if (m > 8) m = 8;          // scratch stays <= 8.7 MB at 128 rows
            sk_max_env = m;
        }
    }
    uint nsplit = 1u;
    // The split kernel is the twin of table entry 1 (r2/nr2/rg32) and of
    // nothing else, so the split arm follows the shape that was actually
    // selected - not the mode number, which is "auto" by default.
    const bool sk_shape = (g_sofuu_gqa_last == 1 && ratio == 2u);
    if (sk_max_env > 0 && sk_shape && ctx->pso_attn_gqa_sk != nil) {
        id<MTLBuffer> pb = (__bridge id<MTLBuffer>)pos_buf_ptr;
        const uint32_t* pp = (const uint32_t*)((const char*)pb.contents + pos_offset);
        // The chunk's own length bounds the useful split count: a slice must
        // still carry >= 32 keys, which is also the widest causal overhang a
        // row group of RG keys can waste on the mask.
        uint32_t tmax_chunk = pp[rows - 1u] + 1u;
        uint32_t want = tmax_chunk / 32u;
        if (want > (uint32_t)sk_max_env) want = (uint32_t)sk_max_env;
        if (want > 8u) want = 8u;
        if (want < 1u) want = 1u;
        nsplit = want;
    }
    uint rec = hd + 4u;
    if (nsplit > 1u) {
        size_t need = (size_t)nsplit * (size_t)rn * (size_t)nh * (size_t)rec * sizeof(float);
        if (ctx->attn_sk_scratch == nil || ctx->attn_sk_cap < need) {
            id<MTLBuffer> nb = [ctx->device newBufferWithLength:(need + need / 2)
                                                       options:MTLResourceStorageModePrivate];
            if (nb == nil) {
                nsplit = 1u;           // no scratch -> the shipped kernel
            } else {
                ctx->attn_sk_scratch = nb;
                ctx->attn_sk_cap = need + need / 2;
            }
        }
    }
    // SOFUU_ATTN_DUP=N re-issues the (idempotent) attention dispatch N extra
    // times inside the same encoder. Nothing this function produces depends on
    // how many times it ran - both variants overwrite out/scratch
    // unconditionally - so the total-time delta between N=0 and N=1 is the
    // marginal cost of ONE attention dispatch measured in situ, with the real
    // grid, cache state and neighbouring dispatches. That receipt stays inside
    // +/-2% under machine load where wall-clock section subtraction does not.
    static int attn_dup = -2;
    if (attn_dup == -2) {
        attn_dup = 0;
        if (const char *d = getenv("SOFUU_ATTN_DUP")) {
            int v = atoi(d);
            if (v > 0) attn_dup = (v > 8) ? 8 : v;
        }
    }
    for (int rep = 0; rep <= attn_dup; ++rep) {
        if (nsplit > 1u) {
            uint nsp = nsplit;
            [ctx->batch_enc setComputePipelineState:ctx->pso_attn_gqa_sk];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
            [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:5];
            [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:6];
            [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:7];
            [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:8];
            [ctx->batch_enc setBytes:&stride length:sizeof(uint) atIndex:9];
            [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:10];
            [ctx->batch_enc setBuffer:ctx->attn_sk_scratch offset:0 atIndex:11];
            [ctx->batch_enc setBytes:&nsp length:sizeof(uint) atIndex:12];
            [ctx->batch_enc setBytes:&rec length:sizeof(uint) atIndex:13];
            [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid_x, row_groups, nsplit)
                            threadsPerThreadgroup:MTLSizeMake(nthr, 1, 1)];
            // Split -> combine is a hard producer/consumer dependency, so it
            // gets an unconditional buffer barrier (sofuu_batch_barrier honors
            // the SOFUU_NO_BARRIER / SOFUU_BARRIER_EVERY ablation knobs, and a
            // skipped barrier here reads as "attention returned zeros").
            [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [ctx->batch_enc setComputePipelineState:ctx->pso_attn_sk_combine];
            [ctx->batch_enc setBuffer:ctx->attn_sk_scratch offset:0 atIndex:0];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:2];
            [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:3];
            [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:4];
            [ctx->batch_enc setBytes:&stride length:sizeof(uint) atIndex:5];
            [ctx->batch_enc setBytes:&nsp length:sizeof(uint) atIndex:6];
            [ctx->batch_enc setBytes:&rec length:sizeof(uint) atIndex:7];
            [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:8];
            [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:9];
            [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rn, nh, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [ctx->batch_enc setComputePipelineState:pso];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
            [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:4];
            [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:5];
            [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:6];
            [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:7];
            [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:8];
            [ctx->batch_enc setBytes:&stride length:sizeof(uint) atIndex:9];
            [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:10];
            [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(grid_x, row_groups, 1)
                            threadsPerThreadgroup:MTLSizeMake(nthr, 1, 1)];
        }
        if (rep != attn_dup) { [ctx->batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    }
    return 1;
}

// Same contract, but the shipped whole-row kernel (llama_attn_pre_w32): one
// threadgroup per (row, head), 32 warps splitting the key range. Kept as the A/B
// arm of SOFUU_CHAIN_ATTN: it streams the K/V rows*n_heads times instead of
// rows/PAR_ROWS, so it is the slower of the two on a 128-row chunk, but it is
// the kernel that was validated at every position and it also exercises the 2D
// grid (rows*n_heads > 1024 threadgroups) the 128-row chunk needs.
int sofuu_metal_batch_llama_attn_w32_out(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t out_stride
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->pso_llama_attn_pre_w32 == nil) return 0;
    if (!q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr || !pos_buf_ptr) return 0;
    if (!sofuu_chain_attn_gate(ctx, pos_buf_ptr, pos_offset, rows, head_dim, n_heads,
                               n_kv_heads, max_seq, out_stride)) return 0;
    sofuu_batch_barrier(ctx);
    uint rpt = 1u;                 // the kernel takes a whole row; ABI slack
    uint n_chunks_max = 1u;        // ditto (both are (void)-cast in the kernel)
    uint hd = head_dim, nh = n_heads, nkv = n_kv_heads, ms = max_seq;
    uint stride = out_stride, rn = rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_pre_w32];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:5];
    [ctx->batch_enc setBytes:&hd length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&nh length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&nkv length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&ms length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&stride length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&rn length:sizeof(uint) atIndex:12];
    [ctx->batch_enc setBytes:&rpt length:sizeof(uint) atIndex:13];
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(rows * n_heads)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return 1;
}

void sofuu_metal_batch_llama_attn_fin_b4(
    void* ctx_ptr,
    void* meta_buf_ptr, size_t meta_offset,
    void* out_buf_ptr, size_t out_offset,
    void* pos_buf_ptr, size_t pos_offset,
    uint32_t rows,
    uint32_t n_heads, uint32_t head_dim,
    uint32_t n_chunks_max, uint32_t meta_stride
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !meta_buf_ptr || !out_buf_ptr) return;
    if (ctx->pso_llama_attn_fin_b4 == nil) return;
    sofuu_batch_barrier(ctx);
    // The copy only matches when the part pass took the whole-row fast path, so
    // ask it rather than re-deriving the gate here (it cannot be re-derived
    // faithfully: n_kv_heads is not a parameter of this function).
    const int fin_window_ok =
        (ctx->pre_window_active != 0 && ctx->pso_llama_attn_copy_b4 != nil && rows > 0u) ? 1 : 0;
    if (fin_window_ok != 0) {
        [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_copy_b4];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:2];
        [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:3];
        [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:4];
        [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:5];
        [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:6];
        [ctx->batch_enc dispatchThreadgroups:sf_grid2d(rows * n_heads)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return;
    }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_fin_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)meta_buf_ptr offset:meta_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)pos_buf_ptr offset:pos_offset atIndex:2];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&n_chunks_max length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&meta_stride length:sizeof(uint) atIndex:6];
    uint grid = rows * n_heads;
    [ctx->batch_enc dispatchThreadgroups:sf_grid2d(grid)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU2.1: RMSNorm with F32 weights (GGUF norm tensors). Weight bound at a
// 256-byte-aligned offset with the element remainder passed as a constant.
int sofuu_metal_batch_rmsnorm_f32w(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL) / 4;   // element remainder (f32)
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm_f32w];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// B-row RMSNorm (spec chain): rows contiguous [B][dim], grid = B threadgroups.
// Returns 1 when the dispatch was enqueued, 0 when the pipeline is missing or
// the grid would exceed the kernel's single-dimension threadgroup index (the
// caller then falls back to the per-row dispatch, never to a silent skip).
int sofuu_metal_batch_rmsnorm_f32w_b4(
    void* ctx_ptr,
    void* x_buf_ptr, size_t x_offset,
    void* w_buf_ptr, size_t w_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t dim, float eps, size_t rows
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !x_buf_ptr || !w_buf_ptr || !y_buf_ptr || dim == 0 || rows == 0) return 0;
    if (ctx->pso_rmsnorm_f32w_b4 == nil) return 0;
    // the kernel reads threadgroup_position_in_grid.x only: more than 1024 rows
    // would be silently truncated, so refuse rather than norm half the chunk.
    if (rows > 1024u) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL) / 4;
    uint d = (uint)dim;
    uint r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_rmsnorm_f32w_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:6];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(r, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// Per-kv-head: optional f32 head-RMSNorm, split-half (mode 0) or interleaved
// (mode 1) RoPE on k, append k(roped)/v(raw) at `pos` of this layer's slice.
int sofuu_metal_batch_llama_kv_write(
    void* ctx_ptr,
    void* k_buf_ptr, size_t k_offset,
    void* v_buf_ptr, size_t v_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* kln_buf_ptr, size_t kln_offset,
    uint32_t has_norm,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs,
    uint32_t ring
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !k_buf_ptr || !v_buf_ptr || !kc_buf_ptr || !vc_buf_ptr) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = kln_buf_ptr ? (kln_offset & ~255ULL) : 0;
    uint w_rem = kln_buf_ptr ? (uint)(kln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_kv_write];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)k_buf_ptr offset:k_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)v_buf_ptr offset:v_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:3];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(kln_buf_ptr ? kln_buf_ptr : k_buf_ptr)
                      offset: aligned_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    if (ctx->attn_noop == 1 || ctx->attn_noop == 3) { pos = 0xFFFFFFFFu; }
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:11];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:13];
    sofuu_bind_rope(ctx, 14, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, k_buf_ptr);
    // KV4 ring mode: 0 = shipped absolute row addressing.
    [ctx->batch_enc setBytes:&ring length:sizeof(uint) atIndex:20];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

// Per-q-head GQA decode attention with optional f32 q-norm + RoPE, softmax
// over the full cache prefix, weighted V into head_out.
// Decode attention with the warp-cooperative kernel. Same signature and bindings as
// sofuu_metal_batch_llama_attn, one threadgroup of 32 per head.
extern "C" int sofuu_metal_batch_llama_attn_fast(
    void* ctx_ptr, void* q_ptr, size_t q_offset,
    void* kc_ptr, size_t kc_offset, void* vc_ptr, size_t vc_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t p,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_llama_attn_fast == nil) return 0;
    if (p >= 4096u) return 0;
    sofuu_batch_barrier(ctx);
    uint w_rem = (uint)(qln_offset & 255ULL) / 4u;
    if (ctx->attn_noop) { p = 0xFFFFFFFFu; }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_fast];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_ptr offset:kc_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_ptr offset:vc_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_ptr offset:(qln_offset & ~255ULL) atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&p length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}

extern "C" int sofuu_metal_batch_llama_attn_w4(
    void* ctx_ptr, void* q_ptr, size_t q_offset,
    void* kc_ptr, size_t kc_offset, void* vc_ptr, size_t vc_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t p,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_llama_attn_w4 == nil) return 0;
    if (p >= 4096u || head_dim > 256u) return 0;
    sofuu_batch_barrier(ctx);
    uint w_rem = (uint)(qln_offset & 255ULL) / 4u;
    if (ctx->attn_noop) { p = 0xFFFFFFFFu; }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_w4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_ptr offset:kc_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_ptr offset:vc_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_ptr offset:(qln_offset & ~255ULL) atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&p length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}


extern "C" int sofuu_metal_batch_llama_attn_w8(
    void* ctx_ptr, void* q_ptr, size_t q_offset,
    void* kc_ptr, size_t kc_offset, void* vc_ptr, size_t vc_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t p,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_llama_attn_w8 == nil) return 0;
    if (p >= 4096u || head_dim > 256u) return 0;
    sofuu_batch_barrier(ctx);
    uint w_rem = (uint)(qln_offset & 255ULL) / 4u;
    if (ctx->attn_noop) { p = 0xFFFFFFFFu; }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_w8];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_ptr offset:kc_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_ptr offset:vc_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_ptr offset:(qln_offset & ~255ULL) atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&p length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}


extern "C" int sofuu_metal_batch_llama_attn_w16(
    void* ctx_ptr, void* q_ptr, size_t q_offset,
    void* kc_ptr, size_t kc_offset, void* vc_ptr, size_t vc_offset,
    void* qln_ptr, size_t qln_offset,
    void* out_ptr, size_t out_offset,
    uint32_t has_norm, uint32_t p,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_llama_attn_w16 == nil) return 0;
    if (p >= 4096u || head_dim > 256u) return 0;
    sofuu_batch_barrier(ctx);
    uint w_rem = (uint)(qln_offset & 255ULL) / 4u;
    if (ctx->attn_noop) { p = 0xFFFFFFFFu; }
    // One threadgroup per query head, 32 warps deep (1024 threads). Decode attention
    // here is bound by resident warps, not by K/V bandwidth: the 16-warp kernel ran
    // the batch at 14.2 ms and this one at 10.3 ms, which puts the K/V stream at
    // ~171 GB/s — this machine's measured streaming ceiling — so the remaining lever
    // there is gone. Wider heads fall through to the shipped 16-warp kernel
    // ("SOFUU_ATTN_W32=0" forces it back).
    if (sofuu_attn_w32_on() != 0 && ctx->pso_llama_attn_w32 != nil
        && head_dim <= 128u && n_kv != 0u && n_heads % n_kv == 0u) {
        uint w_rem = (uint)(qln_offset & 255ULL) / 4u;
        [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_w32];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_ptr offset:kc_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_ptr offset:vc_offset atIndex:2];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_ptr offset:(qln_offset & ~255ULL) atIndex:3];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
        [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
        [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
        [ctx->batch_enc setBytes:&p length:sizeof(uint) atIndex:7];
        [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
        [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
        [ctx->batch_enc setBytes:&n_kv length:sizeof(uint) atIndex:10];
        [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
        [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
        [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
        [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
        sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                        rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                        head_dim, rope_n_rot, rope_mscale, q_ptr);
        [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        return 1;
    }
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn_w16];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_ptr offset:kc_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_ptr offset:vc_offset atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)qln_ptr offset:(qln_offset & ~255ULL) atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    [ctx->batch_enc setBytes:&p length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    return 1;
}


void sofuu_metal_batch_llama_attn(
    void* ctx_ptr,
    void* q_buf_ptr, size_t q_offset,
    void* kc_buf_ptr, size_t kc_layer_offset,
    void* vc_buf_ptr, size_t vc_layer_offset,
    void* qln_buf_ptr, size_t qln_offset,
    void* out_buf_ptr, size_t out_offset,
    uint32_t has_norm,
    uint32_t pos,
    uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_seq,
    float eps, float rope_base, uint32_t rope_mode,
    void* rope_tab_ptr, size_t rope_tab_offset, uint32_t has_table,
    uint32_t rope_n_rot, float rope_mscale,
    void* rope_adj_ptr, size_t rope_adj_offset, uint32_t rope_n_pairs
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !q_buf_ptr || !kc_buf_ptr || !vc_buf_ptr || !out_buf_ptr) return;
    sofuu_batch_barrier(ctx);
    size_t aligned_offset = qln_buf_ptr ? (qln_offset & ~255ULL) : 0;
    uint w_rem = qln_buf_ptr ? (uint)(qln_offset & 255ULL) / 4 : 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_llama_attn];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)q_buf_ptr offset:q_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)kc_buf_ptr offset:kc_layer_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)vc_buf_ptr offset:vc_layer_offset atIndex:2];
    [ctx->batch_enc setBuffer: (__bridge id<MTLBuffer>)(qln_buf_ptr ? qln_buf_ptr : q_buf_ptr)
                      offset: aligned_offset atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf_ptr offset:out_offset atIndex:4];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&has_norm length:sizeof(uint) atIndex:6];
    if (ctx->attn_noop == 1 || ctx->attn_noop == 2) { pos = 0xFFFFFFFFu; }
    [ctx->batch_enc setBytes:&pos length:sizeof(uint) atIndex:7];
    [ctx->batch_enc setBytes:&head_dim length:sizeof(uint) atIndex:8];
    [ctx->batch_enc setBytes:&n_heads length:sizeof(uint) atIndex:9];
    [ctx->batch_enc setBytes:&n_kv_heads length:sizeof(uint) atIndex:10];
    [ctx->batch_enc setBytes:&max_seq length:sizeof(uint) atIndex:11];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:12];
    [ctx->batch_enc setBytes:&rope_base length:sizeof(float) atIndex:13];
    [ctx->batch_enc setBytes:&rope_mode length:sizeof(uint) atIndex:14];
    sofuu_bind_rope(ctx, 15, rope_tab_ptr, rope_tab_offset, has_table,
                    rope_adj_ptr, rope_adj_offset, rope_n_pairs,
                    head_dim, rope_n_rot, rope_mscale, q_buf_ptr);
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// GPU3: batched (B=4) q4_K GEMV — token-major x[4][cols] -> y[4][rows].
void sofuu_metal_batch_gguf_b4(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    if (ctx->pso_gemv_q4_k_b4 == nil) return;
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    uint r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_b4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// G2: GGUF dequant-dot GEMV, one op in the live batch. ty is the GGML type
// id (2 = q4_0, 8 = q8_0, 12 = q4_K, 14 = q6_K). Bindings: 0=w 1=x 2=y
// 3=cols 4=w_rem.
int sofuu_metal_batch_gguf(
    void* ctx_ptr,
    uint32_t ty,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return 0;
    // Decode (b=1) q4_K GEMV: one whole 144-byte block per thread. This is the
    // measured winner over the shipped per-row kernel ("SOFUU_GEMV_B1=0" restores
    // that one, which is also the fallback for any shape this branch declines).
    if (ty == 12 && sofuu_gemv_b1_arm() != 0 && ctx->pso_gemv_q4_k_b1blk != nil
        && cols % 256 == 0 && cols <= 4096 && (w_offset & 15) == 0) {
        // SOFUU_GEMV_B1: 6 (default) = gemv_q4_k_b1v2, the shipped decode kernel;
        // 1/4 = gemv_q4_k_b1blk, the kernel it replaces, kept as the A/B
        // reference for SOFUU_GEMV_BENCH_AB; 0 = the older per-row c2 path.
        int arm = sofuu_gemv_b1_arm();
        id<MTLComputePipelineState> kp = (arm == 6) ? ctx->pso_gemv_q4_k_b1v2
                                                    : ctx->pso_gemv_q4_k_b1blk;
        if (kp == nil) { kp = ctx->pso_gemv_q4_k_b1blk; }
        uint nb = (uint)(cols >> 8);
        uint rpt = 256u / nb;               // whole rows this threadgroup covers
        sofuu_batch_barrier(ctx);
        size_t aligned = w_offset & ~255ULL;
        uint w_rem = (uint)(w_offset & 255ULL);
        uint c = (uint)cols, r = (uint)rows;
        [ctx->batch_enc setComputePipelineState:kp];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
        [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
        [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
        [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
        [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
        [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + rpt - 1) / rpt, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return 1;
    }
    id<MTLComputePipelineState> pso = nil;
    switch (ty) {
        case 8:  pso = ctx->pso_gemv_q8_0; break;
        case 2:  pso = ctx->pso_gemv_q4_0; break;
        case 12: pso = ctx->pso_gemv_q4_k; break;
        case 14: pso = ctx->pso_gemv_q6_k; break;
        default: return 0;
    }
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;
    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}


// ── G0: real multi-op batching ─────────────────────────────────────────────
// batch_open retains ONE uncommitted command buffer with a LIVE encoder;
// every batch_gemv appends an op reading caller-owned MTLBuffers (so ops can
// chain y→x with zero CPU staging); batch_end ends the encoder, commits once,
// waits. One GPU submission per batch, independent of op count.
// GPU-frame capture state (env-gated, shared between batch_open/batch_end;
// single-threaded engine — no sync needed)
static NSString *g_cap_path = nil;
static int g_cap_skip = -1, g_cap_n = 1, g_cap_cycles = 0;
static bool g_cap_active = false;

// Temporary instrumentation: measure achievable read bandwidth over nbytes of the
// weight buffer using the same 144-byte block pattern the q4_K GEMV walks, at a
// caller-chosen thread count and dispatch count inside ONE commit.
extern "C" float sofuu_metal_bw_probe(void* ctx_ptr, void* w_buf_ptr, size_t nbytes,
                                      size_t threads, size_t iters, size_t lane_count, size_t tgsz, size_t mode, size_t slice_blocks) {
    if (!ctx_ptr || !w_buf_ptr || nbytes == 0 || iters == 0 || threads == 0) return 0.0f;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->pso_bw_probe_q4k == nil) { fprintf(stderr, "[bw-probe] pso nil\n"); return 0.0f; }
    uint nb = (uint)(nbytes / 144);
    id<MTLBuffer> obuf = [ctx->device newBufferWithLength:threads * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    if (obuf == nil) { fprintf(stderr, "[bw-probe] obuf nil\n"); return 0.0f; }
    if (sofuu_metal_batch_open(ctx_ptr) != 0) { fprintf(stderr, "[bw-probe] open failed\n"); return 0.0f; }
    [ctx->batch_enc setComputePipelineState:ctx->pso_bw_probe_q4k];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:obuf offset:0 atIndex:1];
    [ctx->batch_enc setBytes:&nb length:sizeof(uint) atIndex:2];
    uint lanes = lane_count;
    [ctx->batch_enc setBytes:&lanes length:sizeof(uint) atIndex:3];
    uint sl = (uint)(slice_blocks ? slice_blocks : nb);
    uint db = sl;   // stride must equal the slice, or dispatches overlap and hit cache
    [ctx->batch_enc setBytes:&sl length:sizeof(uint) atIndex:6];
    uint md = (uint)mode;
    [ctx->batch_enc setBytes:&md length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:0 atIndex:5];   // x stands in for w
    size_t tpt = tgsz ? tgsz : (threads < 1024 ? threads : 1024);
    if (tpt > 1024) tpt = 1024;
    size_t ntgs = (threads + tpt - 1) / tpt;           // round up: never launch fewer
    MTLSize tg = MTLSizeMake(tpt, 1, 1);
    MTLSize grid = MTLSizeMake(ntgs, 1, 1);
    for (size_t it = 0; it < iters; ++it) {
        uint start = (uint)((db ? (it * db) : 0) % (nb ? nb : 1));
        [ctx->batch_enc setBytes:&start length:sizeof(uint) atIndex:7];
        [ctx->batch_enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    auto t0 = std::chrono::steady_clock::now();
    int ok = sofuu_metal_batch_end(ctx_ptr);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) { fprintf(stderr, "[bw-probe] end failed\n"); return 0.0f; }
    double secs = std::chrono::duration<double>(t1 - t0).count();
    if (secs <= 0.0) return 0.0f;
    return (float)((double)sl * 144.0 * (double)iters / 1e9 / secs);
}


// One dispatch per group of independent projections (q/k/v, gate/up). secs is a
// host array of {w_off, y_off, rows} triples, passed by value.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_secs(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_base,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr,
    size_t cols,
    const uint32_t* secs, size_t n_sec
) {
    if (!ctx_ptr || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_gemv_q4_k_secs == nil) return 0;
    if (cols == 0 || cols % 256 != 0 || !secs || n_sec == 0 || n_sec > 8) return 0;
    uint32_t max_rows = 0;
    for (size_t s = 0; s < n_sec; ++s) {
        if (secs[s * 3 + 2] > max_rows) max_rows = secs[s * 3 + 2];
    }
    if (max_rows == 0) return 0;
    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_secs];
    uint c = (uint)cols;
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:secs length:3 * sizeof(uint32_t) * n_sec atIndex:4];
    unsigned long long wb = (unsigned long long)w_base;
    [ctx->batch_enc setBytes:&wb length:sizeof(unsigned long long) atIndex:5];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(max_rows, n_sec, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}


// 8 rows per threadgroup q4_K GEMV: same lane mapping as the shipped c2 kernel,
// one warp per row, so a projection launches 8x fewer threadgroups.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_r8(
    void* ctx_ptr, void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset, void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_gemv_q4_k_c2r8 == nil) return 0;
    if (!w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols, r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_c2r8];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    size_t ntgs = (rows + 7) / 8;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(ntgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}


// Split-K q4_K GEMV: pass 1 is a flat thread->block sweep (the access pattern a
// probe measured at 111-169 GB/s), pass 2 reduces the nb partials of each row.
// The partial buffer grows lazily and is reused across calls.
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_split(
    void* ctx_ptr, void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset, void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_gemv_q4_k_split1 == nil || ctx->pso_gemv_q4_k_split2 == nil) return 0;
    if (!w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0 || cols % 256 != 0) return 0;
    size_t nb = cols / 256;
    size_t need = rows * nb * sizeof(float);
    if (ctx->gemv_partial == nil || ctx->gemv_partial.length < need) {
        ctx->gemv_partial = [ctx->device newBufferWithLength:need options:MTLResourceStorageModePrivate];
        if (ctx->gemv_partial == nil) return 0;
    }
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols, r = (uint)rows, nbu = (uint)nb;
    size_t total = rows * nb;

    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_split1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:ctx->gemv_partial offset:0 atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    size_t tpt1 = 256;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((total + tpt1 - 1) / tpt1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tpt1, 1, 1)];

    sofuu_batch_barrier(ctx);
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_split2];
    [ctx->batch_enc setBuffer:ctx->gemv_partial offset:0 atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:1];
    [ctx->batch_enc setBytes:&nbu length:sizeof(uint) atIndex:2];
    size_t tpt2 = 128;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake((rows + tpt2 - 1) / tpt2, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tpt2, 1, 1)];
    (void)r;
    return 1;
}


// Residual add + RMSNorm in one dispatch: the pair runs every layer over the same
// 4 KB, and the dispatch is most of its cost.
extern "C" int sofuu_metal_batch_add_rmsnorm_f32w(
    void* ctx_ptr,
    void* x_buf, size_t x_off,
    void* r_buf, size_t r_off,
    void* xs_buf, size_t xs_off,
    void* y_buf, size_t y_off,
    void* w_buf, size_t w_off,
    size_t dim, float eps
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_add_rmsnorm_f32w == nil) return 0;
    if (!x_buf || !r_buf || !xs_buf || !y_buf || !w_buf || dim == 0) return 0;
    sofuu_batch_barrier(ctx);
    size_t wal = w_off & ~255ULL;
    uint w_rem = (uint)(w_off & 255ULL) / 4;   // element remainder (f32), like rmsnorm_f32w
    uint d = (uint)dim;
    [ctx->batch_enc setComputePipelineState:ctx->pso_add_rmsnorm_f32w];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf offset:x_off atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)r_buf offset:r_off atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)xs_buf offset:xs_off atIndex:2];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf offset:y_off atIndex:3];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf offset:wal atIndex:4];
    [ctx->batch_enc setBytes:&d length:sizeof(uint) atIndex:5];
    [ctx->batch_enc setBytes:&eps length:sizeof(float) atIndex:6];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:7];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return 1;
}


// Flat q4_K GEMV (one dispatch): 64 rows per threadgroup, one 144-byte block per
// thread, per-row reduction in threadgroup memory. Cols must be a multiple of 256
// and at most 12 blocks per row (3072 cols).
extern "C" int sofuu_metal_batch_gguf_gemv_q4k_flat(
    void* ctx_ptr, void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset, void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols
) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || ctx->pso_gemv_q4_k_flat == nil) return 0;
    if (!w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return 0;
    if (cols % 256 != 0 || cols / 256 > 12) return 0;
    sofuu_batch_barrier(ctx);
    size_t aligned = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols, r = (uint)rows;
    [ctx->batch_enc setComputePipelineState:ctx->pso_gemv_q4_k_flat];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc setBytes:&r length:sizeof(uint) atIndex:5];
    size_t ntgs = (rows + 63) / 64;
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(ntgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return 1;
}

int sofuu_metal_batch_open(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    sofuu_metal_sync(ctx_ptr);
    @autoreleasepool {
        // Env-gated GPU-frame capture for the interleaved-vs-standalone GEMV
        // attribute pass: SOFUU_METAL_CAPTURE=<out.gputrace>,
        // SOFUU_METAL_CAPTURE_SKIP=<cycles> and SOFUU_METAL_CAPTURE_N=<n>
        // (default 1) — skip the process's warmup cycles, then record n
        // batch_open..batch_end command buffers of the steady-state pass.
        if (g_cap_skip < 0) {
            const char* p = getenv("SOFUU_METAL_CAPTURE");
            const char* s = getenv("SOFUU_METAL_CAPTURE_SKIP");
            const char* n = getenv("SOFUU_METAL_CAPTURE_N");
            g_cap_n = n ? atoi(n) : 1;
            if (g_cap_n < 1) g_cap_n = 1;
            g_cap_skip = s ? atoi(s) : 0;
            g_cap_path = p ? [NSString stringWithUTF8String:p] : nil;
        }
        if (g_cap_path != nil && !g_cap_active && g_cap_cycles >= g_cap_skip) {
            MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
            if ([mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
                MTLCaptureDescriptor *d = [MTLCaptureDescriptor new];
                d.captureObject = ctx->device;
                d.destination = MTLCaptureDestinationGPUTraceDocument;
                d.outputURL = [NSURL fileURLWithPath:g_cap_path];
                NSError *err = nil;
                if ([mgr startCaptureWithDescriptor:d error:&err]) {
                    g_cap_active = true;
                    fprintf(stderr, "[metal] gpu capture STARTED (cycle %d, %s)\n",
                            g_cap_cycles, g_cap_path.UTF8String);
                } else {
                    fprintf(stderr, "[metal] gpu capture FAILED: %s\n",
                            err.localizedDescription.UTF8String ?: "(nil)");
                    g_cap_path = nil;
                }
            } else {
                fprintf(stderr, "[metal] gpu capture unsupported (no GPUTraceDocument) — needs Xcode runtime support\n");
                g_cap_path = nil;
            }
        }
        g_cap_cycles++;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        ctx->batch_enc = (__bridge id<MTLComputeCommandEncoder>)(void*)CFBridgingRetain([cmd computeCommandEncoder]);
        ctx->pending_cmd = (__bridge id<MTLCommandBuffer>)(void*)CFBridgingRetain(cmd);
        ctx->pending_y_dst = nullptr;
        ctx->pending_y_bytes = 0;
        if (ctx->batch_enc == nil) return -1;
    }
    return 0;
}

void sofuu_metal_batch_gemv(
    void* ctx_ptr,
    void* w_buf_ptr, size_t w_offset,
    void* x_buf_ptr, size_t x_offset,
    void* y_buf_ptr, size_t y_offset,
    size_t rows, size_t cols, int dtype
) {
    if (!ctx_ptr) return;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !w_buf_ptr || !x_buf_ptr || !y_buf_ptr || rows == 0 || cols == 0) return;
    // Cross-dispatch buffer visibility needs an explicit barrier; ops chain
    // y->x inside ONE encoder (one commit per batch).
    sofuu_batch_barrier(ctx);
    id<MTLComputePipelineState> pso = ctx->pso_gemv_bf16;
    if (dtype == 1) pso = ctx->pso_gemv_f16;
    else if (dtype == 2) pso = ctx->pso_gemv_f32;

    size_t aligned_offset = w_offset & ~255ULL;
    uint w_rem = (uint)(w_offset & 255ULL);
    uint c = (uint)cols;

    [ctx->batch_enc setComputePipelineState:pso];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)w_buf_ptr offset:aligned_offset atIndex:0];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)x_buf_ptr offset:x_offset atIndex:1];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)y_buf_ptr offset:y_offset atIndex:2];
    [ctx->batch_enc setBytes:&c length:sizeof(uint) atIndex:3];
    [ctx->batch_enc setBytes:&w_rem length:sizeof(uint) atIndex:4];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

int sofuu_metal_batch_end(void* ctx_ptr) {
    if (!ctx_ptr) return 1;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) { sofuu_metal_sync(ctx_ptr); return 1; }
    @autoreleasepool {
        [ctx->batch_enc endEncoding];
        CFRelease((__bridge CFTypeRef)ctx->batch_enc);
        ctx->batch_enc = nil;
        id<MTLCommandBuffer> cmd = ctx->pending_cmd;
        if (cmd != nil) {
            [cmd commit];
            bool ok = wait_and_report(cmd, "batch gemv");
            ctx->pending_cmd = nil;
            CFRelease((__bridge CFTypeRef)cmd);
            // 0 = the GPU rejected the buffer: outputs are untouched — the
            // caller MUST treat this pass as not-happened and fall back
            // (reading the slab after a failed buffer is how garbage
            // KV/logits reached the CPU repack on the 1B under load).
            return ok ? 1 : 0;
        }
        // paired with the startCapture in batch_open: a captured cycle ends
        // exactly at its batch_end so the trace holds whole passes
        if (g_cap_active) {
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            g_cap_active = false;
            fprintf(stderr, "[metal] gpu capture STOPPED (cycle %d)\n",
                    g_cap_cycles - 1);
        }
    }
    return 1;
}

// W1 (CPU+GPU row split): commit the open batch WITHOUT waiting for it. This
// is the producer/consumer half of the split: the host commits the GPU's rows
// for a layer group and immediately goes back to running the CPU's rows on the
// host, so the two arms share one wall-clock window instead of serialising.
// The command buffer is retained in ctx->inflight until
// sofuu_metal_batch_wait_all (or sofuu_metal_sync) drains it. Returns 1 when a
// buffer was committed, 0 when there was nothing open.
int sofuu_metal_batch_commit_async(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil) return 0;
    [ctx->batch_enc endEncoding];
    CFRelease((__bridge CFTypeRef)ctx->batch_enc);
    ctx->batch_enc = nil;
    id<MTLCommandBuffer> cmd = ctx->pending_cmd;
    ctx->pending_cmd = nil;
    ctx->pending_y_dst = nullptr;
    ctx->pending_y_bytes = 0;
    if (cmd == nil) return 0;
    const int cap = (int)(sizeof(ctx->inflight) / sizeof(ctx->inflight[0]));
    if (ctx->inflight_n >= cap) {
        // Safety valve: never grow the in-flight set without bound. Draining is
        // always safe here - the caller has already encoded everything it needs.
        sofuu_metal_batch_wait_all(ctx_ptr);
    }
    [cmd commit];
    ctx->inflight[ctx->inflight_n++] = (void*)CFBridgingRetain(cmd);
    CFRelease((__bridge CFTypeRef)cmd);
    return 1;
}

// W1: open the next batch encoder on a NEW command buffer while the buffers
// committed above are still executing. The queue is serial, so the new batch
// simply executes after them - no wait, no ordering surprise.
int sofuu_metal_batch_open_async(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc != nil) return -1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        ctx->batch_enc = (__bridge id<MTLComputeCommandEncoder>)(void*)CFBridgingRetain([cmd computeCommandEncoder]);
        ctx->pending_cmd = (__bridge id<MTLCommandBuffer>)(void*)CFBridgingRetain(cmd);
        ctx->pending_y_dst = nullptr;
        ctx->pending_y_bytes = 0;
        if (ctx->batch_enc == nil) return -1;
    }
    return 0;
}

// W1: drain every command buffer committed through
// sofuu_metal_batch_commit_async. Returns the number that did NOT complete
// successfully (a rejected buffer leaves its outputs untouched, so the caller
// must treat the whole split pass as not-happened and fall back to one arm).
int sofuu_metal_batch_wait_all(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    int bad = 0;
    for (int i = 0; i < ctx->inflight_n; ++i) {
        id<MTLCommandBuffer> c = (__bridge id<MTLCommandBuffer>)ctx->inflight[i];
        if (c != nil) {
            if (!wait_and_report(c, "split layer group")) bad++;
            CFRelease(ctx->inflight[i]);
        }
        ctx->inflight[i] = nullptr;
    }
    ctx->inflight_n = 0;
    return bad;
}

#else

// Stubs for non-macOS platforms (Linux, Android, Windows)
bool sofuu_metal_available(void) { return false; }
int sofuu_metal_batch_open(void* ctx) { (void)ctx; return -1; }
void sofuu_metal_batch_gemv(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, int d) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)d; }
int sofuu_metal_batch_end(void* ctx) { (void)ctx; return 0; }
int sofuu_metal_batch_commit_async(void* ctx) { (void)ctx; return 0; }
int sofuu_metal_batch_open_async(void* ctx) { (void)ctx; return -1; }
int sofuu_metal_batch_wait_all(void* ctx) { (void)ctx; return 0; }
void sofuu_metal_batch_mlx4(void* ctx, void* w, size_t wo, void* s, size_t so, void* b, size_t bo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t g, int bf) { (void)ctx;(void)w;(void)wo;(void)s;(void)so;(void)b;(void)bo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)g;(void)bf; }
void sofuu_metal_batch_gemv_staged(void* ctx, void* w, size_t wo, const float* x, void* y, size_t yo, size_t r, size_t c, int d) { (void)ctx;(void)w;(void)wo;(void)x;(void)y;(void)yo;(void)r;(void)c;(void)d; }
int sofuu_metal_batch_add(void* ctx, void* acc, size_t ao, void* add, size_t ado, size_t n) { (void)ctx;(void)acc;(void)ao;(void)add;(void)ado;(void)n; return 0; }
int sofuu_metal_batch_add_chan_bias(void* ctx, void* y, void* bias, const uint32_t* secs, size_t nsec, size_t b) { (void)ctx;(void)y;(void)bias;(void)secs;(void)nsec;(void)b; return 0; }
int sofuu_metal_batch_strided_copy_f32(void* ctx, void* d, size_t dob, size_t ds, void* s, size_t sob, size_t ss, size_t n, size_t r) { (void)ctx;(void)d;(void)dob;(void)ds;(void)s;(void)sob;(void)ss;(void)n;(void)r; return 0; }
int sofuu_metal_batch_resid_split_add_f32(void* ctx, void* y, size_t yo, void* pa, size_t pao, size_t na, void* pb, size_t pbo, size_t nb, size_t c, size_t r) { (void)ctx;(void)y;(void)yo;(void)pa;(void)pao;(void)na;(void)pb;(void)pbo;(void)nb;(void)c;(void)r; return 0; }
int sofuu_metal_batch_swiglu(void* ctx, void* g, size_t go, void* u, size_t uo, size_t n) { (void)ctx;(void)g;(void)go;(void)u;(void)uo;(void)n; return 0; }
void sofuu_metal_batch_rmsnorm(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps; }
void sofuu_metal_batch_conv_gate(void* ctx, void* bcx, size_t bco, void* cw, size_t cwo, void* st, size_t sto, void* out, size_t oo, size_t dim) { (void)ctx;(void)bcx;(void)bco;(void)cw;(void)cwo;(void)st;(void)sto;(void)out;(void)oo;(void)dim; }
void sofuu_metal_batch_kv_write(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t p, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)p;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb; }
void sofuu_metal_batch_attn(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* out, size_t oo, uint32_t p, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)out;(void)oo;(void)p;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb; }
int sofuu_metal_batch_rmsnorm_f32w(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps; return 0; }
int sofuu_metal_batch_rmsnorm_f32w_b4(void* ctx, void* x, size_t xo, void* w, size_t wo, void* y, size_t yo, size_t dim, float eps, size_t rows) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)y;(void)yo;(void)dim;(void)eps;(void)rows; return 0; }
int sofuu_metal_batch_llama_kv_write(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t hn, uint32_t p, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm, void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)hn;(void)p;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np; return 0; }
void sofuu_metal_batch_llama_attn(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* out, size_t oo, uint32_t hn, uint32_t p, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm, void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)out;(void)oo;(void)hn;(void)p;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np; }
int sofuu_metal_batch_head_rmsnorm(void* ctx, void* x, size_t xo, void* w, size_t wo, uint32_t hd, uint32_t nh, uint32_t hn, uint32_t p, float e, float rb, uint32_t rm, void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)hd;(void)nh;(void)hn;(void)p;(void)e;(void)rb;(void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np; return 0; }
int sofuu_metal_batch_llama_attn_part(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* qln, size_t qlo, void* meta, size_t mo, uint32_t hn, uint32_t p, uint32_t hd2, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)qln;(void)qlo;(void)meta;(void)mo;(void)hn;(void)p;(void)hd2;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm; return 0; }
int sofuu_metal_batch_llama_attn_fin(void* ctx, void* meta, size_t mo, void* out, size_t oo, uint32_t p, uint32_t nh, uint32_t hd) { (void)ctx;(void)meta;(void)mo;(void)out;(void)oo;(void)p;(void)nh;(void)hd; return 0; }
void sofuu_metal_batch_llama_kv_write_b4(void* ctx, void* k, size_t ko, void* v, size_t vo, void* kc, size_t kco, void* vc, size_t vco, void* kln, size_t klo, uint32_t hn, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm, void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np) { (void)ctx;(void)k;(void)ko;(void)v;(void)vo;(void)kc;(void)kco;(void)vc;(void)vco;(void)kln;(void)klo;(void)hn;(void)pb;(void)po;(void)rows;(void)hd;(void)nkv;(void)ms;(void)e;(void)rb;(void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np; }
void sofuu_metal_batch_head_rmsnorm_b4(void* ctx, void* x, size_t xo, void* w, size_t wo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t hn, float e, float rb, uint32_t rm, void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np) { (void)ctx;(void)x;(void)xo;(void)w;(void)wo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)hn;(void)e;(void)rb;(void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np; }
void sofuu_metal_batch_llama_attn_part_b4(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* meta, size_t mo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t ncm, uint32_t mstr) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)meta;(void)mo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)ncm;(void)mstr; }
void sofuu_metal_batch_llama_attn_fin_b4(void* ctx, void* meta, size_t mo, void* out, size_t oo, void* pb, size_t po, uint32_t rows, uint32_t nh, uint32_t hd, uint32_t ncm, uint32_t mstr) { (void)ctx;(void)meta;(void)mo;(void)out;(void)oo;(void)pb;(void)po;(void)rows;(void)nh;(void)hd;(void)ncm;(void)mstr; }
int sofuu_metal_batch_llama_attn_1c_b4(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* out, size_t oo, void* pb, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)out;(void)oo;(void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms; return 0; }
int sofuu_metal_batch_gguf_b4r(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, void* acc, size_t ao, size_t r, size_t c, size_t ad) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)acc;(void)ao;(void)r;(void)c;(void)ad; return 0; }
int sofuu_metal_batch_gguf(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c; return 0; }
void sofuu_metal_batch_gguf_b4(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c; }
int sofuu_metal_batch_gguf_gemm(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_gemm_tiled(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_gemm_mma(void* ctx, void* w, size_t wo, void* x, size_t xo, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)w;(void)wo;(void)x;(void)xo;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_llama_attn_flash_b4(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* o, size_t oo, void* p, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t ht) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)o;(void)oo;(void)p;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)ht; return 0; }
int sofuu_metal_batch_llama_attn_row(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* o, size_t oo, void* p, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t os) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)o;(void)oo;(void)p;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)os; return 0; }
int sofuu_metal_batch_llama_attn_w32_out(void* ctx, void* q, size_t qo, void* kc, size_t kco, void* vc, size_t vco, void* o, size_t oo, void* p, size_t po, uint32_t rows, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t os) { (void)ctx;(void)q;(void)qo;(void)kc;(void)kco;(void)vc;(void)vco;(void)o;(void)oo;(void)p;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)os; return 0; }
int sofuu_metal_batch_gguf_gemm_tiled_multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t yo, size_t rt, size_t c, size_t B, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)yo;(void)rt;(void)c;(void)B;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf_gemm_tiled_acc(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* res, size_t ro, size_t rs, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)res;(void)ro;(void)rs;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_gemm_multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t yo, size_t rt, size_t c, size_t B, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)yo;(void)rt;(void)c;(void)B;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf_gemm_acc(void* ctx, uint32_t ty, void* w, size_t wo, void* x, size_t xo, void* res, size_t ro, size_t rs, void* y, size_t yo, size_t r, size_t c, size_t B) { (void)ctx;(void)ty;(void)w;(void)wo;(void)x;(void)xo;(void)res;(void)ro;(void)rs;(void)y;(void)yo;(void)r;(void)c;(void)B; return 0; }
int sofuu_metal_batch_gguf_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf6_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf8_b4r4(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
int sofuu_metal_batch_gguf8_b4multi(void* ctx, uint32_t ty, void* w, size_t wbo, void* x, size_t xo, void* y, size_t rt, size_t c, const uint32_t* secs, size_t ns) { (void)ctx;(void)ty;(void)w;(void)wbo;(void)x;(void)xo;(void)y;(void)rt;(void)c;(void)secs;(void)ns; return 0; }
void* sofuu_metal_init(void) { return NULL; }
void sofuu_metal_free(void* ctx) {}
void* sofuu_metal_buffer_create_zerocopy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void* sofuu_metal_buffer_create(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void* sofuu_metal_buffer_create_copy(void* ctx, const void* ptr, size_t size_bytes) { return NULL; }
void sofuu_metal_buffer_free(void* buf) {}
void sofuu_metal_gemv_bf16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f16(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_gemv_f32(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols) {}
void sofuu_metal_rmsnorm(void* ctx, const float* x, void* w_buf, size_t w_offset, float* y, size_t dim, float eps) {}
void sofuu_metal_swiglu(void* ctx, float* g, const float* u, size_t size) {}
void sofuu_metal_conv1d_3(void* ctx, void* conv_w_buf, size_t conv_w_offset, const float* state, float* out, size_t channels) {}
void sofuu_metal_gemv_mlx4(void* ctx, void* w_buf, size_t w_offset, void* scales_buf, size_t scales_offset, void* biases_buf, size_t biases_offset, const float* x, float* y, size_t rows, size_t cols, size_t group_size, int is_bf16) {}
void sofuu_metal_gemv_qkv_multi(void* ctx, void* q_buf, size_t q_offset, size_t q_rows, void* k_buf, size_t k_offset, size_t k_rows, void* v_buf, size_t v_offset, size_t v_rows, const float* x, float* q_out, float* k_out, float* v_out, size_t cols, int dtype) {}
void sofuu_metal_gemv_gate_up_multi(void* ctx, void* gate_buf, size_t gate_offset, void* up_buf, size_t up_offset, const float* x, float* gate_out, float* up_out, size_t inter_rows, size_t cols, int dtype) {}
void sofuu_metal_gemv_async(void* ctx, void* w_buf, size_t w_offset, const float* x, float* y, size_t rows, size_t cols, int dtype) {}
void sofuu_metal_sync(void* ctx) {}
void sofuu_metal_gemv_qkv_mlx4(void* ctx, void* q_buf, size_t q_offset, size_t q_rows, void* q_scales_buf, size_t q_scales_offset, void* q_biases_buf, size_t q_biases_offset, void* k_buf, size_t k_offset, size_t k_rows, void* k_scales_buf, size_t k_scales_offset, void* k_biases_buf, size_t k_biases_offset, void* v_buf, size_t v_offset, size_t v_rows, void* v_scales_buf, size_t v_scales_offset, void* v_biases_buf, size_t v_biases_offset, const float* x, float* q_out, float* k_out, float* v_out, size_t cols, size_t group_size, int is_bf16) {}
void sofuu_metal_gemv_gate_up_mlx4(void* ctx, void* gate_buf, size_t gate_offset, size_t gate_rows, void* gate_scales_buf, size_t gate_scales_offset, void* gate_biases_buf, size_t gate_biases_offset, void* up_buf, size_t up_offset, size_t up_rows, void* up_scales_buf, size_t up_scales_offset, void* up_biases_buf, size_t up_biases_offset, const float* x, float* gate_out, float* up_out, size_t cols, size_t group_size, int is_bf16) {}


// KV4: no Metal, so the class is unavailable (the Rust side keeps f32 KV).
int sofuu_metal_kv4_support(void* ctx, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t bits) {
    (void)ctx;(void)hd;(void)nh;(void)nkv;(void)bits; return 0;
}
int sofuu_metal_batch_kv4_quant(void* ctx, void* kr, size_t kro, void* vr, size_t vro,
    void* qk, size_t qko, void* qks, size_t qkso, void* qkm, size_t qkmo,
    void* qv, size_t qvo, void* qvs, size_t qvso, void* qvm, size_t qvmo,
    uint32_t hd, uint32_t nkv, uint32_t ring, uint32_t nblk, uint32_t b0, uint32_t nb, uint32_t bits) {
    (void)ctx;(void)kr;(void)kro;(void)vr;(void)vro;(void)qk;(void)qko;(void)qks;(void)qkso;
    (void)qkm;(void)qkmo;(void)qv;(void)qvo;(void)qvs;(void)qvso;(void)qvm;(void)qvmo;
    (void)hd;(void)nkv;(void)ring;(void)nblk;(void)b0;(void)nb;(void)bits; return 0;
}
int sofuu_metal_batch_llama_attn_w32_kv4(void* ctx, void* q, size_t qo, void* kr, size_t kro,
    void* vr, size_t vro, void* qln, size_t qlno, void* out, size_t outo, uint32_t hn, uint32_t pos,
    uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm,
    void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np,
    void* qk, size_t qko, void* qks, size_t qkso, void* qkm, size_t qkmo,
    void* qv, size_t qvo, void* qvs, size_t qvso, void* qvm, size_t qvmo,
    uint32_t nblk, uint32_t ring, uint32_t bits) {
    (void)ctx;(void)q;(void)qo;(void)kr;(void)kro;(void)vr;(void)vro;(void)qln;(void)qlno;
    (void)out;(void)outo;(void)hn;(void)pos;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;
    (void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np;
    (void)qk;(void)qko;(void)qks;(void)qkso;(void)qkm;(void)qkmo;(void)qv;(void)qvo;(void)qvs;
    (void)qvso;(void)qvm;(void)qvmo;(void)nblk;(void)ring;(void)bits; return 0;
}
int sofuu_metal_batch_llama_attn_pre_w32_kv4(void* ctx, void* q, size_t qo, void* kr, size_t kro,
    void* vr, size_t vro, void* out, size_t outo, void* pb, size_t po, uint32_t rows, uint32_t hd,
    uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t os,
    void* qk, size_t qko, void* qks, size_t qkso, void* qkm, size_t qkmo,
    void* qv, size_t qvo, void* qvs, size_t qvso, void* qvm, size_t qvmo,
    uint32_t nblk, uint32_t ring, uint32_t bits) {
    (void)ctx;(void)q;(void)qo;(void)kr;(void)kro;(void)vr;(void)vro;(void)out;(void)outo;
    (void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)os;
    (void)qk;(void)qko;(void)qks;(void)qkso;(void)qkm;(void)qkmo;(void)qv;(void)qvo;(void)qvs;
    (void)qvso;(void)qvm;(void)qvmo;(void)nblk;(void)ring;(void)bits; return 0;
}
int sofuu_metal_batch_llama_attn_w32_kv4g(void* ctx, void* q, size_t qo, void* kr, size_t kro,
    void* vr, size_t vro, void* qln, size_t qlno, void* out, size_t outo, uint32_t hn, uint32_t pos,
    uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t ms, float e, float rb, uint32_t rm,
    void* rt, size_t rto, uint32_t ht, uint32_t nr, float msc, void* ra, size_t rao, uint32_t np,
    void* qk, size_t qko, void* qks, size_t qkso, void* qkm, size_t qkmo,
    void* qv, size_t qvo, void* qvs, size_t qvso, void* qvm, size_t qvmo,
    uint32_t nblk, uint32_t ring, uint32_t bits) {
    (void)ctx;(void)q;(void)qo;(void)kr;(void)kro;(void)vr;(void)vro;(void)qln;(void)qlno;
    (void)out;(void)outo;(void)hn;(void)pos;(void)hd;(void)nh;(void)nkv;(void)ms;(void)e;(void)rb;
    (void)rm;(void)rt;(void)rto;(void)ht;(void)nr;(void)msc;(void)ra;(void)rao;(void)np;
    (void)qk;(void)qko;(void)qks;(void)qkso;(void)qkm;(void)qkmo;(void)qv;(void)qvo;(void)qvs;
    (void)qvso;(void)qvm;(void)qvmo;(void)nblk;(void)ring;(void)bits; return 0;
}
int sofuu_metal_batch_llama_attn_pre_row_kv4(void* ctx, void* q, size_t qo, void* kr, size_t kro,
    void* vr, size_t vro, void* out, size_t outo, void* pb, size_t po, uint32_t rows, uint32_t hd,
    uint32_t nh, uint32_t nkv, uint32_t ms, uint32_t os,
    void* qk, size_t qko, void* qks, size_t qkso, void* qkm, size_t qkmo,
    void* qv, size_t qvo, void* qvs, size_t qvso, void* qvm, size_t qvmo,
    uint32_t nblk, uint32_t ring, uint32_t bits) {
    (void)ctx;(void)q;(void)qo;(void)kr;(void)kro;(void)vr;(void)vro;(void)out;(void)outo;
    (void)pb;(void)po;(void)rows;(void)hd;(void)nh;(void)nkv;(void)ms;(void)os;
    (void)qk;(void)qko;(void)qks;(void)qkso;(void)qkm;(void)qkmo;(void)qv;(void)qvo;(void)qvs;
    (void)qvso;(void)qvm;(void)qvmo;(void)nblk;(void)ring;(void)bits; return 0;
}

#endif

extern "C" int sofuu_metal_peak_fma(void* ctx_ptr, void* out_buf, size_t n_blocks) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !out_buf || !ctx->pso_peak_fma) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_peak_fma];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

extern "C" int sofuu_metal_peak_fma16(void* ctx_ptr, void* out_buf, size_t n_blocks) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !out_buf || !ctx->pso_peak_fma16) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_peak_fma16];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

extern "C" int sofuu_metal_peak_mma(void* ctx_ptr, void* out_buf, size_t n_blocks) {
    if (!ctx_ptr) return 0;
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->batch_enc == nil || !out_buf || !ctx->pso_peak_mma) return 0;
    [ctx->batch_enc setComputePipelineState:ctx->pso_peak_mma];
    [ctx->batch_enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
    [ctx->batch_enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return 1;
}

// LEDGER #164 probe: CPU-visible contents pointer of a Metal buffer (shared storage).
extern "C" void* sofuu_metal_buffer_contents(void* buf) {
    if (!buf) return NULL;
    return [(id<MTLBuffer>)buf contents];
}
