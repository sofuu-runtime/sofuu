# PLAN-GPU-METAL — full GPU support for the Sofuu engine

**Status: EXECUTION IN PROGRESS (2026-10-16).** Done so far:
G0 ✓ GREEN & PUSHED (80c81254f) — `sofuu_metal_batch_open/gemv/batch_gemv_staged/batch_end`
(real command batching; 32 chained bf16 GEMVs in ONE commit = 10.77×, 431→40 µs/op,
bit-exact at every step; the earlier "visibility bug" was a test slot-map error, retracted).
G1 primitive ✓ (eb956c128) — `batch_mlx4` quantized GEMV op with bindings mirrored from the
per-op kernel (w/s/b aligned offsets + rems[4]); Rust `MetalEngine::batch_mlx4`.

**GPU1 wave (2026-10-16, this session) — whole-decode-step batches + G2 kernels. LANDED
(pending commit):**
- **G1-client ✓ lfm2 whole-step decode** (`infer/lfm2.rs` `Lfm2GpuDecode` +
  `forward_gpu`, `SOFUU_GPU_DECODE=1`, macOS): EVERY op of a decode token — RMSNorms,
  gated short conv (new `lfm2_conv_gate`), KV-write + GQA attention (new
  `lfm2_kv_write` + `lfm2_attn`, threadgroup scratch scores, head-RMSNorm+RoPE folded
  in), SwiGLU, all GEMVs — enqueued into ONE command buffer over page-aligned
  activation/KV slabs wrapped zero-copy (`register_zerocopy_only` refuses a copy
  fallback: CPU and GPU must share pages). CPU does the embedding row + sampler.
  CPU does the embedding row + sampler. Parity receipt
  (`gpu_decode_logits_match_cpu_rms`, SOFUU_TEST_LFM2_DIR=liquid-1.3B): 24 random-token
  steps, greedy top-1 identical through step 17; one near-tie flip at step 18 under
  accumulated f32-order residue; logits RMS 0.05-0.15 vs CPU (per-op-hybrid tolerance
  class), gate < 0.2 PASS.
- **Kernel fleet in `src/metal/kernels.metal`** (16 kernels total): slot-based
  `batch_rmsnorm`/`batch_swiglu` reuse (elementwise kernels hardened with `n`
  guards so grid rounding cannot spill into the next slab slot), `add_inplace`
  (residual), `lfm2_conv_gate`, `lfm2_kv_write`, `lfm2_attn`, plus **G2 GGUF
  dequant-dot GEMVs `gemv_q8_0` / `gemv_q4_0` / `gemv_q4_k` / `gemv_q6_k`** wired
  through `sofuu_metal_batch_gguf` (ty = GGML id: 2/8/12/14). G2 parity pinned:
  q8_0 & q4_0 & q6_K **bit-exact** vs the CPU vec_dot; q4_K 3.3e-7 (f32 order).
  q4_K/q6_K engine specifics are mirrored from the CPU kernels, NOT from upstream
  ggml: this engine's q4_K places LO nibbles of the 32 quants in values 0..31 and
  HI nibbles in 32..63 (q advances 32 bytes per 64 values), and q6_K reads its f16
  d at offset 208 with the interleaved scale order sc[8n + l/16 + {0,2,4,6}].
  Q6_K_XL verified against the MiniCPM5-2B file empirically (the file decodes
  coherently through the CPU block_q6_K path = same layout the kernel mirrors).
- **The measured wall (WHY the 1.3× decode bar stops here):** on this M2 Pro the
  batched whole-step GPU decode runs liquid at ~28 t/s vs CPU ~33-35 t/s. Attribution
  (temp knobs, since removed): +1.08 GB/token of FFN GEMV work = +22.2 ms/s/token →
  **~49 GB/s GEMV stream rate, linear in bytes**; per-op barriers ≈ free (+30 ops ≈ 0);
  lm_head (268 MB bf16 read) ≈ 2 ms; kernel shape changes (uint4 512B bursts, div-free
  group indexing, 8-rows-per-256-thread threadgroups, group-major order) **all land at
  the same ~45-50 GB/s**; driver-managed copied weights (vs wrapped mmap) identical.
  The CPU path decodes the same bytes at the same rate (4 threads, ~46 GB/s) — decode
  here is DRAM-stream-bound and both engines sit at the same wall. G0's win is banked
  (dispatch: per-op hybrid 7.7-10.9 t/s → one batch per token 28-29 t/s), but
  **≤20 ms/token on liquid is unreachable and the plan's stop rule (dispatch/wall
  fallback = CPU decode on this Mac) FIRES for lfm2**. SOFUU_GPU_DECODE stays an
  opt-in, correct (parity-verified) path; the G4 flip table below carries the
  per-model verdicts.
- **GPU2 wave (2026-10-16, same day) — GGUF llama decode client (`model.rs`)**: the
  seven per-layer GEMVs (q/k/v, attn_output, gate/up/down) run as THREE Metal
  commits per layer (`[q,k,v]`, `[attn_output]`, `[gate, up, GPU-swiglu, down]`)
  plus a fourth for the lm_head, over a page-aligned zero-copy slab. Norms,
  QK-norms, RoPE (both interleave modes), CPU KV cache and attention are shared
  verbatim with the CPU path (`batched_attention_ext`, biases, qk_norm all keep
  working); KIVI/dynamic-FFN/streaming fall back to CPU per token. Eligible
  dtypes q4_0/q8_0/q4_K/q6_K — arm fails LOUD otherwise; `--cpu` honored via
  `from_gguf_with_gpu`. Fleet receipts over the storm tail (load 13-39):
  GPU>CPU on ALL FOUR GGUF models in equal windows: 0.6B 14.1 vs 9.1,
  Llama-1B 20.7 vs 10.3, MiniCPM-2B q6_K 6.4 vs 3.6, Qwen3-8B 3.4 vs 1.2 t/s —
  calibration against the older calm-window CPU rows still pending a clean
  battery; the streaming-wall receipt for the WHOLE-STEP design does not apply
  here (this is the fewer-commits hybrid, CPU attention intact).
- **GPU2.1 wave (2026-10-18) — ONE commit per layer for the llama/GGUF shape**:
  three new kernels (`rmsnorm_f32w` f32-weight RMSNorm, `llama_kv_write`
  per-kv-head QK-norm + RoPE (split-half AND interleaved) + KV append,
  `llama_attn` per-q-head GQA decode attention over the full cache prefix)
  close the CPU glue inside the batch: Norms read f32 weights straight from
  the weight mmap (256-byte-aligned bind + element-remainder constant), the
  resume-flags (`has_norm`, `rope_mode`) are bind-time constants, and q/k/norm
  biases gate at arm (any bias / sliding window / head_dim>128 / f16 norms /
  misaligned offset → the 3-commit client, loudly labeled in the arm line).
  The KV cache is no longer duplicated: `k_cache`/`v_cache` are allocated
  BEFORE `arm()` and registered zero-copy, so GPU kernels and the CPU
  fallback read and write the SAME memory (576MB saved + fallback coherence
  by construction; 8B = 36 layers x 8 kv heads x 4096 x 128 f32 x 2).
  `layers_one_commit` runs 14 batch ops per layer behind ONE
  batch_open/batch_end (was 3 commits + host round trips); x and the KV
  caches stay command-buffer-resident across layers, x syncs back once per
  token.
  Receipts (2026-10-18, window load 6.3 — opencode peer on 74% CPU, TAINTED):
  Qwen3-8B-q4_K_M armed "ONE commit/layer" (36 layers, act 0MB, kv-shared
  576MB); temperature-0 decode output is IDENTICAL to the `--cpu` run on the
  same prompt ("The loader measures panels on all cores...") — first
  end-to-end GPU-vs-CPU text parity receipt for the llama shape including the
  new QK-norm/RoPE-fused attention kernels. Llama-3.2-1B-q8_0 through the same
  one-commit path: 27.7 t/s decode, 243 t/s prefill. 8B decode stayed ~2.1 t/s
  in the tainted window — same as the 3-commit client — which LOCALIZES the
  remaining 8B cost to the GEMV kernels' streaming bandwidth itself
  (one-row-per-simdgroup 4B scalar loads => ~10 GB/s effective on the ~5GB
  weight stream), NOT to commit waits or the new attention path: commit-wait
  elimination was necessary but not sufficient, and the panel-GEMV rewrite
  (2-8 rows per threadgroup, 16B loads, llama.cpp-class) is now the measured
  #1 lever for 8B.
- **GPU3 wave A (2026-10-18, whole-token + attribution, ~uncommitted→commit)**:
  the WHOLE TOKEN now runs in ONE command buffer (embed + 36 layers + final
  norm + lm_head; CPU waits once; SOFUU_GPU_LAYER_COMMIT=1 restores per-layer
  waits) — measured effect on 8B decode: ZERO (2.03 vs 2.09 t/s), text parity
  kept. Combined with a raw-operations ablation matrix (SOFUU_GPU_SKIP="attn,
  ffn,head" — timing-only) and three in-GPU micro-benches (op-overhead chain,
  real-layer mimic with ablations, weights geometry incl. file-backed vs
  anon), the 8B per-token budget is now ATTRIBUTED at stable-window
  (profile: host_encode 0.2-0.3ms, GPU busy ~264ms/token):
    FFN gate/up/swiglu/down  ~123 ms  (q4_K GEMVs at ~25-27 GB/s = expected)
    attention (kv_write + per-head attn) ~85 ms  (KV reads ~375MB -> kernel
        runs at ~4 GB/s effective: one 32-lane threadgroup per head walking
        1,270 positions serially = LATENCY-BOUND, the standout defect)
    head (lm_head q6_K 510MB) ~25-30 ms;  q/k/v + o-proj ~23-25 ms;
    norms/adds/swiglu glue ~0-5 ms;  commit waits ~0 (proved by 36->1 = no-op)
  Micro-bench cross-checks: chained q4_K at the FFN shape runs 38 GB/s
  in-batch; a compact-buffer clone of a whole layer (same seven GEMVs) sums
  to ~102-115 ms/token vs the client's 264 — closed per-op/geometry/file-
  backing hypotheses (all equivalent warm); _so the 264-vs-115 gap remains
  UNEXPLAINED_ and is the next diagnostic after the attention fix.
  Chunked attention (head_rmsnorm+rope dispatch, then per (head, 128-position
  chunk) partial softmax+V to a per-layer meta region, then a rescale finish
  kernel) PASSES an isolated real-shape parity test (hd128, 32/8 heads,
  pos 1300: max_err 3.75e-2 vs the CPU two-pass softmax) but produces GARBAGE
  in the full client (old serial kernel in the same client = correct text) —
  shipped default-OFF at first; the push-review MAJOR finding 'chunked
  attention metadata overlaps' (kernels.metal) then exposed the actual bug:
  head 0's partial-V region started at meta[0], the same bytes as the
  {max,sum} block — head 0's own writes corrupted its softmax metadata
  (tolerance-invisible in the isolated test, amplified to garbage in the
  client). Fix = partial region begins after the heads×chunks×2 pairs; text
  parity restored; chunked attention is now DEFAULT and measures 85 -> ~49
  ms/token at ctx ~1.3k (7.7 GB/s KV-effective — pass-3 still latency-bound;
  further t-chunk x d-split is the next attention step).
  q4_K GEMV variant #7 (aligned uint4 loads + fully static unrolls +
  algebraic scale fold, still 1-row/32-lane) REMOVED after A/B: bandwidth
  tied the scalar kernel (18.7-20.0 vs 19.8-20.8 GB/s same-window) AND it
  cannot pass the bit-exact q4_K gate (the fold regroups f32 addition order)
  — 7 designs now bracket the same ~20 GB/s per-threadgroup streaming
  plateau; the kernel-class breakthrough (llama.cpp references ~100+ GB/s)
  remains an open craft problem and gate = bench_gguf_gemv_bandwidth.
  G3 wave B (2026-10-19, SHIPPED & PARITY-BIT-EXACT): the B=4 batched q4_K GEMV —
  the kernel that serves BOTH prefill (weight stream amortized over prompt tokens)
  and spec verification — now passes gguf_b4_matches_cpu at max_rel=0.000e0 (4
  tokens x 8 rows, cols=1024, receipted oracle = quant::vec_dot_raw_q4_k) and the
  test is un-ignored (permanent suite gate). Root cause of the FALSE row-
  invariance alarm was twofold, NOT kernel math: (1) the kernel used
  [[thread_position_in_grid]] = FLAT global id (tg*32+lane), so guarded writes
  scattered at 32x stride — a guarded double-probe (slot0=1000, slot32=1032
  stamped, slots 1-7 quiet) pinned the semantics and the fix is the plain
  [[threadgroup_position_in_grid]] every shipped GEMV uses; (2) the test's hand
  CPU reference folded ALL groups with sc=1 while the crafted scale bytes decode
  subgroups {2,3,6,7} to d=0 — replaced with the receipted vec_dot_raw_q4_k
  oracle (v1 sanity row: max_rel 0.0 too). A v1 cross-check prints b4/v1/cpu row
  tri-agreement (74.5). NEXT: wire spec verify (4 tokens/pass) through the
  one-commit client, then the prefill chunk path (b4 serves it too), then the
  calm two-pass battery. Old kernel remains the default. Everything is
  behind SOFUU_GPU_DECODE=1 as before; CPU-only mode untouched.
  G4 wave (2026-10-19 evening) — spec verify + prefill chunk fully WIRED,
  receipts honest about the economics:
  - B4 KERNEL unrolled: a shared-l-loop rewrite (nibble decoded once, 4
    accumulators, per-token f32 order IDENTICAL to v1) measured 7.5ms
    until #pragma clang loop unroll(full) — the compiler silently refused
    to unroll the fat body; with the pragma b4 = 2.30ms vs v1 x4 = 2.99ms
    same-window (~49 vs ~38 GB/s effective) — real amortization, still
    bit-exact vs the vec_dot_raw_q4_k oracle. Float-floor nibble decode
    measured catastrophic (7.5ms) — integer &15/>>4 only. The unroll
    lesson extends the MSL array: runtime-indexed local arrays, uchar16,
    AND fat loops without an explicit unroll pragma all silently cost 2x+.
  - GPU chain-spec verify (forward_speculative_tree route): root + width-1
    chain (SOFUU_SPEC_DEPTH<=3) in ONE commit per pass; b4 GEMVs; per-row
    attention (row t's kv_write lands before row t's attn — in-order
    barrier visibility = chain causality); row-0 head IN-commit; accepted
    rows get a LAZY GPU head commit (spec_chain_head_logits). The lm_head
    is ~514MB q6_K: 4 rows in-commit = 2GB/pass (measured worse), CPU
    matvec = ~290ms/row under load, lazy GPU row = ~13ms. mod.rs shapes
    the draft as the width-1 chain when gpu_spec_chain_armed(). Text
    token-identical vs GPU serial on 96-token receipts (two revisions).
  - Measured economics: spec 2.2 t/s vs serial 5.2 t/s at load 3-5.5 —
    the pass pays GEMV-for-GEMV (issue-bound plateau) + 4x op bubbles;
    battery #9 rows appended load-labeled (serial 1.22/1.44 vs spec
    1.38/1.41 decode, 16-tok serve window). The calm-window two-pass
    receipt is still the deciding artifact.
  - GPU prefill chunk route wired (forward_prefill_batch_chunk_internal,
    b<=4, head on the LAST row, SOFUU_GPU_PREFILL=1 opt-in): generate()'s
    prefill already prefers CPU tiles (chunks > 4 bail to them), so
    default fleet behavior is unchanged; engages with
    SOFUU_PREFILL_BATCH=4 + the env flag.
  - KERNEL-CLASS BREAKTHROUGH (2026-10-19 night, same-window receipts): the
    COALESCED-LANE q4_K class landed — gemv_q4_k_c (single) + gemv_q4_k_b4c
    (batched, lane l reads byte l of each 32B qs span = full-sector waves,
    per-lane partials + one simd_sum) and gemv_q4_k_b4c2 (SOFUU_Q4K_COALESCED=2:
    8 lanes x 16B uint4 spans cover a superblock, 4 superblocks per wavefront,
    component-parallel float4 x loads + float4 accumulators, one horizontal
    sum). FFN-tensor 4-token groups, same window: v1 x4 = 3.07ms/36.9 GB/s,
    b4c = 1.64ms/69.2, b4c2 = 1.13ms/100.0 GB/s — 2.7x the shipped class,
    parity 2.6e-7 vs vec_dot_raw_q4_k (q4k_coalesced_matches_cpu_and_bench,
    #[ignore], env-classed). The env flips BOTH single+batch PSOs at engine
    init so serial/spec/b4 share one class per run (text parity internal;
    bundled-metallib fallback degrades to v1 rather than silent no-ops).
    Live 8B at load ~4.9: serial 5.1 t/s, spec 3.3 t/s (text token-identical);
    the spec pass is now BUBBLE-bound (its GEMV mass is ~40ms at 100GB/s;
    the ~1600-op commit chain + per-row glue dominate) — next spec levers:
    batch the per-row elementwise ops across the 4 rows and a multi-row
    attention kernel.
  - G4 CHAIN-SPEC CLIENT (2026-10-19 late night): the spec pass runs root +
    <=3 candidates at consecutive positions in ONE commit — B=4 slot regions
    (x/xn/qa/ka/va/ao/xo/gate/up x4, logits xB, posb) with per-row position
    bounds in the attention kernels (chain causality), B-row glue
    (rmsnorm_f32w_b4; flat swiglu/add reused over n=B*rows), and LAZY per-row
    head commits (lm_head = ~514MB q6_K: a 4-row in-commit head = 2GB/pass —
    CPU matvec 290ms/row under load vs ~13ms lazy GPU row). Multi-row
    attention family (kv_write/head_rmsnorm/attn_part/attn_fin with a
    positions buffer parameter) parity 1.04e-7 against a garbage-prefilled
    cache (causality receipt).
  - PASS-WALL TRAJECTORY + FIRST SPEC>SERIAL CROSSOVER (all load ~3.7-4.9):
    pass wall 635 -> 332 -> 232ms (class matrix fix below); 96-token Count
    serial 5.3 vs spec 6.0 t/s — first net spec win; Paris natural text
    serial 11.1 / spec 4.6 (acceptance-limited on natural text, as on Pi 5).
    Battery #9: 12-14 rows, all load-labeled (calm pair 12.0 serial / 6.0
    spec at load ~3.7).
  - FUSION WAVE — MEASURED NEUTRAL UNDER LOAD, shipped behind a kill switch:
    llama_attn_1c_b4 (part+fin fused, pos<128, no meta) + gemv_q4_k_b4c2r
    (resid-add epilogue for o/down where rows==acc_dim, one writer per
    element), SOFUU_GPU_FUSE=0 disables. Same-window client pass: no delta
    beyond run noise — the "each dependent op pays the prior op's ~2.5ms
    barrier drain" model (true per-op in isolation) does NOT express in the
    client pass under load; window load governs. Fusion stays available but
    is not a lever on its own.
  - CLASS MATRIX CORRECTION + FIX (2026-10-19 night 5, commit 6c084e4c2):
    the "b4c2 = 100 GB/s" receipts above were MISLABELED — the bench engine
    ran at SOFUU_COALESCED=1, so those numbers belong to byte-lane b4c.
    TRUE same-window 2x2 (level x buffer): level-1 b4c compact 106.7 /
    in-5GB-mmap 50.1 GB/s; level-2 b4c2 compact 45.7 / in-5GB 37.3 — the
    vec4 batch variant is inferior everywhere. FIX: SOFUU_Q4K_COALESCED=2
    now means BEST-PER-SHAPE — serial single-row GEMVs use the vec4 class
    (gemv_q4_k_c2: 12.0-12.4 t/s serial stands on it), spec-pass batched
    GEMVs use byte-lane b4c (pass wall 332 -> 232ms). Also REVERTED with
    receipt: a 5GB anonymous weight copy to dodge mmap collapses (load
    +20-30s, no live delta — serial c2-single does not suffer the collapse;
    batched kernels still measure ~24 GB/s in the real 5GB space vs 93-104
    compact — the OPEN SPEC-PASS ANOMALY, needs a Metal GPU-frame capture
    to attribute the client pass's remaining ~300ms-class overhead at calm).
  - MEASUREMENT HAZARDS (pinned): (1) every fresh process pays ~7-8s of GPU
    page-onboard for the 4.9GB weight space — runs shorter than ~60 tokens
    are contaminated; deciding receipts are 96-token. (2) zsh `env $E` does
    not word-split multi-var env strings -> stale invariance data RETRACTED;
    all later receipts use direct `env VAR=.. VAR2=.. cmd` forms. (3) window
    load swings standalone kernel rates 3-10x — only same-window A/Bs are
    legal evidence.
  - DECIDING CALM ARTIFACTS (armed 2026-10-19 ~21:00): watcher 1 waits for
    load < 2.5 then runs serial x2 + spec x2 96-token receipts on
    SOFUU_Q4K_COALESCED=2 into battery #9 (auto); watcher 2 fires behind it
    in the same window: ~2.6k-token long-prompt prefill receipts (CPU tile
    route + SOFUU_GPU_PREFILL=1 route — the input-side 80-150 t/s baseline)
    plus a SOFUU_SPEC_DEPTH=1/2/3 acceptance sweep on the chain route.
    Calm projections to beat if the gap to 40-50 persists: spec pass
    ~90-150ms -> spec ~10-14 t/s, serial ~15-20 t/s; the remaining gap is
    byte/stream class (read-all-weights is not a floor: sparsity/spec/MTP/
    KIVI per the order of battle).

- **Panel GEMV rewrite (2026-10-18) — ATTEMPTED, MEASURED DEAD END, not shipped**:
  four kernel variants were built, parity-verified and A/B'd against the shipped
  one-row-per-simdgroup scalar GEMVs with a new in-GPU bandwidth bench
  (`bench_gguf_gemv_bandwidth`, #[ignore], q4_K 12288x4096 = 28.3MB — the
  Qwen3-8B FFN shape): (1) 16B-area windows via uint4 + byte-skew extraction,
  4 rows/128-thread groups; (2) scalar weights + panel dispatch only; (3)
  scalar weights + float4-x via thread-array dynamic indexing; (4) float4-x
  with FULLY STATIC unrolled component access (no dynamic index at all).
  Same-window results: HEAD scalar 18.5-20.9 GB/s vs best variant 18.8, worst
  pairs 4.7-7.7 — every variant LOSES to HEAD; the shipped kernel is the
  measured local optimum on this GPU. Two transferable lessons: thread
  arrays/vec-components indexed by RUNTIME values spill to local memory and
  level-2-cache thrash (variants 1-3); only STATIC component access (variant
  4) avoids it, but plain scalar byte loads into L1 are already the better
  shape here. The storm window also shifted absolute numbers ~2x (4.7-28.3
  GB/s across the session at load 3-24), so same-window A/B is mandatory.
  Consequence for 8B: measured in-batch q4_K ~19-21 GB/s -> pure-GEMV floor
  ~215ms/token for the ~4.3GB weight stream -> ~4.6 t/s ceiling with THESE
  kernels; observed 2.1 t/s still holds ~2x of unexplained per-token cost —
  the next lever is measuring per-op time inside the real 14-op one-commit
  batch (norm/kv/attn/add ops, barriers, x staging), not wider loads. A
  llama.cpp-class hand-tuned multi-row kernel family (simd-shuffle scale
  sharing, static unrolls) remains an OPEN larger program with this bench as
  its gate.
Not landed, deferred with receipts: Spark/Qwen35/block_graph/gptx3 whole-step
clients (bespoke ports of the same pattern; block_graph already has a per-op
hybrid), ternary PTQ1_0/142 GPU kernels (CPU slab optimal per plan), and G3
prefill-on-GPU (GEMM-class, the one remaining plausible win). — same
  wall receipt applies to each (their CPU decodes already run 40-56 GB/s per the
  battery best-of table); G3 prefill-on-GPU (untouched): GEMM-class arithmetic
  intensity is the one place an iGPU can still beat AMX+dequant CPU prefill; needs
  its own tiled-kernel phase.

## FOCUSED PROGRAM (2026-10-18): flip Qwen3-8B first, then the fleet one by one

Focus model: Qwen3-8B Q4_K_M (charter model, worst gap: GPU 2.1 vs CPU 13.7 t/s calm).
Rule above the plan: every phase ends with same-window measurement + parity receipts;
no phase starts without the previous phase's numbers; failed attempts recorded.

- Phase 0 — baseline (DONE, receipts above): CPU 13.7 t/s; GPU 2.1; ~4.3GB/token; pure-GEMV
  floor ~215ms (20 GB/s kernels); measured ~470ms -> ~220ms unattributed.
- Phase 1 — stopwatch on a token (1 session): per-op timing inside the real one-commit
  batch (norms, qkv, kv_write, attn, o-proj, adds, swiglu, head, every batch_end wait).
  EXIT: every ms of the 470 attributed to ±10%; the 220ms split into commit-drains vs
  barriers vs small-ops vs staging.
- Phase 2 — ONE commit per TOKEN (1-2 sessions): port the proven lfm2 whole-step pattern
  to the llama shape (embed + 36 layers + lm_head in a single command buffer; CPU waits
  once). If Phase 1 shows barriers as the larger sink, rework slot rotation to drop
  redundant barriers in the same pass. EXIT: >= 4 t/s end-to-end AND temp-0 text still
  identical to CPU.
- Phase 3 — hand-tuned readers (bench-gated increments): llama.cpp-class MSL: static
  unrolls only (measured pitfall: runtime-indexed thread arrays spill to local memory),
  simd-shuffle scale sharing, multi-block per thread. Gate each step on
  bench_gguf_gemv_bandwidth: 20 -> 60 -> 100+ GB/s. EXIT: pure-GEMV floor < 90ms/token.
- Phase 4 — the flip (1 session): calm two-pass battery; GPU must beat 13.7 t/s.
  Projection after 2+3: ~8-12 t/s = parity-class, not the deep win; the deep win is the
  bandwidth arsenal layered on top of a GPU-primary path: KIVI 2-bit KV, speculative/MTP
  verify on GPU, dynamic-FFN sparsity — each its own battery receipt.
- Phase 5 — fleet roll-out, one model at a time, same recipe mostly config by then:
  Llama-3.2-1B Q8_0 (small loop, races CPU spec decode) -> Qwen-0.6B -> MiniCPM5-2B
  Q6_K_XL -> Spark-4B. 27B ternary last (resident-weights CPU path is a special optimum).
- Constraint that stands: CPU-only mode is never removed (fallback + battery baseline +
  the honest judge every GPU win must beat same-window).

Remaining: lfm2 decode client ✓ (see GPU1), gemma wiring, G2 GGUF MSL kernels ✓ (see GPU1),
G3 optional prefill, G4 per-model flip table + iOS check.
Companion to PLAN-CPU-DECODE-PEAK
(CPU-REMAINING backlog C1–C5 frozen). Builds directly on the 2026-10-15 GPU
survey: the Metal engine (src/metal/metal_engine.mm + infer/metal.rs) has
correct per-op kernels (gemv mlx4/bf16/f16/f32 + fused qkv/gate_up/gemv_multi)
used today by the decode-GEMV path, but `sofuu_metal_gemv_async` is
fake-async — it syncs on entry, allocates a fresh command buffer per matvec,
and stages x/y through shared internal buffers. Measured consequence: the
per-matvec hybrid decode lost to CPU (dispatch-bound), which is why CPU decode
is the current default.

Think of the target shape as: **one GPU submission per decode step, all model
GEMVs inside it, CPU only for the serial glue** — then, where kernels allow,
the whole graph on GPU.

## Rules (same as the CPU plans)
Calms receipts, env-gated `SOFUU_GPU_DECODE=1` / phase flags, default-on only
after receipts and only per model where it beats CPU ≥1.3× at BOTH short and
battery context; output gates: f32/bf16 within bench RMS tolerance, 4-bit
kernels bit-identical to the CPU dot kernels; peer files read-only; commits
straight to main.

## Phase G0 — real command-buffer batching (the unlock; scoped from survey)
- C side (`src/metal/metal_engine.mm`): `sofuu_metal_batch_begin(ctx)` opens a
  retained encoder; enqueue variants take CALLER-provided x/y staging (drop
  the shared-internal buffers — the hazard that forces per-call sync);
  `sofuu_metal_batch_end(ctx)` = commit + wait. Keep the old per-op path as
  the fallback.
- Rust (`infer/metal.rs`): `MetalDecodeBatch` handle — begin, per-layer
  `enqueue(gemv_*)` into it, `end()`. A small pool of MetalBuffer y-staging
  (one per in-flight op, cycled) sized at model load.
- First client: lfm2 decode (30 matvecs/token) — the CPU baseline to beat is
  10.9→30.6 tok/s-era numbers per the earlier samples; bar: shorten the
  33 ms token to ≤20 ms on liquid, else the phase stops with the receipt.
- Acceptance: one commit per token (assert via a submission counter in
  debug); output parity vs CPU decode within f32 tolerance; clean build on
  the AAR (no-NDK JNI path) untouched.

## Phase G1 — whole-decode-step graphs, per arch (model-agnostic handle)
- lfm2 first (proven kernels + my lane), then block_graph (gemma), then the
  GGUF llama backend (model.rs — coordinate with the kernel owner; decode
  currently runs C++ q4_K/q8_0 GEMVs on CPU at 85–87% of the wall, so the GPU
  bar there is high: only win if KV+glue amortize better in-graph).
- Structure: CPU does embedding lookup, norm folding, RoPE/index math, top-1
  + sampler; GPU does every GEMV + the SwiGLU pair math where fused kernels
  exist. One begin/end per step; zero per-op commits.
- Acceptance per arch: token-parity test vs CPU decode (RMS-gated), calm A/B,
  and the two-context battery rows recorded.

## Phase G2 — the missing GPU kernels (backend-coverage matrix)
Current coverage: mlx4/bf16/f16/f32 gemv ✓; fused qkv/gate_up ✓. Missing for
full "all models on GPU":
- q4_K / q6_K / q8_0 / q4_0 MSL dequant-dot gemv kernels (GGUF fleet: Spark,
  Qwen3-8B, MiniCPM5-2B Q6_K_XL — note Q6_K_XL may be an extension layout;
  verify against the file's header before writing the kernel).
- KV cache resident on GPU for "whole-graph" steps (zero-copy via unified
  memory; page the KV slab per context length at load).
- PTQ1_0/142 ternary kernels (last — the 27B's CPU slab path is the fallback
  and its arithmetic is cheap; MSL has no sdot, so a LUT/dequant kernel is a
  research item with the CPU slab as the acceptance baseline).
Each kernel lands bit-identical vs its CPU counterpart where the CPU path is
bit-stable, tolerance-gated elsewhere, with a pinned test per kernel.

## Phase G3 — prefill on GPU (optional, only after decode receipts)
Bar to pass: beat the CPU AMX batched prefill (104.5 tok/s liquid-class,
273 tok/s 0.6B, 27B 2.46 tok/s w/ direct tile) by ≥1.3× on at least two
models with bit/tolerance gates. Likely winners: mlx4/bf16 mid-size gemms at
PRE_CHUNK tiles on the iGPU while CPU serves other requests; skip for f32
models where AMX is already near the bus.

## Phase G4 — per-model flip table + fleet battery (GPU vs CPU decode)
- Battery runs each model twice (SOFUU_GPU_DECODE=0/1) in one JSONL pass;
  the result becomes the shipping default table (per MODEL, not global).
- iOS: same engine; verify on device only after desktop receipts (sim Metal
  quirks documented in ios-sim-metal memory; no-copy SIGTRAP workaround
  already coded around).

## Risks / stop points
- Dispatch overhead wins again: if G0's batched step doesn't beat CPU by the
  bar on liquid, STOP and re-scope (the fallback is CPU forever on this Mac —
  receipts say which).
- 16 GB pressure: GPU staging pools + KV slabs share the same RAM; measure
  wired/compressed before enabling per model; the residency flag and GPU
  staging must not stack on 27B-class files.
- Peer concurrent wave on src/simd: metal_engine.mm coordination before
  touching shared headers.
- Order of battle remains: G0 → G1(lfm2) → receipt gate → rest.
