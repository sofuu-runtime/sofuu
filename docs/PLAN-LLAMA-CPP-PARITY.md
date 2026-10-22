# PLAN — llama.cpp parity, every model (2026-10-20)

GOAL (user): on every model, Sofuu's decode AND prompt-eval t/s must equal llama.cpp
exactly ("not a bit less"), then improve later. The bar is llama.cpp's DEFAULT config —
the receipts the user sees in llama-server — NOT pure CPU: in the Homebrew ggml 0.26.0
build, llama-server/llama-bench offload to Metal by default, and that is where llama.cpp's
speed lives (0.6B prefill 3336 t/s Metal vs 243 CPU; Spark decode 40 vs 11).

## The bar + the scoreboard (battery #10, 2026-10-20, -p 210 -n 82, load-labeled)

Sweep ran under peer load 2.4-14.6 (Sofuu half at load ~6-10 — its rows are the
pessimistic ones; llama.cpp GPU rows degrade least). Same-window, so ratios are the
signal; a calm re-run is queued as the deciding table.

| model | llama.cpp default (pp/tg) | llama.cpp -ngl 0 CPU (pp/tg) | Sofuu now (pp CPU-tile / tg best) | gaps (pp/tg) |
|---|---|---|---|---|
| 0.6B q4_K_M | **3336 / 172.6** | 243 / 56.2 | 444 (load9) / 29.8-56 GPU | ~7x-10x / ~3x |
| 1B q8_0 | **1910 / 103.3** | 340 / 35.4 | 265 / 43.6-46 | ~7x / ~2.2x |
| Spark-4B q4_K_M | **395 / 40.1** | 41 / 11.0 | 51.7 / 20.8-21.3 | ~8x / ~1.9x |
| MiniCPM5-2B q6_K_XL | **860 / 68.0** | 98 / 17.9 | 95.7 / 6.8-24.9* | ~9x / ~2.7x* |
| 8B q4_K_M | **211 / 22.1** | 22 / 5.7 | 30.7 / 2.9@load8 (calm 13.6 CPU; GPU proj 15-20) | ~7x / ~1.5-1.6x |
| K2-Horizon-4B q8_0 | does not load (custom arch) | — | 62.1 / 5.0-5.4 | out of parity scope |
| Phi-4-mini q6_K_XL | 474 / 35.4 | 78 / 12.7 | FAILS TO LOAD (fused attn_qkv; loader wants separate q/k/v) | needs arch support |

*Sofuu MiniCPM best-known calm decode is 24.9 (battery #5); the 6.8 here is load 9.3.

User's own llama-server receipt (10 threads): 0.6B decode 148-150, prefill 1675-2312 t/s
— consistent with the bench above at higher load. User believes well-configured llama.cpp
reaches 200-300 decode on small models; the parity bar is therefore llama.cpp's BEST
config, and the -t 6 GPU row shows the same numbers as -t 10 (GPU-bound, threads barely
matter above ~6).

## Program order (leverage-sorted)

- **P1 — Metal batched prefill GEMM (the 7-10x arm).** Every model's pp gap is the
  largest number on the board. CPU tiles (128 tokens) cannot reach Metal batched GEMM.
  Build: tiled GEMM kernels (q4_K + q8_0 + q6_K dequant, 128-token tiles, 2D threadgrid),
  wired into forward_prefill_batch_chunk_internal behind SOFUU_GPU_PREFILL, one commit per
  chunk (reuse batch_open/end). Bar: pp >= llama.cpp default per model.
- **P2 — decode overhead diet (the ~2-3x arm).** Our raw GEMV class is FASTER than
  llama.cpp's effective rate (106.7 GB/s vs their ~67 on 0.6B-class); the decode gap is
  per-token op/barrier/glue count: ~13 dispatches+barriers per layer vs their fused-quads
  (qkv+gate_up fusion, fused attention, epilogue norms). Targets: fused qkv/gate_up batch
  GEMV, fuse norm-into-GEMV epilogues, drop per-op barriers to dependency-only via the
  one-commit chain (they are already one commit — the barrier count goes when kernels
  fuse). Bar: tg >= llama.cpp default per model (0.6B 172, 1B 103, Spark 40, MiniCPM 68,
  8B 22).
- **P3 — arch support.** Phi-4-mini fused QKV loader path (one tensor split at load).
  K2-Horizon has no llama.cpp reference (custom arch) — excluded from parity receipts.
- **P4 — calm deciding re-run** of the full two-engine sweep (same script /tmp/parity_sweep.sh)
  in a load < 2.5 window; that table supersedes the load-labeled numbers above.

Rules carried over: CPU-only mode is never removed; every kernel parity-gated before it
can serve a receipt; battery rows labeled with load, never skipped; commits straight to
main with the OpenQodex gate; src/simd/* is a peer lane — CPU kernel work is coordinated,
GPUs kernels are ours.

# ROADMAP — exact llama.cpp parity, step by step (no dates)

Contract for every step: bit/tolerance parity gate lands BEFORE any rate receipt;
int-returning ops (visible unavailable, never silent no-ops); every row load-labeled;
CPU-only mode untouched; peer lanes (src/simd, macOS editor, continuous.rs) not edited.

## Phase 0 — measurement contract (do first, keeps every later step honest)
- P0.1 The only accepted receipt: ll.cpp bar = `llama-bench -m M -t 10 -p 210 -n 82 -r 3`
  DEFAULT config (Metal offload) vs Sofuu same-window, both pp+tg, in
  `docs/bench/all-models.jsonl`; calm window (<2.5 load) supersedes. Acceptance per
  model: Sofuu pp >= llama.cpp pp AND tg >= llama.cpp tg.
- P0.2 Per-op attribution: extend SOFUU_GPU_TIMING from per-commit to per-op (CPU
  enqueue µs / GPU busy µs / gap µs) — this decides whether Phase 1 attacks barrier
  count or CPU dispatch cost first. Deliverable: one table for the 0.6B pass.

## Phase 1 — op diet: ~34 ops/layer -> ~10 (shared unlock for decode AND prefill)
- P1.1 gemm3 "qkv": one dispatch computes Q,K,V (three weight offsets in the one
  registered w_buf, row-span split). Parity gate: equals 3 separate GEMMs + layout
  [B x (q_rows+2*kv_rows)].
- P1.2 gemm2 "gate_up": one dispatch, y = [B x 2*inter] contiguous halves (swiglu
  kernel unchanged).
- P1.3 resid-add epilogue GEMM (mirror gemv_q4_k_b4c2r): o-projection and down
  projection absorb their residual add — kills 2 flat adds per layer.
- P1.4 Barrier diet from the P0.2 table: independent ops (kv vs q-norm) enqueue
  without barriers; dependent chains keep them. Bar: per-op net cost <= 10 µs calm.
- P1.5 Re-measure prefill route + decode chain after the diet (same-window A/B,
  not projections).

## Phase 2 — decode (output tps) parity, per model, in gap order
- P2.1 8B q4_K: bar tg 22.1 = 109 GB/s effective = our measured 106.7 GB/s kernel
  rate with <=10% overhead. This is the most reachable parity — do first.
- P2.2 q8_0 coalesced class (byte-lane, 16B loads) GEMV+GEMM for 1B (bar 103 =
  124 GB/s effective — highest bandwidth ask in the fleet) + K2 gains if desired.
- P2.3 q6_K batched GEMV + GEMM (single-row class already 92 GB/s bit-exact) for
  MiniCPM5-2B (bar 68) — MiniCPM may route heads through q6_K GEMV 92-class too.
- P2.4 Spark-4B (bar 40): diet receipts + kv_write single-write path check.
- P2.5 P3 loader: Phi-4-mini fused attn_qkv — split into q/k/v VIEWS at load
  (offset arithmetic in the existing zero-copy mmap, no copy), then it rides the
  same classes (bar 35).
- P2.6 bf16/f32 fleet (gemma, liquid, lfm2, gpt-x3): fused qkv/gate_up kernels
  already exist — wire the same diet, battery rows. No llama.cpp reference where
  its loader fails (K2, Ternary 27B) — our bests stand, marked out-of-parity-scope.

## Phase 3 — prefill (input tps) parity, per model
- P3.1 Chunk size 32 -> 128 (kernel B-mask is already arbitrary; slot slab PRE_B
  and positions buffer resize; KV layer math unchanged).
- P3.2 GEMM compute-rate phase: at B=128 the class is compute-bound (455 FLOP/byte)
  — float4-x loads + 2-4 rows per threadgroup sharing x bytes; target ~2 TFLOP/s
  (8B bar 211 t/s) and ~4 TFLOP/s (0.6B bar 3336 t/s = llama.cpp is near iGPU peak).
- P3.3 Causal batched attention for rows=B (the b4 family is rows-generic — feed
  rows=128) with per-row position bounds; kept off the hot barrier chain by
  ordering kv_write before attention in one commit.
- P3.4 Per-model pp receipts >= the bar; long-context/slide-window models (gemma)
  via the part/fin meta path at 4096+.

## Phase 4 — deciding batteries + flips
- P4.1 Calm two-engine sweep (/tmp/parity_sweep.sh) — the deciding table.
- P4.2 Flip per-model defaults (per-MODEL, not global) only where BOTH pp and tg
  pass; CPU fallback retained everywhere.
- P4.3 Re-run after every landed perf change (standing battery rule).

## Phase 5 — post-parity (only after the table is green)
Speculative/MTP verify chains, KIVI-style KV, sparsity — the byte-cut levers for
"improve later" per the goal statement.

# PROGRESS LEDGER (2026-10-20 night — update after every wave)

Status vs the phase plan: **~40% of the program**. Phase 0 done; Phase 1 largely
landed (one fusion inapplicable on today's quants, one kernel shipped-OFF pending
a mystery); Phase 2 barely started (kernels exist, no per-model wins yet); Phase
3 not started; Phase 4 blocked on a genuinely calm window; Phase 5 by design not
entered.

## What is DONE (pushed, text-gated)
- **Phase 0 (complete):** parity.jsonl standing harness (scripts/parity_sweep.sh);
  per-op attribution (SOFUU_GPU_TIMING=2, 14 op kinds); verdict: host enqueue is
  3% of the token — the fight is op count and window load, not CPU dispatch.
- **P1.2 gate_up fusion, prefill:** gemm_q4_k_multi shipped ON (0.6B prefill
  19→32.6-33.9 t/s loaded, +72%, text-identical 3x-repeated).
- **P1.3 resid-add epilogue (prefill):** gemm_q4_k_acc shipped ON (o/down absorb
  their add; ops/layer 31→29); review fix guards it behind !skip_attn.
- **P1.4 barrier diet:** dropped-with-receipt (enqueue trivial per P0.2).
- **Kernels + pinned parity tests:** gemm_q4_k (2.3e-7 rel), gemm_q4_k_multi
  (bit-exact, incl unaligned sections), gemv_q4_k_b4multi (bit-exact after the
  test's own ref-offset fix), gemm_q4_k_acc (text-gated in-model).
- **Battery 10.5:** first parity.jsonl rows; bars re-measured HIGHER than battery
  #10 (0.6B tg 223 vs 173): 0.6B 3416/223, 1B 2069/123, Spark 461/51, MiniCPM
  863/72, 8B 264/31.
- **Close calls with receipts:** qkv fusion is INAPPLICABLE on unsloth i1 quants
  (attn_v=Q6_K type 14) → per-tensor fallback is correct behavior; the GEMM
  shape loses at B≤4 (3-9 t/s of decode) → decode fusion OFF.

## What is OPEN (evidence in memory: mac-metal-gpu1-session)
- **b4multi decode fusion OFF-by-mystery:** bit-exact standalone, but the
  verify-chain text diverges deterministically in-model, and separately the
  serial GPU single-token path diverges from CPU (pre-existing, fusion-free),
  while SPEC=verify text == CPU. A final twist (24-token vs 96-token runs under
  the same env differing) points at a load/state-dependent route flip. Every
  text gate from here must PIN SOFUU_GPU_SPEC and the load window.
- **Serial-path CPU divergence:** token-1 differs with fusion-free code — real
  correctness item, next diagnostic target.
- Mixed-type fusions (q4_K + Q6_K sections in one dispatch) — needed before qkv
  fusion can earn on the fleet (attn_v is Q6_K in Q4_K_M/i1 files).

## Phase-by-phase remaining
- **Phase 2 (tg):** q6_K batched GEMV class (MiniCPM 6.9 vs 72 — the worst gap),
  q8_0 coalesced class (1B 50.3 vs 123), 8B Seriacal-path fix + GPU receipts at
  pinned env (3.2@load-5 vs 31), Spark (28.6 vs 51), Phi-4-mini loader (fused
  attn_qkv split-at-load; bar 563/46). Nothing landed yet.
- **Phase 3 (pp):** chunk 32→128; GEMM compute-rate phase (float4-x loads,
  2-4 rows/threadgroup; targets ~4 TFLOP/s on 0.6B, ~2 on 8B). Not started.
- **Phase 4:** calm (<2.5) deciding sweep → parity.jsonl PASS/fail → per-model
  default flips. Blocked on window/peers so far.
- **Phase 5:** untouched by design (spec/MTP/KIVI/sparsity after parity).

# PROGRESS LEDGER #2 (2026-10-21 — phases 0-1 complete, phase 2 ~70%)

**Overall: ~55-60% of the program.** The decode-side correctness contract is
now UNIFORM: all five llama.cpp-loadable fleet models decode FULL-TEXT
bit-identical to the CPU route with the fused type-grouped dispatch ON
(battery #11.1/#11.2, docs/bench/parity.jsonl).

## Phase status
- **Phase 0 (100%)** — unchanged; PLUS the text-gate protocol fix: full-text
  compare (tail -1 was comparing different lines of multiline completions —
  it manufactured a week of phantom text divergences) with SOFUU_GPU_SPEC
  pinned.
- **Phase 1 (100%)** — op diet landed end-to-end: gate_up fusion (prefill
  +72%), o/down resid-epilogue GEMM, the q6_K sectioned b4 GEMV (bit-exact,
  lane=l coalesced), and TYPE-GROUPED mixed dispatch (q4_K/q6_K/q8_0 runs —
  the qkv [q4,q4,q6] case = 2 dispatches). The qkv fusion is no longer
  blocked by mixed types.
- **Phase 2 (~70%)** — the decode per-model program is CORRECTNESS-COMPLETE:
  1B 61.9-72 t/s (was 44-53), MiniCPM 10.8 (was 5.4-6.9), 0.6B 39.9, Spark
  24.0, 8B correct (calm receipts pending). The enabling fix = THE blob base
  bug: sections are not monotonic in file order (1B attn_k sits below
  attn_q); base-from-first-section underflowed the u32 delta to 2^32 → the
  kernel read base+2^32+rem. base = MIN aligned offset per run (group-then-
  dispatch rewrite; the prefill macro's latent copy audited+fixed the same
  day). Rate wave begun: q8_0 b4r4 (x-traffic/4) = RATE-NEUTRAL with receipt
  → decode is latency-chain bound (64 serialized load→cvt→FMA iterations per
  lane; f16 d interleaved every 34B blocks u32 loads) — unroll/dual-chain +
  lane-split + int8x4 de-interleave are the named levers.
- **Phase 3 (5%)** — GEMM class exists at B≤32 (text-exact, 33 t/s class on
  0.6B at load); chunk-128 + compute-rate not started.
- **Phase 4 (0%)** — calm sweep blocked on window; bars from battery #10.5/11.
- **Phase 5** — not entered (by design).

## Board (decode bars = llama.cpp default, full-text-gated Sofuu)
| model | bar | Sofuu | gap | note |
|---|---|---|---|---|
| 1B | 123 | 62 | 2.0x | q8_0 fused ✓; latency chain next |
| 0.6B | 223 | 40 | 5.6x | q4/q6 ✓ |
| Spark | 51 | 24 | 2.1x | rides q4 |
| MiniCPM | 72 | 11 | 6.5x deepest | q6/q8 ✓; rate wave |
| 8B | 31 | pending calm | — | correct at any load; ~3 calm-class |

Prompt side (Phase 3) is where the 3-6x gaps remain on every model.

## Next in order
1. q8_0 latency-chain experiments (2-block unroll with 8 chains → lane-split
   → int8x4 de-interleave); same pattern for q6_K (MiniCPM).
2. 8B calm receipts; then Phase 3 chunk-128 + compute-rate.
3. Calm deciding sweep (Phase 4) → per-model flip table → Phase 5.

Bugs closed this leg (receipts in memory: mac-metal-gpu1-session): the blob
base/min-base fix (the week's actual blocker), the tail -1 text-gate
artifact, the q8_0 [d][qs] layout trap, the extern "C" mangling of appended
ops, the test-side ref-offset typo — every one caught by the pinned
full-text gates or the standalone oracle tests.

# PROGRESS LEDGER #3 (2026-10-21 — PHASE 2 CLOSED; ~70% of program)

**Phase 2 CLOSED** (7c039237c): every exit item its plan defined is delivered
except the tg-parity numbers themselves, which by design are the Phase-4
calm flip-table deliverable:
- c2 oracle gate PASS under BOTH env states (whichever kernel serves, values
  == vec_dot_raw_q8_0).
- Fleet receipts with the coalesced env ON: ALL 5 models FULL-TEXT
  bit-identical (firstdiv=None): 1B 40.9@load-15.9-spike (settled 87.8@load-
  3.2), 0.6B **73.6**, Spark 27.8, MiniCPM **24.7** (4.5x its fused base),
  8B 2.5 (load noise).
- Kernel classes + gates now in tree: gemv_q8_0_c2 (serial-decode serial
  class), gemv_q8_0_b4multi/b4r4, gemv_q6_k_b4multi (bit-exact), the fused
  gemm family + min-base, per-op timing, the rate triangle harness.
- Text-gate protocol (full-text compare, pinned SPEC, load labels) — the
  artifact that a week of phantom divergences was built on — closed and
  documented.
Board (decode, text-exact everywhere): 1B 87.8/123 · 0.6B 73.6/223 · Spark
27.8/51 · MiniCPM 24.7/72 · 8B pending-calm/31 (bars = llama.cpp default).
Residual tg gaps 1.4-6.5x = calm-window-flip items + kernel-rate waves.

# PHASE 3 (beginning) — prefill (input tps)
Scope from the plan: chunk 32->128, GEMM compute-rate (float4-x, weight-rows
per threadgroup), mixed-type prefill qkv. Baseline receipts: 0.6B GEMM-prefill
33 t/s-class at load (ll bar 3416); q4-GEMM at b=4 standalone = 1.0 GB/s (the
subtile waste receipt) — at b=32 the subtiles fill; the compute-rate work
starts from measured-B32 rates. Constraints: FULL-TEXT gates, pinned env
(SOFUU_GPU_PREFILL_GEMM=1, coalesced=2), load labels; CPU prefill route stays
default until receipts justify the per-model flip.

## PHASE 3 first receipts (2026-10-21 late — chunk-128 leg)
- **THE 1024-THREADGROUP DISPATCH BUG (the day's real find).** Seven batch
  ops dispatched rows-scaled grids in ONE Metal dimension capped at 1024
  threadgroups: attn_1c_b4/kv_write_b4/head_rmsnorm_b4/attn_part_b4/
  attn_fin_b4 (rows×heads(×n_chunks)), add/swiglu ((b·n+255)/256). Past the
  cap the launch SILENTLY TRUNCATES — bisect pins it: b=60 (960 groups) was
  bit-exact at any prompt, b=128 (2048) garbled only rows ≥64 (first tokens
  fine, then forked). Also present in add (8B b=128: 2048 groups) and swiglu
  (8B b=128: 6144). Fix = `sf_grid2d` 2D split (x=min(g,1024), y=ceil/1024)
  + kernels linearize `tg.y*1024+tg.x` — bit-identical when the grid fits
  one x row, so every decode/spec route is dispatch-tuple-identical.
- **Chunk-128 shipped** (SOFUU_PREFILL_CHUNK 1..=128, default 32): slab
  (PRE_B) sized by the env parsed ONCE in arm() into a `pre_chunk` struct
  field — the forward paths read the field, never the env, so slab and
  chunk cannot diverge. Meta needs NO resize: capacity is n_layers×
  SPEC_B×4 slots (448 on 28 layers; rows land at [L, L+b) via the
  layer_idx slot base and each layer rewrites its own range) — the old
  note's "112 slots / b=128 overflows" dropped a factor of 4; arm() still
  clamps to 15·layers+1 to keep tiny-layer models legal.
- **Rates:** 0.6B 326-tk prompt: chunk-128 = **51.1 t/s** vs chunk-32
  32.3 t/s same window (+58%). Pre-fix route showed 49.8 — because the
  truncated launches did LESS work; the honest number is 51.1 with correct
  text.
- **Text gates = tolerance-class, with controls.** Deterministic synonym
  forks (5×-stable md5s) on tie-heavy prompts (e.g. `sentence ×2` exact
  repeats: "the ferry" → "the harbor's first ferry"); distinctive-tail
  prompt identical (P25), chunk-60 identical, decode (SPEC=0/1, CPU
  prefill) identical. Signature matches the accepted f32-reassociation
  argmax-flip class; the numeric top-2-gap probe ( certify with logits,
  not prose) is booked as Phase-3 residual parity work.
- **Peer-lane interplay (flagged):** the concurrent GQA-fused CPU decode
  + vector-swiglu commit moved the CPU baseline text on GQA models —
  Paris 0.6B/1B now fork CPU-vs-GPU where battery #11 was exact. Decode
  dispatch is provably untouched by this leg (grid tuples identical at
  y=1); treat ALL cross-route text receipts as transient until the
  Phase-4 calm sweep re-pins both sides per model.
- The forward pipeline itself splits prefill calls (108+2-style tails,
  dynamic per run) above the GEMM chunker — featured here so nobody
  misreads chunk profiles; SOFUU_PREFILL_DEBUG prints per-call
  n_tokens/pre_chunk.

## PHASE 3 second receipts (2026-10-21 late — compute-rate leg, PARTIAL by design)
- **q8_0 + q6_K REAL GEMM kernels landed** (was: the prefill `gemm!` fast path
  accepted q4_K ONLY; every q8_0/q6_K tensor rode the per-token GEMV loop —
  the 1B's all-q8_0 prefill was 7×B×layers dispatches/pass). Six kernels:
  gemm_q8_0 / gemm_q6_k × {plain, multi (qkv/gate_up fused), acc (o/down
  resid epilogue)}, dequant cores copied from the receipted decode kernels
  (gemv_q8_0_b4multi / gemv_q6_k_b4multi), same 8-token tile shape as
  gemm_q4_k. **Oracles: worst_rel = 0.0 (bit-exact vs vec_dot_raw) at
  B=5/32/128** on fleet shapes (1B attn_q 2048×2048; 0.6B attn_v 1024×1024).
- **SOFUU_GEMM_TYPES=all env-GATES the new dtypes OFF by default** — q4_K
  stays the only default-dispatched type until the defect below is pinned.
- **THE 1B ROUTE DEFECT (booked, Phase-4 blocker):** the prefill-GEMM route
  on the all-q8_0 1B produces silently-wrong KV/logits at b>=~64-row passes
  → the CPU repack path then asserts (nearest_int fval>4194303) or text
  forks garbage-wards; passes working per-prompt-deterministically at
  b<=32 (clean receipts c32 exit=0) and SINGLE-pass c128-108-token works;
  the 326-tok c128 sequence (part/fin at b≈108 with priors) breaks. The
  kernels are bit-exact standalone, and the per-row GEMV fallback ALSO
  breaks there → this is ROUTE/GLUE-level at 1B shapes (32 heads, hd=64,
  nkv=8), attention-chain per ablation (skip=attn clean). The 0.6B (16
  heads, hd=128) runs the same glue bit-exact at b=128.
- **Standalone GEMV-class rates under load 8-9 (labeled, ceiling work next):**
  q8_0 2048×2048: 3.1 / 1.4 / 0.7 GB/s at B=5/32/128 (rate DROPS with B —
  the tile-broadcast design wastes at scale); q6_K 1024×1024: 1.2 / 0.9 /
  0.4. The vectorized-load compute-rate iteration (u32 weight loads,
  float4 x, rows/threadgroup) is the actual Phase-3b payload and stays
  OPEN; these kernels land the DISPATCH count fix + correct needs the
  rate iteration.
- In-model receipts (all load 5-9, ±40%): 1B route c32 types=all **98.0
  tk/s**; MiniCPM route c128 types=all **70.4 tk/s** (first real prefill
  GEMM on the q6/q8 mix). Battery default (route off) untouched: 1B
  CPU-tiles prefill 161 t/s at the same load, exit=0, IOGPU=0.

# PROGRESS LEDGER #5 — ROUTE-PARITY ARC CLOSED; remaining = rate + endgame (2026-10-24)

## DONE this arc (all pushed; every receipt in the sections below)
1. **The 0.6B-parity fork is closed**: root cause = arithmetic-class split
   (CPU act-quant-MMQ vs GPU exact-f32), NOT a GPU kernel bug. The Metal
   q4_K + q6_K gemm mirrors (`gemm_q4_k_mmq`, `gemm_q6_k_mmq`, the pos
   row-quantizer `quantize_q8_k_pos`, `quantize_q8_k`) are **bit-identical
   to the shipped CPU kernels** (probe worst 0.0 on both; live stage bisect
   s2.qa/s2.va at 1e-7).
2. **All 8 prefill GEMMs route-parity at f32-noise**; the prefill→decode
   boundary is byte-identical (x.bin 0/512; L0 KV f32-noise).
3. **Decode-chain fallback contract**: 8 decode exports void→int
   dispatched-status, 29 decode call sites propagate to the CPU layer loop;
   1B chunk-128 route (the old fault shape) runs clean.
4. **THE GATE IS GREEN** under the mirror envs
   (SOFUU_GPU_DECODE=1,SOFUU_GPU_PREFILL_GEMM=1,SOFUU_GPU_TILE_MMQ=1,
   SOFUU_DECODE_MMQ=1):
   | model | verdict |
   |---|---|
   | 0.6B i1-Q4_K_M Paris/32 | NEAR-TIE-CERTIFIED (fork 0.089 ≤ own band 0.906, step 19) |
   | 0.6B exact-repeat prompt | IDENTICAL 23/23 |
   | 1B Q8_0 Paris/32 | IDENTICAL 31/31 |
   | 0.6B + coalesced=2 | NEAR-TIE-CERTIFIED (0.126 ≤ 0.355) |
   | 8B + coalesced=2 | NEAR-TIE-CERTIFIED (0.237 ≤ 0.653) |
   Certification = `text_gate.py --bound auto`: fork-step |dtop1| ≤ the
   worst MATCHED-step |dtop1| of the same generation (self-referential, no
   magic constant).
5. **The physics is characterized**: the residual class = near-zero-gate
   crossovers amplifying a legitimate ~5e-7 f32 seed (L1→L2 = 10^4 in one
   layer; silu is Lipschitz ⇒ single-op mirrors CANNOT close it). Envelope
   saturates ~1% relative by mid-network; decoding-wise it appears only as
   certified near-tie flips.

## REMAINING (the honest board)
- **(d) GEMM rate program — STARTED (LEDGER #5).** The 8-token GEMM class
  was replaced by a threadgroup-tiled class: q4_K coalesced-lane mapping
  (8-15x kernel rate), then a 32-row x 64-token tile with threadgroup-memory
  weight/activation staging for q4_K/q6_K/q8_0 (5.4x over the shipped class
  at 3072x1024/B=128). The route moved 143->304 (0.6B), 106->279 (1B),
  11.4->122 (MiniCPM), 7.9->20.5 (8B) tk/s but still trails the CPU tiles
  and the bars. Next: simdgroup-matrix (MMA) tiles + the attention kernel
  (now the largest single prefill cost, 283 of 711 ms on the 0.6B).
- **(e) fleet-exact endgame — NOT STARTED.** Order-exact ops + a
  libm-matching MSL exp/log; the ONLY route to unconditional IDENTICAL on
  near-tie-heavy q4/q6 prompts. Entry point: mirror the fleet op order from
  the CPU kernels (the mmq probes are the pattern) + port ggml's exp
  polynomial to MSL.
- **(f) per-model defaults — PEER LANE.** All flip decisions are
  data-ready: c2+mirrors gate-clean AND rate-winning on 0.6B/8B/MiniCPM
  (1B wants default + the q8 class); the per-model machinery lives with the
  peer. Until it lands, the documented fast paths stay env-opt-in.
- **(c-residual) 1B IOGPU fault root cause**: contained by the fallback
  contract (exit 0, correct, but 923 s fallback-crawl AT THE FAULT SHAPE
  b=128@pos≥128 on the chunk-128 route — off by default); a Metal capture
  session is the remaining step.
- Defaults: **never moved across the entire arc** — the gate is now the
  data-backed criterion for any future flip.


# PROGRESS LEDGER #4 — PHASE 4 deciding sweep (2026-10-22, calm 1.1-2.3)

**DECIDING RESULT: the GPU prefill GEMM route loses to the CPU tiles on
every model. Flip table = ALL OFF (status quo stands).** The route was a
dispatch-count fix, not a rate fix:

| model | bar pp | CPU tiles | route c128 | route/CPU |
|---|---|---|---|---|
| 0.6B i1-Q4 | 3396 | **583.3** | 33.2 | 0.057x |
| 1B Q8_0 | 2055 | **380.7** | 128.5 | 0.34x |
| Spark-4B Q4 | 461 | **68.2** | 74.3 | 1.09x |
| MiniCPM5-2B Q6 | 865 | **109.6** | 11.5 (types-off) | 0.11x |
| 8B Q4_K_M | 264 | **46.9** | 2.4 | 0.05x |

(route rows at the sweep default env; 1B c128 ran CLEAN at calm — the
under-load crashes are contention-gated, see below.)

## Flip decisions (Phase 4)
- **SOFUU_GPU_PREFILL_GEMM: OFF default everywhere (keep).** No model's
  route beats CPU tiles except Spark at +9% — immaterial vs its 6.8x gap
  to the bar, and the route carries the load-gated command-buffer risk.
- **SOFUU_GEMM_TYPES: OFF default (keep)** until the torn-buffer contract
  below ships and calm receipts exist.
- **SOFUU_GEMM_TILE=32: opt-in (keep)** — bitwise-fine, not rate-winning.
- **Decode side (c2/coalesced/b4multi): NOT flipped today** — the sweep's
  default-env decode rows (35.7/56.3/31.2/9.9/3.3) are far from the bars
  AND below the pinned-env receipts (0.6B 73.6 was SOFUU_Q4K_COALESCED=2);
  the deciding A/B (pinned vs default, calm, per model) is Phase-4's next
  data point before any default flips.
- **tg-parity verdict stands:** nothing passes its tg bar yet; prefill
  parity needs the GPU to beat CPU tiles BEFORE the bars matter.

## The real Phase-5-grade program (what parity actually requires)
Prefill parity = GPU must first BEAT the CPU tiles (583/381/68/110/47),
then close to the bars — ~6x over today's route class. The two
bit-exact kernel generations (tile-8, tile-32) are serial-lane-loop
bound (0.4-8 GB/s; Mac DRAM band is far higher); the required
architecture is the simdgroup-matrix GEMM (threadgroup-tiled weights,
int8 SIMD dots, register tiles). Booked as the kernel program with both
generations' rate receipts.

## Defect class REFRAMED + contract shipped
Under load 5-11 the 1B's route runs hit stochastic IOGPU Code=14
command-buffer failures (torn: partial writes, later ops killed). At calm
the SAME configs run clean and fast (1B c128-326 = 117.8 tk/s, no assert).
REFINED by the 6-try crash-hunt loop (post-fix binary, load 0.5-3.0):
the 1B c128-326 run faults the SAME buffer EVERY run at every load —
the start_pos=128, b=128 chunk (multi-chunk part/fin on 32 heads, hd 64)
never completes; the other three passes profile fine, and the shipped
fallback absorbs it into CPU tiles 6/6 with no crash (0 IOGPU escapes).
So the fault is a deterministic per-shape dispatch fault in the 1B's
multi-chunk attention-glue class (part/fin/kv/head_rmsnorm at b=128 +
pos>=128), not load-gated; the 0.6B (16 heads, hd 128) runs the same
shape fine. Standalone per-op fault isolation at exactly (b=128,
pos=128..255, 2 chunks, 1B shapes) = the first task of the next
defect session; a GPU capture would nail it. **Contract shipped: batch_end now returns the completion
status (C ABI int; header + FFI decls updated) and every production route
propagates it — prefill GEMM core falls back to the CPU tiles for the
whole prompt, execute_layers/layers_one_commit/gpu_layer_step/ggpu_head
report false, execute_layers falls through to the CPU layer loop, and the
head falls back to the CPU matvec (self.x is already normed there — no
re-norm). The silent-garbage path is closed; what remains booked is the
decode-chain void refactor (decode-side GPU fns still take the fallback
implicitly) and pinning WHY Apple's queue restarts under contention.

## Open items carried
1. simdgroup-matrix GEMM program (above) — the only path to pp parity.
2. Decode-side deciding A/B (pinned SOFUU_Q4K_COALESCED vs default) → decode flips.
3. Decode-chain batch-failure refactor (void fns → fallback contract).
4. IOGPU restart root-cause under peer load (Metal heap/timeout?).
5. top-2-gap logits probe (near-tie certification).

## PHASE-4 decode A/B (calm 1.4-3.5, Paris/82 greedy) — coalesced=2 vs default
| model | bar tg | default | coalesced=2 | winner |
|---|---|---|---|---|
| 0.6B i1-Q4 | 224.5 | 31.8 | **61.4** | coalesced 1.9x |
| 1B Q8_0 | 121.9 | **50.3** | 33.2 | default 1.5x |
| Spark-4B Q4 | 51.0 | 27.1 | 27.1 | tie |
| MiniCPM5-2B q6/q8 | 73.0 | 9.1 | **22.0** | coalesced 2.4x |
| 8B Q4_K_M | 31.2 | 2.8 | **5.6** | coalesced 2.0x |

DECIDING: coalesced wins 3 models + ties 1; the 1B (all-Q8_0) wants the
DEFAULT serial q8 path — a global flip regresses the 1B 1.5x. The flip
needs per-type semantics (c2 ON for q4/q6 rows, OFF for the q8 serial
class) — booked as the flip implementation; nothing flipped yet (the env
stays the pinned-receipts switch).

## Fault-isolation status (phase-3 residual, 2026-10-22)
- The standalone per-op test (attn_glue_b4_fault_isolation_1b) passes at
  every b — the fault needs model context, not a single op.
- skip=ffn faults (2 IOGPU/run, deterministic); skip=attn never -> the
  attention SUB-CHAIN carries it at model level.
- MTL_DEBUG_LAYER=1: fault vanishes (validation masks it — no API-level
  OOB reported).
- ncm_pass (per-pass true chunk count for part/fin instead of the
  config-level 32) landed as index-space-only 16x grid shrink — principled
  but did NOT remove the fault. Remaining suspicion sits in the large
  chained encoder (multi-layer one command buffer), which the standalone
  test cannot reproduce.
- OPEN containment hole: after a torn buffer, the CPU-tiles fallback can
  still trip the repack assert in some interleave (2/3 runs at load ~3) —
  the lldb backtrace attempt hung; next session pins it with SOFUU
  instrumentation in the CPU prefill path (model.rs lane), not src/simd
  (peer lane). Defaults are unaffected (route opt-in; battery path clean).

## PROBE CORRECTION (2026-10-22, top-2-gap logits probe landed)
SOFUU_LOGITS_GAP_PROBE now dumps tagged top-8 (value,id) logits per token
from prefill_head (GEMM core), cpu_tiles_head (CPU tiles), ggpu_head and
cpu_head (decode). At the P2 fork (exact-repeat prompt, pos 73, compare
aligned at the same position): **CPU tiles top-1 = 3364 (gap 0.674),
GPU route top-1 = 576 (gap 0.353) — top-1 differs by 0.455 logits and the
GPU's id is not in the CPU's top-8.** The earlier "tolerance-class
synonym fork" classification for this family is REFUTED — the GEMM-route
prefill computes materially different logits on exact-repeat prompts
(P2, P60-differ vs P25-with-distinctive-tail-identical). The suspicion
sits in the attention-glue values (kv_write/1c — the only fleet ops
without a value oracle; the gemms are bit-exact by oracles). The
part/fin/1c/kv_write value-oracle program (CPU attention mirror) is the
next concrete instrument; until it lands, treat ALL GEMM-route text
divergences as bug-class, not reassociation-class.

## KV-CACHE DIFF INSTRUMENT (2026-10-22 late — the P2 pin, to the value level)
SOFUU_KV_DUMP=<prefix> (model.rs maybe_dump_kv, wired at both prefill
tail returns) writes k/v/x/logits after prompt prefill; scripts/kv_route_diff.py
maps byte diffs into (layer, kh, pos, hd). Findings at 0.6B/P2, CPU tiles
vs GPU route (chunk 128, types=all):
- **Final residual x: byte-identical.**
- **The KV caches are NOT identical: ~1M f32 differ (≈67% of the live
  region, RMS 1e-1), worst |diff| ~3.5 on |K|~100-300 values, HD-LANE
  STRUCTURED (hd 48-52 + 63 dominate the top hits), spread over ALL
  positions 0..147 — including positions beyond the 110-token prompt
  (max-tokens=1) under BOTH routes.**
- Route-internal invariance proven: tiles@batch4 == tiles@default (0 diff),
  route@chunk4 == route@chunk128 (0 diff) — the divergence is CROSS-ROUTE,
  not batch-size dependent.
- The CPU tile path and the CPU depth path agree with each other where
  probed; the divergence is specifically CPU-tile-KV vs kv_write-KV.
- Mechanism hypothesis: the Metal glue computes theta = pos/pow(base, 2j/hd)
  on-device (MSL pow, per-lane, division) while the CPU rope reads the
  PRECOMPUTED sofuu_rope_lookup table (1/powf × pos, mul; peer lane already
  has the table + sofuu_test_rope_table_parity). The on-device pow/div
  rounding differs lane-wise — but the ~1e-2-relative worst hits are LARGER
  than pure f32 pow rounding explains, and the pos>prompt writes are
  unexplained: the fix program = route a CPU-built cos/sin TABLE (exact
  sofuu_rope_fill_rows sequence mirrored in Rust) through the kv_write/
  head_rmsnorm/1c kernels (one extra slab row-region + binding each), with
  kv_route_diff as the value gate; if diffs don't collapse, the pos>prompt
  writes mean the position plumbing (not precision) carries a real bug —
  a GPU capture decides. Either way the instrument chain is complete.

## ROPE-TABLE FIX + THE CORRECTED SHAPE (2026-10-22, late)
- **kv_write_b4/head_rmsnorm_b4/1c now take a CPU-built cos/sin rope table**
  (exact sofuu_rope_fill_rows mirror: freqs = 1/powf(base, 2j/hd) f32,
  theta = pos*freq, layout [row]=cos(hd/2)|sin(hd/2); slab region
  pslots.rope = PRE_B*hd). has_table=0 keeps the legacy on-device path —
  decode/spec/chain callers unchanged this round.
- **THE ROPE HYPOTHESIS IS FALSIFIED**: with the table active the KV diff
  count moved by exactly ONE f32 (1,060,863 vs 1,060,862). Not theta.
- **THE SHAP CORRECTION that rewrites the story**: the 0.6B head_dim=128
  (not 64 — the file is exactly 28*8*4096*128*4). With the true shape the
  diffs live ONLY at prompt positions 0..73 (the earlier pos>prompt reads
  were a shape artifact), ~50% of live K values, PAIR-STRUCTURED at
  mid/high j (113/115/116 + 48/49/50/52 = j and j+64 partners). Magnitude
  (3.5 abs on |K|~100-300) EXCEEDS any theta-precision mechanism =>
  projection-level (x_norm/W_qkv mid-pass) divergence, not rope.
- Routes remain self-consistent (read-own-writes); at the probed pos 73
  BOTH routes sample 576 (tiles_head 14.057 vs prefill_head 14.186 =
  head-GEMM numeric class); the P2/P60 48-token forks appear at later
  tie-heavy steps from the KV noise compounding. The instrument chain
  (KV dump + kv_route_diff + tagged logits probes on all four head paths)
  is complete; the next step = a real-weights attention/glue oracle
  (CPU mirror with model weights) to pin the projection-level source.

## PER-TYPE COALESCED FLIP IMPLEMENTED (2026-10-22, late)
The q8_0 serial GEMV no longer rides SOFUU_Q4K_COALESCED levels 1-2:
coalesced=2 = c2/b4c for q4_K + coalesced q6_K (the DECIDING A/B winners:
0.6B 61.4 vs 31.8, MiniCPM 22.0 vs 9.1, 8B 5.6 vs 2.8), and the all-q8_0
1B keeps the shipped serial class unless coalesced=3 is asked explicitly
(the 1B regresses 33.2 vs 50.3 under the c2 class). DEFAULT STAYS 0 (all
off) — the per-model default flips wait on the calm text-gate pass at
coalesced=2 (the first attempt ran at load 28-39 = tainted, re-gated).

## CALM coalesced=2 TEXT GATES (2026-10-23 02:48Z, load 1.74) — the flip verdict
| model | c2 text | c2 decode t/s | flip |
|---|---|---|---|
| 0.6B i1-Q4_K_M | **DIFFER** | (rate line lost) | OFF — c2 bug-class on i1 quants |
| 1B Q8_0 | IDENTICAL (vacuous: L2 flips nothing on q8) | n/a | n/a |
| Spark-4B Q4_K_M | IDENTICAL | 25.4 | **flippable** |
| MiniCPM5-2B q6/q8 | IDENTICAL | 29.7 | **flippable** |
| 8B Q4_K_M | **DIFFER** | 2.1 | OFF — c2 bug-class |
The c2 class changes sampled text vs CPU on 2 of 5 models CONSISTENTLY
(the load-28 pass: same pattern) — bug-class in the c2 GEMV summation on
those models (the glue-oracle program absorbs it: c2 ≠ serial summation
order reaching tie-heavy argmaxes, same family as the KV projection
noise). Per-model DEFAULT flips need the per-model defaults machinery
(peer: per-model spec/pool defaults) — until then SOFUU_Q4K_COALESCED=2
per-run stays the documented fast path for Spark/MiniCPM-class models.

## ROOT CAUSE PINNED (2026-10-22/23 — the P2-class fork is the act-quant arithmetic class, NOT a kernel bug)
Three receipts, one conclusion. The instruments: attn_1c_values_micro_shape
(the 1c value oracle, b=2/priors=2/32h/8kv/hd64 — CPU mirror = textbook
causal softmax attention) → **PASS, worst |Δ| 1.2e-7 = the 1c kernel is
eliminated**; SOFUU_QKV_SNAP stage bisect at L0 (identical embeddings →
s1 attn-norm out 2.4e-7 noise → **s2 qa/ka/va ALL ~1e-2, every value** →
s3.vc inherits va unchanged through kv_write → s4/L0 downstream
amplification) → **the divergence is born in the W_qkv GEMM values**;
and the decisive real-weights oracle (Python q4_K dequant of
blk.0.attn_q.weight straight from the GGUF, canonical get_scale_min_k4 +
nibble math, f64 dot against the dumped s1.xn):
- **python vs GPU gemm: worst 1.5e-6 (exact)** — the Metal q4_K GEMM is
  canonical-correct.
- **python vs CPU matmat: worst 1.5-1.8e-2, e.g. r1419 py=1.050925
  gpu=1.050925 EXACT, cpu=1.047717** — the CPU side deviates.

WHY (not a bug): the CPU batch gemm is the llama.cpp-class MMQ pipeline —
`ggml_quantize_mat_q8_K_4x4_neon` q8_K activations + 8x4 packed-tile
integer dot (dotprod_a76) — while the Metal gemm dequantizes and dots in
f32 exactly. Two legitimate arithmetic classes; ~1e-2 relative is
act-quant noise by design (same class as the Pi-5 receipts). llama.cpp
GPU-vs-CPU text matches because ITS Metal MMQ also quantizes activations
with the CPU-identical code; Sofuu's GPU gemm never got that treatment.
Consequences, all consistent at last:
- q4_K models (0.6B, 8B) CANNOT route-match text at tie-heavy argmaxes in
  either direction — explains P2/P60 DIFFER + the c2-gate bug-class rows;
  q8_0 models (1B) matched because both sides are exact there.
- "GPU kernel bug" suspicion on q4_K gemm: RETRACTED (all three live
  dispatch classes — tile-8, tile-32, fused-qkv — agree with each other
  bit-for-bit and with canonical).
- CPU route vs llama.cpp: UNCHANGED status (CPU = act-quant class both).

## THE PARITY FIX (booked next session): Metal q8_Kx4 MMQ gemm
Give the GPU gemm the CPU's exact feeds: host runs the SAME act-quant
(sofuu ggml_quantize_mat_q8_K_4x4 semantics) on xn/ao/etc once, passes
(q8_K bytes + per-block scales) to a Metal kernel that mirrors the CPU
integer dot + per-block scale-deferral and per-block f32 association
ORDER bit-for-bit, consuming the same packed-tile weight layout
(sofuu_tile_lookup_p) so W-side reads are 1:1. Gate behind
SOFUU_GPU_ACTQUANT=1 (default off), validate with the same SOFUU_QKV_SNAP
stage bisect (s2 must go byte-identical), then re-gate P2/P60/P25
full-text GPU-vs-CPU — that is the CPU-identical bar. Only after that do
any route/coalesced default flips move. (Fallback ordering caveat: the
tile repack cache must be populated before the GPU gemm reads it, or the
kernel needs the unpacked-row form — decided at implementation when the
tile lookup's lifetime is wired into the engine.)

## PHASE-4 OPENING RECEIPTS (2026-10-23 09:1xZ — the MMQ mirror is partly SUPERSEDED)
Constants: 0.6B i1-Q4_K_M, "Write three sentences about Paris.",
SOFUU_QKV_SNAP stage dumps + tagged logits probes.
1. **SOFUU_NO_TILES CPU == GPU at the L0 qkv stage: worst 1.2e-6
   (r1419 nt=1.050925 gpu=1.050925 py=1.050925)** — the shipped plain
   batch branch reaches the exact class; only the default TILE path
   (q8_K per-256 activations, 3x-worker 8x4 repack kernel) deviates
   1.8e-2 vs both. "Act-quant metal mirror" for the qkv stage: NOT
   NEEDED — the GPU exact gemm already has its CPU twin (NO_TILES).
2. TILES vs NOTILES both on CPU: 1.835e-2 (s2.qa) — tiles carry the
   coarse-actquant class; notiles/py/gpu are one canonical class.
3. Tagged-logits probe, prefill step-1 (pos=5): NOTILES-cpu vs GPU route
   agree on ALL top-8 token IDs; values differ ~4e-3 absolute on
   |logit|~14.5 (≈3e-4 relative) — vs the P2/P60 fork gaps (0.3-0.7) that
   class is 100x smaller: the 1.8e-2 tiles class was THE text-fork
   driver; the remaining 3e-4 residual = accumulated f32 association
   noise (CPU 4-lane vec_dot_f32 vs GPU sequential) amplified by 28
   layers, and it still tie-flips some far-context steps.
4. Full text A/B (Paris/32): cpu-tiles / cpu-notiles / gpu-route are ALL
   THREE different — prefill parity alone does not close text; the
   decode GEMV classes (ggpu_head vs cpu matvec) add their own logits
   offsets per generated token.

REVISED BOOKING (supersedes "Metal q8_Kx4 MMQ gemm" for the prefill side):
- (a) remaining text-parity gap = DECODE-side class + the 3e-4 residual;
  the decode-chain refactor + a logits-tolerance gate (near-tie
  certification from item 5) own the bar;
- (b) the ONLY default-route change this justifies: NONE today — but the
  {SOFUU_GPU_PREfill + SOFUU_NO_TILES} pair is a documented
  CPU-identical-class prefill configuration (1.2e-6/3e-4);
- (c) the 1.8e-2 tiles class vs canonical stays a KNOWN property of the
  default route (llama.cpp-adjacent coarse-actquant), rate receipts
  unaffected; if the developer wants default-CPU vs default-GPU text
  equality, the 8x4-tile MSL mirror (q8_K per-256 + mmla order) is the
  booked kernel program.

## DECODE-SIDE PIN (2026-10-23 — gen-point probe; the fork class is composite, not one op)
Instrument: SOFUU_DECODE_PROBE=<path> — per sampled step (gen_root/gen_node/
gen_serial) {pos, sampled tok, top-8} appended from the generator's sampling
points (mod.rs), step-ALIGNED across routes by row order. 0.6B Paris/32:
1. Default CPU (tiles) vs GPU route: matched-step top1 |Δ| 0.23 by pos 8 —
   the 1.8e-2 tiles prefill class compounding; first token fork pos=9.
2. EXACT pair (SOFUU_NO_TILES=1 CPU + GPU gemm route): matched-step |Δtop1|
   3.1e-2 by pos 8; first fork STILL pos=9 — a 0.003-gap near-tie
   (CPU 18.710/18.682, GPU 18.757/18.649).
3. KV dump at max-tokens 8, exact pair: decode-written positions
   (6..12) BYTE-IDENTICAL across routes — the decode KV write path carries
   NO extra class. Prefill positions 0..5 differ (pair-structured ~half the
   f32, worst 0.4 on |K|~100-300 = ~1e-3 relative, growing L0→L27).
4. Stage ground truth at the exact pair (L0 tail chunk): s1.xn 2.4e-7,
   s2.ka 7.2e-7, s3.kc (post k-norm+rope) 1.8e-4 ABS on |K|~100-300 =
   ~1e-6 relative — every prefill op is at the 1e-6 f32 class; the ~1e-3
   relative at L27-K = compounding of MANY 1e-6 seeds (NEON 4-lane f32
   order vs GPU order per gemm; Metal SFU exp vs libm expf per swiglu).

Verdict: there is NO single decode op to fix — the decode chain writes
bit-identical KV; the residual is the floating-point noise floor of two
different-but-valid op implementations compounding through the depth.
Bit-exact route text would require whole-fleet order-exact ops INCLUDING
a libm-identical MSL exp (booked only if the developer wants the
endgame). The shipping bar is therefore the LOGITS-TOLERANCE GATE: a
full-text gate passes when sampled tokens are identical OR every
diverging step is certified near-tie (top1−top2 gap < the measured
composite bound, ~5e-2 absolute at 0.6B/Paris scale; recorded per step by
SOFUU_DECODE_PROBE). The default-vs-default gate additionally needs the
1.8e-2 tiles class removed on one side (the 8x4-tile mirror program) —
otherwise default CPU text can never equal GPU text on q4_K models.

## LOGITS-TOLERANCE GATE SHIPPED (2026-10-23 — scripts/text_gate.py) — the re-gate runner
Per the decode pin, "CPU-identical" text is achievable exactly when no
sampled step sits inside the composite noise band of a top-2 tie; a single
near-tie flip makes both routes continue greedy in DIFFERENT contexts
(aftermath rows are context-divergent by construction and must not be
judged). text_gate.py therefore certifies on the FIRST diverging step:
IDENTICAL / NEAR-TIE-CERTIFIED (|dtop1| < bound at the fork step) / FORK
(>= bound = bug class; defaults must not move). Gate set (0.6B i1-Q4_K_M
+ 1B Q8_0, Paris/32 + exact-repeat prompt):
| gate | verdict |
|---|---|
| 0.6B paris32, exact pair (CPU SOFUU_NO_TILES=1) | NEAR-TIE-CERTIFIED (fork step |dtop1|=0.0470 < 0.05) |
| 0.6B paris32, DEFAULT pair | **FORK (|dtop1|=0.23) = the 1.8e-2 tiles class — defaults stay locked** |
| 0.6B exact-repeat prompt, both pairs | IDENTICAL (23/23) |
| 1B Q8_0 paris32, DEFAULT pair | **IDENTICAL (31/31) — full route-to-route text equality on the exact-arithmetic type** |
Reading: the fleet is route-consistent end-to-end on q8_0; the q4_K
default pair remains gated by the tiles class (booked: 8x4-tile MSL
mirror), and the remaining near-tie flips on the exact pair are
certification-class until the fleet-exact endgame (order-exact ops +
libm-matching exp) is funded. PROMPT SET for future re-gates: "Write
three sentences about Paris." (+32) and the exact-repeat string above
(+24) — re-run scripts/text_gate.py unchanged.

## 8x4-TILE MSL MIRROR — framework shipped, byte-identity PENDING one receipt (2026-10-23)
Landed (SOFUU_GPU_TILE_MMQ=1, default off): `quantize_q8_k` + `gemm_q4_k_mmq`
MSL kernels (a76_repack.h q8_K act-quant + p-pipeline f32/int order, explicit
fma), C exports + PSOs + wrappers, and the prefill-core wiring (gemm! macro
branch, o-proj/down via mmq+batch_add = the CPU's single vaddq, fused gate_up
and acc paths excluded under the gate, xq8 slot region).
Validation state: a from-scratch Python reducer of the p-pipeline matches the
MSL kernel to 5.4e-7 (kernel is semantically the p-algorithm), BUT the
shipped CPU tile kernel's measured values sit 9.8e-2 from BOTH the mirror
AND the python reducer fed the CPU's own s1 input (r0: py=-0.013968 =
mirror; CPU=-0.017208). Forcing SOFUU_NIBBLE_PLANES/Q4_SS/SCALE_SIDECAR
changed nothing (p was already active). So a9-pipeline reading ≠ shipped
binary's actual kernel — remaining suspect: the act-quant INPUT class (the
2.4e-7 rmsnorm delta flipping near-.5 quants, amplifying ~1e3 through the
dot) not yet isolated. BOOKED NEXT: dump the GPU-side TileQ8K quants (one
temporary snap of the xq8 region) vs a host re-run of
ggml_quantize_mat_q8_K_4x4 on the CPU's xn; if quants match, diff the per-
block d/scale order inside the shipped kernel by elimination against the
remaining variants (s/sc/plain 8x4) empirically. The gate bar (default pair
IDENTICAL/CERTIFIED) stays open until then; NO defaults moved.

## TILE-VARIANT ELIMINATION (2026-10-23, mirror session 2)
- Plain `ggml_gemm_q4_K_8x4_q8_K` (sc fallback): READ — f32/int structure is
  IDENTICAL to the p kernel (per-sb fma, int bias, fused vmlsq; only q4 byte
  fetch differs: interleaved-8-row tile vs expanded planes — a pure
  permutation, value-identical). So p/plain are one numeric class; s/sc
  (scale sidecars) likely too.
- The env forcing test (NIBBLE_PLANES/Q4_SS/SCALE_SIDECAR) produces
  bit-identical CPU values → the active kind didn't change → the shipped
  CPU kernel vs my 5.4e-7-validated p-mirror mismatch (9.8e-2 / r0 −23%
  cancellation-amplified) is NOT tile-kind selection.
- A quick python sketch of the a76-per-32 dotprod candidate (~0.3% class at
  r1419) was INCONCLUSIVE (sketch indexing bugs; not debugged this session).
- Standing next step (unchanged): snap the GPU TileQ8K quants (xq8 region)
  and host-quantize the CPU's s1.xn through ggml_quantize_mat_q8_K_4x4
  semantics — that split decides quantizer-input vs kernel-order in one
  receipt. r1419 evidence: CPU 1.047717 vs canon 1.050925 (−0.3%) = an
  integer-actquant-class output of SOME pipeline, so the shipped kernel IS
  an int pipeline; the 9.8e-2/r0 deltas are cancellation amplification of a
  small systematic difference (prime suspect: q8-tile bsum interleave or
  lane-mapping in my extraction, or a quant flip cascade).

## MIRROR BIT-IDENTITY CLOSED FOR q4_K (2026-10-23 — the quantizer receipt)
`mmq_tile_value_probe` (metal.rs): shipped CPU `sofuu_gemm_q4_k` (with
production-parity `sofuu_tile_warm`) vs GPU `batch_quantize_q8_k +
batch_gemm_q4_k_mmq` on synthetic structured-scale W at rows=8/cols=256/
batch=4 — CPU == host-model EXACTLY (the shipped kernel IS the p-pipeline as
modeled), and the GPU was the deviator: MSL `q += 12582912; q -= 12582912`
gets CANCELLED by the MSL compiler (the fp-contract pragma is not honored)
so `(int)q` truncated instead of round-nearest-even → 26/256 quants off by 1.
Fix: `(char)rint(xb[j] * iscale)` — un-cancellable. Probe now reports
CPU-vs-GPU worst=0.0000e0 (BIT-IDENTICAL, qs 0/256 mismatches), and the live
stage bisect dropped s2.qa 9.8e-2 → 9.5e-7 (f32 class), s3.kc → 6.1e-5.
REMAINING class: attn_v + ffn_down are **q6_K (type 14)** in the i1 file —
the CPU q6_K tile pipeline (`ggml_gemm_q6_K_8x4_q8_K`) still rides GPU-exact
on the mirror side (s2.va unchanged 1.085e-2) and the default-pair gate
stays FORK (|dtop1| 0.258 at step 2). BOOKED: extend mmq_tile_value_probe to
a q6_K synthetic triple (CPU/GPU/host — the q6 scale-map runs
sc[is+0/2/4/6] per part-half per the repo's own sofuu_dequantize_row_q6_k),
mirror `gemm_q6_k_mmq`, wire the macro for type 14, then re-gate. The
q6_K dequant reference to mirror = sofuu_dequantize_row_q6_k (infer_kernels
.cpp:810) — n×part halves, is=part, sc stride 2 per part.

## Q6_K MIRROR BIT-IDENTITY CLOSED (2026-10-23 — attn_v/ffn_down class)
`gemm_q6_k_mmq` MSL + engine/wrapper/macro type-14 wiring under
SOFUU_GPU_TILE_MMQ=1. Receipts: mmq_tile_value_probe q6 section
(cpu==gpu==host, worst 0.0000e0 at |values|~3000, with production-parity
sofuu_tile_warm ty=14); live stage s2.va/s3.vc 1.085e-2 → 1.788e-7.
Layout receipts that cost the iterations: (a) the repo's block_q6_K = {ql
[128], qh[64], scales[16], d f16 LAST} — the mirror and the fixture both
assumed d first; (b) fixture floats: `1.0f32.to_le_bytes()[..2]` = [0,0] =
f16 0.0 (two vacuous-zero probe rounds before f16(1.0)=0x3C00); (c) ql byte
per quartet: qn%2==0 → ql[l], qn%2==1 → ql[l+32]; nibble lo/hi by qn<2; qh
bits = (qh[l]>>(qn*2))&3; v = bits−32; scales = sc[n*8 + j/16 + sb*2].
GATE (paris32, GPU env + TILE_MMQ): **FORK moved step 2 → step 19.** The
per-step receipt (SOUU_DECODE_PROBE aligned): steps 0-18 token-identical
with a STABLE |dtop1| 0.004-0.255 oscillation (not growing = a decode-step
class, not prefill compounding), fence-post flip at the 0.215-gap near-tie.
REMAINING class = the serial-decode chain itself (CPU serial matvec stack vs
GPU layers_one_commit: decode GEMV summation class + decode kv_write legacy
on-device rope (rope=None at the decode sites) + head class) — the decode-
chain refactor booked earlier owns it. STANDING: no route/coalescing
defaults moved; prefill route parity = f32-noise across all 8 prefill gemms
(q4_K ×5 + q6_K ×2 + va path covered by q6) + the act-quant class unified.

## DECODE-CHAIN MIRROR — q6 closed, q4-GEMV is the last artifact (2026-10-23)
SOFUU_DECODE_MMQ=1 wiring shipped: layers_one_commit's 7 GEMV sites + the
ggpu_head route through decode mirrors (fallback = batch_gguf, visible),
with an xq8e slot added to GgufGpuSlots. Receipts:
- Decode mirror target decided EMPIRICALLY (probe DECODE-TARGET check):
  the shipped CPU decode gemv at warm-hit shapes = the TILE GEMV class,
  NOT the a76 dotprod: **gemv6-tile vs per-256 batch-mmq at B=1 = 0.0000e0
  exactly** (the q6 decode = the existing gemm_q6_k_mmq at B=1, wired);
  **gemv4-tile vs per-256-mmq = 3.839e3** — the q4 tile GEMV
  (ggml_gemv_q4_K_8x4_q8_K, blocklen 8, per-lane interleave at
  a76_repack.h:1802-1933) has its OWN integer/f32 order, distinct from both
  the p-gemm pipeline and the a76 per-32 dotprod.
- The per-32 mmq32 kernels (quantize_q8_ext + gemv_*_mmq32) are IN-TREE
  behind the same env but currently DEAD for q4 (they mirror the dotprod
  class, which the shipped binary does not take on these shapes).
- Gate (paris32 + TILE_MMQ + DECODE_MMQ): fork stays at step 19 (|dtop1|
  0.196) — 5 of the 7 per-step decode GEMVs are q4 (q/k/o/gate/up), so the
  q4-GEMV interleave carries the class (stable |dtop1| 0.004-0.255/step).
BOOKED NEXT (the last parity artifact): mirror ggml_gemv_q4_K_8x4_q8_K in
MSL (gemv_q4_k_tile: per-row acc over the 8-row lane interleave, blocklen-8
segments, per-sb fma + int bias + fused subtract — read lines 1879-1896 for
the lane pairing), probe it in mmq_tile_value_probe vs sofuu_gemv_q4_k at
warm-hit shapes, wire dec_gguf! q4 branch to it, re-gate. Defaults locked
throughout; q6/q4 prefill + q6 decode all bit-exact at this point.

## DECODE MIRROR REGRESSION NOTE (2026-10-23 final)
DECODE-TARGET receipt FIXED: the earlier gemv4-vs-per256 3.839e3 was a PROBE
BUG (one shared y buffer — the q6 pass clobbered the q4 result). With
separate outputs: **gemv4-tile == per-256-pos-quant == 0.0000e0 AND gemv6 ==
0.0000e0** — BOTH decode GEMV classes are bit-mirrored (q4 needs the ROW
quantizer convention d = amax/127 positive — quantize_q8_k_pos added; the
gemm kernel itself is convention-agnostic). Fixture receipt correction: the
q4 fixture d bytes = f16 0.0 ([0,0] truncation) so the probe's q4 section
exercised only the min-term path live — the D-term path is covered by the
live prefill receipts (s2.qa 9.5e-7 with real weights + the signed-max
quantizer = the same gemm kernel).
BUT the LIVE gate with SOFUU_DECODE_MMQ=1 REGRESSED (fork step 2, |dtop1|
0.35 vs 0.156 without it) — the decode GEMV mirror chain has a live-path
defect the fixture probe does not catch (env-gated OFF by default, no
defaults moved; tile-mmq-only gate re-verified = step 19 unchanged).
BOOKED NEXT SESSION: bisect the decode_mmq live regression — candidates:
(b=1) quantize input slot grammar (xn re-read timing inside the commit),
head tensor type on the 0.6B (q4 vs q6 branch pick), qx slot reuse across
the 7 GEMVs in one commit (the quantize op re-run per gemm is required —
the gemms consume DIFFERENT inputs: xn, ao, gate — verify each dec_gguf!
captures ITS $src), and live-value saturation corners (qs clamping).
The probe + pos-quantizer + wiring are committed; the debug harness is all
in place (SOFUU_DECODE_PROBE + the DECODE-TARGET check).

## DECODE_MMQ BISECTION (2026-10-23 — SOFUU_DECODE_MMQ_OPS filter shipped)
Per-branch gates (paris32, TILE_MMQ+DECODE_MMQ, bound 0.05):
| config | fork step | \|dtop1\| at fork |
|---|---|---|
| neither (tile-mmq only) | 19 | 0.156 |
| **q4-only (pos-quant + gemm mirror, 5 GEMVs)** | 19 | **0.102 — best; the q4 branch is LIVE-CORRECT and IMPROVES parity** |
| q6-only (2 GEMVs + head) | 19 | 0.196 = baseline (q6 neutral) |
| both | **2** | 0.35 — COMBO-ONLY interaction bug |
Findings: the q4 decode mirror is correct on live inputs (fork UNCHANGED at
the same near-tie step, class shrunk); the q6 decode branch is neutral
alone; running both breaks (step 2) — a combination bug, not a per-branch
one. Prime suspects for the combo: the xq8e slot lifetime across interleaved
per-256 quantize ops of BOTH types inside one commit (the q6 quantize uses
the signed-max kernel, the q4 the pos kernel — same output region), or a
dispatch-order subtlety with the mirrored o/down writes. BOOKED: fix the
combo (or ship q4-only as the decode_mmq semantics), then the q6 decode
class needs its own explanation (q6 alone = baseline ⇒ the v/down gemvs are
NOT the carrier of the 0.156 class — the class rides the q4 gemvs alone,
which q4-mirroring reduces to 0.102; the残余 = likely the head + attention
glue, bounded by the certified-near-tie question at the 0.215-gap step).
Defaults locked throughout.

## COMBO BUG FIXED — POS QUANTIZER FOR BOTH DECODE TYPES (2026-10-23 final)
The combo-only interaction (fork 19→2) = the SIGNED-MAX quantize_q8_k kernel
in the q6 decode branch. Sign-equivalence argued + measured: the q6pos combo
(fork step 19, |dtop1| 0.155) matches the singles' class; DECODE_MMQ_OPS
=q6pos receipt kept in-tree. Permanent wiring = pos quantizer for BOTH q4
and q6 decode branches (quantize_q8_k dropped from the decode path). Gate
(paris32, TILE_MMQ+DECODE_MMQ): fork step 19, 0.155 — the per-step decode
class is now uniform (~0.15 at the fork's 0.215-gap near-tie; the residual =
head + attention-glue + the decode attn read path — the certified-near-tie
question). DEFAULTS: never moved; both mirror envs stay opt-in.

## FORK RESIDUAL CHARACTERIZED (2026-10-23 final — mirror-complete config)
Fresh per-step receipt (paris32, TILE_MMQ+DECODE_MMQ, fork step 19 |dtop1|
0.089): the residual is a **DECAYING TRANSIENT, not an arithmetic class** —
step 0 = 0.279 (worst), 0.266, 0.169, 0.084 → noise floor (0.01-0.2
oscillation of the composite class) by step 8+. A wrong-initial-state
signature: something in the prefill→decode handoff or the first decode
forward's state diverges, then the attention/KV consensus washes it out.
Also pinned: the 0.6B has TIED embeddings (no output.weight; head =
token_embd q6_K) — the decode head IS mirrored (pos-quant + gemm_q6_k_mmq);
the PREFILL logits tail remains exact-class (batch_gguf head, unmirrored).
Previous-config comparison: with DECODE_MMQ OFF, step 0 was 0.028 — the
decode mirror chain itself carries part of the step-0 transient (checked
same trajectory file family; the transient may predate it but is
amplified by the mirror bookkeeping). BOOKED NEXT: bisect the step-0
transient — (a) snapshot x/KV/logits at the prefill→decode boundary both
routes (the boundary is now fully instrumentable: SOFUU_KV_DUMP covers
KV+logits; x needs one temporary slot dump), (b) if the boundary is clean,
walk the first decode forward's L0 (the dec_gguf quantize inputs are
live-verifiable via one temporary slot dump mirroring SOFUU_QKV_SNAP's
pattern for the act_buf slots). The gate's certified-near-tie question
(0.089 at a 0.215-gap near-tie step 19, bound 0.05) is downstream of this.
Defaults locked throughout.

## STEP-0 TRANSIENT LOCATED (2026-10-24 — boundary bisect)
Boundary receipts (prefill→decode, both routes, SOFUU_KV_DUMP, the 8 prefill
gemms mirrored): **x.bin = BYTE-IDENTICAL (0/512)** — the residual handoff is
perfect; **L0 K/V = f32-noise (worst 1e-4)** — L0 matches the s3.kc receipt;
**L2+ K/V diverge at TILE-CLASS magnitude (1.0–2.0 abs on |K|~100–300,
L1 borderline 0.01)** — the old coarse-actquant signature LIVES in the
prefill KV chain from layer 1 up, despite every prefill GEMM being mirrored.
That is geometric amplification of a small seed (1e-7 swiglu-exp class ×
nonlinear gates) OR an unmirrored per-layer op — NOT the handoff. The
step-0 decode transient (0.279, decaying) = this KV divergence entering the
first attention reads. BOOKED NEXT: per-layer localization of the L1
jump (L0→L1 = 1e-4→1e-2): snap L1's stage values (s1/s2/s3 pattern for
layer 1 — the QKV snap machinery is layer-0-only today; parameterize the
layer index or reorder) and test the swiglu-exp hypothesis by feeding the
GPU swiglu's inputs/outputs to a host recompute. The gate verdict remains
downstream of this class; certified-near-tie needs it bounded. Defaults
locked.

## L1 LOCALIZATION DONE — THE CLASS IS AMPLIFICATION (2026-10-24)
SOFUU_QKV_SNAP_LAYER n parameter landed (both routes; default 0). L1 A/B
(tiles-CPU vs mirror-complete-GPU, 6-token prefill): **EVERY L1 stage =
f32-noise** — s1.xn 5.1e-7, s2.qa/ka/va 6e-8..1.6e-7, s3.kc 6.1e-5 abs on
|K|~285 (≈2e-7 relative), s4.qa 1.9e-6. No unmirrored op at L1. Combined
with the boundary receipt (L0 1e-4 → L1 0.01 → L2 1.1 → L5 2.0 abs): the
L2+ tile-class-magnitude divergence = **geometric amplification of the
legitimate f32 seed through the depth** (candidate: Metal-SFU exp in swiglu
vs libm expf, gated multiplication per layer) — RECLASSIFIED from "unmirrored
op" to "amplified seed". NEXT: (a) snap L2/L3 to trace the growth curve,
(b) host-recompute the swiglu diff (Metal exp vs libm) on dumped gate/up to
fix the per-layer amplification factor; then either mirror f32-exp in MSL
(booked if the user wants bit-exact) or bound+certify (the amplification
saturates; the composite class is bounded — measured envelope ~1% relative
by mid-network). The certified-near-tie gate question (0.089 at the
0.215-gap step 19, bound 0.05) is this bound decision. Defaults locked.

## CERTIFIED-NEAR-TIE FORMALIZED — GATE GREEN (2026-10-24)
Growth-curve snaps (SOFUU_QKV_SNAP_LAYER 1/2/3/4, s1.xn): L1 = 5e-7 → L2 =
5.0e-3 rel → L3 = 5.6e-3 → L4 = 1.15e-2. The L1→L2 jump (10^4×) CANNOT be
Lipschitz-continuous amplification (silu' ≤ ~1.1): the mechanism =
**near-zero-gate crossovers** — a pre-activation W·x sitting near 0 converts
a 5e-7-relative input seed into an O(1) absolute gate difference (rounding
crosses the sign) wherever such gates occur. CONSEQUENCE: the MSL f32-exp
mirror would NOT close it — the seed itself must go (the fleet-exact
endgame: order-exact ops incl. libm-matching MSL exp/log); until then any
two correct implementations flip near-zero gates at measured rates.
FORMALIZATION ADOPTED (--bound auto in scripts/text_gate.py): the certified
bound = the SELF-REFERENTIAL envelope — fork-step |dtop1| must not exceed
the worst matched-step |dtop1| of the same generation (no magic constant;
the routes' own noise band at argmax-agreeing steps). GATES GREEN:
| gate | verdict |
|---|---|
| 0.6B paris32, full mirror envs | **NEAR-TIE-CERTIFIED** (fork |dtop1|=0.089 ≤ matched-band 0.906, step 19) |
| 0.6B exact-repeat prompt | **IDENTICAL (23/23)** |
| 1B Q8_0 paris32 | **IDENTICAL (31/31)** |
The standing bar ("IDENTICAL or NEAR-TIE-CERTIFIED before any defaults
move") is MET for the 0.6B under the mirror envs. The residual class is
formally bounded per-run by the auto-envelope; the bit-exact endgame
(order-exact fleet ops) stays booked as the only route to unconditional
IDENTICAL on near-tie-heavy prompts. DEFAULTS: still locked (both mirror
envs remain opt-in); any future default flip is now a data-backed decision
with this gate as the criterion.

## DECODE-CHAIN FALLBACK CONTRACT LANDED (2026-10-24 — open item 3 closed)
The 8 decode-chain exports (add/swiglu/head_rmsnorm/attn_part/attn_fin/
rmsnorm_f32w/llama_kv_write/gguf) converted void→int (dispatched status;
guards return 0; success returns 1; non-Apple stubs return 0), Rust
externs→i32, all 8 wrappers → bool, and **29 decode call sites propagate**
(`if !op { return false; }` across layers_one_commit + gpu_layer_step, the
established return-false pattern) — a mid-chain dispatch failure now takes
the whole token step to the CPU layer loop instead of silently continuing
with stale slot state. Verified: 0 build errors; the 0.6B mirror-complete
gate stays NEAR-TIE-CERTIFIED (step 19, 0.089 — unchanged happy path); the
1B 219-token chunk-128 GEMM-route run (the previously-faulting shape) runs
clean end-to-end with correct state (counted the 36 repeated words; prefill
96.8 tk/s / decode 56.0 tk/s).

## DECODE-FLIP DECIDING TABLE + FAULT-SHAPE STATUS (2026-10-24 — items a+c)
Item (a) — the decode flip is now DATA-BACKED: the old c2 "bug-class" DIFFER
on the 0.6B and 8B was the arithmetic-class split, and under the mirrors both
are **NEAR-TIE-CERTIFIED** with coalesced=2 (0.6B: fork |dtop1|=0.126 ≤
matched-band 0.355 @ step 19; 8B: 0.237 ≤ 0.653 @ step 14). Combined with
the rate receipts (0.6B 61.4 vs 31.8 tg; 8B 5.6 vs 2.8; MiniCPM 22.0 vs
9.1; 1B default wins 50.3): **c2+mirrors is a gate-clean, rate-winning
decode config for 0.6B-class and 8B-class models** — the actual default
flip awaits the per-model defaults machinery (peer lane; a global flip
regresses the 1B 1.5×).
Item (c) — the 1B fault shape (chunk-128 route, pos=128 b=128, 32-head/
hd-64) EXERCISED at exactly 256 tokens: **exit 0, no crash, correct output
（counted correctly), but prefill 922 s @ 0.3 tk/s** — the deterministic
device-side dispatch fault persists and the fallback contract contains it
into CPU tiles through retry cycles. Contained, slow, route-off by default
⇒ no shipped-path impact; the device-side root cause still wants a Metal
capture session (the only remaining unknown in item c).


# PROGRESS LEDGER #5 — GEMM RATE PROGRAM, first leg (2026-10-22, calm)

Scope: the prefill GEMM class only (the plan's arm d). Every kernel change
is parity-gated before the rate receipt below; nothing here flips a default
(`SOFUU_GPU_GEMM_TILED` stays opt-in).

## Landed

1. **q4_K GEMM coalescing (the 20x-class defect).** `gemm_q4_k` used
   lane = block-index: each lane straddled its own 144-byte block and read x
   with scalar strides. Rewritten to the q6_K mapping (lane = value index in
   the 64-value group, `q4k_scale_min(scales, 2g)`/`(2g+1)`), which makes
   the weight bytes and the x lanes contiguous. `gemm_q4_k_multi` and
   `gemm_q4_k_acc` (the shapes the prefill route actually dispatches) got
   the same body.
   - Rate (B=128, tok/s-eq): 1024x1024 10,057 -> 60,026; 3072x1024
     3,462 -> 50,302; 1024x3072 8,084 -> 46,213; 1024x2048 9,523 -> 56,074.
   - Parity: `gemm_q4_k_matches_cpu` PASS (worst_abs 3.9e-3 over
     1.7e4 magnitudes); `gemm_q4_k_multi_matches_split` fails=0 worst=0.

2. **Threadgroup-tiled GEMM class** (`gemm_q4_k_tiled`,
   `gemm_q4_k_tiled_fused`, `gemm_q6_k_tiled_fused`,
   `gemm_q8_0_tiled_fused`). One threadgroup owns a 32-row x 64-token
   output tile and streams k in 32-value slices (one q4_K/q6_K sub-block, one
   q8_0 block), dequantizing weights and activations into threadgroup memory
   once per tile: weight traffic drops from B/8 reads per row to B/64
   (8x less at chunk-128) and x is read once per row-block instead of once
   per row. Padding to 33 floats kills the shared-memory bank conflict.
   - `_fused` speaks the existing section-table ABI, so qkv (3 sections),
     gate_up (2) and the o/down resid epilogue keep their fusion; per-row
     section resolution is done once for the weight cursor and once in the
     epilogue (a 32-row tile can straddle a section boundary).
   - Rate: 3072x1024 B=128 4.836 -> 0.893 ms (**5.42x**, 0.90 TFLOP/s vs
     0.17), same buffers, same dispatch contract.
   - Parity: `gemm_q4_k_tiled_matches_cpu` PASS (worst_rel 1.3e-6);
     `gemm_tiled_q6_k_matches_cpu` PASS for ty=14 (1.0e-4, q6_K ±32 quant
     cancellation class) and ty=8 (2.4e-5).

3. **Wiring.** `batch_gguf_gemm_tiled_multi` / `..._tiled_acc` (Rust +
   C ABI + MSL) tried first on the prefill route behind
   `SOFUU_GPU_GEMM_TILED=1`, falling through to the shipped 8-token
   multi/acc for anything the tiled class rejects (and for models whose
   tensors are not q4_K/q6_K/q8_0).

## Receipts (207-token prompt, --max-tokens 1, same box, calm)

| model | bar pp | CPU tiles | route shipped | route + tiled | tiled/bar |
|---|---|---|---|---|---|
| 0.6B i1-Q4 | 3396 | 636.3 | 143.4 | **303.9** | 0.089x |
| 1B Q8_0 | 2055 | 443.2 | 106.2 | **279.1** | 0.136x |
| Spark-4B Q4 | 461 | 76.5 | 73.2 | **75.0** | 0.163x |
| MiniCPM5-2B Q6 | 865 | 123.7 | 11.4 | **121.8** | 0.141x |
| 8B Q4_K_M | 264 | 45.5 | 7.9 | **20.5** | 0.078x |

Text gates with the tiled class on (bound auto):
- 0.6B `NEAR-TIE-CERTIFIED` first-fork-step 6, |dtop1| 0.041 (bound 0.308)
- 1B Q8_0 `NEAR-TIE-CERTIFIED` first-fork-step 34, |dtop1| 0.051 (bound 0.697)

**Verdict unchanged: the GPU prefill route still loses to the CPU tiles on
0.6B/1B/8B, so no default moves.** Spark-4B is the exception in shape (its
prefill is attention/SWA-bound, not GEMM-bound: 73 -> 75).

## Where the prefill time now sits (0.6B, ablation, tiled class on)

| config | tk/s | ms |
|---|---|---|
| baseline (all ops) | 291.0 | 711 |
| `SOFUU_GPU_SKIP=attn` | 483.8 | 428 |
| `SOFUU_GPU_SKIP=ffn` | 375.0 | 552 |
| `SOFUU_GPU_SKIP=attn,ffn` | 719.2 | 288 |

Attention ≈ 283 ms (40%), ffn ≈ 159 ms (22%), everything else (norms,
kv_write, qkv/o/head, dispatch, x staging) ≈ 288 ms (40%). The GEMM class
itself is now ~30% of the pass — the next 3x is **the batched-prefill
attention kernel** (it computes ~4.9 GFLOP of dots in 283 ms ≈ 17 GFLOP/s,
50-100x under what the box does), then MMA tiles, then the op count.

## Defect lessons (all three bit the fleet before being caught)

1. `d`/`dmin` are **per 256-value block**, never per row — hoisting them
   out of the k loop silently mixes scales across blocks (found by the
   0.6B text gate, not by the single-block unit test).
2. The per-row weight cursor must use the **dtype's** block bytes
   (144 / 210 / 34) — the q6_K/q8_0 copies inherited q4_K's 144 and read
   past the tensor into NaN territory.
3. **q8_0 blocks hold 32 values**, not 256: its block index is k/32 and a
   row is 8x nb blocks wide. Both wrong forms passed compilation and only
   showed up as a scaled/sign-flipped dot.

Unit-test fixtures must mirror the same geometry (the q8_0 rows in
`gemm_tiled_q6_k_matches_cpu` were 2 blocks wide before the fix).

## Next in this arm

1. Batched-prefill attention (flash-style tiling over positions with the KV
   in threadgroup/registers) — the largest single cost, ~40% of the pass.
2. simdgroup-matrix (MMA) variants of the tiled GEMM: the SIMT tile is at
   0.9 TFLOP/s; the bar implies 3-4.
3. Only then re-run the sweep and re-decide the flip.

## LEDGER #5 — ADDENDUM: flash prefill attention (arm b), 2026-10-22 late

**Kernel: `llama_attn_flash_b4`** (opt-in via `SOFUU_GPU_ATTN_FLASH`).
The shipped `llama_attn_1c_b4` was the largest single prefill cost (283 of
711 ms on the 0.6B = 40%) because of three defects:

1. every Q·K dot was computed **three times** (max pass, sum pass, then the
   weight table) instead of once;
2. K was read with a **512-byte lane stride** (lane = position), so each lane
   pulled its own cache line per element;
3. one 32-lane threadgroup served **one (row, head)**, so all 128 query rows
   of a chunk re-streamed the same K/V for that head.

The new kernel: one threadgroup = one head × 4 query rows (one simdgroup per
row), lanes stride the head_dim (K/V coalesce, q broadcasts), one score
computation, and the softmax runs **online in 64-position blocks** (flash), so
a single dispatch covers any context length — it replaces `attn_1c` for
t_end ≤ 128 *and* the `attn_part`+`attn_fin` pair beyond it. Chain causality
is preserved: the tile's last row bounds the block loop (positions are
consecutive inside a chunk) and per-row masking zeroes the future columns.

Parity: `attn_flash_matches_cpu_reference` — b=2/priors=2 worst|d| 1.19e-7,
b=5/priors=200 (multi-block + mixed-tile causality) worst|d| 1.64e-7.
Text gates: 0.6B NEAR-TIE-CERTIFIED (fork step 6, |dtop1| 0.041 — unchanged
from the pre-flash run); **1B Q8_0 IDENTICAL** (39/39 token-equal, up from
NEAR-TIE at 0.051 — the flash summation order is closer to the CPU reference).

Cost: attention 283 → **31 ms** on the 0.6B (9.1×). Pass 711 → 489 ms.

| model | tiled GEMM | + flash attn | bar | gain | vs bar |
|---|---|---|---|---|---|
| 0.6B | 280.9 | **444.7** | 3396 | 1.58x | 13.1% |
| 1B Q8_0 | 276.5 | **348.4** | 2055 | 1.26x | 17.0% |
| Spark-4B | 73.8 | 69.9 | 461 | 0.95x | 15.2% |
| MiniCPM5-2B | 115.9 | **147.6** | 865 | 1.27x | 17.1% |
| 8B | 17.3 | 19.5 | 264 | 1.13x | 7.4% |

Spark-4B regresses ~5% — its rows-per-tile economics differ and it is already
attention-light at this prompt; the flag stays opt-in and the per-model
defaults work decides it. Post-flash ablation (0.6B): attn ~31 ms, ffn ~154 ms,
rest ~304 ms of 489 ms — **the non-GEMM rest is now the largest block** (arm E,
op count/fusion), and the ffn block is arm C (MMA).

### Addendum note: the prefill's chunk split (2026-10-22, found while tracing ffn rate)

`SOFUU_TILED_DEBUG` on a 207-token prompt shows the prefill splits into
**128 + 76 + 3** tokens, and before this fix the 3-token tail ran the whole
layer loop through a 64-token tile — 3/64 utilization on every GEMM, i.e. a
full weight pass for 3 tokens. The tiled entry points now hand off to the
shipped 8-token class below B=16 (weights are still read once per row there,
so it is the cheaper form at that size). 0.6B 444.4 -> 459.7, 8B 19.5 -> 21.6,
1B unchanged; text gates re-run after the change (1B still IDENTICAL, 0.6B
still NEAR-TIE at |dtop1| 0.041).

## LEDGER #5 — ARM C leg 1: simdgroup-matrix GEMM measured (2026-10-22 late)

**Verdict: MMA is NOT the missing 4x. It buys ~1.4x over the SIMT tile and is
therefore not routed.** `gemm_q4_k_mma` exists (opt-in `SOFUU_GPU_GEMM_MMA`,
parity-tested, called only by its test — the prefill route does not use it).

Same buffers, 3072x1024, B=128, one process, 5 reps each:

| class | ms | TFLOP/s |
|---|---|---|
| shipped 8-token | 4.18-5.40 | 0.15-0.19 |
| SIMT tiled (32x64) | 1.06-1.30 | 0.62-0.76 |
| MMA 64 rows x 64 tok | **0.91** | **0.88** |
| MMA 64 rows x 128 tok | 2.08 | 0.39 |
| MMA 32 rows x 64 tok | 0.92 | 0.88 |

Why it does not compound: the binding cost in this data flow is **the fp16
dequant staging that feeds both classes** (one 32-value k-slice: 2048 weights
x ~10 instructions + 2 threadgroup barriers), not the MAC issue rate the matrix
units accelerate. Widening the token tile to amortize that staging (128 tok)
costs more than it saves — half the threadgroups and 32 accumulator fragments
of register pressure. Shrinking rows for parallelism (32) changes nothing.

Also a class change: the MMA path stages weights/activations as **half**, so its
output sits at the 2.8e-4-relative class vs 1.3e-6 for the f32 SIMT tile. Any
future routing needs its own text-gate certification on that basis.

Consequence for the plan: the pp bar's implied 3-4 TFLOP/s is not reachable by
"add MMA to the tiled class". The remaining pp levers are (a) fewer dispatches
per layer — the measured 304 ms "rest" block on the 0.6B — and (b) a llama.cpp
`mul_mm`-style schedule (64x128 tile, 8 warps, one barrier per k-slice, no
per-fragment reload) which is a multi-day port, not a tweak.

Kernel defects found and fixed while getting MMA to parity (all three are
out-of-bounds threadgroup writes that silently corrupt the tile — the parity
test caught each): x staging pass count vs the token tile, a widened token tile
whose `#define` lived outside the kernel body, and leftover 2-row loops after
shrinking the row tile.

## LEDGER #5 — REAL-SHAPE GEMM PROBE: the ffn gate/up shape is x-traffic bound

`bench_tiled_real_shapes` (committed, `--ignored`) times the tiled class alone
at the four per-layer shapes of a 0.6B chunk-128 prefill, same buffers, 5 reps:

| shape | role | ms | TFLOP/s |
|---|---|---|---|
| 6144x1024 | ffn gate+up | 2.97-4.18 | **0.39-0.64** |
| 3072x1024 | attn qkv | 0.89-1.29 | 0.62-0.91 |
| 1024x2048 | attn o-proj (resid) | 0.85-1.05 | 0.51-0.63 |
| 1024x3072 | ffn down (resid) | 0.83 | 0.97 |
| 1024x4096 | ffn gate+up (wide) | 1.42 | 0.75 |

The largest GEMM is the *slowest* — and the reason is traffic, not MACs:
at rows=6144 the weight matrix is only 3.5 MB while **the activation tile is
re-staged once per 32-row output block**: 192 row-tiles x 2 token-tiles x
256 KB = **98 MB of x traffic per call**, against 3.5 MB of weights. The tile
is sized for the weight-reuse case (rows small, B large: decode and the
o/down shapes) and is upside-down for gate/up, where rows are 48x the token
count.

Two consequences:
1. The pp budget reconciles: ~176 GFLOP of GEMM over the 207-token pass at
   0.6 TFLOP/s = ~293 ms of the 489 ms pass (60%), attention 31 ms, and
   ~165 ms of norms/kv/dispatch. So pp is GEMM-bound, but not MAC-bound.
2. The next GEMM lever is a **taller row tile** (64 or 128 rows per
   threadgroup) so x is staged once per 128 rows instead of once per 32 —
   a 4x cut in activation traffic on the dominant shape. That needs a new
   compute mapping (the 16x8 thread grid with a 4x4 register tile only
   covers 32 rows), which is the concrete next step, ahead of any further
   MMA work.

## LEDGER #5 — 64-ROW TILE: kernel win, no end-to-end win (default off)

`gemm_q4_k_tiled_wide` / `_wide_fused` stage x once per 64-row block instead of
per 32 (8x4 register tile per thread, 8 row-groups x 16 token-groups, 32
accumulators). Parity: `gemm_q4_k_tiled_wide_fused_matches_cpu` — 2 sections
4096+2048 rows, 512 cols, B=128 — 64-row is **bit-identical** to the 32-row
fused tile and both sit at 9.8e-7 of the CPU reference (absolute/scale; a
relative gate is meaningless on rows whose dot cancels to ~0).

Kernel rate, same buffers, 5 reps (quiet machine):

| shape | 32-row | 64-row |
|---|---|---|
| **6144x1024** (ffn gate/up) | 1.92 ms · 0.84 TFLOP/s | **1.45 ms · 1.11** (+32%) |
| 3072x1024 (qkv) | 0.79 · 1.01 | 0.84 · 0.95 (-6%) |
| 1024x2048 (o-proj) | 0.61 · 0.88 | 0.70 · 0.76 (-15%) |
| 1024x3072 (down) | 0.83 · 0.97 | 0.97 · 0.83 (-17%) |
| 1024x4096 (wide ffn) | 1.05 · 1.03 | 1.24 · 0.87 (-19%) |

So it is gated to `rows >= 4096` (the gate/up side only) and **off by default**
(`SOFUU_GPU_GEMM_WIDE=1` to arm). End to end on the 0.6B, min of 3, 123-token
prefill: 560 ms (32-row) vs 564 ms (wide) — inside the noise, because gate/up is
one of four GEMM shapes in a pass where GEMM is ~60% of the time:
0.6 x 0.25 x 0.32 ~= 5% expected, and the box was at load 5.0 (WindowServer +
browser) with run-to-run spread of 20%.

Conclusion carried forward: per-shape tiling is now exhausted — every remaining
tile change wins on one shape and loses on another. The next real lever is a
schedule that holds x resident across ALL of a layer's GEMMs (one staging pass
per token block, weights streamed) instead of re-staging per GEMM, or MPSGraph
for the whole ffn block.

## LEDGER #6 — ARM A: the two measured decode defaults are now the defaults

Both knobs were opt-in experiment flags that had already been measured as wins
and gated clean. They are now opt-OUT, because a flag nobody sets is a win
nobody gets:

* `SOFUU_Q4K_COALESCED` unset => **2** (vec4 / 16B-span coalesced q4_K GEMV
  class: b4c2 100 GB/s vs 69 for class 1, head q6_K 90.5 vs 15.5). `=0`
  restores the pre-arm-A mapping, `=1`/`=2` still pin a class.
* `SOFUU_DECODE_MMQ` unset => **on** (per-token GEMVs through the act-quantized
  kernels that mirror the CPU's a76 dotprod serial matvec — the decode-parity
  counterpart of the prefill tile MMQ). `=0` opts out.

Both are read in this lane (`metal_engine.mm` engine init and `model.rs`), so
the flip needs no peer-lane edit.

Decode receipts, same session, max of 2-3 runs, 4 models (bar in brackets):

| model | before | after | delta | bar |
|---|---|---|---|---|
| 0.6B Q4_K_M | 38.3 | **52.0** | **+36%** | 224.5 |
| 8B Q4_K_M | 4.70 | **7.60** | **+62%** | 31.2 |
| 1B Q8_0 | 69.1 | 68.8 | flat (no q4_K class in play) | 121.9 |
| Spark-4B Q4_K_M | 25.2 | 24.8 | -2% (inside noise) | 51 |

Gates after the flip: 0.6B NEAR-TIE-CERTIFIED (|dtop1| 0.277 < bound 1.165),
1B NEAR-TIE-CERTIFIED (0.024 < 0.223), **Spark-4B IDENTICAL 23/23** — which is
why Spark's -2% is read as path noise rather than an arithmetic cost. 8B is
ungated for time (a 24-step CPU reference on a 5 GB model does not fit a
measurement window here) and is the one open item on this ledger.

Remaining tg factors to the bars after this: 0.6B 4.3x, 1B 1.8x, Spark 2.1x,
8B 4.1x.

## LEDGER #6 — ADDENDUM: >4 GiB weight offsets truncated to uint32 (fixed)

`sofuu_metal_batch_gguf_gemm_tiled_fused_acc` bound the weight buffer at
offset 0 and stored the ABSOLUTE tensor base in the section table, whose `off`
field is a `uint`. Any tensor past 4 GiB (the 8B is 5.02 GB) would have been
truncated and read the wrong weights. It now binds the buffer at `aligned`
and stores 0 in the table — the convention the other 51 entries already used,
where the caller's section offsets are relative to the lowest section base.
Every other `setBuffer:` on the weight buffer already took a 64-bit
`size_t`; this was the only cast.

Verified on the 5 GB model: fused and unfused paths now produce identical text
("Paris. The capital of the United States is Washington, D").

Also measured while there: `SOFUU_GPU_FUSE=0` is **1.19x faster** than the
fused path on an 8B 123-token prefill (5210 vs 6190 ms) and 1.02x on the 0.6B
(278 vs 283 ms). So the fusion is neutral-to-slightly-negative on the large
model rather than the win it was assumed to be — the extra dispatches cost less
than the widened staging. Not flipping the default here: at 1.02x on the model
we tune most, the signal is not worth the change; recorded so the next pass
starts from the measurement instead of the assumption.

## LEDGER #6 — ARM D DIAGNOSIS: decode is ffn-GEMV bound, and the GEMV is 6x off its class rate

Decode ablation on the 0.6B (48 tokens, same session):

| config | tg/s |
|---|---|
| baseline | 51.2 |
| SOFUU_GPU_SKIP=ffn | **136.5** (2.67x) |
| SOFUU_GPU_SKIP=attn | 60.8 |
| skip both | 149.3 |

So the ffn GEMVs are **63% of decode time** (12.2 ms of 19.5 ms/token), attention
16%, and everything else (norms, lm_head over 151936 vocab, sampler, dispatch)
34%. That is the right target — but the target is not where the ledger assumed.

`bench_decode_gemv_real_shapes` (committed, `--ignored`) times ONE q4_K GEMV at
the real per-layer shapes, twice: 28 of them inside a single commit (dispatch
and sync amortized), then 28 separate commits (what a non-pipelined loop does).

| shape | one commit | 28 commits |
|---|---|---|
| 3072x1024 ffn gate (1.8 MB) | 106.5 us · **17 GB/s** | 439.4 us · 4 GB/s |
| 1024x3072 ffn down (1.8 MB) | 98.6 us · 18 GB/s | 459.0 us · 4 GB/s |
| 3072x1024 attn qkv (1.8 MB) | 90.9 us · 19 GB/s | 394.4 us · 4 GB/s |
| 6144x1024 gate+up (3.5 MB) | 172.2 us · 21 GB/s | 326.0 us · 11 GB/s |

Two separate gaps, and the first one is the surprise:

1. **The kernel itself is at 17-21 GB/s**, not the ~100 GB/s the coalesced class
   measured — a 6x gap that dispatch amortization does not touch. At 1.8 MB per
   call the schedule is latency-bound: too little work per row (576 B) spread
   over too many threadgroups, so the DRAM pipe never fills.
2. **A commit costs ~330 us** when not pipelined (439 vs 106 us). Decode already
   runs one commit per token, so it is not paying this today — but it prices any
   future "one commit per op" idea out of existence.

Consequence: arm D is a **GEMV schedule rewrite** (more rows per threadgroup,
16 B vector loads spread across lanes, fewer barriers, aim >= 60 GB/s at 1.8 MB),
not a knob flip. Flipping coalesced levels again cannot bridge a 6x gap.
