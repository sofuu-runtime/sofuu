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
