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
