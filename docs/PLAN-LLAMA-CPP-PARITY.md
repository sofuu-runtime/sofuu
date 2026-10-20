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
