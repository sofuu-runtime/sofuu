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
