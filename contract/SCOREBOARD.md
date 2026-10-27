# Sofuu performance scoreboard (generated)

**This is the inference-parity engineering bar, not the product claim.**
The bar is llama.cpp's default build (Metal offload on macOS) on the same
machine; the app-level comparison is the two-build benchmark in
`benchmark/`. A failing row is a measurement, not a marketing result — the
parity program that closes it is tracked in `docs/PLAN-LLAMA-CPP-PARITY.md`
(local-only by repo policy).

Every row is a measured receipt; the headline is the **latest valid receipt
per model**, i.e. the newest row that recorded both pp and tg (zero-metric
rows are listed under *unmeasured* instead of being scored). The ledger is
append-only and receipts carry the configuration they were taken under, so
the table is a chronological status, not a normalized benchmark. Generated
by `python3 scripts/scoreboard.py` — **do not hand-edit**.

**6 models measured · 41 receipts · pp parity 0/6 · tg parity 0/6**

| Model | llama.cpp pp | Sofuu pp | pp | llama.cpp tg | Sofuu tg | tg | text | measured |
|---|---:|---:|:--:|---:|---:|:--:|:--:|---|
| `Llama-3.2-1B-Q8_0` | 2,055.1 | 128.5 | ❌ | 121.9 | 56.3 | ❌ | — | 2026-10-22 |
| `Qwen-3-0.6B-Claude-dist-i1` | 3,396.5 | 33.2 | ❌ | 224.5 | 35.7 | ❌ | — | 2026-10-22 |
| `Qwen-3-0.6B-i1-Q4_K_M` | 3,416 | 605 | ❌ | 223 | 23.4 | ❌ | — | 2026-10-20 |
| `Qwen3-8B-Q4_K_M` | 264.3 | 2.4 | ❌ | 31.2 | 3.3 | ❌ | — | 2026-10-22 |
| `Sharp-MiniCPM5-2B-Q6_K_XL` | 865 | 11.5 | ❌ | 73 | 9.9 | ❌ | — | 2026-10-22 |
| `Spark-X2.5-4B-Q4_K_M` | 461.4 | 74.3 | ❌ | 51 | 31.2 | ❌ | — | 2026-10-22 |

`text` = the exact-text gate ran and matched; `—` = not recorded on that receipt.

## Configuration of the headline receipts

- `Llama-3.2-1B-Q8_0` (2026-10-22T00:53): PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,load=1.25
- `Qwen-3-0.6B-Claude-dist-i1` (2026-10-22T00:53): PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,load=1.48
- `Qwen-3-0.6B-i1-Q4_K_M` (2026-10-20T09:18): post-wave-2 sweep, ll load 3.0-13, sofuu rows load 11.2-13.1
- `Qwen3-8B-Q4_K_M` (2026-10-22T00:53): PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,load=2.00
- `Sharp-MiniCPM5-2B-Q6_K_XL` (2026-10-22T00:53): PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,load=1.41
- `Spark-X2.5-4B-Q4_K_M` (2026-10-22T00:53): PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,load=2.27

## Unmeasured receipts

- `Llama-3.2-1B-Ins`: 1 receipt(s), no valid pp+tg pair (last 2026-10-21) — post-minbase-fix pinned receipts (full-text gate, fused defaults ON)
- `Phi-4-mini-reasoning-UD-Q6_K_XL`: 2 receipt(s), no valid pp+tg pair (last 2026-10-22) — PHASE-4 deciding sweep (calm 1.1-2.3, PROMPT-DECIDING); chunk=c128,no-route (loader gate),load=1.28
- `Qwen-3-0.6B-Clau`: 1 receipt(s), no valid pp+tg pair (last 2026-10-21) — post-minbase-fix pinned receipts (full-text gate, fused defaults ON)
- `Qwen-3-0.6B-Claude-distilled-i1`: 2 receipt(s), no valid pp+tg pair (last 2026-10-21) — P3 chunk-32 prefill GEMM control, same window as the 51.1 row
- `Sharp-MiniCPM5-2`: 1 receipt(s), no valid pp+tg pair (last 2026-10-21) — post-minbase-fix pinned receipts (full-text gate, fused defaults ON)
- `Spark-X2.5-4B-Q4`: 1 receipt(s), no valid pp+tg pair (last 2026-10-21) — post-minbase-fix pinned receipts (full-text gate, fused defaults ON)

## How this is fed and gated

- `scripts/parity_sweep.sh` measures each local model against `llama-bench`
  and appends receipts to `contract/scoreboard/parity.jsonl` (and the
  local-only `docs/bench/parity.jsonl`).
- `python3 scripts/scoreboard.py` regenerates this page, `contract/scoreboard.json`
  and the conformance page from the tracked sources.
- `python3 scripts/scoreboard.py --check --live-report <report>` is the CI
  gate: it fails on drift, on a conformance FAIL/posture regression, and on
  any metric more than 10% below the frozen baseline.
  Accepting a new floor is deliberate: `--freeze-baseline`.

Measured alongside: 3 conformance snapshots (linux-arm64, linux-x86_64, macos-arm64) — see `contract/CONFORMANCE.md`.
