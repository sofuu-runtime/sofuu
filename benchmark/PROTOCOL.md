# The "substantially better" benchmark protocol (PLAN §8)

Build the **same** application twice, from the same spec:

> Streaming chat + tool calls (2 tools) + persistent memory + resume after
> restart + local fallback when offline + HTTP endpoint (OpenAI-compatible).

- **Build A — Sofuu In-Process Substrate:** [`build-a/app.js`](build-a/app.js) —
  single file, public runtime API only, zero npm dependencies, one process,
  no daemons. The binary IS the app (`sofuu` < 5 MB).
- **Build B — The Fragmented Frankenstein Stack:** [`build-b/`](build-b/README.md)
  — Node + LangChain + Ollama daemon + Chroma daemon + Express; two supply
  chains (npm + pip), three background processes/ports.

## The table (record per release into `table.md`)

| Metric | Build A (Sofuu) | Build B (fragmented) | Measured by |
|---|---|---|---|
| lines of app code | `wc -l build-a/app.js` | `wc -l build-b/app.js` | repo |
| dependencies (direct/transitive) | 0 / 0 | 6 npm + 4 pip / hundreds | manifests |
| processes / daemons / ports | 1 / 0 / 1 | 3 / 2 / 3 | compose |
| cold start → first token | `bash scripts/bench_build_a.sh` | compose up + warm the daemons | script |
| peak RSS of the app process | same | same | script |
| works offline (features) | full (local model) | no (daemons need images/models fetched) | doc |
| model swap: cloud ↔ local | 1 env var (`SOFUU_BASE_URL`) | rewrite the provider wiring | diff |
| move laptop → phone | same binary family; the session doc ports (R4) | re-host the daemons | doc |
| resume after kill | `session.json` + events.ndjson (R4/R6) — asserted | rebuild Chroma state | script |

Reproduce with: `bash scripts/bench_build_a.sh` (Build A rows), and
`docker compose up` in `build-b/` for Build B. A stranger should be able to
re-run both from this directory alone.
## Decode-path microbenchmarks (GPU kernel A/B)

Kernel work on the decode path is measured on **device** time, not wall time:
this box runs other agents' benchmarks, so host timings swing 2-3x while a
kernel's GPUStartTime/GPUEndTime barely moves.

- `SOFUU_GPU_TIMING=1` prints `SOFUU_CMD_TIMING kernel=<name> gpu_us=<...>` per
  committed command buffer. With the decode client armed there is ONE commit per
  token, so this is the per-token GPU cost; `SOFUU_GPU_LAYER_COMMIT=1` splits it
  into one commit per layer for a per-layer breakdown (device times stay valid;
  wall time is inflated by 28 host round trips).
- `SOFUU_GEMV_BENCH=1` replays the real decode GEMV plan (28x gate 3072x1024 +
  28x attn_q 2048x1024 = 56 dispatches, 82.6 MB) inside one commit;
  `SOFUU_GEMV_BENCH=head` replays the lm_head Q6_K instead;
  `SOFUU_GEMV_BENCH_N=<n>` truncates the plan.
- `SOFUU_GEMV_BENCH_AB="4,6"` (with `SOFUU_GEMV_BENCH_AB_N=<rounds>`) is the
  comparison harness: same plan, one commit per arm, **adjacent commits,
  ABBA-ordered inside one process**, device time per commit. Contention lands in
  both arms of a pair instead of between two process runs, which is what makes
  +2% differences resolvable. Arms are `SOFUU_GEMV_B1` values:
  `6` = `gemv_q4_k_b1v2` (shipped decode kernel), `4` =
  `gemv_q4_k_b1blk` (the kernel it replaced, kept as the reference),
  `0` = the older per-row `gemv_q4_k_c2` path.
- `SOFUU_DUMP_LOGITS_DEC=<file>` writes the decode path's full logit vector for
  the first generated token (SFLOG01 container, raw LE f32) so two kernels can
  be compared bit for bit — `SOFUU_DUMP_LOGITS` only ever covered prefill, which
  never runs a b=1 GEMV.
