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

## Closed: ANE forward-path offload (do not re-attempt without new hardware)

Measured twice, on real weights and real engine hidden states, and rejected both times. Recorded here
because the idea is attractive and will be proposed again.

**1. Whole transformer block on the ANE: rejected on numerics.** A real-weight fp16 MLProgram for a full
Qwen3-0.6B block dispatches perfectly (MLComputePlan 82/82 device-assigned ops on neuralEngine, zero CPU,
zero GPU) and is fast (1.05-1.28 ms/block vs 3.2-4.8 ms in-engine). But on real hidden states the block's own
contribution is 51% wrong at L0 and 95% wrong at L13 vs a validated f32 reference. Mechanism: Qwen3's per-head
q/k RMSNorm gains are large (max 96.5), pushing attention scores into the tens, where fp16 ulp is ~0.03 and
128 accumulating adds give ~0.1 error - the same size as the p10 top-2 score gap. Near-tied heads collapse
(per-head cosine down to 0.29) while peaked heads survive (0.99). The engine keeps prefill attention in f32
(proved: CPU and GPU prefill routes are bit-identical at every layer), so a whole-block ANE program cannot
reproduce it, and forcing fp32 ops pushes them off the ANE.

**2. GEMM-only split (norms+qkv on ANE, attention in the engine f32 kernel, o+FFN on ANE): numerically CLEAN
but does NOT pay.** It passed every gate: per-block rel error vs the f32 chain 9.96e-3 (L0) / 2.77e-4 (L13),
cos 0.9999508 / 0.9999998; a 28-layer chained emulation fed each layer into the next and produced 16/16
identical greedy tokens vs the f32 chain; the ANE's own logit contribution was 0.258 - six times SMALLER than
the engine's own Q8_K activation quantisation gap of 1.518. Implemented as a default-off flag with the seam at
the FFN boundary (so the fused attention epilogue is untouched and its loss is exactly zero), loud
load-time MLComputePlan refusal, programs loaded at model load, and no Core ML in the default binary
(otool/nm/strings all clean). Prefill-only, interleaved ABBA:

  prompt    comparison    med     min
  124 tk    k=0 vs k=7   0.887   1.052
  1454 tk   k=0 vs k=7   1.047   1.214
  1454 tk   k=0 vs k=14  0.987   0.994
  1454 tk   k=0 vs k=18  0.991   0.991

A wash at 124 tokens, a wash at 1454 with k=14/18, a loss at k=7/1454. Why: forcing one batch_end/batch_open
per offloaded layer costs 1.5-7% of a long prefill on its own (measured 4780 -> 4850 ms with the seam
inert), the seam adds two 0.5 MB copies per layer, and the ANE's 0.63 ms per 128-row FFN beats the GPU's
~1.2 ms by less than the seam consumes. The earlier synthetic 2.2x-4.5x projection came from comparing ANE
block time against the ENGINE's per-block time while ignoring the seam and the remaining f32 attention.

**3. Decode on the ANE: rejected on bandwidth.** B=1 is a streaming problem: GPU 8.4 ms vs ANE 14.5 ms per
0.44 G-param token; ANE streams 57-61 GB/s vs the GPU's 104. At B=32 all three engines tie at ~48-50 GB/s of
effective weight streaming, confirming the mechanism. int4 palettised reaches parity at best, and is ~5 dB
worse than Q4_K on real weights (17.9-18.3 dB vs 22.9 dB SNR), so it is not a drop-in either.

Conclusion: on this M2 Pro, with a correct fp16 implementation and the seam cost included, the ANE does not
beat the GPU for either prefill or decode of this workload. The machine's usable structure is GPU + CPU
together, one commit per token, which is what ships. Keep /tmp/ane-lab scripts (bench.swift, build_ffn.py,
chain28.py, RESULTS.md) for a future SoC where the seam or the bandwidth ratio differs.
