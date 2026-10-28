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

### Fusing projection dispatches (`SOFUU_GEMV_SECS`) — bit-exact, and a measured loss

196 of a decode token's 254 dispatches are the per-layer projections, so the
question is whether a stage can share one grid. Only same-x, disjoint-output
groups can, and only when they need the **same thread-group geometry**:

* this model's stage is `q(12) k(12) v(14)`, `o(12)`, `gate(12) up(12)`,
  `down(14)` (`ffn_down` and `attn_v` are the q6_K ones — check the GGUF, not the
  family reputation), and `lm_head` is q6_K;
* q4_K's bit-exact body (`gemv_q4_k_b1v2`) needs 256 threads/group with `rpt` rows
  per group; q6_K's (`gemv_q6_k`) needs 32 threads/group with **one** row. A
  dispatch has one thread-group size, so q(12)+k(12) can fuse and k(12)+v(14)
  cannot, and "one dispatch per layer" is not expressible at all without a
  grid-wide barrier;
* so the floor for this graph is **7 -> 5 per layer, 196 -> 140 per token**, and
  `=1` reaches exactly that.

The sectioned kernels keep each section's shipped body verbatim
(`gemv_q4_k_secs2`, `gemv_q6_k_secs`), and the dispatch counter confirms the
counts. Measured with `SOFUU_GPU_TRACE=1`, which prints
`[proj-dispatch] N projection dispatches this token`:

| arm | dispatches/token | dump md5 | 64-token md5 |
|---|---|---|---|
| frozen (`=0`) | **196** | `8f6739367691258cc4268f9ccf3b79e2` | `39703b5415de4eb91da72370c006be31` |
| `=qkv` (q+k fused) | **168** | identical | identical |
| `=gu` (gate+up fused) | **168** | identical | identical |
| `=1` (both) | **140** | identical | identical |

So the fusion is bit-exact by construction and in fact. It is also **slower** —
paired in-token device time, 3 runs per arm:

| arm | short prompt | 1427-token prompt |
|---|---|---|
| frozen (`=0`) | 6551 us | 10052 us |
| `=qkv` | 6855 us (**+4.6%**) | 10397 us (+3.4%) |
| `=gu` | 6960 us (**+6.2%**) | 10437 us (+3.8%) |
| `=1` | 6969 us (**+6.4%**) | 10533 us (+4.8%) |

Both fused stages carry the same penalty, and it is far larger than the ramp it
removes: gate+up is 99 MB of the token's stream (~1.3 ms of work at 74 GB/s), and
fusing it costs +409 us, i.e. **~30% on that work**, while deleting one dispatch
per layer. The sectioned kernel is therefore intrinsically slower than the shipped
per-projection kernel (most likely register pressure: it carries the section
prologue and dynamic `rows`/offsets into the same body), and the per-dispatch ramp
it was meant to recover is already hidden behind the GEMV stream. Conclusion: the
default stays the frozen path, `=qkv`/`=gu` remain opt-in for anyone re-testing
the idea, and **`=gu` should not be armed**. What would make fusion pay is a
sectioned kernel that matches `gemv_q4_k_b1v2`'s occupancy — the ramp is worth
less than 4% of a projection's own time, so the kernel has to be free.

### Why there is no `numeric_effect=yes` arm in the decode path

The first candidate was the Q6_K head: `gemv_q6_k` gives a row to a whole
32-lane simdgroup, but a 1024-column row is 4 blocks, so lanes 4-31 retire
immediately and 7/8 of every warp is idle on 19% of the token. Packing 8 rows per
warp (4 lanes each, one block per lane, butterfly sum) was built and measured:

| | frozen `gemv_q6_k` | 8 rows/warp |
|---|---|---|
| decode dump md5 | `8f6739367691258cc4268f9ccf3b79e2` | `46055397b582b654e98ba196c12eb2fb` |
| max abs dlogit | none | **7.63e-06** on `|logit|` 17.1, mean 1.11e-06 |
| logits differing | none | 143419 / 151936 |
| argmax | reference | same |
| 64-token text | `39703b5415de4eb91da72370c006be31` | identical |
| head plan, 1021 MB, device time | 9381 us = **108.8 GB/s** | 44043 us = **23.2 GB/s** |
| full plan, 262.9 MB | 3560 us = 73.8 GB/s | 5259 us = 50.0 GB/s |
| per-token device time | 6589 us | 12564 us (+90.7%) |

The arm is **0.213x**: interleaving 8 rows per warp turns each warp's contiguous
840-byte row stream into eight streams 840 bytes apart, so the lanes that stop
retiring are paid for with coalescing, which costs more. The frozen kernel sits at
108.8 GB/s against the ~117 GB/s zero-math ceiling for that mapping, i.e. the idle
lanes were never what limited it. The arm was therefore deleted rather than
shipped behind a flag: no decode-path kernel with a relaxed determinism tier
ships, and the only numeric knob this work added (`SOFUU_GEMV_SECS`) is backed by
bit-identical receipts, so its determinism tier is `exact`.

**The rule this leaves behind**, for whoever arms the next one: a decode arm may
change logits by ~1 ULP (about 1e-05 absolute at a logit magnitude of ~17) only
while (a) the argmax is unchanged on the decode dump, (b) the 64-token text md5 is
unchanged, and (c) the doc carries both arms' md5s. That tolerance is three orders
of magnitude below the engine's own activation quantisation (Q8_K, ~1.4e-02
relative), so it cannot move a sampling decision that quantisation does not
already move; what it does move is *reproducibility* — a bit-exact replay claim
requires the flag off, and the arm id has to be recorded with any receipt that
used it.

## Closed: DCOCO contextual correction over n-gram lookahead (2026-10-27)

DCOCO (github.com/two-delta/DCOCO, arXiv 2610.12570) fixes a parallel drafter that picks each slot without
seeing its predecessors: keep top-K per slot, score the (B-1)K^2 adjacent transitions in one batched pass,
then walk prefix-first. Our n-gram lookahead has the same defect (each level proposed independently), so it
was implemented in full (+558 lines, additive) and measured. **It loses. Default stays 0.**

| model | tree off | DCOCO K=4 | delta |
|---|---|---|---|
| SmolLM2-360M Q8_0 (GPU chain, default) | 2.952 tok/pass | 2.773 | **-6.1%** |
| SmolLM2-360M (CPU tree rows=8) | 3.158 | 2.952 | -6.5% |
| Qwen3-8B Q4_K_M | 1.125 | 1.059 | -5.9% |
| Qwen3-0.6B Q4_K_M | 1.000 | 1.000 | 0 (spec gated off after 8 zero-accept steps) |

Cost was never the problem: the pair pass is 5.27 us/draft, **0.016%** of a 32-34 ms draft cycle. The reason
is that the tree fails on **recall, not ranking** - at K=64 the model token is still absent from the lattice on
5/21 deep breaks (SmolLM2) and **16/16** (Qwen3-8B), and only 3/21 deep breaks were rerankable at all (+14% at
oracle precision is the whole headroom). An AR n-gram drafter already puts the approximate joint argmax in each
slot top-1, so pair scoring is near-degenerate. Output stayed bit-identical (greedy md5 + full-vocab prefill and
decode dump md5 across all 8 arms), so this is a pure throughput loss: no correctness risk, simply no win.

**Finding that outlived the experiment - speculation is currently slower than not speculating.** SmolLM2,
interleaved ABBA x5, n=10/arm: serial **130.70** tk/s, tree verify **83.55**, tree+DCOCO **77.90**. A 4-row
verify accepts 2.95 tokens but costs ~4.6x a serial token, where amortising the weight read should make it
~1.2x. The gate (`min_accept=1.15`) is acceptance-only, so it keeps a net loss enabled. Same root cause as
batched decode failing to amortise at b=2: a multi-row forward costs ~N x a single row instead of ~1x. Do not
add more drafter work (DCOCO, Medusa-style heads, wider lattices) until multi-row cost is fixed - recall and
verify cost both have to be solved, and recall is cheap to fix only after the cost is.

Also found here, pre-existing and unrelated: the CPU-tree **rows=5** route diverges from serial greedy text
(SmolLM2, Qwen3-8B) with DCOCO off, and `SOFUU_NO_TILES=1` does not unify it - unlike the TASKS.md:3741 note.
GPU chain (rows=3) and rows=8 do match serial. Full report: `docs/DCOCO-REPORT.md` (local-only, gitignored).

## Closed: quantized KV on the GPU decode path (2026-10-28)

kv4 (`--kv-dtype 8|4`) was built on the promise "less KV memory, and at long context maybe faster". Measured
on the shipped GPU decode path it is **slower than the f32 cache at every context tested**, by a gap that
grows with context, and it aborts the process at the window. **The default stays f32; the flag stays opt-in
and is not promoted.** Receipt: `kv-evidence/REPORT-kvlong.md`. Operator-facing summary:
`contract/KV4-OPERATOR-NOTE.md`.

Method: Qwen3-0.6B Q4_K_M, default 4096 window, `SOFUU_GPU_PREFILL=1 SOFUU_GEMM_MMA8=1`, serial decode
(`--no-lookahead`; the arms generate identical token streams, so speculation would add risk and nothing
else), prompts 493 / 1497 / 2993 / 3994 tokens from `/tmp/gen_prompt.txt`. Per point: 3 rounds of ABBA =
6 runs per arm, 72 runs total. Decode rate is the median of the 31 per-token `forward_ns` values from
`--trace`, not the CLI line, so the cold first step cannot bias an arm. Load (`sysctl -n vm.loadavg`) was
1.5-3.8 for the whole matrix except one transient 8.87 at the tail of the 2993 point that decayed over about
three minutes; it did not bias the 4k point (all three arms drifted up together, the inter-arm gap held at
+8.3 → +8.6 ms/token, fits still r2 >= 0.999). The instrument is deterministic: all 6 reps per arm and point
produced bit-identical token streams AND logprobs (max |dlp| = 0).

| prompt tok | f32 decode tk/s | 8-bit | 4-bit | 8-bit delta | 4-bit delta | f32 prefill tk/s | 8-bit | 4-bit |
|---|---|---|---|---|---|---|---|---|
| 493 | 111.6 | 100.1 | 98.3 | -10.3% | -12.0% | 1616 | 603 | 605 |
| 1497 | 86.3 | 69.6 | 69.8 | -19.3% | -19.1% | 1412 | 278 | 275 |
| 2993 | 66.2 | 46.9 | 47.5 | -29.2% | -28.3% | 1058 | 146 | 148 |
| 3994 | 56.3 | 38.8 | 38.6 | -31.0% | -31.4% | 874 | 111 | 111 |

Prefill deltas: -62.7% / -80.3% / -86.2% / -87.3% (8-bit), -62.6% / -80.5% / -86.0% / -87.3% (4-bit);
0.30 s → 35.9 s of prefill at ~4k tokens. That is the largest single effect in the experiment and the
opposite of the "quantized KV shrinks the prefill attention read" hope. 4-bit prefill == 8-bit prefill at
every point (603/605, 278/275, 146/148, 111/111 tk/s).

### Why it loses: 1.82x the f32 slope, not a fixed offset

Decode ms/token fits `a + b*ctx` with r2 >= 0.999 in all three arms:

| arm | a (fixed ms/token) | b (us per context token) | r2 |
|---|---|---|---|
| f32 | 7.76 | 2.49 | 0.99938 |
| 8-bit | 7.69 | 4.53 (**1.82x**) | 0.99984 |
| 4-bit | 7.78 | 4.49 (**1.80x**) | 0.99914 |

The fixed per-token cost is identical, so the arms differ *only* in attention. The quantized attention kernel
has 1.8x f32's slope while reading 3.2x (8-bit) / 5.3x (4-bit) fewer bytes, so it is **not bandwidth-bound**:
4-bit and 8-bit are indistinguishable in time (±2%) even though 4-bit reads 1.7x fewer bytes than 8-bit. The
dequant ALU/occupancy cost dominates the byte saving, which makes the gap a *slope* penalty — no context
length turns it into a win. As ctx → ∞ the penalty approaches 1.82x ms/token (~-45% tk/s); it is already -31%
at 4096. Fair-comparison note: with the default `SOFUU_ATTN_CHUNK_MIN=4096` the f32 arm uses
`llama_attn_w32` at every point below 4096 — the same kernel family and threadgroup geometry as
`llama_attn_w32_kv4` — so this is like-for-like, not a fused-vs-chunked artifact. Above 4096 f32 would also
get the chunked path that kv4 never gets, and kv4 aborts there anyway.

Engagement was proven three ways per point, not assumed: the `[kv4] quantized KV armed: ... 280MB vs 896MB
(3.2x), 2560 B/token/layer` receipt printed 6/6 runs of every quantized arm (4-bit: 168MB, 5.3x); logprobs
differed from f32 on 30-31 of 31 steps (max |dlp| 0.0005-0.022 for 8-bit, 0.015-1.55 for 4-bit); 8-bit
reproduced f32's greedy tokens 31/31 at every point (4-bit diverged 8/31 at 493 tokens). No arm was inert.

Contrast, Llama-3.2-1B Q8_0 (64 KiB KV/token, 3.5x smaller than Qwen; 2 rounds ABBA per point): 8-bit decode
-13.4% at 1497 tokens and -20.9% at 2993, prefill -53.3% / -65.9%, slope 2.07x f32's. The smaller-KV model
shows the smaller penalty (there is a byte-count term), but the sign never flips and it still grows with
context.

### What it does buy, and the claim that had to be retracted

The KV region is genuinely smaller — 896 MB f32 vs 280 MB (8-bit) vs 168 MB (4-bit) at ctx 4096 = 3.2x / 5.3x,
from the binary's own receipt. Peak **phys_footprint** falls with it: 1830 MB f32 vs 1701 MB 8-bit at a
493-token prompt, 2134 vs 1704 MB at 3994 (per-arm spread 1-3 MB). Peak **RSS** is not reportable at long
context: the *same arm* reads 1798 MB or 1283 MB depending on the rep, and 4-bit and 8-bit read identically.
So quote footprint deltas, never RSS deltas at long context, and do not claim the full ~570 MB the buffer
arithmetic predicts (footprint recovers only part of it, and the ~112 MB 4-bit should additionally save over
8-bit measures 0 MB).

### It aborts at the window, and the obvious escape is a trap

- Crossing 4095 with kv4 armed **SIGABRTs** (rc = -6) right at the boundary, after ~100 already-streamed
  tokens, same for 4-bit. The f32 arm on the identical input finishes all 200 tokens (rc = 0, 52.7 tk/s)
  because it can slide.
- A prompt longer than the window is refused cleanly by both arms (rc = 1, "prompt is 7988 tokens but the
  context window is only 4096 — ... raise --ctx or shorten the prompt"). Not kv4-specific.
- **`--ctx 8192` does not rescue it.** At that window the GPU decode client fails to build at all — even for
  f32 ("decode client NOT armed: building it failed ... this model decodes on the CPU") — and the kv4 arm
  additionally prints "the quantized cache could not be armed - restoring the full f32 KV cache (8192
  tokens)". kv4 above 4096 is unreachable on this 16 GB M2 Pro, so a user who follows the old "raise --ctx"
  advice silently lands on the CPU path instead.

### Decided

- Default stays f32. `--kv-dtype 8|4` stays opt-in; it is never promoted on throughput.
- The kv4 FATAL advice no longer offers "raise --ctx": it states that packed blocks are absolute-position
  indexed, that a sliding history needs `--kv-dtype f32`, and that 8192 does not arm the GPU client (one
  string in `infer/mod.rs`, `kv4_full_stop`).
- `--kivi` no longer silently disarms the GPU client: on a Metal build it refuses with the measured facts and
  requires `SOFUU_KIVI_CPU=1` (5.1 tk/s with `--kivi` vs 99.3 tk/s f32 on the GPU at ~890 tokens, 19x);
  CPU-only builds keep the flag because there the CPU layer loop is the only path.
- The quantized-KV *line* is not dead in principle — the loss is a property of this kernel's
  dequant-by-threadgroup structure (1.8x the ALU per attended token, insufficient parallelism at 16
  threadgroups), not of quantization. Re-attempt only with a kernel whose slope matches f32.
- Limits: one machine (M2 Pro, 10 cores, 16 GB); two small models (224 and 64 KiB KV/token) — no 8B-class
  big-KV model was timed, so "bandwidth never wins" is proven only up to 224 KiB/token; one synthetic prompt
  family; temp 0; serial decode.
