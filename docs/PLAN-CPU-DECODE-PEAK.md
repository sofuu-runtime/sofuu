# PLAN-CPU-DECODE-PEAK

**Status: DRAFT — awaiting approval. Do not start any phase until the user says go.**
(2026-10-15. Companion to `PLAN-CPU-PREFILL-PEAK.md`, which is complete: prefill
ladder shipped and default-on. This plan targets the decode side of the same
budget using the already-proven recipes and new structural levers.)

---

## Ground truth (measured 2026-10-15, M2 Pro 6P+4E, calm unless noted)

Decode = one full weight sweep per token, strictly serial across layers.
Realistic streaming ceiling for a dependent GEMV chain ≈ 80 GB/s (llama.cpp's
own numbers agree; the 200 GB/s bus spec is not reachable for M=1 chains).

| Model | Bytes/token | Ceiling | Sofuu now (short ctx) | Battery ctx (~1300) | % of ceiling |
|---|---|---|---|---|---|
| Qwen-0.6B Q4 | ~0.35 GB | ~230 t/s | 55–65 | 55.4 | ~25% |
| Llama-1B Q8 | ~1.1 GB | ~72 t/s | 40 | 37.8 | ~55% |
| liquid-1.3B mlx4 | ~0.8 GB | ~97 t/s | 30 | 8.9 | ~31% |
| gemma-2b mlx4 | ~1.1 GB | ~75 t/s | 16 | 5.4 | ~21% |
| Spark-4B Q4 | ~2.4 GB | ~33 t/s | 28.6 | 15.6 | ~87% |
| Qwen3-8B Q4 | ~4.9 GB | ~16 t/s | 13.6 | 9.5 | ~85% |
| 27B ternary | 5.5 GB (2.75 GB slab) | ~29 t/s | ~1.0 (resident) | 0.61 | ~3–5% |

Reference (same box, llama.cpp CPU, calm): 0.6B 60.4 · 1B 41.3 · Spark 13.3 ·
8B 7.8 — we are at 0.9–2.2× llama.cpp everywhere on GGUF. The gap that remains
is to the bandwidth wall, not to llama.cpp.

## The four walls (what the samples showed)

1. **Serial glue fraction** — norm/RoPE/softmax/SwiGLU/sampler + GEMV
   fan-out/join quanta run single-threaded between the big calls. Dominates
   small models: 0.6B token ≈ 16 ms of which the weight sweep is ~4 ms.
2. **O(context) attention, mostly single-threaded** — KV re-read per token per
   layer; the 2–3× short→battery-T drop on every model.
3. **Dispatch quanta** — 40–140 GEMVs/token each split across 6 cores with a
   fixed join cost; hard floor for tiny models (~55–65 t/s).
4. **27B specifics** — per-token trit expansion (slab shipped, receipt
   pending), sequential GDN recurrence (48 independent heads), whole output
   head read every token.

Every escape below breaks exactly one term: tokens-per-sweep, bytes-per-sweep,
speed-of-sweep, or sidesteps the sweep.

---

## Phase D0 — receipts closure (housekeeping, ~1 calm window)

- [ ] 27B slab sustained-rate receipt: `SOFUU_T143_SLAB=1` serve, one
      `max_tokens=40` request after the ~6-min 1262-token prefill (the short
      counting prompt EOSes after 1 token). Record vs the 1.0 t/s packed-dot
      baseline. Calm box required (prepass thrashed at <100 MB free).
- [ ] Calm-window battery re-run and TASKS.md comparison row (battery #4 ran
      at load 8–10 mid-run; #3 was load-poisoned — the standing record needs
      one clean pass).

## Phase D1 — speculative decoding (tokens per sweep; walls 1+3)

- **Expected: 1.8–2.5× per-request decode on 8B / Spark / 27B; quality exactly
  unchanged (rejection sampling).** Small models draft about as fast as they
  decode, so pairs are big-draft←small-draft only.
- Design: draft model (Qwen-0.6B, liquid-350) proposes k∈{4,8} tokens greedily;
  target verifies all k in one weight sweep through its existing prefill path
  (AMX/batched — verification IS prefill shape, and prefill is our strongest
  lane by 1.7–2× vs llama.cpp). Accept the longest verified prefix by the
  standard reject/resample rule on the target's logits. Per-request adaptive k
  (acceptance-rate accumulator, no ML).
- Files: serve/eval loop + a draft/target coordinator in `infer/`
  (new `speculative.rs`), model-agnostic via existing backend traits. Peer
  files read-only.
- Flag: `SOFUU_SPEC=1` (+ `SOFUU_SPEC_K`, draft model path). Default-off until
  receipts.
- Acceptance: (a) greedy streams from target-only vs spec-decoded are
  token-identical on a 500-token chain test (quality gate, pinned); (b) ≥1.7×
  wall on 8B and Spark calm A/B; (c) acceptance-rate logged per run in the
  JSONL.
- Review stop: after (a) passes on one pair, before fleet roll-out.

### D1b — self-speculative, same model as its own draft (EXPERIMENTAL, user direction 2026-10-15)

**Status: experimental only — do not build until D1 (separate-draft) receipts
exist or the user okays it explicitly.** The idea: draft with part of the
target model itself instead of a second model. This is the family the field
converged on — layer-skip Draft & Verify (Elhoushi 2024), early-exit/Kangaroo,
Medusa/EAGLE extra heads, and training-free n-gram/Jacobi Lookahead decoding
(Fu 2024). Note the repo already carries two relevant pieces:
`infer/lookahead.rs` (n-gram propose + 2D tree-causal candidate evaluation,
from 49be50eff) and a lazy-LM-head speculative-decoding commit (5b453f772).

Why it is attractive on this box: zero extra memory, same tokenizer, shared
KV/mmap (page-cache friendly), and the verifier rides our strongest lane.
Why it stays experimental: a same-model draft is not free — a shallow pass
that keeps a fraction p of the layers still sweeps p of every byte, per draft
token. Break-even: accepted Σα over a k-token cycle must beat (k·p + 1)/k
sweeps; with p=0.5, k=4 that needs α ≈ 0.63 average, which half-model drafts
rarely hold across positions; realistic configs (p≤0.25, or multi-token
drafting in one pass via the Medusa/Lookahead shapes) are the only ones the
math favors. CPU bonus: verification arithmetic rides AMX, and ternary sdot
is arithmetically cheap — the 27B self-spec math is the most favorable of the
fleet (after the D0 slab receipt, not before).

Gate before anything gets built: the offline α-study — log per-position
acceptance of truncated-model drafts vs the full model on real prompts
(probe script + a layers-skip env on one target, no infra). If Σα per cycle
does not clear break-even with margin, this phase closes with the receipt
recorded.

**GATE RESULT (2026-10-15, 0c5e7e696): the plain truncated-layer variant is
CLOSED.** Mechanism shipped (`draft_layers` in lfm2.rs, per-token + batched,
shared mmap; ignore-tagged `spec_self_draft_acceptance_study`). Receipt on
liquid-1.3B (30 layers, k=4, 48 cycles): depth 22/30 α
0.60/0.29/0.02/0.00 (accepted 0.9/cycle); shallower depths worse to zero.
Sweep math: even natural-text acceptance nets ~1.7× slower than plain
decode — every draft token sweeps p of every byte. **Live branch within
D1b: zero-sweep proposals** — n-gram/Lookahead drafting (sketched in
`infer/lookahead.rs`) proposes at 0 sweeps, so verify = 1 sweep per up-to-k
candidates and tokens/sweep = accepted+1, quality unchanged; needs only
tree-attention verify wiring. Medusa/EAGLE heads need training, out of scope.

## Phase D2 — megakernel fused decode layer (walls 1+3)

- **Expected: the 20–30%-of-ceiling models (liquid, gemma, 0.6B) to 50–70% of
  ceiling** — liquid 30 → ~60 short-T; battery-context ~2×. Same class of win
  as prefill glue fusion (47.8 → 72.1).
- Design: one C++ (or Rust+NEON) per-layer call with resident threads and
  internal barriers — norm → qkv → RoPE → attention → output → FFN — no return
  to the dispatcher between stages. Reuses existing pieces: fused QKV (Spark
  has it), paired gate+up sweeps (LFM2 prefims), pool as persistent barrier
  group rather than per-GEMV dispatch. Attention prep/mix split stays
  token-sequential-safe (prefill lesson).
- Per-arch ports, one at a time, each bit-exactness-gated vs today's path
  (env-gated, default new path only after gates+receipts).
- Acceptance: per-arch batch-vs-fused equivalence test (bit-identical logits)
  + calm A/B ≥1.5× on liquid-1.3B before moving to the next arch.
- Review stop: liquid receipt before porting #2.

## Phase D3 — cheap decode kernels: repacked q4_K, KV quant, threaded heads (wall 2 + scale chain)

- **Expected: long-context decode +25–50% on GGUF fleet; matches the Pi-5 T1
  work.**
- Design: (a) repack q4_K at load time so decode GEMV reads pre-decoded
  sub-block scales (the peer's Pi-5 repack trick pointed at the GEMV lane);
  (b) KV cache quantization hooks (q4/q8 KV as opt-in, model.rs peer
  lane — coordinate); (c) split attention over independent heads across the
  pool (heads are data-independent; deterministic row split like the GEMV
  split).
- Acceptance: bit-exact vs per-token decode for (c) (same fold order);
  tolerance-gated for (a)/(b) quantized paths with recorded drift; calm
  battery-context A/B on 8B + Llama-1B.

## Phase D4 — MoE GGUF backend (sidestep)

- **Expected: 30B-total/3B-active-class models decode at 3B-dense rates
  (30–40 t/s) on this box** — the biggest absolute jump available.
- Design: GGUF MoE parser (expert tensor layout, router softmax + top-k),
  per-expert GEMV through the existing vectorized kernels (decode lane is
  already exactly the right shape — fewer matrices per token), page-cache-aware
  expert loading (16 GB box will not hold a full 30B file; touch-rate-aware
  mmap + optional expert cache).
- Note: the nemotron MoE GGUF was deleted 2026-10-14 as unsupported; re-acquire
  one test model when the parser exists.
- Flag: `SOFUU_MOE=1` until receipts.

## Phase D5 — 2-bit weights + LUT kernels (bytes term; plan P4 continued)

- **Expected: 8B ceiling 16 → ~30 t/s (IQ2-class GGUF); 27B T-MAC path to the
  20–40 t/s band.**
- Design: adopt/complement i-quants for decode-heavy GGUFs; T-MAC-style LUT
  bit-parallel kernels where dequant arithmetic (not bandwidth) binds —
  primarily the 27B ternary path (bit-identical-to-sdot requirement from P4
  stands).
- Large effort; only after D1–D3 receipts; keep on the researched notes in
  PLAN-CPU-PREFILL-PEAK.md (arXiv 2407.00088).

---

## Rules (same as the prefill plan)

- One phase at a time; D0 first, D1 next in line.
- Calm-machine receipts (`uptime` recorded) — battery rows only from calm runs.
- Env-gate every behavior change; default-on only after receipts.
- Bit-exactness gates pinned per path where the math allows; tolerance gates
  for genuinely quantized paths, with drift recorded.
- Peer files read-only; narrow staging; commits straight to main.
- Deliberate review stop after each phase's first receipt.

## What D1–D5 do NOT do

- Single-request dense decode past ~2–3× its bandwidth ceiling is physically
  closed; spec-decode works by changing the ratio of tokens to sweeps, not by
  breaking bandwidth. The honest per-request ceiling sequence on this box is:
  D2 kernel efficiency → D1 tokens-per-sweep → D3/D5 bytes-per-sweep.
- GPU offload (fused whole-graph Metal submission) is deliberately out of
  scope: it reopens the CPU-only constraint. Available if that constraint is
  ever relaxed as a product decision.

## Phase DG — GPU (Metal) decode, ordered by measured blockers (2026-10-15 survey)

Status: survey done, first increment scoped. The Metal engine has per-op
kernels (gemv mlx4/bf16/f16/f32, fused qkv/gate_up variants) but the async seam
is fake: `sofuu_metal_gemv_async` syncs on entry, makes a fresh command buffer
per matvec, and shares internal x/y staging buffers — a data hazard that
*requires* the per-call sync (and exactly the 30-submissions-per-token overhead
the CPU-default flip measured). Ordered work:

1. **DG1 — real command-buffer batching (the unlock).** C side
   (`src/metal/metal_engine.mm`): `batch_begin/open encoder`, per-op enqueue
   with CALLER-provided y staging (Rust: a small pool of MetalBuffers, one per
   layer op), `batch_end = commit + wait`. one submission per decode step (+
   k for spec verify). Rust (`infer/metal.rs`): `DecodeBatch` handle wrapping
   the open encoder; lfm2/block_graph/model backends enqueue their per-layer
   GEMVs into it. Flip under `SOFUU_GPU_DECODE=1`, A/B against the
   CPU-decode default; flips only where it wins (measured bar: >1.3× vs CPU
   short- and battery-context).
2. **DG2 — whole-graph encoding per token** (norm/attention glue on CPU,
   all GEMVs on GPU in one buffer), the shape that finally beats CPU on
   the 2B–8B band if DG1's numbers support it.
3. Acceptance: pinned outputs equivalence (GPU vs CPU within bench tolerance
   for f32/bf16; bit-identical for mlx4 dot kernels vs CPU gemv), calm A/B
   rows in the battery, explicit per-model GPU on/off table in TASKS.md.

---

# CPU-REMAINING — the complete "milk CPU to 100%" backlog (2026-10-15, frozen while GPU builds)

State: prefill ladder fully milked (default-on AMX/int8/tile, at/above llama.cpp
everywhere). Mid/large GGUF decode milked to the bandwidth wall (Spark 87%,
8B 85%). The decode items below are every measured, known-cause gap left, in
priority order. Each is env-gated + bit- or tolerance-gated per the plan rules.

## C1 — tiny-batch fused decode (was D2; the highest-value CPU item)
- Why: small models sit at 20–31% of their bandwidth ceiling (liquid 31%,
  gemma 21%). A token = ~4ms weight sweep + ~23ms glue/dispatch. Spinning the
  6-thread pool per-GEMV (~130+/token) with serialize-between stages.
- Fix: one fused per-layer decode call with resident threads + internal
  barriers (norm → qkv → rope → attention → out → FFN), no dispatcher return.
  Double-purpose: this SAME primitive is the fast small-batch verifier that
  unlocks speculative decoding fleet-wide (measured verifier-bound 0.13× →
  2–5× once a 5-token batch costs ~1 token).
- Evidence: spec A/B receipt (88cc8942f); gflop/streaming math in this file.
- Acceptance: liquid-1.3B ≥50 tok/s short-ctx (from 30), bit-exact vs
  per-token decode; then gemma/0.6B ports.
- Files: lfm2.rs/block_graph.rs/model.rs lanes per owner; C++ candidates.

## C2 — attention-head parallelism + KV quant (was D3)
- Why: battery-context decode is 2–3× worse than short everywhere (liquid
  30→13.5, gemma 16→6.5): single-threaded O(T) attention, unquantized KV.
- Fix: (a) split attention over independent heads across the pool (bit-exact,
  row-split precedent exists in every kernel); (b) KV q4/q8 quant hooks
  (tier-gated, drift recorded); (c) q4_K repack-time pre-scaled decode GEMV
  (port the Pi-5 T1 trick to the Mac decode lane).
- Acceptance: battery-context decode ≥1.6× on liquid + 8B; bit-exact (a),
  recorded-drift (b)/(c).

## C3 — 27B multipliers (three independent items)
- a) GDN recurrence parallelization: 48 heads are independent, processed
  serially — head-parallel split (my lane, qwen35.rs).
- b) LM head is ~a third of every token's sweep (65k vocab read per token):
  LUT/bit-packed head decode or head-slab residency — measure first.
- c) Slab sustained-rate receipt (D0): mechanism shipped (SOFUU_T143_SLAB),
  perf receipt pending a calm window (prepass thrashed under peer pressure;
  ~6-min prefill precedes any sustained probe).
- Expected: with slab + (a), 0.87 → 3–6 tok/s battery-context plausible
  before any speculative layer.

## C4 — speculative decode with the fast verifier (unlocked by C1)
- Proven: 96% acceptance, zero-sweep proposals (88cc8942f). Add per-model k
  caps from sweep profiles (k=6–8 fat-sweep, 2–3 tight), miss→plain fallback.
- First target: 27B-slab (plain decode pathological, batched marginal small).

## C5 — hygiene items
- liquid-350 EOS-collapse ≥~50 tokens (correctness; gating its battery row).
- Prefill AMX w32-slab alloc churn (~104 ms/call): buffer reuse — a few % on
  prefill, more relevant if reused for C1 chains.
- Battery rows only from calm windows (load poisoning bit #3 and #4).

Ceiling check after C1–C4: liquid ~55–70 short-ctx / 25–35 battery; 27B
3–6 battery; spec-decode multiplies tokens-per-sweep on top. That is ~100%
of what a 6P+4E CPU without batching tricks can express for single-request
decode; C1's small-batch primitive is the linchpin — it also feeds GPU DG1.
