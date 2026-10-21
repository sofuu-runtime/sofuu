# PLAN-CPU-DECODE-ANDROID

**Status: DRAFT — awaiting developer approval. No phase starts before the
developer says go. No phase has a timeline; every phase has a measured
acceptance gate instead.**

Companion to `PLAN-CPU-DECODE-PEAK.md` (the C1–C5 backlog and measured
ground truth on the M2 Pro reference box). This document is the Android
edition: same walls, same recipes, measured on the vivo I2220
(SM8650 / Snapdragon 8 Gen 3, 1×X4 + 3×A715 + 4×A520, LPDDR5X) by the
developer through the app's own readouts.

## Goal

One number: **0.6B Q4_K_M decode ≥ 90 tok/s sustained**, measured by the
app's speed line over a ≥3-minute generation, stable (no thermal decay past
20%), on CPU only. Secondary, same levers proportional: Llama-1B Q8 and
gemma-2b-class models; TTFT for a 200–300-token prompt ≤ 400 ms warm.

Everything here is **architecture-general**: every change lives in the
common inference layers (kernel pool, batching scheduler, GEMV/quant
kernels, GGUF-arch forward interfaces). No change may be specific to one
model family. Non-goals at the bottom are the ones the literature rejected.

## Ground truth on-device (measured)

- 16.3 tok/s with the 2026-10-16 pinning build (regressed build), **40
  tok/s** after the audit revert (current build, decode 4 threads, no
  pinning). Earlier measurement ≤50 on the pre-metrics build.
- Prompt sizes on-device: a bare "hi" carries a 213-token input (system +
  tools + history) — prefill is a real TTFT term even for tiny prompts.
- Bandwidth math: 0.6B Q4_K_M ≈ 0.35 GB/token → 40 tok/s ≈ 16 GB/s
  effective; ~100 tok/s ≈ 40 GB/s. LPDDR5X practical single-chain GEMV
  streaming is ~30–40 GB/s → the target sits AT the wall, not below it.
  (Reference box context: the M2 plan measured the same model at 55–65 of a
  230 t/s ceiling — every thin-modelSoC is short of its ceiling and wins
  from exactly the same fixes.)

## The walls on this device (ranked by measured cost)

1. **Serial glue + dispatch quanta** (C1's argument, stronger here): a
   token is ~120+ separate GEMVs across ~120 ms; per-GEMV pool fan-out and
   stage serialization dominate thin models.
2. **Bandwidth saturation**: 4 big-core threads with uncoordinated streams
   ≈ 16 GB/s achieved of ~35+ available.
3. **Unverified scheduling variables**: core pinning (now opt-in),
   thread-count defaults — cheap to iterate but currently unmeasured
   on-device.
4. **Prefill**: batched path exists and scales with the performance
   cluster, but shares the pool with decode and has never been measured
   on-device separately.

## The parts (ordered; each has an acceptance gate and a rollback)

### Part 0 — Baseline + evidence (sofuu-android, mine; mostly shipped)

- Console→file capture, per-model auto-tune with per-candidate rows, tuning
  persistence. Complete except: a fixed measurement protocol row
  (same prompt, fresh chat, reported tok/s + TTFT) written into this file's
  receipts as the developer runs it.
- Gate: one baseline table row (0.6B + Llama-1B, decode auto, prefill auto).
- Rollback: none needed (read-only instrumentation).
- **Status 2026-10-16: instrumentation COMPLETE.** Console→file capture
  (`files/engine-console.log`), auto-tune per-candidate rows with TTFT, and
  per-model receipts (`files/bench-receipts.jsonl`: device, threads, n, ms,
  ttft, tps — one line per measurement + a `decode-winner` line). Added:
  the **phase metrics** instrument (`SOFUU_PHASE_METRICS`, Settings toggle
  "Phase metrics") — every 256 tokens the scheduler appends
  `fwd/sample/arm/glue per-token µs` + prefill aggregates to
  `files/phase-metrics.log`, splitting the sweep from the glue ON DEVICE.
- **Part 1 prerequisite satisfied by the same instrument:** the Part 1
  coordinated patch (C1 fused decode) starts only after one
  phase-metrics-on run names the glue fraction on this device.
- Gate fulfillment needs two developer runs: select the 0.6B, select
  Llama-1B (auto-tune + receipts fire per selection), one substantive
  generation each with Phase metrics ON. Pull via
  `adb shell run-as com.sofuu cat files/bench-receipts.jsonl` (and
  phase-metrics.log / engine-console.log).

### Part 1 — C1 on Android: fused tiny-batch decode (coordinated lane)

- Mechanism (from `PLAN-CPU-DECODE-PEAK.md` C1): one fused per-layer decode
  call with resident threads and internal barriers (norm → GEMV → rope →
  attention → out → FFN), no per-GEMV dispatcher return. Kills wall #1 and
  is the fast small-batch verifier Part 2 requires.
- Files: `model.rs` / `block_graph.rs` / `lfm2.rs` lanes per owner; C++
  kernels candidates — peer-coordinated patch, staged with measured
  receipts from this device.
- Acceptance: 0.6B ≥ 1.5× current tok/s short-context, bit-exact vs
  per-token decode on the existing fixture; llama-1B same-shape win to
  prove generality.
- Rollback: the fused path ships behind an engine opt-out so the per-token
  path stays callable.

### Part 2 — streaming bandwidth (coordinated lane; can run before/with 1)

- Mechanism: (a) channel-interleaved row→thread mapping in the decode GEMV
  (consecutive rows land on the same DDR channel today and serialize);
  (b) software prefetch depth in the dotprod loops; (c) Q4_K block-scale
  hoisting so payload streams dominate per-block reads. These are the
  classic llama.cpp-class ARM levers; bit-exact by construction for (a)/(b)
  if accumulation order is preserved — (c) must be float-order-checked.
- Files: `infer_kernels.cpp` + `arch/arm_cortex_a76.h` (coordinated).
- Acceptance: measured effective bandwidth ≥ 30 GB/s (tok/s ÷ 0.35 GB) on
  the 0.6B, bit-exact parity on the kernel fixture, ≥1.25× end-to-end
  tok/s.
- Rollback: kernel selection flags per change, default to current kernel.

### Part 3 — speculative decode with the fast verifier (depends on 1)

- What changed for Android: the 2026-10-16 cost model (spec verifier-bound
  at 0.13× on liquid) assumed the per-token verifier. Part 1's batched
  verify at performance-cluster thread counts is the unlock; on-device
  numbers replace the M2 receipts before any wide rollout.
- Mechanism: `NgramCache.propose()` (already shipped in
  `infer/lookahead.rs`) drafts k tokens from prompt + output history; the
  scheduler verifies the block with ONE batched forward
  (`forward_prefill_all_with_logits(tokens, pos)` on the prefill pool),
  accepts the longest argmax prefix, and commits. Rejected suffix KV is
  overwritten by the next round's batched forward — no rollback API
  needed.
- Files: `continuous.rs` (mine) + ONE addition in `model.rs`
  (coordinated): per-position argmax out of the batched forward, or a
  `verify_block` returning accepted length+tokens. `lookahead.rs` needs no
  changes.
- Gates: (a) correctness — bit-exact committed text vs plain decode on the
  fixture suite; (b) net ≥1.2× tok/s end-to-end on list/code/echo
  workloads at ≥40% acceptance; (c) per-model k caps (4 for 0.3–1B, 6 for
  ≤4B) with miss→plain fallback; (d) the readout gains "spec m/k" so
  acceptance is visible.
- Scope guard (generalization): first target = standard dense GGUF archs
  (llama/qwen3). Hybrid GDN2 archs (gptx3/lfm2/spark) are deferred until
  batched verify is proven on dense — recurrence state makes their verify
  semantics per-arch work.
- Rollback: env flag (`SOFUU_SPEC=0`), default ON only after gates.

### Part 4 — prefill lane (coordinated + mine)

- Current: batched prefill exists and rides the 8-thread performance
  cluster. Gap on-device: TTFT for the standard 213-token prompt was 4.0 s
  on a cold model and is unmeasured warm.
- Items: (a) C5 prefill slab/alloc reuse (coordinated); (b) keep the
  prefill pool at the full performance cluster while decode is 4 (already
  in the phase pool); (c) measure and record warm prefill ms/token on this
  device after Part 1 (the fused chains help prefill too).
- Acceptance: 213-token warm prompt ≤ 250 ms prefill; cold load documented
  separately (model-load time is a model property, not a regression).

### Part 5 — app-side (mine, continuous during all parts)

- The readout gains: spec acceptance m/k, thermal decay trace (tok/s over
  generation time). The per-model auto-tune keeps working unchanged — it
  tunes decode threads through the real scheduler, whatever the engine
  becomes under it.
- Thermal gate for the final number: ≥90 tok/s must hold over a 3-minute
  sustained generation; if decay >20% is measured, the decode thread
  default drops to the highest count that sustains it (auto-tune reruns).

## Coordination / file ownership (respecting concurrent lanes)

- Mine today: `sofuu-android/**`, `infer/continuous.rs`,
  `platform_affinity.h` (Android pinning).
- Coordinated with the kernels/infer owner (tiny, receipted, staged):
  `infer_kernels.cpp` (+`arm_cortex_a76.h`), `model.rs` (verify addition),
  and the C1 fused-decode lanes. Each lands behind a gate with a rollback.

## Rejected by research (for the record)

- **PowerInfer / PowerInfer-2** sparsity: 11.68 tok/s for a 47B on a phone
  is real, but it requires ReLU-fied/sparse-TRAINED models + per-model
  neuron profiling and predictors — dense Qwen/Llama get no benefit. Fails
  the generalization bar.
- **LayerSkip**: 2.16× but requires per-model training (layer dropout +
  early-exit loss). Fails the no-retraining bar.
- **Apple LLM-in-a-Flash**: reduces FLASH reads for models larger than
  RAM; ours are RAM-resident and DRAM-bandwidth-bound. Wrong layer.

## Definition of done

0.6B decode ≥ 90 tok/s×3-min stable on-device; Llama-1B and gemma-class
proportionally improved by the same parts; committed text bit-exact (or
recorded-drift where quantized KV lands); every permanent engine change
behind a gate; every claim traceable to a numbered run in this file or the
app receipts.
