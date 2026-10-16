# PLAN-GPU-METAL — full GPU support for the Sofuu engine

**Status: EXECUTION IN PROGRESS (2026-10-16).** Done so far:
G0 ✓ GREEN & PUSHED (80c81254f) — `sofuu_metal_batch_open/gemv/batch_gemv_staged/batch_end`
(real command batching; 32 chained bf16 GEMVs in ONE commit = 10.77×, 431→40 µs/op,
bit-exact at every step; the earlier "visibility bug" was a test slot-map error, retracted).
G1 primitive ✓ (eb956c128) — `batch_mlx4` quantized GEMV op with bindings mirrored from the
per-op kernel (w/s/b aligned offsets + rems[4]); Rust `MetalEngine::batch_mlx4`.
Remaining: lfm2 decode client (activation slab + layer wiring under SOFUU_GPU_DECODE=1,
bar 33→20 ms/token on liquid), gemma wiring, G2 GGUF MSL kernels (q4_K/q6_K/q8_0/q4_0;
verify Q6_K_XL layout), G3 optional prefill, G4 per-model flip table + iOS check.
DRAFT awaiting approval for the remaining phases. Companion to PLAN-CPU-DECODE-PEAK
(CPU-REMAINING backlog C1–C5 frozen). Builds directly on the 2026-10-15 GPU
survey: the Metal engine (src/metal/metal_engine.mm + infer/metal.rs) has
correct per-op kernels (gemv mlx4/bf16/f16/f32 + fused qkv/gate_up/gemv_multi)
used today by the decode-GEMV path, but `sofuu_metal_gemv_async` is
fake-async — it syncs on entry, allocates a fresh command buffer per matvec,
and stages x/y through shared internal buffers. Measured consequence: the
per-matvec hybrid decode lost to CPU (dispatch-bound), which is why CPU decode
is the current default.

Think of the target shape as: **one GPU submission per decode step, all model
GEMVs inside it, CPU only for the serial glue** — then, where kernels allow,
the whole graph on GPU.

## Rules (same as the CPU plans)
Calms receipts, env-gated `SOFUU_GPU_DECODE=1` / phase flags, default-on only
after receipts and only per model where it beats CPU ≥1.3× at BOTH short and
battery context; output gates: f32/bf16 within bench RMS tolerance, 4-bit
kernels bit-identical to the CPU dot kernels; peer files read-only; commits
straight to main.

## Phase G0 — real command-buffer batching (the unlock; scoped from survey)
- C side (`src/metal/metal_engine.mm`): `sofuu_metal_batch_begin(ctx)` opens a
  retained encoder; enqueue variants take CALLER-provided x/y staging (drop
  the shared-internal buffers — the hazard that forces per-call sync);
  `sofuu_metal_batch_end(ctx)` = commit + wait. Keep the old per-op path as
  the fallback.
- Rust (`infer/metal.rs`): `MetalDecodeBatch` handle — begin, per-layer
  `enqueue(gemv_*)` into it, `end()`. A small pool of MetalBuffer y-staging
  (one per in-flight op, cycled) sized at model load.
- First client: lfm2 decode (30 matvecs/token) — the CPU baseline to beat is
  10.9→30.6 tok/s-era numbers per the earlier samples; bar: shorten the
  33 ms token to ≤20 ms on liquid, else the phase stops with the receipt.
- Acceptance: one commit per token (assert via a submission counter in
  debug); output parity vs CPU decode within f32 tolerance; clean build on
  the AAR (no-NDK JNI path) untouched.

## Phase G1 — whole-decode-step graphs, per arch (model-agnostic handle)
- lfm2 first (proven kernels + my lane), then block_graph (gemma), then the
  GGUF llama backend (model.rs — coordinate with the kernel owner; decode
  currently runs C++ q4_K/q8_0 GEMVs on CPU at 85–87% of the wall, so the GPU
  bar there is high: only win if KV+glue amortize better in-graph).
- Structure: CPU does embedding lookup, norm folding, RoPE/index math, top-1
  + sampler; GPU does every GEMV + the SwiGLU pair math where fused kernels
  exist. One begin/end per step; zero per-op commits.
- Acceptance per arch: token-parity test vs CPU decode (RMS-gated), calm A/B,
  and the two-context battery rows recorded.

## Phase G2 — the missing GPU kernels (backend-coverage matrix)
Current coverage: mlx4/bf16/f16/f32 gemv ✓; fused qkv/gate_up ✓. Missing for
full "all models on GPU":
- q4_K / q6_K / q8_0 / q4_0 MSL dequant-dot gemv kernels (GGUF fleet: Spark,
  Qwen3-8B, MiniCPM5-2B Q6_K_XL — note Q6_K_XL may be an extension layout;
  verify against the file's header before writing the kernel).
- KV cache resident on GPU for "whole-graph" steps (zero-copy via unified
  memory; page the KV slab per context length at load).
- PTQ1_0/142 ternary kernels (last — the 27B's CPU slab path is the fallback
  and its arithmetic is cheap; MSL has no sdot, so a LUT/dequant kernel is a
  research item with the CPU slab as the acceptance baseline).
Each kernel lands bit-identical vs its CPU counterpart where the CPU path is
bit-stable, tolerance-gated elsewhere, with a pinned test per kernel.

## Phase G3 — prefill on GPU (optional, only after decode receipts)
Bar to pass: beat the CPU AMX batched prefill (104.5 tok/s liquid-class,
273 tok/s 0.6B, 27B 2.46 tok/s w/ direct tile) by ≥1.3× on at least two
models with bit/tolerance gates. Likely winners: mlx4/bf16 mid-size gemms at
PRE_CHUNK tiles on the iGPU while CPU serves other requests; skip for f32
models where AMX is already near the bus.

## Phase G4 — per-model flip table + fleet battery (GPU vs CPU decode)
- Battery runs each model twice (SOFUU_GPU_DECODE=0/1) in one JSONL pass;
  the result becomes the shipping default table (per MODEL, not global).
- iOS: same engine; verify on device only after desktop receipts (sim Metal
  quirks documented in ios-sim-metal memory; no-copy SIGTRAP workaround
  already coded around).

## Risks / stop points
- Dispatch overhead wins again: if G0's batched step doesn't beat CPU by the
  bar on liquid, STOP and re-scope (the fallback is CPU forever on this Mac —
  receipts say which).
- 16 GB pressure: GPU staging pools + KV slabs share the same RAM; measure
  wired/compressed before enabling per model; the residency flag and GPU
  staging must not stack on 27B-class files.
- Peer concurrent wave on src/simd: metal_engine.mm coordination before
  touching shared headers.
- Order of battle remains: G0 → G1(lfm2) → receipt gate → rest.
