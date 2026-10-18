# PLAN-GPU-METAL — full GPU support for the Sofuu engine

**Status: EXECUTION IN PROGRESS (2026-10-16).** Done so far:
G0 ✓ GREEN & PUSHED (80c81254f) — `sofuu_metal_batch_open/gemv/batch_gemv_staged/batch_end`
(real command batching; 32 chained bf16 GEMVs in ONE commit = 10.77×, 431→40 µs/op,
bit-exact at every step; the earlier "visibility bug" was a test slot-map error, retracted).
G1 primitive ✓ (eb956c128) — `batch_mlx4` quantized GEMV op with bindings mirrored from the
per-op kernel (w/s/b aligned offsets + rems[4]); Rust `MetalEngine::batch_mlx4`.

**GPU1 wave (2026-10-16, this session) — whole-decode-step batches + G2 kernels. LANDED
(pending commit):**
- **G1-client ✓ lfm2 whole-step decode** (`infer/lfm2.rs` `Lfm2GpuDecode` +
  `forward_gpu`, `SOFUU_GPU_DECODE=1`, macOS): EVERY op of a decode token — RMSNorms,
  gated short conv (new `lfm2_conv_gate`), KV-write + GQA attention (new
  `lfm2_kv_write` + `lfm2_attn`, threadgroup scratch scores, head-RMSNorm+RoPE folded
  in), SwiGLU, all GEMVs — enqueued into ONE command buffer over page-aligned
  activation/KV slabs wrapped zero-copy (`register_zerocopy_only` refuses a copy
  fallback: CPU and GPU must share pages). CPU does the embedding row + sampler.
  CPU does the embedding row + sampler. Parity receipt
  (`gpu_decode_logits_match_cpu_rms`, SOFUU_TEST_LFM2_DIR=liquid-1.3B): 24 random-token
  steps, greedy top-1 identical through step 17; one near-tie flip at step 18 under
  accumulated f32-order residue; logits RMS 0.05-0.15 vs CPU (per-op-hybrid tolerance
  class), gate < 0.2 PASS.
- **Kernel fleet in `src/metal/kernels.metal`** (16 kernels total): slot-based
  `batch_rmsnorm`/`batch_swiglu` reuse (elementwise kernels hardened with `n`
  guards so grid rounding cannot spill into the next slab slot), `add_inplace`
  (residual), `lfm2_conv_gate`, `lfm2_kv_write`, `lfm2_attn`, plus **G2 GGUF
  dequant-dot GEMVs `gemv_q8_0` / `gemv_q4_0` / `gemv_q4_k` / `gemv_q6_k`** wired
  through `sofuu_metal_batch_gguf` (ty = GGML id: 2/8/12/14). G2 parity pinned:
  q8_0 & q4_0 & q6_K **bit-exact** vs the CPU vec_dot; q4_K 3.3e-7 (f32 order).
  q4_K/q6_K engine specifics are mirrored from the CPU kernels, NOT from upstream
  ggml: this engine's q4_K places LO nibbles of the 32 quants in values 0..31 and
  HI nibbles in 32..63 (q advances 32 bytes per 64 values), and q6_K reads its f16
  d at offset 208 with the interleaved scale order sc[8n + l/16 + {0,2,4,6}].
  Q6_K_XL verified against the MiniCPM5-2B file empirically (the file decodes
  coherently through the CPU block_q6_K path = same layout the kernel mirrors).
- **The measured wall (WHY the 1.3× decode bar stops here):** on this M2 Pro the
  batched whole-step GPU decode runs liquid at ~28 t/s vs CPU ~33-35 t/s. Attribution
  (temp knobs, since removed): +1.08 GB/token of FFN GEMV work = +22.2 ms/s/token →
  **~49 GB/s GEMV stream rate, linear in bytes**; per-op barriers ≈ free (+30 ops ≈ 0);
  lm_head (268 MB bf16 read) ≈ 2 ms; kernel shape changes (uint4 512B bursts, div-free
  group indexing, 8-rows-per-256-thread threadgroups, group-major order) **all land at
  the same ~45-50 GB/s**; driver-managed copied weights (vs wrapped mmap) identical.
  The CPU path decodes the same bytes at the same rate (4 threads, ~46 GB/s) — decode
  here is DRAM-stream-bound and both engines sit at the same wall. G0's win is banked
  (dispatch: per-op hybrid 7.7-10.9 t/s → one batch per token 28-29 t/s), but
  **≤20 ms/token on liquid is unreachable and the plan's stop rule (dispatch/wall
  fallback = CPU decode on this Mac) FIRES for lfm2**. SOFUU_GPU_DECODE stays an
  opt-in, correct (parity-verified) path; the G4 flip table below carries the
  per-model verdicts.
- **GPU2 wave (2026-10-16, same day) — GGUF llama decode client (`model.rs`)**: the
  seven per-layer GEMVs (q/k/v, attn_output, gate/up/down) run as THREE Metal
  commits per layer (`[q,k,v]`, `[attn_output]`, `[gate, up, GPU-swiglu, down]`)
  plus a fourth for the lm_head, over a page-aligned zero-copy slab. Norms,
  QK-norms, RoPE (both interleave modes), CPU KV cache and attention are shared
  verbatim with the CPU path (`batched_attention_ext`, biases, qk_norm all keep
  working); KIVI/dynamic-FFN/streaming fall back to CPU per token. Eligible
  dtypes q4_0/q8_0/q4_K/q6_K — arm fails LOUD otherwise; `--cpu` honored via
  `from_gguf_with_gpu`. Fleet receipts over the storm tail (load 13-39):
  GPU>CPU on ALL FOUR GGUF models in equal windows: 0.6B 14.1 vs 9.1,
  Llama-1B 20.7 vs 10.3, MiniCPM-2B q6_K 6.4 vs 3.6, Qwen3-8B 3.4 vs 1.2 t/s —
  calibration against the older calm-window CPU rows still pending a clean
  battery; the streaming-wall receipt for the WHOLE-STEP design does not apply
  here (this is the fewer-commits hybrid, CPU attention intact).
- **GPU2.1 wave (2026-10-18) — ONE commit per layer for the llama/GGUF shape**:
  three new kernels (`rmsnorm_f32w` f32-weight RMSNorm, `llama_kv_write`
  per-kv-head QK-norm + RoPE (split-half AND interleaved) + KV append,
  `llama_attn` per-q-head GQA decode attention over the full cache prefix)
  close the CPU glue inside the batch: Norms read f32 weights straight from
  the weight mmap (256-byte-aligned bind + element-remainder constant), the
  resume-flags (`has_norm`, `rope_mode`) are bind-time constants, and q/k/norm
  biases gate at arm (any bias / sliding window / head_dim>128 / f16 norms /
  misaligned offset → the 3-commit client, loudly labeled in the arm line).
  The KV cache is no longer duplicated: `k_cache`/`v_cache` are allocated
  BEFORE `arm()` and registered zero-copy, so GPU kernels and the CPU
  fallback read and write the SAME memory (576MB saved + fallback coherence
  by construction; 8B = 36 layers x 8 kv heads x 4096 x 128 f32 x 2).
  `layers_one_commit` runs 14 batch ops per layer behind ONE
  batch_open/batch_end (was 3 commits + host round trips); x and the KV
  caches stay command-buffer-resident across layers, x syncs back once per
  token.
  Receipts (2026-10-18, window load 6.3 — opencode peer on 74% CPU, TAINTED):
  Qwen3-8B-q4_K_M armed "ONE commit/layer" (36 layers, act 0MB, kv-shared
  576MB); temperature-0 decode output is IDENTICAL to the `--cpu` run on the
  same prompt ("The loader measures panels on all cores...") — first
  end-to-end GPU-vs-CPU text parity receipt for the llama shape including the
  new QK-norm/RoPE-fused attention kernels. Llama-3.2-1B-q8_0 through the same
  one-commit path: 27.7 t/s decode, 243 t/s prefill. 8B decode stayed ~2.1 t/s
  in the tainted window — same as the 3-commit client — which LOCALIZES the
  remaining 8B cost to the GEMV kernels' streaming bandwidth itself
  (one-row-per-simdgroup 4B scalar loads => ~10 GB/s effective on the ~5GB
  weight stream), NOT to commit waits or the new attention path: commit-wait
  elimination was necessary but not sufficient, and the panel-GEMV rewrite
  (2-8 rows per threadgroup, 16B loads, llama.cpp-class) is now the measured
  #1 lever for 8B.
Not landed, deferred with receipts: Spark/Qwen35/block_graph/gptx3 whole-step
clients (bespoke ports of the same pattern; block_graph already has a per-op
hybrid), ternary PTQ1_0/142 GPU kernels (CPU slab optimal per plan), and G3
prefill-on-GPU (GEMM-class, the one remaining plausible win). — same
  wall receipt applies to each (their CPU decodes already run 40-56 GB/s per the
  battery best-of table); G3 prefill-on-GPU (untouched): GEMM-class arithmetic
  intensity is the one place an iGPU can still beat AMX+dequant CPU prefill; needs
  its own tiled-kernel phase.

Remaining: lfm2 decode client ✓ (see GPU1), gemma wiring, G2 GGUF MSL kernels ✓ (see GPU1),
G3 optional prefill, G4 per-model flip table + iOS check.
Companion to PLAN-CPU-DECODE-PEAK
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
