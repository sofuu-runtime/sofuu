# Ternary-Bonsai-2-27B (qwen35 / PTQ1_0) Support Session — 2026-10-13/14

## Scope

Add CPU-only inference support to the Sofuu engine for:

```text
/Users/priyanshuboruah/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

and benchmark it against the other local models (all comparisons CPU-forced:
`--cpu` on the Sofuu side, `OLLAMA_LLM_LIBRARY=cpu` for ollama, `-ngl 0` for
llama.cpp tools). No Metal/GPU code paths were touched.

## Model facts (established this session)

- `general.architecture = "qwen35"`, 27B, GGUF v3, 851 tensors, file 5.5 GB.
- 64 layers: 48 GDN/SSM layers (`(i+1) % 4 != 0`) + 16 gated full-attention
  layers (`full_attention_interval = 4`).
- Shapes: emb 5120, vocab 248320, heads 24 / kv 4 (rep 3), head dim 256,
  GDN inner 6144, conv width 4, state 128, qkv fused [5120, 10240] =
  q 2048 + k 2048 + v 6144, `attn_q` fuses q with per-head sigmoid gate
  (12288 cols = 256 q + 256 gate per head), FFN 17408 gated-silu,
  rope base 1e7 with sections [11, 11, 10, 0] (NEOX pairing on the first
  64 dims of q/k), eos 248046, add_bos off.
- **PTQ1_0 = GGML type 143**, invented by PrismML for this model:
  group of 128 values in a 28-byte block = `qs[24]` (120 values via
  5-trits-per-byte) + `qh[2]` (8 values, 4-trits-per-byte) + f16 scale `d`
  last. Trit decode: `q = (byte * 3^n) mod 256`, `trit = (q * 3) >> 8`,
  value = `(trit - 1) * d`; emit order = qs chunks [0..16) then [16..24)
  (c=32 never fits), n-major/m-minor, then qh n-major/h-minor.
- **prism.hadamard** metadata: 401 big tensors are folded `W̃ = W·D·H` with
  a 1024-block unsorted normalized Sylvester/Walsh–Hadamard H
  (`H[i][j] = (±1)^popcount(i&j) / √1024`); sign vectors have widths
  [5120, 6144, 17408] read from `prism.hadamard.sign_values[28672]`.
  Runtime input transform = signs, then per-1024-block H.
  `token_embd` is *inverse*-folded with REVERSED order (H first, then
  signs); `ssm_out` additionally permutes its 6144-wide input from tiled
  [hd 128, nk 16, rep 3] to [hd, rep, nk] before signs+H.
  `ssm_alpha`/`ssm_beta` are NOT hadamard-folded.
- GDN recurrence per v-head (48): `g = ssm_a · softplus(alpha + dt)`
  (softplus threshold 20), `S ← S·exp(g)` (scalar), `delta = β·(v − Sᵀk)`,
  `S += k⊗delta`, read after update; q/k L2-normed by
  `max(‖v‖₂, 1e-6)`; gated RMS-norm per v-head then `silu(z) ⊙ norm`.
- Attention: plain RMS q/k-norm [256], partial rope, causal softmax at
  1/√256, output gate `sigmoid`.

## What was done

### 1. PTQ1_0 quantization kernel (`crates/sofuu-core/src/infer/quant.rs`)

`GGML_TYPE_PTQ1_0 = 143`, `QK_PTQ1_0`, `ptq1_0_dequant_row(src, out)`.
Bit-exact against an independent Python reference (`/tmp/ptq_ref.py`):
sha256 of the first 128 dequantized f32 values of `token_embd`:

```text
9eb7c4d8cc8dea5c60698da2eca8d03d7492ba5b56bb3aca6bf6ffb34df1d206
```

### 2. Backend (`crates/sofuu-core/src/infer/qwen35.rs`)

From-scratch CPU backend (single-threaded decode path uses the validated
math; prefill uses the batched path below). Public API:
`from_gguf(gguf, max_seq_len)`, `reset_cache`, `compact_kv_cache`
(reset + 0 — GDN conv/SSM state cannot shift losslessly), `forward`,
`forward_prefill`, `forward_prefill_batch`, `forward_prefill_all_with_logits`,
`pub config`. Unit tests: FWHT/Sylvester matrix identity and a full
load-prefill-greedy test against the real 27B file (dev and release both
pass; release e2e 120.7 s).

Validation: greedy "The capital of France is" → `" Paris."`; serve-side
benchmark rated `QUALITY_OK` with a coherent multi-country continuation.

### 3. Dispatch wiring (`crates/sofuu-core/src/infer/mod.rs`)

`ModelBackend::Qwen35` arm wired through `reset_cache`, `compact_kv_cache`,
`forward`, `forward_prefill`, `forward_prefill_batch`,
`forward_prefill_all_with_logits`; forward_embed/layer_range/head stubbed
zero-ish like the other non-split backends; `from_gguf` routes
`arch == "qwen35"` to the new backend. Before this, the model failed to
load with `tensor not found: blk.0.attn_q.weight` (llama-style fallback).

### 4. Serve-side long-prefill fix (`rt/http_server.rs`, `main.rs`)

The http server's net-3 deadlines (30 s headers / 75 s idle) killed
connections while local inference prefill was still thinking — any prompt
whose TTFT exceeded 75 s got "Remote end closed connection without
response". Added a one-shot env escape hatch
(`SOFUU_HTTP_HEADERS_MS` / `SOFUU_HTTP_IDLE_MS`, milliseconds, `0` = no
deadline; defaults unchanged) and `cmd_serve` now opts into "no deadline".
Streaming completions of any length now survive; library defaults and the
slowloris posture are untouched elsewhere (http_server tests 7/7).

## Performance work (native thread sampling proof)

Initial state: a 5-token prompt + 8-token generation did not finish in
4 minutes. Sampling showed one thread grinding in `gemv_t143 → dot8` with
all other cores idle, plus per-token re-decode of the whole weight set.

- Row-parallel GEMV (`std::thread::scope`, contiguous deterministic row
  chunks, bit-identical per-row math, per-thread row scratch;
  `PAR_MAC_THRESHOLD` + `thread_budget()` cap 8).
- Batched prefill (`forward_batch_impl`/`forward_chunk`): each weight
  matmul dequantizes its PTQ tile once per token chunk and runs an f32
  GEMM over the chunk; conv/GDN state, KV writes, attention stay
  sequential in token order. Measured prefill: ~15 s/token → ~0.94
  s/token at 200 tokens (TTFT 15.7 s for 5 tok, 46.5 s for 60 tok,
  188.6 s for 200 tok on the fixed serve).
- Decode (~12.5 s/token, ~0.1 tok/s) remains the honest bottleneck: the
  single-token path re-reads and re-dequantizes ~5.5 GB of ternary weights
  per token, and on this 16 GB machine the mmap is page-cache bound.
  Next wins, in order: packed-domain ternary int8 dot (no f32 dequant per
  token), weight residency (madvise/pinned), prefill progress events.

## CPU benchmark (2026-10-14, 10-core Apple Silicon 6P+4E, 16 GB)

Full table in `/tmp/sofuu_bench/results.md` (also reproduced here):

| Model (all CPU, temp 0) | prefill | decode |
|---|---|---|
| **Ternary-Bonsai-2-27B-PTQ1_0 (Sofuu)** | ~1.0 tok/s (471-tok prompt, TTFT 487.7 s) | ~0.1 tok/s |
| Qwen3-8B-Q4_K_M | 12.5 tok/s | 7.8 tok/s |
| Spark-X2.5-4B-Q4_K_M | 19.7 tok/s | 13.3 tok/s |
| gemma-2b | 15.2 tok/s | 11.8 tok/s |
| Llama-3.2-1B-Q8_0 | 82.7 tok/s | 41.3 tok/s |
| Qwen-3-0.6B-Q4_K_M | 162.1 tok/s | 60.4 tok/s |
| gpt-x3-150m | 199.5 tok/s | 106.9 tok/s |

Load 0.1 s (mmap; no full-file dequant at load). Quality run:
`Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spa…`
— QUALITY_OK.

## Compatibility matrix (why every comparison engine is N/A)

- **llama-bench 0.26.0 / llama-cli:** cannot load PTQ1_0 — upstream ggml
  has no type 143 (`load_backend BLAS → fail`, confirmed locally).
- **ollama 0.20.6:** registers the `qwen35` architecture, but its bundled
  ggml has no PTQ1_0 decoder; 402 of 851 tensors are undecodable (static
  binary evidence; a live `ollama create/import` was skipped for disk
  space — <10 GB free). With `OLLAMA_LLM_LIBRARY=cpu` it would fail at
  model load.

Sofuu is currently the only local engine that can run this model.

## Files touched (this session's work)

- `crates/sofuu-core/src/infer/quant.rs` — PTQ1_0 dequant kernel
- `crates/sofuu-core/src/infer/qwen35.rs` — NEW backend
- `crates/sofuu-core/src/infer/mod.rs` — Qwen35 dispatch arms + arch routing
- `crates/sofuu-core/src/rt/http_server.rs` — deadline escape hatch
- `crates/sofuu-core/src/main.rs` — `cmd_serve` opts into it
- `src/simd/infer_kernels.cpp` — hoisted `template tile_scratch` out of the
  `extern "C"` block (7 C++ errors had been blocking every build for ~10 h;
  code otherwise preserved verbatim)
- `docs/SESSION-TERNARY-27B-PTQ1_0-2026-10-14.md` — this file

The tree also carried concurrent in-progress work by the developer during
this session (tile-kernel GEMM/GEMV in quant.rs + infer_kernels, Spark25
backend, android/ios binding edits, gptx3/model/rt-loop touches) — left
untouched and included as-is in the same commit.

## Test status

- `infer::qwen35::tests` — 2/2 pass in dev AND release (real 27B model).
- `rt::http_server` — 7/7 pass after the deadline change.
- Full `cargo test -p sofuu-core` — 522 pass / 1 fail; the failure is the
  pre-existing in-progress `test_tile_gemv_q8_0_matches_reference` tile
  kernel (not part of this work).
