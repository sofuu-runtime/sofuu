# Sofuu Custom Inference Engine — Detailed Plan (`engine:"sofuu"`)

> The proprietary endgame of Track C: a tiny, self-contained transformer that runs
> **QTSQ-quantized weights** directly and produces **QTSQ KV** natively — one format end to end.
> llama.cpp is the **default backend, the correctness oracle, and the long-term fallback**.
>
> This is the deep plan. The high-level roadmap lives in `ROADMAP.md` (Track C, milestones M5–M6).
>
> *Updated 2026-08-06: the engine will be written **Rust-first** (Track D). The numeric kernels
> (matmul, quant, SIMD) may use `unsafe` + explicit SIMD intrinsics where needed, but the
> architecture, model config, loader, KV cache, and sampling are safe Rust. QTSQ stays C (FFI).*

---

## 0. Goal, Non-Goals, Success

**Goal.** Generate correct, fast text from a decoder-only LLM entirely in our own C code, with
weights and KV both in QTSQ, runnable on phone / SBC / Worker / server unchanged.

**Non-goals (at least initially).** Training/fine-tuning. GPU kernels (CPU-first; GPU later).
Every architecture on day one (one arch first). Beating llama.cpp on speed at M5 (correctness first).

**Definition of done (M5 → M6).**
- **M5 (parity):** one architecture, greedy decoding matches llama.cpp top-1 token for token on a
  fixed prompt; logits cosine ≥ 0.999 vs an f32 reference; perplexity within tolerance on a small set.
- **M6 (default):** ≥3 architectures; within ~1.5–2× llama.cpp tokens/sec per HW tier; passes the full
  parity + soak + fuzz suite; `engine:"sofuu"` flipped to default, llama.cpp kept as fallback.

---

## 1. The Two-Layer Weight Model (the spine — get this right first)

`qtsq_decompress_tensor()` returns **full float32**. Materializing weights as f32 in RAM defeats
edge inference (a 4-bit 1.5B model → ~6 GB f32). So we separate two concerns:

| Layer | Format | Where | Purpose |
|-------|--------|-------|---------|
| **L1 — disk/download codec** | QTSQ container streams (lossless over our packed bytes) | `model.qtsq` file | tiny download, integrity (CRC/recovery), encryption, wormholes |
| **L2 — in-RAM numeric format** | our **packed quant blocks** (int4/int8 + per-block scale) | resident RAM | what the matmul kernels consume — *never* expanded to f32 |

**Load path:** `qtsq_read` container → for each tensor stream `qtsq_decompress` to the **packed quant
bytes** (NOT `qtsq_decompress_tensor` → f32) → keep packed blocks resident. We control quantization
at convert time, so we store our own block bytes as **raw streams** (`qtsq_compress_raw` /
`qtsq_container_add_stream`), and QTSQ only loss­lessly codecs+packs them. This keeps QTSQ as the
container/codec and lets the engine own the numeric format.

**Block format (mirrors the proven llama.cpp Q4_0/Q8_0 idea):**
```c
/* src/infer/engine/quant.h */
#define QK 32                     /* elements per block */
typedef struct { uint16_t d;      uint8_t qs[QK/2]; } block_q4;  /* fp16 scale + 32×4-bit */
typedef struct { uint16_t d;      int8_t  qs[QK];   } block_q8;  /* fp16 scale + 32×8-bit */
```
Map to QTSQ precisions: our q4 ↔ `QTSQ_TENSOR_F4`, q8 ↔ `QTSQ_TENSOR_F8`. (We may later add a QTSQ
accessor that yields blocks+scales directly to skip even the raw round-trip; not required for M5.)

**mmap option (later):** store L2 packed bytes *uncompressed* in the container so they can be
`mmap`'d and demand-paged on RAM-tight devices — trades disk size for lower RSS. Decision deferred to C3.

---

## 2. Target Architecture (pick ONE first)

**Llama/Qwen2-style decoder-only.** Per layer:
1. `h = x + Attn(RMSNorm(x))`
2. `y = h + MLP(RMSNorm(h))`

- **Attention:** Q/K/V projections with **GQA** (`n_head` query heads, `n_kv_head` ≤ `n_head`),
  **RoPE** on Q,K, causal scaled-dot-product with KV cache, output projection.
- **MLP:** **SwiGLU** = `down( silu(gate(x)) * up(x) )`.
- **Norm:** **RMSNorm** (no bias). **Final:** RMSNorm → `lm_head` (handle tied embeddings).

Config struct lifted from GGUF metadata:
```c
typedef struct {
  int n_vocab, n_ctx, n_embd, n_layer, n_head, n_kv_head, n_ff;
  float rope_theta, rms_eps;
  int   rope_dim;            /* usually n_embd/n_head */
  bool  tied_embeddings;
} sofuu_model_cfg;
```
First concrete target: **Qwen2.5-0.5B** or **TinyLlama-1.1B** (small, popular, GQA, SwiGLU, RoPE —
exercises every code path while staying debuggable).

---

## 3. File / Module Layout (new `crates/sofuu-core/src/infer/`)

> Rust-first (Track D): the engine lives inside the existing `crates/sofuu-core/` workspace,
> alongside the already-ported memory/SSE/MCP/TS/bundler/npm modules. QTSQ stays C via
> `sofuu-ffi`.

```
crates/sofuu-core/src/infer/
  infer.rs                # sofuu_infer_backend vtable (trait) + shared types (see ROADMAP §2)
  backend_llamacpp.rs     # Track B — wraps llama.cpp via FFI (oracle + default)
  backend_sofuu.rs        # custom engine: implements the vtable
  engine/
    model.rs              # model struct, cfg, weight handles, lifetime
    loader.rs             # QTSQ container -> resident quant blocks (L1->L2)
    quant.rs              # block_q4/q8, dequant, (de)quantize helpers
    forward.rs            # the layer loop / forward pass
    ops.rs                # rmsnorm, rope, softmax, swiglu, attention (scalar REFERENCE)
    ops_simd.rs           # NEON/AVX2 fast paths (unsafe intrinsics, guarded)
    kvcache.rs            # native QTSQ KV: export/inject, paging
    sample.rs             # greedy, temp, top-k, top-p, repetition penalty
  convert/
    gguf2qtsq.rs          # offline: GGUF -> Sofuu quant blocks -> QTSQ container
tests/infer/
  oracle_pytorch/         # white-box per-layer golden activations (bring-up)
  oracle_llamacpp/        # black-box end-to-end logit/token parity (real models)
  parity_test.rs          # the heartbeat test
```

Build: `SOFUU_LLM=1` gates inference; the custom engine has **zero external deps** (pure Rust +
our SIMD + QTSQ FFI), so it stays in the tiny binary even when llama.cpp is compiled out.

---

## 4. Phased Build (expands ROADMAP Track C0–C4)

### C0 — Foundations & the oracle harness  *(correctness scaffolding before any kernel)*
1. **Vtable wiring:** `backend_sofuu.c` implements `sofuu_infer_backend` (stubs that return errors)
   so the JS path `provider:"local", engine:"sofuu"` reaches our code.
2. **PyTorch/NumPy reference (white-box oracle):** implement the chosen arch in ~150 lines of
   PyTorch; for a **fixed tiny random model + fixed input**, dump per-tensor golden activations
   (embeddings, each RMSNorm, Q/K/V, post-RoPE, attn out, MLP out, final logits) to `.bin`.
3. **Scalar tensor core (`ops_scalar.c`):** rmsnorm, rope, softmax, silu, swiglu, naive attention,
   naive f32 matmul. **No SIMD, no quant yet.** Wire `forward.c` to run f32 weights.
4. **Parity test:** feed the same tiny model+input through `forward.c`; assert every stage matches
   the PyTorch golden within `1e-4`. **Exit criteria: full f32 forward pass bit-close to reference.**

> Rationale: a pure-f32, pure-scalar forward that matches PyTorch isolates *math* bugs from *quant*
> bugs from *SIMD* bugs. Every later layer is validated against this same harness.

### C1 — Quantization + SIMD kernels  *(the memory/speed win; shares `src/simd/` C kernels via FFI, or `ops_simd.rs` intrinsics)*
1. **`quant.c`:** quantize f32 → block_q4/q8 and dequant back; unit-test round-trip error bounds.
2. **`loader.c`:** read QTSQ container streams → resident packed blocks (L1→L2 of §1). Verify a
   loaded-then-dequantized weight equals the convert-time quantization (no drift).
3. **Quantized matmul:**
   - **W4A32 first** (dequant block→f32 fused into the dot): simplest correct quantized path.
   - then **W8A8 / W4A8** (quantize activations per-row to int8; int8·int{4,8} → int32 accumulate →
     scale): the fast path. NEON `vdotq_s32`, AVX2 VNNI `_mm256_dpbusd_epi32` (scalar fallback).
4. **Parity:** swap f32 weights for quantized; compare logits to the C0 f32 reference — require
   **logit cosine ≥ 0.999** and **top-1 token match** on the fixed prompt.
   **Exit criteria: quantized forward matches f32 forward within tolerance.**

### C2 — End-to-end generation + native QTSQ KV
1. **KV cache (`kvcache.c`):** ring/paged f32 (then quantized) KV; GQA-aware; causal masking.
2. **Sampling (`sample.c`):** greedy → temperature → top-k → top-p → repetition penalty.
3. **Native QTSQ KV bridge:** implement the vtable's `kv_export`/`kv_inject` directly on our KV
   buffers (no llama.cpp state-API conversion) — this is where the custom engine *beats* the
   llamacpp backend: zero-copy into the QTSQ adapter, unified format.
4. **Black-box oracle (`oracle_llamacpp`):** run the **same real GGUF** through llama.cpp and through
   our engine (after `gguf2qtsq`); compare greedy token streams and per-step top-1.
   **Exit criteria (M5): greedy parity with llama.cpp on one real model.**

### C3 — Breadth, performance, edge
1. **`gguf2qtsq.c` converter** hardened: read GGUF tensors+metadata, requantize to our blocks, write
   QTSQ container. Add as a `sofuu convert model.gguf -o model.qtsq` subcommand.
2. **More architectures:** Phi, Gemma (handle norm variants, biases, partial-RoPE, logit softcap).
3. **Perf:** threading (libuv threadpool or pthreads), kernel tuning per ISA, KV quant, prompt-batch
   prefill, optional `mmap` weights (§1). Target within ~1.5–2× llama.cpp tok/s per HW tier.
4. **Edge budgets:** per-tier RAM caps, int4 default on phone/SBC; soak + power-loss recovery share
   Track A2.

### C4 — Promote to default
- Flip default `engine` to `"sofuu"` once M6 criteria hold; llama.cpp stays as `engine:"llamacpp"`
  broad-compat fallback. Document the switch; ship tiny-binary artifact.

---

## 5. The Oracle Strategy (why this de-risks "a transformer from scratch")

Two oracles, used at different granularities — this is the single biggest correctness lever:

- **White-box (PyTorch/NumPy), C0–C1:** per-layer activation parity on a *tiny fixed* model.
  Catches the exact op + the exact layer where math diverges. Tolerance `~1e-4` (f32), looser for quant.
- **Black-box (llama.cpp), C2+:** end-to-end on *real* GGUF models. Metrics: greedy top-1 token
  agreement %, logit cosine, KL divergence, and perplexity on a small corpus. Tolerance: 100% greedy
  agreement on short prompts; perplexity delta within a few %.

`tests/infer/parity_test.c` runs both and is wired into `make test` + CI. **No kernel merges without
its parity test.** This is the engine's heartbeat.

---

## 6. Risks (engine-specific)

| Risk | Mitigation |
|------|------------|
| Subtle numerical bug in one op | Two-oracle parity at per-layer granularity (C0); one arch before breadth. |
| Quant degrades quality | W4A32 baseline first; compare cosine/perplexity each step; expose precision knob; keep sensitive tensors (e.g. lm_head) higher precision. |
| QTSQ f32-only decompress kills RAM win | §1 two-layer model: store/keep **packed blocks**, never materialize f32. |
| SIMD divergence across ISAs | Scalar reference is ground truth; NEON/AVX2 must match scalar bit-for-bit (int paths) / within ulp (f32). |
| RoPE / GQA / tied-embeddings edge cases | Lift all params from GGUF metadata; assert against llama.cpp config dump. |
| Scope creep (many archs) | Arch behind a small "arch ops" table; add one at a time, each gated by parity. |
| Engine blocks shipping | Rides the shared vtable; llama.cpp backend ships M1–M4 regardless; promote only at M6. |

---

## 7. First Concrete Steps (the C0 spike)

1. Create `src/infer/infer.h` (vtable) + `backend_sofuu.c` stub reachable from `provider:"local",
   engine:"sofuu"`.
2. Write the PyTorch reference for Qwen2.5-0.5B-style arch; dump golden activations for a fixed
   tiny config + fixed token input.
3. Implement `ops_scalar.c` + `forward.c` (f32, scalar) and `parity_test.c`.
4. Make the f32 forward pass match the golden activations within `1e-4`. **That green test is M5's
   foundation** — everything after is swapping in quant (C1) and SIMD without breaking it.
