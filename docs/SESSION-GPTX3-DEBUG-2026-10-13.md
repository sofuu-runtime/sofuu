# GPT-X3 Backend Debug Session — 2026-10-13

## Scope

Focused on the Sofuu inference backend for `GPTX3ForCausalLM`, especially the hybrid
GDN2 + GQA architecture in `crates/sofuu-core/src/infer/gptx3.rs`.

Reference model:

```text
/Users/priyanshuboruah/Downloads/gpt-x3-150m
```

The real Hugging Face model was loaded through a temporary local custom-code view at
`/tmp/gptx3_hf`, using `configuration_gptx3.py` and `modeling_gptx3.py`.

Expected HF greedy output for prompt:

```text
The capital of France is
```

is:

```text
 Paris.
Paris is the capital of
```

Token sequence after the prompt:

```text
[5242, 523, 3604, 41677, 856, 779, 5706, 803]
```

---

## What Was Done

### 1. Restored reliable references

Used three references:

1. The existing NumPy reference `/tmp/gptx3_ref.py`.
2. The upstream HF implementation `/tmp/modeling_gptx3.py`.
3. The real HF model loaded with `trust_remote_code=True`.

The NumPy reference was useful for early layer comparison, but it incorrectly
normalized GQA queries across all heads. The real HF implementation normalizes each
query head independently, so later validation switched to HF as the source of truth.

### 2. Fixed prefill position bug

In `Gptx3Model::forward_prefill_all_with_logits`, the final prompt token was being
processed at the wrong position:

```rust
start_pos + tokens.len() - 1
```

after the loop had already advanced `start_pos`.

Fixed to:

```rust
start_pos
```

This affected all GQA layers and recurrent context handling.

### 3. Fixed GDN recurrence read-out indexing

The recurrent state is stored as:

```text
state[row, column]
```

The output must be:

```text
out[d] = sum_j q[j] * state[j, d]
```

The implementation was effectively computing a row-wise dot product instead of the
correct column-wise read-out.

This was fixed in `gdn_forward`.

After this fix, layer 0 recurrence state and raw GDN output matched the NumPy
reference exactly.

### 4. Fixed GDN output RMSNorm epsilon placement

The old code effectively used:

```text
(sum_sq + eps) / head_dim
```

The reference uses:

```text
sum_sq / head_dim + eps
```

This made the GDN output normalization scale too large.

Fixed to:

```rust
1.0 / (sum_sq / gdn_dim as f32 + 1e-5).sqrt()
```

### 5. Fixed transformer residual normalization

The original code normalized `self.x` in place before attention and MLP. That
destroyed the residual stream.

HF semantics require:

```rust
x_norm = rmsnorm(x, layer.pre_norm)
x = x + attention(x_norm)
x_norm = rmsnorm(x, layer.post_norm)
x = x + mlp(x_norm)
```

Changed both attention and MLP paths to normalize into `x_norm` while preserving
`x`.

After this, GDN layer 0 attention output matched the NumPy reference exactly.

### 6. Fixed GQA query/gate layout

`q_gate_proj` produces interleaved per-head data:

```text
[q_head_0, gate_head_0, q_head_1, gate_head_1, ...]
```

The Rust code initially treated it as:

```text
[all query heads, all gates]
```

Fixed the slicing to deinterleave query and gate per head.

After this, normalized and RoPE-applied GQA queries/keys matched HF.

### 7. Fixed GQA KV cache layout

The cache read path assumes:

```text
[layer, kv_head, position, head_dim]
```

The write path was using:

```text
[layer, position, kv_head, head_dim]
```

This made heads 3+ read zero vectors.

Changed KV cache writes to store each KV head at:

```rust
cache_base
  + kv_head * max_seq_len * head_dim
  + position * head_dim
```

After this:

- Token 0 matched HF through all 24 layers.
- Final layer 23 GQA attention output matched HF.
- Final layer 23 MLP output matched HF.
- Final norm input for token 0 matched HF.
- Computed final norm from Rust matched HF.

### 8. Updated QTSQ context-size mirror

The vendored `qtsq_context_t` grew by 4 bytes because of `stream_codec_id`.

Updated:

- `QTSQ_CONTEXT_SIZE` from `5024` to `5032`
- Added `OFF_STREAM_CODEC_ID = 5024`
- Updated the build-time static assertions
- Updated the layout test expectations

This was needed because a concurrent memory-module change had introduced the new
field while the Rust mirror and guard still expected the old size.

### 9. Fixed unrelated compile blockers

During the session, concurrent edits introduced several unrelated compile errors.
Minimal compatibility fixes were made so inference work could continue:

- `crates/sofuu-core/src/chat.rs`: corrected `ALL_COMMANDS` array length.
- `crates/sofuu-core/src/rt/memory.rs`: temporarily handled in-progress
  `BrainLayout` changes.
- `crates/sofuu-core/src/infer/model.rs`: fixed a borrow-scope issue.
- `crates/sofuu-core/src/rt/memory.rs`: QTSQ layout mirror update above.

These should be reviewed before push because the working tree contains unrelated
in-flight changes from other work.

---

## Current Correctness Status

### Token 0

Matches the real HF implementation through:

- GDN layers
- GQA layers
- MLP layers
- final norm
- logits

### Tokens 1+

Still diverge.

Layer-0 short-conv outputs for all five prompt tokens match the NumPy reference
within small floating-point error. Therefore the remaining bug is after the
depthwise convolution.

Layer-0 recurrent state diverges starting at token 1.

Current layer-0 state comparison:

```text
token 0: matches
token 1: diverges
token 2: diverges
token 3: diverges
token 4: diverges
```

This means the likely remaining issue is one of:

1. The normalized query/key/value recurrence inputs.
2. The decay/gate tensors.
3. The erase/update recurrence math across time.
4. Some state layout or aliasing issue after the first token.

---

## Temporary Debug Code Still Present

The following debug hooks remain in `gptx3.rs` and `infer/mod.rs`:

- `SOFUU_DEBUG_GPTX3`
- `SOFUU_DEBUG_TOPK`
- `SOFUU_DEBUG_TOKENS`

These must be removed or made production-safe before push.

---

## What Is Left

### 1. Fix token-1+ GDN recurrence

Compare layer 0 for token 1 against HF/reference:

- normalized `q`
- normalized `k`
- raw `v`
- `b`
- `w`
- decay `g`
- erase vector
- update vector
- previous and next recurrent state

The short convolution is already proven correct, so the issue is downstream.

### 2. Validate all prompt tokens through HF

After the recurrence fix, compare:

- every layer hidden state
- every token
- final norm
- logits
- greedy generation

Expected final result:

```text
 Paris.
Paris is the capital of
```

### 3. Remove temporary debug instrumentation

Remove or gate out all temporary debug prints.

### 4. Run correctness and regression checks

Minimum checks:

```bash
cargo test -p sofuu-core infer
cargo test -p sofuu-core
cargo build --release -p sofuu-core --bin sofuu
make size-check
```

Also run a broader smoke test across:

```text
/Users/priyanshuboruah/Downloads/gpt-x3-150m
/Users/priyanshuboruah/Downloads/liquid
/Users/priyanshuboruah/Downloads/liqiud-350
/Users/priyanshuboruah/Downloads/gemma-2b
/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf
/Users/priyanshuboruah/Downloads/gpt-x3-150m
/Users/priyanshuboruah/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf
/Users/priyanshuboruah/Downloads/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
```

### 5. Benchmark CPU inference

Benchmark after GPT-X3 correctness is fixed:

- Sofuu
- llama.cpp
- Ollama
- MLX-LM

Metrics to record:

- load time
- prompt prefill TPS
- decode TPS
- TTFT
- total latency
- output quality

### 6. Review and push

The repository requires OpenQodex review before push.

Before pushing:

1. Separate or carefully review unrelated dirty work.
2. Remove debug hooks.
3. Re-run tests and size check.
4. Run:

```bash
npx -y openqodex@0.12.0 review
```

5. Show the receipt.
6. Do not push on a blocked verdict unless the developer explicitly says to proceed.
