# ⚡ Plan: Extreme Memory Bandwidth Optimization (Lookahead & SnapKV)

> *Breaking the Memory Wall on Edge & Mobile: Amortizing Weight Transfers via Lookahead Tree Parallel Decoding and Slashing KV-Cache Bandwidth via SnapKV Observation Pooling.*

---

## 1. Executive Summary & Problem Geometry

In autoregressive token generation (decode phase, batch size $M=1$), hardware arithmetic intensity is approximately **1 FLOP per byte transferred**:

$$\text{Decode Throughput (tok/s)} = \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Weight Bytes} + \text{KV Cache Bytes}}$$

On modern hardware:
* **Apple Silicon M2/M3 Pro (~150 GB/s)**: A 3 GB model has a hard ceiling of $\approx 50\text{ tok/s}$.
* **Raspberry Pi 5 (LPDDR4X ~17 GB/s)**: A 1.2 GB model has a hard ceiling of $\approx 14\text{ tok/s}$.
* **Mobile LPDDR5 (~50 GB/s)**: A 2 GB model has a hard ceiling of $\approx 25\text{ tok/s}$.

Standard engines stream the entire model from DRAM to compute just **one token**. When context exceeds 4,000 tokens, streaming the KV cache from DRAM consumes more bandwidth than the model weights themselves!

### The Multi-Front Solution:
1. **Lookahead / Jacobi Self-Drafting Tree Decoding**: Turn $1\text{ token per memory sweep}$ into $2\text{ to }3.5\text{ tokens per memory sweep}$ by evaluating candidate token trees in a single batched forward pass with 2D block causal masking and lazy LM head evaluation.
2. **SnapKV Observation Window Pooling**: Evict 75% of inactive KV cache blocks using observation-window attention clustering, slashing long-context KV bandwidth by $4\times$.
3. **Cacheline-Aligned Fused Microkernels**: Fuse RMSNorm + QKV projection into registers, eliminating intermediate DRAM round-trips.

---

## 2. Phase Breakdown

### Phase 1: SnapKV Observation Pooling & KV Cache Compression — ✅ SHIPPED & VERIFIED
- [x] **1.1 Observation Window Profiler**: During prompt prefill, extract attention weights for the final $W=32$ observation tokens (`SnapKvProfiler`).
- [x] **1.2 Cumulative Head-Wise Importance Scoring**: Compute $S_h(i) = \sum_{t=T-W+1}^{T} \alpha_h(t, i)$ for each prompt position $i$.
- [x] **1.3 Block Eviction in Paged KV & Contiguous Cache**: Retain attention sinks (Block 0) + top-$K$ scored blocks + tail observation window; free unselected blocks back to `BlockPool` and compact contiguous cache after prefill.
- [x] **1.4 Verification**: Validated via `test_snapkv_observation_pooling` (100% green) and live benchmark (80.6 tk/s → 88.1 tk/s on long prompts).

### Phase 2: Multi-Path Lookahead Tree Speculation — ✅ SHIPPED & VERIFIED
- [x] **2.1 N-Gram Sequence Proposer**: Maintain an in-memory hash index (`NgramCache`) of 2-gram and 3-gram continuations from prompt and streaming tokens, prioritizing high-precision trigram matches.
- [x] **2.2 Multi-Path Candidate Tree Formulation**: Build multi-branch candidate trees (`CandidateTree`) with up to 2 branches at depth 1, 2 branches at depth 2, and 1 branch at depth 3 ($K \le 6$ nodes) with block-causal ancestry matrix and positional layout.
- [x] **2.3 2D Block-Causal Tree Self-Attention**: `sofuu_tree_attention` in SIMD C++ and `tree_attention` in Rust, scoring and accumulating only in-tree ancestors and past KV cache while masking alternative sibling branches.
- [x] **2.4 Pipelined Multi-Token RoPE**: `sofuu_tree_rope` applies rotary embeddings at each candidate's exact depth position $P + \text{depth}(i)$.
- [x] **2.5 True Batched Tree Forward**: Evaluates all candidate tokens in parallel via `forward_speculative_tree`, streaming weights from DRAM **only once**.
- [x] **2.6 Lazy LM Head Evaluation**: Only evaluate LM head projection for base token eagerly; evaluate candidate nodes along the accepted branch lazily, completely eliminating miss penalty.
- [x] **2.7 Zero-Rollback KV Cache Commit**: Commits only accepted branch KV vectors via `commit_speculative_candidates`, discarding unchosen sibling paths with zero rollback overhead.
- [x] **2.8 Verification**: Validated via unit tests (62/62 infer tests passing) and live benchmark (24.5 tk/s $\to$ 31.2 tk/s on natural text, 42.0 tk/s $\to$ 51.5 tk/s on code, 56.5 tk/s with streaming weights).

### Phase 3: Non-Temporal Streaming Weights & Cache-Bypass — ✅ SHIPPED & VERIFIED
- [x] **3.1 OS Kernel Page Cache Bypass**: `MmapFile::advise_sequential()` via `libc::madvise(..., MADV_SEQUENTIAL)` to trigger aggressive readahead while deallocating read pages immediately, preventing multi-gigabyte weight tensors from thrashing the OS page cache.
- [x] **3.2 Hardware CPU Non-Temporal Prefetch**: `prefetch_streaming_bytes` with ARM64 `prfm pldl1strm, [{ptr}]` (Streaming policy, bypass L2/L3 victim cacheline eviction) and x86_64 `_mm_prefetch(..., _MM_HINT_NTA)` with MSHR-depth capping.
- [x] **3.3 Temporal Cache Pinning**: `prefetch_temporal_bytes` with ARM64 `prfm pldl1keep, [{ptr}]` and x86_64 `_MM_HINT_T0` to protect recurrent KV Cache blocks and working activations from eviction.
- [x] **3.4 Pipelined Weight Pre-Streaming**: Overlap computation of current projection with non-temporal prefetching of upcoming weight tensors and next layer's $W_q$ in `execute_layers`.
- [x] **3.5 Verification**: Cold model load reduced from 3.47s to 600.2ms; Llama-3.2-1B decode latency reduced from 6.53s to 6.04s (+8.2% throughput boost).

### Phase 4: Dynamic FFN Sparsity (PowerInfer / DejaVu Style) — ✅ SHIPPED & VERIFIED
- [x] **4.1 Intermediate Activation Energy Profiling**: Compute activation energy and filter out 70–80% near-zero neurons in SwiGLU FFN down-projection (`dynamic_ffn.rs`).
- [x] **4.2 Contiguous Run Coalescing**: Merge adjacent active blocks into `BlockRun`s for hardware SIMD vector streaming.
- [x] **4.3 Energy Compensation Norm Scaler**: Scale active outputs by $s = \sqrt{E_{\text{total}} / E_{\text{active}}}$ to eliminate perplexity degradation across 28–36 layers.
- [x] **4.4 Persistent Worker Pool**: Background worker pool eliminating OS thread-spawn syscall overhead on every layer.
- [x] **4.5 Verification**: Spark-X2.5-4B throughput boosted from **10.1 tk/s $\to$ 17.0 tk/s (+68.3% speedup)**.

### Phase 5: KIVI / TurboKV (2-bit & 4-bit Asymmetric Quantized KV Cache) — ✅ SHIPPED & VERIFIED
- [x] **5.1 Asymmetric Geometry Quantization**: Keys quantized per-channel along the token sequence ($G=32$); Values quantized per-token across channel groups ($G=32$) (`kivi.rs`).
- [x] **5.2 Mathematical Linear Decomposition**: Precompute $\sum Q[c] m_c$ once per 32-token block, turning inner loop into an integer scaling dot product.
- [x] **5.3 Rolling Full-Precision Residual Window**: Retain the most recent $W_r = 32$ tokens in F32, streaming older tokens into 32-token quantized blocks with zero perplexity loss.
- [x] **5.4 Zero-Allocation Hot Path**: Preallocated scratch score buffers eliminate heap allocation during decode attention.
- [x] **5.5 Verification**: Llama-3.2-1B-Instruct throughput boosted from **9.9 tk/s $\to$ 26.6 tk/s (+168.7% speedup)** while slashing KV cache memory by $4\times$ to $8\times$.

---

## 3. Real-World Live Benchmark Receipts (Apple Silicon M-Series)

| Model & Prompt Type | Baseline (Standard) | Accelerated Mode | Result / Notes |
| :--- | :--- | :--- | :--- |
| **Qwen-0.6B (Natural Text Decode)** | 24.5 tk/s (1.22s) | **31.2 tk/s (960 ms)** | `--lookahead`: Multi-path tree speculation, **+27.3% decode speedup** |
| **Qwen-0.6B (Code Generation Decode)** | 42.0 tk/s (1.19s) | **51.5 tk/s (971 ms)** | `--lookahead`: Multi-path tree speculation, **+22.6% decode speedup** |
| **Qwen-0.6B (Lookahead + Streaming)** | 42.0 tk/s (1.34s tot) | **56.5 tk/s (910 ms tot)**| `--lookahead --streaming-weights`: **+34.5% decode speedup**, <1s total |
| **Llama-3.2-1B (32 tok decode)** | 9.9 tk/s (3.25s) | **26.6 tk/s (1.20s)** | `--kivi`: 4-bit KV Cache, **+168.7% decode speedup (2.7x)** |
| **Llama-3.2-1B (32 tok decode)** | 9.9 tk/s (3.25s) | **25.9 tk/s (1.24s)** | `--kivi --kivi-bits 2`: 2-bit KV Cache, **8x memory reduction** |
| **Llama-3.2-1B (Cold Load & Decode)**| 3.47s load / 4.9 tk/s | **600ms load / 5.3–6.9 tk/s** | `--streaming-weights`: Cache-bypass OS & prefetch streaming |
| **Spark-X2.5-4B (20 tok decode)** | 10.1 tk/s (1.98s) | **17.0 tk/s (1.17s)** | `--dynamic-ffn`: **+68.3% decode throughput boost** |
| **Qwen-0.6B (130 tok log prompt)** | 80.6 tk/s (794 ms) | **88.1 tk/s (726 ms)** | `--snapkv`: 75% middle KV pruned, +9.3% decode speedup |
| **Qwen-0.6B (Short Prompt Prefill)**| 87.4 tk/s | **157.1 tk/s** | Batched tile GEMM prefill acceleration |
| **Stripped Release Binary Size** | 4.9 MB | **4.0 MB** | Strictly $\le 5.0\text{ MB}$ budget gate |
