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
1. **Lookahead / Jacobi Self-Drafting Tree Decoding**: Turn $1\text{ token per memory sweep}$ into $2\text{ to }3.5\text{ tokens per memory sweep}$ by evaluating candidate token trees in a single batched forward pass with 2D block causal masking.
2. **SnapKV Observation Window Pooling**: Evict 75% of inactive KV cache blocks using observation-window attention clustering, slashing long-context KV bandwidth by $4\times$.
3. **Cacheline-Aligned Fused Microkernels**: Fuse RMSNorm + QKV projection into registers, eliminating intermediate DRAM round-trips.

---

## 2. Phase Breakdown

### Phase 1: SnapKV Observation Pooling & KV Cache Compression
- [ ] **1.1 Observation Window Profiler**: During prompt prefill, extract attention weights for the final $W=32$ observation tokens.
- [ ] **1.2 Cumulative Head-Wise Importance Scoring**: Compute $S_h(i) = \sum_{t=T-W+1}^{T} \alpha_h(t, i)$ for each prompt position $i$.
- [ ] **1.3 Block Eviction in Paged KV**: Retain 4 attention sinks + top-$K$ scored blocks + local window $W_{\text{local}}$; free unselected blocks back to `BlockPool`.
- [ ] **1.4 Mathematical Invariance Test**: Verify perplexity on long-context sequences remains within $\Delta < 0.05$.

### Phase 2: Lookahead Jacobi Parallel Tree Decoding
- [ ] **2.1 N-Gram Sequence Proposer**: Maintain an in-memory trie / ring buffer of recent $N$-grams from prompt and output tokens.
- [ ] **2.2 Candidate Tree Generator**: Formulate a $K$-token candidate tree ($K=4\dots 6$) from high-confidence branches.
- [ ] **2.3 2D Block-Causal Masking**: Construct a 2D attention mask enabling the target model to evaluate all tree candidates simultaneously in a single forward pass.
- [ ] **2.4 Speculative Rejection & Tree Acceptor**: Verify tree logits against standard temperature/top-$p$ sampler rules in parallel; commit accepted tokens in a single step.
- [ ] **2.5 Benchmark**: Measure tokens/sec speedup on standard code, JSON, and chat benchmarks.

### Phase 3: Fused Pre-Norm + GEMV Microkernels
- [ ] **3.1 In-Register Normalization**: Keep RMSNorm normalized activations in CPU SIMD registers (NEON / AVX2) or Metal threadgroups.
- [ ] **3.2 Direct Weight Streaming**: Stream weights straight into dot product accumulators without writing intermediate normalized vectors back to RAM.
- [ ] **3.3 Non-Temporal Streaming Hints**: Apply `PLDL1STRM` cacheline hints so weight streaming does not evict active KV cache blocks from L2/L3 cache.

### Phase 4: Integration, CLI Flags & Verification
- [ ] **4.1 Engine Configuration**: Add `--lookahead` and `--snapkv` flags to `sofuu infer` and `sofuu serve`.
- [ ] **4.2 Regression Testing**: Ensure zero regression across GGUF, SafeTensors, PyTorch, and all architectures (Llama, Qwen, Gemma, LFM2).
- [ ] **4.3 Binary Size Gate**: Keep stripped release binary strictly $\le 5.0\text{ MB}$.
