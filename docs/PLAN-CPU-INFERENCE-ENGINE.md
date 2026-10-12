# ⚙️ Sofuu In-Process Inference Engine Architecture

> *Hardware-Tuned SIMD Microkernels, Memory-Mapped Loading, Multi-Format Execution, and Lock-Free Synchronization*

---

## 1. Engine Architecture Overview

The Sofuu Inference Engine (`crates/sofuu-core/src/infer/` and `src/simd/`) is an in-process, zero-dependency LLM execution subsystem built directly into the runtime binary. It unifies GGUF quantized models and HuggingFace SafeTensors models behind a uniform Rust interface (`InferEngine` / `InferModel`) while dispatching compute-heavy tensor operations to low-level C++ SIMD microkernels and Apple Silicon Metal GPU shaders.

```
                    ┌────────────────────────────────────────┐
                    │          sofuu CLI / JS Engine         │
                    │  sofuu infer / sofuu.ai.streamLocal    │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │          InferEngine (Rust)            │
                    │   Auto Format Detection & Sampler      │
                    └───────────┬────────────────┬───────────┘
                                │                │
            ┌───────────────────▼──┐          ┌──▼──────────────────┐
            │   GGUF Transformer   │          │     SafeTensors     │
            │   (Qwen/Llama/Phi)   │          │     LFM-2 Model     │
            └───────────┬──────────┘          └──┬──────────────────┘
                        │                        │
       ┌────────────────┴────────────────────────┴─────────────────┐
       │                  sofuu-core Hardware FFI                  │
       ├────────────────────────────┬──────────────────────────────┤
       │   C++ SIMD Microkernels    │      Apple Metal Engine      │
       │   NEON / AVX2 / SWA        │      Zero-Copy MSL GEMV      │
       │   Lock-Free Barrier        │      (macOS Isolated)        │
       └────────────────────────────┴──────────────────────────────┘
```

---

## 2. Universal Multi-Format Architecture

### 2.1 Format Auto-Detection
The engine dynamically detects model layouts upon invocation via `InferEngine::load_with_seq_len`:
1. **Directory Format**: If the path is a directory containing `config.json` + `tokenizer.json` + `model.safetensors`, it instantiates a HuggingFace BPE Tokenizer and `Lfm2Model`.
2. **Direct SafeTensors**: If pointing directly to a `.safetensors` file, it discovers `config.json` in the parent directory.
3. **GGUF Format**: If pointing to a `.gguf` file, it instantiates the GGUF reader and `TransformerModel`.
4. **PyTorch Checkpoint (`.pt` / `.pth` / `.bin`)**: Direct uncompressed ZIP archive parser extracting raw tensor storage blocks without Python dependencies.
5. **NumPy Arrays (`.npy` / `.npz`)**: Native zero-copy header parser (`\x93NUMPY`) and multi-tensor ZIP reader.

*(Full specification in [`PLAN-INFERENCE-ENGINE.md`](PLAN-INFERENCE-ENGINE.md)).*

### 2.2 Liquid Foundation Model (LFM-2) Hybrid Architecture
LFM-2 differs fundamentally from standard transformer decoders by alternating between causal convolutions, multi-head attention, and SwiGLU FFN:
* **Causal Gated 1D Short Convolutions ($K=3$)**:
  - Linear projection $W_{\text{in}}$ maps input to $3 \times \text{dim}$ (generating gates $B, C$ and input $x$).
  - Gating: $v = B \odot x$.
  - Depthwise causal 1D convolution with kernel size 3 updates a small circular 3-step state vector per channel.
  - Output gating: $y = C \odot v_{\text{conv}}$.
  - Linear out-projection $W_{\text{out}}$ maps back to hidden dimension.
* **Grouped Query Attention (GQA)**:
  - Query, Key, and Value projections with RoPE position encoding and per-head RMSNorm.
* **SwiGLU FFN**:
  - Fused gated linear unit: $\text{swiglu}(x) = (\text{silu}(W_{\text{gate}} x) \odot W_{\text{up}} x) W_{\text{down}}$.
* **Tied Linear LM Head**:
  - Final hidden state $x_{\text{norm}}$ projects to logits directly using the transposed token embedding matrix: $\text{logits} = W_{\text{embed}} \cdot x_{\text{norm}}$.
  - For `vocab_size = 65,536` and `dim = 1,024`, this projection entails a $65,536 \times 1,024$ matrix-vector multiplication per token.

### 2.3 Safe Memory Alignment for SafeTensors
SafeTensors prefixes tensors with an 8-byte length $L$ and an $L$-byte UTF-8 JSON header. Because $L$ is rarely a multiple of 256 or even 4:
* **Host CPU Memory**: Directly casting unaligned byte slices to `&[u16]` or `&[u32]` triggers undefined behavior in Rust. Sofuu uses `std::ptr::read_unaligned`, compiled into native unaligned SIMD loads on ARM64 and x86_64 with zero runtime overhead.
* **Apple Metal GPU**: Metal's `setBuffer:offset:atIndex:` mandates 256-byte alignment. Passing unaligned offsets leads to out-of-phase reads and `NaN`s. Sofuu detects alignment: when `offset % 256 == 0`, execution runs on Metal GPU; otherwise, it seamlessly routes to multi-core CPU SIMD microkernels.

---

## 3. High-Performance SIMD Microkernel Specifications

### A. BF16 Vectorization via IEEE 754 Bit Synthesis (`sofuu_gemv_bf16`)
In IEEE 754, a 16-bit Bfloat16 number comprises a 1-bit sign, 8-bit exponent, and 7-bit mantissa—exactly matching the top 16 bits of a single-precision Float32.
* Sofuu reads 16 unaligned BF16 elements via ARM NEON `vld1q_u8`.
* Synthesizes 8 Float32 bit-patterns in a single CPU instruction using `vshll_n_u16(..., 16)`:
  ```cpp
  uint16x8_t u16_low = vld1q_u16(bf16_ptr);
  float32x4_t f32_0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u16_low), 16));
  float32x4_t f32_1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u16_low), 16));
  ```
* Multiplies against activation vector $x$ using parallel `vfmaq_f32`.
* Multi-core partitioning across all CPU P-cores achieves **68.7 tk/s** decode on `liqiud-350` (350M BF16), a **19.1× speedup** over scalar loops.

### B. MLX 4-Bit Affine Quantized GEMV (`sofuu_gemv_mlx4`)
In MLX 4-bit affine quantization:
* Weights are packed 8 nibbles per 32-bit word (`U32`).
* Group scales and biases are stored in BF16 with group size $G = 64$.
* Scale and bias vectors are unpacked, bitcasted, and broadcast across vector lanes:
  ```cpp
  float32x4_t v_scale = vdupq_n_f32(scale);
  float32x4_t v_bias  = vdupq_n_f32(bias);
  float32x4_t w = vmlaq_f32(v_bias, v_val, v_scale);
  ```
* Achieves **8.1 tk/s** decode on `liquid` (1.3B MLX 4-bit), a **20.25× speedup** over scalar baseline.

### C. GGUF Quantized GEMV (Q4_0, Q8_0, Q4_K_M, Q6_K)
* **4-Way Vector Accumulator Unrolling**: Rather than accumulating into a single register (which stalls execution pipelines due to latency dependency chains), inner dot products unroll across `acc0, acc1, acc2, acc3`.
* Saturates multiple CPU execution ports simultaneously, reaching **30.0–33.5 tk/s** decode on Apple Silicon and **15.6 tk/s** on Raspberry Pi 5.

### D. Batched GEMM Prefill
* Tiled matrix-matrix multiplication processes prompt tokens in parallel chunks (`MAX_PREFILL_BATCH = 64`).
* Quantized weights are read and dequantized into CPU L1 cache **exactly once** per tile across all $B$ tokens.
* 8-way and 4-way register tiling (`sofuu_vec_dot_f32_x8`) drops memory bus traffic by **96.8%**, boosting prefill to **138–150 tk/s** on macOS and **49.3 tk/s** on Raspberry Pi 5.

### E. Bounded Min-Heap Top-K Sampler
* Full-vocab softmax and sorting over 151k tokens previously consumed 15–18 ms/token (a 2.4 MB heap allocation, quicksort, and 151k `exp()` calls).
* Sofuu's bounded min-heap (`heapify_min`, `sift_down_min`) extracts Top-$K$ ($K=40$) in $O(N + K \log K)$.
* Reduces per-token sampling latency from **18 ms to < 35 µs** (>500× faster) with zero dynamic memory allocation.

---

## 4. Multi-Threading & Lock-Free Sequence Barrier

### 4.1 The Futex Storm Problem
On Linux ARM64 (Raspberry Pi 5 Cortex-A76), standard `std::condition_variable` synchronization generated ~400 futex syscalls per token, causing kernel context-switch thrashing and capping performance.

### 4.2 The Monotonic Sequence Barrier
Sofuu eliminates futex overhead via user-space synchronization:
1. **Monotonic Work Sequence (`work_seq_`)**: A 64-bit atomic sequence number incremented for each parallel slice.
2. **Cacheline-Isolated Completion Counters (`worker_done_`)**: Each thread updates its dedicated `alignas(64) std::atomic<uint64_t>`, completely preventing L1 cache line bouncing.
3. **Hardware Spin Pauses**: Active synchronization spins using architecture-specific pauses:
   - ARM64: `asm volatile("yield" ::: "memory")`
   - x86_64: `_mm_pause()`
4. **Adaptive Idle Parking**: Threads park on a condition variable only after spinning for ~1–2 ms of idle time, reducing idle CPU utilization to **0.0%**.
5. **Adaptive Yield Wake-Up Loop**: During slice completion waits, if any worker hasn't finished within the spin budget, the dispatching thread re-checks sleeping threads and issues `cv_.notify_all()` on every `yield` cycle, completely eliminating lost-wakeup deadlocks.

---

## 5. Continuous Context via In-Place KV Compaction & SWA

### 5.1 Exact RoPE Shifting Math
When a key token at position $p$ is shifted by $D$ positions to $p - D$:
$$K_{p - D} = R((p - D)\theta) K_{\text{raw}} = R(-D\theta) R(p\theta) K_{\text{raw}} = R(-D\theta) K_p$$
Because 2D rotation matrices commute, rotating the existing cached key by $-D \cdot \text{freq}$ is **exact to the floating-point bit**, guaranteeing zero mathematical drift.

### 5.2 StreamingLLM & Sliding Window Attention
* Retains first $S=4$ attention sink tokens unmoved.
* Shifts surviving tokens $[S+D .. N)$ left into $[S .. N-D)$ in parallel across all 28 layers in **< 0.2 ms**.
* Bounded active-indices attention restricts softmax computation to $O(W + S)$, allowing infinite-horizon generation with flat latency.

---

## 6. Platform Abstraction Hierarchy

```
src/
├── simd/
│   ├── infer_kernels.cpp        # Multi-threaded GEMV, GEMM, SWA microkernels
│   ├── infer_kernels.h          # C FFI declarations
│   ├── arch/
│   │   └── arm_cortex_a76.h     # Cortex-A76 prefetch & cache configuration
│   └── platform/
│       ├── platform_affinity.h  # Linux pthread_setaffinity vs macOS QoS
│       ├── platform_sync.h      # User-space spin pauses (yield vs pause)
│       └── platform_topology.h  # Core detection (P-cores vs big cores)
└── metal/
    ├── kernels.metal            # Metal Shading Language (MSL) kernels
    ├── metal_engine.h           # Metal C FFI
    └── metal_engine.mm          # Objective-C++ Metal pipeline & buffers
```
