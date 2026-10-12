# PLAN-EDGE-INFERENCE-STANDARD — The Sofuu Edge & Mobile Standard

> **Mission:** Establish Sofuu as the unquestioned global standard for Edge, Phone, Mac, and Small-Scale Serverless AI inference — just as vLLM is the standard for high-scale datacenter clusters.
> **Scope:** Edge devices (Raspberry Pi, Orange Pi, Rockchip SBCs), Mobile (iOS & Android phones), Laptops (Apple Silicon Mac), Heterogeneous CPU+GPU Co-Processing, and Distributed Edge Swarms.
> **Constraint Gate:** Standalone release binary/library strictly $\le 5.0\text{ MB}$, zero background daemons, zero external dependencies, cold-start latency $< 30\text{ ms}$, 0.0% idle battery consumption.

---

## 1. Executive Strategy & Market Reality

### 1.1 The Core Positioning
Datacenter engines (vLLM, TensorRT-LLM, Hugging Face TGI) are designed exclusively for 80GB enterprise GPUs sitting in liquid-cooled server racks consuming 700 watts with infinite network bandwidth. They completely ignore the other 95% of compute on Earth:
- **Phones**: Hundreds of millions of iPhones and Android devices with powerful ARM NEON engines, fast unified RAM, and mobile GPUs.
- **SBCs (Single Board Computers)**: $50–$100 devices (Raspberry Pi 5, Orange Pi 5, RK3588) operating at 5–15 watts.
- **Macbooks**: Unified memory Apple Silicon M-series laptops capable of streaming weights at 100–400 GB/s.
- **Old PCs & Low-End GPUs**: Millions of desktop machines with 2GB–4GB VRAM cards (GTX 1050/1060, RX 580) or quad-core CPUs collecting dust.

**Sofuu will not be a generic cloud inference engine.** Sofuu is purpose-built to dominate **Edge, Phone, Mac, and Distributed Edge Swarms**.

### 1.2 The Edge Competitive Landscape

| Engine | Major Shortcoming at the Edge | Sofuu Edge Architecture |
|---|---|---|
| **llama.cpp** | Monolithic C++ codebase (hundreds of flags, 50+ backends), heavy memory footprint, no built-in HTTP server, no native agent runtime, clumsy embedding in mobile apps. | **< 5.0 MB single standalone binary/library**, sub-30ms zero-copy cold start, embedded QuickJS runtime, built-in HTTP server, native agent loop, MCP tool support. |
| **ExecuTorch / ONNX** | Heavy Python dependency chains, brittle compilation pipelines, slow cold starts, poor out-of-the-box support for low-end SBCs. | **Direct GGUF & `.qtsq` zero-copy execution**, pure native machine code, zero Python runtime, instant execution. |
| **Ollama / vLLM** | Heavy background server daemons (>500MB idle RAM overhead), Docker requirement, unusable on iOS/Android background tasks or 1GB RAM SBCs. | **Zero daemon overhead (8MB idle RSS)**, adaptive sleep to 0.0% CPU/battery when idle, runs directly on mobile background threads. |

---

## 2. Architectural Pillars

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SOFUU EDGE INFERENCE STANDARD                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Zero-Overhead Mobile Runtime   │ < 5MB binary, < 30ms cold load,    │
│                                   │ 0% idle battery, pure C/Rust.      │
├───────────────────────────────────┼────────────────────────────────────┤
│ 2. Microarchitecture Specialization│ Per-core tuned ARM NEON, Cortex    │
│                                   │ A76/A55, Apple P-cores, DotProd.   │
├───────────────────────────────────┼────────────────────────────────────┤
│ 3. Heterogeneous Edge Split       │ Apple Silicon Metal (unified),     │
│                                   │ Android Vulkan/Adreno + CPU split. │
├───────────────────────────────────┼────────────────────────────────────┤
│ 4. Distributed Edge Swarms        │ 4 KB activation pipelining over    │
│                                   │ LAN across cheap SBCs and GPUs.    │
├───────────────────────────────────┼────────────────────────────────────┤
│ 5. The Edge Micro-Hosting Stack   │ Embedded HTTP server, QuickJS loop,│
│                                   │ MCP tools, zero-byte embeddings.   │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Pillar 1: Zero-Overhead Mobile & Serverless Runtime
- **Sub-30ms Cold Start:** Operating system virtual memory paging (`mmap`) reads model headers and super-blocks on demand. Zero heap copying of weights.
- **True Micro-Serverless:** Engine wakes from 0% CPU on an incoming HTTP request, streams the answer in milliseconds, and returns immediately to 0.0% CPU/battery usage via adaptive worker parking.
- **Strict Binary Ceiling:** Entire runtime (LLM inference + JS engine + HTTP server + vector ops + local encrypted storage) fits strictly inside $\le 5.0\text{ MB}$.

### Pillar 2: Microarchitecture Specialization (Never Generic Code)
All hardware-specific optimizations are isolated into modular platform headers:
- `src/simd/platform/platform_sync.h`: Spin pauses and memory barriers per architecture (`asm volatile("yield")` for ARM, `_mm_pause()` for x86).
- `src/simd/platform/platform_topology.h`: Core topology detection (macOS P-cores vs Linux physical cores vs Android big cores).
- `src/simd/arch/`: Dedicated microkernels for ARM Cortex-A76 (`FEAT_DotProd`: `sdot`/`udot`), Apple Silicon (ARMv8.5-A), and Cortex-A55 (in-order execution pipelines).

### Pillar 3: Heterogeneous CPU + GPU Co-Processing (Division of Labor)
Instead of forcing all computations onto one chip, Sofuu splits execution according to physics:
1. **GPU (Heavy Dense Math):**
   - High-throughput matrix multiplications ($W_Q, W_K, W_V, W_O$, SwiGLU FFN up/gate/down).
2. **CPU (Logic, Memory Relief & Asynchronous Operations):**
   - **Logit Sampling:** Bounded min-heap Top-K (< 35 µs) runs on CPU, completely avoiding PCIe logit transfers (600 KB per token) and fragmented GPU micro-kernels.
   - **KV Cache Compaction & Sliding-Window Attention:** In-place RoPE position shifting ($R(-\Delta \cdot \text{freq})$) and KV compaction run in CPU RAM in < 0.2 ms, leaving **100% of GPU VRAM free for model weights**.
   - **Pipelined Staggering:** Layers 0..K execute on GPU while CPU prepares next-token activations and manages memory, eliminating idle stalls.

### Pillar 4: Distributed Edge Swarms (Clustering Cheap Hardware)
Pool clusters of cheap SBCs (e.g. 4x Raspberry Pi 5 = $200 total, 32GB RAM) or heterogeneous mixed devices (an old PC with a 2GB GPU + two Raspberry Pis) over local Wi-Fi or Gigabit Ethernet:
- **Activation-Only Pipelining:** Between transformer layers, only the hidden activation vector is transmitted (1024 floats = 4 KB for 0.6B–3B models; 20 KB for 14B models).
- **Near-Zero LAN Latency:** 4 KB over a 1 Gbps Ethernet switch transfers in **32 microseconds (0.032 ms)**—completely negligible compared to the 10–30 ms layer computation time.
- **Running 14B–32B Models on $50 Boards:** By spreading layers across nodes (Node 0: Layers 0..6; Node 1: Layers 7..13; Node 2: Layers 14..20; Node 3: Layers 21..27), small devices can cooperatively host models that normally demand a $2,500 workstation.

### Pillar 5: The Complete Edge Node (The Sofuu Advantage)
A standalone inference library only multiplies matrices. To build an app or server, developers still need Python, Node.js, and background gateways.
**Sofuu delivers the complete autonomous AI node in a single binary:**
- `sofuu.http.createServer()`: Host an OpenAI-compatible API directly from the device.
- `sofuu.ai.streamLocal()`: Native JavaScript API for local generation.
- `sofuu.agent.run()`: Autonomous agent loop with MCP tool execution.
- Offline embeddings (0 bytes, ~2 µs) and encrypted local brain persistence (`.qtsq`).

---

## 3. Milestone-Driven Roadmap (No Timelines)

This roadmap is strictly **milestone-driven and dependency-ordered**. Work progresses sequentially through verification gates, prioritizing uncompromising quality and speed over arbitrary calendar deadlines.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SOFUU EDGE ROADMAP MILESTONES                        │
├─────────┬──────────────────────────────────────────┬───────────────────┤
│ Stage   │ Milestone Focus                          │ Target Hardware   │
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 1 │ ARM64 Edge CPU Specialization (DotProd)  │ Pi 5, SBCs, Phones│
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 2 │ Apple Silicon Unified GPU Engine (Metal) │ MacBooks, iOS     │
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 3 │ Heterogeneous Split (CPU + GPU Co-Ops)   │ Low-VRAM GPUs, SBC│
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 4 │ Mobile GPU Compute (Android / Vulkan)    │ Snapdragon, Mali  │
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 5 │ Distributed Edge Swarm (Layer Pipelining)│ SBC Clusters, LAN │
├─────────┼──────────────────────────────────────────┼───────────────────┤
│ Stage 6 │ Edge Micro-Hosting & Monetization Node   │ Homelabs, Edge VPS│
└─────────┴──────────────────────────────────────────┴───────────────────┘
```

---

### Stage 1: ARM64 Edge CPU Specialization & Micro-Kernel Tuning
**Objective:** Squeeze maximum instruction throughput and memory bandwidth saturation out of quad-core and octa-core ARM64 edge processors without GPU assistance.

- [x] **1.1 Futex Storm Elimination:** Lock-free monotonic sequence barrier (`work_seq_`), cacheline-isolated completion counters (`worker_done_[i]`), hardware spin-pause loop, adaptive idle parking to 0.0% CPU/battery.
- [x] **1.2 Factorized Q4_K Scale/Min Arithmetic:** Algebraic elimination of per-element vector multiplication/subtraction in `sofuu_vec_dot_q4_k`, boosting Raspberry Pi 5 prefill to 57 tk/s.
- [ ] **1.3 Native ARMv8.2-A `FEAT_DotProd` Microkernel (`sdot`/`udot`):**
  - Implement integer dot product accumulation for Q4_K and Q8_0, processing 4 integer dot products per cycle per NEON execution unit.
  - Dedicated architecture file: `src/simd/arch/arm_cortex_a76.h`.
- [ ] **1.4 Dual-Block Hardware Memory Prefetch Tuning:**
  - Fine-tune software prefetching (`__builtin_prefetch`) for LPDDR4X / LPDDR5 edge memory bus latencies on Cortex-A76 and Cortex-A55.
- [ ] **1.5 Target Compiler Flags in Cross-Build:**
  - Support `-mcpu=cortex-a76` and architecture feature flags in `scripts/cross/zig-cc.sh` without breaking cross-compilation.

**Verification Gate:** Raspberry Pi 5 decode throughput reaches $\ge 20\text{ tk/s}$ on `Qwen-3-0.6B-Q4_K_M` (up from 14.9 tk/s).

---

### Stage 2: Apple Silicon Unified GPU Engine (Metal)
**Objective:** Accelerate MacBooks and iOS devices via Apple's unified memory architecture, achieving zero-copy weight sharing between CPU and GPU.

- [ ] **2.1 Minimalist Dependency-Free Metal Backend:**
  - Pure Objective-C / Metal framework bindings linked dynamically on macOS/iOS (zero external library bloat, preserving $\le 5.0\text{ MB}$ binary).
- [ ] **2.2 Unified Memory Zero-Copy Buffer Mapping:**
  - Share resident GGUF model memory directly between CPU virtual space and `MTLResourceStorageModeShared` Metal buffers without duplicating RAM.
- [ ] **2.3 Hot GEMV / GEMM Metal Shaders:**
  - Write high-performance compute shaders for Q4_K, Q6_K, Q8_0 matrix-vector projections.
- [ ] **2.4 macOS Performance Benchmark:**
  - Verify decode throughput on Apple Silicon M-series laptops.

**Verification Gate:** Decode throughput on M-series host exceeds $\ge 60\text{ tk/s}$ on 0.6B and $\ge 35\text{ tk/s}$ on 1.5B with zero dynamic memory duplication.

---

### Stage 3: Heterogeneous CPU + GPU Co-Processing
**Objective:** Intelligently divide transformer layers and operational phases between CPU and GPU to maximize throughput on hardware with constrained VRAM.

- [ ] **3.1 Dynamic Layer Splitting (`--gpu-layers <N>`):**
  - Allocate the first $K$ layers to GPU VRAM and remaining $N-K$ layers to CPU system RAM.
- [ ] **3.2 Asynchronous Pipelined Overlap:**
  - Stagger execution: while CPU executes final hidden layer normalization and token sampling, GPU asynchronously processes prompt projections for the next stream cycle.
- [ ] **3.3 CPU-Resident KV Cache with Zero-VRAM Footprint:**
  - Keep KV cache in system RAM; CPU threads execute in-place compaction and RoPE position shifts while GPU handles heavy weight multiplication.

**Verification Gate:** Run a 3B parameter model on a 2GB VRAM device with zero out-of-memory crashes and consistent sustained generation.

---

### Stage 4: Mobile GPU Acceleration (Vulkan for Android & SBCs)
**Objective:** Tap into mobile GPUs (Adreno on Snapdragon, Mali on Raspberry Pi 5 and Rockchip) on phones and edge boards.

- [ ] **4.1 Lightweight Vulkan Compute Pipeline:**
  - Minimal Vulkan 1.2 compute pipeline using pure SPIR-V shaders (no external Android NDK dependencies beyond standard `libvulkan.so`).
- [ ] **4.2 Adreno & Mali Optimized Workgroups:**
  - Tailor compute shader workgroup sizes to mobile GPU warp/wavefront widths (e.g. 64 for Adreno, 16 for Mali).
- [ ] **4.3 Thermal & Battery Throttling Protection:**
  - Dynamic workload throttling to prevent mobile device heating and battery depletion during prolonged inference.

**Verification Gate:** Android phone (Snapdragon 8 Gen 1/2) achieves $\ge 25\text{ tk/s}$ decode with stable thermals over 500 consecutive generated tokens.

---

### Stage 5: Distributed Edge Swarm (Layer Pipelining Across LAN)
**Objective:** Cluster multiple cheap SBCs, old PCs, and phones over local Wi-Fi or Gigabit Ethernet to pool RAM and run larger models.

- [ ] **5.1 Zero-Copy LAN Activation Protocol:**
  - High-performance, low-latency UDP/TCP socket stream for transmitting hidden activation vectors (4 KB–20 KB) between nodes.
- [ ] **5.2 Zero-Config Node Auto-Discovery:**
  - Local network mDNS / UDP broadcast handshake (`sofuu swarm join` / `sofuu swarm host`) automatically discovering sibling nodes on the subnet.
- [ ] **5.3 Adaptive Layer Partitioning:**
  - Benchmark memory and compute speed of discovered nodes; automatically slice model layers proportionally across the cluster.
- [ ] **5.4 Fault-Tolerant Node Reconnection:**
  - Gracefully handle node dropouts without crashing the master node.

**Verification Gate:** A cluster of 4x Raspberry Pi 5s (32GB combined RAM) successfully runs and streams a 14B Q4_K model over Gigabit Ethernet at $\ge 5\text{ tk/s}$.

---

### Stage 6: Edge Micro-Hosting & Monetization Node
**Objective:** Allow ordinary users with low-end or spare hardware to host AI endpoints and monetize idle compute.

- [ ] **6.1 Single-Command Edge API Server (`sofuu serve`):**
  - Standalone command exposing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) directly from the binary.
- [ ] **6.2 Built-in Rate Limiting, API Keys & Auth:**
  - Zero-dependency bearer token validation and IP rate limiting configured via `~/.sofuu/config.json`.
- [ ] **6.3 Native Cluster Gateway:**
  - The hosting node accepts incoming API requests and transparently drives the underlying Distributed Swarm cluster.
- [ ] **6.4 Local Metrics Dashboard:**
  - Lightweight console / HTTP metrics endpoint displaying real-time tokens/sec, active connections, RAM consumption, and uptime.

**Verification Gate:** A single `sofuu serve` node on Raspberry Pi 5 handles 5 concurrent client streaming connections with zero dropped requests.

---

## 4. Architectural Rules & Invariants

1. **Standalone Binary Ceiling:** The shipped executable and headless C/Rust library must remain strictly $\le 5.0\text{ MB}$.
2. **Zero External Daemons:** No background background server dependencies (no Python, no Docker, no Ollama).
3. **Platform Isolation:** Device-specific logic must be strictly quarantined in `src/simd/platform/` and `src/simd/arch/`. A change to Apple Silicon Metal code must never alter, touch, or regress Linux ARM64 or Android builds.
4. **Zero Thermal & Battery Surprises:** Edge and mobile code must feature adaptive parking and power-aware scheduling, dropping to 0.0% CPU when idle.
