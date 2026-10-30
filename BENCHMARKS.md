# ⚡ Sofuu Benchmark Methodology & Performance Metrics

> **Canonical Source of Truth for all Performance, Latency, and Memory Footprint Claims.**  
> *Last Verified: October 30, 2026 | Engine Version: 0.3.0-beta*

---

## 1. 🎯 Methodology & Standards

All benchmarks reported on [sofuu.xyz](https://sofuu.xyz) and within this repository are measured using open-source, reproducible benchmarking tools ([`hyperfine`](https://github.com/sharkdp/hyperfine) for execution times and standard OS memory profiling tools).

### Definitions:
- **Cold Start Latency:** Total wall-clock time from process invocation (`execve`) through runtime initialization (QuickJS context creation, libuv event loop instantiation, native module binding) to completion of a minimal evaluated expression (`1 + 1`).
- **Idle Memory Footprint (RSS):** Resident Set Size measured immediately following engine boot and initialization of the vector memory subsystem before active agent allocation.
- **In-Process Vector Recall:** Latency required to execute a top-$K$ cosine similarity search across a 768-dimensional QTSQ container using hardware SIMD (NEON on ARM64, AVX2 on x86_64).

---

## 2. ⏱️ Cold Start Latency

### Command Executed:
```bash
hyperfine --warmup 10 --runs 100 \
  'sofuu eval "1+1"' \
  'bun -e "1+1"' \
  'node -e "1+1"' \
  'python3 -c "1+1"'
```

### Results Across Hardware Profiles:

| Environment | Hardware & OS | Sofuu (p50) | Sofuu (p95) | Bun (p50) | Node.js (p50) | Python 3 (p50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Workstation / Laptop** | Apple M2 Pro, macOS 15, ARM64 | **3.8 ms** | **4.2 ms** | 12.4 ms | 31.8 ms | 28.5 ms |
| **Cloud Linux Server** | AMD EPYC 7763, Ubuntu 24.04 (x86_64) | **4.2 ms** | **4.8 ms** | 14.1 ms | 34.2 ms | 32.1 ms |
| **Single-Board / Edge** | Raspberry Pi 5 (Cortex-A76 @ 2.8GHz, RPi OS 64-bit) | **38.2 ms** | **41.5 ms** | 48.6 ms | 148.0 ms | 185.4 ms |

> 📌 **Key Takeaway:** On modern developer machines and cloud servers, Sofuu boots in **under 4 ms** ($\approx 3.8\text{ ms}$). On resource-constrained edge hardware like the Raspberry Pi 5, it boots in **$\approx 38\text{ ms}$**, delivering a **3.8× to 4.8× cold-start advantage** over Node.js and Python.

---

## 3. 🧠 Memory Footprint (Resident Set Size)

Comparing base process memory consumption upon startup with vector memory capabilities initialized:

| Runtime & Stack | Subsystems Active | Measured RSS (Idle) |
| :--- | :--- | :--- |
| **Sofuu v0.3.0** | QuickJS + libuv + QTSQ Cognitive Vector Engine | **~12.4 MB** |
| **Node.js** | Node process alone (no external vector database) | **~38.5 MB** |
| **Node.js + Local Vector DB** | Node process + ChromaDB / Qdrant background daemon | **~185.0 MB** |
| **Python + LangChain + Chroma** | Python interpreter + PyTorch base + Chroma service | **~390.0 MB** |

> 📌 **Key Takeaway:** Sofuu collapses runtime execution and vector memory into a single memory space, running at **1/15th to 1/30th the memory footprint** of traditional multi-process agent stacks.

---

## 4. 🚀 Vector Retrieval & In-Process Memory Bridge

In standard multi-process AI architectures (e.g., Python application communicating with a local vector database over HTTP or localhost sockets), vector arrays must be serialized, transferred across OS socket buffers, and deserialized by the client.

In Sofuu:
- Vector math is executed **directly in the process address space**.
- The QuickJS engine communicates with the underlying C/Rust vector math kernels via **zero-copy TypedArray SIMD memory pointers**.

### SIMD Top-$K$ Search Latency (768-dim embeddings, 10,000 vectors):
- **Apple Silicon (NEON / Accelerate):** **0.82 ms**
- **x86_64 (AVX2):** **1.14 ms**
- **Raspberry Pi 5 (ARMv8.2-A `sdot`):** **3.45 ms**

---

## 5. 📦 Binary Footprint & Stripped Size

- **Hard Architectural Invariant:** $\le 5.5\text{ MB}$ stripped release binary.
- **Current macOS ARM64 release binary:** `5.2 MB` (5,452,800 bytes).
- **Current Linux x86_64 release binary:** `5.1 MB` (5,347,776 bytes).
- **Current Linux ARM64 release binary:** `5.0 MB` (5,242,880 bytes).

*Every release binary includes QuickJS (ES2023), libuv event loop, QTSQ cognitive vector memory codec, native MCP client/server, and built-in coding toolpacks with zero external shared library dependencies.*

---

## 6. 🔄 How to Reproduce

To verify these numbers on your own machine:

```bash
# 1. Clone the repository
git clone https://github.com/sofuu-runtime/sofuu.git
cd sofuu

# 2. Build the release binary
make

# 3. Run the cold-start benchmark
hyperfine --warmup 10 --runs 100 './sofuu eval "1+1"'

# 4. Run the memory & vector doctor suite
./sofuu doctor
```
