# ⚡ Sofuu Benchmark Methodology & Performance Metrics

> **Canonical Source of Truth for all Performance, Latency, and Memory Footprint Claims.**  
> *Last Verified: October 31, 2026 | Engine Version: 0.3.0-beta*

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

> **How to read this table.** Sofuu is a full-suite single-binary AI runtime (inference, encrypted
> vector memory, agents, MCP, HTTP, security — with a JS/TS engine as the orchestration surface);
> Bun, Node.js and Python are general-purpose language runtimes. They are **interpreter baselines
> for reference, not peer products** — Sofuu does not compete with them as a JS runtime. The
> like-for-like comparison is the AI **stack** Sofuu replaces: a Node.js process plus an agent
> framework, a vector database daemon, and an inference server (Python/LangChain/Chroma/Ollama),
> measured in §3 below.

| Environment | Hardware & OS | Sofuu (p50) | Sofuu (p95) | Bun (p50) | Node.js (p50) | Python 3 (p50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Workstation / Laptop** | Apple M2 Pro, macOS 15, ARM64 | **3.8 ms** | **4.2 ms** | 12.4 ms | 31.8 ms | 28.5 ms |
| **Cloud Linux Server** | AMD EPYC 7763, Ubuntu 24.04 (x86_64) | **4.2 ms** | **4.8 ms** | 14.1 ms | 34.2 ms | 32.1 ms |
| **Single-Board / Edge** | Raspberry Pi 5 (Cortex-A76 @ 2.8GHz, RPi OS 64-bit) | **38.2 ms** | **41.5 ms** | 48.6 ms | 148.0 ms | 185.4 ms |

> 📌 **Key Takeaway (revised 2026-10-31):** the 3.8 ms and 38.2 ms figures in this table are historical. Measured on an M2 Pro, 31 runs, wall clock including process spawn: **22.5 ms median** for `sofuu eval 1+1` on the current tree, against **61.5 ms** for the published 0.3.0-beta artifact measured the same way. The gap to Node.js and Python was not re-measured this session and is not claimed here — and those columns are interpreter baselines anyway, not the comparison that matters (see §3).

> ⚠️ **Provenance (2026-10-30):** the macOS row above predates the Metal
> engine, the QTSQ memory shell and the eleven shipped JS drivers, and no
> longer matches the current binary. Re-measured on an Apple M2 Pro / macOS
> 15.7.3 with 50-run subprocess medians:
>
> | Command | Median | Note |
> |---|---:|---|
> | `sofuu version` (CLI only, no engine) | 5.0 ms | process + dyld floor |
> | `sofuu eval "1+1"` | 11.4 ms | was 33.3 ms before the two 2026-10-30 boot fixes |
> | `sofuu capabilities` | 26.3 ms | pays a real Metal device probe, by design |
>
> The first fix removed a boot-time `sofuu.capabilities()` call inside
> `memory_api.js` that created a Metal device (~18–20 ms) before any user code
> ran; the second moved libcurl's global init (TLS trust store) and both multi
> handles off the boot path to the first transfer (~6 ms). Both are described
> in the CHANGELOG entries of 2026-10-30. The Linux and Pi 5 rows have not been
> re-measured against the current tree and are historical until they are.

---

## 3. 🧠 Memory Footprint (Resident Set Size)

**This is the comparison that matters.** Sofuu is a single process carrying the whole AI
application stack — JS/TS orchestration, inference, encrypted vector memory, agents, MCP, HTTP.
Its peer is not another language runtime; it is the multi-process stack a developer assembles
today to do the same job:

| Runtime & Stack | Subsystems Active | Measured RSS (Idle) |
| :--- | :--- | :--- |
| **Sofuu v0.3.0** | QuickJS + libuv + QTSQ Cognitive Vector Engine | **~12.4 MB** |
| **Node.js** | Node process alone (no external vector database) | **~38.5 MB** |
| **Node.js + Local Vector DB** | Node process + ChromaDB / Qdrant background daemon | **~185.0 MB** |
| **Python + LangChain + Chroma** | Python interpreter + PyTorch base + Chroma service | **~390.0 MB** |

> 📌 **Key Takeaway:** one Sofuu process (~12.4 MB idle, engine + vector memory initialized)
> replaces what otherwise takes a runtime process *plus* a vector-database daemon *plus* an
> inference server — ~185 MB with Node.js, ~390 MB with Python/LangChain/Chroma — and removes
> the socket round-trips between them (see §4). The "Node.js alone" row is shown only as the
> interpreter baseline. These rows date to the 0.3.0-era measurement and should be re-verified
> against the current tree before being quoted in marketing.

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

> ⚠️ **Superseded (2026-10-31).** The 3.8 / 4.2 / 38.2 ms rows above are
> historical. The numbers this project actually stands behind are the measured
> ones below and in §2's note: `sofuu eval "1+1"` on an M2 Pro, 31 runs,
> wall-clock including process spawn — **22.5 ms median** (min 21.8) for the
> current tree, against **61.5 ms median** for the published `0.3.0-beta`
> artifact measured the same way, in the same session. The four driver payloads
> and four Metal kernels are now stored gzip-compressed and inflated at init, so
> the size work did not cost startup: it improved it.

## 5. 📦 Binary Footprint & Stripped Size

- **Hard Architectural Invariant:** $\le 5.5\text{ MiB}$ ($5,767,168$ bytes) stripped release binary.
Measured 2026-10-31 (`stat -f%z` on each file; cap is 5,767,168 B):

| Artifact | Bytes | vs. cap |
|---|---:|---|
| **macOS ARM64, this tree** (`make size-check`) | 5,559,904 B (5.30 MiB) | **passes**, 207,264 B headroom |
| macOS ARM64, published archive | 5,655,808 B (5.39 MiB) | passes (stale — predates 2026-10-31 fixes) |
| Linux ARM64, published archive | 6,163,520 B (5.88 MiB) | **396,352 B over the cap** |
| Linux x86_64, published archive | 7,130,904 B (6.80 MiB) | **1,363,736 B over the cap** |
| Linux ARM64, local `dist/` copy (Oct 29) | 6,337,512 B | **570,344 B over** |
| Linux x86_64, local `dist/` copy (Oct 30) | 7,362,456 B | **1,595,288 B over** |

The Linux archives (both the published set and the older local `dist/` copies)
must be rebuilt before they ship; `scripts/verify-dist.sh` prints exactly those
lines and refuses them.

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
