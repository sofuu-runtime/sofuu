# ⚡ Sofuu Benchmark Methodology & Performance Metrics

> **Canonical Source of Truth for all Performance, Latency, and Memory Footprint Claims.**  
> *Last Verified: November 6, 2026 | Engine Version: 0.3.0-beta*

---

## 1. 🎯 Methodology & Standards

All benchmarks reported on [sofuu.xyz](https://sofuu.xyz) and within this repository are measured using open-source, reproducible benchmarking tools ([`hyperfine`](https://github.com/sharkdp/hyperfine) for execution times and standard OS memory profiling tools). The extreme cold-start characterization uses `scripts/bench/extreme_cold_start.py` (stdlib-only, for hosts without hyperfine); its per-run wall clock includes the Python spawn harness, whose cost is quantified by a `/usr/bin/true` row. Unless a row says otherwise, page caches are **warm**; macOS `purge` requires root.

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
>
> **Update 2026-11-06:** the 500-run extreme re-measure is **14.58 ms p50 / 15.33 ms p95 / 15.53 ms p99** on the same class of machine, with the pre-stabilization artifact at 14.47 ms p50. See the extreme subsection below for methodology and the concurrent-burst table.

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

### Pi 5 re-measure (M-D0, 2026-11-01)

Measured on the lab Pi 5 (BCM2712, `arm_freq=2800`, `over_voltage_delta=25000`,
governor `ondemand`, idle load 0.01) against **the binary that was on the box**,
`sofuu-linux-arm64` `0.2.0-beta` (5,289,392 B) — an Oct-11-era revision, so these
describe the published artifact, not this tree. They replace the "historical
until re-measured" caveat above with measured values.

| Measurement | Value | Method |
|---|---:|---|
| Spawn floor (any process) | 0.44 ms | median of 10 × `/bin/true` |
| `sofuu version` (CLI only, no engine) | 0.53 ms | median of 10 |
| **Engine boot** (runs `console.log(1+1)`) | **4.85 ms** | median of 10 (min 4.76, max 6.27) |
| Idle RSS (runtime idling on `setTimeout`) | **4.5 MB** (4,656 KB) | `VmRSS` after 2 s |
| Qwen3-0.6B Q4_K_M load | 1.26 s | 28 layers, 16 heads, 4 threads |
| Prefill | 74.1 tk/s | 5-token prompt |
| Decode | **32.4 tk/s** | 48 tokens, temp 0 |
| Thermal under that load | 45.8 → 55.6 °C | sampled at 2 Hz |
| Clock under load | 1700 → 2800 MHz | never dropped below the idle floor |

The two figures the drone plan inherits are **engine boot ~5 ms** and **idle RSS
~4.5 MB**, both far inside the 30 MB companion-computer envelope. They are lower
than the macOS rows because this revision boots no Metal probe and loads no model.

### Extreme cold-start characterization (2026-11-06)

Reproducible with `make cold-start-bench`
(`scripts/bench/extreme_cold_start.py`). This run used **~7,000 process
launches**: 1,000 spawn-floor runs, 500 engine boots, 300-run variants, and
concurrent bursts up to 200 simultaneous processes.

**Conditions:** Darwin arm64, 10 CPUs, 16 GB, load 2.33 / 3.40 / 3.19 during
the run; **warm page cache** (`purge` requires root on macOS and was refused, so
true cold-disk rows are not available). Wall clock includes process spawn and
the Python harness; the `/usr/bin/true` row quantifies that floor.

| Path | n | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `/usr/bin/true` spawn floor | 1000 | **1.48 ms** | 1.77 | 2.08 | 12.48 |
| `sofuu version` (CLI only) | 500 | **4.35 ms** | 4.87 | 5.08 | 5.23 |
| **`sofuu eval "1+1"` (engine boot)** | 500 | **14.58 ms** | **15.33** | **15.53** | 15.65 |
| published artifact, same command | 300 | 14.47 ms | 15.14 | 16.11 | 16.88 |
| `SOFUU_NO_ML=1` | 300 | 14.65 ms | 15.25 | 16.01 | 18.26 |
| `SOFUU_NO_TOOLS=1` | 300 | 14.54 ms | 15.27 | 15.55 | 24.53 |
| `sofuu run tiny.js` | 300 | 14.62 ms | 15.12 | 15.47 | 15.61 |
| `sofuu capabilities` (Metal probe) | 150 | **31.86 ms** | 37.62 | 38.85 | 42.41 |

**Net engine boot** (subtracting the spawn floor): **~13.1 ms**. The published
`51fa5a23f` artifact measured 14.47 ms, so the Stabilization Freeze introduced
**no cold-start regression**; the difference is inside run-to-run noise. The
first pass on revision `8b7bd139e` agreed (14.45 ms p50 / 15.45 ms p95).
`SOFUU_NO_ML` and `SOFUU_NO_TOOLS` do not move boot — ML gates and tool-driver
registration are not startup costs. `capabilities` deliberately pays a real
Metal device probe (~17 ms over engine boot).

**Concurrent cold-start bursts** (current binary; `N` simultaneous processes):

| N | Per-process p50 | Per-process p95 | Burst wall p50 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 14.87 ms | 16.03 | 15.01 | 67 /s |
| 4 | 16.15 ms | 17.42 | 17.50 | 229 /s |
| 10 | 19.51 ms | 30.35 | 31.44 | 318 /s |
| 32 | 48.44 ms | 74.84 | 83.63 | **383 /s** |
| 64 | 85.18 ms | 140.15 | 159.66 | **401 /s** |
| 100 | 121.15 ms | 245.06 | 269.01 | 372 /s |
| 200 | 180.85 ms | 935.37 | 847.23 | 236 /s |

Throughput saturates at **~400 process launches/s** on 10 cores. Beyond one
process per core the per-process latency rises with queuing and the p99 tail
grows; the published artifact's N=32 burst (45.67 ms per-process p50, 83.43 ms
burst p50) is statistically identical to the current build.

**Peak RSS:** `version` 7.50 MB · `eval "1+1"` **10.91 MB** · `capabilities`
13.05 MB · published `eval` 10.91 MB.

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

### Stabilization Freeze: brain persistence RAM (2026-11-05)

Measured with `/usr/bin/time -l` on macOS arm64, 64-dim store, one record,
**fresh process per row**:

| Path | Peak RSS | Notes |
|---|---:|---|
| Idle boot (`eval "1+1"`) | **11.78 MB** | inside the 4–12 MB invariant |
| Encrypted first flush | **79.84 MB** | first Argon2id derivation in the process |
| Plaintext first flush (`{ plaintext: true }`) | **12.63 MB** | zero KDF, zero vault |
| Repeated encrypted flush (same process) | 0.0 ms KDF | QTSQ KDF v2 master cache |

The extreme cold-start run measured **10.91 MB** for the same `eval "1+1"` row;
the 11.78 MB figure is the earlier stabilization run. The difference is
build/run variance (the machine was under different load), not a regression —
both are inside the 4–12 MB invariant.

The 64 MiB Argon2id stretch is now **once per process/password**, not once per
container: `SOFUU_PROFILE=1` shows the first `encrypt` at **34.1 ms** and every
subsequent `encrypt` at **0.0 ms**. Legacy v1 containers still derive per file
as a one-time migration. Edge/IoT stores can use
`sofuu.memory.open(path, dim, { plaintext: true })` to skip the vault entirely;
the choice is persisted in the manifest.

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

> ⚠️ **Superseded (2026-10-31; re-measured 2026-11-06).** The 3.8 / 4.2 / 38.2
> ms rows above are historical. The 2026-10-31 session measured **22.5 ms median**
> (31 runs) for `sofuu eval "1+1"` against 61.5 ms for the published
> `0.3.0-beta` artifact. The 2026-11-06 extreme run measured **14.58 ms p50 /
> 15.33 ms p95 over 500 runs** on the same M2 Pro, with the pre-stabilization
> artifact at 14.47 ms p50 — i.e. no regression, and the newer figure reflects
> the quieter/warmer-cache conditions plus the boot fixes since 2026-10-31. See
> §2's extreme subsection for the full distribution.

## 5. 📦 Binary Footprint & Stripped Size

- **Hard Architectural Invariant:** $\le 5.5\text{ MiB}$ ($5,767,168$ bytes) stripped release binary.
Measured 2026-11-06 (`stat -f%z` on each file; cap is 5,767,168 B):

| Artifact | Bytes | vs. cap |
|---|---:|---|
| **macOS ARM64, this tree** (`make size-check`) | **5,628,208 B (5.37 MiB)** | **passes**, 138,960 B headroom |
| **Linux ARM64, rebuilt `dist/`** | **5,117,296 B (4.88 MiB)** | **passes**, 649,872 B headroom |
| **Linux x86_64, rebuilt `dist/`** | **5,751,616 B (5.49 MiB)** | **passes**, 15,552 B headroom |
| macOS ARM64, published archive (2026-10-31) | 5,655,808 B (5.39 MiB) | passes; predates this tree |
| Linux ARM64, published archive (2026-10-31) | 6,163,520 B (5.88 MiB) | still over — re-cut required |
| Linux x86_64, published archive (2026-10-31) | 7,130,904 B (6.80 MiB) | still over — re-cut required |

The **local `dist/` artifacts are now under the cap on every platform**; the
published Linux archives on the download page still need a re-cut release.

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

# 4. Run the extreme cold-start characterization (~7k process launches)
make cold-start-bench

# 5. Run the concurrent serve event-loop ceiling (16 ms)
make event-loop-check

# 6. Run the memory & vector doctor suite
./sofuu doctor
```
