# Sofuu (素風) — Architecture & Project Overview

## Overview

Sofuu is an early-stage, bootstrapped, open-core project founded and built by **Haruhito**.

Sofuu is an ultra-lightweight, AI-native **JavaScript/TypeScript runtime** designed to make building, running, and deploying AI applications simpler, faster, and self-contained. 

Instead of requiring developers to assemble and maintain a bloated constellation of separate packages and background daemons—a JavaScript runtime (Node.js/Bun), AI SDKs (LangChain), agent frameworks, background inference daemons (Ollama/vLLM), vector databases (Chroma/Pinecone), MCP clients, and HTTP web servers—Sofuu brings every layer together inside a **single, standalone < 5 MB binary (4.0 MB stripped release)**.

The goal is to make local, edge, and cloud AI development feel like running a single native executable with zero external dependencies and zero setup.

---

## What Sofuu Provides

Sofuu executes JavaScript and TypeScript while providing first-class AI primitives directly at the runtime layer:

* **In-Process Local LLM Inference**: Native execution of GGUF (v2/v3), SafeTensors, PyTorch (`.pt`/`.pth`), and NumPy checkpoints directly on CPU (NEON/AVX2) and Apple Metal with zero daemons and zero Python.
* **Continuous Memory Architecture (CMA)**: Persistent, on-device encrypted semantic vector memory that automatically manages episodic recall, decay, and reinforcement without external vector databases.
* **Autonomous Agents & Tool Loops**: Built-in agent execution loops with multi-step reasoning, self-reflection, planning, and recursive execution.
* **Native Model Context Protocol (MCP)**: First-class MCP client and server support to connect models directly to external tools, databases, and APIs without extra libraries.
* **High-Throughput Streaming & Completion APIs**: Low-latency token generation with asynchronous backpressure handling.
* **SIMD Vector Mathematics**: Microsecond vector operations (cosine similarity, dot product, L2 distance) powered by hand-tuned NEON and AVX2 kernels.
* **Offline Hash & Semantic Embeddings**: Instant zero-network 768-dim hash embeddings and local semantic vector spaces.
* **Built-in Async HTTP Server**: Expose streaming AI endpoints directly over HTTP from the same process without Express or external frameworks.
* **Interactive AI Terminal CLI (`sofuu chat`)**: Immediate interactive assistant with persistent memory, zero coding required for end users.
* **Developer Tooling**: Built-in runner, bundler, and evaluator for modern JS/TS scripts (`sofuu run app.ts`).
* **Ultra-Small Standalone Footprint**: Single binary strictly under 5 MB (`4.0 MB stripped release`) with `< 4 ms` cold startup.

---

## Runtime Architecture

Sofuu is engineered as a hybrid high-performance runtime:
* **JavaScript Engine**: Embedded **QuickJS** providing lightweight, sandboxed ES2023+ execution with instant cold starts and minimal RAM overhead.
* **Async Event Loop**: **libuv** driving cross-platform asynchronous I/O, timers, networking, and filesystem operations.
* **Engine Shell & Core**: Written in **Rust** (`crates/sofuu-core`, `sofuu-capi`, `sofuu-ffi`) for memory safety, concurrency, encrypted memory subsystems, and CLI tooling.
* **SIMD & Metal Kernels**: Hand-tuned **C++ / ARM NEON / x86_64 AVX2** and **Apple Metal Shaders** (`src/simd/infer_kernels.cpp`, `src/metal/infer_metal.metal`) executing tensor operations at hardware saturation.

### Key Architectural Pillars:
* **Strict Binary Budget**: Shipped in a **4.0 MB stripped binary** (strictly $< 5.0\text{ MB}$ ceiling).
* **Instant Startup**: Launches in **$< 4\text{ ms}$**; zero-copy mmap tensor loading initializes multi-gigabyte models in **$17–38\text{ ms}$**.
* **Zero Daemons, Zero IPC**: Eliminates HTTP localhost overhead and background process crashes—the runtime process *is* the inference and memory engine.
* **Local-First & Private**: All data, vector indexes, sessions, and memory are stored in local AES-256-GCM encrypted brain segments.

---

## In-Process Local Inference Engine

Sofuu features an embedded inference engine capable of loading and executing foundation models on local hardware without Python, PyTorch, CUDA toolkits, or Ollama:

* **Universal Formats**: Ingests **GGUF** (v2/v3), **SafeTensors**, **PyTorch checkpoints**, and **NumPy arrays** with zero format conversion.
* **Dynamic Block Graph**: Dynamically routes and executes any transformer or hybrid architecture (LLaMA, Qwen 2.5/3, Mistral, SmolLM, Gemma 1/2, Phi-3/4, Liquid LFM-2, and ternary quantized models).
* **Hardware Acceleration**:
  * **Apple Silicon**: Zero-copy Metal Performance Shaders (MPS) unified memory execution.
  * **ARM NEON**: Cache-line aligned 8-wide float vectorization and specialized Raspberry Pi 5 nibble-plane optimizations.
  * **x86_64 AVX2 / FMA**: 8-wide FMA matrix kernels with cache-friendly tiled layouts.
* **Advanced Bandwidth & Latency Optimizations**:
  * Non-temporal streaming weights (cache-bypass streaming for large weight matrices).
  * Multi-path lookahead tree speculation.
  * Sliding Window Attention (SWA) with attention sink preservation.

---

## Continuous Memory Architecture (CMA v2) & Native Structured Storage

Sofuu includes an integrated cognitive vector memory and structured database platform powered 100% by **QTSQ**:
* **100% QTSQ, Zero SQLite**: Replaces third-party databases with proprietary, encrypted QTSQ containers (`brain.qtsq` and `storage.qtsq`). Zero SQLite dependency, zero unencrypted plaintext disk leaks.
* **Encrypted On-Disk Storage**: Associative facts, episodic turns, and vector embeddings are stored in `.qtsq` files encrypted with hardware AES-256-GCM.
* **1-Line Learning Facade (`brain.learn`)**: Ingests unstructured text with automatic subword morphological TF-IDF projection into vector space, eliminating vector boilerplate.
* **Local-First Asynchronous Fact Extractor**: Automatically distills key user preferences, updates, and facts in the background using local models (e.g. 0.6B / 1B) for $0 cost, 0 ms chat delay, and 100% privacy.
* **Automated Contradiction Resolution**: Detects superseded facts and tombstones conflicting historical records from the HNSW graph automatically.
* **Dual-Tier Associative Retrieval**: Combines fast 768-dimensional hash embeddings for instant keyword/syntactic lookup with 64-dimensional semantic embedding spaces and BM25 rare-token re-ranking.
* **Self-Pruning & Reinforcement**: Built-in exponential Ebbinghaus decay, reinforcement on positive user feedback, and semantic diversity floors to prevent memory drift.
* **Native Structured Columnar Storage (`sofuu.storage`)**: Strict-schema columnar tables (10 data types), 128-row bounded chunking, and 48-byte checkpoint skip indexing enabling sub-millisecond chronological pagination.
* **Dual-Strategy Adaptive Compressor**: Direct White Dwarf RAW storing with SHA-256 integrity for bit-sized / sub-KB data (< 1,024 B), and QTC entropy compression for large blocks with automatic zero-error fallback ladders.
* **Weekly Container Partitioning (`storage_YYYY_Www.qtsq`)**: Prevents monolithic database growth, isolates weekly history, and eliminates corruption cascades.
* **Zero External Dependencies**: Eliminates the need for external vector databases (Pinecone, Chroma, Milvus, or sqlite-vss) and relational engines.

---

## Built-In Agent Tooling & MCP

Sofuu makes agentic workflows a native primitive:
* **Core Agent Tool Belt**: Native tools available directly to agents (`read`, `write`, `edit`, `grep`, `glob`, `list_dir`, `bash`, web search, and persistent brain recall).
* **Model Context Protocol (MCP)**: Native client and server capabilities. Agents can connect to MCP servers over stdio or HTTP SSE to orchestrate local tools, enterprise systems, and remote services with zero external SDKs.

---

## Developer Experience & Scripting

### 1. Zero-Code Terminal Assistant
Users who do not write JavaScript can use Sofuu directly out of the box:
```bash
sofuu chat
```
Launches an interactive terminal chat with local models, persistent memory, and agent capabilities.

### 2. Native AI Scripting (JavaScript / TypeScript)
Developers can build custom AI pipelines and agents with clean runtime APIs:

```ts
// In-process local inference
import { ai, memory, agent } from "sofuu";

// Stream tokens from a local GGUF model
for await (const token of ai.stream("Explain KV-cache optimization", { model: "./model.gguf" })) {
  process.stdout.write(token);
}

// Store in encrypted persistent memory
await memory.remember("User prefers concise SIMD explanations", { tags: ["preferences"] });

// Run an autonomous agent with native tools
const result = await agent.run("Audit the src/ directory for memory leaks and generate a report");
```

### 3. Built-In Async HTTP Server
Serve AI streaming APIs directly without Express or Fastify:
```ts
import { serve, ai } from "sofuu";

serve({ port: 3000 }, async (req) => {
  const { prompt } = await req.json();
  return ai.stream(prompt, { model: "./model.gguf" });
});
```

---

## Cross-Platform & Edge Support

Sofuu compiles into a static standalone binary across:
* **macOS** (Apple Silicon M-series & Intel)
* **Linux** (x86_64, aarch64, Raspberry Pi 5)
* **Windows** (x86_64)
* **Edge & Embedded Targets**: C-API (`sofuu-capi`) and Swift/Kotlin bindings for direct integration into iOS and Android applications.

One-line installation:
```bash
curl -fsSL https://sofuu.org/install | sh
```

---

## Open-Core Model

Sofuu is licensed under the **MIT License**.

The runtime, inference kernels, agent loops, memory subsystems, and developer CLI are open source and developed publicly. 

Future commercial layers (managed cloud deployments, hosted collaborative agent fleets, enterprise authentication, and centralized telemetry for private clusters) will build on top of the open-source core without restricting the standalone runtime.

---

## Project Identity & Links

* **Founder & Lead Developer**: **Haruhito**
* **Website**: [https://sofuu.org](https://sofuu.org) / [https://sofuu.xyz](https://sofuu.xyz)
* **Source Repository**: [https://github.com/sofuu-runtime/sofuu](https://github.com/sofuu-runtime/sofuu)
* **Landing Page**: [https://github.com/sofuu-runtime/landing-page](https://github.com/sofuu-runtime/landing-page)

---

## Short GitHub Organization Description

> Sofuu is an open-core, AI-native JavaScript/TypeScript runtime built by Haruhito. In-process local LLM inference, encrypted memory (CMA), autonomous agents, native MCP, and HTTP serving in a single < 5 MB binary (4.0 MB stripped). Zero daemons, zero Python.

---

## Recommended GitHub Organization README (`.github/profile/README.md`)

```markdown
# Sofuu (素風)

The AI-native JavaScript runtime. In-process local LLM inference, encrypted persistent memory, autonomous agents, and native MCP in a single **< 5 MB binary (4.0 MB stripped)**.

Zero background daemons. Zero Python dependencies. Zero external vector databases.

```bash
curl -fsSL https://sofuu.org/install | sh

# 1. Interactive terminal chat with local models & memory (zero code needed)
sofuu chat

# 2. Run an AI script or agent
sofuu run app.ts
```

## Highlights

- **Single `< 5 MB` Binary**: Shipped as a 4.0 MB stripped executable with `< 4 ms` cold startup.
- **In-Process Inference Engine**: Runs GGUF, SafeTensors, PyTorch, and NumPy checkpoints natively on CPU (NEON/AVX2) and Apple Metal with zero daemons.
- **Continuous Memory Architecture (CMA)**: Encrypted on-device semantic and episodic associative memory.
- **Native MCP Client & Server**: Direct Model Context Protocol tool calling without third-party frameworks.
- **Built-in Agent Belt**: Reasoning loops, reflection, and file/terminal tools out of the box.
- **Integrated HTTP Server**: Stream LLM tokens and serve agent endpoints directly from JS.
- **Cross-Platform**: macOS, Linux, Windows, Raspberry Pi 5, iOS & Android native bindings.

## Architecture

Sofuu combines:
- **QuickJS** for ultra-lightweight, sandboxed JavaScript execution.
- **libuv** for cross-platform asynchronous I/O and networking.
- **Rust** for memory safety, encrypted brain subsystems, and engine orchestration.
- **Hand-tuned C++ SIMD & Apple Metal kernels** for bare-metal tensor execution.

## License

Sofuu core is open source under the [MIT License](LICENSE). Built in public by **Haruhito**.
```
