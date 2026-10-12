# 🧠 Plan: Sofuu Core — The AI-Native Runtime & Cognitive OS

> *The foundational engine, QuickJS/libuv event loop, TypeScript stripper, Cognitive Memory Architecture (CMA), QTSQ encrypted brain, agents, and native MCP in a single ≤5.0 MB binary.*

---

## 1. Executive Vision & Scope

Sofuu is fundamentally an **AI-native server-side JavaScript/TypeScript runtime and Edge-AI Operating System**. While engines like Node.js, Deno, and Bun were designed for traditional web servers and microservices, Sofuu is purpose-built from the ground up for local AI workflows, autonomous agents, and long-lived edge intelligence.

Every capability an AI application needs is integrated directly into the core binary with zero external npm dependencies:
* **~3 ms Cold Boot**: QuickJS (ES2023 complete) paired with libuv's battle-tested async event loop.
* **Zero-Config TypeScript**: Instant in-memory TS stripping and ESM bundling with zero compiler installation.
* **Cognitive Memory Architecture (CMA)**: 4-tier human-like memory model with Ebbinghaus forgetting decay, k-means consolidation, and HNSW graph retrieval.
* **QTSQ Persistent Encrypted Brain**: Proprietary tensor container (`.qtsq`) storing memories and KV pages locally across restarts.
* **Autonomous Agent Loop**: Budget-controlled multi-step reasoning, sub-agent delegation, and sandboxed coding tools.
* **Native Model Context Protocol (MCP)**: In-tree JSON-RPC client and server connecting tools and data sources.
* **Offline Vector Subsystem**: Instant zero-byte character-trigram embeddings (`hash-768`), learned semantic projectors, and joint text/image embedding.
* **Context-Economy ML Gates**: 5 on-device learned models (~8k params each) preventing token waste.

---

## 2. Core Architectural Subsystems

```
┌────────────────────────────────────────────────────────────────────────┐
│                        sofuu CLI / Embeddable API                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         Sofuu Runtime Shell                            │
├───────────────────────────────────┬────────────────────────────────────┤
│     QuickJS Engine (ES2023)       │      libuv Async Event Loop        │
│     Pure C, Zero-JIT Interpreter  │      Timers, Sockets, FS, Signals  │
├───────────────────────────────────┴────────────────────────────────────┤
│                     In-Tree Rust Runtime Subsystems                    │
│                                                                        │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────┐  │
│  │   Agent & Tool Loop   │  │   CMA & HNSW Memory  │  │  Native MCP │  │
│  │   • Budgets & Traces  │  │   • 4-Tier Decay     │  │  • Client   │  │
│  │   • Code & Web Tools  │  │   • Dream Consolidation│ │  • Server  │  │
│  └───────────────────────┘  └──────────────────────┘  └─────────────┘  │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────┐  │
│  │  TypeScript Stripper  │  │   QTSQ Brain Codec   │  │   RLM Loop  │  │
│  │  • Zero-Config AST    │  │   • Encrypted At-Rest│  │  • Sandboxed│  │
│  │  • ESM Bundler        │  │   • SSD Paging & LRU │  │    Reader   │  │
│  └───────────────────────┘  └──────────────────────┘  └─────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │             Offline Embeddings & Context ML Gates                │  │
│  │    • hash-768 (2 µs, 0 B)  • SEM1/SEM2  • Image Projector        │  │
│  │    • Freshness / Supervisor / Compaction / Relevance Gates       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Deep Dive & Specifications

### 3.1 QuickJS & libuv Integration
* **Why QuickJS**: V8 is 30+ MB of C++ complexity and requires JIT (banned on iOS). QuickJS is ~100 KB of pure C, implements ES2023 completely, and boots in ~3 ms. For AI workloads, network and tensor math dominate execution time, making raw interpreter micro-benchmarks irrelevant.
* **Why libuv**: Battle-tested event loop powering Node.js. Provides cross-platform asynchronous file I/O, timers, DNS, TCP/HTTP sockets, and process management.
* **Memory Ownership**: QuickJS `JSRuntime` and `JSContext` are thread-local. All cross-thread operations dispatch messages via libuv async handles (`uv_async_send`) to prevent data races.

### 3.2 TypeScript Execution & Bundling
* **Instant Stripping (`ts.rs`)**: Strips type annotations, interfaces, generics, access modifiers, and enum declarations in a single linear pass over the byte stream. TypeScript runs directly (`sofuu run script.ts`) without generating `.js` files or requiring `tsconfig.json`.
* **ESM Bundler (`bundler.rs`)**: Resolves relative imports, traverses local dependency graphs, and bundles projects into a single executable JavaScript artifact (`sofuu bundle entry.ts -o dist.js`).

### 3.3 Cognitive Memory Architecture (CMA) & HNSW
* **4-Tier Cognitive Hierarchy**:
  1. **Working Memory**: Active session context, short-term buffer with immediate recall.
  2. **Episodic Memory**: Specific conversation turns and tool events, subject to Ebbinghaus forgetting decay.
  3. **Semantic Memory**: Consolidated facts and knowledge extracted from episodes via k-means clustering.
  4. **Entity Memory**: Named entities, user preferences, and stable identities pinned indefinitely.
* **Ebbinghaus Forgetting Decay**: Memory retention decays exponentially over time:
  $$R = e^{-\frac{t}{S}}$$
  where $S$ is memory stability increased by positive recall reinforcement (`markPositive`).
* **HNSW Vector Store (`hnsw.rs`)**: Pure-Rust Hierarchical Navigable Small World graph supporting high-dimensional nearest-neighbor search with logarithmic complexity.
* **TF-IDF Dream Consolidation**: Offline background consolidation passes summarize decaying episodic memories into compact semantic facts during idle cycles.

### 3.4 QTSQ Encrypted Brain (`.qtsq`)
* **Format Structure**:
  - `memories`: Compressed flat Float32 vectors stored at F4, F8, F16, or F32 precision.
  - `metadata`: JSON stream containing textual records, roles, memory tier tags, and relational graph links.
* **SSD Paging & RAM LRU**: Uses a 2-page RAM LRU buffer with direct SSD paging, enabling devices with 512 MB RAM to search gigabyte-sized memory stores without loading the entire file into memory.
* **At-Rest Encryption**: Encrypted using AES-256 with a machine-local key, ensuring user data privacy on shared or edge devices.

### 3.5 Autonomous Agent Runtime & Tools
* **Agent Engine (`sofuu.agent`)**:
  - Declarative definition: `sofuu.agent.define({ name, system, tools, budget, memory, agents })`.
  - Multi-agent delegation: Parent agents can delegate sub-tasks to specialized sub-agents with hierarchical budget trees (`maxSteps`, `maxDepth`, `maxWallMs`).
  - Cancellation Tree: Cancelling a root task propagates cleanly down the entire agent tree via cooperative interruption tokens.
  - Live Trace Visualizer: Renders formatted execution trees showing tool calls, intermediate observations, and reasoning branches.
* **Built-in Coding & Web Tools**:
  - Filesystem: `read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, `glob`.
  - Sandboxed Shell: `bash` (jailed to the project directory; strippable via `SOFUU_NO_SHELL=1`).
  - Web Search: Built-in keyless search via DuckDuckGo, plus optional Brave/Tavily keys.

### 3.6 Native Model Context Protocol (MCP)
* **Client Architecture**: Connects to external MCP child servers over `stdio` or HTTP SSE. Parses JSON-RPC 2.0 messages safely in Rust (`mcp/jsonrpc.rs`).
* **Multi-Server Routing**: Supports multiple concurrent servers with automatic tool name clash resolution.
* **Server Architecture**: Allows Sofuu itself to expose local tools, brain memory, and inference as an MCP server (`sofuu.mcp.serve`) for Claude Desktop, Cursor, or other AI clients.

### 3.7 Vector Embeddings & Context-Economy ML Gates
* **Offline Vector Spaces**:
  - `hash-768`: 768-dimensional character-trigram projection using MurmurHash3 and L2-normalization. Runs in $< 2\ \mu\text{s}$ per doc with **0 bytes** model footprint.
  - `sem1-64` / `sem2-64`: Compact learned linear/table projectors for 64-dim memory indexing.
  - `img1-64`: Joint text/image embedding (`image.rs`) projecting image bytes into the same metric space as text.
* **Context-Economy ML Gates (`sofuu.ml`)**:
  - 5 tiny on-device ML models (~8k params each) that advise the agent loop:
    1. **Freshness Gate**: Predicts if retrieved context is stale.
    2. **Supervisor Gate**: Evaluates if the agent's current step is making progress or looping.
    3. **Compaction Gate**: Identifies erasable conversation turns before hitting context limits.
    4. **Relevance Gate**: Filters out irrelevant retrieved memories.
    5. **Budget Gate**: Predicts remaining steps needed to solve the task.

---

## 4. Current State & Quality Gaps

| Component | Status | Verified State | Open Items & Next Steps |
|---|---|---|---|
| **QuickJS & libuv** | ✅ Shipped | ~3 ms startup, ES2023 complete, 431 tests green | Evaluate zero-copy ArrayBuffer transfers to Rust. |
| **TypeScript Stripper** | ✅ Shipped | Pure-Rust linear pass, zero compiler installation | Support `.tsx` JSX syntax stripping. |
| **CMA & HNSW** | ✅ Shipped | 4 tiers, decay, consolidation, HNSW graph | Wire automated background "dreaming" on idle ticks. |
| **QTSQ Brain** | ✅ Shipped | Round-trip verified, encrypted at rest, SSD paging | Implement write-ahead logging (WAL) for power-loss crash safety. |
| **Agent Loop** | ✅ Shipped | Delegation, budget trees, cancel, traces, web tool | Wire in-process local inference model tool-calling. |
| **MCP Client/Server** | ✅ Shipped | JSON-RPC 2.0, stdio + SSE, multi-server routing | Add OAuth/Bearer token authentication support for remote MCP. |
| **Offline Embeddings** | ✅ Shipped | `hash-768` (0 B, 2 µs), `sem1/2`, `img1-64` | Complete clean-room `emb-m` pilot model training/quantization. |
| **ML Gates** | ✅ Shipped | Beat constant & logistic baselines on synthetic data | Implement live shadow-mode A/B logging on real user traffic. |

---

## 5. Phased Roadmap (Milestone-Driven)

### Milestone C1: Local Inference & Tool Calling Fusion
- [ ] Connect `sofuu.agent` tool-calling directly to `sofuu.ai.streamLocal()` using JSON schema grammar-guided decoding.
- [ ] Enable local models (Qwen-3, LFM-2) to drive autonomous agents without any remote API key.
- [ ] Implement structured output enforcement on local logits.

### Milestone C2: QTSQ Power-Loss Resilience & WAL
- [ ] Implement write-ahead journaling for the QTSQ persistence engine to prevent brain file corruption during unexpected power loss or device reboots.
- [ ] Add atomic page commits and checksum verification on load.

### Milestone C3: Clean-Room `emb-m` Distilled Embedder
- [ ] Complete pilot training of `s-mem-10m` (agent memory retrieval model) from permissively licensed datasets.
- [ ] Quantize model to 4-bit/8-bit weights and embed runtime forward pass directly in `sofuu-core` ($\le 1.5\text{ MB}$ footprint).
- [ ] Provide drop-in semantic upgrade over `hash-768`.

### Milestone C4: Shadow A/B Telemetry for Context Gates
- [ ] Implement non-invasive session logging to measure real-world token savings produced by the 5 ML gates.
- [ ] Measure tokens-per-completed-task end metric on production workflows.
