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
* **Offline Vector Subsystem**: Instant zero-byte character-trigram embeddings (`hash-768`), learned semantic projectors, and joint text/image embedding. The larger clean-room `emb-m` models are **not bundled** — they run as a separate service consumed through the embedding API (see §3.7).
* **Context-Economy ML Gates**: 5 on-device learned models (~8k params each) preventing token waste — the only ML models bundled in the binary, with an upgrade on real trajectory data in progress (`PLAN-TINY-ML-UPGRADE.md`).

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
* **HNSW Vector Store (`hnsw.rs`)**: Pure-Rust Hierarchical Navigable Small World graph supporting high-dimensional nearest-neighbor search with logarithmic complexity. **Recall fixed 2026-10-13** — the results heap shared the candidate frontier's reversed ordering, so it evicted near neighbours as the graph grew (recall@10 0.82 → 0.07 from 50 to 2000 vectors); it now uses its own naturally-ordered type, with regression tests. See `AUDIT-2026-10-13-BRAIN-CMA.md`.
* **TF-IDF Dream Consolidation**: Offline background consolidation passes summarize decaying episodic memories into compact semantic facts during idle cycles.

> **Roadmap:** the audit-driven fix and upgrade sequence for §3.3/§3.4 lives in
> [`ROADMAP-BRAIN-CMA.md`](ROADMAP-BRAIN-CMA.md) (B0–B7, dependency-ordered, each with an evidence gate).

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
* **Bundled vs. served (decided 2026-10-12)**: The binary bundles only the tiny models — `hash-768` (0 B), the `sem1/2` + `img1` projectors, and the 5 context-economy gates. The clean-room `emb-m` models (`s-mem-10m`, `s-docs-15m`, `s-code-20m`) are **never compiled into the binary**: they run as a separate embedding service and are consumed through the standard API surface (`ai.embed` / `ai.embedBatch` with a provider or custom base URL), exactly like any remote embedding provider. The in-binary `hash-768` default stays the zero-key, zero-network path.
* **Context-Economy ML Gates (`sofuu.ml`)** — bundled, and being upgraded on real trajectory data (`PLAN-TINY-ML-UPGRADE.md`):
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
| **Offline Embeddings** | ✅ Shipped | `hash-768` (0 B, 2 µs), `sem1/2`, `img1-64` | `emb-m` is served via API, **not bundled** (see C3); in-binary path keeps `hash-768` as the zero-byte default. |
| **ML Gates** | ✅ Shipped | Beat constant & logistic baselines on synthetic data | Upgrade the 5 bundled gates on real trajectory data (`PLAN-TINY-ML-UPGRADE.md`) + live shadow-mode A/B logging on real user traffic. |

---

## 5. Phased Roadmap (Milestone-Driven)

### Milestone C1: Local Inference & Tool Calling Fusion
- [x] **Local tool-call protocol + stream parser (2026-10-12).** Tool specs ride the system turn (Hermes/Qwen-3 `<tools>` block), prior assistant calls render back into `<tool_call>` blocks in ChatML history, and tool results use the Qwen `<tool_response>` user-turn convention. The local stream scanner (`rt/local_tools.rs`) parses `<tool_call>` — plus the `<tools>` near-miss small models emit — out of the token stream and surfaces completed calls as OpenAI-shaped `tool_calls` on `ai.stream` / `ai.complete` / `ai.streamLocal`, the exact shape the agent loop already consumes for every remote provider. Conservative by construction: a block becomes a call only when its payload parses as `{name, arguments}`; with no declared tools the local output passes through byte-for-byte.
- [x] **Local models drive autonomous agents with no remote API key — verified live 2026-10-12** on `Qwen3-0.6B-Q4_K_M` (GGUF): agent → `<tool_call>` → `read_file` executed → tool result → final answer carrying the file's real contents (2 LLM calls, 1 tool call, zero keys). LFM-2 remains untested for tool calling.
- [ ] Structured output enforcement on local logits (JSON-schema grammar-guided decoding in the sampler) — deferred until the inference-engine refactor lands; the text-protocol parser above is the compatibility layer until then.

### Milestone C2: QTSQ Power-Loss Resilience & WAL
- [x] **Durable atomic publish (2026-10-12).** Every persistence rename now fsyncs the staged file before the rename and fsyncs the directory after it (`rt/wal.rs::publish`) — without the first fsync the data can be lost while the rename survives, the classic power-loss corruption that left a brain file that exists and is garbage. Wired into the brain save, the KV page flush, the KV index write, and the semantic-migration publish.
- [x] **Write-ahead journal.** `rt/wal.rs::Journal` records the intent (target, staged path, sha256) and fsyncs it BEFORE any rename; `recover` replays it on open — rolling the commit forward when the staged bytes are intact and match, back when they are not. A journal younger than 30 s is treated as a live writer's and never stolen (desktop + CLI on one project).
- [x] **Atomic page commits + orphan adoption.** `index.json` was a bare `fs::write` (a crash mid-write left a truncated index that `kv_store_open` silently read as "no pages", orphaning the entire store) — it is now staged and published like a page, and a corrupt index says so out loud. Pages on disk that the index does not know about (a crash between page write and index flush) are adopted on open instead of being clobbered by the next save's id reuse.
- [x] **Checksum verification on load.** Per-stream content integrity is provided by the codec itself — compress writes sha256(original bytes) into the container header and `qtsq_decompress` verifies it after decode (`qtsq_singularity.c` / `qtsq_radiation.c`). The adapter adds the crash-window digest via the journal and verifies it during recovery.
- [x] **Adjacent fix found while testing:** `ensure_dir` used single-level `create_dir`, so a nested KV store path whose parent did not exist yet never persisted anything — and the old `kv_flush` reported success anyway. Now `create_dir_all`, and `kv_flush` reports a real failure.

### Milestone C3: `emb-m` Embedding Service (API-Served, Never Bundled)
- [ ] Complete pilot training of `s-mem-10m` (agent memory retrieval model) from permissively licensed datasets.
- [ ] Serve `emb-m` through an embedding API (hosted or self-hosted) as a separate service — the models are deliberately **not** compiled into the binary (no `include_bytes!`, no in-binary forward pass, zero binary-footprint cost).
- [ ] Integrate the runtime through the existing provider surface (`ai.embed` / `ai.embedBatch` with a provider or custom base URL) — no new in-binary model loader.
- [ ] Keep `hash-768` as the zero-byte in-binary default; the semantic upgrade is a configuration choice, not a binary change.

### Milestone C4: Bundled Tiny-Model Upgrade & Shadow A/B Telemetry
- [ ] Upgrade the 5 bundled context-economy gates with real-world trajectory data per `PLAN-TINY-ML-UPGRADE.md` (≤15k params each, <50 µs CPU forward pass, weights still compiled in via `include_bytes!`).
- [ ] Implement non-invasive session logging to measure real-world token savings produced by the 5 ML gates.
- [ ] Measure tokens-per-completed-task end metric on production workflows.
