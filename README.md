<p align="center"><img src="docs/sofuu-logo.png" width="118" alt="Sofuu logo"></p>

# ⚡ Sofuu (素風)

> *The general-purpose runtime for AI — built to run anywhere from servers to smart devices. A compact, sovereign systems engine (Rust/C core, QuickJS/libuv) combining autonomous agent loops, the Kaze agent language, encrypted cognitive vector memory (QTSQ), and native MCP in a single self-contained native binary (under 10 MB). Zero Python environments, zero external daemons, zero setup baggage.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0--beta-orange.svg)](https://github.com/sofuu-runtime/sofuu/releases)
[![Binary Size](https://img.shields.io/badge/binary-%3C%2010%20MB-emerald.svg)](https://sofuu.xyz)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://sofuu.xyz/install)
[![TypeScript](https://img.shields.io/badge/TypeScript-Native-blue.svg)](https://sofuu.xyz)
[![MCP Native](https://img.shields.io/badge/MCP-Native-purple.svg)](https://sofuu.xyz/docs#plugins)
[![Docs](https://img.shields.io/badge/docs-sofuu.xyz-blue.svg)](https://sofuu.xyz/docs)
[![X](https://img.shields.io/badge/follow-%40sofuu__runtime-black?logo=x&logoColor=white)](https://x.com/sofuu_runtime)

> 🚧 **Active development — iterating fast.** Open-source and MIT-licensed.

---

## Why Sofuu?

Today's AI stack is fragmented and bloated: running an agent or workflow typically requires Python virtualenvs, Node.js packages, external vector databases (Chroma/Pinecone), background servers (Ollama/vLLM), and multi-gigabyte Docker containers.

**Sofuu collapses this entire stack into a single lightweight binary (< 10 MB).**

Whether you write in modern **TypeScript/JavaScript** or declare pure autonomous agents in **Kaze (`.kaze`)**, Sofuu executes everything in-process with zero external dependencies:

* 🚀 **Instant Startup (< 20 ms cold start):** Single native Mach-O / ELF binary. Zero JIT warmup pauses, making it 100% compliant with strict platform boundaries (iOS App Store, RTOS, edge IoT).
* 🧠 **Built-in Cognitive Vector Memory (QTSQ):** Persistent semantic memory in encrypted `.qtsq` containers. Zero-byte offline embeddings, sub-2ms hardware SIMD search, zero localhost database servers.
* 🤖 **Autonomous Agents & Kaze Language:** Full reasoning loops, cooperative scheduling, domain toolpacks, fail-closed capability gates (`needs: ...`), and compile directly to single-binary native executables (`sofuu compile`).
* 🔌 **First-Class Model Context Protocol (MCP):** Connect to or host MCP tools over stdio, SSE, or direct in-process transports with zero npx/Node overhead.
* 🛡 **Unified R7 Security Firewall:** Deterministic capability sandboxing governing tools, shell, filesystem, and network with fail-closed policies.
* 📱 **Runs Everywhere (`libsofuu`):** Cloud servers, developer workstations, Raspberry Pi 5, edge IoT gateways, and embedded mobile apps (iOS & Android).

---

## Quick Start (30 Seconds)

### 1. Install Sofuu

```bash
# One-line install (Linux & macOS, auto-detects architecture)
curl -fsSL https://sofuu.xyz/install | sh

# Or via npm
npm install -g sofuu
```

Official pre-compiled single binaries are available for macOS (Apple Silicon) and Linux (`x86_64` and `arm64` / Raspberry Pi 5).

---

### 2. Native AI Memory in Standard JavaScript / TypeScript

Most frameworks require API keys or 2 GB model downloads just to test vector memory. With Sofuu, offline semantic memory works instantly out of the box:

```javascript
// memory.js — Run directly: $ sofuu run memory.js
const brain = sofuu.memory.open("./brain.qtsq", 768);

// 1. Embed and persist facts into encrypted on-disk vector storage (0-byte model)
brain.remember(
  sofuu.ai.embedLocal("native AI runtime"),
  "Sofuu executes AI natively with zero background daemons."
);

// 2. Query offline vector memory via hardware SIMD in microseconds
const [match] = brain.recall(sofuu.ai.embedLocal("daemons"), 1);
console.log("Recalled from vector brain:", match.text);
```

```bash
$ sofuu run memory.js
Recalled from vector brain: Sofuu executes AI natively with zero background daemons.
# Completed in ~2ms · 0 background daemons · 0 cloud API calls
```

---

### 3. Build an Autonomous Agent with Tools

```javascript
// researcher.js — Run: $ sofuu run researcher.js
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Search the web, then synthesize answer with sources.",
  tools: ["coding", "web", "memory"],
  memory: "agent",
});

const result = await sofuu.agent.run("researcher", "What features shipped in QuickJS 2024?");
console.log(result.answer);
```

---

### 4. Pure Kaze Agent & Direct Native Compilation

You can program 100% pure autonomous AI agents in Kaze (`.kaze`) with **zero JavaScript or Python**, and compile them directly into standalone native executables:

```kaze
# agent.kaze — Pure Declarative Autonomous Agent
needs: ai, tool, policy, memory, log, alert

filter monitor_and_act(source, metric_val, threshold, max_limit):
    log.info("Telemetry ingest: " + source + " = " + metric_val)

    # 1. Fail-closed security: block credential leaks before execution
    if policy detects secrets in source:
        reject "Security refusal: credential leak in source"

    # 2. AI reasoning with continuous vector memory
    if metric_val > threshold:
        let analysis = ai.prompt("Synthesize incident analysis for " + source)
        remember analysis as source + "_incident"

        # 3. Guardrailed action dispatch
        if metric_val >= max_limit:
            tool.call("actuator", source + ":TRIGGER_FAILSAFE")
            send alert "CRITICAL: Failsafe triggered on " + source
            allow "Emergency action taken"

        tool.call("notifier", source + ":DISPATCH_WARNING")
        allow "Warning dispatched"

    allow "Nominal telemetry"
```

Compile directly into a standalone, single-file native binary (under 10 MB) with production 12-factor configuration:

```bash
# 1. Compile pure Kaze directly into a standalone single binary:
$ sofuu compile agent.kaze -o my-agent

# 2. Run anywhere with JSON config or CLI parameters (zero runtime dependencies):
$ ./my-agent --config agent.config.json

# 3. Or run continuously 24/7 with the autonomous supervisor daemon:
$ ./my-agent --config agent.config.json --daemon --every 5s
```

---

## Core Runtime vs. Experimental Preview

To maintain rock-solid production reliability, Sofuu clearly delineates core guaranteed subsystems from preview modules:

| Subsystem | Readiness | What it covers |
| :--- | :---: | :--- |
| **Autonomous Agent Engine** | 🟢 **Core** | `sofuu.agent`, cooperative scheduling (`scheduler.yield`), step budgets, resilient editing (`smart_edit`), and real-time trace streaming. |
| **Domain Toolpacks** | 🟢 **Core** | Modular toolpacks (`coding`, `web`, `memory`, `system`), single-instruction fuel sandbox, numbered unified diffs. |
| **Standalone Compiler** | 🟢 **Core** | `sofuu compile`: native executables (under 10 MB) with 32-byte `SOFUU_BIN_V1` trailer, ad-hoc signing, sub-5ms cold starts. |
| **Cognitive Memory (QTSQ)** | 🟢 **Core** | On-device `.qtsq` vector containers, cached KDF v2, plaintext zero-KDF mode, hardware SIMD search. |
| **Model Context Protocol (MCP)** | 🟢 **Core** | Native client & server over stdio, SSE, and direct in-process transports; dynamic tool discovery. |
| **Kaze Agent Language** | 🟢 **Core** | Declarative agent language, linear arena VM, zero GC pauses, fail-closed policy gates, 12-factor configuration. |
| **Unified R7 Security Firewall** | 🟢 **Core** | Deterministic path traversal blocking, private CIDR SSRF filtering, credential shielding (`.env`, `id_rsa`). |
| **Local Inference Engine** | 🧪 **Preview** | Built-in custom GGUF/SafeTensors engine. (For deployed local inference, use `sofuu-plugin-llamacpp` or remote providers). |
| **Robotics & MAVLink** | 🧪 **Preview** | `sofuu drone`, MAVLink v2 codec, and GPIO line timers through R7 hardware safety gates. |
| **Native Frontend Engine** | 🧪 **Preview** | `sofuu dev` streaming JSX/TSX transpiler and reactive micro-runtime. |

---

## Core Capabilities

### 1. Model Context Protocol (MCP) & Standalone Micro-Binaries

Sofuu features first-class native MCP client and server support with **zero external dependencies** — no Node.js, Python, or `npx` required.

#### Compile Any MCP Server into a Standalone Native Binary (`sofuu mcp build`)
Turn any TypeScript or JavaScript MCP server into a single, self-contained native binary (~2.05 MiB) with instant startup and sub-5 MB memory footprint:

```bash
# 1. Compile MCP server directly into a standalone native micro-binary (~2.05 MiB):
sofuu mcp build server.ts -o dist/weather-mcp

# 2. Inspect tools and execute calls directly from the terminal (zero code):
sofuu mcp inspect ./dist/weather-mcp --call get_weather --args '{"city":"Tokyo"}'

# 3. Synchronize server definitions directly into your IDE:
sofuu mcp link cursor           # Link to Cursor IDE
sofuu mcp link claude-desktop   # Link to Claude Desktop
```

* **Ultra-Lean Footprint (~2.05 MiB):** Targets the dedicated `sofuu-micro` native base. Dead-code elimination strips over 3.2 MB of unused GPU and tensor machinery, yielding sub-5 MB RSS and 0 ms cold execution.
* **Lightweight Multi-Server Workspaces:** Run 10+ concurrent MCP servers in your editor or autonomous agent swarms with virtually zero system resource drain.

#### Write & Connect MCP Servers in Code
```javascript
// 1. In-process MCP server with zero serialization overhead
const server = sofuu.mcp.serve({ name: "math-service", version: "1.0.0" });
server.tool("fibonacci", {
  description: "Calculate nth fibonacci",
  schema: { type: "object", properties: { n: { type: "number" } }, required: ["n"] }
}, async ({ n }) => ({ result: 55 }));

server.start(); // Spawns over stdio for any standard MCP client

// 2. Connect to external or direct in-memory MCP servers
const client = await sofuu.mcp.connect("npx -y @modelcontextprotocol/server-filesystem /tmp");
const tools = await client.listTools();
await client.call("read_file", { path: "/tmp/notes.txt" });
client.disconnect();

// 3. Bind directly to an autonomous agent loop with zero IPC hops:
const directClient = await sofuu.mcp.connect(server, { transport: "direct" });
const agent = sofuu.agent.define({ name: "solver", tools: [directClient] });
```

Full guide: [docs/MCP.md](docs/MCP.md).

---

### 2. Five-Shield Plugin System & Extensibility

Sofuu is extensible directly from JavaScript or TypeScript with **no build step**: drop a file into `./plugins/` (or run `sofuu plugin add`) and it is discovered, validated, and registered at startup.

Every plugin is isolated behind the **Five Shields Pipeline**, ensuring that an untrusted or crashing plugin can never bring down the host runtime:

| Shield | Protection Layer | Failure Code | Behavior |
| :--- | :--- | :---: | :--- |
| **1. Schema** | Input & Output Validation | `E_SCHEMA` | Inputs validated before execution; callable never receives bad payloads. |
| **2. Policy Jail** | Network & Filesystem Boundaries | `E_DENIED` | Enforces `permissions.net` allowlist, private CIDR SSRF block, and jailed `ctx.fs`. |
| **3. Circuit Breaker** | Cascading Failure Defense | `E_CIRCUIT_OPEN` | Automatically trips after 5 consecutive failures / 30s to prevent hanging. |
| **4. Watchdog** | Hardware Timer Deadlines | `E_TIMEOUT` | Bounded execution budget (default 15s) with cooperative abort signal (`ctx.signal`). |
| **5. Panic Guard** | Runtime Crash Insulation | `E_TOOL` | Unhandled exceptions, rejected promises, or circular references return typed errors. |

#### Five Extensible Categories:
* **Tools (`defineTool`):** LLM-callable actions with auto-generated schemas and permission gates.
* **Inference Providers (`defineProvider`):** Custom LLM backends (e.g. `sofuu plugin add llamacpp`, Ollama, or local HTTP endpoints).
* **Storage & Vector Stores (`defineVectorStore` / `defineStorage`):** Connect remote databases (e.g. pgvector, Pinecone) as an asynchronous dual-write mirror behind local QTSQ memory.
* **Lifecycle Services (`defineService`):** Background hooks for `onStart`, `onTurnStart`, `onToolCall`, and `onTurnEnd`.
* **OpenAPI & MCP Ingest (`useOpenAPI`, `ingestMcp`):** Zero-code bridging of external REST APIs and MCP servers.

```bash
# Plugin CLI Workflow
sofuu plugin add llamacpp                   # Install official local LLM provider bridge
sofuu plugin init weather --category=tool   # Scaffold new tool in ./plugins/weather.ts
sofuu plugin validate plugins/weather.ts    # Parse, check types, and verify registration
sofuu plugin test weather '{"city":"Oslo"}' # Invoke tool once and print execution envelope
sofuu plugin list                           # View all registered plugins and circuit breaker states
```

Full guide: [docs/PLUGINS.md](docs/PLUGINS.md).

---

### 3. Cloud LLM Streaming (Any Provider)

Stream from OpenAI, Anthropic, DeepSeek, OpenRouter, or any OpenAI-compatible server with zero SDKs to install:

```javascript
// server.js
const server = sofuu.http.createServer(async (req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  const stream = sofuu.ai.stream("Explain SIMD vector registers in simple terms.", {
    provider: "openai",
    model: "gpt-4o",
  });
  for await (const chunk of stream) {
    res.write(chunk.text);
  }
  res.end();
});

server.listen(3000, "0.0.0.0");
```

---

### 4. Universal Execution Sandbox

`sofuu.sandbox` executes JavaScript or TypeScript in an isolated QuickJS realm with strict memory and instruction-fuel limits:

```javascript
// Fresh realm, run untrusted code, and destroy
const report = sofuu.sandbox.run("1 + 2 * 3", {
  preset: "strict", // 2s deadline, 5M instruction fuel meter, 32MB memory cap
});
console.log(report.result);   // 7
console.log(report.fuelUsed); // instructions charged
```

---

### 5. 24/7 Agent Observability & Crash Recovery

Built-in append-only JSONL tracing designed for autonomous supervisor loops:

* **Append-Only Traces:** Turns and tool steps append to `.sofuu/logs/agent-<run_id>.jsonl` with total-order sequence numbers.
* **Secret Redaction:** API keys, authorization tokens, and private paths are automatically scrubbed at the write choke point before touching disk.
* **Crash Post-Mortems:** Unhandled crashes generate diagnostic cards in `.sofuu/crashes/<crash_id>.json` capturing the failed step and suggested fixes.
* **Resumable Execution:** Transparently resume interrupted runs with `sofuu resume <run_id>`.

```bash
sofuu logs                     # List past agent runs
sofuu logs show <run_id>       # Render colored, step-by-step transcript
sofuu logs --last-error        # View post-mortem card for the latest failure
sofuu resume <run_id>          # Resume an interrupted run from its last checkpoint
```

---

## CLI Cheatsheet

```bash
# Execution & REPL
sofuu run <file.js|ts>         # Run a JavaScript or TypeScript file
sofuu eval "<code>"            # Evaluate an expression inline
sofuu repl                     # Interactive JS/TS REPL

# Autonomous Agents & Kaze
sofuu agent run <name> "task"  # Run an autonomous agent headless
sofuu script run <file.kaze>   # Run a pure Kaze agent routine
sofuu compile <file.kaze|ts>   # Compile agent into standalone native binary (under 10 MB)
sofuu daemon install|run       # Manage 24/7 autonomous supervisor daemon

# Model Context Protocol (MCP) & Standalone Micro-Binaries
sofuu mcp build <server.ts> -o <out> # Compile MCP server to standalone native micro-binary (~2.05 MiB)
sofuu mcp inspect <server|bin> --call <tool> --args '<json>' # Inspect & invoke tools directly
sofuu mcp test "npx -y <srv>"  # Protocol handshake and tool inventory verification
sofuu mcp link cursor|claude   # Synchronize server configuration into Cursor & Claude Desktop

# Extensibility & 5-Shield Plugins
sofuu plugin add <llamacpp|ollama>   # Install official plugin bridges
sofuu plugin init <name> --category=tool # Scaffold new plugin
sofuu plugin validate <file.ts>      # Validate registration, schemas & shields
sofuu plugin test <name> '<json>'    # Test run a tool with circuit state reporting
sofuu plugin list                    # List registered plugins & circuit breaker states

# Memory & Serving
sofuu brain verify <path>      # Verify QTSQ cognitive memory container
sofuu serve --brain <path>     # Host OpenAI-compatible HTTP server

# System & Health
sofuu doctor                   # Verify QTSQ linkage, brain round-trip, and system health
sofuu version                  # Display semver, commit revision, and engine info
```

---

## Embedded Integration (`libsofuu`)

Sofuu ships an embeddable C ABI library (`libsofuu`) with first-class Swift and Kotlin bindings for iOS, Android, and desktop apps:

```swift
// Swift (iOS / macOS)
import Sofuu

let sofuu = try Sofuu()
let brain = try Brain.openDefault(sofuu: sofuu)
try brain.remember("user prefers dark mode")
let matches = try brain.recall("theme preference", k: 1)
```

```kotlin
// Kotlin (Android)
val sofuu = SofuuRuntime.create()
val vec = sofuu.embedLocal("offline embeddings")
```

```c
// Native C ABI
#include "sofuu_embed.h"

SofuuRuntime *rt = sofuu_rt_new(NULL);
char *out = NULL;
sofuu_rt_call(rt, "memory.open", "{\"path\":\"./brain.qtsq\",\"dim\":768}", &out);
sofuu_free(out);
sofuu_rt_free(rt);
```

---

## Verified Benchmarks Snapshot

| Metric | Measured Value | Comparison / Notes |
| :--- | :--- | :--- |
| **Binary Size** | **Under 10 MB** (single binary) | macOS Apple Silicon (~5.4 MiB), Linux (~4.9–5.5 MiB). Bundles QuickJS, libuv, Rust core, QTSQ memory. |
| **Cold Start** | **14.6 ms median** (wall clock) | `sofuu eval "1+1"` including process spawn (Apple M2 Pro). Process floor `sofuu version` is 5.0 ms. |
| **HTTP Time to First Response** | **39.4 ms measured** | Native in-process server (Node 22: 32.4 ms, Bun 1.3: 25.8 ms). |
| **Idle Memory (RSS)** | **~14.9 MB measured** | Clean process footprint with zero background daemons. |
| **Vector Memory Query** | **< 2 ms** | In-process SIMD cosine/L2 search over encrypted `.qtsq` files. |

> 👉 *For complete hardware benchmark receipts across 40+ local models, Apple Silicon Metal GPU kernels, and Raspberry Pi 5 runs, see [BENCHMARKS.md](BENCHMARKS.md).*

---

## Building from Source

```bash
# Prerequisites: Rust (cargo), clang, cmake, make
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu

make deps                      # Build vendored libuv
make                           # Build release binary (./sofuu)
make test                      # Run test suite
./sofuu doctor                 # Health check
```

---

## License & Governance

Sofuu is open source under the **MIT License** — see [LICENSE](LICENSE).  
The underlying [QTSQ](https://github.com/haruhito2666/qtsq) cognitive memory engine is open source under the Apache License 2.0.

* **Author & Maintainer:** Built and maintained by **Haruhito**.
* **Governance:** Kaze and Sofuu are maintained exclusively by the author. External pull requests are not accepted.
* **Website:** [sofuu.xyz](https://sofuu.xyz)
* **Documentation:** [sofuu.xyz/docs](https://sofuu.xyz/docs)
* **X (Twitter):** [@sofuu_runtime](https://x.com/sofuu_runtime)
