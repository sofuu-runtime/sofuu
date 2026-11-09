<p align="center"><img src="https://sofuu.xyz/logo.png" width="118" alt="Sofuu logo"></p>

# Sofuu (素風)

> *A lightweight native runtime for AI applications, tools, and agents (under 10 MB). Written in Rust, QuickJS, and libuv. Unites autonomous agent loops, declarative Kaze routines, on-device vector memory (QTSQ), and native Model Context Protocol (MCP) in a single binary — zero Python environments, zero external daemons.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0--beta-orange.svg)](https://github.com/sofuu-runtime/sofuu/releases)
[![Binary Size](https://img.shields.io/badge/binary-%3C%2010%20MB-emerald.svg)](https://sofuu.xyz)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://sofuu.xyz/install)
[![TypeScript](https://img.shields.io/badge/TypeScript-Native-blue.svg)](https://sofuu.xyz)
[![MCP Native](https://img.shields.io/badge/MCP-Native-purple.svg)](https://sofuu.xyz/docs#plugins)
[![Docs](https://img.shields.io/badge/docs-sofuu.xyz-blue.svg)](https://sofuu.xyz/docs)
[![X](https://img.shields.io/badge/follow-%40sofuu__runtime-black?logo=x&logoColor=white)](https://x.com/sofuu_runtime)

> **Active development — iterating fast.** Open-source and MIT-licensed.

---

## Why Sofuu?

Today's AI development stack is fragmented: running an agent or workflow typically requires Python virtualenvs, Node.js packages, external vector databases (Chroma/Pinecone), background servers (Ollama/vLLM), and multi-gigabyte Docker containers.

**Sofuu collapses this entire stack into a single lightweight binary (< 10 MB).**

Whether you write in modern **TypeScript/JavaScript** or declare autonomous agents in **Kaze (`.kaze`)**, Sofuu executes everything in-process with zero external dependencies:

* **Instant Cold Starts (< 20 ms):** Single native Mach-O / ELF binary without JIT warmup delays. Complies with strict execution environments like the iOS App Store and RTOS.
* **On-Device Vector Memory (QTSQ):** Persistent semantic memory stored in encrypted `.qtsq` files. Zero-byte offline embeddings and sub-2ms SIMD vector search without running database servers.
* **Autonomous Agents & Kaze Language:** Multi-step agent loops, cooperative scheduling, domain toolpacks, fail-closed capability gates (`needs: ...`), and single-binary compilation (`sofuu compile`).
* **Native Model Context Protocol (MCP):** Complete client and server implementation supporting stdio, HTTP, SSE, and in-memory direct transports with zero Node.js/npx overhead.
* **Deterministic Sandboxing:** Fine-grained policy gates governing tool execution, filesystem access, network egress, and shell commands.
* **Universal Deployment (`libsofuu`):** Cloud servers, workstations, Raspberry Pi 5, edge IoT devices, and mobile apps (iOS & Android).

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

### 2. Everything in One Process: Write AI as Normal Code

Sofuu collapses vector databases, agent orchestrators, and tool sandboxes into standard JavaScript / TypeScript inside a single lightweight binary (< 10 MB):

```javascript
// quickstart.js - memory · native tools · agents · zero daemons
const brain = sofuu.memory.open("./brain.qtsq", 768);
brain.remember(sofuu.ai.embedLocal("runtime"), "AI runtime. Under 10 MB.");

// Microsecond SIMD vector recall (< 2ms, zero database servers)
const [hit] = brain.recall(sofuu.ai.embedLocal("speed"), 1);

// Sandboxed native tool executed in-process with policy gates
const sys = sofuu.tools.execute("system_info", {});

// Autonomous agent - define once with memory & tools (0 models needed to test)
const agent = sofuu.agent.define({
  name: "assistant",
  system: "Cite sources.",
  tools: ["coding", "memory", "web"],
});

console.log(hit.text, "•", sys.platform || process.platform, "• agent:", agent.name);
// -> AI runtime. Under 10 MB. · darwin · agent: assistant
```

*Install with line 1 · run with line 2 · memory, tools, and agents — all in-process, zero daemons, 22.5 ms median cold start measured on macOS arm64 (31 runs, wall clock incl. spawn).*

Run it immediately with zero setup, or compile it into a standalone native executable:

```bash
# 1. Run directly with sub-20ms cold start
$ sofuu run quickstart.js

# 2. Compile into a standalone native executable with zero dependencies
$ sofuu compile quickstart.js -o my-agent
$ ./my-agent
```

> *Vector memory (`brain.remember` / `brain.recall`) and native tool execution run 100% offline with zero external servers. When invoking `sofuu.agent.run()`, connect your local GGUF/llama.cpp model or any standard remote streaming provider.*

---

### 3. Instant Terminal Testing (No Files Needed)

You can also test Sofuu's capabilities directly from your terminal:

```bash
# Launch interactive AI chat with persistent session memory
$ sofuu chat

# One-liner vector memory test
$ sofuu eval 'const b = sofuu.memory.open("./brain.qtsq", 768); b.remember(sofuu.ai.embedLocal("fast"), "Sofuu starts in <20ms"); console.log(b.recall(sofuu.ai.embedLocal("speed"), 1)[0].text);'
```

---

### 4. Prefer Pure Declarative Agents? Try Kaze (`.kaze`)

For production agents with zero JavaScript or Python, define declarative capability gates (`needs: ...`) and compile directly into native binaries:

```kaze
# agent.kaze — Declarative Autonomous Agent
needs: ai, tool, policy, memory, log

filter monitor_and_act(source, metric_val, threshold):
    log.info("Telemetry ingest: " + source + " = " + metric_val)

    # 1. Fail-closed policy check: block credential leaks before execution
    if policy detects secrets in source:
        reject "Security refusal: credential leak in source"

    # 2. Reasoning with continuous on-device vector memory
    if metric_val > threshold:
        let analysis = ai.prompt("Synthesize incident analysis for " + source)
        remember analysis as source + "_incident"
        tool.call("notifier", source + ":DISPATCH_ALERT")
        allow "Alert dispatched"

    allow "Nominal telemetry"
```

```bash
# Compile pure Kaze directly into a standalone single-file binary:
$ sofuu compile agent.kaze -o my-agent

# Run anywhere with CLI flags or 12-factor config:
$ ./my-agent --config agent.config.json
```

---

### 5. Production Showcase Agents (`example-agents/`)

The repository includes pre-built showcase agents written 100% in pure Kaze (zero JavaScript, zero package dependencies):

* **Quant Sentinel (`quant_sentinel/`):** Algorithmic risk sentinel enforcing leverage hard-caps and secret-leak detection.
* **Med Protocol (`med_protocol/`):** Clinical triage agent automating ESI emergency escalations from patient vitals.
* **Agri Sentinel (`agri_sentinel/`):** Autonomous greenhouse climate monitor controlling irrigation and heat relays.
* **Paper Trader (`paper_trader/`):** Simulated multi-asset exchange with portfolio risk guardrails.

Compile all agents in one step, or test their standalone native executables:

```bash
# 1. Compile all showcase agents with one command:
$ ./example-agents/build.sh

# 2. Test Quant Sentinel (Within 3.0x leverage cap -> Order Routed):
$ ./example-agents/quant_sentinel/quant-sentinel --arg ticker=BTC --arg signal=BUY --arg leverage=2 --arg price=65000
# -> [VERDICT] ALLOW "Synthesized market thesis: BTC target $68,200..."

# 3. Test Risk Policy Gate (Leverage > 3.0x -> Instant Fail-Closed Rejection):
$ ./example-agents/quant_sentinel/quant-sentinel --arg ticker=BTC --arg signal=BUY --arg leverage=5 --arg price=65000
# -> [VERDICT] REJECT Risk policy violation: leverage 5 exceeds hard cap of 3.0x

# 4. Test Clinical Vitals Triage (Hypoxia + Tachycardia -> ESI Level 1 Escalation):
$ ./example-agents/med_protocol/med-protocol --arg patient_id=P101 --arg heart_rate=145 --arg spo2=88 --arg systolic_bp=85
# -> [VERDICT] REJECT CRITICAL: Severe vitals decompensation -> Emergency code triggered
```

---

## Core Runtime vs. Experimental Preview

Sofuu maintains a clear demarcation between core guaranteed subsystems and preview modules:

| Subsystem | Status | Description |
| :--- | :---: | :--- |
| **Autonomous Agent Engine** | **Core** | `sofuu.agent`, cooperative scheduling (`scheduler.yield`), step budgets, resilient editing (`smart_edit`), and real-time trace streaming. |
| **Domain Toolpacks** | **Core** | Modular toolpacks (`coding`, `web`, `memory`, `system`), single-instruction fuel sandbox, numbered unified diffs. |
| **Standalone Compiler** | **Core** | `sofuu compile`: native executables (under 10 MB) with 32-byte `SOFUU_BIN_V1` trailer, ad-hoc signing, sub-5ms cold starts. |
| **Vector Memory (QTSQ)** | **Core** | On-device `.qtsq` vector containers, cached KDF v2, plaintext zero-KDF mode, hardware SIMD search. |
| **Model Context Protocol (MCP)** | **Core** | Native client & server supporting stdio, HTTP, SSE, and direct in-process transports; dynamic tool discovery. |
| **Kaze Agent Language** | **Core** | Declarative agent language, linear arena VM, zero GC pauses, fail-closed policy gates, 12-factor configuration. |
| **Security Sandbox** | **Core** | Deterministic path traversal blocking, private CIDR SSRF filtering, credential shielding (`.env`, `id_rsa`). |
| **Local Inference Engine** | **Preview** | Built-in custom GGUF/SafeTensors engine. (For deployed local inference, use `sofuu-plugin-llamacpp` or remote providers). |
| **Robotics & MAVLink** | **Preview** | `sofuu drone`, MAVLink v2 codec, and GPIO line timers through hardware safety gates. |
| **Native Frontend Engine** | **Preview** | `sofuu dev` streaming JSX/TSX transpiler and reactive micro-runtime. |

---

## Core Capabilities

### 1. Model Context Protocol (MCP)

Sofuu provides a complete native MCP client and server implementation (protocols `2025-06-18` and `2024-11-05`) with **zero external dependencies** — no Node.js, Python, or `npx` required.

#### MCP Client
Connect to local or remote MCP servers to discover tools, read resources, and execute actions:

```javascript
// Connect via stdio, Streamable HTTP, or legacy SSE
const client = await sofuu.mcp.connect("npx -y @modelcontextprotocol/server-filesystem /tmp");
// or: await sofuu.mcp.connect({ url: "http://127.0.0.1:9000/mcp", transport: "http" });

// Discover tool inventory and call tools
const tools = await client.listTools();
const result = await client.call("read_file", { path: "/tmp/notes.txt" });
console.log(result.text);

client.disconnect();
```

* **Transports Supported:** `stdio` (command spawning with newline-delimited JSON-RPC), `http` (Streamable HTTP with session handling), `sse` (HTTP+SSE), and `direct` (in-memory).
* **Configuration & Policy (`~/.sofuu/mcp.json`):** Configure persistent servers with per-server tool allow/deny filters (`tools: { allow: ["read_file"], deny: ["write_file"] }`) and secret store resolution (`TOKEN: { secret: "MCP_TOKEN" }`).

#### MCP Server
Build standalone or in-process MCP servers in TypeScript or JavaScript:

```javascript
const server = sofuu.mcp.serve({ name: "math-service", version: "1.0.0" });

// Register tools with schema validation
server.tool("fibonacci", {
  description: "Calculate the nth Fibonacci number",
  schema: { type: "object", properties: { n: { type: "number" } }, required: ["n"] }
}, async ({ n }) => {
  return { result: fib(n) };
});

// Register resources and prompts
server.resource("system://metrics", { mimeType: "application/json" }, async () => {
  return JSON.stringify({ cpu: process.cpuUsage(), memory: process.memoryUsage() });
});

// Composable onion middleware
server.use(async (ctx, next) => {
  console.log(`[MCP] Invoking ${ctx.toolName}`);
  await next();
});

server.start(); // Spawns over stdio for Cursor, Claude Desktop, or custom clients
```

#### In-Memory Direct Transport for Agents
Connect an in-process MCP server directly to an autonomous agent without process spawning or serialization overhead:

```javascript
const directClient = await sofuu.mcp.connect(server, { transport: "direct" });
const agent = sofuu.agent.define({
  name: "calculator-agent",
  tools: [directClient],
});
```

#### CLI Tooling & Standalone Micro-Binary Compilation
Sofuu includes CLI commands to test, inspect, link, and compile MCP servers:

```bash
# Test handshake, protocol negotiation, and tool count
sofuu mcp test "npx -y @modelcontextprotocol/server-filesystem /tmp"

# Inspect and execute a tool directly from the terminal
sofuu mcp inspect "npx -y some-server" --call echo --args '{"text":"hello"}'

# Synchronize server configurations into your editor
sofuu mcp link cursor           # Configure in Cursor IDE
sofuu mcp link claude-desktop   # Configure in Claude Desktop

# Compile any MCP server into a standalone single-file micro-binary (~2 MB)
sofuu mcp build server.ts -o dist/my-mcp-server
```

* **Micro-Profile Optimization (`sofuu-micro` base):** `sofuu mcp build` automatically targets the minimal `micro` compilation profile (~2 MB). It strips out cognitive agent loops, chat TUI, and local inference machinery, keeping only QuickJS, libuv, and the streamable MCP protocol engine.
* **Instant Start & Zero Dependencies:** The resulting standalone executable runs with `< 5 ms` cold start and `< 5 MB` idle RSS, completely eliminating Node.js, `npx`, and Python dependencies from your MCP toolchain.

Documentation: [sofuu.xyz/mcp](https://sofuu.xyz/mcp)

---

### 2. Plugins & Extensibility

Sofuu is extensible directly from TypeScript or JavaScript with **no build step**. Drop a file into `./plugins/` (or `~/.sofuu/plugins/`), or install via `sofuu plugin add`, and it is discovered, validated, and registered at startup.

#### Plugin Categories
* **Tools (`defineTool`):** Actions callable by LLMs with Zod or JSON Schema input validation and fine-grained permissions.
* **Providers (`defineProvider`):** Custom or local model backends (e.g. `sofuu plugin add llamacpp`, Ollama, or custom HTTP endpoints).
* **Storage & Vector Stores (`defineVectorStore` / `defineStorage`):** Connect remote databases (e.g. pgvector, Pinecone) as an asynchronous dual-write mirror behind local QTSQ memory.
* **Services (`defineService`):** Background lifecycle hooks (`onStart`, `onTurnStart`, `onToolCall`, `onToolResult`, `onTurnEnd`, `onStop`).
* **OpenAPI & MCP Ingest (`useOpenAPI`, `ingestMcp`):** Automatic zero-code bridging of external REST APIs and MCP servers.

#### Defining a Sandboxed Tool
```typescript
import { defineTool, z } from "sofuu";

export default defineTool({
  name: "weather",
  description: "Get current weather for a city",
  schema: z.object({ city: z.string() }),
  timeoutMs: 5000,
  permissions: { net: ["api.weatherapi.com"] }, // Network allowlist
  circuitBreaker: { failureThreshold: 5, resetTimeoutMs: 30000 },
  async run({ city }, ctx) {
    const res = await ctx.fetch(`https://api.weatherapi.com/v1/current.json?q=${city}`);
    return res.json();
  },
});
```

#### Fault Isolation & Execution Safety
Every plugin runs with isolated boundaries:
* **Input & Output Validation:** Arguments are validated against the schema before handler execution (`E_SCHEMA`).
* **Resource Sandboxing:** `ctx.fs` is scoped to the project directory, and network access is restricted to declared hosts (`E_DENIED`).
* **Timeouts & Circuit Breakers:** Execution is bounded by hardware deadlines (`E_TIMEOUT`), and failing services trip circuit breakers to prevent request stalls (`E_CIRCUIT_OPEN`).
* **Error Containment:** Exceptions, rejected promises, or circular references return typed error structures without crashing the host process (`E_TOOL`).

#### CLI Workflow
```bash
sofuu plugin add llamacpp                   # Install official local LLM provider bridge
sofuu plugin init weather --category=tool   # Scaffold new tool in ./plugins/weather.ts
sofuu plugin validate plugins/weather.ts    # Parse, check types, and verify registration
sofuu plugin test weather '{"city":"Oslo"}' # Invoke tool once and print execution envelope
sofuu plugin list                           # View all registered plugins and breaker states
```

Documentation: [sofuu.xyz/docs](https://sofuu.xyz/docs)

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

### 6. Continuous Context Persistence & Local Model Reliability

Running multi-step agents against local inference servers (`llama.cpp`, Ollama, vLLM) with MCP tools often breaks on conventional runtimes due to schema token bloat and KV cache limits. Sofuu provides a native continuous context engine for zero-interruption agent loops:

* **Real KV Cache Discovery:** Discovers active `meta.n_ctx` limits directly from local `/v1/models` endpoints, ensuring prompt budgeting aligns with true server KV cache allocation.
* **90% MCP Schema Slimming:** When context is tight ($\le 16384$ tokens), tool descriptions and parameter schemas are automatically compacted ($\le 130$ and $\le 80$ chars), reducing 24 MCP tools from $\approx 7,500$ tokens to $\approx 720$ tokens without losing argument validation.
* **Small Model Tool Steering:** Directs local models (0.6B–3B) to execute native tool calls rather than reciting markdown descriptions of available functions.
* **Proactive Budgeting & Self-Healing:** Preflight budgeting includes tool overhead and live calibration. If an endpoint breaches its context limit (HTTP 400), Sofuu dynamically caches the ceiling limit, auto-compacts earlier history into a distilled memory gist, and retries seamlessly with zero interruption.

---

### 7. Standalone Compiler & Minimum Size Profiles (`sofuu compile`)

Sofuu can compile any JavaScript, TypeScript, or Kaze (`.kaze`) agent directly into a self-contained single-file executable with **zero external dependencies**. Standalone binaries append bytecode and metadata behind a 32-byte `SOFUU_BIN_V1` trailer with automatic Mach-O / ELF ad-hoc signing and sub-5ms cold starts.

To achieve minimal binary sizes across different deployment targets, Sofuu provides dedicated compiler profiles:

| Profile | Flag | Base Runtime | Target Size | Included Subsystems | Best For |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Micro** | `--profile micro` | `sofuu-micro` | **~2.0–2.5 MB** | Minimal QuickJS + libuv + MCP stream parser. No agent loops, no TUI, no inference. | Single-purpose tools, microservices, MCP servers (`sofuu mcp build`). |
| **Lean** *(Default)* | `--lean` | `sofuu-agent` | **~5.4 MB** | Full QuickJS, libuv, vector memory (`sofuu.memory`), sandboxed tools, policy gates, and agent loop (`sofuu.agent`). Drops local GGUF engine and chat TUI. | Autonomous agents, background daemons, CLI tools. |
| **Full** | `--full` / `--with-inference` | `sofuu` | **~6.8 MB** | Entire engine: local GGUF/SafeTensors CPU inference, MAVLink robotics, JSX transpiler, and full CLI. | Fully self-contained local offline inference systems. |

#### Compiler CLI Usage

```bash
# 1. Compile an agent with the lean profile (default, under 10 MB):
sofuu compile agent.js -o dist/my-agent

# 2. Compile an MCP server with the ultra-lean micro-profile (~2 MB):
sofuu mcp build server.ts -o dist/my-mcp-server
# Or equivalently:
sofuu compile server.ts --profile micro -o dist/my-mcp-server

# 3. Carry on-device vector memory inside the binary trailer:
sofuu compile agent.js --embed-brain ./knowledge.qtsq -o dist/agent-with-brain

# 4. Compile pure declarative Kaze agents:
sofuu compile agent.kaze -o dist/kaze-agent
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

# Model Context Protocol (MCP)
sofuu mcp list                      # List configured MCP servers
sofuu mcp test "npx -y <server>"    # Test handshake and protocol negotiation
sofuu mcp inspect <server> --call <tool> --args '<json>' # Inspect and invoke a tool
sofuu mcp link cursor|claude-desktop # Sync server configuration to IDEs
sofuu mcp build <server.ts> -o <out> # Compile MCP server to standalone native binary (~2 MB)

# Plugins & Extensibility
sofuu plugin list                   # List registered plugins and circuit breaker states
sofuu plugin add <llamacpp|ollama>  # Install provider plugin
sofuu plugin init <name> --category=tool # Scaffold a new plugin
sofuu plugin validate <file.ts>     # Validate schemas and registration
sofuu plugin test <name> '<json>'   # Test run a tool with circuit state reporting

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

> *For complete hardware benchmark receipts across 40+ local models, Apple Silicon Metal GPU kernels, and Raspberry Pi 5 runs, see [BENCHMARKS.md](BENCHMARKS.md).*

---

## Building from Source

Sofuu links the **QTSQ vector memory codec** (`libqtsq.a`) statically as a core runtime dependency. To build from source, compile the QTSQ codec first, then pass its path via `SOFUU_QTSQ_DIR` when building Sofuu:

### 1. Prerequisites
* **Rust & Cargo:** (1.80+) `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* **C/C++ Toolchain:** `clang` or `gcc`, `cmake`, `make`
* **System Libraries:**
  * **macOS:** Xcode Command Line Tools (`xcode-select --install`)
  * **Linux (Ubuntu / Debian / AWS Linux):** `sudo apt-get install -y zlib1g-dev libcurl4-openssl-dev libssl-dev`

### 2. Build the QTSQ Vector Codec
```bash
# Clone QTSQ into your workspace
git clone https://github.com/haruhito2666/qtsq ~/projects/qtsq
cd ~/projects/qtsq

# Build the static archive (libqtsq.a)
make
export SOFUU_QTSQ_DIR=$(pwd)
cd ..
```

### 3. Build Sofuu
```bash
# Clone Sofuu
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu

# 1. Build vendored libuv
make deps

# 2. Build release binary (statically links libqtsq.a)
SOFUU_QTSQ_DIR=$SOFUU_QTSQ_DIR make

# 3. Verify health check and QTSQ memory linkage
./sofuu doctor

# 4. Run test suite
make test
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
