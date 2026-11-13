<p align="center"><img src="https://sofuu.xyz/logo.png" width="118" alt="Sofuu logo"></p>

# Sofuu (素風)

> *An early-stage native runtime for AI applications and agents. Written in Rust, QuickJS, and libuv, Sofuu includes JS/TS agent loops, Kaze scripting, QTSQ vector memory, and Model Context Protocol (MCP) support. Some workflows require model files, external services, or system libraries.*

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

Sofuu brings agent loops, tools, local vector memory, and MCP into a native runtime for AI applications. Selected local workflows can run in one process; cloud models, remote tools, MCP servers, and some inference providers require external services.

The project is beta. Features and evidence vary by platform and execution path:

* **Native runtime:** QuickJS runs without a JIT. See [BENCHMARKS.md](BENCHMARKS.md) for measured startup conditions, not a universal latency bound.
* **On-Device Vector Memory (QTSQ):** Local vector storage is encrypted by default; plaintext mode and unencrypted exports are available. `embedLocal` uses a 768-dimensional trigram hash, not a trained semantic model.
* **Agents & Kaze:** JS/TS agent loops and domain toolpacks; Kaze routines and standalone compilation. The real Kaze model-and-tool execution path remains an open milestone.
* **Model Context Protocol (MCP):** Client and server support for selected protocol versions and transports; see the limitations below.
* **Policy and sandbox tools:** Selected tool, filesystem, network, and shell paths have policy checks; boundaries and approval handling depend on the execution surface.
* **Embedding:** A C ABI and mobile wrappers are available; support and testing vary by platform. Hardware and app-store acceptance are not universally established.

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

The following JavaScript example creates local vector memory, calls a native tool, and defines an agent. It does not invoke a model:

```javascript
// quickstart.js - memory · native tools · agents · zero daemons
const brain = sofuu.memory.open("./brain.qtsq", 768);
brain.remember(sofuu.ai.embedLocal("runtime"), "AI runtime. Under 10 MB.");

// Recall from local memory using the same embedding space
const [hit] = brain.recall(sofuu.ai.embedLocal("speed"), 1);

// Native tool executed in-process
const sys = sofuu.tools.execute("system_info", {});

// Define an agent; invoking it requires a configured model provider
const agent = sofuu.agent.define({
  name: "assistant",
  system: "Cite sources.",
  tools: ["coding", "memory", "web"],
});

console.log(hit.text, "•", sys.platform || process.platform, "• agent:", agent.name);
// -> AI runtime. Under 10 MB. · darwin · agent: assistant
```

*This example does not invoke an agent or model. Local memory does not require a database server.*

Run it immediately with zero setup, or compile it into a standalone native executable:

```bash
# 1. Run directly
$ sofuu run quickstart.js

# 2. Compile into a native executable
$ sofuu compile quickstart.js -o my-agent
$ ./my-agent
```

> Local memory and selected local tools can work offline. Model-backed agents need a supported local model/backend or a configured remote provider; remote services need network access.

---

### 3. Instant Terminal Testing (No Files Needed)

You can also test Sofuu's capabilities directly from your terminal:

```bash
# Launch interactive AI chat with persistent session memory
$ sofuu chat

# One-liner vector memory test
$ sofuu eval 'const b = sofuu.memory.open("./brain.qtsq", 768); b.remember(sofuu.ai.embedLocal("fast"), "Local vector memory"); console.log(b.recall(sofuu.ai.embedLocal("speed"), 1)[0].text);'
```

---

### 4. Kaze: a domain-specific scripting language

Kaze provides a parser, checker, VM, capability declarations and standalone compilation for bounded agent and device routines. It is not a general-purpose language. A script or compiled agent that declares AI or tool needs runs through an asynchronous bridge that routes AI calls through Sofuu's AI API and tool calls through its built-in tools or MCP client under the applicable policies. **A real Kaze model-and-authorized-tool journey has been exercised on a debug build with a configured remote model and a real MCP server, through `sofuu.kaze.runAsync`, a compiled agent, and `sofuu script run` for source and bytecode.** A script that declares neither capability keeps the engine-free synchronous runner and refuses unavailable AI and tool backends. Current source also includes an explicitly selected, per-run in-memory CMA backend; persistent Kaze memory and alert, sensor and device operations remain unavailable. Physical simulation is explicitly selected for drone or rover testing. Released binaries, local-GGUF weights and live hardware still need separate verification. Do not use the examples for real-world actions or treat their output as evidence of verified live integrations.

See [embedded Kaze capability status](KAZE-EMBEDDED.md) for policy, backend and verification details.

The [Kaze examples](example-agents/README.md) can be inspected and checked without executing their capabilities:

```bash
sofuu script check example-agents/quant_sentinel/agent.kaze
```

Compilation packages supported routines as one executable; system libraries and any real backends needed by the workload remain separate.

## Platform evidence (2026-11-10)

| Platform | Available evidence | Limits |
| :--- | :--- | :--- |
| macOS arm64 | Published CLI archive and local build; startup and size measurements in [BENCHMARKS.md](BENCHMARKS.md). | An inspected compiled executable dynamically links system libraries. Measurements apply only to the recorded workload. |
| Linux arm64 and x86_64 | Published CLI archives and local builds are listed in [BENCHMARKS.md](BENCHMARKS.md). | Published Linux artifacts exceed the project's size cap; re-release and platform-specific verification are pending. |
| Android | Android source and an SDK package are described at [sofuu.xyz/embed](https://sofuu.xyz/embed). | Capability coverage varies by device and build; verify the artifact and device before use. |
| iOS / macOS SDK | C ABI and Swift wrapper sources exist. | SwiftPM/CocoaPods coordinates at the cited 0.3.0-beta tag were not installable in the 2026-10-31 site audit; verify a current release before using them. No app-store approval is claimed. |
| Watches, rings, RTOS, other smart devices | Future targets. | No device, OS, or app-store acceptance evidence established in this audit. |

## Core Runtime vs. Experimental Preview

Sofuu groups the main runtime surfaces and experimental modules below. Core means included in the product; it does not imply production readiness or a security guarantee.

| Subsystem | Status | Description |
| :--- | :---: | :--- |
| **Autonomous Agent Engine** | **Core** | `sofuu.agent`, cooperative scheduling (`scheduler.yield`), step budgets, resilient editing (`smart_edit`), and real-time trace streaming. |
| **Domain Toolpacks** | **Core** | Modular toolpacks (`coding`, `web`, `memory`, `system`), single-instruction fuel sandbox, numbered unified diffs. |
| **Standalone Compiler** | **Core** | `sofuu compile`: native executables with a 32-byte `SOFUU_BIN_V1` trailer; signing and dependencies vary by platform. |
| **Vector Memory (QTSQ)** | **Core** | On-device `.qtsq` vector containers, cached KDF v2, plaintext zero-KDF mode, hardware SIMD search. |
| **Model Context Protocol (MCP)** | **Core** | Native client & server supporting stdio, HTTP, SSE, and direct in-process transports; dynamic tool discovery. |
| **Kaze Agent Language** | **Core** | Domain-specific scripting language and VM; real model and tool integration remains incomplete. |
| **Security Sandbox** | **Core** | Deterministic path traversal blocking, private CIDR SSRF filtering, credential shielding (`.env`, `id_rsa`). |
| **Local Inference Engine** | **Preview** | Built-in custom GGUF/SafeTensors engine. (For deployed local inference, use `sofuu-plugin-llamacpp` or remote providers). |
| **Robotics & MAVLink** | **Preview** | `sofuu drone`, MAVLink v2 codec, and GPIO line timers through hardware safety gates. |
| **Native Frontend Engine** | **Preview** | `sofuu dev` streaming JSX/TSX transpiler and reactive micro-runtime. |

---

## Core Capabilities

### 1. Model Context Protocol (MCP)

Sofuu provides native MCP client and server support for protocol versions `2025-06-18` and `2024-11-05`. Client transports include stdio, Streamable HTTP, legacy SSE, and in-process direct transport. Server behavior varies by transport; stdio resource and prompt handlers must be synchronous. External servers have their own requirements: the `npx` example below needs Node.js, npx, and access to the package.

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
* **Current tool limitations:** A lost MCP response or deadline after a potentially effectful call is reported as `E_MCP_OUTCOME_UNKNOWN`; inspect the operation before retrying. Built-in and MCP calls requiring `approval.mode: "always"` currently fail with `E_POLICY_APPROVAL_REQUIRED`, even after an agent prompt, until approval receipts are propagated to the dispatch boundaries.

#### MCP Server
Build standalone or in-process MCP servers in TypeScript or JavaScript:

```javascript
const server = sofuu.mcp.serve({ name: "math-service", version: "1.0.0" });
const fib = n => {
  let a = 0, b = 1;
  for (let i = 0; i < n; i++) [a, b] = [b, a + b];
  return a;
};

// Register tools with schema validation
server.tool("fibonacci", {
  description: "Calculate the nth Fibonacci number",
  schema: { type: "object", properties: { n: { type: "integer", minimum: 0, maximum: 78 } }, required: ["n"] }
}, async ({ n }) => {
  return { result: fib(n) };
});

// Register resources and prompts
server.resource("system://platform", { mimeType: "application/json" }, () => {
  return JSON.stringify({ platform: process.platform });
});

// Composable onion middleware
server.use(async (ctx, next) => {
  console.error(`[MCP] Invoking ${ctx.toolName}`);
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
* **Deployment:** The compiled executable does not need a separate Node.js or Python installation for supported code. It may still depend on OS libraries and any external servers or services it calls.

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
Plugin tools run in-process and must be trusted as code; their helpers provide checks but are not an OS-level sandbox:
* **Input & Output Validation:** Arguments are validated against the schema before handler execution (`E_SCHEMA`).
* **Guarded helpers:** `ctx.fs` scopes supported operations to the project directory, and `ctx.fetch` checks declared network hosts (`E_DENIED`).
* **Timeouts & Circuit Breakers:** Watchdogs report timeouts and signal cancellation; repeated failures can open a circuit. In-process code may not cooperate with cancellation.
* **Error Handling:** The wrapper converts supported exceptions and rejected promises to typed results (`E_TOOL`); it cannot contain every crash or resource exhaustion.

The plugin tool wrapper rejects missing, invalid, or failed policy evaluation.
`sofuu.plugins.invoke` handles policy approval requests; callers using a tool's
`execute` method directly are responsible for handling an `ask` decision.

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

### 5. Agent Observability & Recovery

Built-in append-only JSONL tracing designed for autonomous supervisor loops:

* **Append-Only Traces:** Turns and tool steps append to `.sofuu/logs/agent-<run_id>.jsonl` with total-order sequence numbers.
* **Secret Redaction:** Recognized credential and path patterns are redacted in supported diagnostic paths; coverage is not exhaustive.
* **Crash Post-Mortems:** Diagnostic cards may be written to `.sofuu/crashes/<crash_id>.json` on a best-effort basis.
* **Resumable Execution:** `sofuu resume <run_id>` requires a usable checkpoint; redacted or truncated checkpoints can be refused.

```bash
sofuu logs                     # List past agent runs
sofuu logs show <run_id>       # Render colored, step-by-step transcript
sofuu logs --last-error        # View post-mortem card for the latest failure
sofuu resume <run_id>          # Resume an interrupted run from its last checkpoint
```

---

### 6. Continuous Context Persistence & Local Model Reliability

Running multi-step agents against local inference servers (`llama.cpp`, Ollama, vLLM) with MCP tools often breaks on conventional runtimes due to schema token bloat and KV cache limits. Sofuu includes context-management techniques that may help within a model's limits; neither retention nor uninterrupted execution is guaranteed:

* **Real KV Cache Discovery:** Discovers active `meta.n_ctx` limits directly from local `/v1/models` endpoints, ensuring prompt budgeting aligns with true server KV cache allocation.
* **MCP Schema Slimming:** Under tight context budgets, tool descriptions and parameter schemas can be compacted; savings vary with the tool set and model.
* **Small Model Tool Steering:** Attempts to help supported local models produce tool calls; results depend on the model and task.
* **Budgeting & Retry:** Preflight budgeting includes tool overhead; some context-limit errors can trigger compaction and a retry. A retry can still fail or lose context.

---

### 7. Standalone Compiler & Minimum Size Profiles (`sofuu compile`)

Sofuu can compile supported JavaScript, TypeScript, and Kaze (`.kaze`) programs into a single executable with a 32-byte `SOFUU_BIN_V1` trailer. A separate Node.js or Python installation is not needed for supported code; OS libraries and workflow-specific models, servers, or providers may still be required.

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
sofuu daemon install|run       # Manage a supervisor daemon

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
| **Binary Size** | Varies by artifact | Published macOS arm64: 5.39 MiB; published Linux arm64: 5.88 MiB, x86_64: 6.80 MiB. See [BENCHMARKS.md](BENCHMARKS.md). |
| **Startup** | **14.58 ms p50** | `sofuu eval "1+1"`, 500 runs, macOS arm64, warm page cache, including process spawn, 2026-11-06; see [conditions and revisions](BENCHMARKS.md). |
| **Encrypted first flush** | **79.84 MB peak RSS** | One record, 64-dimensional store, fresh process, macOS arm64, 2026-11-05; see [BENCHMARKS.md](BENCHMARKS.md). |
| **Vector Memory Query** | Varies by device | See the 10,000-vector results and their limitations in [BENCHMARKS.md](BENCHMARKS.md). |

> *For benchmark conditions, historical measurements, and limitations, see [BENCHMARKS.md](BENCHMARKS.md).*

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
