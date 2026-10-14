<p align="center"><img src="sofuu-landing/public/logo.png" width="118" alt="Sofuu logo"></p>

# ⚡ Sofuu (素風)

> *The vLLM of Edge & Mobile — a fast, private JavaScript runtime with in-process CPU/GPU inference, multi-format (GGUF + SafeTensors), agents, and encrypted memory in a single ≤5 MB binary (4.4 MB shipped). Zero daemons, zero external deps.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0--beta-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()

> 🚧 **Active development — iterating fast.** APIs may change before a stable release.

---

## What is Sofuu?

Sofuu is a **server-side JavaScript/TypeScript runtime** — run your AI backend, API server, or edge automation script. No browser API, no DOM, no renderer.

Everything AI workloads need is built into the binary — no npm packages required: LLM streaming, local CPU inference, agents, MCP client/server, HTTP, SIMD vector math, offline embeddings, speech in/out, and an encrypted local brain. It runs **unchanged** on a phone, a Raspberry Pi, a laptop, or a cloud server.

---

## Features

- **JS/TS runtime** — QuickJS (ES2023) + libuv event loop, ~3 ms startup, TypeScript runs with no tsconfig or compiler step, `sofuu bundle` ships one file
- **Local edge inference** — `sofuu infer`, `sofuu.ai.streamLocal()`, `sofuu chat -m <model>`; multi-format (GGUF, SafeTensors, PyTorch `.pt`, NumPy `.npy`/`.npz`), mmap load in 17–38 ms, up to 150 tok/s prefill, up to 68.7 tok/s decode, in-place KV cache compaction & sliding-window attention past context limits
- **Cloud LLM streaming** — `sofuu.ai.stream` / `complete` over OpenAI, Anthropic, or any OpenAI-compatible endpoint (BYO key); per-model context/effort registry, no default model
- **Agents + sub-agents** — `sofuu.agent.define/run`, built-in `code` + `web` tools, budgets, cancel, trace rendering; the same loop powers chat
- **Encrypted local brain** — `sofuu.memory` over `.qtsq` files: recall across sessions, `/remember` pins facts, portable brain cards (`/share` + `/import`), optional HTTP serving
- **Offline embeddings** — `ai.embed` (768-dim, zero-byte model), `ai.embedImage` (joint text/image space), SIMD `similarity`/`dot`/`l2`
- **Speech in/out** — `ai.transcribe` / `ai.speak` via provider endpoints or on-device OS engines
- **Long-context Q&A** — `sofuu.rlm.query` answers over contexts far larger than any model window via a sandboxed reader loop
- **Context-economy gates** — `sofuu.ml` freshness/supervisor/compaction/relevance models (~8k params each) advise the loop to cut token waste; hard off via `/ml off` or `SOFUU_NO_ML=1`
- **MCP + HTTP + web** — `sofuu.mcp.connect/serve`, `sofuu.http.createServer`, keyless `sofuu.web.search/open`
- **Embeddable** — `libsofuu` C ABI with Swift/Kotlin wrappers; QuickJS has no JIT, so iOS App Store policy is satisfied by design

---

## Install

```bash
# 1. curl | sh (no package manager)
curl -fsSL https://sofuu.xyz/install | sh

# 2. npm
npm install -g sofuu      # or: npx sofuu

# 3. Manual
#    → https://sofuu.xyz/downloads  (macOS tarball + .sha256)
```

> **Platform: macOS is the only host with a prebuilt binary today.** Building from source works anywhere (see below).

---

## Quick Start

**Pick a provider on first launch — there is no default model:**

```bash
sofuu        # guided setup asks for provider + model (+ key)
             # later: /provider to switch, /model to change model
```

| endpoint format              | example provider | example `model`     | API key              |
|------------------------------|------------------|---------------------|----------------------|
| OpenAI (`/chat/completions`) | `"openai"`       | `"gpt-5"`           | `OPENAI_API_KEY`     |
| OpenAI (`/chat/completions`) | `"openrouter"`   | `"z-ai/glm-4.6"`    | `OPENROUTER_API_KEY` |
| Anthropic (`/v1/messages`)   | `"anthropic"`    | `"claude-sonnet-4-6"` | `ANTHROPIC_API_KEY` |
| local server (`/api/chat`)   | `"ollama"`       | `"llama3"`          | none                 |

Any other OpenAI-compatible or Anthropic endpoint works too — add it with `/provider`. Settings persist in `~/.sofuu/config.json`.

**Stream an AI response — no packages, no SDK setup:**

```js
// server.js
const server = sofuu.http.createServer(async (req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  const stream = sofuu.ai.stream("Explain how SIMD works.", {
    provider: "openai",
    model: "gpt-4o",   // a model is REQUIRED
  });
  for await (const chunk of stream) {
    res.write(chunk.text);
  }
  res.end();
});
server.listen(3000, "0.0.0.0");
```

```bash
sofuu run server.js
```

**TypeScript just runs:**

```bash
sofuu run hello.ts
```

**Run an agent with web search — still zero packages:**

```js
// research.js
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Search first, then answer with sources.",
  tools: ["code", "web"],
  memory: "agent",
});

const r = await sofuu.agent.run("researcher", "What shipped in QuickJS 2024?");
console.log(r.answer);
```

```bash
sofuu run research.js
```

---

## Local Inference

Sofuu provides a zero-dependency, in-process inference engine supporting **GGUF**, **HuggingFace SafeTensors**, raw **PyTorch checkpoints (`.pt` / `.pth`)**, and **NumPy arrays (`.npy` / `.npz`)**:

```bash
# One-shot generation from a quantized GGUF model:
sofuu infer ./Qwen-3-0.6B-Q4_K_M.gguf "Explain virtual memory paging." --max-tokens 200

# One-shot generation from a HuggingFace SafeTensors directory (LFM-2 BF16 or MLX4):
sofuu infer ./liqiud-350 "Explain quantum entanglement in simple terms." --max-tokens 50

# Extreme long-context with KIVI 2-bit & 4-bit Asymmetric Quantized KV Cache:
sofuu infer ./model.gguf "Summarize this long document" --kivi --kivi-bits 4

# Eliminate CPU cache thrashing via Non-Temporal Streaming Weights (Cache-Bypass):
sofuu infer ./model.gguf "Explain distributed consensus" --streaming-weights

# Accelerate decode throughput via PowerInfer/DejaVu Dynamic FFN Sparsity:
sofuu infer ./model.gguf "Explain general relativity" --dynamic-ffn

# Generate past context limits with in-place KV compaction & sliding-window attention:
sofuu infer ./model.gguf "Write an essay" --ctx 256 --swa 128 --max-tokens 1000

# Interactive chat powered by any local model:
sofuu chat -m ./Qwen-3-0.6B-Q4_K_M.gguf
```

```javascript
// In-process streaming in JS/TS — no network calls, no background servers:
const stream = sofuu.ai.streamLocal("Explain vector quantization.", {
  model: "./Qwen-3-0.6B-Q4_K_M.gguf",  // or "./liqiud-350"
  temperature: 0.7,
  maxTokens: 500,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

### Verified Hardware Benchmarks

| Hardware Target | Model & Precision | Cold Load | Prompt Prefill | Autoregressive Decode | Notes |
|---|---|---|---|---|---|
| **Apple Silicon Mac** | `Llama-3.2-1B-Instruct` (Q8_0 GGUF) | **12.5 ms** | **42.3 tk/s** | **26.6 tk/s** | **+168.7% speedup** with `--kivi` 4-bit KV cache (vs 9.9 tk/s baseline) |
| **Apple Silicon Mac** | `Llama-3.2-1B-Instruct` (Q8_0 GGUF) | **600 ms** | **14.6 tk/s** | **5.3–6.9 tk/s** | Cache-bypass with `--streaming-weights` (vs 3.47s cold load) |
| **Apple Silicon Mac** | `Spark-X2.5-4B` (Q4_K_M GGUF) | **18.2 ms** | **5.2 tk/s** | **17.0 tk/s** | **+68.3% speedup** with `--dynamic-ffn` (vs 10.1 tk/s dense) |
| **Apple Silicon Mac** (M2 Pro) | `liqiud-350` (BF16 SafeTensors) | **127.9 ms** | — | **68.7 tk/s** | 19.1× speedup via NEON bit-extension |
| **Apple Silicon Mac** (M2 Pro) | `liquid` (MLX 4-bit SafeTensors) | **119.2 ms** | — | **8.1 tk/s** | 20.25× speedup via group scale/bias broadcast |
| **Apple Silicon Mac** (M2 Pro) | `Qwen-3-0.6B` (Q4_K_M GGUF) | **17.7 ms** | **138–150 tk/s** | **30.0–33.5 tk/s** | 4-way unrolled vector accumulators |
| **Raspberry Pi 5** (ARM64 musl) | `Qwen-3-0.6B` (Q4_K_M GGUF) | **38.15 ms** | **49.3–49.8 tk/s** | **15.6–15.7 tk/s** | Zero futex storms, core-pinned, LPDDR4X prefetch |

---

## CLI Reference

```bash
sofuu                        # Interactive AI chat
sofuu chat [-m <model.gguf>] # Chat (cloud API or local model)
sofuu infer <model> "<p>"    # Pure CPU LLM inference (GGUF, zero deps)
sofuu repl                   # JavaScript evaluation REPL
sofuu run <file.js|ts>       # Run a JavaScript/TypeScript file
sofuu eval "<code>"          # Evaluate a JS expression inline
sofuu bundle <entry.js> [-o <out.js>]  # Bundle into one file
sofuu install / add <package>          # npm packages
sofuu agent list             # List agent definitions (~/.sofuu/agents/*.js)
sofuu agent run <name> "task" [--json] # Run one agent headless
sofuu serve --brain <path>   # Serve the brain over HTTP
sofuu doctor                 # Health check: QTSQ link, brain round-trip, ctx
sofuu version / help
```

**Chat slash commands** (`sofuu` / `sofuu chat`):

```text
/help  /model <name>  /provider <name>  /effort <low|medium|high|max>
/compact  /clear  /brain [on|off]  /rlm [on|off|auto]
/tools  /agents  /version  /exit

/remember <fact>  /why  /share [path]  /import <path>  /cost  /ghost [on|off]
@file[:start-end]  @name <task>  /watch <path>  /hooks  /resume [id]  /serve
/ml [on|off|info]  /sessions  /context [id]  /ctx [<tokens>]  /maxout [<tokens>]
/work <desc>  /done  /note <msg>  /notify <msg>  /sync [on|off]
```

`sofuu doctor` verifies the binary is healthy — QTSQ linked, brain write→read round-trip passes, and the current model's context window resolves. A non-zero exit means a hard dependency is missing.

---

## API Reference

### `sofuu.ai` — LLM Streaming, Vectors, Embeddings, Speech

```js
const stream = sofuu.ai.stream("What is the speed of light?", {
  provider: "openai", model: "gpt-5",   // model is REQUIRED
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}

const result = await sofuu.ai.complete("What is 2 + 2?", { provider: "openai", model: "gpt-4o" });
result.text;    // "4"
result.tokens;  // { input: 12, output: 1 }

sofuu.ai.similarity(a, b)   // Cosine similarity → [-1, 1]
sofuu.ai.dot(a, b)          // Dot product
sofuu.ai.l2(a, b)           // Euclidean distance

const v = sofuu.ai.embedLocal("hello world");              // 768-dim, offline, no model file
const vs = await sofuu.ai.embed(["a", "b"], { space: "hash-768" });
const batch = await sofuu.ai.embedBatch(["a", "b", "c"]);  // one call
const e = await sofuu.ai.embed("hi", { provider: "openai", model: "text-embedding-3-small" }); // BYO key
const iv = sofuu.ai.embedImage(await sofuu.fs.readFileBytes("shot.png"));  // image → joint space

const t = await sofuu.ai.transcribe(audioBytes, { provider: "openai", model: "whisper-1" });
const s = await sofuu.ai.speak("hello", { provider: "openai", model: "tts-1", voice: "alloy" });
```

### `sofuu.agent` — Agents & Sub-Agents

```js
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Cite sources.",
  tools: ["web"],
  memory: "agent",   // "shared" | "agent" | "off"
  budget: { maxSteps: 12, maxDepth: 2, maxWallMs: 300_000 },
  agents: ["librarian"],   // sub-agents reachable via `delegate`
});

const r = await sofuu.agent.run("researcher", "compare Rust and Zig GC stories");
r.answer; r.subRuns; r.usage; r.trace; r.stopped;

await sofuu.agent.runMany([{ agent: "researcher", task: "…" }], { concurrency: 4 });
sofuu.agent.cancel(runIdOrSignal);
sofuu.agent.renderTrace(r);   // human-readable run tree
```

Built-in `code` tools: `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `list_dir`, `bash` (jailed to the project dir; `SOFUU_NO_SHELL=1` strips `bash`).

### `sofuu.rlm` — Long-Context Q&A

```js
const res = await sofuu.rlm.query(bigText, "What error caused the outage?", {
  provider: "openai", model: "gpt-4o",
  maxRounds: 16, maxLlmCalls: 24, maxWallMs: 120000, trace: true,
});
res.answer; res.calls; res.rounds; res.stopped; res.trace;
sofuu.rlm.route(ctxTokens, windowTokens, question)  // → "rlm" | "plain"
```

### `sofuu.ml` — Context-Economy Gates

Tiny on-device models (~8k params each, offline, advise-only — the LLM always decides):

```js
JSON.parse(sofuu.ml.freshness.score(text, task, JSON.stringify({ kind: "web" })));
JSON.parse(sofuu.ml.supervisor.check(JSON.stringify({ run, step, tool, task, budget: 20 })));
JSON.parse(sofuu.ml.compaction.plan(JSON.stringify({ task, summary, recent, segments }), JSON.stringify({ budget: 1200 })));
JSON.parse(sofuu.ml.relevance.plan(JSON.stringify({ task, recent, candidates }), JSON.stringify({ kept: [0] })));
sofuu.ml.info();
```

### `sofuu.web` — Web Search & Page Reading (keyless default)

```js
const r = await sofuu.web.search("rust vs zig 2026", { count: 8 });
r.results;   // [{ title, url, snippet }]
const page = await sofuu.web.open("https://example.com/article");
page.text;
```

### `sofuu.mcp` — Model Context Protocol

```js
const client = await sofuu.mcp.connect("npx @modelcontextprotocol/server-filesystem /tmp");
await client.listTools();
await client.call("read_file", { path: "/tmp/notes.txt" });
client.disconnect();

const server = sofuu.mcp.serve();
server.tool("get_weather", { description: "…", schema: { … } }, async ({ city }) => { … });
server.start();   // stdio — works with any MCP client
```

### `sofuu.http` · `sofuu.fs` · `sofuu.spawn` · Globals

```js
const server = sofuu.http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok" }));
});
server.listen(8080, "0.0.0.0");
const res = await fetch("https://api.example.com/data");

const text = await sofuu.fs.readFile("./data.json", "utf8");
await sofuu.fs.writeFile("./output.txt", "Hello from Sofuu!");

setTimeout(() => console.log("done"), 1000);
await sleep(500);
process.env.API_KEY; process.argv; process.cwd();
```

---

## Embed Sofuu (libsofuu)

```swift
// Package.swift — or: pod 'Sofuu', '~> 0.2'
import Sofuu

let sofuu = try Sofuu()
let vec = try sofuu.embed("the sky is blue")     // 768-dim, offline, 0-byte model
let brain = try Brain.openDefault(sofuu: sofuu)  // encrypted, local, yours
try brain.remember("the sky is blue")
let hits = try brain.recall("what colour is the sky?", k: 3)
```

```kotlin
// Gradle: implementation 'com.sofuu:sofuu-android:0.2.0'
val vec = sofuu.embedLocal("hello")   // FloatArray, 768-dim, no network
```

```c
#include "sofuu_embed.h"
SofuuRuntime *rt = sofuu_rt_new(NULL);
char *out = NULL;
sofuu_rt_call(rt, "memory.open", "{\"path\":\"/tmp/b.qtsq\",\"dim\":768}", &out);
sofuu_free(out);
sofuu_rt_free(rt);
```

```bash
npx sofuu            # chat
npx sofuu run app.ts
```

From source: `make libsofuu`, `make dist-macos`, `make dist-ios` (xcframework), `make dist-android` (per-ABI .so).

---

## Build from Source

```bash
# Requirements: Rust (cargo), clang, make. libcurl for HTTP features.
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
export SOFUU_QTSQ_DIR=/path/to/black-hole-disk   # the encrypted-brain codec
make                 # cargo build --release
make test            # JS e2e suite
make size-check      # fails if the binary exceeds 5MB
./sofuu doctor       # confirms QTSQ linkage + a real brain round-trip
```

`make` refuses to build without the QTSQ codec (pass `SOFUU_ALLOW_NO_QTSQ=1` for a memory-less binary) — a codec-less build silently no-ops memory, so it is opt-out, not opt-in.

Key gates: `cargo test --release` · `make test` · `make headless-test` (all 5 C samples assert) · `make abi-check` · `make size-check` · `./sofuu doctor`.

---

## Open core

The runtime is **fully open source (MIT)** — engine, CLI, JS/TS APIs, agents, MCP, HTTP, vector ops, memory, embeddings, SDK bindings. Use, modify, and distribute it freely, commercially or not.

Proprietary are only: the **QTSQ format specification** ([terms](LICENSES/QTSQ-FORMAT.txt) — using it *through Sofuu*, commercially or not, is free and needs no approval) and **future hosted/cloud/enterprise services** as separate offerings. Nothing is removed from the core to create them.

---

## Project and founder

Built by **Haruhito** — early-stage, bootstrapped, open-core, in active development.

- Website: [sofuu.xyz](https://sofuu.xyz)
- Source: [sofuu-runtime/sofuu](https://github.com/sofuu-runtime/sofuu)
- Contact: [hello@sofuu.xyz](mailto:hello@sofuu.xyz)

---

## License

MIT — see [LICENSE](LICENSE). The QTSQ format spec stays proprietary per [LICENSES/QTSQ-FORMAT.txt](LICENSES/QTSQ-FORMAT.txt).

© 2026 Haruhito
