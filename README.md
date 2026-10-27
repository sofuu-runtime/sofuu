<p align="center"><img src="docs/sofuu-logo.png" width="118" alt="Sofuu logo"></p>

# ⚡ Sofuu (素風)

> *The AI-Specific General-Purpose Runtime — a fast, private JavaScript/TypeScript execution environment with in-process CPU/GPU inference, multi-format (GGUF + SafeTensors), autonomous agents, and cognitive vector memory (CMA) in a single < 5 MB binary (4.8 MB stripped release). Zero daemons, zero external deps.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0--beta-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()

> 🚧 **Active development — iterating fast.** APIs may change before a stable release.

---

## What is Sofuu?

Sofuu is **the AI-specific general-purpose JavaScript/TypeScript runtime** — the unified execution environment for the AI era.

Sofuu is **not just a terminal coding agent or a local model runner**; the terminal assistant (`sofuu chat`) is merely one first-party demonstration application running on the engine. Sofuu is the foundational operating substrate built to collapse the fragmented AI stack (Python/Node + LangChain + Chroma + Ollama + FastAPI) into a single in-process runtime.

Everything AI workloads need is built directly into the single native binary — no npm packages, no Python environments, and no background daemons required:
- **In-process local LLM inference** (GGUF, SafeTensors, PyTorch, NumPy on Metal / NEON / AVX2) and cloud streaming
- **Autonomous agent loops** with native tools and sandboxed execution
- **Continuous Memory Architecture (CMA v2)** — on-device encrypted vector brain in `.qtsq` containers
- **Model Context Protocol (MCP)** — first-class client and server support
- **Unified R7 Security Policy** — deterministic firewall for tools, shell, filesystem, network, and MCP
- **Async HTTP server & SIMD vector mathematics**

Sofuu runs **identically across surfaces**: as an embeddable mobile SDK (`libsofuu` for iOS/Android), an edge agent on a Raspberry Pi, an API microservice in the cloud, or an interactive terminal workspace on your laptop.

---

## Features

- **JS/TS runtime** — QuickJS (ES2023) + libuv event loop with cooperative scheduler (W3C `scheduler.yield()`, microtask starvation protection), AOT bytecode snapshotting, zero-copy TypedArray SIMD memory bridge, native `AsyncLocalStorage` context propagation, deterministic instruction-metered sandboxing, ~3 ms startup, TypeScript runs with no compiler step, and `sofuu bundle` ships one self-contained file
- **Local edge inference** — `sofuu infer`, `sofuu.ai.streamLocal()`, `sofuu chat -m <model>`; multi-format (GGUF, SafeTensors, PyTorch `.pt`, NumPy `.npy`/`.npz`), up to ~195 tok/s prefill and ~69 tok/s decode on this battery's hardware, in-place KV cache compaction & sliding-window attention past context limits
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

# Multi-Path Lookahead Tree Speculation (evaluate candidate token trees via single DRAM weight sweep):
sofuu infer ./model.gguf "def fibonacci(n):" --lookahead

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
| **Apple Silicon Mac** | `Qwen-3-0.6B` (Q4_K_M GGUF) | **111.3 ms** | **157.1 tk/s** | **51.5–56.5 tk/s** | **+22.6% to +34.5% speedup** with `--lookahead` (vs 42.0 tk/s baseline) |
| **Apple Silicon Mac** | `Llama-3.2-1B-Instruct` (Q8_0 GGUF) | **12.5 ms** | **42.3 tk/s** | **26.6 tk/s** | **+168.7% speedup** with `--kivi` 4-bit KV cache (vs 9.9 tk/s baseline) |
| **Apple Silicon Mac** | `Llama-3.2-1B-Instruct` (Q8_0 GGUF) | **600 ms** | **14.6 tk/s** | **5.3–6.9 tk/s** | Cache-bypass with `--streaming-weights` (vs 3.47s cold load) |
| **Apple Silicon Mac** | `Spark-X2.5-4B` (Q4_K_M GGUF) | **18.2 ms** | **5.2 tk/s** | **17.0 tk/s** | **+68.3% speedup** with `--dynamic-ffn` (vs 10.1 tk/s dense) |
| **Apple Silicon Mac** (M2 Pro) | `liqiud-350` (BF16 SafeTensors) | **127.9 ms** | — | **68.7 tk/s** | 19.1× speedup via NEON bit-extension |
| **Apple Silicon Mac** (M2 Pro) | `liquid` (MLX 4-bit SafeTensors) | **119.2 ms** | — | **8.1 tk/s** | 20.25× speedup via group scale/bias broadcast |
| **Apple Silicon Mac** (M2 Pro) | `Qwen-3-0.6B` (Q4_K_M GGUF) | **17.7 ms** | **138–150 tk/s** | **30.0–33.5 tk/s** | 4-way unrolled vector accumulators |
| **Raspberry Pi 5** (ARM64 musl) | `Qwen-3-0.6B` (Q4_K_M GGUF) | **880 ms** | **195–196 tk/s** | **26.0–26.1 tk/s** | Ahead of llama.cpp-11041 CPU on the same prompt (+11% prefill / +9% decode); q4_K tile GEMM |
| **Raspberry Pi 5** (ARM64 musl) | `Qwen-3-0.6B` (Q3_K_S GGUF) | **560 ms** | **145 tk/s** | **30.3 tk/s** | ~2× llama.cpp prefill, decode +7% (cost-aware speculative gate, serial after warmup) |
| **Raspberry Pi 5** (ARM64 musl) | `Llama-3.2-1B-Instruct` (Q8_0 GGUF) | **1.1 s** | **141–142 tk/s** | **10.1 tk/s** | Prefill 2.4× llama.cpp, decode +15%; speculative tree on longer generations (20+ tk/s) |
| **Raspberry Pi 5** (ARM64 musl) | `Sharp-MiniCPM5-2B` (Q6_K_XL GGUF) | **1.4 s** | **38.4–39.2 tk/s** | **5.8–5.9 tk/s** | Ahead of llama.cpp (+9% prefill / +6% decode, sampled — its greedy first token is EOS on the raw prompt) |
| **Raspberry Pi 5** (ARM64 musl) | `Spark-X2.5-4B` (Q4_K_M GGUF) | **2.1 s** | **25.3–25.4 tk/s** | **4.4 tk/s** | Ahead of llama.cpp (+11% prefill / +5% decode), matched prompt & context |
| **Raspberry Pi 5** (ARM64 musl) | `Ternary-Bonsai-2-9B` (PQ2_0 GGUF) | **440 ms** | **1.4 tk/s** | **0.7 tk/s** | Loadable only here (ternary-weight format); v1 dequant path, not yet optimized |

Pi 5 receipts: 2026-10-26 battery, current main, 4× Cortex-A76 with NEON+dotprod at the adopted `arm_freq=2800` + 25 mV boot config, `performance` governor, 4 threads, temp 0, 234–252-token prompt, fresh llama.cpp build 11041 references from the same session. Load times are warm page-cache.

---

## CLI Reference

```bash
sofuu                        # Interactive AI chat
sofuu chat [-m <model.gguf>] # Chat (cloud API or local model)
sofuu dev [dir]              # AI-native full-stack dev server (--port, --host, --model)
sofuu infer <model> "<p>"    # Pure CPU LLM inference (GGUF, zero deps)
sofuu repl                   # JavaScript evaluation REPL
sofuu run <file.js|ts>       # Run a JavaScript/TypeScript file
sofuu eval "<code>"          # Evaluate a JS expression inline
sofuu bundle <entry.js> [-o <out.js>]  # Bundle into one file
sofuu install / add <package>          # npm packages
sofuu agent list             # List agent definitions (~/.sofuu/agents/*.js)
sofuu agent run <name> "task" [--json] # Run one agent headless
sofuu serve --brain <path>   # OpenAI-compatible HTTP server (--model, --model-id, --host, --port)
sofuu doctor                 # Health check: QTSQ link, brain round-trip, ctx
sofuu version / help
```

---

## Native Frontend & Full-Stack Engine (`sofuu dev`)

Sofuu embeds a streaming JSX/TSX transpiler, file-based router, and React-compatible reactive micro-runtime directly into the native binary. Develop full-stack reactive web applications with zero npm dependencies and zero background daemons.

```bash
sofuu dev [dir] --port 3000 --host 127.0.0.1
```

* **Zero-AST On-the-Fly Transpilation**: Converts JSX and strips TypeScript types in RAM with sub-millisecond latency.
* **Drop-In React Compatibility**: Write standard React code (`import React, { useState, useEffect } from "react"`, `import ReactDOM from "react-dom"`) with zero `npm install`.
* **Complete Primitives Suite (38 primitives)**:
  - **Hooks & Lifecycle**: `useState`, `useReducer`, `useEffect`, `useLayoutEffect`, `useCallback`, `useMemo`, `useRef`, `useId`, `createContext`, `useContext`, `forwardRef`, `memo`, `useSyncExternalStore`, `useTransition`, `useDeferredValue`, `createPortal`, `Children`.
  - **Client-Side SPA Routing (`sofuu/router`)**: `<Link>`, `useRouter`, `usePathname`, `useSearchParams`.
  - **Data Fetching & Cache**: `useQuery`, `useMutation` with automatic SWR revalidation and caching.
  - **Global Reactive Store (`sofuu/store`)**: `createStore`, `useStore` without external state libraries.
  - **AI UI Streaming (`sofuu/ai`)**: `useChat()`, `useCompletion()`, `<ThinkingBlock />`, `<Markdown />`, `<CodeBlock />`.
* **Zero-Dependency Live Reload**: Built-in file watcher streams instant reloads on save via `/_sofuu/live` SSE.


**Chat slash commands** (`sofuu` / `sofuu chat`):

```text
/help  /model <name>  /provider <name>  /effort <low|medium|high|max>
/compact  /clear  /brain [on|off|info]  /rlm [on|off|auto]
/tools  /agents  /version  /exit

/remember <fact>  /why  /share [path]  /import <path>  /cost  /ghost [on|off]
@file[:start-end]  @name <task>  /watch <path>  /hooks  /resume [id]  /serve
/ml [on|off|info]  /sessions  /context [id]  /ctx [<tokens>]  /maxout [<tokens>]
/work <desc>  /done  /note <msg>  /notify <msg>  /sync [on|off]
```

`sofuu doctor` verifies the binary is healthy — QTSQ linked, brain write→read round-trip passes, and the current model's context window resolves. On a QTSQ build it also inspects the real brain store: vector space, segment manifest, per-tier histogram, capacity headroom, journal state, and last compaction. A non-zero exit means a hard dependency is missing.

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

### `sofuu serve` — OpenAI-compatible HTTP server

```bash
export SOFUU_SERVE_TOKEN=$(openssl rand -hex 24)  # --token works, but `ps` can see it
sofuu serve --model ./qwen3-0.6b-q4_k_m.gguf --model-id qwen3-0.6b \
            --host 127.0.0.1 --port 7707 --brain ./mem.sofuu
```

Every route except `GET /health` requires `Authorization: Bearer $SOFUU_SERVE_TOKEN`
(constant-time compare). `policy.http.auth: "none"` opts out explicitly and logs
a startup warning; `policy.http.deny` path prefixes are evaluated per request.

| Route | What it does |
| --- | --- |
| `GET /health` | liveness only, no auth, no data |
| `GET /v1/models` | configured ids (`--model-id`, default: the model file name) |
| `POST /v1/chat/completions` | OpenAI chat. `stream: true` → SSE `chat.completion.chunk` events + `data: [DONE]`; otherwise one JSON `chat.completion`. Honors `messages`, `max_tokens`, `temperature`, `stop` (up to 4), `tools`, and `stream_options.include_usage` |
| `POST /v1/completions` · `/infer` | raw text completion, no chat template |
| `POST /v1/embeddings` | OpenAI embeddings: 1–64 string inputs, 8192-token cap per input |
| `POST /v1/sessions` · `GET /v1/sessions` · `GET /v1/sessions/:id/turns` | process-local sessions and their recorded turns |
| `POST /v1/agent/runs` | `sofuu.agent.run`; SSE events when `stream: true`, else JSON |
| `GET /v1/tools` · `POST /v1/tools/:name/call` | tool inventory, then policy-gated execution |
| `POST /remember` · `GET /recall?q=&k=` · `GET /memories` · `POST /forget` · `POST /share` | brain-backed memory (requires `--brain`) |

Errors use the OpenAI envelope `{error:{message,type,param,code}}`: `401`
`invalid_api_key`, `404` `model_not_found`, and policy denials are `403` with
`code: "policy_denied"` plus the `rule` that fired (the same verdict
`sofuu.policy.evaluate` gives in-process tool calls).

```bash
curl -s localhost:7707/v1/chat/completions \
  -H "Authorization: Bearer $SOFUU_SERVE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Name three prime numbers"}]}'
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
make bc-check        # fails if a shipped driver .bc lags its .js
./sofuu doctor       # confirms QTSQ linkage + a real brain round-trip
```

`make` refuses to build without the QTSQ codec (pass `SOFUU_ALLOW_NO_QTSQ=1` for a memory-less binary) — a codec-less build silently no-ops memory, so it is opt-out, not opt-in.

Key gates: `cargo test --release` · `make test` · `make headless-test` (all 5 C samples assert) · `make abi-check` · `make size-check` · `make bc-check` · `./sofuu doctor`.

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
