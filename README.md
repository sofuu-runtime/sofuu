# ⚡ Sofuu (素風)

> *A fast, private JavaScript runtime built in **Rust** (shell, parsers, memory logic) with a **C** core (QuickJS engine, libuv event loop, SIMD kernels, QTSQ codec) — designed from day one for AI-era workloads.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0--beta-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()

> ⚠️ **Beta software.** Sofuu is under active development. APIs may change before a stable release.

---

## What is Sofuu?

Sofuu is a **server-side JavaScript runtime** — the kind of thing you'd use to run your AI backend, your API server, or your edge automation script. It has no browser API, no DOM, and no renderer.

What makes it different from Node.js, Deno, or Bun is that all the things you normally have to install separately — LLM streaming, vector math, MCP client/server, HTTP — are just built in at the C level. No npm packages needed for any of it.

Beyond that, it ships the parts people usually pay for: **offline embeddings with no model file** (0 bytes, ~2 µs a document — median over 200 samples; treat it as a magnitude, not a benchmark constant), **image embeddings** in the same space as text, **speech in and out** (provider endpoints or the OS's own engines, offline), an **encrypted local brain** that persists across sessions, and five **tiny context-economy models** (~8k parameters each) that decide what enters the context, what actions are worth taking, and what stays. All of it runs offline, deterministically, with no API key.

You build your frontend (React, Next.js, plain HTML — whatever you like) completely separately. Sofuu runs the server side.

---

## Open core

The Sofuu runtime is **fully open source (MIT)** — engine, CLI, JS/TS APIs, agents, MCP, HTTP, vector ops, local memory, embeddings, and the SDK bindings. Use, modify, and distribute it freely, commercially or not.

Proprietary are only: the **QTSQ format specification** ([terms](LICENSES/QTSQ-FORMAT.txt)) and **future hosted/cloud/enterprise services**, which will be separate offerings around the core. Nothing is removed from the core to create them. Full boundary: [`docs/OPEN-CORE.md`](docs/OPEN-CORE.md).

---

## Install

> **Platform: macOS (Apple Silicon and Intel) is the only host we ship a
> binary for today.** CI builds and tests macOS only, and the release has no
> Linux or Windows artifact — the installer says so plainly instead of
> failing on a missing download. Building from source works anywhere,
> including Linux and Windows: the cross-compile targets (`make
> linux-x86_64`, `make windows-x86_64`) and `scripts/cross/*.sh` are all
> still in the tree, just not exercised by CI for now.
>
> The **SDK packs** — SwiftPM, CocoaPods, the Android AAR — are unaffected;
> those target iOS/Android *devices*, not desktop hosts.

**The CLI** — pick whichever you prefer; all three install the same binary.

```bash
# 1. curl | sh (no package manager)
curl -fsSL https://sofuu.xyz/install | sh

# 2. npm — the JS ecosystem's default
npm install -g sofuu      # or: npx sofuu

# 3. Homebrew-style manual
#    → https://sofuu.xyz/downloads  (macOS tarball + .sha256)
```

Or build it yourself from source:

```bash
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
export SOFUU_QTSQ_DIR=/path/to/black-hole-disk   # the encrypted-brain codec
make
./sofuu version
```

> `make` refuses to build without the QTSQ codec unless you pass
> `SOFUU_ALLOW_NO_QTSQ=1`. A codec-less build is *invisibly* broken — memory
> no-ops and session persistence fails — so it is opt-out, not opt-in. Run
> `./sofuu doctor` to confirm a build is healthy.

**Embedding the runtime in your own app** is a different, packaged surface —
Swift Package Manager, CocoaPods, and Gradle. See [Embed Sofuu](#embed-sofuu-libsofuu).

---

## Sofuu Desktop (macOS)

A native Tauri app wrapping the same engine — same turn engine
(`src/js/chat.js`), same session mesh, same agents; API keys live in the
macOS Keychain. Design + build details: `PLAN-DESKTOP.md`.

```bash
cd sofuu-desktop
npm install
npm run tauri dev      # dev window
npm run tauri build    # → target/release/bundle/macos/Sofuu.app (+ .dmg)
```

The desktop app adds a permission gate: read-only tools run automatically;
anything that writes or executes asks first (allow / deny / always-allow).

---

## Quick Start

**Pick a provider on first launch — there is no default model:**

```bash
sofuu        # first run: guided setup asks for provider + model (+ key)
             # later: /provider to switch, /model to change model
```

Sofuu speaks three ENDPOINT formats (a "provider" is just a saved endpoint +
key + format). Settings persist in `~/.sofuu/config.json`:

| wire (endpoint format)        | example provider | example `model`   | API key              |
|-------------------------------|------------------|-------------------|----------------------|
| OpenAI (`/chat/completions`)  | `"openai"`       | `"gpt-5"`         | `OPENAI_API_KEY`     |
| OpenAI (`/chat/completions`)  | `"openrouter"`   | `"z-ai/glm-4.6"`  | `OPENROUTER_API_KEY` |
| Anthropic (`/v1/messages`)    | `"anthropic"`    | `"claude-sonnet-4-6"` | `ANTHROPIC_API_KEY` |
| local server (`/api/chat`)    | `"ollama"`       | `"llama3"`        | none                 |

Any other host with an OpenAI-compatible or Anthropic endpoint works too —
add it with `/provider` (the wire format is auto-detected from the URL).

**Endpoints are formats, not companies.** The OpenAI endpoint
(`/chat/completions`) is what MOST providers speak — deepseek, qwen, grok,
llama, mistral and countless gateways serve their own models through it; it
is not limited to OpenAI models. The Anthropic endpoint (`/v1/messages`) is
named for where it originated but any provider/model may use it. Sofuu keeps
these orthogonal: request SYNTAX comes from the endpoint, limits (context
window, max output, thinking) come from the model's capability registry — so
a Claude model behind an OpenAI-format gateway, or a non-Claude model behind
an Anthropic-format one, both just work.

**Stream an AI response — no packages to install, no SDK to configure:**

```js
// server.js
const server = sofuu.http.createServer(async (req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });

  const stream = sofuu.ai.stream("Explain how SIMD works.", {
    provider: "openai",   // any of the table above; a model is REQUIRED
    model: "gpt-4o",
  });

  for await (const chunk of stream) {
    res.write(chunk.text);
  }

  res.end();
});

server.listen(3000, "0.0.0.0");
console.log("Server running on port 3000");
```

```bash
sofuu run server.js
```

**TypeScript works out of the box — no tsconfig, no compiler step:**

```typescript
interface Message { role: string; content: string; }
const greet = (msg: Message): string => `[${msg.role}] ${msg.content}`;
console.log(greet({ role: "user", content: "Hello, Sofuu!" }));
```

```bash
sofuu run hello.ts
```

**Run an AI agent that searches the web — still zero packages:**

```js
// research.js — an agent with the built-in coding + web tools (no API key
// needed for search; the duckduckgo engine is keyless by default)
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Search first, then answer with sources.",
  tools: ["code", "web"],       // code: read/write/edit/grep/glob/list_dir/bash
                                 // web: web_search + web_open
  memory: "agent",              // private long-term memory per agent
});

const r = await sofuu.agent.run("researcher", "What shipped in QuickJS 2024?");
console.log(r.answer);
console.log(sofuu.agent.renderTrace(r));   // the full run tree
```

```bash
sofuu run research.js
```

---

## Why Sofuu?

*Competitor cells read "add a package" when you need an extra dependency; Sofuu's
AI features are in the binary itself.*

| | Node.js | Deno | Bun | **Sofuu** |
|---|---|---|---|---|
| Binary size | ~100 MB | ~80 MB | ~60 MB | **3.2 MB** (measured) |
| Startup time | ~50 ms | ~30 ms | ~7 ms | **~3 ms** |
| Language | C++ | Rust | Zig | **Rust-first + C (low-level)** |
| LLM streaming | add a package | add a package | add a package | **Built in** |
| MCP client + server | add a package | add a package | add a package | **Built in** |
| SIMD vector math | add a package | add a package | add a package | **Built in (NEON/AVX2)** |
| Embeddings (offline, 0 bytes) | add a package | add a package | add a package | **Built in (`ai.embed`, hash-v1)** |
| Image embeddings | add a package | add a package | add a package | **Built in (`ai.embedImage`)** |
| Speech in / out | paid API | paid API | paid API | **Built in (`ai.transcribe`/`speak` + OS bridges)** |
| Persistent local brain | paid service | add a package | add a package | **Built in (encrypted `.qtsq`)** |
| TypeScript support | Separate compiler | Built in | Built in | **Built in (Rust stripper)** |
| AI agents + sub-agents | add a package | add a package | add a package | **Built in (`sofuu.agent`)** |
| Web search for agents | add a package | add a package | add a package | **Built in (`sofuu.web`, keyless default)** |
| Context-economy ML gates | — | — | — | **Built in (`sofuu.ml`, ~8k params each)** |
| Memory safety of shell | ❌ | ✅ | ✅ | **✅ (Rust shell)** |

---

## Embedding benchmarks

Measured, not estimated. Reproduce with:

```bash
bash scripts/bench/fetch_datasets.sh      # BEIR SciFact + STS test split
cargo run -p ml-train --release -- bench # SOFUU_BENCH_JSON=out.json to capture
```

**Text — BEIR SciFact** (5,183 scientific abstracts, 300 judged queries,
official qrels) and **STS test** (1,379 human-rated sentence pairs,
Spearman ρ). Apple M2 Pro, release build, single thread, no network:

| space | dim | params | artifact | median embed | SciFact R@1 | R@5 | nDCG@10 | STS ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 *(reference, measured here)* | — | — | 0 | — | 0.537 | 0.750 | **0.663** | — |
| **hash-v1** *(shipped default)* | 768 | **0** | **0 B** | **~0.002 ms** | 0.307 | 0.493 | **0.410** | **0.613** |
| sem1-64 *(opt-in)* | 64 | 13,392 | 13.7 KB | ~0.05 ms | 0.003 | 0.017 | 0.014 | 0.421 |
| sem2-64 *(opt-in)* | 64 | 30,032 | 29.9 KB | ~0.08 ms | 0.037 | 0.090 | 0.081 | 0.517 |

Read this honestly:

- The **BM25 row is the calibration check** — our implementation lands at
  0.663 nDCG@10 against the ~0.665 published for SciFact, so the harness,
  the qrels and the metrics agree with the literature.
- **hash-v1 is the default because it is the one that works.** It reaches
  62% of BM25's nDCG with **zero parameters, zero bytes and no model
  file** — the whole embedder is arithmetic over trigram hashes, at ~2 µs
  a document. It is also the only space that beats nothing: it is what
  `ai.embed` uses by default, and it is what makes the brain work with no
  download and no API key.
- **The 64-dim learned spaces are not general-domain retrievers.** On
  out-of-domain text they reach 12% (sem2) and 2% (sem1) of BM25's nDCG.
  They exist to be tiny and to win on their own in-domain gates; they are
  opt-in, and the repo's own pre-registered gate (`embed-eval`) currently
  returns **FAIL** on two of them, so they must not replace the default.
  We are reporting this rather than quietly dropping the table.
- We did **not** run a hosted-model baseline (it costs money per run);
  expect a 1536-dim API embedder to be well ahead of every row here. That
  is the trade: they need a network round-trip, a key and cents per
  million documents.

**Image — IMG1** (`ai.embedImage`, 7.4 KB projector, 144 features → 64-d):
text→image **R@1 0.581 / R@5 1.000** on 93 held-out procedurally
generated scenes across 39 unseen tag combinations, versus 0.247 for raw
image features and 0.538 for a lexical caption baseline. **This is a
synthetic, in-domain set** — it shows the projector learns its intended
mapping, not that it matches a real-image retriever. We publish it
labelled as synthetic because there is no honest real-image number to put
beside it yet.

The internal synthetic suites (`embed-eval`, `img-eval`) report friendlier
numbers (fused 0.917 vs hash 0.875 on our own 8-category corpus). They
are training gates, in-domain by construction, and are not comparable to
the external table above.

---

## CLI Reference

```bash
sofuu                        # Start interactive AI chat (Claude-Code-style)
sofuu chat                   # Same interactive AI chat
sofuu chat -m <model>        # Chat with a specific model
sofuu chat -p <provider>     # Chat with a specific provider
sofuu chat --effort <lvl>    # Reasoning effort (low|medium|high|max)
sofuu repl                   # JavaScript evaluation REPL
sofuu run <file.js>          # Run a JavaScript file
sofuu run <file.ts>          # Run a TypeScript file (auto-transpiled)
sofuu eval "<code>"          # Evaluate a JS expression inline
sofuu bundle <entry.js>      # Bundle everything into one distributable file
sofuu bundle <entry.js> -o <out.js>
sofuu install                # Install packages from package.json
sofuu add <package>          # Add and install an npm package
sofuu agent list             # List agent definitions (~/.sofuu/agents/*.js)
sofuu agent run <name> "task"       # Run one agent headless
sofuu agent run <name> "task" --json  # …with a JSON result
sofuu serve --brain <path>    # Serve the brain over HTTP (--port, --host, --token)
sofuu doctor                 # Health check: QTSQ link, brain round-trip, ctx resolution
sofuu version                # Print version information
sofuu help                   # Print usage help
```

`sofuu doctor` answers the three questions you cannot otherwise check —
is the brain actually linked and persisting, where does it live, and what
context window does the current model resolve to (with the evidence behind
it):

```text
QTSQ / brain
  ✓ QTSQ linked into this binary (brain persistence available)
  ✓ brain write→read round-trip passed (temp store 5.2 KB)
  · brain file not created yet: /path/.sofuu/brain/brain.qtsq

context window
  · model stealth/space-bunny-alpha @ https://openrouter.ai/api/v1/chat/completions
  · resolved window 32768 (source: default) · max output 384000 (config)
  · discovered model caps: 677 entries cached
```

A non-zero exit means a hard dependency is missing (an unlinked QTSQ
build silently no-ops memory — this is what catches it).

**Building from source** requires the QTSQ codec checkout, because a
binary without it cannot persist a brain or a session:

```bash
SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> make
```

`make` refuses to build without it (pass `SOFUU_ALLOW_NO_QTSQ=1` if you
really want a memory-less binary). The chat banner never claims memory
persists when the codec is missing.

**Chat slash commands** (inside `sofuu` / `sofuu chat`):

```text
/help  /model <name>  /provider <name>
/effort <lvl>  /compact  /clear  /brain [on|off]  /rlm [on|off|auto]
/tools  /agents  /version  /exit

# Brain & memory
/remember <fact>   # Pin a fact to the brain (survives decay)
/why               # Show which memories shaped the last answer
/share [path]      # Export brain as a portable card
/import <path>     # Import a brain card from someone else
/cost              # Token usage + spend breakdown + budget
/ghost [on|off]    # Toggle ghost prompt completion

# Context
@file[:start-end]  # Attach file contents to a prompt
@name <task>        # Run a loaded agent directly (also @agent:name <task>)
/watch <path>       # Watch a directory for changes (surfaces in chat)
/hooks              # Show ~/.sofuu/hooks.js user middleware info
/resume [id]        # Browse + resume a past session
/serve              # Info on serving the brain over HTTP
/ml [on|off|learn|adopt|discard|reset|wrong|wasted|info]  # Context-economy gates + supervisor online learning

# Session mesh (same project shares context in real time)
/sessions           # List sessions on this project
/context [id]       # Show a session's full context
/ctx [<tokens>]     # View/set the context window (default = the model's real window)
/maxout [<tokens>]  # View/set max output tokens (default = the model's real max output)
/work <desc>        # Announce what you are working on
/done               # Clear your current task
/note <msg>         # Record a personal note (peers see it)
/notify <msg>       # Broadcast a critical notice to all sessions
/sync [on|off]      # Toggle session-mesh polling
```

### Dynamic limits & reasoning effort (per-model, not hardcoded)

Sofuu resolves every limit from a per-model capability registry
(`sofuu.ai.modelCaps(model)`) — never from flat constants:

- **Max output (output-only)** — `max_tokens`/`max_completion_tokens` is strictly output limit per model
  (e.g. 64k for `claude-sonnet-4.x`, 128k for `gpt-5`, 100k for o-series).
  Unknown models on OpenAI omit the field (provider default); Anthropic unknown gets endpoint-driven `~71k` so any model can use thinking. `/maxout <n>` overrides; 384k is the config ceiling. Output limit never defines thinking.
- **Context window (total `input + thinking + output`)** — e.g. `1M` total. History budgeting, `@file` attachment budgets, recall budgets and tool-result caps all scale with `ctxWindow`. `/ctx <n>` overrides. **Per-answer limits are not session caps:** `32k` thinking and `64k` output are *per-answer* limits (how much the model can think/output in one answer). Across many turns inside the `1M` window you can think `≫32k` and output `≫64k` cumulatively — each turn's `thinking+output` just appends to `history` until the window fills. When the window is nearly full (`70%` of `ctxBudget`), CLI **auto-compresses** by summarizing the whole context into one system message and resetting usage to `~0` so you can continue indefinitely.
- **Thinking effort** — `/effort low|medium|high|max` is strictly what you picked (`off` → no thinking on any wire, never more than selected):
  - **Anthropic wire (`/v1/messages`) `64k` is *thinking* max, not context window** (`200k`/`1M` etc). It is endpoint-driven — any model on that endpoint can use it (not only Claude). Budget = fraction of `ctxWindow*0.30` capped at `64k` (`low 6%`, `medium 25%`, `high 50%`, `max 90%`). E.g. `200k` ctx → `60k` thinking cap → `max` `54k` thinking. Thinking lives inside total context, does not define output limit (`max_output` stays output-only).
  - **OpenAI wire (`/chat/completions`) is generic** — hosts almost all providers/models, not only OpenAI's `gpt-5.5` etc. Models with a discrete ladder (`o-series`, `GPT‑5` → `minimal/low/medium/high`) accept only their supported levels; picking `max` (Anthropic-only) **falls back to `high`** strictly, never more than selected. Non-reasoning models omit the field; `off` → no thinking on any wire.
  - If a model rejects thinking at runtime anyway (undocumented endpoint),
    sofuu detects it from the API error, marks the model as non-thinking
    (persisted), tells you, and retries the turn once without effort. The
    `/effort` picker then reports: *this model does not support thinking*.

Settings (provider, model, effort, brain) persist in `~/.sofuu/config.json` —
the file is written on first setup (the first-run wizard), there's no
built-in default. Cost tracking fields (`pricing`, `budget_usd`,
`spend_total_usd`), verification (`verify`, `verify_model`), and ghost
completion (`ghost`) also persist there.

**One agent loop for everything:** every chat turn runs the runtime's agent
loop (`sofuu.agent.run`) with the MCP tools below plus built-in `web_search`
/ `web_open` tools — the model plans, calls tools, delegates, and streams the
final answer. **Zero-config MCP tools:** add `~/.sofuu/mcp.json` and the chat
connects to each server on demand, routing every tool call to the server that
owns it (`/tools` shows the live list, `/agents` lists agent definitions):

```json
[{"name": "fs", "command": "npx @modelcontextprotocol/server-filesystem /tmp"}]
```

### Chat Features — Brain, Cost, Verify, Watch, Hooks, Ghost

The interactive chat has a set of brain-level features that make it more than
a thin wrapper over `sofuu.ai`:

**Brain memory** (`/brain on` to enable): the chat uses the runtime's CMA
memory — every user prompt is embedded (offline TF-IDF by default) and
stored in the brain (`~/.sofuu_brain.qtsq`). Relevant memories are recalled
and injected into the system prompt each turn. `/remember <fact>` pins a fact
directly (survives decay); `/why` shows which memories shaped the last answer
(with similarity scores and roles).

**Cost tracking** (`/cost`): every turn's token usage is costed against a
pricing table (seeded with popular models; add your own in config under
`pricing`). `/cost` shows per-turn, session-total, and lifetime spend.
Set `budget_usd` in config to cap spend — the chat preflight-blocks turns
when the budget is exceeded.

**File mentions** (`@file` or `@file:start-end`): type `@path` in a prompt to
attach file contents. Paths resolve against cwd; `..` escapes are rejected.
The attachment budget scales with the model's context window (~25% of it,
2048–65536 tokens). History stores only the manifest — follow-up turns don't
re-send files.

**Agent mentions** (`@name <task>` or `@agent:name <task>`): run a loaded
agent (`~/.sofuu/agents/*.js`) directly with its own definition — a focused
run with the agent's system prompt, tools and memory scope; no chat history
is injected. Agents win over same-named files; unknown names fall through to
the `@file` path. Esc stops the run like any turn.

**Per-model capability registry**: context windows, max output tokens,
thinking support and effort ladders come from a built-in per-model table
(`rt/model_caps.rs`, exposed as `sofuu.ai.modelCaps(model)`). RLM routing
(`/rlm auto`) uses the real window (GPT/Claude/Llama/Qwen/Grok/…) so
oversized turns route correctly; output requests default to the model's
real max output instead of a flat constant.

**Context-window detection** (evidence ladder, strongest wins):
a real provider 400 → caps the endpoint itself publishes for this exact
model (harvested from its `/models` listing) → the built-in registry →
an honest `32,768` default. `/model` and `/provider` force a fresh
harvest for the endpoint you just selected and print what was found, so a
1M model never sits silently at 32k:

```text
/model <name>
  ✓ Model → some/model
  ✓ detected 1,048,576 context for some/model (discovered, 214 models)
```

Overrides behave differently on purpose:

- `/ctx <n>` is **explicit** — you typed it for this model, so it is
  obeyed exactly as given, even above everything Sofuu knows. If it sits
  above the known evidence you get one advisory line naming the bound; if
  the provider disagrees, its 400 teaches the real limit and the session
  corrects itself.
- A `ctx_window` **inherited** from `config.json` (or another model) still
  shrinks to the selected model's real bound, so a stale global can never
  shadow a smaller model.

`/ctx default` clears both. `sofuu doctor` prints the current resolution
and its evidence without starting a chat.

**Directory watching** (`/watch <path>`): watches a directory for changes
(mtime + size poll on the 1s tick, 2k file cap, skips `.git`/`node_modules`/
`target`). Changed files surface as one-liners in the chat context. `/watch`
with no arg lists watched paths; `/watch off` clears.

**User middleware** (`~/.sofuu/hooks.js`): write a `pre({ text, cfg })` and/or
`post({ text, answer, usage })` function to transform prompts or answers.
Each call has a 3s timeout; three consecutive failures auto-disable hooks
for the session.

**Ghost completion** (`/ghost on`): as you type, the chat embeds your prefix
(sync `sofuu.ai.embedLocal` + `brain.recall`), and if a past user prompt
matches at cosine ≥ 0.9, it suggests the suffix. Off by default.

**Session resume** (`/resume [id]`): browse past sessions (stored as `.qtsq`
files under `<project>/.sofuu/sessions/`), pick one, and load its turn
history back into the chat.

**Brain sharing** (`/share [path] [label]` + `/import <path>`): export the
brain as a portable JSON card (`sofuu-brain-card@1` schema with record count
and brain path); import reads and validates a card from someone else.

**Brain server** (`sofuu serve --brain <path>`): serve the brain over HTTP
for remote access — `GET /health`, `POST /remember`, `GET /recall?q=…&k=…`,
`POST /share`. Bearer-token auth (auto-generated or `--token`). Remote bind
(`0.0.0.0`) requires an explicit token.

### Embed Sofuu (libsofuu)

Sofuu ships as an embeddable runtime for any host — iOS, Android, desktop,
server, edge. QuickJS is a pure interpreter (no JIT), so iOS App Store
executable-memory policy is satisfied by design. The pitch in one line:
**a private AI agent with memory, offline, in a ~3 MB library.**

**iOS / macOS (Swift):**

```swift
// Package.swift — or: pod 'Sofuu', '~> 0.2'
import Sofuu

let sofuu = try Sofuu()
let vec = try sofuu.embed("the sky is blue")     // 768-dim, offline, 0-byte model

let brain = try Brain.openDefault(sofuu: sofuu)  // encrypted, local, yours
try brain.remember("the sky is blue")
let hits = try brain.recall("what colour is the sky?", k: 3)
```

**Android (Kotlin):**

```groovy
dependencies { implementation 'com.sofuu:sofuu-android:0.2.0' }
```

```kotlin
val vec = sofuu.embedLocal("hello")   // FloatArray, 768-dim, no network
```

**C (macOS / Linux):**

```c
#include "sofuu_embed.h"
SofuuRuntime *rt = sofuu_rt_new(NULL);
char *out = NULL;
sofuu_rt_call(rt, "memory.open", "{\"path\":\"/tmp/b.qtsq\",\"dim\":768}", &out);
sofuu_free(out);
sofuu_rt_free(rt);
```

**CLI on npm:**

```bash
npx sofuu            # chat
npx sofuu run app.ts
```

Every `examples/headless/` sample **asserts** its calls, so a broken SDK
fails `make headless-test` rather than shipping:

| Sample | Proves |
|---|---|
| `c_embed.c` | eval + funnel + embed + a full brain round-trip |
| `c_embed_vec.c` | vector ABI: spaces, batch, error guards |
| `c_llm.c` | a real LLM call (mock in CI, live with a key) |
| `c_rlm.c` / `c_agent.c` | RLM long-context Q&A / agent run + stream + cancel |

From source: `make libsofuu`, `make dist-macos`, `make dist-ios`
(xcframework), `make dist-android` (per-ABI .so). `make dist-linux` still
exists and still works, but CI does not build or publish it for now. See
[`docs/EMBEDDING-DIST.md`](docs/EMBEDDING-DIST.md) (artifacts + install) and
[`docs/EMBEDDING.md`](docs/EMBEDDING.md) (the contract). Design + status:
[`PLAN-HEADLESS.md`](PLAN-HEADLESS.md).

### Full features on Windows & Linux (QTSQ port)

> **Shipped platform: macOS only, for now.** The QTSQ codec genuinely ports
> to all three desktop OSes and the build path for each is intact — the
> Makefile's `linux-x86_64` / `linux-arm64` targets and `scripts/cross/*.sh`
> still cross-compile. What is switched off for now is *building and
> shipping* them: CI runs macOS only, and no Linux/Windows release artifact is
> published. Re-enabling either is a small, self-contained change.

All Sofuu features — including **QTSQ session persistence** (brain/memory,
session store, `.qtsq` container save/load, vault, secure_text, deniable
encryption, authorship anchoring) — build on all three desktop platforms.
The QTSQ codec is a proprietary local checkout; each OS links it natively:

| Platform | QTSQ artifact | Built by | Linked by `build.rs` |
|---|---|---|---|
| macOS | `libqtsq.a` + `compressor/libqtc.a` | `make` in the checkout | static=qtsq, static=qtc, `-lz` |
| Linux | `libqtsq.a` + `compressor/libqtc.a` | same Makefile (gcc/clang — clean-build proof: docker `gcc:13`, all 21 FFI symbols link, 8 layout guards pass) | same as macOS |
| Windows | single `qtsq.lib` (qtc + crypto + shim baked in) | `cmake -S windows -B windows/build-win64 && cmake --build windows/build-win64 --config Release` in the checkout — an MSVC port with a Win32 POSIX-compat shim (`windows/shim/`: pthreads→Win32 threads, mmap→CreateFileMapping, dirent→FindFirstFile, clock/time/rename/sysconf equivalents) so turbo keeps **parallel** worker threads | `SOFUU_QTSQ_DIR` + `SOFUU_QTSQ_LIB`; zlib from vcpkg `x64-windows-static-md` (`SOFUU_ZLIB_DIR` to override); bcrypt/advapi32 auto-linked |

Point the build at a checkout with `SOFUU_QTSQ_DIR` (default:
`~/projects/black-hole-disk`); a missing checkout degrades to the no-QTSQ
build (session persistence disabled) instead of failing. The build script
also re-runs an 8-assert `_Static_assert` layout guard against the checkout's
headers on every OS (via the `cc` crate: clang/gcc on POSIX, cl.exe on
Windows), so a format drift fails the build instead of corrupting files.
Windows QTSQ is excluded from CI (proprietary checkout can't ship there);
CI artifacts stay no-QTSQ/fail-closed. Windows verification status: full
compile+link closure proven with `zig cc -target x86_64-windows-gnu`
(100/100 sources, 21-symbol link → PE32+, layout guards pass); final
native-MSVC proof happens on a Windows machine. See the QTSQ checkout's
`windows/README.md` for the port's internals.

---

## API Reference

### `sofuu.ai` — LLM Streaming & Vector Math

```js
// Stream tokens as they arrive — non-blocking, C-backed
const stream = sofuu.ai.stream("What is the speed of light?", {
  provider: "openai",   // endpoint format: "openai" | "anthropic" | "local"
                        // (any other name = a custom OpenAI-compatible host)
  model: "gpt-5",       // required — there is no default model
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}

// Per-model capability registry (powers the dynamic limits)
JSON.parse(sofuu.ai.modelCaps("claude-sonnet-4-6"))
// → { known: true, ctxWindow: 200000, maxOutput: 64000,
//     thinking: "budget", maxThinkingBudget: 64000, efforts: [] }

// One-shot completion
const result = await sofuu.ai.complete("What is 2 + 2?", { provider: "openai", model: "gpt-4o" });
console.log(result.text);    // "4"
console.log(result.tokens);  // { input: 12, output: 1 }

// SIMD-accelerated vector similarity (runs on CPU vector registers directly)
// Accepts Float32Array (zero-copy) or plain Array
sofuu.ai.similarity(a, b)   // Cosine similarity  → number in [-1, 1]
sofuu.ai.dot(a, b)           // Dot product        → number
sofuu.ai.l2(a, b)            // Euclidean distance → number

// Bundled offline embeddings (pure Rust, no network — the chat's default
// memory path; remote embedding providers are opt-in via config):
// text → 768-dim unit Float32Array
const v = sofuu.ai.embedLocal("hello world");
// Learned semantic spaces (offline, bundled): SEM1 64-dim, SEM2 64-dim
const s = sofuu.ai.embedLocalSemanticV2("hello world");
JSON.parse(sofuu.ai.embedInfoV2());  // → { id, dim: 64, ... } space manifest

// ai.embed defaults to the bundled local model (SEM2-64, offline):
const v2 = await sofuu.ai.embed("hello world");                 // → Float32Array(64)
const vs = await sofuu.ai.embed(["a", "b"], { space: "hash-768" }); // → Array<Float32Array>
const batch = await sofuu.ai.embedBatch(["a", "b", "c"]);       // → Array<Float32Array>, one call
// Network embeddings (explicit provider — BYO key, billed by them):
const e = await sofuu.ai.embed("hello world", { provider: "openai", model: "text-embedding-3-small" });

// Vision input: user messages carry images (data URLs) on both wires
// (OpenAI image_url parts + Anthropic base64 blocks, count/size capped)
const r = await sofuu.ai.complete({
  messages: [{ role: "user", content: "read this", images: ["data:image/png;base64,…"] }],
  provider: "openai", model: "gpt-5",
});

// Image embeddings (offline, bundled IMG1 projector — PNG/JPEG bytes in,
// 64-dim joint-space vector out; text queries retrieve images):
const iv = sofuu.ai.embedImage(await sofuu.fs.readFileBytes("shot.png"));

// Multimodal recall recipe (image store + text query):
const store = sofuu.memory.open("/tmp/mm.qtsq", 64, "image-projector-v1");
store.remember(iv, "login dialog", "user", 0);
store.recall(sofuu.ai.embedLocalSemanticV2("login screen"), 3);

// Provider voice (OpenAI-compatible audio endpoints, BYO key):
const t = await sofuu.ai.transcribe(audioBytes, { provider: "openai", model: "whisper-1" });
console.log(t.text);
const s = await sofuu.ai.speak("hello", { provider: "openai", model: "tts-1", voice: "alloy" });
// → { audio: Uint8Array, format: "mp3" }

// Headless SDK: sofuu_embed_local/batch/image/info + sofuu_voice_transcribe/speak
// C ABI (libsofuu) + Swift/Kotlin wrappers (incl. on-device OS speech:
// Apple Speech/AVSpeech, Android SpeechRecognizer/TTS — offline, free).
// Run a command and capture its output
const r = await sofuu.exec("echo", ["hi"]);
console.log(r.code, r.stdout, r.stderr);   // 0 "hi\n" ""
```

### `sofuu.rlm` — Long-Context Q&A (RLM scaffold)

Answers questions about contexts far larger than any model's window. The
context never enters the prompt whole: it lives host-side, and the model
reads it through a hermetic QuickJS sandbox (`chunk()`, `grep()`, `peek()` …)
and can recurse on focused sub-questions (`llm()`), with hard budgets.

```js
const res = await sofuu.rlm.query(bigText, "What error caused the outage?", {
  provider: "openai", model: "gpt-4o",   // required — no default model
  maxRounds: 16,        // model turns      (default 16)
  maxLlmCalls: 24,      // llm() sub-calls  (default 24)
  maxWallMs: 120000,    // wall clock       (default 120s)
  maxDepth: 3,          // recursion depth  (default 3; v1 sub-calls are plain)
  chunkChars: 16000,    // context chunking (default 16000)
  trace: true,          // include the full event trace
});
res.answer;        // the answer string
res.calls;         // llm() sub-calls made
res.rounds;        // model rounds used
res.stopped;       // null | "budget_*" | "loop_detected" | "aborted"
res.trace;         // [{t, kind:"chunk|grep|peek|llm|final|…", …}]
```

Routing helpers for "should I even use this?":

```js
sofuu.rlm.route(ctxTokens, windowTokens, question)  // → "rlm" | "plain"
// Every decision can be logged to ~/.sofuu/routing_log.jsonl (data for a
// future learned router):
sofuu.rlm.logRoute({ ctxTokens, windowTokens, question, route, latencyMs });
```

In chat, `/rlm auto` runs this router per turn; `/rlm on` forces every turn;
`/rlm off` (default) disables. Security: the sandbox is frozen-whitelist
only (no fs/net/process), time is deterministic, answers/traces are size-
capped, and API keys never appear in traces or logs. Note the trust
boundary: context *slices* go to your configured provider — don't run RLM
over content you wouldn't send that provider. Full design + status:
[`PLAN-RLM.md`](PLAN-RLM.md).

v1 limits, honestly: `llm()` sub-calls are plain completions *unless*
`recurseVia: { agent }` routes them through a tool-using agent (nested
episodes land next); Esc aborts between rounds; routing windows are a
static per-provider table; validated against a mock provider so far — the
first live-model run may tune the driver prompt.

### `sofuu.agent` — Agents & Sub-Agents

First-class agents, headless-first: an agent is a named definition (system,
tools, model, memory scope, budget); delegation is just a `delegate` tool
call the model makes — composition is the model's job. The same loop powers
interactive chat.

```js
// Define an agent (inline tools, MCP servers, or the built-in web tools)
sofuu.agent.define({
  name: "researcher",
  system: "You research precisely. Cite sources.",
  tools: ["web"],                      // built-ins: web_search + web_open
  // tools: [{ name, description, parameters, execute: async (args) => … }],
  // tools: [{ mcp: [{ name: "fs", command: "npx @modelcontextprotocol/server-filesystem /tmp" }] }],
  memory: "agent",                     // "shared" | "agent" (private) | "off"
  budget: { maxSteps: 12, maxDepth: 2, maxTokens: 200_000, maxWallMs: 300_000 },
  agents: ["librarian"],               // sub-agents reachable via `delegate`
});

const r = await sofuu.agent.run("researcher", "compare Rust and Zig GC stories");
r.answer;    // final text
r.subRuns;   // child AgentResults (delegation tree)
r.usage;     // { promptTokens, completionTokens, llmCalls, toolCalls, wallMs }
r.trace;     // [{t, kind:"plan|tool|delegate|answer|…", payload}]
r.stopped;   // null | "budget_steps" | "budget_tokens" | "budget_wall" | "cancelled"
```

More surface:

```js
// Parallel fan-out (single loop, promise pool; default concurrency 4)
const rs = await sofuu.agent.runMany([{ agent: "researcher", task: "…" }], { concurrency: 4 });

// Map-reduce over huge contexts (the 1M-token document Q&A shape):
// chunk → parallel tool-using sub-agent runs → reduce merge
const r = await sofuu.agent.mapContext(hugeText, "find every license issue", {
  agent: "researcher", chunkChars: 16_000, concurrency: 4, reduceAgent: "writer",
});

sofuu.agent.cancel(runIdOrSignal);      // kills the run + its children
sofuu.agent.list();                     // registry summary
sofuu.agent.loadDir();                  // load ~/.sofuu/agents/*.js (broken files listed, never fatal)
sofuu.agent.renderTrace(r);             // human-readable run tree
sofuu.agent.contextWindow(model);       // per-model context window used for RLM routing (read-only)
```

Safety rails are structural, not advisory: a run at `maxDepth` gets **no**
`delegate` tool at all (recursion is impossible), delegation cycles are
detected via the chain and surfaced as tool errors, children inherit the
*remaining* tree budget, every tool call has a timeout (default 30s), and
memory scopes are enforced host-side (`"off"` agents provably write nothing).
RLM integrates both directions: oversized agent turns route through
`sofuu.rlm` automatically (`rlm: "auto"`, with clean fallback), and
`sofuu.rlm.query(ctx, q, { recurseVia: { agent: "researcher" } })` makes the
sandbox's `llm()` sub-calls run through a tool-using agent. Runs can log one
JSON line each to `~/.sofuu/logs/agent_runs.jsonl` (`opts.logs`).

Agent definitions live in `~/.sofuu/agents/*.js` (plain scripts that call
`sofuu.agent.define({...})` — trusted like any user script). Run them from
a shell with `sofuu agent run`, or headless from any script. Design + status:
[`PLAN-AGENTS.md`](PLAN-AGENTS.md).

**Built-in coding tools** (`tools: ["code"]`, also active in every chat
turn): `read_file` (cat -n style, so exact text can be quoted into edits),
`write_file`, `edit_file` (exact unique-match replacement — refuses
ambiguous edits and asks for more context), `grep` (regex over the project,
skips .git/node_modules/target), `glob`, `list_dir`, and `bash` (timeout +
output caps, killed on expiry). Rails: writes are jailed to the project
directory; `SOFUU_NO_SHELL=1` strips `bash` entirely; every tool result is
additionally capped by the loop's truncation. Pick individual tools with
`tools: ["read_file", "edit_file", ...]`, or `{ builtin: "grep" }`.

### `sofuu.ml` — Context-Economy Gates (PLAN-ML-GATES)

Tiny on-device models (~8.5k params each — offline, deterministic,
zero-API) that ADVISE the agent loop: they never filter, cap, reorder, or
delete anything on the data path. Guidance rides the ephemeral context
message or the tool result; the LLM decides. `/ml off` (persisted) or
`SOFUU_NO_ML=1` disables everything; every call is try/catch-guarded so a
gate failure can never break a turn.

```js
// Freshness — is this material current for this task? (trained, baked
// weights: 28→112→48→1, threshold tuned for recall; §5/§10 of the plan)
const v = JSON.parse(sofuu.ml.freshness.score(text, task,
  JSON.stringify({ kind: "web" })));   // kind: web|memory|tool|file
// → {"score":0.99,"stale":true,"years":6.0,"reason":"dated 2020"}

// Supervisor — is this NEXT call worth making? (trained, baked weights:
// 33→104→44→1, threshold 0.82; §11 of the plan). Mechanical rules (exact
// repeat calls, re-reads of unchanged files) speak first; where they are
// silent, the net speaks. Advise only — the call still runs; the nudge
// rides the tool result ("[supervisor: …]").
const c = JSON.parse(sofuu.ml.supervisor.check(JSON.stringify(
  { run, step, tool, sig, target, argsText, task, skipTargets: [], budget: 20 })));
// → {"ok":false,"reason":"dup_call","source":"rule","score":1.0,
//    "nudge":"identical call already made at step 1 …"}

// Loop-boundary checkpoint — is the RUN itself spinning/stalled/over
// budget? (the "__loop__" pseudo-action; silent under 3 recorded calls)
const l = JSON.parse(sofuu.ml.supervisor.loop(JSON.stringify(
  { run, step, budget: 20 })));
// → {"ok":false,"reason":"loop_spinning","source":"model","nudge":"…"}

// Online-learning feedback (§13, off by default — see /ml learn below).
// conf (optional, default 1.0) is the label's confidence and becomes its
// weight; the agent abstains on ambiguous results rather than guess.
sofuu.ml.feedback(JSON.stringify(
  { kind: "outcome", model: "supervisor", run, step, wasted: true, conf: 0.6 }));

// Compaction — which history segments are mechanical junk? (trained,
// baked weights: 33→104→44→1; §12 of the plan). The net SELECTS; the
// caller acts. Free tiers (dup/boilerplate/retrievable) are safe to drop
// without an LLM; "summarize" segments need one.
const p = JSON.parse(sofuu.ml.compaction.plan(JSON.stringify(
  { task, summary, recent, segments }), JSON.stringify({ budget: 1200 })));
// segments: [{ text, tokens, age, kind, retrievable, compacted }]
// → {"compact":[2,4,5,3],"keep":[0,1,6,7],"freeable":1018,
//    "tiers":["dup","dup","dup","dup"],"scores":[...]}
// Relevance — within a fixed budget, which candidates deserve the space?
// (trained, baked weights: 37→104→44→1, threshold 0.95 recall-first; §6 of
// the plan). Pre-retrieval ADVISOR: it scores a menu of candidates against
// the task and returns use/skip advice — it never mutates the menu.
const r = JSON.parse(sofuu.ml.relevance.plan(JSON.stringify(
  { task, recent, candidates }), JSON.stringify({ kept: [0] })));
// candidates: [{ text, kind: "file|memory|web|other", strength, role, path }]
// → {"use":[0],"skip":[1,2,3,4,5],"scores":[...]}  (use ordered best-first)

sofuu.ml.info();     // gate states + working-set counters (JSON)
sofuu.ml.track(json) // feed the in-memory context working set

**Compaction is the one irreversible thing the chat does**, so it runs
under two guards that do not depend on the model being right:

- **A per-pass cap** — at most a quarter of the complete turn blocks, and
  never fewer than two survive. A gate that flags 200 of 200 blocks still
  leaves 150 turns in place.
- **A lexical floor** — a turn whose *user* message asks a question, gives
  an instruction, or records a decision/approval is never removed,
  whatever the net says. The last line of defence is deliberately not the
  component under suspicion. Skipped turns are counted and reported, and
  they never consume the drop budget.

**Do the gates earn their bytes?** `ml-train gate-eval` grades the shipped
weights against two references on held-out *families* (whole content
constructions, never random rows):

| gate | params | MLP F1 | logistic F1 | constant F1 |
|---|---:|---:|---:|---:|
| freshness | 8,721 | 1.000 | 0.571 | 0.000 |
| compaction | 8,201 | 0.850 | 0.553 | 0.545 |
| relevance | 8,617 | 0.989 | 0.484 | 0.000 |
| supervisor | 8,201 | 1.000 | 0.832 | 0.769 |
| alloc | 6,321 | 0.991 | 0.973 | 0.000 |

Every gate beats both a constant predictor and a logistic regression on
the same features. Compaction's benchmark was rebuilt to include a family
that is provably non-separable (verified by exhaustive search, not
assumed) — before that, linear also scored 1.000 and the comparison could
not tell whether the network was doing anything.

**The honest caveat**, printed by the command on every run: all five
training sets are synthetic with mechanically derived labels, so a pass
means "learnable on the distribution we generate" — not "correct on real
user traffic". The instrument that would answer that is a shadow-mode A/B
on live sessions; it does not exist yet.
```

When the freshness gate fires, the turn gets ONE evidence-carrying notice
(`[freshness] … dated 2019 … verify its currency …`) on the next context
boundary. Acceptance bar (held-out construction families, measured once):
accuracy 0.997, precision 1.000, recall 0.990 — beating logistic
regression (0.976) on identical features; committed weights re-verified by
`cargo test` fixtures on every build.

Chat uses the compaction model as an opportunistic gate in front of the
one-shot cliff: while usage is past half the context budget, each pass
drops the oldest turns whose user prompt AND assistant answer are both
free-tier junk (verbatim dups, boilerplate, re-fetchable reads) — never
the newest turn, never a half-flagged turn, zero LLM calls. Measured on
the repeated-prompt E2E (`tests/chat_compact_e2e.sh`): the gate frees the
junk a turn at a time, keeps usage below the 70% cliff, and makes exactly
one provider request per turn; with `SOFUU_NO_ML=1` the cliff alone still
fires exactly as before. Compaction acceptance bar (held-out families):
test accuracy 1.000, precision 1.000, recall 1.000 (8,201 params,
threshold 0.41 = margin midpoint); a deterministic dedupe rule (cosine
≥ 0.95) sits below the model so exact repeats are caught even when the
recent window is full of similar junk.

The agent uses the relevance model as a pre-retrieval advisor over the
recalled-memory menu: before the turn it scores the candidates against the
task and renders ONE `[relevance] … likely on-task: …; likely tangential: …
Advisory only` notice on the ephemeral context message — nothing is
filtered, capped, or dropped; the LLM decides what to rely on. Two
deterministic rules sit below the model (a near-verbatim repeat of an
already-kept candidate, and the never-use class). Relevance acceptance bar
(held-out construction families): test accuracy 0.962, precision 1.000,
recall 0.924 (8,617 params, threshold 0.95) — beating logistic regression
(0.934) on identical features; the model leans on BM25 over the candidate
set plus a 5-char stem channel to catch morphological echoes
(migrates↔migration) that a flat cosine/overlap threshold misses.

The agent runs the supervisor as a per-call pre-check AND a loop-boundary
checkpoint: before each tool call, `check()` layers the trained net over
the certain rules (exact repeat calls, re-reads of unchanged files) and the
nudge rides the tool result in-band (`[supervisor: …]`); at each loop
boundary the `__loop__` pseudo-action asks whether the RUN itself is
spinning, stalled, or over budget (one notice per class per run — no
nagging). A write invalidates prior reads of that file, so a re-read after
an edit is never flagged. Supervisor acceptance bar (held-out construction
families): test accuracy 1.000, precision 1.000, recall 1.000 (8,201
params, threshold 0.82 = margin midpoint of the val fold) — beating
logistic regression (0.945) on identical features.

Online learning (§13) is the escape hatch for messiness the synthetic
training set cannot foresee — and it is OFF by default. `/ml learn` adapts
ONLY the supervisor's output layer from labeled runtime examples, under six
guardrails: confidence-weighted labels (the agent's waste proxy is a
4-band scheme — errored → wasted conf 1.0, tiny result → wasted conf 0.6,
large result → clean conf 0.8, and ambiguous 80–319-char results ABSTAIN
rather than send a coin-flip label; explicit `/ml wrong` / `/ml wasted`
gold labels weigh 2.0), a trust region (‖Δw‖ ≤ 0.25·‖w₀‖, projected),
replay anchors (8 canonical fixtures in every batch), a batch floor (16
examples or it refuses), an adoption gate over 31 fixtures (the 8 anchors
plus 23 margin-filtered representatives of the training families, baked
into the binary — the candidate passes only if every one stays correct at
the baked threshold), and two-step adoption: `/ml learn` holds the
candidate PENDING in memory (never persisted unadopted) until `/ml adopt`
applies + persists it or `/ml discard` drops it. Adopted deltas persist
separately (`~/.sofuu/ml/supervisor_online.f32`), keyed to the pretrained
blob's hash so a re-bake invalidates them. `/ml reset` clears the
adaptation, `/ml info` reports threshold + online state + pending flag +
working set.

The 5th model, `alloc` (§21), is the model-aware config allocator +
pre-flight guard for the context window — it fixes the class of failures
where fixed allocations (one budget ratio, one compaction cliff, one tool
cap for every model) send a request the selected model cannot accept. It
fetches the selected model's details FIRST — the capability registry
(context window, max output, thinking kind), then limits learned from the
provider's own error messages, then conservative defaults — and allocates
against them. Three layers: a mechanical output guard on the wire (outgoing
max_tokens is always clamped to the model's cap; the unknown-model fallback
is a conservative 4,096, not a fixed 71,680); a pre-flight fit check at
every request-build point (agent tool loop, toolless answer path, chat
driver, RLM) with a corrective ladder — re-cap tool results, clamp the
output reserve, drop oldest plain messages, truncate the largest — so the
request that goes out always fits, each correction printed as an `allocgate`
line; and error learning — if a provider still 400s with a limit error, the
real limit is parsed from the message, cached per model, and the turn
retries once per kind. On top sits a tiny net (24→96→40→1, 6,321 params,
threshold 0.51; test accuracy 0.986 vs majority 0.810 / logistic regression
0.969, recall 0.994) producing one scalar — context pressure — that a
deterministic clamped policy turns into the per-turn allocation: when to
compact (0.70 of window slack → 0.50 tight), how much tool result to keep,
recall and attachment budgets, and the output reserve. Advise-only within
the mechanical bounds; `/ml off` restores the old fixed ratios exactly.

### `sofuu.web` — Web Search & Page Reading

Provider-neutral web access over the runtime's own `fetch` — **no API key
required** by default.

```js
// Search — duckduckgo engine (default) needs NO key
const r = await sofuu.web.search("rust vs zig 2026", { count: 8 });
r.results;   // [{ title, url, snippet }]
r.engine;    // "duckduckgo"

// Optional API engines — better reliability, built for LLM consumers:
//   engine: "brave"   → needs BRAVE_API_KEY (or opts.api_key)
//   engine: "tavily"  → needs TAVILY_API_KEY (or opts.api_key)
// SOFUU_WEB_ENGINE selects a default; SOFUU_WEB_ENDPOINT overrides the
// duckduckgo endpoint (proxies / self-hosting).

// Read a page as clean text (scripts/styles/tags stripped, ~20k cap)
const page = await sofuu.web.open("https://example.com/article");
page.text;
```

Agents and chat get these as the built-in `web_search` / `web_open` tools
automatically. The keyless endpoint is best-effort scraping — if it
rate-limits, switch `SOFUU_WEB_ENGINE=brave|tavily` with a key.

### `sofuu.mcp` — Model Context Protocol

```js
// Connect your script to any MCP-compatible tool server
const client = await sofuu.mcp.connect("npx @modelcontextprotocol/server-filesystem /tmp");
const tools   = await client.listTools();
const result  = await client.call("read_file", { path: "/tmp/notes.txt" });
client.disconnect();

// Build your own MCP tool server to expose tools to AI agents
const server = sofuu.mcp.serve();

server.tool("get_weather", {
  description: "Returns the weather for a given city.",
  schema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
}, async ({ city }) => {
  const res  = await fetch(`https://wttr.in/${city}?format=3`);
  return await res.text();
});

server.start(); // Listens over stdio — works with Claude, any MCP client
```

### `sofuu.http` — HTTP Server & Client

```js
// Native HTTP server — no Express or other frameworks needed
const server = sofuu.http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok" }));
});
server.listen(8080, "0.0.0.0");

// Standard fetch API
const res  = await fetch("https://api.example.com/data");
const json = await res.json();

// Streaming bodies — Response.body is a real async iterator of Uint8Array
// chunks (headers/status resolve on the first byte, body streams in)
const res2 = await fetch("https://api.example.com/events");
for await (const chunk of res2.body) {
  process.stdout.write(new TextDecoder().decode(chunk));
}
```

### `sofuu.fs` — File System

```js
const text    = await sofuu.fs.readFile("./data.json", "utf8");
await sofuu.fs.writeFile("./output.txt", "Hello from Sofuu!");
const entries = await sofuu.fs.readdir("./src");
await sofuu.fs.mkdir("./out", { recursive: true });  // recursive parent creation
await sofuu.fs.rm("./out/tmp.txt");                  // file (or empty dir)
const bytes   = await sofuu.fs.readFileBytes("./img.bin");  // → Uint8Array
```

### `sofuu.spawn` — OS Processes

```js
const proc = sofuu.spawn("python3", ["process_data.py"]);
proc.stdout.on("data", (chunk) => console.log(chunk));
await proc.wait();
```

### Globals

```js
// Standard globals available everywhere — no imports needed
setTimeout(() => console.log("done"), 1000);
setInterval(() => console.log("tick"), 500);
await sleep(500);             // Native non-blocking sleep

process.env.API_KEY           // Read environment variables
process.argv                  // CLI argument array
process.cwd()                 // Current working directory
process.exit(0)               // Exit the process

console.log / warn / error    // Full console support
```

---

## Architecture

**Rust** — the runtime shell, engine lifecycle, module loader, chat UI, REPL,
CLI, and all security-sensitive parsing. The only remaining C is the **engine
layer**: **QuickJS** (ES2023 JS engine), **libuv** (event loop), **SIMD**
kernels (NEON/AVX2), the **vendored http-parser**, and the proprietary **QTSQ**
codec — all reached through a thin FFI layer (`crates/sofuu-ffi`).

```
sofuu run agent.ts
       ↓
  [Rust core]             — crates/sofuu-core: CLI, chat (rt/tui.rs), config,
                            REPL (rt/engine.rs), ESM module loader, memory
                            (CMA, KV page store, agent prefetch), SSE, MCP
                            JSON-RPC, TS stripper, bundler, npm safety,
                            agents + web search (shipped JS: src/js/),
                            chat features (brain pin/recall, cost tracking,
                            second-model verify, fs watcher, hooks.js
                            middleware, ghost completion, @file mentions,
                            session resume, brain share/import, brain server),
                            async spine + leaf I/O (rt/ + modules/:
                            loop, promises, timers, fs, subprocess, console,
                            process incl. TTY readline, AI client, HTTP client
                            + server, engine lifecycle — rt/engine.rs)
       ↓
  [sofuu-ffi]             — safe bindings to the C engine layer (all `unsafe`
                            lives here); QuickJS value construction helpers,
                            libuv handle/request size shims, curl multi bridge,
                            QTSQ session store
       ↓
  [QuickJS engine]        — ES2023, native ESM module support  (C)
       ↓
  [libuv event loop]      — non-blocking I/O, timers, subprocess  (C)
       ↓
  C engine layer (via FFI):
    ├── QuickJS (src/ not used — compiled from deps/quickjs/)
    ├── libuv (deps/libuv — prebuilt)
    ├── SIMD kernels (src/simd/{neon,avx}.c — called by rt/ai.rs)
    ├── http-parser (deps/http-parser — vendored, optional M11 swap)
    └── QTSQ codec (external libqtsq.a — brain/KV persistence)
```

The `src/` directory contains **only** `simd/` (3 files, 181 lines of NEON/AVX
intrinsics) and three shipped JS drivers — `js/rlm.js` (RLM), `js/agent.js`
(the agent runtime) and `js/web.js` (web search) — data assets eval'd once per
engine context by the `shipped.rs` seam. Every line of glue, logic, and
binding — `engine`, `tui`, `repl`, `process`, `http`, `mcp`, `npm`, `memory`,
`io` — is Rust (~30k lines in `crates/`).

Rust-ported modules (in `crates/sofuu-core/`): `memory` (CMA — HNSW, decay,
consolidation, dream, gravity, resonance), `http::sse`, `mcp::jsonrpc`
(builders + inbound parse), `ts`, `bundler`, `npm` (spec validation, SHA-1,
safe tar extraction, walk-up resolve), `rt::ai` (sofuu.ai — LLM
streaming/completions/embeddings over curl+libuv, the offline TF-IDF
embedLocal, and SIMD-backed vector math; kernels stay C), `rt::memory`
(sofuu.memory/kv/agent JS shells + the QTSQ brain/KV adapter). All C
duplicates (`memory/hnsw.c`, `memory/dream.c`, `ts/stripper.c`,
`bundler/bundler.c`, `engine/engine.c`, `io/tui.c`, `repl/repl.c`,
`sofuu.c`, `ffi_shim.c`) are deleted; `make c-only` is removed.

---

## Build from Source

```bash
# Requirements: Rust (cargo), clang, make. libcurl for HTTP features.
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu

export SOFUU_QTSQ_DIR=/path/to/black-hole-disk   # the encrypted-brain codec
make                 # cargo build --release (build.rs compiles QuickJS+SIMD+http-parser)
make test            # JS e2e suite
make size-check      # fails if the binary exceeds 5MB
./sofuu doctor       # confirms QTSQ linkage + a real brain round-trip
make install         # copies the binary to /usr/local/bin
```

> **Why `SOFUU_QTSQ_DIR` is required:** the encrypted brain is the product, and
> a build without the QTSQ codec is *silently* broken — memory calls no-op and
> every session persist fails while the UI still claims persistence. A release
> tarball built that way shipped once (2026-09-22). `make` now refuses it
> unless you pass `SOFUU_ALLOW_NO_QTSQ=1` (what CI does), and
> `sofuu doctor` catches it in one command.

### The gates

| Command | What it proves |
|---|---|
| `cargo test --release` | 395 `sofuu-core` unit tests **and** the 36 `sofuu-capi` tests (funnel, memory handles, streaming, cancel, multi-instance, the doc anti-drift test) |
| `make test` | JS e2e/parity suite — 46 passed / 0 failed / 1 skipped (the skip is an opt-in live test, `SOFUU_LIVE_TEST=1`) |
| `make headless-test` | compiles **and runs** all 5 C samples against `libsofuu`; every sample asserts, so a broken SDK fails the build |
| `make abi-check` | exported C symbols still match `scripts/abi_symbols.txt` (a new symbol must be acknowledged) |
| `make size-check` | 5 MB cap (currently 3.2 MB) |
| `./sofuu doctor` | QTSQ is linked, a real write→flush→reopen→recall round-trips, caps resolve |
| `ruby scripts/check_podspec.rb` | the CocoaPods manifest is valid |
| `swift package dump-package` | the SwiftPM manifest parses |
| `xcrun swiftc -typecheck` | the typed Swift layer still matches `sofuu_embed.h` |

Other suites: `./sofuu run tests/agent_test.js` (agents + web E2E over a
scripted mock provider and real MCP child servers), `tests/rlm_mock_test.js`
(RLM, network-free), `tests/verify_memory_test.js` (CMA brain: recall, dedup,
entity upsert, persistence, decay, consolidation), `tests/long_horizon_test.js`
(endurance: 429 storms, mid-answer transport cuts, 25-round tool loops,
sub-agent budget inheritance, provider-outage salvage).

- The C engine layer is compiled by `crates/sofuu-ffi/build.rs`: QuickJS
  (`deps/quickjs/`), SIMD kernels (`src/simd/`), vendored http-parser
  (`deps/http-parser/`), and prebuilt libuv (`deps/libuv/build/libuv.a` —
  build with `cmake -S deps/libuv -B deps/libuv/build && cmake --build deps/libuv/build`).
- Cross-compilation: `make linux-x86_64` / `make linux-arm64` (requires Zig —
  `make zig-install` once).

**Embedding the runtime** (not just the CLI): see [Embed Sofuu](#embed-sofuu-libsofuu),
[`docs/EMBEDDING.md`](docs/EMBEDDING.md) (the contract) and
[`docs/EMBEDDING-DIST.md`](docs/EMBEDDING-DIST.md) (artifacts + install).
Platform packs: `make dist-macos` · `dist-ios` (xcframework) ·
`dist-android` (per-ABI `.so`). These build libuv/curl with CMake + Ninja, so
they need `brew install cmake ninja curl` first (CI installs all three); a
plain `make` only needs a Rust toolchain.

---

## Documentation map

| I want to… | Read |
|---|---|
| Install and run the CLI | this README · [sofuu.xyz/docs](https://sofuu.xyz/docs) |
| Put the runtime inside my app | [`docs/EMBEDDING.md`](docs/EMBEDDING.md) (contract) · [`docs/EMBEDDING-DIST.md`](docs/EMBEDDING-DIST.md) (install) |
| Understand the architecture | [`CONTEXT.md`](CONTEXT.md) · [Architecture](#architecture) above |
| See what's done and what isn't | [`TASKS.md`](TASKS.md) (status board) · [`ROADMAP.md`](ROADMAP.md) |
| Read a design decision | `PLAN-*.md` at the repo root (indexed in [ROADMAP.md](ROADMAP.md)) |
| Reproduce the embedding numbers | [`docs/EMBEDDING.md`](docs/EMBEDDING.md) §13 + `scripts/bench/fetch_datasets.sh` |
| Check a build is healthy | `./sofuu doctor` |

Design plans worth knowing about: [`PLAN-HEADLESS.md`](PLAN-HEADLESS.md)
(the embeddable SDK) · [`PLAN-MULTIMODAL-EMBEDDINGS.md`](PLAN-MULTIMODAL-EMBEDDINGS.md)
(embeddings/image/voice) · [`PLAN-ML-GATES.md`](PLAN-ML-GATES.md) (the five
context-economy models) · [`PLAN-POSITIONING-2026.md`](PLAN-POSITIONING-2026.md)
(why the SDK is the product).

---

## License

The main execution runtime of Sofuu is open-source and licensed under the **MIT License**.

This means you can use, modify, and distribute it freely for both commercial and non-commercial projects.

> ⚠️ **IMPORTANT NOTE ON QTSQ FORMAT:**
> The MIT License applies **only** to the execution runtime codebase of Sofuu. It DOES NOT apply to the design, mathematics, or specification of the **QTSQ tensor format**, nor to the associated QTSQ quantization models. The QTSQ mathematical format and specifications remain proprietary. 
> 
> Please refer to [LICENSES/QTSQ-FORMAT.txt](LICENSES/QTSQ-FORMAT.txt) for specific licensing terms regarding the QTSQ infrastructure.

See [LICENSE](LICENSE) for the full terms, and [`docs/OPEN-CORE.md`](docs/OPEN-CORE.md) for the open-core boundary (what's MIT vs. commercial).

© 2026 Priyanshu Boruah
