# ⚡ Sofuu (素風)

> *A fast, private JavaScript runtime built in **Rust** (shell, parsers, memory logic) with a **C** core (QuickJS engine, libuv event loop, SIMD kernels, QTSQ codec) — designed from day one for AI-era workloads.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0--beta-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

> ⚠️ **Beta software.** Sofuu is under active development. APIs may change before a stable release.

---

## What is Sofuu?

Sofuu is a **server-side JavaScript runtime** — the kind of thing you'd use to run your AI backend, your API server, or your edge automation script. It has no browser API, no DOM, and no renderer.

What makes it different from Node.js, Deno, or Bun is that all the things you normally have to install separately — LLM streaming, vector math, MCP client/server, HTTP — are just built in at the C level. No npm packages needed for any of it.

You build your frontend (React, Next.js, plain HTML — whatever you like) completely separately. Sofuu runs the server side.

---

## Install

```bash
curl -fsSL https://sofuu.xyz/install | sh
```

Or build it yourself from source:

```bash
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
make
./sofuu version
```

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

| | Node.js | Deno | Bun | **Sofuu** |
|---|---|---|---|---|
| Binary size | ~100 MB | ~80 MB | ~60 MB | **~2 MB** |
| Startup time | ~50 ms | ~30 ms | ~7 ms | **~3 ms** |
| Language | C++ | Rust | Zig | **Rust-first + C (low-level)** |
| LLM streaming | npm install | npm install | npm install | **Built in** |
| MCP client + server | npm install | npm install | npm install | **Built in** |
| SIMD vector math | npm install | npm install | npm install | **Built in (NEON/AVX2)** |
| TypeScript support | Separate compiler | Built in | Built in | **Built in (Rust stripper)** |
| AI agents + sub-agents | npm install | npm install | npm install | **Built in (`sofuu.agent`)** |
| Web search for agents | npm install | npm install | npm install | **Built in (`sofuu.web`, keyless default)** |
| Memory safety of shell | ❌ | ✅ | ✅ | **✅ (Rust shell)** |

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
sofuu version                # Print version information
sofuu help                   # Print usage help
```

**Chat slash commands** (inside `sofuu` / `sofuu chat`):

```text
/help  /models  /providers  /model <name>  /provider <name>
/effort <lvl>  /compact  /clear  /brain [on|off]  /rlm [on|off|auto]
/tools  /agents  /version  /exit

# Brain & memory
/remember <fact>   # Pin a fact to the brain (survives decay)
/why               # Show which memories shaped the last answer
/share [path]      # Export brain as a portable card
/import <path>     # Import a brain card from someone else
/cost              # Token usage + spend breakdown + budget
/verify <provider> # Second-model verification pass with diff
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

**Second-model verification** (`/verify <provider>`): after the final answer,
the chat runs a second completion against a different provider and reports a
word-agreement diff (>95% = ✓ verified; mismatches are surfaced inline).

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
real max output instead of a flat constant. Explicit `/ctx` and `/maxout`
overrides always win.

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

### Embed Sofuu (libsofuu C ABI)

Sofuu ships as an embeddable library (`libsofuu`) for any host — iOS, Android,
desktop, server, edge. QuickJS is a pure interpreter (no JIT), so iOS App Store
executable-memory policy is satisfied by design.

```bash
make libsofuu       # builds dist/libsofuu.{a,dylib} + dist/sofuu_embed.h
make headless-test  # compiles + runs c_embed.c and c_rlm.c against libsofuu
make abi-check      # verifies the exported symbol surface
```

**C (macOS / Linux):**

```c
#include "sofuu_embed.h"
int main(void) {
    SofuuRuntime *rt = sofuu_rt_new(NULL);
    char *out = NULL;
    sofuu_rt_eval(rt, "1 + 1", &out);   // → {"ok":true,"result":2}
    sofuu_free(out);
    sofuu_rt_free(rt);
}
```

See `examples/headless/c_embed.c` (eval + funnel), `c_rlm.c` (RLM long-context
Q&A), and `c_agent.c` (agent define/run/stream/cancel) for complete working examples.

**iOS (Swift):**

```swift
let sofuu = SofuuBridge()
sofuu.eval("1 + 1")                    // → {"ok":true,"result":2}
sofuu.embedLocal("hello")             // → [Float] (768-dim, no network)
```

See `examples/headless/SwiftSample/SofuuBridge.swift` for the full wrapper.

**Android (Kotlin/JNI):**

```kotlin
val sofuu = SofuuBridge()
sofuu.eval("1 + 1")                   // → {"ok":true,"result":2}
sofuu.embedLocal("hello")             // → FloatArray (768-dim, no network)
```

See `examples/headless/KotlinSample/` for the wrapper + JNI bridge + CMake config.

Platform packs: `make dist-macos` (arm64+x86_64), `make dist-linux` (x86_64+arm64
musl), `make dist-ios` (xcframework), `make dist-android` (per-ABI .so). See
[`dist/README.md`](dist/README.md) and [`docs/EMBEDDING.md`](docs/EMBEDDING.md)
for the full embedding contract. Design + status: [`PLAN-HEADLESS.md`](PLAN-HEADLESS.md).

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
# Requirements: Rust (cargo), clang, make. libcurl is needed for HTTP features.
git clone https://github.com/sofuu-runtime/sofuu
cd sofuu
make                 # cargo build --release (build.rs compiles QuickJS+SIMD+http-parser)
make size-check      # fails if binary exceeds 5MB
make install         # copies binary to /usr/local/bin
```

- `cargo test` — runs the Rust unit tests (memory, SSE, JSON-RPC, TS,
  bundler, npm, rt loop/promises/timer/fs/spawn, console, MCP, RLM, engine,
  shipped drivers, chat features — 177 tests).
- `./sofuu run examples/agent_test.js` — agents + web E2E battery over a
  scripted mock provider and real MCP child servers (46 checks).
- `./sofuu run examples/rlm_mock_test.js` — RLM E2E, network-free.
- `make test` — JS parity test suite (priority1/priority2/ts_test/simd_test).
- `make size-check` — fails if binary exceeds 5MB (current: 2.0MB).
- The C engine layer is compiled by `crates/sofuu-ffi/build.rs`: QuickJS
  (from `deps/quickjs/`), SIMD kernels (`src/simd/`), vendored http-parser
  (`deps/http-parser/`), and prebuilt libuv (`deps/libuv/build/libuv.a` —
  build with `cmake -S deps/libuv -B deps/libuv/build && cmake --build deps/libuv/build`).
- Cross-compilation: `make linux-x86_64` or `make linux-arm64` (requires Zig —
  `make zig-install` once; builds via cargo + zig as linker).

---

## License

The main execution runtime of Sofuu is open-source and licensed under the **MIT License**.

This means you can use, modify, and distribute it freely for both commercial and non-commercial projects.

> ⚠️ **IMPORTANT NOTE ON QTSQ FORMAT:**
> The MIT License applies **only** to the execution runtime codebase of Sofuu. It DOES NOT apply to the design, mathematics, or specification of the **QTSQ tensor format**, nor to the associated QTSQ quantization models. The QTSQ mathematical format and specifications remain proprietary. 
> 
> Please refer to [LICENSES/QTSQ-FORMAT.txt](LICENSES/QTSQ-FORMAT.txt) for specific licensing terms regarding the QTSQ infrastructure.

See [LICENSE](LICENSE) for the full terms.

© 2026 Priyanshu Boruah
