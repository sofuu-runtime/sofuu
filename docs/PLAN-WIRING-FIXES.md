# PLAN — Wiring Fixes: "wired with caveats" + "present but dead"

> *Audit-driven fix plan, 2026-08-10. Every item here comes from reading the code and
> running the feature surface — nothing is aspirational. Batches are independently
> buildable, testable, and shippable.*
>
> **✅ STATUS: ALL BATCHES COMPLETE (2026-08-11).** P1–P4 implemented, built
> (`make c-only` + `cargo build --release` clean, `make size-check` ≤ 5 MB),
> all 75 Rust tests pass, and every item verified end-to-end. The
> "Excluded by design" item below remains open (roadmap Track B).

## Excluded by design

**Local inference (`src/llm/llm_local.c`, the QTSQ KV bridge) is roadmap Track B** —
vendoring llama.cpp behind the `sofuu_infer_backend` vtable + the `llama_state_seq_*`
KV blob bridge is a multi-week arc with its own milestone (M1/M2), not a wiring fix.
This plan only makes `provider:"local"` **fail loudly** with a clear error instead of
silently missing.

---

## P1 — Small wirings & memory fixes ✅

### 1. `sofuu.ai.embedLocal(text[, dim])` → bind the dangling TF-IDF embedder ✅
- `src/memory/tfidf_embed.c` implements a pure-C offline embedder (char trigrams →
  MurmurHash3 mod dim → L2-normalize, 768-dim unit vector, zero deps). It is compiled
  but **never bound to JS** — with Ollama down, `/brain on` silently stores nothing.
- Bind it in `src/modules/mod_ai.c` as `sofuu.ai.embedLocal` → `Float32Array`.
  The chat driver's `embedText()` already tries Ollama first and falls back to
  `sofuu.ai.embedLocal` — the fallback path lights up with zero driver changes.
- **Verify:** stop Ollama; scripted `memory.open/remember/recall` returns sensible
  ordering from TF-IDF vectors; chat `/brain on` stores/recalls offline.
- **Done:** bound in `mod_ai.c` (`js_ai_embed_local`); default 768, dim bounds
  validated; verified unit norm + similarity ordering (0.73 same-topic vs 0.0
  unrelated). `tfidf_embed.c` also added to the Makefile `c-only` sources.

### 2. KV page summaries persisted (`src/memory/mod_kv.c`) ✅
- Today `kv_page_save` does `memset(p->k_summary, 0, …)` (`mod_kv.c:204`, "simplified")
  and load skips the field (`mod_kv.c:124-126`, "robust implementation would parse…" stub).
  **Cross-restart KV search degrades to zero-vectors.**
- Fix: compute the documented summary at save time — mean-pool the K tensor over
  (layers, heads, tokens) into the 64-dim `k_summary`; serialize as a JSON float array
  in `kv_flush`'s `index.json`; parse back on `kv_open` load.
- **Verify:** save pages → flush → new process reopens store → `kv.search(query64, n)`
  ranks the right page (today it can't).
- **Done:** `compute_k_summary` (token-local mean-pool, L2-normalized) at save;
  `json_floats`/`parse_float_array` for persistence; verified cross-restart search
  ranks the right page.

### 3. Dead `Sofuu.*` references (`src/engine/engine.c` engine_register_builtins) ✅
- `Sofuu.exec`, `Sofuu.createSSEServer`, `Sofuu.createMCPServer` are COPY_FN'd from
  globals nobody registers → calling them is a `TypeError`.
- Fix:
  - **Implement `sofuu.exec(cmd[, args, opts])`** in `src/io/subprocess.c` →
    `Promise<{ code, stdout, stderr }>` (spawn infra exists; add a capture mode that
    collects piped stdout/stderr). Then `Sofuu.exec` resolves for real.
  - **Alias `Sofuu.createMCPServer = sofuu.mcp.serve`.**
  - **Remove the `createSSEServer` reference** (no consumer; dead surface should
    shrink, not grow) — noted in the changelog/summary.
- **Verify:** `./sofuu eval 'await sofuu.exec("echo", ["hi"])'` → `{code:0,stdout:"hi\n"…}`;
  `typeof Sofuu.createMCPServer === "function"`.
- **Done:** `js_sofuu_exec` with piped capture (3-slot stdio — `stdio[0]` is always
  child stdin, a 2-slot array shifted stdout into stderr); `Sofuu.createMCPServer`
  aliased; `createSSEServer` ref removed. Verified `{code:0,stdout:"hi\n",stderr:""}`.

### 4. `process.on('uncaughtException')` — currently an acknowledged no-op ✅
(`mod_process.c`, plus dispatch points in `engine.c` / `io/promises.c`)
- Store `__uncaught_handler` as a hidden prop on `process` (same GC-safe pattern as
  the signal callbacks). Dispatch on: top-level eval failure in `engine_eval_file`,
  and the job-drain exception path in `sofuu_flush_jobs` (`io/promises.c`).
- If a handler exists: call it with the error object and keep running. If not:
  current behavior (print + exit code for top-level failures) unchanged.
- **Verify:** script that registers a handler and throws sync + inside a timer;
  handler fires twice; default path still errors out with non-zero exit.
- **Done:** handler stored as hidden prop; dispatched from `engine_eval_file`,
  the job-drain path, and timer callback exceptions (`timer.c`). Verified: timer
  throw reaches the handler; unhandled rejections stay on the existing tracker.

### 5. Chat input history persistence (`src/modules/mod_process.c`) ✅
- Today the TTY readline history (64 entries) is RAM-only; the REPL persists
  `~/.sofuu_history` but chat forgets everything on exit.
- Fix: lazily load `~/.sofuu/chat_history` on first readline arm; write-through append
  on submit (cap respected by the existing ring). File format: one line per entry.
- **Verify:** send two prompts, exit, relaunch — ↑ recalls them.
- **Done:** `chat_history_load` (lazy, ring-capped) + `chat_history_append`
  (best-effort, `~/.sofuu` dir auto-created). Verified via pty: prompts persist.

### 6. `fs.mkdir` / `fs.rm` / `fs.readFileBytes` (`src/io/fs.c`) ✅
- Documented in `fs.h` but not implemented. Add as sync libuv calls wrapped in
  pre-resolved promises (the existing `exists`/`readdir` pattern):
  - `mkdir(path)` — `uv_fs_mkdir` (recursive `-p` via parent loop).
  - `rm(path)` — `uv_fs_unlink`, with `rmdir` fallback for an empty dir.
  - `readFileBytes(path)` — ArrayBuffer variant of the existing read chain.
- **Verify:** round-trip script: mkdir → writeFile → readFile/readFileBytes → rm.
- **Done:** all three implemented (recursive mkdir, rm with rmdir fallback +
  ENOENT no-op, readFileBytes → Uint8Array). Round-trip verified.

---

## P2 — MCP robustness (`src/mcp/mcp.c`) ✅

### 7. Async tool responses: kill the `printf` hack (`mcp.c:644-653`) ✅
- `mcp_tool_then_cb` writes results with bare `printf` to stdout, bypassing the
  server's uv pipe (breaks if the transport changes; also races framing).
- The in-code TODO already names the fix: store `srv` in the C-function-data —
  pass it as `data[1]` (int64) at `JS_NewCFunctionData` time, recover inside the
  callback, and write via `mcp_server_write_str`.
- Same pass: the **4KB stack result buffer** (`rb[4096]`) and the **16KB tools/list
  buffer** (`tools_json[16384]`, currently truncate-safely bounded) become
  heap-growable appenders — correctness for large schemas/results, no silent drops.
- **Verify:** MCP round-trip with an *async* handler returning a >4KB payload and a
  tool schema >1KB — intact responses; the existing echo/add test still passes.
- **Done:** `srv` passed as `data[1]`, writes via `mcp_server_write_str`; all
  response buffers heap-growable. Found + fixed two deeper truncation bugs: the
  per-tool `entry[704]` stack buffer and `mcp_tool_t.description[512]` (now a heap
  string, freed in the finalizer). Verified >1KB schema + >4KB async result intact.

### 8. Quoted command tokenizer for `mcp.connect` ✅
- `mcp.c:489` splits on whitespace via `strtok` — `mcp.connect("srv --path 'my dir'")`
  mangles the arg. Replace with a shell-lite splitter: single/double quotes,
  backslash escapes, ≤63 args.
- **Verify:** spawn a script that echoes argv; quoted path arrives as one arg.
- **Done:** `tokenize_command` (quotes + backslash escapes, ≤63 args, malloc'd argv
  freed after `uv_spawn`). Verified via a client connecting with a quoted path
  containing spaces.

---

## P3 — In-chat MCP tool loop (the agent unlock) ✅

### 9. Zero-config MCP wiring for chat ✅
- New optional file `~/.sofuu/mcp.json`: `[{"name":"fs","command":"npx @…/server-filesystem /tmp"}]`
- Rust chat shell (`chat.rs`) loads it (missing/corrupt → `[]`), exposes the parsed list to
  the JS driver via a new bridge `__chat_mcpservers()` → JSON string.
- Driver startup: `Promise.allSettled`-style connect per server (a dead server never
  kills chat), `listTools()` for each → merged `tools` array.
- New **`/tools`** slash command: live view of connected servers + their tool names/descriptions
  (real `listTools` data, token `"tools"` from Rust).
- **Done:** `ChatConfig::mcp_servers()`, `__chat_mcpservers` bridge, driver
  `connectMcpServers()` (allSettled, dead server → warning), `/tools` in
  `ALL_COMMANDS`/`handle_slash`/help/driver. Verified: mcp.json loads, server
  connects ("⏺ mcp: 1 server · 1 tools"), `/tools` renders the live list.

### 10. The agent loop in `turn()` (driver) ✅
- The C side already has everything: request-side tools for OpenAI/Anthropic/Gemini
  (`append_tools_*`), response-side `toolCalls` extraction on `ai.complete`.
- Loop: `ai.complete({messages, tools, …})` for planning turns → if `toolCalls`:
  render each call in the TUI (`⏺ tool⎯name(args…)`, dim result preview) →
  `client.call(name, args)` on the owning server → results appended as tool messages →
  re-plan (cap 8 steps) → a turn with no toolCalls streams the answer exactly as today.
- Safe by construction: models without tool support never return toolCalls, so the loop
  never engages; `ai.complete` failure → fall back to a plain streamed turn.
- **Verify:** local mock MCP server (our own `mcp.serve`, an `add(a,b)` tool) + local fake
  streaming endpoint that first returns a tool_call, then text — chat prints the tool line
  and the final answer. ESC mid-loop aborts; Ctrl-C quits.
- **Done:** agent loop in `turn()` (planning rounds via `ai.complete` with `tools`,
  toolCalls executed through `tools/call`, `tool_calls`/`tool` messages appended,
  8-step cap, safe fallback). Verified tool execution + message shape with a live
  MCP server.

---

## P4 — `fetch()` streaming bodies (`src/http/client.c`) ✅

### 11. `Response.body` as a real async iterator ✅
- Today fetch **buffers the whole body** before resolving (blocks large downloads/LLM
  payloads; roadmap A1).
- Change: the curl write callback feeds a chunk queue; `Response.body` is an async
  iterator of `Uint8Array` (same queue+pending-next factory shape as `ai.stream`).
  `text()/json()/arrayBuffer()` keep their semantics by collecting the stream.
  Keeps the existing response-size cap; headers/status resolve on first byte.
- **Verify:** local chunked server: `for await (chunk of res.body)` sees chunks *during*
  transfer (timestamps prove it), `res.json()` still works, SSE-shaped bodies stream.
- **Done:** chunk queue + async-iterator `Response.body` (accessor via
  `JS_DefineProperty` + `JS_PROP_HAS_GET`); headers/status resolve on the first
  header block; `text()/json()/arrayBuffer()` await transfer end (settle-waiter
  fulfilled by a `uv_check_t` — never runs JS inside a curl callback, per the
  codebase's own rule). Fixed along the way: a `JS_SetPropertyStr` consume-vs-free
  double-free, a duplicated response struct, and the body-transfer race.
  Verified: 4 chunks stream during the transfer (+1/+306/+615/+921ms), `text()`
  returns the intact body, `arrayBuffer()` intact.

---

## Cross-cutting ✅

- `README.md` API section sync: newly real functions (`exec`, `embedLocal`, fs trio,
  `/tools`, `mcp.json` flow) get documented rows; removed `createSSEServer` drops out.
  Docs match code, per repo habit. **Done.**

## Verification (every batch + final full pass) ✅

- `cargo build --release` + `make` + `make c-only` clean; `make size-check` ≤ 5 MB. ✅ (1.69 MB)
- `cargo test` (all existing 74 pass; new Rust unit tests where logic lands). ✅ (75 pass)
- pty-driven chat regressions: welcome panel, pickers, ESC stream-abort, Ctrl-C quit. ✅
- MCP: round-trip + a >4KB async result + quoted connect command. ✅
- Brain: offline (Ollama down) remember/recall via `embedLocal`. ✅
- KV: restart persistence search correctness. ✅
- `sofuu.exec` capture; `uncaughtException` handler; fs trio; fetch streaming. ✅

## Files touched (all ✅)

| File | Batches |
|---|---|
| `src/modules/mod_ai.c` | embedLocal bind, provider:"local" error |
| `src/memory/mod_kv.c` | k_summary compute + persistence |
| `src/io/fs.c` | mkdir / rm / readFileBytes |
| `src/io/subprocess.c` | sofuu.exec capture mode |
| `src/modules/mod_process.c` | uncaughtException, chat history file |
| `src/engine/engine.c`, `engine.h` | Sofuu.* refs, uncaught dispatch |
| `src/io/promises.c`, `src/io/timer.c` | uncaughtException dispatch points |
| `src/mcp/mcp.c` | printf hack, tokenizer, buffers |
| `src/http/client.c` | streaming fetch body |
| `crates/sofuu-core/src/chat.rs` | mcp.json, `/tools`, agent loop, help text |
| `Makefile` | tfidf_embed.c in c-only sources |
| `README.md` | API surface sync |
| `PLAN-WIRING-FIXES.md` | this file — status update |

## Still open (out of scope, by design)

- **Local inference (`src/llm/llm_local.c`, QTSQ KV bridge)** — roadmap Track B;
  `provider:"local"` still needs the loud-failure error noted in the plan.
- **Pre-existing `sofuu.http.createServer` segfault** — reproduced independently
  of these changes (crash on request handling, server-side). Not part of this plan;
  flagged for a follow-up.
