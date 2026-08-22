# Sofuu — Task & Status Board

> *Status as of 2026-08-21. Everything marked ✅ was verified by running it (builds + scripted/pty/end-to-end tests), not by reading code claims.*
>
> Legend: ✅ done & verified · 🟡 in progress · ⬜ not started
>
> Active plan docs (2026-08-21): `PLAN-MEMORY-TOKENS.md` · `PLAN-RUST-MIGRATION.md` · `PLAN-CHAT-FEATURES.md` · `PLAN-RLM.md` · `PLAN-HEADLESS.md` · `PLAN-AGENTS.md` (repo root).

---

## ✅ Completed — 2026-08-22 full audit + P0 fix pass

Full-codebase audit (wiring/dead code, FFI/unsafe correctness, security,
JS-driver logic — see `AUDIT-2026-08-22.md`), the tree committed to git
(e6979e256, first commit of the Rust codebase), and every P0 memory-safety
finding fixed + verified. Verified after fixes: `cargo test` **186 green**
(was 185; +1 new regression test) · `make test` 19 green · size 2,125,632
bytes ≤ 5MB · brain server E2E (401/401/200 auth matrix, /remember,
/recall, CSPRNG token, remote-bind guard) ✓.

### P0 fixes
- ✅ **P0-1 git:** entire Rust implementation was untracked → committed
  (snapshot commit e6979e256: 212 files, +60,244/−11,239). Note: two files
  remain unstaged (the security hook blocks `git add` on them — stage
  manually: the installer script and the memory-verify script).
- ✅ **P0-2 spawn double-free:** failure path freed the process storage
  directly AND via the fence (`spawn_free`). Now all four handles close via
  `spawn_close_cb` (closing=4). Regression test:
  `rt::process_spawn::tests::spawn_failure_and_post_exit_use_are_clean`.
- ✅ **P0-3 mcp.connect UAF:** `uv_close(pipe, None)` + immediate `free` —
  libuv's closing pass touches freed storages. Storages now freed in the
  close callback (`free_pipe_cb`); also fixed `write_str_dup` ignoring the
  `uv_write` rc (per-call leak on EPIPE).
- ✅ **P0-4 NUL over-read:** 8 sites passed the original length to
  QuickJS after `CString::new(...).unwrap_or_default()` collapsed to a
  1-byte buffer (fetch `.text()`, `headers.get`, 6× SSE/JSON parse in
  `rt/ai.rs`). All now use the CString's byte length.
- ✅ **P0-5 brain server was nonfunctional end-to-end:** (a) native HTTP
  server never parsed headers → `req.headers` undefined → every request
  401; now wired (on_header_field/value/headers_complete, 32KB/128-header
  caps, lowercase keys, comma-joined duplicates); (b) auto-token was
  epoch micros (zero entropy) → now 16 bytes from /dev/urandom
  (`sofuu-<32hex>`, persisted 0600); (c) remote-bind guard prefix-sniffed
  `sofuu-` (bypassable with `--token sofuu-anything`) → now a real
  auto-generated flag; (d) auth compare is constant-time (XOR-accumulate
  in the serve script); (e) `/remember` read `req.body` (a String) as a
  chunk stream (O(n²)) → uses it directly; (f) HOME-derived paths
  validated against `..` components.
- ✅ **P0-6 stale subprocess opaque:** the fence freed `SpawnReq` while
  the JS object's opaque still pointed at it (post-exit `.write()`/`.kill()`
  = UAF). `SpawnReq` now holds a dup'd `self_obj`; `spawn_free` nulls the
  opaque before freeing; `kill`/`write` throw `process already exited`.
  Also fixed the finalizer leaking the three callback refs.

### NEW P0s found BY the fix pass (not in the original audit)
- ✅ **argv not NULL-terminated** in `sofuu.spawn`/`sofuu.exec` (execvpe
  walks argv until NULL — EFAULT when the byte after the Vec wasn't zero;
  only ever "worked" by heap-layout luck; exposed deterministically by the
  new regression test). Both sites now push `NULL`.
- ✅ **Failed-spawn process handle left in `loop->handle_queue`:** the
  vendored libuv inits+queues the process handle before spawning and its
  error-dequeue path is `#if 0`'d (deps/libuv/src/unix/process.c:1017) —
  freeing the storage directly leaves a dangling entry that `uv_walk`
  visits at loop close (SIGSEGV, reproduced under lldb). All three
  failure paths (`spawn`, `exec`, `mcp.connect`) now `uv_close` the
  process handle (with `handle->data` set first — the fence reads it).
- ✅ **`sofuu serve --brain` died instantly:** the serve script's async
  IIFE made `var server` function-local → unreachable at resolution → the
  JS finalizer closed the listening socket. Server + brain are now pinned
  to globals. (Deeper runtime footgun noted below.)

### Known-issue notes (documented, not fixed here)
- ⚠️ **JS-object reachability controls native server lifetime** — a server
  created inside a function scope dies when the scope does. Structural
  fix (ref the tcp handle into the loop) is a follow-up.
- ⬜ P0-7 multi-engine teardown on the shared loop (capi multi-runtime
  mode) remains open — architectural, needs per-engine handle tracking.
- ⬜ P1/P2 audit findings (agent budget contract, redirect header leak,
  npm inflate caps, dual brain handles, MCP server leaks, dead
  `ffi_exports` shims…) — full list + fix order in `AUDIT-2026-08-22.md`.

---

## ✅ Completed — audit fixes + remaining chat items (2026-08-21)

Full-codebase audit (memory safety, security, dead code) plus the last four
deferred chat/agent/RLM items. Verified: `cargo test` 180 green · `make test`
19 green · chat E2E 5 green · size 2.1MB ≤ 5MB · headless C samples ✓ ·
ABI symbols match baseline exactly.

### Audit fixes
- ✅ **macOS SIGKILL fix:** stale ad-hoc linker signature killed the binary
  after rebuilds (`make test` was 18× "Killed: 9"); the Makefile now re-signs
  on Darwin (`codesign --force --sign -`).
- ✅ **capi eval-length bug:** `rt_call_and_capture` passed a hardcoded len 22
  for the 24-char `JSON.stringify(__capi_r)` read expression — worked only by
  accident (QuickJS scans to NUL, violating its documented contract). Now
  computed. Dead code removed from capi (`success_envelope`, unused
  `CapiConfig.config`), `CapiRuntime` made pub; stale "let's re-do" comment in
  `sofuu_rt_call_stream` replaced with the real events-only contract.
- ✅ **server_test.js used the non-existent `res.json()`** → handler TypeError,
  no response, fetch stalled the full 120s curl timeout, then the process hung
  FOREVER (listening socket keeps the loop alive; no server.stop/close). Test
  fixed to the documented API (`writeHead`+`end`) and exits explicitly.
- ✅ **@file mention path containment tightened** (chat driver): absolute paths
  outside cwd are rejected (documented contract), not just `..`.
- ✅ **@file mention await bug:** `expandMentions` never awaited
  `sofuu.fs.readFile` — every attachment embedded `[object Promise]`, missing
  files leaked unhandled rejections. Now async + awaited (found by the new E2E).
- ✅ **Dead code removed:** `active_entry_mut` / `provider_env_name` (chat.rs),
  `CURLOPT_HEADER` const (http_client.rs); all cargo dead-store/mut warnings
  cleaned (remaining warnings are intentional C-API-mirroring names).
- ✅ **Suite robustness:** `fetch_test.js` / `priority1_test.js` skip when
  httpbin.org answers upstream 5xx (observed a real 503 outage mid-run).

### The four deferred items — all landed & verified
- ✅ **Per-model context windows** (`src/js/agent.js`): `MODEL_CTX` prefix
  table + read-only `sofuu.agent.contextWindow(model)`; RLM routing uses it
  (def `/ctx` override wins). 10 CTX checks in `tests/agent_test.js`, incl.
  llama3 (8192) routing an oversized turn where flat-32768 would not.
- ✅ **RLM mid-request Esc abort** (`src/js/rlm.js`): asks run stream-first
  (abortable) with complete() fallback; a 100ms watcher kills in-flight ask
  streams when `__rlm_aborted` is set; abort throws fold into
  `stopped:'aborted'` instead of rejecting. Drip probe in
  `tests/rlm_mock_test.js`: abort at 400ms killed the stream at 2/14 chunks;
  mock serves SSE for stream requests (wire fidelity).
- ✅ **`@agent` mentions in chat** (chat driver): `@name <task>` and
  `@agent:name <task>` run the loaded agent directly with its own definition
  (focused run, no chat history), Esc-cancellable via `signal:'chat'`; agents
  win over same-named files, unknown falls through to `@file`; `via <name>`
  summary line; `/at` help + README updated. New
  `tests/chat_mention_e2e.sh` + `tests/mock_llm_server.js` (isolated HOME +
  scripted endpoint): 5 checks ALL PASSED.
- ✅ **Live-provider verification harness:** env-gated
  `tests/live_provider_test.js` registered in `run_js_tests.sh` — runs a real
  completion + tool-using agent run only with `SOFUU_LIVE_TEST=1` + a provider
  key; otherwise SKIP exit 0.

---

## ✅ Completed — PLAN-MEMORY-TOKENS.md (P1–P7, full plan, landed 2026-08-21)

Token efficiency + RAM hygiene across chat/agent/RLM/CMA. Verified: `cargo test`
181 green (3 new units), `agent_test.js` ALL PASSED (13 new checks),
`rlm_mock_test.js` green, `make test` 19/19, size 2.0MB ≤ 5MB.

**Tokens (spend them on the right things):**
- ✅ **P1 minimal system prompt:** one byte-stable core (`sofuu.agent.CORE_PROMPT`, ~25 tokens: identity + directness + tool discipline) shared by the chat def and every agent default; recall/mesh/watch context moved OUT of the system string into an ephemeral user-role message placed after it (never leaks into stored history); RLM driver prompt compressed ~250→~170 tokens (`episode.rs`, contract-preserving unit test asserts every API name/rule).
- ✅ **P2 recall gating:** similarity floor 0.30 (per-def `recallMin`) → dedupe vs last-4 history turns + task → 1024-token budget fill highest-score-first (per-def `recallBudget`). Zero relevant hits = zero injected bytes. Recall events now carry `dropped`/`budgetCut`; chat's `/why` reads exactly what gating let through (was reading a never-populated store).
- ✅ **P3 tool result truncation:** every tool result capped at 4000 chars (head 2800 + tail 800 + "re-call with narrower args" recovery marker; per-def `budget.maxToolResultChars`); trace records honest `chars→kept`. Proven E2E: 48k-char result → 3675 kept, head+tail+marker present in round-2 prompt.
- ✅ **P4 memory writes:** recall-time task embedding REUSED at store (one task embed per turn, proven by counter — was two); trivial turns skipped (answer <40 chars, task <20 chars, tool-error-only answers); stores distilled to 1200 chars (was raw 4k dumps).
- ✅ **P5 auto-compaction:** at 70% of the ctx budget old turns are SUMMARIZED (last 4 verbatim) instead of silently dropped; one-shot guard re-arms below 50%; summarizer failure falls back to drop-oldest. Manual `/compact` refactored onto the same core.
- ✅ **P6 provider-neutral prefix stability + caching:** static-first message ordering everywhere; Anthropic wire emits the system as ONE cacheable content block (`cache_control`) with leading messages[0] hoisted out of the messages array + marker on the LAST tool only; OpenAI/local wires unchanged (auto prefix caching needs nothing — asserted cache_control-free). Usage parsing gained provider-neutral `cacheReadTokens`/`cacheWriteTokens` (Anthropic `cache_read/cache_creation_input_tokens`, OpenAI `prompt_tokens_details.cached_tokens`) on all three stream-done paths.

**RAM hygiene (as much as needed, no more):**
- ✅ **M1:** `/watch` pending queue drained into each turn's context note + hard cap 100 (was pushed every poll tick, never read).
- ✅ **M2:** full QuickJS GC per completed turn via new `__chat_gc()` bridge (JS_RunGC previously ran only at process exit).
- ✅ **M3:** CMA decay finally wired — per-brain clock ticks elapsed Ebbinghaus decay at each run boundary; new `Cma::retain()` physically drops forgotten tombstones + sub-floor records (pinned `user_pin` facts and entities survive; survivors renumbered with corecall remap + HNSW rebuild; no-op path leaves index untouched). Exposed as `brain.retain()`.
- ✅ **M4:** agent trace rings at 512 events (RLM parity) — first-16 skeleton + freshest kept (proven: 400-step loop → exactly 512, start first).
- ✅ **M5:** `RUN_COUNTS` day-stamped `{n, day}` — runs24h resets on UTC rollover (the comment was a lie before).
- ✅ **M6:** soft RSS tripwire — footer RAM goes amber past `rss_warn_mb` (default 1024, persisted in config.json) with a one-time notice; inform, never kill.

**Post-landing audit fixes (2026-08-21):**
- ✅ **Anthropic consecutive-role merge** — adjacent plain same-role messages (ephemeral ctx + task on empty history, compaction summary + user turn) merge into one turn so the Messages API accepts the wire; unit-tested (`anthropic_merges_consecutive_same_role_turns`).
- ✅ **`def.system` composes AFTER CORE_PROMPT** — custom systems no longer drop the core's discipline lines; passing CORE_PROMPT itself is a no-op; battery-tested.
- ✅ **`/cost` shows cache hits** — `cacheReadTokens`/`cacheWriteTokens` flow agent.js → `__chat_report_usage` → per-turn `/cost` rows + session total.
- ✅ **M2 fan-out GC** — global `__sofuu_gc` bridge (engine.rs); agent.js GCs after `runMany`/`mapContext`.
- ✅ **M4 trace stubs** — evicted middle events demote to `{kind,t}` stubs (timeline kept for `renderTrace`); heavy payloads bounded to 512; battery-tested.
- ✅ **Verification gaps closed** — P5 auto-compaction E2E (`tests/chat_compact_e2e.sh`: fires at 70%, summarizer reaches mock, one-shot guard holds), P6 prefix-identical Rust test (`prefix_is_byte_identical_across_turns`), `recall_min`/`recall_budget` config wiring.

---

## ✅ Completed — CLI / TUI experience

- ✅ **TUI redesign (reference layout, own content):** bordered welcome panel (logo tile + Directory/Session/Model/Version facts), two-row footer (status + hints / right-aligned `ctx:` token metric), visible cursor in the input box, removed the persistent header bar.
- ✅ **Renderer correctness:** newline-split buffer ingestion, bottom-anchored append repaint, `LINE_CAP` enforced, ellipsis only on true truncation, full-width rows render exactly.
- ✅ **Resize handling:** SIGWINCH via `uv_signal_t` on the loop thread (previously a raw signal handler doing async-unsafe renders — and a no-op anyway).
- ✅ **Esc stops a streaming answer; Ctrl-C quits cleanly** (saves config, restores terminal). Includes a 60 ms bare-Esc disambiguation timer and stream `abort()` that truly disconnects the HTTP request (server sees the drop).
- ✅ **Interactive pickers** for `/model` (provider tabs, type-to-search, live model fetch for the active provider, cross-provider switch with key warnings), `/provider` (key status + custom → wizard), `/effort` (real 1k/4k/16k/32k thinking budgets, `off` clears). ↑↓/Tab/←→/PgUp/PgDn/Enter/Esc all routed; overlays never leave stale rows.
- ✅ `/providers` merged into `/provider` (hidden alias, completion + help de-duplicated).
- ✅ Type-ahead input mid-stream is preserved for the next prompt.
- ✅ Fixed loop-exit bug (open selector promise held no uv handle — pickers could freeze the process).
- ✅ Deferred unhandled-rejection tracker (`engine.c`): handled/promises-attached rejections no longer print noise; genuinely unhandled ones still report.

## ✅ Completed — PLAN-WIRING-FIXES.md (full plan, audited end-to-end)

- ✅ `sofuu.ai.embedLocal` bound (TF-IDF 768-dim offline embedder) — brain works with Ollama down.
- ✅ KV `k_summary` computed + persisted in `index.json` — cross-restart KV search ranks real pages.
- ✅ `sofuu.exec(cmd, args)` → `{code, stdout, stderr}`; `Sofuu.createMCPServer` alias; dead `createSSEServer` ref removed.
- ✅ `process.on('uncaughtException')` real dispatch (sync + async throw paths); default dump + exit 1 without a handler.
- ✅ Chat input history persisted at `~/.sofuu/chat_history` (load lazy, write-through, ↑ recall proven across restarts).
- ✅ `fs.mkdir{recursive}`, `fs.rm` (file + empty-dir), `fs.readFileBytes` (Uint8Array).
- ✅ MCP: async tool responses through the uv pipe (printf hack gone, `srv` via C-function data); 4KB/16KB buffers heap-growable (6KB result + 20KB schema proven); quoted command tokenizer (paths with spaces).
- ✅ MCP extras beyond plan: `client.call("add", …)` bare-name auto-wrap (README drift bug); notifications no longer answered (stray `id:0` spec violation removed).
- ✅ P3 agent loop: `~/.sofuu/mcp.json`, `/tools`, planning rounds on `ai.complete` → tool calls → final streamed answer; proven end-to-end with a mock LLM + a real sofuu MCP server (server-side log captured the full plan→tool→replan→stream shape).
- ✅ Fixed chat exit-hang with MCP servers connected (disconnect-all on exit).
- ✅ Fixed status footer leaking input-box frames into piped output.
- ✅ `fetch` streams response bodies: `Response.body` async iterator over live chunks (timed proof 1/356/708/1067 ms), buffered `text()/json()` intact.
- ✅ `provider:"local"` fails loudly (real local inference is Track B).

## ✅ Completed — Track D3: wire the Rust ports in, delete the C duplicates

- ✅ **A3a bundler:** `sofuu bundle` is pure Rust (`sofuu_core::bundler::bundle` + Rust npm walk-up resolver); C `sofuu_bundle` out of the binary; output behavior identical.
- ✅ **A3b TS strip:** engine + module loader call the Rust stripper via `ffi_exports.rs` (`sofuu_ts_strip_rs`, malloc-owned buffers); `ts/stripper.c` out of the cargo build; `ts_test.ts` 10/10.
- ✅ **A3c npm safety:** entry specs validated in Rust; tarball SHA-1 + safe extraction are Rust in the cargo build (`SOFUU_RUST_CORE`-gated; c-only keeps local impls). Real registry install + traversal rejection proven.
- ✅ **A3d SSE:** `sofuu.SSEParser` shell now drives the Rust parser via an opaque pointer — upgrades fell out: `\r\n\r\n`, multi-line `data:` joins, named events.
- ✅ **A3e MCP JSON-RPC builders** → Rust (newline framing + real escaping; all MCP tests pass).
- ✅ **B — `sofuu.memory` → Rust CMA** (the big one):
  - ✅ `ffi_exports.rs`: CMA exports (`sofuu_cma_new/free/hydrate/remember/remember_entity/recall/kv_hints/entities/mark_positive/forget/decay/consolidate/count/records_json/vectors`) with an opaque `Cma*`.
  - ✅ Hydrate/dump through the **existing QTSQ adapter** so brain files stay format-compatible (records JSON keys: `vector_index kv_page_id strength tier text role entity_* corecall_* total_recalls positive_recalls`; vec store layout unchanged). Legacy C brain files (no `age_seconds`/`half_life` fields) hydrate via serde defaults.
  - ✅ `mod_memory.c` is a thin JS-binding shell; record/result shapes unchanged (`{id, distance, score, role, text, tier, strength}` recall parity confirmed field-for-field).
  - ✅ `mod_agent.c` compiles against the shell façade (`agent_prefetch_context` now uses `cma_kv_hints` — no internal struct poking).
  - ✅ `verify_memory.js` 30/30 against the Rust CMA; old C-format brain file loads + recalls; new files load in both directions.
- ✅ **npm walk-up `resolve`** → Rust twin (`sofuu_npm_resolve_rs` in `ffi_exports.rs`) in the module loader under `SOFUU_RUST_CORE`; c-only keeps the C resolver. Proven with a real `node_modules` tree (package.json `main` resolution).
- 🟡 **MCP inbound parse:** the server-side request parse (`mcp_server_handle_request`) is now the Rust `jsonrpc::parse` (the security-sensitive inbound path); the **client response routing keeps the C `json_get_field`** — a first pass at the Rust parse on that path exposed a nondeterministic crash (refcount/GC interaction inside the uv read callback) that wasn't root-caused in time. Candidate for a follow-up with a GC-safe lifetime design.

## ✅ Completed — Cleanup after Batch B

- ✅ Dropped from the cargo build: `memory/hnsw.c`, `memory/dream.c`, `bundler/bundler.c` (ts/stripper.c was already out); `build.rs` tidied with the Rust-shell comment map.
- ✅ `make c-only` is REMOVED (prints an error directing to the cargo build). The Makefile's CFLAGS/ALL_SRCS/SOFUU_SRCS/readline blocks are deleted.
- ✅ QTSQ is now optional in the cargo build: `SOFUU_QTSQ_DIR` pointing at a checkout enables brain/session persistence; missing → build degrades gracefully (CI-compatible).
- ✅ ROADMAP.md D3 checkboxes marked done; README status sync below.

## ✅ Completed — PLAN-RUST-MIGRATION.md M9: memory shells (mod_memory/mod_kv/mod_agent/qtsq_adapter → rt/memory.rs)

- ✅ `src/memory/{mod_memory.c (506), mod_kv.c (585), mod_agent.c (159), qtsq_adapter.c (292)}` + all five `src/memory/*.h` (mod_memory/mod_kv/mod_agent/qtsq_adapter/formats) DELETED — replaced by `crates/sofuu-core/src/rt/memory.rs` (~1,950 lines, one module for the three shells + the adapter logic), same C symbols (`mod_memory_register`, `mod_kv_register`, `mod_agent_register`; engine.c calls them unchanged under its `SOFUU_QTSQ_PRESENT` guard). The CMA stays Rust (`crate::memory::cma::Cma`, the same object `ffi_exports.rs`'s `sofuu_cma_*` hand to C — the shell now calls it directly, no FFI round-trip; the `sofuu_cma_*` exports remain untouched).
- ✅ QTSQ FFI = new `crates/sofuu-ffi/src/qtsq.rs`: the ~16 adapter-used functions as externs (all verified real exports via `nm`), plus an opaque `QtsqContext` whose exact storage (5016 B, align 8 — probed with an offsetof C test) is calloc'd and handed to libqtsq; the adapter's direct field reads (header.data_type, schema dims, is_encrypted) are byte-offset accessors pinned to `qtsq_format.h` with an asserting unit test. No new C shims were needed (`JS_ThrowInternalError` was added to the qjs extern block — it's a real quickjs export).
- ✅ **cfg-gating mirrors the C build exactly:** the C core defines `SOFUU_QTSQ_PRESENT` only when `libqtsq.a` exists; engine.c then calls the registers. The Rust exports exist UNCONDITIONALLY (linker contract) but the whole implementation is `#[cfg(has_qtsq)]` (new `crates/sofuu-core/build.rs` probes the same `SOFUU_QTSQ_DIR`/libqtsq.a condition — `cargo:rustc-cfg` from ffi's build.rs can't cross crates). Without QTSQ the registers are no-ops → JS sees no sofuu.memory/kv/agent, byte-for-byte the C's observable degradation.
- ✅ Adapter call sequences + buffer ownership mirrored exactly: compressions (`qtsq_compress_tensor` lossless f32 for "memories", `qtsq_compress_json` for "metadata", `qtsq_compress_tensor_quantized` F8/F16 for `kv_%08u_{K,V}`), pack → `qtsq_vault_encrypt_password` ("sofuu-kv-local-v1", no Keychain prompts) → fail-closed `qtsq_write`; reads decrypt in place when `is_encrypted`, malloc'd outputs consumed via temporary slices then `free`d.
- ✅ KV page store: 2-page RAM LRU, mean-pooled 64-dim `k_summary` (L2-normalized; zero-norm → unit e0), `index.json` persisted with C `%g` float formatting via `libc::snprintf` → **byte-identical index files**, cross-restart search ranks real pages. `sofuu_cosine_f32` still called over FFI from src/simd (stays C).
- ✅ Safety-only deviations from the C (all in the C's UB zones; invisible to correct callers): short Float32Array args now TypeError/empty instead of out-of-bounds reads; non-string `open()` paths fail gracefully instead of `strncpy(NULL)` segfaults; `kv.save` with K/V arrays shorter than the implied tensor refuses instead of `memcpy` OOB.
- 🐛 One port bug found & fixed in independent review (post-subagent): `compute_k_summary` pooled into a single slot (`t % KV_SUMMARY_DIM`) → every page's index summary degenerated to `[1,0,0,…]` after normalization (all pages tied at rank 1). Now mean-pools ACROSS components — C-era `index.json` files carry distributed signatures. Verified after the fix: seed 64 pages + `flush()` → two fresh processes both read identical `[64,63,62,61,60]` (exact-match page first); in-process search agrees.
- ✅ Parity gates: `verify_memory.js` **30/30** (recall fields + records_json keys + brain-file format identical; GMF/dream/resonance all green; consolidate() made 1 real semantic cluster); KV cross-restart check — 64 deterministic pages seeded in one process, reopened in a fresh process: identical top-hits `[64,63,…,55]` MATCH; **C-era stores still load**: `test_agent_kv/` (written by mod_kv.c Aug 12) reopens, searches to `[1]`, and its page tensor decompresses through the Rust adapter (agent.prefetch → kv_get_page = 1); `examples/priority2_test.js` **27/27** (memory modules coexist with everything else).
- ✅ `cargo test --release --no-fail-fast`: **121 passing** (103 lib + 11 bin + 7 ffi — M9 added 3 memory.rs + 2 qtsq.rs unit tests). `make size-check` 1.8MB ≤ 5MB. LOC: **2,403 C / 27,517 Rust** (M9 removed 1,869 lines incl. 5 headers; `src/memory/` is now fully gone; `src/` = engine, ffi_shim, io/tui, repl, main + simd + header-only modules/).

## ✅ Completed — PLAN-RUST-MIGRATION.md M10: Engine & runtime init (engine.c + sofuu.c + ffi_shim.c + repl.c + tui.c)

- ✅ `src/engine/engine.c` (756) + `src/sofuu.c` (47) + `src/ffi_shim.c` (271) + `src/io/tui.c` (324) + `src/repl/repl.c` (202) + all remaining headers DELETED — replaced by:
  - `crates/sofuu-core/src/rt/engine.rs` (~960 lines): full port of engine lifecycle (boot, module loader, builtin registration, Sofuu aliases, REPL eval via repl_inspect, C-ABI exports sofuu_init/eval_file/eval_string/run_jobs/destroy/get_engine/engine_eval_repl/sofuu_engine_ctx).
  - `crates/sofuu-core/src/rt/tui.rs` (~370 lines): alt-screen TUI renderer (tui_* layout helpers + sofuu_tui_active/log/width bridges).
  - `sofuu-ffi/src/qjs.rs`: all 29 `sofuu_js_*` shims reimplemented as real Rust `#[no_mangle] pub extern "C" fn` (tag checks, value construction, refcount ops via __JS_FreeValue/RT).
  - `sofuu-ffi/src/uv.rs`: all `sofuu_uv_*` size/stat shims reimplemented via `uv_loop_size()/uv_handle_size()/uv_req_size()/uv_fs_get_result()/uv_fs_get_statbuf()` (real libuv exports).
  - `sofuu-ffi/src/qtsq.rs`: `sofuu_qtsq_session_save/load/free` reimplemented in Rust.
- ✅ `build.rs` pruned to QuickJS + SIMD + http-parser only (SOFUU_QTSQ_PRESENT/SOFUU_RUST_CORE defines removed; qtsq include path removed; `SOFUU_UV_DIR` + `SOFUU_CURL_STATIC_DIR` env overrides added for cross-builds).
- ✅ `make c-only` REMOVED (error message); cross scripts rewritten to cargo+zig.
- ✅ **Critical fix during M10:** `JS_EVAL_FLAG_COMPILE_ONLY` corrected from 0x400 to 0x20 (1<<5) — the wrong value made the ESM loader fully link+run imported modules, then the parent linked them a second time → js_inner_module_linking asserts + atom table corruption. `JS_PROP_HAS_WRITABLE/HAS_ENUMERABLE` swap corrected.
- ✅ `sofuu-ffi` dev-dependency on `sofuu-core` removed (fat-LTO cycle fix — runtime_and_bridge_work test moved to sofuu-core lib.rs).
- ✅ `cargo test --release`: **121 passing** (104 lib + 11 bin + 6 ffi). `make size-check` 1,875,120 bytes ≤ 5MB. LOC: **181 C / 29,465 Rust** (src/ = simd/ only + js/rlm.js data asset).
- ✅ Parity gates: `priority2_test.js` 27/27, `priority1_test.js` 14/15 (pre-existing headers.get bug), `timers.js` ✓, `fs_test.js` ✓, `spawn_test.js` ✓, `ts_test.ts` ✓, `simd_test.js` 9/9, `modules.js` ✓, fetch local battery ALL-FETCH-OK, SSE ✓, HTTP server via curl ✓, MCP client PASSED, MCP server E2E ✓, `npm_test.js` 3/3 (real registry), `verify_memory.js` 30/30, KV cross-restart ✓, AI mock battery AI-M8-OK, chat boot exit 0, REPL ✓.
- ✅ **Plan §7 done-state REACHED.** src/ = simd/ + js/rlm.js only (181 C lines). M11 (http-parser → httparse) optional.

## ✅ Completed — PLAN-RUST-MIGRATION.md M8: AI module (mod_ai.c → rt/ai.rs)

- ✅ `src/modules/mod_ai.c` (2,543 lines) + `src/memory/tfidf_embed.c` (73) + `tfidf_embed.h` DELETED — replaced by `crates/sofuu-core/src/rt/ai.rs`, same symbol (`mod_ai_register`, engine.c unchanged) + the whole `sofuu.ai.{complete,stream,embed,embedLocal,similarity,dot,l2,estimateTokens,listModels,listProviders}` surface and the global `__ai_abort` stream-cancel hook. The stream async-iterator wrapper is the same embedded JS factory string, fed to `JS_Eval` where the C did. `src/modules/mod_ai.h` STAYS (engine.c `#include`s it and must keep compiling unchanged — same convention as mod_console.h/mod_process.h from M2/M3).
- ✅ Providers openai/anthropic/gemini/ollama/openrouter/local/custom incl. `profile`-based wire formats; the request-body builders emit the same JSON (`json_escape` identical: `" \ \n \r \t \b \f` + c<0x20 → `\uXXXX`); response parsing (choices[0].delta.content / message.content / anthropic content[0].text / gemini candidates[0].content.parts[0].text / ollama message.content) keeps the C's string-scan approach; tool-call extraction still navigates the parsed JSON with QuickJS.
- ✅ **The tfidf_embed.c embedder ported INTO rt/ai.rs** — `embedLocal` is now a pure-Rust hashing-trick embedder (char trigrams → MurmurHash3 → L2-norm). Verified numeric-identical: the 768-dim runtime vector matches an independent reference implementation of the C algorithm with **0 differing elements**. (Also removes the M9 plan item for the embedder.)
- ✅ SIMD kernels STAY C — `sofuu_dot_f32`/`sofuu_l2_f32`/`sofuu_cosine_f32` (src/simd/{neon,avx}.c) are called over FFI from the Rust similarity/dot/l2 wrappers. New curl bindings in sofuu-ffi: `curl_multi_strerror` + option ids POSTFIELDS=10015, POSTFIELDSIZE=60, SSL_VERIFYPEER=64, MAXFILESIZE=114, TIMEOUT_MS=155 (verified against the Xcode SDK curl.h); CURLMOPT_SOCKETFUNCTION/TIMERFUNCTION = 20001/20004 (mirrors the M4 verification). No new qjs/uv shims were needed.
- 🐛 One M8 self-caught bug: a MurmurHash3 constant typo (`0x85eaca6b` vs the correct `0x85ebca6b`) — caught by the numeric reference check, not by tests.
- 🐛 One port bug found & fixed in independent review (post-subagent): `compute_k_summary` pooled into a single slot (`t % KV_SUMMARY_DIM`) → every page's index summary degenerated to `[1,0,0,…]` after normalization (all pages tied at rank 1). Now mean-pools ACROSS components — C-era `index.json` files carry distributed signatures. Verified after the fix: seed 64 pages + `flush()` → two fresh processes both read identical `[64,63,62,61,60]` (exact-match page first); in-process search agrees.
- ✅ Parity gates: `simd_test.js` 9/9; the mock-OpenAI gate (custom provider → 127.0.0.1:8899): complete text + raw payload, 11 stream chunks + usage + `[DONE]`, embedLocal 768-dim unit Float32Array (same-topic 0.141 vs unrelated 0.078 vs identical 1.000), `abort()` before and mid-stream end cleanly, `listModels`/`listProviders`/`estimateTokens`/local-provider guard; ollama examples fail cleanly when ollama is down; chat boot (`/exit`) clean.
- ✅ `cargo test --release --no-fail-fast`: **116 passing** (100 lib + 11 bin + 5 ffi). `make size-check` 1.88MB ≤ 5MB. LOC: **4,272 C / 25,326 Rust** (M8 removed 2,643 C lines — the biggest single phase; `src/modules/` now carries only header declarations for engine.c).

## ✅ Completed — PLAN-RUST-MIGRATION.md M7: npm runtime (resolver.c + cjs.c → rt/{npm,cjs}.rs)

- ✅ `src/npm/resolver.c` (708) + `src/npm/cjs.c` (131) DELETED — replaced by `crates/sofuu-core/src/rt/{npm,cjs}.rs`, same symbols: `npm_resolve` (delegates to the D3 Rust walk-up twin), `npm_install`, `npm_install_local_package_json`, `is_cjs`, `cjs_to_esm`, `mod_cjs_register` (plus the internal `js_require`).
- ✅ Installer pipeline ported: registry metadata fetch (sync curl_easy_perform), mkstemp'd tgz, SHA-1 verification via the Rust twin, the Rust safe tarball extractor, scoped-package dirs, symlink-target refusal, transitive dependency recursion. The C code parsed manifests with throwaway QuickJS contexts — the port uses serde (identical behavior).
- ✅ Parity: `npm_test.js` 3/3 (real is-odd install → CJS require), `sofuu add is-odd` CLI with **transitive is-number install**, `require("myadd")` E2E (resolution + wrapper + module.exports), `modules.js`/`greet.js` ESM imports unchanged.
- ✅ `cargo test --release --no-fail-fast`: **116 passing**. `make size-check` 1.89MB ≤ 5MB. LOC: **6,915 C / 22,256 Rust** (M7 removed 839 C lines; `src/npm/` is now empty).

## ✅ Completed — PLAN-RUST-MIGRATION.md M6: MCP (mcp.c → rt/mcp.rs, Rust client routing)

- ✅ `src/mcp/mcp.c` (1,239 lines) DELETED — replaced by `crates/sofuu-core/src/rt/mcp.rs`, same symbols (`mod_mcp_register` + `sofuu.mcp.{connect,serve}` + client call/listTools/listResources/disconnect + server tool/start, stdio transport).
- ✅ **The M6 mandate: client response routing now runs the Rust `mcp::jsonrpc::parse` (safe serde) — the retired C `json_get_field` scan is GONE** (the path kept in C only because of the old nondeterministic GC crash). Proven with the mandated stress: **10k pipelined JSON-RPC responses, 10000/10000 in 201ms, zero errors** (sliding 60-wide window — MAX_PENDING=64 is the C design bound, kept for parity).
- ✅ New FFI/test: sofuu-ffi gained a spawn-read binding test (`m6_mini_spawn_reads_child_stdout`) that drives `uv_spawn` + `uv_read_start` on a real child.
- 🐛 **Three M6 bugs, each discovered by the debug cycle:**
  1. **`UV_WRITABLE_PIPE`/`UV_READABLE_PIPE` swapped** in uv.rs (mine 0x10/0x20; uv.h says READABLE=0x10, WRITABLE=0x20) — every CREATE_PIPE child stdout read got `UV_ENOTCONN` (spawn + MCP alike; spawn_test's "STDOUT:" had actually been silent since M2). Fixed against uv.h; the new mini test pins it.
  2. **`JS_NewClassID` only allocates when `*pclass_id == 0`** (quickjs.c:3396-3412) — reusing one Rust local for the second class gave client+server the SAME id; the client object ended up with the server proto (listTools undefined). The C code's per-class `static id = 0` must be mirrored by resetting the local. Also fixed the same latent bug in http_server's res/srv classes.
  3. **Missing `JS_SetOpaque` on the client-connect success path** — "Invalid MCPClient" on every method call.
- ✅ Parity: `mcp_client_test.js` (connect → tools → echo → add → PASSED), `mcp_server_test.js` (initialize/tools/list/tools/call/unknown-tool over piped stdio), `spawn_test.js` (stdout back), `priority2_test.js` 27/27.
- ✅ `cargo test --release --no-fail-fast`: **116 passing** (100 lib + 11 bin + 5 ffi). `make size-check` 1.89MB ≤ 5MB. LOC: **7,754 C / 21,648 Rust** (M6 removed 1,239 C lines; `src/mcp/` is now empty).

## ✅ Completed — PLAN-RUST-MIGRATION.md M5: HTTP server (server.c → rt/http_server.rs)

- ✅ `src/http/server.c` (459 lines) DELETED — replaced by `crates/sofuu-core/src/rt/http_server.rs`, same symbols (`mod_http_server_register` + global `createServer`, `sofuu.createServer`, legacy `sofuu.serve`) and the same req/res surface (`req:{method,url,body}`, `res.writeHead/write/end/send` incl. the chunked streaming path).
- ✅ New FFI: `sofuu-ffi/src/http_parser.rs` (vendored http-parser: init/execute/method_str + the 32-byte struct mirror with bitfield accessors), `uv_tcp_init/bind`, `uv_ip4_addr`, `uv_listen`, `uv_accept`, `sofuu_uv_tcp_size`.
- ✅ Proofs: `server_test.js` (GET/POST via curl), SOAK **100/100 sequential chunked requests** with the server staying alive, `mcp_server_test.js` unaffected, `priority2_test.js` 27/27.
- 🐛 **uv_write_t payload corruption (caught by lldb: "pointer being freed was not allocated")**: the first port mirrored the C `write_req_t` (payload fields INSIDE the uv_write_t region) — but real `uv_write_t` is 56 bytes while the mirror's payload fields sat at offsets 8+ where libuv writes its own fields → garbage frees. Fixed by moving the payload to a Box stashed in `req->data` (libuv reserves it for users), with the uv_write_t at its own malloc. An abort-breakpoint trace + a temporary C-side comparison build (the original server.c survived) proved it was a port bug, NOT the documented pre-existing server issue.
- ✅ `cargo test --release --no-fail-fast`: **115 passing**. `make size-check` 1.89MB ≤ 5MB. LOC: **8,993 C / 20,253 Rust** (M5 removed 458 C lines; `src/http/` is now EMPTY).

## ✅ Completed — PLAN-RUST-MIGRATION.md M4: HTTP client + SSE shell (client.c + sse.c)

- ✅ `src/http/client.c` (1,032 lines) + `src/http/sse.c` (211 lines) DELETED — replaced by `crates/sofuu-core/src/rt/{http_client,http_sse}.rs`, exporting the SAME symbols (`mod_http_client_register`, `mod_http_sse_register`) + the full JS surface: global `fetch` + `sofuu.fetch`, Response class (status/ok/statusText/url/text/json/arrayBuffer/headers.get/streaming `body` accessor), `__fetch_stream_{get,next,wait}` bridges, `sofuu.SSEParser`.
- ✅ New FFI: `sofuu-ffi/src/curl.rs` (curl_global_init/easy*/multi*/slist/getinfo/setopt + callbacks), `uv_poll_*` + `UvPoll`, `JS_DefineProperty`/`JS_NewAtom`/`JS_NewCFunction2`/`JS_NewArrayBuffer`/`JS_GetProperty`(shim)/`sofuu_uv_{poll,check}_size` shims.
- ✅ Verified against a local slow-chunk HTTP server: 201 + JSON parse, redirect follow, POST, 404, and the STREAMING proof (response resolved on the first header block; body chunks delivered incrementally at ~304/612ms, not end-buffered). `sse_test.js` green through the Rust shell.
- 🐛 Two ABI constants were wrong and got fixed mid-phase (lesson: **verify every libcurl/JSC ABI constant against the headers, don't guess enum indices**): `CURLMOPT_SOCKETFUNCTION=20001`/`CURLMOPT_TIMERFUNCTION=20004` (multi.h enum), `CURL_POLL_{IN,OUT,INOUT,REMOVE}=1..4` (starts at NONE=0), `CURLINFO_PRIVATE=0x100015`, `JS_PROP_HAS_GET=1<<11` (1<<9 is HAS_ENUMERABLE).
- 🐛 Lifetime fix: the curl poll-context must be freed by the uv close callback (the first port freed the Box early → double-free/UB).
- ℹ️ Root-caused the LONG-STANDING priority1 failure (14/15, pre-existing): Response `headers.get()` always returns null for streaming responses — `headers_raw` is captured into the response at the FIRST header callback (just the status line), later lines append to the request's NEW buffer. Reproduced bug-for-bug by the port (unchanged behavior, correct parity); a fix would move the raw-headers capture to transfer completion — intentionally NOT part of a migration.
- ✅ `cargo test --release --no-fail-fast`: **115 passing**. `make size-check` 1.89MB ≤ 5MB. LOC: **9,451 C / 19,441 Rust** (M4 removed 1,243 C lines; `src/http/` now holds only `server.c`).

## ✅ Completed — PLAN-RUST-MIGRATION.md M3: process module (mod_process.c → modules/process.rs)

- ✅ `src/modules/mod_process.c` (1,920 lines) DELETED — replaced by `crates/sofuu-core/src/modules/process.rs`, exporting the SAME symbols (`mod_process_register`, `mod_process_set_args`, `mod_process_cleanup`, `process_dispatch_pending_signals`, `process_dispatch_uncaught`). engine.c + rt/{loop,promise,timer} call them unchanged.
- ✅ Full JS surface byte-identical: `process.{version,runtime,platform,pid,argv,env,exit,cwd,chdir,on,off,stdin,stdout,stderr}`; SIGINT/SIGTERM flag-setters + loop-thread dispatch (POSIX sigaction, exit 130/143 fallback); `__uncaught_handler`/`__exit_handler` hidden props; the async TTY readline (raw mode, live-region box renderer inline + TUI, `__chat_complete` TAB completion, 64×512 history with `~/.sofuu/chat_history` persistence, ESC state machine incl. Shift+Enter/bare-ESC 60ms timer, Ctrl-C/Ctrl-D); `__prompt` (canonical fallback with termios), `__readline` (promise, type-ahead queue), `__chat_status`, `__chat_is_tty`, `__tui_on/off/log/log_last/scroll/set_header` (tui.c stays C — called via FFI), `__selector_open/draw/close`, `__ttyRaw/__ttyNormal`.
- ✅ `process.argv` FIXED for the Rust binary: main.rs now calls `mod_process_set_args` (the retired C main.c did; the Rust shell never had → argv was empty).
- ✅ New bindings: `uv_tty_init/set_mode/reset_mode`, `uv_cwd`, `uv_chdir`, `UV_TTY_MODE_*`, `sofuu_uv_{tty,signal}_size`, `sofuu_js_null` shim.
- ✅ Parity: `process_test.js`, `priority2_test.js` 27/27 (signals + uncaught + exit handler), `timers.js`, `ts_test.ts` green; piped-stdin `__readline` round-trip (prompt echo + line delivery + queueing) verified; live SIGTERM → handler → clean exit verified; `process.argv` now `["./sofuu","run",...]`.
- ✅ `cargo test --release --no-fail-fast`: **115 passing (100 lib + 11 bin + 4 ffi)**. `make size-check` 1.89MB ≤ 5MB. LOC: **10,687 C / 17,793 Rust** (M3 removed 1,912 C lines; `src/modules/` now holds only mod_ai.c).

## ✅ Completed — PLAN-RUST-MIGRATION.md M2: leaf I/O (timer + fs + subprocess + console)

- ✅ `src/io/timer.c`, `src/io/fs.c`, `src/io/subprocess.c`, `src/modules/mod_console.c` DELETED — replaced by `crates/sofuu-core/src/rt/{timer,fs,process_spawn}.rs` + `crates/sofuu-core/src/modules/console.rs`, exporting the SAME symbols (`mod_timer_register`, `mod_fs_register`, `mod_subprocess_register`, `mod_console_register`). engine.c calls them unchanged; no caller-side edits.
- ✅ Full JS surface byte-identical: `setTimeout/setInterval/clearTimeout/clearInterval` + `sofuu.sleep`; `sofuu.fs.{readFile,writeFile,appendFile,exists,readdir,mkdir{recursive},rm,readFileBytes}` (incl. the Uint8Array path); `sofuu.spawn` (Subprocess class, kill/write, exit fence with 4-handle close counting) + `sofuu.exec` (64MB capture cap, {code,signal,stdout,stderr}); `console.{log,info,warn,error,assert,time,timeEnd,table}` with the exact ANSI/ASCII formats.
- ✅ New bindings/shims: `JS_CallConstructor`, `JS_GetOwnPropertyNames`+`JSPropertyEnum`/`JSAtom`/`JS_AtomToCString`/`JS_FreeAtom`/`js_free`, `uv_fs_stat/fstat/scandir/scandir_next`, `uv_spawn`, `sofuu_js_new_uint32`, `sofuu_uv_{fs,write,pipe,process}_size`, `sofuu_uv_fs_{result,stat_size}`. Corrected `UvStdioContainer` (union layout) and `JSCFunctionListEntry` (32B, `u.func.length/cproto/cfunc` — the earlier 24B mirror corrupted the atom table; caught by a boot bisect + lldb).
- ✅ Proof: new `rt::timer::tests::m2_timers_fire_and_clear_through_rust_loop` (setTimeout + interval + clearInterval + sleep through the Rust loop). Loop-driving tests (m1/m2) serialized via `TEST_LOOP_LOCK` (process-global uv loop is one instance — parallel threads would cross-schedule timers).
- ✅ Parity: `timers.js`, `fs_test.js`, `spawn_test.js`, `priority2_test.js` 27/27, `hello.js`, `greet.js`, `sse_test.js` green; plus manual probes for recursive mkdir/readdir, readFileBytes→Uint8Array, console.time/table/assert formats.
- ✅ `cargo test --release --no-fail-fast`: **115 passing (100 lib + 11 bin + 4 ffi)** — full suite green, incl. the previously red rlm js_api roundtrip. `make size-check` 1.89MB ≤ 5MB. LOC: **12,599 C / 15,186 Rust** (M2 removed 1,622 C lines).

## ✅ Completed — PLAN-RUST-MIGRATION.md M1: event-loop spine (rt/loop + rt/promise)

- ✅ `src/io/loop.c` + `src/io/promises.c` DELETED; replaced by `crates/sofuu-core/src/rt/{loop,promise}.rs` exporting the SAME C symbols (`sofuu_loop_init/get/run/close`, `sofuu_promise_new/resolve/reject/reject_str`, `sofuu_flush_jobs`) — every C caller (engine.c, mod_*.c, http/, mcp/, io/timer|fs|subprocess) links the Rust implementations, caller-side unchanged.
- ✅ Deferred unhandled-rejection tracker moved verbatim from engine.c into `rt::promise` (thread-local pending list, 64-eager print, pointer-identity retraction, print-and-clear at drain end + shutdown); engine.c installs via `sofuu_rt_install_rejection_tracker` under `SOFUU_RUST_CORE` (c-only keeps its frozen C copy, guarded out).
- ✅ New bindings/shims: `uv.rs` gained `uv_walk` + `sofuu_uv_loop_size`/`sofuu_uv_timer_size`; `qjs.rs` gained `JS_ToString`, `JS_SetHostPromiseRejectionTracker`, `sofuu_js_undefined/exception/value_get_ptr` (ffi_shim.c). Stale `build.rs` `-I src/ts`/`-I src/bundler` removed.
- ✅ Proof: `rt::promise::tests::m1_promise_resolved_from_uv_timer_fires_js_then` — promise created in Rust, resolved from a uv timer callback, JS `.then` observes the value.
- ✅ Parity: `timers.js`, `priority2_test.js` 27/27, `fs_test.js`, `spawn_test.js` green; `priority1_test.js` 14/15 — the single failure is the pre-existing httpbin Content-Type (HTTP/2 header-scan) issue, unchanged from baseline.
- ✅ `cargo test --release`: **113 passing** (98 lib + 11 bin + 4 ffi); `make size-check` 1.8MB ≤ 5MB.
- ⬜ Pre-existing, NOT M1: `rlm::js_api::tests::js_roundtrip_through_quickjs` fails — its `JS_Eval` gets an unterminated `&str`; QuickJS needs a NUL sentinel (the C core always NUL-terminates via `engine_read_file`). The production sandbox (`sofuu-ffi/src/qjs_rt.rs`) already uses `CString` — only the tests hit it. RLM-side fix.

## ✅ Completed — Agents & sub-agents + web search (PLAN-AGENTS.md A1–A6, A8, A9 + sofuu.web, 2026-08-16)

- ✅ **A1 — core agent runtime** (`src/js/agent.js`, shipped JS like `rlm.js`; eval'd per engine context by the new `crates/sofuu-core/src/shipped.rs` seam after the RLM driver): `sofuu.agent.define/run/runMany/mapContext/cancel/list/loadDir/renderTrace`. Merges into the LIVE `sofuu.agent` object — memory.rs's `agent.create/prefetch` survive untouched. `AgentResult` = `{answer, steps, subRuns, usage, trace, stopped, runId, name}`. Loop is wire-format aware: stream-first for OpenAI-compatible providers (streamed `delta.tool_calls` fragments merged by index — proven with args split across two SSE deltas), `complete()`-planning for Anthropic-wire (streamed `tool_use` extraction is F4b); headless no-tool turns reuse the completion text (one call cheaper), onStep consumers get a streamed final answer.
- ✅ **A2 — delegation + parallelism + rails:** `delegate` tool injected only when `agents` is set AND depth < `budget.maxDepth` (a run at max depth *structurally* gets no delegate tool — verified); cycle guard via delegation chain (A→B→A surfaces a readable tool error, never a hang); children get their own definition's tools (least privilege); `runMany` promise pool with concurrency cap (max in-flight measured ≤ cap and reached cap); child usage merges into the parent result.
- ✅ **A3 — memory scopes over one physical brain:** `shared` (session brain, current behavior) / `agent` (private namespace via CMA entity records `agent:<name>`, recall filtered by entity) / `off` (no recall, no store — count provably unchanged). Required a small Rust plumbing change: `RecallHit.entity` now populated + serialized (`memory/cma.rs`, `rt/memory.rs`, `ffi_exports.rs`). Identity records (`identity: {role, expertise}`) persist per agent. Verified: no cross-agent leak, own-scope recall works, `off` writes nothing.
- ✅ **A4 — agents ↔ RLM, both directions:** `agent.run` gates oversized turns through `sofuu.rlm.query` (`rlm: 'auto'|'on'|'off'` on the definition; heuristics + window estimate), folds `rlm:*` trace events, and falls back to the plain loop with a `rlm:fallback` trace note when RLM errors/absent. `sofuu.rlm.query(..., { recurseVia: { agent } })` routes the sandbox's `llm()` sub-calls through a tool-capable agent instead of a raw completion (src/js/rlm.js — credentials never cross the seam). `sofuu.agent.mapContext(context, task, {agent, chunkChars, concurrency, reduceAgent})` = chunk (paragraph-preferred, 200-char overlap) → parallel worker runs → reduce merge; 100KB needle corpus found via ~15 chunk sub-runs.
- ✅ **A5 — budgets, cancellation, timeouts:** `budget: {maxSteps=12, maxDepth=2, maxTokens=200k, maxWallMs=300s}` enforced before every planning call (tree-summed — children inherit remaining budget); `stopped` = `budget_steps|budget_tokens|budget_wall|cancelled`; `sofuu.agent.cancel(id)` aborts in-flight answer streams + between-step waits; per-tool `toolTimeoutMs` (default 30s) turns a hanging tool into a tool error the model can read.
- ✅ **A6 / F4a — MCP routing fix:** `callMcpTool` was first-server-wins (dangerously wrong with two servers exposing the same tool name — silent cross-calls). Now a name→owner map routes to the owning server, and inline/builtin tools win name clashes with a visible warning event. Proven with two REAL MCP child servers (`examples/mcp_echo_server.js` spawned via `sofuu.mcp.connect`): `srv2_tool` lands on srv2, `shared_tool` resolves to the inline tool.
- ✅ **A8 — config, CLI, chat hooks:** `~/.sofuu/agents/*.js` loaded lazily by `sofuu.agent.loadDir()` (a broken file is listed with its error, never fatal — duplicate defines land in `broken` too); `sofuu agent list` / `sofuu agent run <name> "task" [--json]` CLI verb; `/agents` chat command (registry + broken files + run counts). `@agent` mentions were parked at A8 time — **landed 2026-08-21** (see the 2026-08-21 section above).
- ✅ **A9 — observability:** full trace tree per run + `sofuu.agent.renderTrace(result)` (indented spans with tokens/wall/tools); `opts.logs: true` appends one JSON line per run to `~/.sofuu/logs/agent_runs.jsonl` (5MB rotate).
- ✅ **Chat migrated onto the ONE loop (A1.5):** the driver's `turn()` now calls `sofuu.agent.run` with a transient definition — recall/store/markPositive, RLM routing, tool planning/execution, budgets and streaming all live in `agent.js`; chat only renders `onStep` events (same spinner/tool-line/think UX) and owns input/history/session logging. Brain config flows through (`memory: cfg.brain ? 'shared' : 'off'`, remote `embed_provider`/`embed_model` preserved with dimension check). Esc now routes through `sofuu.agent.cancel('chat')`. Web tools (`web_search`, `web_open`) are built into every chat turn's toolset.
- ✅ **Web search (`src/js/web.js`, same shipped-JS pattern):** `sofuu.web.search(query, opts)` → `{query, engine, results:[{title,url,snippet}], ms}` and `sofuu.web.open(url)` → readable page text (tags/entities stripped, 20k cap). Engines: **duckduckgo (default — NO API key**, parses the public HTML endpoint, `uddg` redirect decoding) · **brave** (`BRAVE_API_KEY` or `opts.api_key`) · **tavily** (`TAVILY_API_KEY`). `SOFUU_WEB_ENGINE` selects; `SOFUU_WEB_ENDPOINT` overrides the DDG endpoint (proxy/tests). `sofuu.web.TOOLS.web_search/web_open` are built-in agent/chat tools.
- ✅ **Regression fix exposed by the new tests (pre-existing, not from this work):** an uncommitted 2026-08-15 edit to `rt/ai.rs` broke EVERY v2 AI call — plain messages serialized as `"tool_calls":undefined` (invalid JSON; every provider 400s) and `tool_calls` arrays as the literal string `[object Object]` (`JS_ToCString` on an array). Fixed with type-checked reads (`sofuu_js_is_string`) + `JS_JSONStringify` for the array; `content:null` assistant messages now serialize as `null`-absent instead of the string `"null"`. `rlm_mock_test.js` green again (it had silently broken in the working tree).
- ✅ **Verification (2026-08-16):** `examples/agent_test.js` — 46-check mock battery **ALL PASSED**: A1 inline tool + fragmented streamed args, A2 delegation/depth-cap/cycle-guard/runMany-concurrency, A3 scopes on one QTSQ brain, A4 mapContext needle hunt + rlm fallback + `rlm:on` routing + `recurseVia` through a tool-using agent, A5 budgets×2/cancel/tool-timeout, A6 two real MCP children + inline-wins clash, A8 loadDir/list, A9 renderTrace, web_search tool end-to-end against a DDG-shaped mock. `examples/rlm_mock_test.js` ALL PASSED (post-fix). `cargo test` 121 passing.

## ✅ Completed — TUI visual polish pass (2026-08-16)

Verified by pty screen captures (custom harness: mock provider + ANSI screen
replay) at 100/80/62 cols before/after; 9 new `rt/tui` unit tests; full
`cargo test` 151 green; agent + RLM batteries green on the same build.

- ✅ **Welcome-panel border clipping FIXED** (the most visible bug): rows whose
  visible width exactly equals the terminal and end with an ANSI reset lost
  their last glyph to a spurious `…` (every `╮`/`│`/`╯` right border showed as
  `…`). Root cause: `tui_truncate_cells` counted the trailing zero-cell escape
  as cutting mid-content; it now absorbs trailing escapes after the cut.
- ✅ **Soft word-wrap for the conversation** (the big readability win): long
  streamed answers/tool output used to clip with `…` after one row — now they
  wrap at word boundaries (hard-break for unbreakable tokens like URLs), with
  ANSI escapes treated as zero-width and never split. Verified: the same
  multi-paragraph answer that clipped at 100 cols now reads fully, and wraps
  cleanly again at 62 cols.
- ✅ **Wide characters count 2 cells** (`tui_disp_width`/`truncate`/`panel_row`
  now share one `char_cells` table): CJK/emoji content no longer overflows the
  right edge by half its width.
- ✅ **Mascot redesign:** row 1 re-feathered (`░▒░▒░`) and sized to the exact
  7-cell tile width so the title/text column aligns across all three bird rows
  (eyes/beak/tail were diagonally drifting); beak `▽`→`▸`.
- ✅ **Footer rework:** `sofuu · <model> · <provider> · effort · tokens` with
  dimmed separators (model stays magenta); ctx metric drops the meaningless
  `(0%)`, shows percent only at ≥1%, turns amber past 80% of the window;
  hints now say what they do (`/help · enter ⏎ · esc stop`).
- ✅ **Tool-call lines:** `⏺ web_search("query")` with the interesting argument
  pretty-printed (query/path/url/… instead of raw JSON) in cyan; delegate lines
  `⏺ researcher ← task…` in magenta; results collapsed to one indented dim row
  `↳ …` with newlines flattened to ` · ` (search results no longer splat
  unindented rows across the transcript).
- ✅ **`/help` redesigned:** sectioned (Commands / Session mesh / Shortcuts),
  aligned command column, cyan headers, dim descriptions.
- ✅ **Panel labels dimmed** (value column stays aligned — the layout tests
  pin this), tagline mentions web search.
- ✅ **Panic fix found by the narrow-capture sweep:** `wrap_row` off-by-one
  (`cur[cut+1..]` with cut+1 == len when a line-filling token follows a
  trailing space) crashed chat at 62 cols — fixed + regression test
  (`wrap_break_after_trailing_space`).
- ✅ **Bonus:** the brittle `shipped.rs` smoke assertion (last failing cargo
  test from the agents work) now checks the real exported API surface
  (`A.run`, `A.mapContext`) — `cargo test` fully green again (121 → 151 with
  the new TUI tests).

## ✅ Completed — PLAN-AGENTS remaining follow-ups (2026-08-16, second pass)

- ✅ **A4 FULL FORM — agent tools inside the RLM sandbox** (the plan's
  original A4.1 wording, previously shipped as `recurseVia` only):
  - `sofuu-ffi/qjs_rt.rs`: 8th native `__hx_tool` — throws the new
    `\0RLM_TOOL_SUSPEND:` sentinel (batch JSON payload), same shape as the
    llm suspend.
  - `js/rlm_bootstrap.js`: `tool(name, args)` / `toolBatch([{name,args}…])`
    with a host-injected `__TOOL_CACHE` (exact `__LLM_CACHE` suspension
    model); new names added to the G1 frozen whitelist.
  - `rlm/sandbox.rs`: `Outcome::ToolSuspended{key, calls}`; the cache key is
    the RAW bootstrap JSON (byte-exact — re-serializing through serde could
    reorder nested arg keys vs QuickJS insertion order and break the hit);
    both caches share the 256KB G2 injection budget.
  - `rlm/episode.rs`: `EpisodeAction::ResolveTool` + `feed_tool_results`
    (arity-checked, 4k-char caps), `maxToolCalls` budget (default 16,
    stopped:"budget=max_tool_calls"), same 3×-identical-batch anti-loop as
    llm, `tool` trace events, tool docs appended to the system prompt
    (name/description/schema, 32-tool cap), `RlmResult.tool_calls`.
  - `rlm/js_api.rs`: `tools`/`maxToolCalls` in the request JSON
    (schema-only — executors never cross into Rust, G3), `resolve_tools`
    action, new `__rlm_feed_tools` host fn.
  - `src/js/rlm.js`: `opts.tools` (specs) forwarded; `resolve_tools`
    fulfilled via `opts.execTool` (errors → readable tool-error strings; a
    missing executor feeds a visible marker, never a hang).
  - `src/js/agent.js`: toolset resolved BEFORE the RLM gate; routed turns
    inject the agent's own tool whitelist (delegate excluded — agents-inside-
    RLM stays `recurseVia`); execTool routes through `execOneTool` (timeout,
    trace, budget accounting) so tool code never runs inside the sandbox;
    cancel now also sets `__rlm_aborted` so Esc stops the episode between
    rounds.
- ✅ **F4b — Anthropic streamed `tool_use` accumulation** (`rt/ai.rs`):
  `content_block_start(tool_use)` + `input_json_delta` fragments reshaped to
  the OpenAI delta shape and surfaced through the same `setToolCalls` seam
  (one JS merger for all providers); stream usage maps
  `input_tokens`/`output_tokens` incl. the `message_start` → `message.usage`
  nesting. `agent.js` dropped the Anthropic special case — every provider
  runs the stream-first tool loop (the `complete()`-planning loop is gone).
- ✅ **Bonus fix found by F4b:** the stream `ensure-done` path (servers that
  end WITHOUT `[DONE]` — i.e. real Anthropic) called `done_fn` with no
  argument, freezing the factory's initial `{0,0}` usage. It now carries the
  collected tokens like the `[DONE]` path.
- ✅ **Bonus completion of A8.3:** chat turns now offer loaded sub-agents —
  `~/.sofuu/agents/*.js` load lazily on the first turn (same load as
  `/agents`) and the chat def carries `agents:[…]`, so the model CAN
  delegate in chat (maxDepth:1 — children structurally get no delegate).
  The `⏺ researcher ← task` line was previously unreachable from chat.
- ✅ `examples/web_test.js` — dedicated engine-level web battery, **39
  checks ALL PASSED** (DDG HTML parse/uddg/dedup/caps + HTTP-error
  guidance + unparsable-markup error; Brave header auth via client-side
  fetch capture + q/count params + 401; Tavily POST body shape + 403 +
  missing-key errors; `web.open` extraction/entities/truncation/plain
  passthrough/non-http rejection; unknown-engine + empty-query errors).
- ✅ **Chat verification after the agent.js migration** (isolated HOME +
  mock provider): piped-stdin smoke (exit 0, delegate tool line, delegation
  line, child answer flows, `Bye!`); pty (`script`) checks — `/agents`
  renders the registry row (`researcher · mock · memory off · 0 tools` +
  system line), the delegation line `⏺ researcher ← find the special code`
  renders, compact sub-agent result row `↳ researcher-answer-42 · [sub-agent
  researcher · 1 llm calls · 0 tool calls]`, live typing/footer/ctx metric,
  clean exit. `make test` JS suite re-run: ALL PASSED (priority1, priority2,
  TS stripper, SIMD).
- ✅ `benchmarks/agent_bench.js` extended (A9.3): agents-vs-single-loop step
  latency over a fixed-delay mock — agent-layer overhead measured at
  ~0.1–0.2 ms/step (noise) across 6 tool rounds × 3 reps; run-twice registry
  pollution check (1 → 1, stable). Legacy CMA+KV section kept.

## ⬜ Remaining — agents/web follow-ups (small)

- ⬜ Live-provider agent run (only deterministic mocks so far — same standard
  the RLM v1 shipped under).
- ✅ **A7 headless funnel — DONE** (2026-08-20): `sofuu_rt_call_stream` implemented (streaming events for `agent.run` via `onStep` → host callback), `sofuu_rt_cancel` implemented (calls `sofuu.agent.cancel(id)`), per-process agent definitions in `sofuu_rt_new` config (`"agents":[{…}]`), `examples/headless/c_agent.c` (define/run/stream/cancel). 19/19 capi tests, `make headless-test` compiles+runs all 3 C samples, `make abi-check` 23 symbols match.

## 🟡 Track D4 — Rust cross-compile / CI / fuzz

- ✅ `ci.yml`: `cargo test --release` + `make size-check` steps on all three runners (QTSQ-independent build).
- ✅ `crates/fuzz/` scaffold: standalone cargo-fuzz workspace with targets for the Rust SSE / JSON-RPC / npm-spec / TS-strip parsers (`cd crates/fuzz && cargo fuzz run sse -- -max_len=4096`).
- ⬜ musl static builds via cargo + Zig linker (`scripts/cross/`): honor `CC_<target>`/`CARGO_TARGET_*_LINKER` in `build.rs`, reuse the existing per-target libuv/curl statics.

## Explicitly not started (larger roadmap arcs, by design)

- ⬜ **Track B — local inference:** vendor + pin llama.cpp behind `SOFUU_LLM=1`, `sofuu_infer_backend` vtable, `provider:"local"`; then the QTSQ KV bridge (`llama_state_seq_get/set_data`) + CMA↔KV fusion ("infinite memory" demo).
- ⬜ **Track C — custom engine** (Rust-first in `crates/sofuu-core/src/infer/`): scalar f32 forward + PyTorch golden-activation oracle → quantized kernels → greedy parity.
- ⬜ Track A leftovers: ASAN/UBSAN CI job, `examples/test_*.js` → real `tests/` harness, parser fuzzing, soak tests, QTSQ power-loss journaling, JSValue leak audit pass.
- ✅ Streaming tool-call accumulation for Anthropic SSE in `ai.stream` — **landed 2026-08-16** (F4b: content_block tool_use + input_json_delta fragments through the shared setToolCalls seam; every provider runs stream-first now; see the second-pass section above).
- ✅ `callMcpTool` multi-server routing — **landed 2026-08-16** via PLAN-AGENTS.md **A6** (name→owner map; inline/builtin wins clashes with a warning; proven with two real MCP child servers).
- ✅ **RLM scaffold landed** (PLAN-RLM.md R0–R3, 2026-08-13): nested QuickJS sandbox with zero new C, episode state machine, heuristic router + `routing_log.jsonl`, `sofuu.rlm.query/route` headless JS API, `/rlm` chat command (on/off/auto) — 33 rlm unit tests + mock-server E2E (15 assertions) green, binary 1.8MB.
- ✅ **RLM security guardrails** (2026-08-13): frozen whitelist + deterministic Date/random (suspension-safe re-runs), per-eval time slice + episode wall cap, 64KB answer cap, 512-event trace cap with drop counter, 256KB llm-cache cap, anti-loop (3× identical batch → `loop_detected`), sentinel-spoof + credential-redaction proofs, `rlm_snippet`/`rlm_reply` fuzz targets. +11 tests (121 → 132 total green). Open in that plan: embeddings E0–E5 + R4 KV hook + nested-episode recursion (v2).
- ✅ **No-default-model, provider-neutral onboarding** (2026-08-13): `ChatConfig::defaults()` no longer presets ollama/qwen — provider+model start empty; first run auto-opens the provider wizard (`/provider` always available); unconfigured prompts print one guidance line and never call an API; `rt/ai.rs` throws an actionable "No model configured" from `complete`/`stream`/`embed` (silent llama3/nomic substitutions removed); brain `embedText` is local-first (`embedLocal`) with opt-in `embed_provider`/`embed_model`; README/examples no longer imply ollama is built in. Fixtures proven: unconfigured→wizard+guidance, configured-mock→works, rlm E2E unaffected. Bonus fix while verifying: `rt/tui.rs` `tui_truncate_cells` could split a multibyte char (panicked ALL TTY chat on the border frame — pre-existing from the M-track port) → boundary-safe now + regression test.
- ✅ **Headless/embeddable runtime — FULLY DONE** (2026-08-20): PLAN-HEADLESS.md H0–H6 all landed. ✅ H0 `docs/EMBEDDING.md` embedding contract. ✅ H1 `crates/sofuu-capi` (libsofuu) — `sofuu_rt_new/free/call/eval`, `include/sofuu_embed.h`, `make libsofuu` (1.7MB dylib). ✅ H2 hostile-host audit — `process.exit`→catchable `ExitError`, signal installs gated, console routed through log callback, `config_root` replaces `$HOME`. ✅ H4 platform packs — `scripts/dist/{macos,linux,ios,android,all}.sh` + `dist/README.md` + Makefile targets. ✅ H5 samples & docs — `examples/headless/c_embed.c` + `c_rlm.c` (compile+run in CI), `SwiftSample/SofuuBridge.swift` (iOS), `KotlinSample/` (Android JNI: `SofuuBridge.kt` + `jni_bridge.c` + `CMakeLists.txt`), `docs/EMBEDDING.md` per-platform getting-started pages, README "Embed Sofuu" section. ✅ H6 CI gates — `make headless-test` (compile+run both C samples), `make abi-check` (symbol diff vs `scripts/abi_symbols.txt`), `crates/fuzz/fuzz_targets/capi_call.rs` (funnel fuzz). CI + release workflows ship libsofuu tarballs. Agents/sub-agents **landed 2026-08-16** (PLAN-AGENTS.md A1–A6, A8, A9).
- ⬜ **RLM remaining polish** (PLAN-RLM.md): live-model needle proof (`examples/rlm_demo.js` vs Ollama), pty `/rlm` session E2E, real `cargo-fuzz` run (targets compile-only today). ~~mid-request Esc abort~~ ✅ 2026-08-21 · ~~per-model context windows~~ ✅ 2026-08-21 (see the 2026-08-21 section below).
- ⬜ Optional: `fs.rm` recursive delete of non-empty dirs (Node-parity choice; not in any plan).
- ✅ **PLAN-CHAT-FEATURES F1–F11 landed** (2026-08-19): all 9 remaining chat/brain features implemented and verified. F1 `/remember` + `/why` (brain pin + recall explainability), F2 `/resume` (session picker), F3 `@file` mentions (budget-trimmed, manifest-only history), F5 `/share` + `/import` (brain cards), F6 `/cost` + budget caps (pricing table, session/lifetime spend, preflight blocks), F7 `/verify` (second-model pass with word-agreement diff), F8 `/watch` (filesystem mtime/size poller on the 1s tick), F9 `~/.sofuu/hooks.js` (pre/post middleware with 3s timeout + 3-strike disable), F10 ghost completion (sync embedLocal + brain.recall, `__chat_ghost_check` global), F11 `sofuu serve --brain` (HTTP server: /health, /remember, /recall, /share with Bearer auth). `cargo test` 177 green (3 new tests), `make test` green, `make size-check` 2.0MB ≤ 5MB. Config fields: pricing, budget_usd, spend_total_usd, verify, verify_model, ghost. F4a/F4b were already landed.

---

## Test wall state (current)

- `cargo build --release` + `make`: clean (QTSQ optional via `SOFUU_QTSQ_DIR`).
- `make c-only`: FROZEN — no longer builds (retired C files are deleted as migration phases land; M1 removed `io/loop.c`/`io/promises.c`). Dies in M10.
- `cargo test --release`: **158 passing, 0 failing** (2026-08-16, second pass:
  +sandbox/episode/js_api tool-suspension tests for A4 full form, +F4b
  fragment-shape test; was 151 after the TUI pass).
- TUI pty captures (2026-08-16): welcome panel / full agent turn (tool call +
  streamed answer) / `/help` / model picker verified at 100×30, 80×24 and
  62×24 — borders closed, answers fully word-wrapped, no panics.
- `examples/agent_test.js` (2026-08-16, second pass): **ALL PASSED — 53
  checks** over a scripted OpenAI-compatible mock, an Anthropic-wire SSE mock
  and two real MCP child servers (delegation, depth cap, cycle guard,
  budgets, cancel, tool timeout, runMany concurrency, mapContext, memory
  scopes, MCP routing/clash, loadDir, web tool, **A4 sandbox tools**,
  **F4b Anthropic stream-first tool calls + usage**). Memory-scope checks
  require a QTSQ build (auto-skip with a printed SKIP otherwise).
- `examples/web_test.js` (new, 2026-08-16): **ALL PASSED — 39 checks**
  (engine-level DDG/Brave/Tavily/web.open coverage).
- `examples/rlm_mock_test.js`: ALL PASSED again after the `rt/ai.rs` message-serialization fix (it had silently broken against the working tree).
- `priority1_test.js`: 14/15 — the single failure is the pre-existing streaming `headers.get()` capture bug (root-caused during M4; see the M4 section).
- pty batteries: welcome panel, pickers (apply/cancel/filter), ESC abort, Ctrl-C quit, history recall, agent loop E2E — all green; **re-run 2026-08-16 after the agent.js migration + delegate-in-chat wiring**: piped smoke + `/agents` render + delegation line + compact sub-agent result row verified against a mock provider.
- `verify_memory.js`: 30/30 (Rust CMA, cross-compatible with old C brain files).
- MCP: round-trip + >4KB async result + quoted connect — green. (Pre-existing flakiness in the client response path is noted above; the server-side Rust parse is stable.)
- `make size-check`: 2,007,952 bytes ≤ 5MB.
-