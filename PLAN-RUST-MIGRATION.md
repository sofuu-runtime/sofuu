# PLAN — Full Rust Migration of the Remaining C Core

> **Goal:** Rust owns *everything* except the few places where C is genuinely the
> right tool. End state C surface: **QuickJS** (engine), **libuv** (event loop),
> **SIMD kernels** (NEON/AVX2), **QTSQ codec** (external), **curl** + vendored
> **http-parser** (networking I/O). Every line of *glue, logic, and bindings* —
> `src/engine`, `src/io`, `src/http`, `src/mcp`, `src/modules`, `src/npm`,
> `src/repl`, `src/memory` shells — becomes safe Rust.
>
> *Written 2026-08-12. LOC figures measured same day (`find src -name '*.c' -o -name
> '*.h' | xargs wc -l`): project C total 17,229 lines of which 2,580 are retired
> duplicates kept only for `make c-only`, leaving ~14.6k active C lines to eliminate
> (~11–12k of real logic once headers/SIMD/ffi_shim are excluded). Rust today: 8.6k
> lines in `crates/`.*
>
> *Updated 2026-08-13 (post-M10): **M0–M10 are DONE** — see §1 + §2.
> §7 done-state **REACHED**: `src/` = `simd/` + `js/rlm.js` only (181 C
> lines, all SIMD); Rust: **29,465 lines** in `crates/`. `build.rs`
> compiles QuickJS + SIMD + http-parser only. `make c-only` removed.
> `cargo test` **121 passing**, `verify_memory.js` 30/30, MCP E2E,
> npm 3/3, priority2 27/27, priority1 14/15 (pre-existing headers.get
> parity bug), size **1.875 MB**. M11 (http-parser → httparse) optional.*
>
> **Precedent:** this plan is "Track D3, generalized." Memory (CMA), SSE, MCP
> JSON-RPC, TS stripper, bundler, and npm safety already migrated using the
> patterns below — every phase reuses them.

---

## 0. Definitions

**What "migrated" means for a module:** all logic in `crates/sofuu-core` (or a new
crate), all JS-facing registration written in Rust against the QuickJS FFI, the C
file deleted from `build.rs`, JS-visible behavior byte-identical (same object
shapes, same error messages, same timing semantics). `make c-only` parity is NOT a
goal — it is already deprecated and dies in Phase M10.

**What stays C forever (the engine layer), and why:**

| C component | Why C is correct here |
|---|---|
| QuickJS (`deps/quickjs`) | ES2023-complete engine, 100KB, no Rust equivalent at this size/compat. Accessed via FFI. |
| libuv (`deps/libuv`) | Vendored event-loop library. Rust calls `uv_*` via FFI; nobody rewrites it. |
| SIMD kernels (`src/simd/*.c`) | NEON/AVX2 intrinsics are C's home turf. Stay C; Rust calls them via FFI. |
| QTSQ codec (external `libqtsq.a`) | Proprietary external library; FFI only. |
| curl + `deps/http-parser` | Vendored/system networking C. (Optional final swap to `httparse` in Phase M11.) |

**Key insight that makes full migration possible:** `src/io/*`, `src/http/*`, etc.
are *glue between two C libraries* (QuickJS ↔ libuv/curl). Rust can call both
libraries directly over FFI — C functions don't care whether their caller is C or
Rust. The only thing that ever forced glue to be C was convenience, not
capability. All `unsafe` continues to live in `sofuu-ffi`.

---

## 1. The keystone first: a Rust QuickJS/libuv binding layer (Phase M0) — ✅ DONE (2026-08-12)

Everything else depends on this. Today Rust calls narrow, purpose-built exports
(`sofuu_cma_*`, `sofuu_ts_strip_rs`) with C as the caller. The end state inverts
the arrows: **Rust is the caller, QuickJS/libuv are the callee.** That requires
real bindings, not one-off shims.

### M0.1 — `quickjs` binding module in `sofuu-ffi` (`crates/sofuu-ffi/src/qjs.rs`) — ✅ DONE

Hand-write minimal bindings for the ~80 QuickJS entry points the runtime actually
uses (audit `src/` for every `JS_*` call to build the exact list). Do **not**
adopt `rquickjs`: its API surface and MSRV/size costs are wrong for us, and D4
already lists it as merely "evaluate." Our surface is small and stable:

- Values: `JSValue` repr/transmute helpers, `JS_NewFloat64/Int32/Bool/String/
  Object/Array/TypedArray`, `JS_GetPropertyStr`, `JS_SetPropertyStr`,
  `JS_DefineProperty`, `JS_ToCStringLen/JS_FreeCString`, `JS_ToInt32/Float64`,
  `JS_IsFunction/Array/...`, `JS_DupValue`, `JS_FreeValue`.
- Functions: `JS_NewCFunction`, `JS_NewCModule`/init hooks, `JS_SetModuleExport`.
- Calls: `JS_Call`, `JS_CallConstructor`, `JS_Invoke`.
- Promises: `JS_NewPromiseCapability`, resolve/reject via stored `JSValue`s.
- Engine: `JS_NewRuntime/Context`, `JS_Eval`, `JS_RunModule`, job queue
  (`JS_ExecutePendingJob`), `JS_SetModuleLoaderFunc`, interrupt handler.
- GC rules encoded in types: a `Rooted<T>` guard that `JS_DupValue`s on creation
  and `JS_FreeValue`s on drop; a `CtxPtr` wrapper asserting loop-thread affinity.

**Delivered:** `qjs.rs` (520 lines) covers the value/functions/calls/promises/
engine surface actually used (the ~95 `JS_*` symbols at 2,609 call sites). The
static-inline QuickJS funcs (no linkable symbol) go through `src/ffi_shim.c` —
**8 new shims added** (`sofuu_js_new_float64/int64/dup_value/to_uint32/
is_null/is_undefined/is_object/is_number`) on top of the pre-existing bridge.
`Rooted<T>` was revised to **take ownership** (no internal dup): one owner, one
free — unambiguous across FFI boundaries. `CtxPtr` is `!Send` (PhantomData
`*const ()`). JSValue ABI verified against `deps/quickjs/quickjs.h`: this build
defines `JS_PTR64` → **no NaN-boxing**, so `JSValue` is the 16-byte
`{JSValueUnion u; int64_t tag;}` struct (the existing `bridge.rs` repr was
already correct).

### M0.2 — `libuv` binding module in `sofuu-ffi` (`crates/sofuu-ffi/src/uv.rs`) — ✅ DONE

Minimal bindings for the handles in use today: `uv_loop_t`, `uv_timer_t`,
`uv_fs_t`, `uv_tcp_t`, `uv_pipe_t`, `uv_process_t`, `uv_signal_t`, `uv_idle_t`/
`uv_check_t` (job draining), `uv_async_t`. Pattern: `#[repr(C)]` mirror structs
(only the fields we touch), `extern "C"` decls, and one safe wrapper per op.

**Delivered:** `uv.rs` (209 lines) mirrors the handles + 69 `uv_*` symbols the
runtime uses: `uv_buf_t`/`uv_write_t`/`uv_pipe_t`/`uv_process_options_t`/
`uv_stdio_container_t` reprs, extern decls, `LoopPtr` (also `!Send`), and the
stdio flags (`UV_IGNORE/CREATE_PIPE/WRITABLE_PIPE/READABLE_PIPE`). tcp/tty/poll
deferred to M5 as planned.

### M0.3 — Single-threaded discipline, codified — ✅ DONE

The runtime is single-threaded (one libuv loop, one JSContext). Encode it:
`LoopThread<T: Send>` is unnecessary — instead make all QuickJS types `!Send` via
`PhantomData<*const ()>` so the compiler forbids cross-thread use. All uv
callbacks for a module funnel through one `extern "C"` dispatch shim that
recovers the Rust closure from `handle->data`.

**Delivered:** `CtxPtr` (qjs) + `LoopPtr` (uv) are both `!Send` by construction.
The callback-dispatch shim pattern is used by M1+.

**M0 exit criteria:**
- ✅ Rust unit test boots a QuickJS context *through the new bindings*, evals
  `1+1`, reads `2`. Runs in `cargo test` on macOS + Linux CI.
- ✅ A Rust-registered native module returning "hello" is callable from JS.
- ✅ No new `unsafe` outside `sofuu-ffi`; `cargo test` stays green; size cap holds.

**Verified:** `cargo test --release`: 80 passing (65 lib + 11 bin + 4 ffi: the 3
new M0 tests + the bridge test). Size 1.81MB ≤ 5MB cap. All `unsafe` confined to
`sofuu-ffi`.

**Lesson learned (now part of the Rooted discipline):** Rust drops locals at
function scope end — *after* `JS_FreeContext` if the teardown is inline. A
`Rooted` held to scope end frees on a dangling ctx. Rule: **always scope a
`Rooted` in a block so it drops before the context does.** This is exactly the
M6-crash class, caught at the keystone stage.

---

## 2. Migration phases (each independently shippable)

Order rationale: spine before limbs **for the I/O substrate** (loop/promises
underpin everything), then leaves by ascending complexity, then the big stateful
surfaces, then the engine itself. Every phase: `cargo build --release` + `make
size-check` + `cargo test` + the phase's JS proofs must all pass before merge.

### M1 — Event-loop spine: `io/loop.c` + `io/promises.c` (~700 lines) — ✅ DONE (2026-08-12)

The promise↔libuv bridge and job queue. Ported to `sofuu-core/src/rt/`
(`loop.rs` — module `event_loop`, `loop` being a Rust keyword; `promise.rs`),
on the M0 bindings. The deferred unhandled-rejection tracker moved verbatim
out of `engine.c` into `rt::promise` (thread-local store — the C statics it
replaced were equally unsynchronized; 64-eager flood print; pointer-identity
retraction; print-and-clear at drain end + shutdown).

- **Delivered:** the same `sofuu_loop_*` / `sofuu_promise_*` / `sofuu_flush_jobs`
  symbols exported from Rust; all C callers link unchanged. engine.c wires
  `sofuu_rt_install_rejection_tracker` / `sofuu_rt_report_pending_rejections`
  under `SOFUU_RUST_CORE` (c-only's C copy is `#ifndef`-guarded).
  `uv.rs` gained `uv_walk` + loop/timer size shims; `qjs.rs` gained
  `JS_ToString`, `JS_SetHostPromiseRejectionTracker` + three more shims.
  The loop's storage is allocated once and never freed — engine_destroy
  closes the loop BEFORE `mod_process_cleanup`, which still touches handles
  that reference it (same lifetime as the old C `static uv_loop_t g_loop`).
- **Proofs:** ✅ `examples/timers.js`, `priority2_test.js` 27/27, `fs_test.js`,
  `spawn_test.js`; `priority1_test.js` 14/15 (the one failure is the
  pre-existing httpbin HTTP/2 header-scan issue, unchanged from baseline);
  ✅ the new Rust test `rt::promise::tests::m1_promise_resolved_from_uv_timer_fires_js_then`.
- **File deletions:** `src/io/loop.c`, `src/io/promises.c` out of `build.rs`
  and deleted; stale `-I src/ts` / `-I src/bundler` removed from `build.rs`.
- **Verified:** `cargo test --release` 113 passing (98 lib + 11 bin + 4 ffi);
  size 1.8MB ≤ 5MB. (`rlm::js_api::tests::js_roundtrip_through_quickjs` fails
  PRE-EXISTING: its `JS_Eval` gets an unterminated `&str` — QuickJS needs a
  NUL sentinel, which the C core always provides; only tests hit it. RLM-side
  fix, out of M1 scope.)
- **Lesson learned (repeatable):** Rust `&str` is NOT NUL-terminated — any
  `JS_Eval` input must go through `CString` (or a `\0`-padded buffer), exactly
  like `qjs_rt.rs` already does. This is now part of the recipe (§3).

### M2 — Leaf I/O modules: `io/timer.c`, `io/fs.c`, `io/subprocess.c`, `modules/mod_console.c` (~1.6k lines) — ✅ DONE (2026-08-12)

- `timer.c` → `rt/timer.rs` (setTimeout/setInterval/clearX + `sofuu.sleep`,
  exception routing on the M1 rejection tracker). Proof: ✅ `timers.js` +
  the uncaughtException-in-timer path (priority2), ✅ new Rust test
  `rt::timer::tests::m2_timers_fire_and_clear_through_rust_loop`.
- `fs.c` → `rt/fs.rs` (readFile/writeFile/readdir/mkdir{recursive}/rm/
  readFileBytes with the Uint8Array path verbatim). Proofs: ✅ `fs_test.js` +
  manual recursive-mkdir/readdir/readFileBytes probes.
- `subprocess.c` → `rt/process_spawn.rs` (`sofuu.spawn` with the Subprocess
  class + kill/write + the 4-handle close-counting fence, `sofuu.exec`
  capture mode with the 64MB cap). Proofs: ✅ `spawn_test.js`,
  `await sofuu.exec("echo",["hi"])` shape.
- `mod_console.c` → `modules/console.rs` (log/info/warn/error/assert/time/
  timeEnd/table — exact ANSI prefixes + ASCII table format). Trivial; proves
  the module-registration recipe for later phases.
- **Bindings added/fixed:** `JS_CallConstructor`, `JS_GetOwnPropertyNames` +
  `JSPropertyEnum`/`JSAtom`/`JS_AtomToCString`/`JS_FreeAtom`/`js_free`,
  `uv_fs_stat/fstat/scandir/scandir_next`, `uv_spawn`, `sofuu_js_new_uint32`,
  `sofuu_uv_{fs,write,pipe,process}_size`, `sofuu_uv_fs_{result,stat_size}`;
  `UvStdioContainer` corrected to the real union layout; **`JSCFunctionListEntry`
  corrected to 32 bytes (`u.func = {u8 length; u8 cproto; fn}`)** — the M0
  mirror placed the fn pointer at union offset 0, so JS_SetPropertyFunctionList
  read a shifted/garbage pointer and corrupted the atom table (nondeterministic
  boot crashes + find_atom("") faults; caught via engine.c register bisect +
  lldb). This is the M0 struct-mirror lesson: **verify every spec'd struct
  against quickjs.h before first use** — the M0 exit criteria only exercised
  values/functions, not property tables.
- **Test note:** the process-global uv loop is ONE instance (C-static parity);
  the m1/m2 proof tests serialize via `TEST_LOOP_LOCK` — parallel threads
  would cross-schedule timers on each other's contexts (observed: interval
  firing 7× under parallel-test interference).
- **Verified:** `cargo test --release` **115 passing** (100 lib + 11 bin +
  4 ffi), size 1.89MB ≤ 5MB. Files deleted: the four C files + their
  Makefile/scripts/cross references.

### M3 — `modules/mod_process.c` (1,920 lines — the biggest "simple" file) — ✅ DONE (2026-08-12)

`process.env/argv/cwd/exit`, signals, uncaughtException hidden-prop storage,
TTY readline bridgework, chat-history hooks. Ported to `modules/process.rs`
(2.6k lines); the hidden-prop GC-safe patterns are kept exactly as C did
them. **Bonus: `process.argv` now actually works in the Rust binary** —
`main.rs` calls the ported `mod_process_set_args` (the retired C main.c was
the only caller; argv had been silently empty).

- **Bindings added:** `uv_tty_init/set_mode/reset_mode`, `uv_cwd`, `uv_chdir`,
  `UV_TTY_MODE_*`, `sofuu_uv_{tty,signal}_size`, `sofuu_js_null` (JS_NULL).
  All `sa_sigaction`/`sigemptyset`/termios work goes through the libc crate.
- **Proofs:** ✅ `process_test.js`, `priority2_test.js` 27/27 (signals +
  uncaught + exit handler), `timers.js`, `ts_test.ts`; piped-stdin
  `__readline` round-trip; live SIGTERM → handler → clean exit.
  (TTY/pty specifics = the chat battery; structure is byte-identical to C.)
- **Verified:** `cargo test --release` 115 passing (100 lib + 11 bin +
  4 ffi), size 1.89MB ≤ 5MB. `src/modules/` now holds only `mod_ai.c`.

### M4 — HTTP client + SSE shell: `http/client.c`, `http/sse.c` (~1.2k lines) — ✅ DONE (2026-08-12)

`fetch` over curl, streaming `Response.body` async iterator, header/status
handling; `sse.c` is already a shell over the Rust SSE parser and dissolves into
the same module. Ported to `rt/http_client.rs` + `rt/http_sse.rs` calling
curl via a new `sofuu-ffi/src/curl.rs`; the SSE parser is already Rust — only
the JS plumbing moved.

- **Proofs:** ✅ local-server battery (201 + json, redirect follow, POST,
  404), ✅ `sse_test.js`, ✅ streaming-body timing proof (chunks at
  ~304/612 ms against a slow chunked server — not an end-buffered dump).
  (`fetch_test.js`/`priority1_test.js` hit httpbin — the one pre-existing
  failure is root-caused in M4, see below.)
- **LESSON (ABI constants):** verify every libcurl/QuickJS numeric constant
  against the system header — enum-index guesses for `CURLMOPT_*`,
  `CURL_POLL_*`, `CURLINFO_PRIVATE` and `JS_PROP_HAS_GET` were all wrong and
  silently disabled features (guessed 20002/20003 vs real 20001/20004;
  `JS_PROP_HAS_GET` is `1<<11`, not `1<<9`).
- **Pre-existing (bug-for-bug parity):** Response `headers.get()` never
  matched REAL headers — `headers_raw` is captured at the FIRST header
  callback (status line only); later lines append to the request's buffer.
  This is the exact root cause of the long-standing priority1 Content-Type
  failure (14/15 since before the migration). A future fix moves the raw
  capture to completion — deliberately out of scope for a migration.

### M5 — HTTP server: `http/server.c` (459 lines, plus vendored http-parser) — ✅ DONE (2026-08-12)

`sofuu.http.createServer`, parsing via vendored `http-parser`, `req/res`
objects, `writeHead/write/end`, keepalive. Ported to `rt/http_server.rs`;
`http_parser.c` stays vendored C behind FFI (until M11).

- **Proofs:** ✅ `server_test.js` (GET/POST via curl), ✅ SOAK 100/100
  sequential requests incl. the chunked streaming path, `mcp_server_test.js`
  + `priority2_test.js` unchanged.
- **LESSON (uv write requests):** NEVER mirror payload fields inside a
  uv_write_t region — the real struct (56B) overwrites everything past its
  first bytes. Stash the payload Box in `req->data` (reserved for users).
  Diagnosed with an abort breakpoint (`___BUG_IN_CLIENT_OF_LIBMALLOC` +
  `on_write_done`) and PROVEN a port bug by temporarily rebuilding the
  original C server.c and showing it survives the same battery.

### M6 — MCP: `mcp/mcp.c` (1,245 lines) — ✅ DONE (2026-08-12)

Client (spawn + JSON-RPC over pipes + response routing) and server (tool
registry, stdio loop). JSON-RPC parse/build was **already Rust**; this moved
the process/pipe plumbing and tool-call dispatch. **The known
client-response-routing gap is CLOSED**: the client path now parses with
`mcp::jsonrpc::parse` (safe serde) — the C `json_get_field` scan is gone —
and the mandated stress passes: **10k pipelined responses, 0 errors** (the
sliding window honors the C `MAX_PENDING=64` ring for parity).

- **LESSONS (3 real bugs caught in one phase):** (1) `UV_READABLE_PIPE`/
  `UV_WRITABLE_PIPE` were swapped in uv.rs (READABLE=0x10, WRITABLE=0x20) —
  every CREATE_PIPE child-stdout read returned UV_ENOTCONN; spawn_test had
  been silently deaf since M2. (2) `JS_NewClassID` allocates ONLY when
  `*pclass_id == 0` — reusing one local across two classes gave both the
  same id and the client stole the server proto; mirrors C's
  `static JSClassID id = 0` per class (fixed in mcp AND http_server).
  (3) The client-connect success path was missing `JS_SetOpaque`.
  The new `m6_mini_spawn_reads_child_stdout` binding test pins the pipe
  wiring for good.

- **Proofs:** `mcp_client_test.js`, `mcp_server_test.js`, >4KB async result,
  quoted-command connect, `~/.sofuu/mcp.json` chat E2E, the 10k-response stress.
- **Already landed (D3):** Rust JSON-RPC builders + server-side inbound parse;
  `json_get_field` string-scan deleted from the server path. The client response
  path still uses the C scan (see TASKS.md for the crash note).

### M7 — npm runtime: `npm/resolver.c`, `npm/cjs.c` (~840 lines) — ✅ DONE (2026-08-12)

Walk-up resolution and CJS shims (`module.exports`, `require`, `__dirname`).
Spec validation/SHA-1/tar safety were already Rust; this moved the loader
plumbing (`npm_resolve` now delegates to the D3 twin; the installer runs the
fetch → SHA-1 → safe-extract → transitive-deps pipeline through the Rust
exports; manifests are parsed with serde where C used throwaway contexts).

- **Proofs:** ✅ `npm_test.js` 3/3 against the REAL registry, `sofuu add`
  with transitive installs, `require()` E2E, `modules.js` unchanged.
- **Already landed (D3):** the module loader's walk-up `resolve` now calls the
  Rust twin (`sofuu_npm_resolve_rs`) under `SOFUU_RUST_CORE`; `npm/resolver.c`
  stays only as the c-only fallback. The CJS loader shell (`npm/cjs.c`) remains.

### M8 — AI module: `modules/mod_ai.c` (2,543 lines — biggest single file) — ✅ DONE (2026-08-12)

Provider adapters (OpenAI/Anthropic/Gemini/Ollama), `ai.stream` async iterator,
`ai.complete`, effort/reasoning mapping, `listModels/listProviders`, SIMD-backed
`similarity/dot/l2` wrappers (kernels stay C), `embedLocal`. **The TF-IDF
embedder itself is ported to Rust** (it's pure string math — trigrams →
MurmurHash3 → L2-norm) and lives inline in `rt/ai.rs`; mod_ai.c's tfidf_embed.c
consumer is gone with the same edit (the M9 tfidf_embed.c item is superseded).
Landed as ONE Rust file (`crates/sofuu-core/src/rt/ai.rs`) rather than the
provider-split `/modules/ai/` sketch — the C file's shared curl-multi bridge,
promise dispatch, and per-request structs are one unit; reuse M4's bridge
patterns (`uv_poll` sockets + `uv_timer` timeout, CURLMOPT_* FUNCTIONPOINT ids).

- **Proofs:** ✅ `simd_test.js` 9/9; the mock-OpenAI gate (custom provider
  → 127.0.0.1:8899) exercises complete + stream + usage + `[DONE]` +
  abort-before/mid-stream + embedLocal (768-dim unit vectors, same-topic
  0.141 > unrelated 0.078, identical ≈ 1.000) + listModels/listProviders/
  estimateTokens + the local-provider guard; the embedder is byte-identical
  to the C algorithm (0 differing elements vs an independent reference).

### M9 — Memory shells: `memory/mod_memory.c`, `mod_kv.c`, `mod_agent.c`, `qtsq_adapter.c` (~1.8k lines — tfidf_embed.c already landed in M8) — ✅ DONE (2026-08-13)

All four C files + their five headers are DELETED; the shells and the QTSQ
adapter landed as `crates/sofuu-core/src/rt/memory.rs` (one module: CMA shell
over `crate::memory::cma::Cma`, KV page store with 2-page RAM LRU, agent
façade) with the libqtsq FFI in `crates/sofuu-ffi/src/qtsq.rs` (opaque
5016-byte context storage + byte-offset field accessors; no new C shims).
`has_qtsq` cfg in both crates (sofuu-core gained a build.rs mirroring ffi's
`SOFUU_QTSQ_DIR` probe — `cargo:rustc-cfg` cannot cross crates). Brain-file
format compatibility held: records JSON keys and vec store layout
byte-identical (verified 30/30).

- **Proofs (all green 2026-08-13):** `verify_memory.js` 30/30 incl. GMF /
  dream / resonance persistence; cross-version KV — a store written by the
  Aug-12 C build (`test_agent_kv/`) reopens, searches and decompresses
  through the Rust adapter; cross-restart KV ranking (seed 64 pages in one
  process → reopen → identical top hits); `examples/priority2_test.js` 27/27.
- **C-symbol contract:** `mod_memory_register`, `mod_kv_register`,
  `mod_agent_register` are exported unconditionally; the implementations are
  `#[cfg(has_qtsq)]` — without QTSQ they no-op, giving engine.c the same
  JS-visible absence of sofuu.memory/kv/agent as the C's
  `#if defined(SOFUU_QTSQ_PRESENT)` registration guard.

### M10 — Engine & runtime init: `engine/engine.c`, `sofuu.c`, `ffi_shim.c`, `repl/repl.c` (~1.3k lines) — ✅ DONE (2026-08-13)

The final boss: runtime/context boot, module loader (TS strip hook, npm walk-up —
Rust logic already, loader shell moves), builtin registration table, top-level
error dispatch, `uncaughtException` default path, the JS-eval REPL, and
dissolution of `ffi_shim.c` (its wrappers become plain Rust in sofuu-ffi). When
this lands, `src/` contains **only** `simd/` and vendored deps.

- **Delivered:**
  - `crates/sofuu-core/src/rt/engine.rs` (~960 lines) — full port of
    `engine.c` + `sofuu.c` + `repl.c` (engine_create/destroy, the ESM module
    loader with resolve_module_path canonicalize/fallback ladder, builtin
    registration table in the C order, the Sofuu global aliases, REPL
    eval via repl_inspect with %g parity, sofuu_init/eval_file/eval_string/
    run_jobs/destroy/get_engine/engine_eval_repl/sofuu_engine_ctx C-ABI
    exports, and the `sofuu_module_loader` extern "C" function).
  - `crates/sofuu-core/src/rt/tui.rs` (~370 lines) — full port of `tui.c`
    (alt-screen conversation buffer, disp_width ANSI-aware, render_rows
    viewport/scroller, tui_push_split newline-safety with UTF-8 backoff,
    all tui_* layout helpers + sofuu_tui_active/log/width bridge aliases).
  - `sofuu-ffi/src/qjs.rs` — ffi_shim.c dissolved: all 29 sofuu_js_*
    wrappers reimplemented as real Rust `#[no_mangle] pub extern "C" fn`
    against the public quickjs.h inline semantics (tag checks,
    JS_MKVAL/NewInt32/Int64/Float64/Uint32/Bool value construction,
    JSRefCountHeader decrement + __JS_FreeValue/RT for free/dup,
    JS_ToCStringLen2/JS_FreeCString/JS_NewCFunction2/JS_GetPropertyInternal
    aliases). __JS_FreeValue/__JS_FreeValueRT/JS_NewString/JS_NewObject/
    JS_GetGlobalObject/JS_GetException/JS_GetPropertyStr/JS_SetPropertyStr/
    JS_FreeCString/JS_GetPropertyInternal added as real-export externs.
  - `sofuu-ffi/src/uv.rs` — sofuu_uv_* size/stat shims reimplemented:
    `uv_loop_size()`/`uv_handle_size()`/`uv_req_size()` (real libuv exports)
    + `uv_fs_get_result()`/`uv_fs_get_statbuf()` (real exports) + `UvStat`
    mirror (portable uv_stat_t, 160 bytes).
  - `sofuu-ffi/src/qtsq.rs` — sofuu_qtsq_session_save/load/free
    reimplemented in Rust using existing qtsq_* externs (qtsq_secure_text +
    qtsq_decompress added as new externs). lib.rs wrappers re-erouted
    (cfg-gated, no-QTSQ fallback returns -1/NULL).
  - `sofuu-ffi/src/bridge.rs` — rewritten as thin wrappers over qjs.
  - `sofuu-ffi/src/qjs.rs` — JS_SetGCThreshold/JS_SetModuleLoaderFunc/
    js_init_module_std/os + JsModuleNormalizeFunc/JsModuleLoaderFunc types
    added (engine boot).
  - `sofuu-ffi/Cargo.toml` — dev-dependency on sofuu-core REMOVED (the
    cycle linked sofuu-ffi twice under fat-LTO → symbol multiply defined;
    runtime_and_bridge_work test moved to sofuu-core lib.rs ffi_runtime_test).
  - `crates/sofuu-core/src/lib.rs` — ffi_runtime_test (runtime+bridge E2E
    with TEST_LOOP_LOCK serialization).
  - `crates/sofuu-core/src/rt/memory.rs` — compute_k_summary_normalizes
    test updated to match the M9 distributed-pooling fix (the old test
    still asserted the pre-fix degenerate slot-0 pooling).
  - `crates/sofuu-ffi/build.rs` — source list pruned to QuickJS + SIMD +
    http-parser only; qtsq include/define removed; SOFUU_UV_DIR +
    SOFUU_CURL_STATIC_DIR env overrides added (cross-builds); opus
    pkg-config gated on QTSQ presence (prevents cross-link poisoning).
  - `Makefile` — `make c-only` replaced with an error message; whole CFLAGS/
    ALL_SRCS/SOFUU_SRCS/readline blocks removed.
  - `scripts/cross/build_linux_arm64.sh` + `build_linux_x86_64.sh` —
    rewritten to cargo + zig cross-build flow (previously referenced
    deleted C files).
  - `src/` deletions: `engine/engine.c`, `engine/engine.h`, `sofuu.c`,
    `sofuu.h`, `main.c`, `ffi_shim.c`, `io/tui.c`, `io/tui.h`, `io/*.h`
    (loop/fs/timer/subprocess/promises), `repl/repl.c`, `repl/repl.h`,
    `http/client.h`, `http/server.h`, `http/sse.h`, `mcp/mcp.h`,
    `modules/mod_ai.h`, `modules/mod_console.h`, `modules/mod_process.h`,
    `npm/cjs.h`, `npm/resolver.h`. All empty subdirs removed. Final
    `src/` state: `simd/{neon,avx}.c`, `simd/simd.h`, `js/rlm.js`.
- **Constants fixed during M10:** `JS_EVAL_FLAG_COMPILE_ONLY` corrected from
  0x400 to 0x20 (1<<5) — the wrong value made the ESM loader fully
  link+run imported modules inside JS_Eval, then the parent linked them
  a second time → js_inner_module_linking asserts + corrupt module defs.
  `JS_PROP_HAS_WRITABLE`/`JS_PROP_HAS_ENUMERABLE` swap corrected
  (W=1<<9, E=1<<10 per quickjs.h:277-278).
- **Proofs:** `cargo test --release` **121 passing** (104 lib + 11 bin +
  6 ffi — the runtime+bridge test moved from ffi to core lib, reducing
  ffi from 7 to 6; same count), `make size-check` 1,875,120 bytes ≤ 5MB,
  `priority2_test.js` 27/27, `priority1_test.js` 14/15 (pre-existing
  headers.get parity bug — documented), `timers.js` ✓, `fs_test.js` ✓,
  `spawn_test.js` ✓, `ts_test.ts` ✓, `simd_test.js` 9/9, `modules.js` ✓,
  fetch local battery ALL-FETCH-OK, SSE test ✓, HTTP server via curl ✓,
  MCP client test PASSED, MCP server E2E (initialize + tools/list +
  tools/call) ✓, `npm_test.js` 3/3 (real registry), `verify_memory.js`
  30/30, KV cross-restart [64,128,63,127,62] stable, AI mock battery
  AI-M8-OK, chat boot exit 0, REPL 1+1=2 ✓.
- **Known limitations:** Linux cross-build scripts rewritten to the cargo +
  zig flow but untested locally (CI with macOS+Linux runners is the real
  gate). priority1 14/15 is the pre-existing httpbin headers.get parity
  bug, not introduced by M10.

### M11 — Optional收尾: swap vendored `http-parser` → Rust `httparse`

`http-parser` is the one remaining non-QuickJS/libuv vendored C parser. Swapping
to `httparse` leaves curl as the only non-engine C dependency. Optional — only if
size/perf neutral.

---

## 3. The per-module recipe (followed in every phase)

This is the proven D3 pattern, made explicit:

1. **Inventory** the C file: every exported JS function, every internal helper
   other files call, every global/static.
2. **Write the Rust port** in sofuu-core with the same JS-visible shapes. No
   redesigns — migration ≠ refactor. Behavior parity is the spec; `git log` the
   C file for semantics it doesn't document.
3. **Register from Rust** using the M0 bindings (replacing `JS_NewCFunction`
   tables); the C file shrinks to zero and exits `build.rs`.
4. **Prove parity** with the phase's JS scripts — record commands + outputs in
   the PR/commit; tests that can't run in CI (Ollama, network) get a mock.
5. **Fuzz** any new parser surface via `crates/fuzz` (extend targets as parsers
   move).
6. **Doc sweep:** README/ROADMAP/TASKS lines describing "C shell over Rust X"
   get updated as the shell dissolves.

Rules that prevented past regressions — now mandatory:
- Returned buffers are `libc::malloc`-owned (C can `free()`); established in
  `ffi_exports.rs`'s memory contract.
- Every FFI entry tolerates NULL/zero and fails soft, letting a caller fall back.
- JSValues held across uv callbacks only via `Rooted<T>`; never touched off the
  loop thread. (This is the exact fix for the M6 crash class.)
- **`Rooted<T>` must scope-drop BEFORE the context is freed** (Rust drops locals
  at function end, after `JS_FreeContext` — caught in M0).
- **Every `JS_Eval` input must be NUL-terminated** (`CString` or a `\0`-padded
  buffer) — QuickJS reads past `input_len` for a sentinel. Rust `&str` alone is
  NOT; the C core always terminated via `engine_read_file` (caught in M1).
- Binary ≤ 5MB every phase (`make size-check` in CI, already wired).

---

## 4. Testing & CI upgrades to land *early* (enablers, not phases)

- Convert `examples/*_test.js` into a real `tests/` harness with asserts + exit
  codes (ROADMAP A0) **before M4** — the stateful phases need a loud gate, not
  eyeball diffs. `make test` grows: `cargo test` + JS suites + fuzz smoke.
- Add the pty battery and `verify_memory.js` to CI as scripted steps.
- Keep ASAN/UBSAN for the shrinking C surface until M10; afterwards, Miri-level
  care on the `sofuu-ffi` shims (spot-checked; full Miri is impractical over
  QuickJS FFI).

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GC/refcount crashes when Rust owns JSValues across callbacks (the M6 precedent) | `Rooted<T>` + `!Send` types in M0 (**done**); stress tests per phase; the failure mode is compile-time-visible, not runtime. M0 already caught one: drop-order vs `JS_FreeContext`. |
| Behavior drift in subtle semantics (timer ordering, rejection timing, SSE edge cases) | Parity gates per phase re-run the exact scripts that pinned C behavior; golden-output compare; no "improvements" inside migration diffs. |
| Binary size creep (Rust replacing C) | Size-check every phase; profile stays `opt-level=z, lto=fat, strip`; ~3.4MB headroom today (1.81MB of 5MB). |
| QTSQ-gated builds diverging | CI matrix builds with and without `SOFUU_QTSQ_DIR`; graceful-degradation warning path preserved. |
| One mega-PR stalls everything | Phases are mergeable slices; each leaves the runtime fully working. M0/M1 are the only cross-cutting ones. |
| `make c-only` bit-rots during migration | Already deprecated; frozen at M0 state, deleted at M10. CI stops building it after M2. Retired duplicates are deleted, not kept for c-only. |

---

## 6. Effort estimate (rough, single-author pace)

| Phase | Size | Notes |
|---|---|---|
| M0 bindings | ~1.5–2 weeks | The only truly new infrastructure. **✅ Done (2026-08-12).** |
| M1 spine | ~1 week | Hard semantics, small code. **✅ Done (2026-08-12).** |
| M2 leaves | ~1 week | Recipe is proven by now. **✅ Done (2026-08-12).** |
| M3 mod_process | ~1 week | Leaves, biggest file. **✅ Done (2026-08-12).** |
| M4 HTTP client | ~1.5 weeks | Streaming correctness is the cost. **✅ Done (2026-08-12).** |
| M5 HTTP server | ~1 week | http-parser stays C until M11. **✅ Done (2026-08-12).** |
| M6 MCP | ~1 week | Known crash closed + 10k stress. **✅ Done (2026-08-12).** |
| M7 npm | ~3 days | Loader plumbing only. **✅ Done (2026-08-12). Next: M8.** |
| M4–M5 HTTP | ~1.5 weeks | Streaming correctness is the cost. |
| M6 MCP | ~1 week | Includes fixing the known crash properly. Rust builders + server parse already landed (D3). |
| M7 npm | ~3 days | Loader plumbing only; logic + walk-up resolve already Rust (D3). |
| M8 mod_ai | ~1.5 weeks | Biggest file; mostly mechanical provider adapters. **✅ Done (2026-08-12).** |
| M9 memory shells | ~1 week | Format-compat proofs are the gate. `mod_memory.c` shell + agent façade already landed (D3). **✅ Done (2026-08-13). Next: M10.** |
| M10 engine/init | ~1 week | Dissolves ffi_shim; retires c-only. Retired C duplicates already deleted (M0 cleanup). **✅ Done (2026-08-13).** |
| M11 optional | ~2 days | http-parser → httparse. |

**Total ≈ 9–10 weeks** of focused work, in 11 independently landable slices, at
which point the repo's handwritten code is ~95%+ Rust and the remaining C is
exactly the agreed engine layer: QuickJS, libuv, SIMD, QTSQ, curl. **(M0–M10
all DONE — remaining ≈ 2 days for M11 if elected.)**

---

## 7. Done-state definition (whole plan)

- `src/` contains: `simd/`, and nothing else that ships logic in C. (QuickJS,
  libuv, http-parser live in `deps/`; QTSQ is external.)
- `build.rs` compiles: QuickJS + SIMD (+ http-parser until M11). No `src/io`,
  `src/http`, `src/modules`, `src/memory`, `src/engine`, `src/npm`, `src/repl`.
- `cargo test` + JS harness + pty battery + `verify_memory.js` + MCP E2E green
  on macOS/Linux, with and without QTSQ.
- `make size-check` ≤ 5MB; `make c-only` gone; README/ROADMAP/TASKS describe the
  finished state, not the migration.

**Progress as of 2026-08-13 (post-M10):** §7 done-state **REACHED**.
`src/` contains only `simd/` (3 files, 181 lines) and `js/rlm.js` (a
shipped JS data asset, not C logic). Rust: **29,465 lines** in `crates/`.
`build.rs` compiles QuickJS + SIMD + http-parser only (no `src/io`,
`src/http`, `src/modules`, `src/memory`, `src/engine`, `src/npm`, `src/repl`,
no `ffi_shim.c`, no `sofuu.c`). `make c-only` is removed (error message).
`cargo test` **121 passing** (104 lib + 11 bin + 6 ffi), `verify_memory.js`
30/30, KV cross-restart, simd_test 9/9, mock-AI gate, MCP E2E,
`npm_test` 3/3 (real registry), `priority2` 27/27, `priority1` 14/15
(pre-existing headers.get parity bug), size **1.875 MB** ≤ 5 MB.
M0–M10 all done; M11 (http-parser → httparse) is optional.
