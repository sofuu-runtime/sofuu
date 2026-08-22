# Sofuu Embedding Contract (C ABI v2)

> **Status:** H0–H6 of `PLAN-HEADLESS.md` — this document is the contract every
> headless/embedding phase implements. Code follows this doc, not the other
> way around. Written 2026-08-18; audited 2026-08-20 against the post-M10 codebase
> (the C core is retired — all audit targets below live in `crates/`). All
> phases (H0–H6) are landed.
>
> **The pitch:** any device or app — iOS, Android, desktop, server, edge —
> embeds Sofuu with **all features and zero TUI**. RLM, embeddings,
> brain/memory, HTTP, MCP, the agent loop — all reachable from a host
> process through a small stable ABI. QuickJS is a pure interpreter (no
> JIT), so iOS App Store executable-memory policy is satisfied by design.

---

## 0. Where things stand today (verified)

- Public C ABI is six functions (`sofuu_init` / `sofuu_eval_file` /
  `sofuu_eval_string` / `sofuu_run_jobs` / `sofuu_destroy` /
  `sofuu_get_engine` + `sofuu_engine_ctx` / `engine_eval_repl`), exported
  from `crates/sofuu-core/src/rt/engine.rs`. `sofuu_init` takes **no
  config** — that is what v2 fixes.
- Safe Rust wrapper: `sofuu_ffi::SofuuRuntime` (`crates/sofuu-ffi/src/lib.rs`)
  — init/eval_file/eval_string/run_jobs/engine_ctx/eval_repl, freed on Drop.
- Shipped JS in **every** engine context: `src/js/{agent,rlm,web}.js`
  (`sofuu.agent`, `sofuu.rlm`, `sofuu.web`); `rt/ai.rs` registers
  `sofuu.ai`; `rt/memory.rs` registers `sofuu.memory`/`kv`/`agent.create`
  (gated on the QTSQ codec being linked — `cfg(has_qtsq)`).
- Known hostile-host facts (the H2 work list, located — all resolved by
  H2, kept here as the audit record):
  - `libc::exit` on reachable paths: `modules/process.rs` — `process.exit`
    (raw `libc::exit(code)`), and the SIGINT/SIGTERM 130/143 fallbacks in
    `process_dispatch_pending_signals`.
  - Signal installs: `install_sigaction(SIGINT/SIGTERM)` on first
    `process.on(...)`, plus a `uv_signal_t` SIGWINCH watcher.
  - `$HOME` derivations: `chat.rs` config dir (`~/.sofuu`),
    `rlm/js_api.rs` routing log (`~/.sofuu/routing_log.jsonl`),
    `src/js/agent.js` default brain path (`$HOME/.sofuu_brain.qtsq`).
  - stdout/stderr writes: `modules/console.rs` prints straight to the
    process streams.
- The libuv loop is **process-global**: one `G_LOOP` allocated once per
  process and never freed (`rt/loop.rs`); tests serialize on
  `TEST_LOOP_LOCK` (`rt/mod.rs`). QuickJS itself supports multiple
  independent runtimes/contexts per process.

---

## 1. Process rule — the library never terminates its host

The library never calls `exit`/`abort` on any path reachable from the
embedding API. Errors **return**; they never kill.

- `process.exit` from JS, when embedded (`embedded:true` in the
  `sofuu_rt_new` config), throws a catchable JS `ExitError` instead of
  terminating (H2).
- The SIGINT/SIGTERM 130/143 re-exits and any REPL/CLI exits are CLI-gated
  — unreachable headless, asserted by the H2 audit test.
- Enforcement: CI grep over `src/` + `crates/` for `exit(`/`abort(` with an
  allowlist of CLI-only files (H6). A crash or exit is a failed fuzz run of
  the funnel (`capi_call`).

**Status:** H1 ✅ done (2026-08-18) — `crates/sofuu-capi` +
`include/sofuu_embed.h` + `examples/headless/c_embed.c` + `make libsofuu`
+ `make headless-test`. H2 ✅ done (2026-08-19) — embedded `process.exit`
throws a catchable `ExitError` (host catches via `sofuu_rt_eval`;
`libc::exit` remains only on non-embedded CLI paths), signal installs
gated behind `enable_signals`, `console.*` routes through a host log
callback (`sofuu_embed_set_log_cb`), `config_root` plumbed through chat
config/history, RLM log routing, and `agent.js`, and `api_keys`/
`brain_path` init-config keys resolve ahead of env vars. Tracked in
`PLAN-HEADLESS.md`.

## 2. Threading model

- **One `SofuuRuntime` is owned by one thread.** All calls on a runtime
  must come from that thread — QuickJS and libuv are single-threaded.
  The handle is `Send` (move it to its thread once) but not `Sync`.
- **Multiple runtimes per process: supported**, each on its own thread —
  with the caveat below.
- **The event loop is centralized.** `rt/loop.rs` holds one process-wide
  libuv loop (`G_LOOP`), mirroring the retired C `static uv_loop_t g_loop`.
  Two runtimes pumping it concurrently would schedule each other's timers
  on the wrong context (this is exactly what `TEST_LOOP_LOCK` serializes
  in the test suite). Until H1 proves otherwise, treat concurrent
  multi-instance I/O as **serialized on the shared loop** — safe to create
  N runtimes on N threads, but their loop work interleaves through one
  loop, and per-loop ctx dispatch must be verified before claiming
  true concurrency.
- H1 ships the verification test: two runtimes on two threads in one C
  test binary, both streaming concurrently → no cross-talk, no shared
  globals corruption (hunts the known process-global HACK class). The
  outcome — concurrent or serialized — gets documented here either way.

**Status:** single-runtime-per-thread is the only verified mode today.
Multi-instance is architecturally tractable (loop already centralized;
per-thread state like the process module's `CTX` is already
`thread_local!`) but unverified — H1 item 5.

## 3. Error model

Every fallible call returns (or conveys, for streaming) a uniform shape:

```json
{ "ok": false, "error": { "code": "rlm_budget", "message": "step budget exhausted after 12 steps" } }
```

- `code` is a **stable string** from a reviewed registry (snake_case,
  `area_reason`). `message` is human-readable and **not** machine-parsed.
- No errno-style integers, no stderr-only errors. Anything a host can act
  on must arrive through the return JSON; stderr is reserved for the log
  callback path (§5, shipped with H2).
- Success shape: `{ "ok": true, ...result }`.
- Unknown funnel method → `{ "ok": false, "error": { "code": "unknown_method", ... } }`;
  malformed JSON args → `"bad_args"`. Neither may crash (H6 fuzz gate).

Initial code registry (grows additively, reviewed per release):

| Code | Meaning |
|---|---|
| `unknown_method` | funnel method not in this release's manifest |
| `bad_args` | args JSON missing/malformed/wrong shape |
| `not_configured` | feature needs config that was not supplied (e.g. API key) |
| `ai_provider_error` | provider rejected the request (status in `message`) |
| `rlm_budget` | RLM step/token/wall budget exhausted |
| `agent_budget` | agent run hit `budget.*` limit |
| `cancelled` | operation cancelled via `sofuu_rt_cancel` |
| `qtsq_unavailable` | build without QTSQ asked for a memory/kv op |
| `io_error` | filesystem/network failure below the feature layer |

## 4. Memory ownership

Mirrors the proven `ffi_exports.rs` contract (`crates/sofuu-core/src/ffi_exports.rs`):

- Every buffer the library hands out is allocated with `malloc`
  (`libc::malloc` in Rust) and NUL-terminated where it is a string.
- The host frees it with `sofuu_free()` — which is exactly `free()`,
  exported so hosts with their own CRT (Windows, some embedders) never
  cross allocators. Never the reverse: the host must not `free()`
  pointers it did not receive from a `sofuu_*` call, and must not pass
  host-owned buffers into the library expecting it to free them.
- All entry points tolerate NULL inputs by returning NULL/`-1`-style
  failures (the `cstr_or_null` / `malloc_cstr` discipline) — a NULL from
  the host is a graceful error, never a crash.
- Streaming callbacks (`on_event`) receive borrow-only strings valid for
  the duration of the callback; copy if you keep them.

## 5. Config model

Every ambient dependency becomes explicit init config. `sofuu_rt_new`
takes a JSON config (NULL = all defaults):

```json
{
  "embedded": true,
  "config_root": "/var/app/sofuu",
  "api_keys": { "anthropic": "sk-ant-...", "openai": "sk-..." },
  "brain_path": "/var/app/sofuu/brain.qtsq",
  "qtsq": true,
  "enable_signals": false,
  "agents": [ { "name": "researcher", "system": "…", "provider": "anthropic",
                "model": "claude-3-5-sonnet", "tools": […], "budget": {…} } ],
  "log": "<host callback, registered via API, not JSON>"
}
```

- `config_root` (default `$HOME/.sofuu`, host-overridable) replaces every
  `$HOME/.sofuu*` derivation: chat config, `mcp.json` discovery, RLM
  routing log, brain default path (H2 item 4, done).
- `api_keys` config values take **precedence** over env vars. Env vars
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `BRAVE_API_KEY`, …) are a
  *fallback*, never a requirement — `rt/ai.rs` resolves explicit opts →
  embed config → env (H2, done).
- `brain_path` overrides the memory brain file (default without it:
  `$HOME/.sofuu_brain.qtsq` via `src/js/agent.js`; H2, done).
- `qtsq: false` forces the QTSQ-free flavor even in a QTSQ-linked build
  (memory/kv report `qtsq_unavailable` instead of touching disk).
- `enable_signals` (default **off** when embedded): no `sigaction`/
  `uv_signal` installs unless the host opts in (H2 item 2, done).
- Log callback: `console.*` routes through a host-provided callback
  registered with `sofuu_embed_set_log_cb` when embedded; default in the
  CLI stays stdout/stderr (H2 item 3, done).
- JS seam: `sofuu_rt_new` injects `globalThis.__sofuu_embed_config`
  (`{ embedded, configRoot, brainPath }`) so bundled JS (`agent.js`) can
  derive its paths from the host config instead of `$HOME`.
- Env vars are a fallback everywhere, never a requirement. Missing
  `$HOME` on iOS/Android sandboxes must degrade to "feature off /
  `not_configured`", never a crash.

## 6. Stability tiers

- **JS API = stable (semver).** `sofuu.{ai,agent,rlm,web,memory,kv,mcp,fs,http}`
  plus the shipped drivers are the embedder's primary surface; breaking
  changes only on major versions. The reviewed diff is the
  `[js/sofuu.api.json]` manifest (scope cap per release).
- **C ABI v2 = `experimental` for one release**, then frozen with symbol
  versioning (Linux version script `libsofuu.map` from day one;
  `-fvisibility=hidden` + explicit export list; `nm`-diff CI guard fails
  on accidental new exports).
- `sofuu_embed_abi_version()` returns the ABI version as a `uint32_t`
  (major in the high 16 bits). Hosts must check it; the v2 surface only
  **extends**, never breaks, the v1 `sofuu.h` six functions.

### The v2 surface (H1 ships `include/sofuu_embed.h`)

```c
uint32_t       sofuu_embed_abi_version(void);
SofuuRuntime  *sofuu_rt_new(const char *config_json);   /* NULL-safe */
void           sofuu_rt_free(SofuuRuntime *);
/* generic funnel — features grow here without new symbols */
int            sofuu_rt_call(SofuuRuntime *, const char *method,
                             const char *args_json, char **out_json);
/* streaming/cancellable variant for ai.stream/rlm/agent */
int            sofuu_rt_call_stream(SofuuRuntime *, const char *method,
                             const char *args_json,
                             void (*on_event)(const char *event_json, void *),
                             void *opaque, uint64_t *out_cancel_id);
int            sofuu_rt_cancel(SofuuRuntime *, uint64_t cancel_id);
/* eval escape hatch (full JS power when the funnel isn't enough) */
int            sofuu_rt_eval(SofuuRuntime *, const char *source,
                             char **out_json);
/* host log sink for console.* when embedded (NULL clears) */
void           sofuu_embed_set_log_cb(
                             void (*cb)(const char *level, const char *msg,
                                        void *opaque),
                             void *opaque);
void           sofuu_free(void *);
```

Initial funnel `method` set (additive per release):
`ai.complete`, `ai.stream`, `ai.embed`,
`memory.open/remember/recall/count/export`,
`rlm.query`, `agent.run`, `agent.list`, `agent.define`, `agent.cancel`,
`http.serve.start/stop`, `mcp.connect/call`.

**Streaming** (A7): `sofuu_rt_call_stream` calls a method that accepts an
`onStep` callback (e.g. `agent.run`) and forwards each event to the host's
callback on the calling thread. For `agent.run`, events are JSON objects:
`{"runId":"…","name":"…","depth":0,"t":<ms>,"kind":"start|plan|tool|delegate|answer_delta|answer|stop","payload":{…}}`.
The host receives a `cancel_id` usable with `sofuu_rt_cancel`.

**Per-process agent definitions** (A7.3): supply agent definitions in
`sofuu_rt_new` config as `"agents":[{"name":"…","system":"…",…}]`. No
filesystem needed — `~/.sofuu/agents/*.js` is the convenience layer, never
the only way.

The eval escape hatch (`sofuu_rt_eval`) means the funnel is a convenience,
not a cage: anything the JS API can do, a host can do today, and the
funnel grows to absorb the common calls.

## 7. Build artifacts & layering

- New crate `crates/sofuu-capi` (H1): `crate-type = ["staticlib", "cdylib"]`,
  thin wrapper over `sofuu-core` + `sofuu-ffi`; owns the exported symbol
  list. Dependency rule: **sofuu-capi may depend on sofuu-core and
  sofuu-ffi; nothing may depend on sofuu-capi.** (Repo lesson: never
  create a dev-dependency cycle between these crates — fat LTO fails with
  "symbol multiply defined"; see the `crates/sofuu-ffi/Cargo.toml` note.)
- `make libsofuu` → `target/release/libsofuu.{a,so/dylib}` +
  `include/sofuu_embed.h` copied to `dist/`. Size cap ≤ 5MB, same as the
  CLI (`make size-check`, `SIZE_LIMIT = 5242880`).
- Release profile is size-tuned (`opt-level = "z"`, fat LTO, strip,
  `panic = abort`) — `panic = abort` makes §1 a hard contract: there is no
  unwind to catch a Rust panic at the boundary, so capi entry points must
  validate inputs and never panic.

## 8. Licensing note for embedders

The Sofuu runtime is **MIT** (`LICENSE`). Builds with the QTSQ codec
linked additionally fall under `LICENSES/QTSQ-FORMAT.txt` (Quantum Tensor
Sequence Proprietary License v1.0, © Haruhito) for the QTSQ file-format
components; **QTSQ-free builds are pure MIT**. Per-platform packs (H4)
ship both flavors and state which artifact is which in `dist/README.md`;
the iOS/Android samples use the QTSQ-off flavor.

## 9. Platform notes

- **iOS:** pre-cleared by architecture — QuickJS interprets, never JITs,
  so no executable-memory allocation; App-Store-safe. QTSQ frameworks
  need a real-device link test before a QTSQ-on flavor ships there
  (degraded-QTSQ build is the fallback).
- **Android:** NDK per-ABI `.so` + JNI sample, minSdk 24.
- **Windows / WASM:** honest **v2** items — libcurl/libuv statics need a
  Windows cross story; WASM is a documented subset (engine + TS +
  bundler + memory-without-QTSQ; networking via host imports later).

## 10. Contract checklist (what each phase must prove)

| # | Rule | Proven by |
|---|---|---|
| 1 | No `exit`/`abort` on reachable paths | H2 audit test + H6 CI grep + funnel fuzz |
| 2 | One runtime ↔ one thread; multi-instance documented | H1 two-runtimes-two-threads test |
| 3 | Uniform `{ok, error:{code, message}}` errors | funnel tests + fuzz |
| 4 | malloc-owned buffers, `sofuu_free` | capi tests under ASAN/LSAN (H6) |
| 5 | Explicit config; env = fallback only | H2 `$HOME`/`getenv` grep audit |
| 6 | Stability tiers + version query | `sofuu_embed_abi_version` + `nm`-diff CI |

---

## 11. Getting embedded — per-platform quick start

### macOS / Linux (C)

```bash
make libsofuu          # → dist/libsofuu.{a,dylib} + dist/sofuu_embed.h
make headless-test     # compiles + runs examples/headless/c_embed.c
```

```c
#include "sofuu_embed.h"
SofuuRuntime *rt = sofuu_rt_new(NULL);
char *out = NULL;
sofuu_rt_eval(rt, "1 + 1", &out);   // → {"ok":true,"result":2}
sofuu_free(out);
sofuu_rt_free(rt);
```

See `examples/headless/c_embed.c` (minimal eval + funnel) and
`examples/headless/c_rlm.c` (RLM long-context Q&A) for full working examples.

### iOS (Swift)

QuickJS is a pure interpreter (no JIT) → no executable-memory allocation →
App-Store-safe by design. This is the key architectural advantage.

```bash
make dist-ios    # → dist/libsofuu.xcframework (device + simulator)
```

Add the xcframework to your Xcode project (General → Frameworks), import the
module, and use `SofuuBridge.swift`:

```swift
import Sofuu

let sofuu = SofuuBridge()
let result = sofuu.eval("1 + 1")              // → {"ok":true,"result":2}
let caps = sofuu.checkCaps()                    // → {"has_ai":true,...}
let vec = sofuu.embedLocal("hello")            // → [Float] (768-dim)
let sim = sofuu.similarity(vec, vec)           // → 1.0
```

See `examples/headless/SwiftSample/SofuuBridge.swift` for the full wrapper.

### Android (Kotlin/JNI)

```bash
make dist-android    # → dist/libsofuu-android-{aarch64,x86_64}.so
```

Copy the `.so` files into `app/src/main/jniLibs/<abi>/`, add the JNI bridge
(`jni_bridge.c`), and use `SofuuBridge.kt`:

```kotlin
val sofuu = SofuuBridge()
val result = sofuu.eval("1 + 1")              // → {"ok":true,"result":2}
val caps = sofuu.checkCaps()                    // → {"has_ai":true,...}
val vec = sofuu.embedLocal("hello")            // → FloatArray (768-dim)
```

See `examples/headless/KotlinSample/` for the full wrapper + JNI bridge +
CMakeLists.txt. minSdk 24.

### Windows / WASM

**v2** — not yet implemented. Windows needs a libcurl/libuv cross story;
WASM is a documented subset (engine + TS + bundler + memory-without-QTSQ;
networking via host imports later).

### Platform pack scripts

| Command | Output |
|---|---|
| `make dist-macos` | `dist/libsofuu-darwin-{arm64,x86_64}.{a,dylib}` |
| `make dist-linux` | `dist/libsofuu-linux-{x86_64,arm64}.{a,so}` (static musl) |
| `make dist-ios` | `dist/libsofuu.xcframework` (device + simulator) |
| `make dist-android` | `dist/libsofuu-android-{aarch64,x86_64}.so` |
| `make dist-all` | Every platform available on the current host |

### CI gates (H6)

Every CI push/PR and release build runs:

1. **`make headless-test`** — compiles `c_embed.c` + `c_rlm.c` against
   `libsofuu` and runs them. Proves the library works standalone (no TUI,
   no `~/.sofuu` touched unless configured).
2. **`make abi-check`** — `nm` symbol diff against
   `scripts/abi_symbols.txt` (23 public API symbols). Missing = breaking
   change (fails CI). New = additive (warns; update the baseline).
3. **`crates/fuzz/fuzz_targets/capi_call.rs`** — feeds arbitrary method/JSON
   to `sofuu_rt_call`; a crash or exit is a failed fuzz run.

### Sizes

| Artifact | Size |
|---|---|
| `libsofuu.a` (macOS arm64, QTSQ-free) | ~1.7 MB |
| `libsofuu.dylib` (macOS arm64, QTSQ-free) | ~1.7 MB |
| `sofuu` CLI (macOS arm64) | ~2.0 MB |

All artifacts enforce a 5MB size cap (`make size-check`).
