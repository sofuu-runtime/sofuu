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
    `rlm/js_api.rs` routing log (`~/.sofuu/routing_log.jsonl`).
    The brain is deliberately NOT one of them — it is project-local
    (`<cwd>/.sofuu/brain/brain.qtsq`), overridable via `brain_path`.
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

Code registry as **actually emitted today**. The funnel is deliberately small
and each code is a real, tested path — an aspirational list of codes that no
code path returns is worse than a short honest one, so this table is what a
host can branch on, not what someone hopes to add:

| Code | Emitted when | Test |
|---|---|---|
| `invalid_arg` | a required pointer argument is NULL | `call_null_runtime_returns_error` |
| `bad_args` | `args_json` is not valid JSON | `call_bad_args_json_returns_error` |
| `unknown_method` | the funnel method does not exist in this build | `call_unknown_method_returns_error_envelope` |
| `js_exception` | the method exists and threw (provider error, budget, unconfigured key, dimension mismatch — all surface here) | `a_throwing_method_is_not_reported_as_unknown_method` |
| `internal` | the engine could not produce a result (null context, unparseable envelope) | — |

The C-level vector ABI additionally returns typed `int` codes for
`sizes/guards` that cannot be expressed in JSON: `SOFUU_ERR_UNKNOWN_SPACE`
(`-7`), `SOFUU_ERR_MODEL_UNAVAIL` (`-8`), `SOFUU_ERR_NOMEM` (`-9`),
`SOFUU_ERR_INVALID_ARG` (`-1`).

`unknown_method` is the one that matters for feature detection: it separates
"this build does not have it" from "it exists and failed", so a host can
branch on capabilities without string-matching messages.

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
  "enable_signals": false,

  "provider": "openai",
  "model": "gpt-4o-mini",
  "base_url": "http://127.0.0.1:11434/v1",

  "agents": [ { "name": "researcher", "system": "…", "provider": "anthropic",
                "model": "claude-3-5-sonnet", "tools": […], "budget": {…} } ]
}
```

A host log callback is installed through the API
(`sofuu_embed_set_log_cb`), not through JSON.

**`provider` / `model` / `base_url` (E2).** Set the request defaults once,
here, instead of repeating them on every call:

```c
/* pin the whole runtime to a local OpenAI-compatible server, once */
sofuu_rt_new("{\"provider\":\"openai\",\"model\":\"llama3.2\","
             "\"base_url\":\"http://127.0.0.1:11434/v1\"}");

/* …and then no call ever names a model again: */
sofuu_rt_call(rt, "ai.complete", "{\"messages\":[…]}", &out);
```

These are consulted **only when a call omits them**, so an explicit per-call
value always wins. They are also read exclusively from the creating
runtime's settings (never process globals), so one runtime's defaults cannot
retarget another's. The CLI is unaffected: chat has no configured default, so
its "no model configured" guidance still fires — a user picks a model in
`/model`, which is the interactive path an embedded host does not have.

> **QTSQ is a build-time property, not a config key.** Whether the encrypted
> brain codec is present is decided when `libsofuu` is compiled, so there is
> no `"qtsq": true` toggle to set. Check what you linked with
> `sofuu doctor`, or read §12 on the QTSQ-free flavor. (An earlier draft of
> this doc advertised a `qtsq` key; it was never parsed and has been removed
> rather than left as a lie.)

- `config_root` (default `$HOME/.sofuu`, host-overridable) replaces every
  `$HOME/.sofuu*` derivation: chat config, `mcp.json` discovery, RLM
  routing log, brain default path (H2 item 4, done).
- `api_keys` config values take **precedence** over env vars. Env vars
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `BRAVE_API_KEY`, …) are a
  *fallback*, never a requirement — `rt/ai.rs` resolves explicit opts →
  embed config → env (H2, done).
- `brain_path` overrides the memory brain file. Default without it:
  **project-local** `<cwd>/.sofuu/brain/brain.qtsq` (`src/js/agent.js` /
  `src/js/chat.js` agree — the old `$HOME/.sofuu_brain.qtsq` default was
  retired 2026-09-09 because it split reads and writes across two files).
  `sofuu doctor` prints the resolved path and round-trips a write.
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

Funnel `method` set — **this list is enforced by a test**
(`documented_funnel_methods_all_resolve`), which resolves every entry below
through a real runtime, so it cannot drift from the implementation again:

| namespace | methods |
|---|---|
| **ai** | `ai.complete`, `ai.stream`, `ai.embed`, `ai.embedLocal`, `ai.embedBatch`, `ai.embedImage`, `ai.embedInfo`, `ai.similarity`, `ai.transcribe`, `ai.speak` |
| **memory** | `memory.open`, `memory.remember`, `memory.recall`, `memory.count`, `memory.flush`, `memory.forget`, `memory.close` |
| **mcp** | `mcp.connect`, `mcp.call`, `mcp.listTools`, `mcp.listResources`, `mcp.disconnect` |
| **agent** | `agent.define`, `agent.run`, `agent.list`, `agent.cancel`, `agent.renderTrace` |
| **rlm** | `rlm.query`, `rlm.route` |
| **http** | `http.serve` |

**Handle-addressed namespaces.** `memory.open`, `mcp.connect` and
`http.serve` return *handles* (small JSON objects) rather than the raw JS
instances, because those instances do not survive JSON serialization. The
subsequent calls take `{"handle":N, …}`; when a host keeps only one brain
or one MCP client, `handle` may be omitted and the most recent one is used.

```c
/* open a brain, remember, recall */
sofuu_rt_call(rt, "memory.open",  "{\"path\":\"/tmp/b.qtsq\",\"dim\":768}", &out);
sofuu_rt_call(rt, "memory.remember",
              "{\"vec\":[…],\"text\":\"the sky is blue\",\"role\":\"user\"}", &out);
sofuu_rt_call(rt, "memory.recall", "{\"vec\":[…],\"k\":5}", &out);
```

**Streaming**: `sofuu_rt_call_stream` supports two shapes with one driver.
Methods that take an `onStep` option (`agent.run`) forward each step event;
methods that return an async iterable (`ai.stream`) are drained, each chunk
delivered as `{"kind":"delta",…}`. Either way the stream ends with exactly
one terminal `{"kind":"done",…}` event — carrying `deltas`/`aborted` on
success, or `error` on failure — so a host that waits for `done` is never
left hanging. The signature has no `out_json`, so that terminal event is
also where the final value arrives. `*out_cancel_id` is written *before* the
stream starts, so a callback can cancel in flight; `sofuu_rt_cancel` stops
both shapes.

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
ship both flavors and state which artifact is which in `docs/EMBEDDING-DIST.md`;
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

> **Prefer a package manager?** SwiftPM (`Package.swift`), CocoaPods
> (`Sofuu.podspec`) and Gradle (`com.sofuu:sofuu-android`) are the supported
> install paths, and the `sofuu` CLI is on npm. The manual instructions below
> are for building from source. Artifact inventory + licensing:
> [`EMBEDDING-DIST.md`](./EMBEDDING-DIST.md).

### iOS / macOS (Swift) — the fastest path

QuickJS is a pure interpreter (no JIT) → no executable-memory allocation →
App-Store-safe by design. This is the key architectural advantage.

**Swift Package Manager**

```swift
dependencies: [
    .package(url: "https://github.com/sofuu-runtime/sofuu.git", from: "0.2.0")
],
targets: [ .target(name: "MyApp", dependencies: [
    .product(name: "Sofuu", package: "sofuu")
]) ]
```

**CocoaPods**

```ruby
pod 'Sofuu', '~> 0.2'
```

Either way you get `libsofuu.xcframework` from the release (checksum-pinned
by `Package.swift`) and a typed Swift layer over it:

```swift
import Sofuu

let sofuu = try Sofuu()

// Offline embeddings — no key, no network, no model file.
let vec = try sofuu.embed("the sky is blue")        // 768-dim unit vector

// The brain: encrypted, local, owned by your user.
let brain = try Brain.openDefault(sofuu: sofuu)
try brain.remember("the sky is blue")
for hit in try brain.recall("what colour is the sky?", k: 3) {
    print(hit.text, hit.score)
}

// Streaming, with a handle you can cancel.
let stream = sofuu.stream("ai.stream", ["prompt": "Say hi"]) { event in
    switch event {
    case .delta(let text): print(text, terminator: "")
    case .done: print()
    case .failure(let e): print("error: \(e)")
    case .step: break
    }
}
stream.cancel()
```

The full typed surface is `bindings/swift/Sources/SofuuBridge/SofuuBridge.swift`.

**From source** (no package manager):

```bash
make dist-ios    # → dist/libsofuu.xcframework (device + simulator)
```

Then add the xcframework to your Xcode project (General → Frameworks), or
call the C ABI directly — see `examples/headless/SwiftSample/SofuuBridge.swift`.

### macOS / Linux (C)

```bash
make libsofuu          # → dist/libsofuu.{a,dylib} + dist/sofuu_embed.h
make headless-test     # compiles + RUNS five asserting samples
```

```c
#include "sofuu_embed.h"
SofuuRuntime *rt = sofuu_rt_new(NULL);
char *out = NULL;
sofuu_rt_eval(rt, "1 + 1", &out);   // → {"ok":true,"result":2}
sofuu_free(out);
sofuu_rt_free(rt);
```

`examples/headless/c_embed.c` is the asserting version: eval → embed → open a
brain → remember → recall → flush. `c_llm.c` is the real-LLM path (mock in
CI, live with `SOFUU_DEMO_KEY`).

### Android (Kotlin/Gradle) — the fastest path

**Gradle**

```groovy
dependencies { implementation 'com.sofuu:sofuu-android:0.2.0' }
```

The AAR bundles the per-ABI `libsofuu.so` and the JNI bridge — no manual
`.so` copying. Full setup and the manual fallback: `bindings/android/README.md`.

<details>
<summary>Android from source (Kotlin/JNI, manual)</summary>

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

</details>

### The CLI on npm

```bash
npx sofuu            # chat
npx sofuu run app.ts
```

`npm/` is the free CLI surface (downloads a checksum-verified prebuilt
binary). To embed the runtime in your own app, use one of the SDK paths
above.

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

> **Published host platform: macOS only, for now.** CI builds and tests
> macOS only and the release attaches no Linux or Windows artifact. Every
> target in that table still works from source on a capable host — they are
> simply not built or published for now. The SDK packs (iOS xcframework,
> Android AAR) are unaffected: those target devices, not desktop hosts.

### CI gates (H6)

Every CI push/PR and release build runs:

1. **`make headless-test`** — compiles `c_embed.c` + `c_rlm.c` against
   `libsofuu` and runs them. Proves the library works standalone (no TUI,
   no `~/.sofuu` touched unless configured).
2. **`make abi-check`** — `nm` symbol diff against
   `scripts/abi_symbols.txt` (30 exported symbols, of which 15 are the
   documented `sofuu_embed.h` surface). Missing = breaking
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

---

## 12. Vector embedding ABI + image + voice (H-E1/M1/M2 shipped)
> Implementation plan: `PLAN-MULTIMODAL-EMBEDDINGS.md` at the repo root.
> Phases H-E1 (text vector ABI + JS local-default), M1 (image embeddings)
> and M2 (voice) SHIPPED 2026-09-23. Hosts must still probe availability at
> runtime (`sofuu_embed_abi_version`, space manifests) rather than assume it.

**Vector ABI (H-E1 SHIPPED).** The JSON funnel (`sofuu_rt_call("ai.embed")`)
stays, but vector workloads get direct float entry points so hosts never
JSON-encode embeddings (`*out` is malloc-owned; host frees with
`sofuu_free()`; `rt` may be NULL — embedders are stateless):

```c
int sofuu_embed_local(SofuuRuntime *rt, const char *text, const char *space,
                      float **out, size_t *out_dim);
int sofuu_embed_batch(SofuuRuntime *rt, const char **texts, size_t n,
                      const char *space,
                      float **out, size_t *out_n, size_t *out_dim);
int sofuu_embed_image(SofuuRuntime *rt, const uint8_t *bytes, size_t len,
                      float **out, size_t *out_dim);
int sofuu_embed_info(SofuuRuntime *rt, char **out_json);
```

Spaces: `hash-768` (**the default** — and the same space the brain stores
in, so embed → remember needs no space argument), `sem1-64`, `sem2-64`, plus
`img1-64` (M1 — joint with sem2-64 text geometry by versioned design, so text
queries retrieve images). Passing `NULL`/`""` selects `hash-768`; pass
`"hash-v1"` too and it is accepted as an alias. Space rule (extends the
brain-file rule): one space per index, except img1-64 which is explicitly
compatible with sem2-64; a cross-space write refuses with a typed error
rather than corrupting recall. Every vector is tagged `{model_id, dim}` via
the space manifest, and `sofuu_embed_info()` reports the live default.

**Image (M1 SHIPPED).** IMG1 tiny distilled projector (144 IMGF1 features →
32 tanh → 64, ~7.4KB artifact, round-1): PNG/JPEG decode → frozen features →
joint-space vector. Eval: text→image R@5 1.000 on held-out tags (+0.75 over
raw features, +0.46 over lexical, zero text degradation). JS:
`sofuu.ai.embedImage(u8bytes)` (sync) + `sofuu.fs.readFileBytes` for files;
memory recipe: `open(path, 64, 'image-projector-v1')` → `remember(vec,
caption, …)` → `recall(sem2TextVec, k)`. Proven by `tests/multimodal_test.js`
(9/9). Agent-loop multimodal fusion is fast-follow.

**Voice (Phase M2, SHIPPED).** Provider audio (`ai.transcribe` /
`ai.speak`, BYO key, OpenAI-compatible `/audio/transcriptions` +
`/audio/speech`, multipart + raw-bytes paths, own curl-multi request tag
with full lifecycle) plus OS speech bridges (`SofuuVoice.swift`:
SFSpeechRecognizer file + live-mic + AVSpeechSynthesizer;
`SofuuVoice.kt`: SpeechRecognizer + TextToSpeech) plus thin C ABI
(`sofuu_voice_transcribe` / `sofuu_voice_speak` funnel routing, hosts
base64 themselves) + Swift/Kotlin provider fallbacks. Full-duplex
`sofuu_voice_turn` orchestration stays a sample-level recipe (OS mic
capture is not portable C). No bundled STT model — the 5MB cap stands.

**JS parity.** `ai.embed` local-by-default (network opt-in);
`ai.embedBatch(texts[])` for ingestion; `ai.transcribe`/`ai.speak` for
prototyping. (There is no `sofuu.voice` namespace — an earlier draft of this
line named one; it never existed.)
CLI embedding surfaces (`sofuu embed`, semantic file search) are fast-follow
and do not gate the ABI.

## 13. Measured quality (Q phase, SHIPPED 2026-09-25)

The in-house suites (`embed-eval`, `img-eval`) are training gates on our
own synthetic corpora. They can only say "we beat our own baselines", so
`ml-train bench` measures the public sets instead — reproduce with
`scripts/bench/fetch_datasets.sh` then `ml-train bench`.

**Retrieval** — BEIR SciFact, 5,183 abstracts, 300 judged queries, official
qrels. **Similarity** — STS test split, 1,379 rated pairs. Apple M2 Pro,
release build, single thread, no network:

| space | dim | params | artifact | embed | R@1 | R@5 | nDCG@10 | ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 *(measured here)* | — | — | 0 | — | 0.537 | 0.750 | **0.663** | — |
| **hash-v1** *(shipped default)* | 768 | **0** | **0 B** | **~0.002 ms** | 0.307 | 0.493 | **0.410** | **0.613** |
| sem1-64 *(opt-in)* | 64 | 13,392 | 13.7 KB | ~0.05 ms | 0.003 | 0.017 | 0.014 | 0.421 |
| sem2-64 *(opt-in)* | 64 | 30,032 | 29.9 KB | ~0.08 ms | 0.037 | 0.090 | 0.081 | 0.517 |

How to read it:

- **The BM25 row is the calibration check.** Our implementation lands at
  0.663 nDCG@10 against the ~0.665 published for SciFact, so the harness,
  the qrels and the metrics agree with the literature. Without that row
  the numbers below mean nothing.
- **Latency is a median over 200 samples on a shared machine** — read it as
  a magnitude, not a constant; it moves run to run.
- **hash-v1 is the default because it is the one that works.** 62% of
  BM25's nDCG with zero parameters, zero bytes and no model file — pure
  arithmetic over trigram hashes at ~2 µs a document. It is what makes the
  brain work with no download and no API key.
- **The 64-dim learned spaces are not general-domain retrievers.** Out of
  domain they reach 12% (sem2) and 2% (sem1) of BM25's nDCG. The repo's own
  pre-registered `embed-eval` gate independently returns FAIL on two gates
  and says "do NOT replace the current embedder", so they must stay opt-in.
  The table ships with the gap rather than quietly omitting the rows.
- **No hosted-model baseline was run** (it costs money per run). Expect a
  1536-dim API embedder to lead every row; that is the trade — quality for
  a network round-trip, a key and cents per million documents.
- **IMG1's R@5 1.000 is synthetic and in-domain** (93 held-out procedural
  scenes, 39 unseen tag combinations). It shows the projector learned its
  intended mapping; it is not a real-image retrieval number, and no honest
  one is claimed.

The in-house numbers (fused 0.917 vs hash 0.875 on our own 8-category
corpus) are in-domain by construction and are not comparable to the table
above.
