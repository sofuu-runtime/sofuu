# PLAN — Full Headless Mode: Sofuu as an Embeddable AI Runtime

> **Goal:** any device or app — iOS, Android, desktop, server, edge — embeds
> Sofuu with **all features and zero TUI**. The TUI is one consumer of the
> runtime, never a dependency. RLM, embeddings, brain/memory, HTTP, MCP, the
> agent loop — all reachable from a host process through a small stable ABI.
>
> *Written 2026-08-12. Supersedes/expands `PLAN-RLM.md` Part 3 (P3 items
> execute here). Complements `PLAN-RUST-MIGRATION.md` (bindings) and
> `PLAN-CHAT-FEATURES.md` (chat-only features stay chat-only, by design).*
>
> **Update 2026-09-27 — the SDK is now installable, and honest.**
> H0–H6 built the library; what was missing was the *install path* and, more
> seriously, truthfulness. An audit found **7 of the 17 documented funnel
> methods did not exist** (the entire brain was unreachable from C), the
> flagship sample called a method that does not exist and printed the error
> as success, `ai.stream` was advertised but delivered zero events, and the
> default embedding space disagreed with the brain's (a silent dimension
> mismatch). All fixed, each pinned by a test:
>
> - **Memory from C** — a handle facade makes `memory.open/remember/recall/
>   count/flush` real; `c_embed.c` does a full round-trip in 30 lines.
> - **Streaming** — `ai.stream` is pumped to delta events, always ends with
>   exactly one terminal `done` event (success *or* failure), and supports a
>   live mid-flight cancel.
> - **Samples assert** — all five headless samples fail the build on a broken
>   call. A new `c_llm.c` covers the real-LLM path (mock in CI).
> - **Default space is `hash-768`** — matching the brain and the measured
>   quality table, so embed → remember just works.
> - **`unknown_method` is a real code**, distinct from `js_exception`, so a
>   host can feature-detect.
> - **Anti-drift test** — `documented_funnel_methods_all_resolve` parses the
>   method table out of `docs/EMBEDDING.md` and fails if any listed method
>   does not resolve. Doc drift is now structurally impossible.
> - **Install paths** — `Package.swift` + `Sofuu.podspec` (iOS/macOS),
>   a Gradle AAR (`bindings/android`, `com.sofuu:sofuu-android`), and an npm
>   package for the CLI. Release CI builds and attaches all four SDK packs
>   and commits the SwiftPM checksum back to the branch.
> - **Config defaults for embedders (E2)** — `provider`/`model`/`base_url`
>   can now be set once in `sofuu_rt_new` config instead of on every call.
>   An embedded host has no interactive `/model` picker, so repeating them
>   per call was pure friction. They apply only when a call omits them (a
>   per-call value still wins) and are read exclusively from the creating
>   runtime's settings, so one runtime's defaults cannot retarget another.
>   The CLI is untouched: chat sets no default, so its "no model configured"
>   guidance still fires.
> - **Typed Swift layer** — `bindings/swift/.../SofuuBridge.swift` gives
>   `embed`, a real `Brain` type, streaming, and cancel; typechecked in CI
>   against the real C header.
>
> Verified: capi 36/0 · JS 46/0/1 · headless 5/5 samples · abi-check green ·
> size 3.2 MB/5 MB. Release prepared in `RELEASE-0.2.0-PREP.md`; the
> tag-and-publish step is deliberately the owner's.
>
> **Status (2026-08-20):**
> - **H0 ✅** — `docs/EMBEDDING.md` written, all 6 contract points grounded in source.
> - **H1 ✅** — `crates/sofuu-capi` complete: `sofuu_rt_new/free/call/eval` + streaming stubs,
>   `include/sofuu_embed.h`, `make libsofuu` → 1.7MB dylib, 13/13 tests pass.
> - **H2 ✅** — hostile-host audit fixes landed: `process.exit` throws a catchable
>   `ExitError` (exitCode prop) when embedded; SIGINT/SIGTERM/SIGWINCH installs gated
>   behind `enable_signals`; console.log/info/warn/error/table route through a host
>   log callback (`sofuu_embed_set_log_cb`); `config_root` replaces `$HOME/.sofuu`
>   derivations (chat history, RLM state, agent brain/logs); `api_keys`/`brain_path`
>   config keys + `globalThis.__sofuu_embed_config` JS seam. 16/16 capi tests,
>   137+15 core tests pass.
> - **H3 ✅** — Folded into PLAN-AGENTS A1 (agent.js shipped to all contexts). Desktop dogfood landed (2026-08-20): `sofuu-desktop/sofuu-bridge.js` rewritten — deleted `__ttyRaw`/`__ttyNormal`/`__readline`, switched from `sofuu.ai.stream` to `sofuu.agent.run` (the A1 loop), reads stdin without raw mode. Zero TTY calls in the bridge.
> - **H4 ✅** — Platform packs landed: `scripts/dist/{macos,linux,ios,android,all}.sh` +
>   `docs/EMBEDDING-DIST.md` with artifact table. macOS (arm64+x86_64), Linux (x86_64+arm64 musl),
>   iOS (xcframework device+sim), Android (NDK per-ABI .so). Win/WASM remain v2.
> - **H5 ✅** — Samples & docs landed: `examples/headless/c_embed.c` (eval + funnel)
>   + `examples/headless/c_rlm.c` (RLM long-context Q&A) both compile & run in CI.
>   `examples/headless/SwiftSample/SofuuBridge.swift` (iOS xcframework wrapper).
>   `examples/headless/KotlinSample/` (Android JNI: `SofuuBridge.kt` + `jni_bridge.c` +
>   `CMakeLists.txt`). `docs/EMBEDDING.md` updated with per-platform getting-started
>   pages. README "Embed Sofuu" section added. `make headless-test` compiles + runs both
>   C samples.
> - **H6 ✅** — CI gates landed: `make headless-test` (compile+run c_embed.c + c_rlm.c), `make abi-check`
>   (nm symbol diff vs `scripts/abi_symbols.txt` baseline), `crates/fuzz/fuzz_targets/capi_call.rs`
>   (funnel fuzz target). CI workflow (`ci.yml`) + release workflow (`release.yml`) run all three
>   gates on every push/PR and ship libsofuu tarballs in releases.
>
> **Audit of today's blockers (verified 2026-08-12 — this is why "embedable"
> is half-true at best right now):**
> - Public C API is six functions (`src/sofuu.h`): init/eval/run-jobs/
>   destroy. No config, no error model, no direct feature calls, no streaming
>   or cancellation story.
> - **`exit()` inside runtime code** — fatal in a hosted library:
>   `repl/repl.c:93` (`exit(0)`), `modules/mod_process.c:263/274` (signal
>   re-exit 130/143), `mod_process.c:1636` (`process.exit` → raw `exit()`).
>   Embedded, any of these kills the host app.
> - **Signal handler installs** in `mod_process.c` — a library must not
>   hijack its host's signals.
> - **`$HOME`/getenv assumptions** in `mod_process.c`, `mod_ai.c`
>   (`~/.sofuu_brain.qtsq`, config under `$HOME`) — apps have no meaningful
>   `$HOME` contract; iOS sandboxes especially.
> - **stdout/stderr writes** baked into `mod_console.c` — a host app needs a
>   log callback, not someone printing into its TTY.
> - **The existing "embedding" proves the pain**: `sofuu-desktop`'s
>   `sofuu-bridge.js` runs the runtime as a *subprocess* and even then must
>   pull in TTY plumbing (`__ttyRaw()`, `__readline`). Today's integration
>   path drags the terminal in with it.
> - **Good news:** `uv_default_loop` is used in exactly one place
>   (`io/loop.h`) — the event loop is already centralized, so the
>   multi-instance story is tractable. QuickJS itself supports multiple
>   independent runtimes/contexts per process.
> - **iOS is pre-cleared by architecture:** QuickJS is a pure interpreter
>   (no JIT), so App Store executable-memory policy is satisfied by design.

---

## H0 — The embedding contract (write it down first — 1 day) ✅ DONE

Everything downstream implements this document (`docs/EMBEDDING.md` skeleton
first, code second):

1. **Process rule:** the library never calls `exit`/`abort` on reachable
   paths — errors return. Enforced by CI grep on `src/` + `crates/`
   (allowlist: CLI-only files).
2. **Threading model:** one `SofuuRuntime` is owned by one thread; all calls
   on a runtime must come from that thread (QuickJS + libuv are
   single-threaded). Multiple runtimes per process: **supported**, each on
   its own thread; serialized if they share the centralized loop (H1
   verifies; documented either way).
3. **Error model:** every fallible call returns/conveys
   `{ok:false, error:{code:"rlm_budget", message:…}}` — stable string codes,
   human messages. No errno-ish integers, no stderr-only errors.
4. **Memory ownership:** buffers the library hands out are malloc-owned;
   host frees with `sofuu_free()`. Never the reverse. Mirrors the proven
   `ffi_exports.rs` contract.
5. **Config model:** every ambient dependency becomes explicit init config —
   config root dir (defaults `$HOME/.sofuu`, host-overridable), API keys
   (config values take precedence over env vars), brain path, QTSQ on/off,
   log callback. Env vars are a *fallback*, never a requirement.
6. **Stability tiers:** JS API = stable (semver). C ABI v2 = `experimental`
   for one release, then frozen with symbol versioning.
   `sofuu_embed_version()` returns the ABI version.

## H1 — `libsofuu`: build, ABI, and the JSON funnel (~3–4 days) ✅ DONE

1. **New crate `crates/sofuu-capi`:** `crate-type = ["staticlib", "cdylib"]`,
   thin wrapper over `sofuu-core` + `sofuu-ffi`. The core crates stay
   binary-first; the capi crate owns the exported symbol list.
2. **Header `include/sofuu_embed.h`** (v2 ABI — extends, never breaks,
   `sofuu.h`):
   ```c
   uint32_t    sofuu_embed_abi_version(void);
   SofuuRuntime *sofuu_rt_new(const char *config_json);   /* NULL-safe */
   void         sofuu_rt_free(SofuuRuntime *);
   /* generic funnel — features grow here without new symbols */
   int          sofuu_rt_call(SofuuRuntime *, const char *method,
                              const char *args_json, char **out_json);
   /* streaming/cancellable variant for ai.stream/rlm */
   int          sofuu_rt_call_stream(SofuuRuntime *, const char *method,
                              const char *args_json,
                              void (*on_event)(const char *event_json, void *),
                              void *opaque, uint64_t *out_cancel_id);
   int          sofuu_rt_cancel(SofuuRuntime *, uint64_t cancel_id);
   /* eval escape hatch (full JS power when the funnel isn't enough) */
   int          sofuu_rt_eval(SofuuRuntime *, const char *source,
                              char **out_json);
   void         sofuu_free(void *);
   ```
   Initial `method` set: `ai.complete`, `ai.stream`, `ai.embed`,
   `memory.open/remember/recall/count/export`, `rlm.query` (as PLAN-RLM R2
   lands), `agent.run`, `http.serve.start/stop`, `mcp.connect/call`.
3. **Symbol hygiene:** `-fvisibility=hidden` + explicit export list; Linux
   gets a version script (`libsofuu.map`); `nm` diff script in CI fails on
   accidental new exports.
4. **`make libsofuu`** → `target/release/libsofuu.{a,so/dylib}` +
   `include/sofuu_embed.h` copied to `dist/`. Size-check ≤5MB, same as the
   CLI.
5. **Multi-instance verification test:** two runtimes on two threads in one
   C test binary, both streaming concurrently → no cross-talk, no shared
   globals corruption (hunts the known `mcp.c` global-state HACK class).

## H2 — Hostile-host audit fixes (~2–3 days) ✅ DONE (2026-08-19)

The "the library must be a polite guest" pass:
1. **exit() → error returns:** `mod_process.c:1636` (`process.exit`) becomes:
   when embedded (`sofuu_rt_new` config flag `embedded:true`), `process.exit`
   throws a catchable JS `ExitError` instead of terminating; the 130/143
   re-exits and `repl.c` exit are CLI-gated (already unreachable headless —
   assert it in the audit test).
2. **Signals:** guard all `uv_signal_init`/signal installs behind
   `config.enable_signals` (default off when embedded).
3. **Logging:** `mod_console.c` stdout/stderr writes route through a
   host-provided callback when embedded (default = current behavior in CLI).
   TUI/tui.c must never init without a TTY — assert via the desktop-bridge
   test (which today must call `__ttyRaw()` — the bridge gets rewritten in
   H3 to prove it isn't needed).
4. **Config root:** `config.config_root` replaces every `$HOME/.sofuu*`
   derivation (`mod_process.c`, `mod_ai.c`, chat history, brain default
   path). Audit grep `getenv(`/`HOME` in `src/` → each hit either reads
   config or is documented as CLI-only.
5. **cwd assumptions:** all relative path resolution documented; API accepts
   absolute paths everywhere.

## H3 — Down-level the composed behaviors (~2 days; makes "ALL features" true) ✅ DONE (2026-08-20)

> *If `PLAN-AGENTS.md` ships first, its A1 subsumes this phase (one loop, one
> owner) — don't build H3 separately in that case, mark it folded-in.*

Today the agent loop, recall injection, and MCP auto-connect live in the
chat driver's embedded JS (`chat.rs:1088` onward: agent loop :1273+, recall
:1227–1240, MCP merge :1104+). An embedder gets primitives only. Fix:
1. **Ship them as a runtime JS module** — `src/js/agent.js` (sibling of
   `think.js`), registered for **every** engine context, exposing:
   ```js
   sofuu.agent.run({ messages, tools, provider, model, maxSteps, recall, brainPath, onStep })
     → { answer, steps, usage, recalls }
   ```
   The chat driver *calls this* instead of owning the loop — chat keeps only
   rendering (`/why`, spinners, pickers). No feature duplication, one
   implementation, chat-tested behavior becomes every embedder's behavior.
2. **Recall injection** becomes the `recall:true` + `brainPath` options on
   `agent.run` (driver's `recallContext`/`markPositive` logic moves).
3. **MCP tool wiring**: `tools` param accepts `{mcp: [{name, command}]}`
   (headless-supplied config) or falls back to `config_root/mcp.json`.
   Fixes today's `$HOME`-only discovery.
4. **Desktop dogfood:** rewrite `sofuu-desktop/sofuu-bridge.js` to consume
   `sofuu_rt_call("ai.stream"/"agent.run")` via direct lib linking (or keep
   subprocess for v1 but with zero TTY calls — delete `__ttyRaw()`) — the
   sample proving TUI isn't dragged in. Bonus: desktop gains the agent loop
   for free.

## H4 — Platform packs (~4–5 days total, staggered) ✅ DONE (2026-08-20)

| Pack | Target triples | Notes |
|---|---|---|
| macOS | `aarch64-apple-darwin`, `x86_64-apple-darwin` | `.a` + `.dylib` + header tarball into `dist/`. |
| Linux | `x86_64/aarch64-unknown-linux-musl` | Static musl; reuses `scripts/cross/` (the D4 gap "build.rs honors CC_<target>" gets fixed here). |
| iOS | `aarch64-apple-ios`, `-ios-sim` | `xcframework` via `xcodebuild -create-xcframework`; **no-JIT = App-Store-safe** (market this). Swift sample wraps the C API in 30 lines. QTSQ frameworks already link ObjC/Metal — verify they survive device linking (desktop-linked frameworks on iOS differ; degraded-QTSQ build is the fallback). |
| Android | `aarch64/armv7/x86_64-linux-android` | NDK via `cargo ndk` or `scripts/cross/` extension; `.so` per ABI + Kotlin (JNI) sample; minSdk 24. |
| Windows | `x86_64-pc-windows-msvc` | **v2** — libcurl/libuv statics need a Windows cross story; honest v2 note, not blockers. |
| WASM/edge | `wasm32-unknown-unknown` | **v2 spike**: QuickJS is portable C, but libuv/curl are not — so the WASM tier is a documented subset (engine + TS + bundler + memory-without-QTSQ; networking via host imports later). A spike report, not a promise. |

Per pack: `scripts/dist/<platform>.sh`, CI job, size recorded next to the
artifact table in `docs/EMBEDDING-DIST.md`.

## H5 — Samples & docs (~2 days) ✅ DONE (2026-08-20)

- `examples/headless/c_embed.c` — ~15 lines: `sofuu_rt_new` →
  `sofuu_rt_call("ai.complete", …)` → print → `sofuu_rt_free`. Compiled and
  **run** in CI on macOS+Linux.
- `examples/headless/c_rlm.c` — RLM over a bundled 400KB text with a mock
  provider, prints the trace (doubles as the RLM headless proof).
- `examples/headless/SwiftSample/` — iOS: xcframework import, one
  `SofuuBridge.swift`, streaming to a `Text` view.
- `examples/headless/KotlinSample/` — Android JNI bridge + Compose screen.
- `docs/EMBEDDING.md` — the H0 contract plus per-platform "getting
  embedded" pages; README gains an "Embed Sofuu" section (with the no-JIT
  iOS line and the 2MB story).
- **Licensing note for embedders** (one paragraph, `docs/EMBEDDING.md`):
  runtime is MIT; QTSQ-enabled builds fall under LICENSES/QTSQ-FORMAT.txt;
  QTSQ-free builds are pure MIT. State which artifacts are which.

## H6 — CI / QA gates (~1–2 days) ✅ DONE (2026-08-20)

- `release.yml`: macOS, Linux musl (x64/arm64) build+size-check;
  `ci.yml`: compiles `c_embed.c` against the artifact and runs it vs a mock
  provider on both runners.
- **ABI guard:** `scripts/check_abi.sh` — `nm` export diff vs a checked-in
  baseline; new public symbol = deliberate PR change.
- **Fuzz the funnel:** `sofuu_rt_call` with arbitrary method/JSON must never
  crash — add `crates/fuzz/fuzz_targets/capi_call.rs`.
- **Leak smoke:** C sample under ASAN/LSAN on Linux completes clean-ish
  (QuickJS baseline leaks documented, no *new* leaks from capi paths).

---

## Sequencing & effort

| # | Item | Effort | Depends on |
|---|---|---|---|
| H0 | Embedding contract doc | 1 day | — |
| H1 | libsofuu crate + ABI + funnel + multi-instance test | 3–4 days | H0 |
| H2 | Hostile-host audit fixes (exit/signals/logs/HOME) | 2–3 days | H1 (flag plumbing) |
| H3 | `sofuu.agent` down-level + desktop dogfood | 2 days | H1 |
| H4 | Platform packs (iOS/Android/Linux/macOS; Win/WASM = v2) | 4–5 days | H1, H2 |
| H5 | Samples + EMBEDDING.md + README | 2 days | H1–H4 samples per pack |
| H6 | CI + ABI guard + fuzz | 1–2 days | H1, H4 |

**Total ≈ 2.5–3 weeks.** Independent of PLAN-RLM except `rlm.query` joining
the funnel when R1 lands (funnel methods are additive — no ABI churn).
Synergy: H1's registration work reuses the `register_global_fn` pattern and
feeds M0 of the Rust-migration plan; H2's audit de-risks Track A leftovers.

## Risks

| Risk | Mitigation |
|---|---|
| Hidden process-global state (mcp HACK class, env vars, curl global init) | H1 multi-instance test + H2 grep audit; curl handled once-per-process behind a `std::once` wrapper. |
| `exit()` missed on some error path | CI grep + fuzz the funnel (`capi_call`) — a crash/exit is a failed fuzz run. |
| Scope creep into full Node-compat | Contract caps the funnel method list per release; `[js/sofuu.api.json]` manifest is the reviewed diff. |
| iOS QTSQ framework drag (Metal/Security on device) | QTSQ-off flavor per pack (pure MIT story, already CI-supported); QTSQ-on flavor ships only after a real-device link test. |
| ABI promises too early | `experimental` tier for one release; version script from day one; ABI guard CI. |
| Desktop refactor snowballs | v1 keeps subprocess bridge, only strips TTY calls; in-process linking is v2 per `sofuu-desktop`'s own roadmap. |

## Done-state definition

- ✅ `cc c_embed.c libsofuu.a` → working AI call on macOS + Linux CI, no
  terminal involved, no `~/.sofuu` touched unless configured.
- ✅ `cc c_rlm.c libsofuu.a` → RLM long-context Q&A headless proof.
- ✅ iOS xcframework + Swift sample (`SofuuBridge.swift`); Android per-ABI `.so` +
  Kotlin sample (`SofuuBridge.kt` + `jni_bridge.c`); both QTSQ-off (pure MIT).
- ✅ `sofuu.agent.run` works identically in chat, `sofuu run`, and via the funnel.
- ✅ ABI guarded (`make abi-check`), funnel fuzzed (`capi_call.rs`).
- ✅ `docs/EMBEDDING.md` published with per-platform getting-started pages;
  README "Embed Sofuu" section live; `docs/EMBEDDING-DIST.md` artifact table.
- ✅ Desktop bridge needs zero TTY functions — `sofuu-bridge.js` rewritten
  (H3, 2026-08-20): `__ttyRaw`/`__ttyNormal`/`__readline` deleted, uses
  `sofuu.agent.run` via stdin/stdout JSON-lines.
- ⬜ iOS simulator / Android emulator run on CI (scripts are ready; CI device
  testing is a v2 item).
