<p align="center"><img src="../sofuu-landing/public/logo.png" width="118" alt="Sofuu logo"></p>

# Sofuu for Android

The Sofuu runtime as a native Android app — Kotlin + Jetpack Compose UI
with the full engine (`libsofuu`) linked in-process through JNI. Same
product as the macOS app (sofuu-desktop): streaming agent chat, coding
tools with approval prompts, the encrypted local brain, session resume,
the slash-command surface, and the same cream/warm-dark design language.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Kotlin / Jetpack Compose (app/src/main/kotlin/com/sofuu)  │
│  ui/        Shell (drawer + topbar), Composer + sheets,   │
│             Settings (6 panes), ChatScreen, Markdown,     │
│             DiffView, CloudBackground                      │
│  store/     AppStore (the engine-event reducer — a port   │
│             of the desktop's store.ts), ConfigStore       │
│             (config.json + Keystore key store), Models    │
├──────────────────────────────────────────────────────────┤
│  com.sofuu.engine.SofuuEngine                             │
│  one worker thread owns the runtime; JS events are pumped │
│  through a queue; cancel/approvals answer on that thread  │
├──────────────────────────────────────────────────────────┤
│  cpp/jni_bridge.c — the documented C ABI (sofuu_embed.h)  │
│  live: abi_version / rt_new / rt_eval / rt_free /          │
│        embed_set_log_cb · shimmed but unused: rt_call,    │
│        rt_call_stream, rt_cancel, embed_info              │
├──────────────────────────────────────────────────────────┤
│  lib/arm64-v8a/libsofuu.so — QuickJS + libuv + curl/OpenSSL │
│  + SIMD + QTSQ (brain + session store)                    │
└──────────────────────────────────────────────────────────┘
```

Turn driving mirrors the desktop backend but without the WebView:
`sofuu.chat.submit(text, {onEvent})` is fired as a promise; a pump
drains the engine's event queue every ≤50 ms (each drain also advances
the libuv loop, so the turn progresses between polls) and pushes events
into a `SharedFlow`. Cancel + approvals ride `chat.js`'s `__host_poke`
on the same thread — the thread is never blocked mid-turn, unlike the
desktop's blocking eval.

Android-specific bindings:

| Desktop | Android |
|---|---|
| NSMenu actions | in-app overflow menu + drawer |
| Sidebar | slide-over workspace drawer (hardware back closes it) |
| Keychain (security-framework) | EncryptedSharedPreferences (Keystore-wrapped) |
| `$HOME/.sofuu` | app-private `filesDir/.sofuu` (`setenv("HOME")` at boot) |
| File picker (NSOpenPanel) | SAF picker; picked files are copied into the workspace and @mentioned, images ride as data URLs |
| Dock badge / notifications | POST_NOTIFICATIONS channel for approvals + finished turns |

## Build

Prerequisites: Android SDK + NDK 27 (`~/Library/Android/sdk/ndk/27.0.12077973`),
JDK 17, `rustup target add aarch64-linux-android x86_64-linux-android`,
and (for the encrypted brain) the QTSQ checkout (`SOFUU_QTSQ_SRC`,
default `~/projects/black-hole-disk`).

```bash
# 1. the engine — per-ABI cross builds (libuv → OpenSSL → curl+TLS →
#    QTSQ → sofuu-capi), then stages the .so into app/src/main/jniLibs
make mobile-engine                 # debug profile (fast iteration)
make mobile-engine-release         # opt-z + fat LTO + stripped (shipping)

# 2. the app
make mobile-apk                    # or: cd sofuu-android && ./gradlew :app:assembleDebug
adb install -r sofuu-android/app/build/outputs/apk/debug/app-debug.apk

# shipping build
make mobile-engine-release
cd sofuu-android && ./gradlew :app:assembleRelease
# → app/build/outputs/apk/release/app-release-unsigned.apk
```

Everything is cached under `dist/`: `libuv-android-*`, `openssl-android-*`,
`libcurl-android-*` (TLS via the OpenSSL we build — the zig cross curl is
deliberately TLS-free), `qtsq-android-*` (libqtsq.a + libqtc.a + libz.a).

Notes:

- The engine `.so` is QTSQ-linked here, so chats + the brain survive app
  restarts. The public SDK pack (`make dist-android`) stays QTSQ-free.
- `libsofuu.so` carries `-Wl,-soname,libsofuu.so`: a Rust cdylib has no
  DT_SONAME by default and the JNI link would otherwise bake a
  build-time absolute path into the APK (dlopen fails on device).
- **Both** native libs link `-Wl,-z,max-page-size=16384`. `targetSdk 35`
  means the APK must load on Android 15+ devices with a 16 KiB page size,
  and the NDK toolchain only aligns the first PT_LOAD by default. Without
  the flag the engine `.so` is rejected at load time on current Pixels.
  Verify with `llvm-readelf -l lib/*/libsofuu*.so` — every LOAD must say
  `0x4000`.
- The dot-product kernel is chosen at **runtime**
  (`sofuu_cpu_has_dotprod` in `src/simd/infer_kernels.cpp` reads
  `AT_HWCAP`), not at compile time. `build.rs` passes
  `-march=armv8.2-a+dotprod` for every aarch64 target, so on a CPU older
  than ARMv8.2 the sdot instructions would be emitted and fault with
  SIGILL; the guard falls back to the generic kernel instead.
- `res/xml/network_security_config.xml` allows cleartext to `10.0.2.2`
  only (the emulator's host alias, for local-model testing); everything
  else is HTTPS-only.
- Min SDK 24, ABIs `arm64-v8a` + `x86_64`.

## What the phone adds

The runtime is the same engine the CLI embeds; a phone cannot do less of
it, and in two places it can do more.

### Host tools

`sofuu_rt_register_host_tool` lets the host hand the agent a native
capability with no JS in the loop — the QuickJS sandbox cannot reach the
Android SDK, so anything platform-specific has to be answered by the host.
It was declared in `include/sofuu_embed.h` and wired through
`crates/sofuu-capi`, and exposed by **no** binding at all until now.

Four ship, all permission-light so none of them silently fails:

| Tool | Does |
|---|---|
| `device_info` | model, Android version, battery, free storage and memory |
| `open_url` | opens a link in the phone's browser |
| `share_text` | opens the Android share sheet |
| `vibrate` | a short buzz, to get the user's attention |

They register as ordinary agent tools, so `/tools` lists them, the
permission profile applies (`plan` refuses them), and they land in the
normal tool timeline. `Settings → General → Phone tools` shows what
actually registered at boot.

To add one, implement `SofuuBridge.HostTool` and call
`bridge.registerHostTool(name, description, schemaJson, tool)`. Return a
JSON string for a synchronous answer, or null and settle later with
`resolveHostTool` — safe from any thread.

### Surfaces the CLI reaches through a terminal

Settings carries panes for the runtime's own capabilities, each routed
through the engine's existing eval path:

- **Agents** — register an agent from the phone (`sofuu.agent.define`)
  and run one headlessly (`sofuu.agent.run`). An agent added here is a
  first-class registry entry the chat turn, `/agents` and `delegate` all
  see, and its tools may include the host tools above.
- **Brain** — count, recall search with per-hit scores, forget,
  consolidate, decay, the embedding-space manifest with unavailable
  artifacts called out, text similarity, and the multimodal tower
  (pick an image, embed it, compare against a text query in img1's space).
  Every call goes through `sofuu.agent.brainFor`, the ONE cached handle —
  a QTSQ flush rewrites the whole file, so a second handle would clobber
  the first on every write.
- **Context gates** — `sofuu.ml.info` and `ml.workset`, gold-label the most
  recent call wrong or wasted (`ml.feedback`), and ask
  `ml.supervisor.check` what a gate would say before the agent acts.

### Engine controls

The library configures itself through `std::env::var`. A CLI can
`export SOFUU_NO_SHELL=1`; a phone had no equivalent at all. General →
Engine controls now exposes:

`SOFUU_NO_TOOLS` · `SOFUU_NO_SHELL` · `SOFUU_NO_ML` ·
`SOFUU_MEMORY_BACKEND` · `SOFUU_WEB_ENGINE` ·
`SOFUU_WEB_ALLOW_INTERNAL` · `SOFUU_STALL_TIMEOUT_SECS` ·
`SOFUU_CHAT_MAX_STEPS`

Most are read inside the agent loop, so they apply to the next message.
`SOFUU_NO_ML` is the exception — it gates *module registration* in
sofuu-core, so it needs a restart and the UI says so rather than leaving a
switch that appears to do nothing.

### Local models

Settings → Models → **On this device** imports a `.gguf` file. The pick is
copied into `filesDir/models/` (the engine opens models by absolute path
through `mmap`, so a `content://` URI is useless to it) and selecting one
mirrors the CLI's `/model x.gguf` branch: `provider` becomes `local`,
`base_url` and `api_key` are cleared, and the caps read out of the GGUF
header are applied instead of guessed.

No extra plumbing is needed to make the turn work — `agent.js` already
calls `sofuu.ai.stream`, and `rt/ai.rs` routes that to the in-process
engine whenever the provider is Local or the model path ends in `.gguf`.
The app checks the model actually generates before switching to it, so a
bad file reports its error instead of failing the next turn.

## Known gaps

- Workspace switching is app-internal for now (the drawer adds the
  app's `filesDir/workspace`); a SAF *tree* picker + path mapping for
  shared storage is the next step.
- The desktop's Shortcuts pane was deliberately **dropped** — macOS
  keyboard shortcuts have no phone equivalent, so it was removed rather
  than shipped as a list of bindings that cannot fire. Settings is
  General / Appearance / Models / Providers / Usage.
- Release signing is wired but unconfigured: drop a `keystore.properties`
  next to `settings.gradle.kts` and `assembleRelease` picks it up. With no
  file present the release APK is **unsigned** and Gradle says so.
- `armeabi-v7a` is not built, so devices older than ~2017 (32-bit only)
  cannot install.
- Switching to a session shows an empty page until the engine resumes it.
  This is deliberate and matches the desktop (`store.ts`: "empty if the
  session never had a transcript in this window"); read a past chat with
  the history sheet (read-only) or `/resume`.
- The whole engine console stream was unreachable until the
  `sofuu_embed_set_log_cb` shim landed; it now appears in logcat under
  the `sofuu-js` tag alongside the app's own `sofuu-engine` tag.
- **MCP is not reachable.** `sofuu.mcp.connect` spawns a child process
  over stdio, and Android 10+ forbids executing files from app-writable
  storage (W^X). There is no shell on a phone and no way to launch an MCP
  server, so no MCP UI is built — a connect button could only ever fail.
  Everything else in `sofuu.mcp` depends on a spawned server and is out
  for the same reason. An in-process or bundled-native MCP server would
  need engine work, not an app change.
- `__sofuu_gc` is exposed under Brain → Memory rather than automatically:
  one runtime lives for the whole process on a phone, so nothing else
  would run the collector at the moment a big fan-out wants it.
- No automated tests. There is no `src/test/` and no `src/androidTest/`
  despite the junit dependencies in `app/build.gradle.kts`, and CI only
  runs `cc -fsyntax-only` over the *SDK's* bridge — it never compiles this
  app.

## License

MIT for the app. A QTSQ-linked engine build additionally falls under
`LICENSES/QTSQ-FORMAT.txt` (see `docs/OPEN-CORE.md`). JetBrains Mono
(OFL-1.1) is bundled in `app/src/main/res/font` — its license is in
`licenses/JetBrainsMono-OFL.txt`.
