# Sofuu for iOS

The iOS sibling of `sofuu-android/`: the same product, the same engine, the
same UI, rebuilt in **Swift + SwiftUI only** — no Kotlin, no Compose, no
JNI, no cross-platform UI framework.

```
sofuu-ios/
├── project.yml                  XcodeGen spec — the .xcodeproj is generated
├── SofuuApp/
│   ├── SofuuApp.swift           @main: owns the store, the theme root, scenePhase
│   ├── Info.plist               ATS local-networking, file sharing, mic usage
│   ├── SofuuApp.entitlements
│   ├── Assets.xcassets/         AppIcon + AccentColor (#2B6CB0 / #84B3E8)
│   ├── JSON.swift               JSONObject + org.json-style opt* accessors
│   ├── Engine/
│   │   ├── SofuuBridge.swift    the C ABI, called directly (no JNI)
│   │   └── SofuuEngine.swift    the worker thread, mailbox, event pump
│   ├── Store/
│   │   ├── Models.swift         AppState, ChatMessage, SessionInfo, Usage, …
│   │   ├── Timeline.swift       verb-first tool rows, icon ids, phase labels
│   │   ├── ConfigStore.swift    config.json + mobile.json + Keychain + UiPrefs
│   │   └── AppStore.swift       @MainActor reducer + full slash surface
│   ├── UI/
│   │   ├── Shell.swift          topbar, drawer/sidebar, history, approvals
│   │   ├── CloudBackground.swift the procedural cloud + its tile fallback
│   │   ├── Notifier.swift       UNUserNotificationCenter
│   │   ├── ComposerSheets.swift model / effort / layers / mode / permissions
│   │   ├── chat/                ChatScreen, Markdown, DiffView
│   │   ├── composer/Composer.swift
│   │   ├── settings/SettingsScreen.swift
│   │   ├── components/Icons.swift
│   │   └── theme/Color.swift    every design token, light + dark
│   └── Shaders/SofuuCloud.metal the desktop cloud shader as MSL
└── .devstub/                    stub `Sofuu` module for fast type-checking
```

## The one architectural difference: no JNI

Android reaches the engine through a C shim called over JNI
(`cpp/jni_bridge.c` → `libsofuu.so`). iOS doesn't need that layer at all —
Swift talks to the same C ABI v2 **directly**, through the `Sofuu` module
that `dist/libsofuu.xcframework` exports:

```swift
import Sofuu

var out: UnsafeMutablePointer<CChar>?
let rc = source.withCString { sofuu_rt_eval(rtPtr, $0, &out) }
guard rc == 0, let out else { return nil }
defer { sofuu_free(out) }
return String(cString: out)
```

That is one less layer and one less copy of the contract to keep in sync.

QuickJS is a pure interpreter with no JIT, so the App Store's
executable-memory policy is satisfied by construction — the same property
the Android build relies on.

## What is *identical* to Android

Deliberately, and by request — this is a port, not a redesign:

* **The whole event-pump architecture.** One thread owns the runtime;
  `chat.submit(text, {onEvent: …})` fires a promise; a pump drains
  `globalThis.__evq` / `__rq` every ≤50 ms, advancing the libuv loop on
  each drain. Cancel and approvals ride `chat.js`'s `__host_poke` on the
  same thread, so it is never blocked mid-turn.
* **The reducer.** `applyStep` / `stepToTimeline` are line-for-line ports,
  including the tool-row matching, the checklist fold, the compact
  lifecycle and the `firstMatch` regexes.
* **The slash surface.** All 20 commands, including the honest
  "this is CLI-only" replies rather than a silent "unknown command".
* **Session discipline.** `MAX_SESSIONS = 7`, the message/ctx caches, the
  park-and-clear dance in `newChat`, resume-only-on-boot.
* **Every design token.** Cream light / warm dark, ported from
  `styles.css` through `Color.kt`, accent included (`#2B6CB0` light,
  `#84B3E8` dark — blue, not the desktop purple, so it matches the cloud).

## Where iOS necessarily differs

| Android | iOS | Why |
|---|---|---|
| `CompletableFuture` | `async`/`await` + `CheckedContinuation` | FIFO deque of continuations, matched by arrival order |
| `LinkedBlockingQueue<Runnable>` | `NSCondition` + `[() -> Void]` mailbox | same semantics, no JDK |
| `org.json` | `JSONSerialization` + hand-written `opt*` | mirrors `org.json` exactly: missing key or null → default, never a throw |
| `EncryptedSharedPreferences` / Keystore | Keychain (`kSecClassGenericPassword`) | platform equivalent |
| `SharedPreferences` | `UserDefaults` | for `UiPrefs` only |
| adaptive **drawer** | drawer on compact, **pinned sidebar** on regular | see below |
| SAF picker (`ACTION_OPEN_DOCUMENT`) | `.fileImporter` + security-scoped URLs | same handler: images → `data:` URLs, other files → copied into the workspace as `@mentions` |
| AGSL `RuntimeShader` (API 33+) | Metal `MTKView` + `.metal` | same shader, same uniforms |
| baked-tile fallback (< API 33) | baked-tile fallback | safety net; essentially unreachable on iOS |
| hardware back | veil tap / drag-to-dismiss | no back button to intercept |

### Layout: the drawer becomes a sidebar on iPad

Android's `Shell.kt` always shows the workspace drawer as an 82 %-wide
slide-over. iOS keeps that on **compact** width, and pins the *same
content* as a permanent sidebar on **regular** width (iPad full-screen,
Max-class iPhones in landscape). The rows, actions and order are identical
— only the chrome changes and the hamburger disappears. A slide-over on a
12.9" screen wastes the entire point of the size class.

## Build

```bash
# 1. Fast feedback — no engine required (~3s, needs only Xcode)
make ios-typecheck        # swiftc -typecheck on all 19 files + compiles the .metal

# 2. The engine framework
make ios-engine           # scripts/dist/ios.sh all → dist/libsofuu.xcframework

# 3. The project + the app
make ios-project          # xcodegen generate  (brew install xcodegen)
make ios-app              # xcodebuild -sdk iphonesimulator
make ios-sim              # install + launch on a booted simulator
```

`make ios-typecheck` exists because the app `import Sofuu`, which only
resolves once the xcframework exists. Without it you cannot check a single
line of Swift until a multi-minute Rust cross-build finishes.
`.devstub/` supplies a header stub and a modulemap declaring the `Sofuu`
module, so the type-checker resolves the import immediately. It never
links and never ships.

## Build status

The **app** is complete and verified:

* All 19 Swift files type-check clean against the iOS 16 simulator SDK
  (Swift 6.2.3 / Xcode 26.2) — zero errors, zero warnings.
* `Shaders/SofuuCloud.metal` compiles to AIR for iOS **and** renders
  correctly. It was rendered offscreen at 512×512 and inspected: the
  light field spans luminance 213–247 (mean 232.6, σ 6.71, 261 distinct
  colours) with the wispy periwinkle lobes and edge fade intact, and the
  dark field spans 30–41. Not a flat screen — which is the exact failure
  the Android source comments warn about.

  The renders are checked in as a regression baseline, so a future shader
  edit can be diffed against them:

  | | |
  |---|---|
  | `docs/cloud-metal-light.png` | the Metal path, light, t=0 |
  | `docs/cloud-metal-dark.png` | the Metal path, warm dark, t=0 |
  | `docs/cloud-tile-seamless-3x3.png` | the fallback tile, tiled 3×3 — the seam test |

The **engine** for iOS now packages too, and the app runs. Measured on this
machine:

| Step | Status |
|---|---|
| OpenSSL 3.3.2 for iOS (device + simulator) | ✅ `platform IOS` / `platform IOSSIMULATOR`, 0 macOS objects |
| libcurl 8.6.0 for iOS, OpenSSL merged in | ✅ 10 MB each, zlib deliberately disabled |
| `libuv` cross-built for iOS | ✅ `platform IOS`, 0 macOS objects |
| All Rust crates compile for both slices | ✅ `sofuu-core`, `sofuu-ffi`, `sofuu-capi` |
| Merged, self-contained engine archive | ✅ 29 MB per slice — all 264 `_uv_*`/`_curl_*` symbols resolved |
| `libsofuu.xcframework` | ✅ 58 MB, `Sofuu.framework` for device + simulator |
| `xcodebuild` | ✅ **BUILD SUCCEEDED** |
| Runs in the iOS Simulator | ✅ launched and rendering — `docs/ios-simulator-running.png` |

> **If you previously attempted a build before the framework existed, run
> Product ▸ Clean Build Folder (⇧⌘K) once.** Xcode caches the *failed* module
> lookup for `Sofuu`; the cache is not invalidated by the framework appearing,
> so a stale DerivedData keeps reporting all 20 `cannot find 'sofuu_*' in
> scope` errors even though nothing is wrong. `xcodebuild clean` is enough —
> deleting DerivedData is not necessary.

### What had to be built for iOS (and why)

Android links a `libsofuu.so` against the NDK sysroot, which already carries
libz and friends. iOS ships **no system libcurl and no system OpenSSL**, so the
whole TLS stack has to be cross-built and folded into the engine archive:

* `scripts/mobile/ios_env.sh` — iOS toolchain plumbing (SDK, `CC`/`AR`,
  per-slice identity, CMake flags). The iOS twin of `ndk_env.sh`.
* `scripts/mobile/build_openssl_ios.sh` — static OpenSSL per slice.
* `scripts/mobile/build_libcurl_ios.sh` — static, TLS-capable libcurl with
  OpenSSL merged into `libcurl.a`, zlib disabled on purpose.

`curl 8.6` has **no SecureTransport backend** (removed in curl 8.0), so "just
use Apple's TLS" was never an option; OpenSSL it is, matching Android.

### Bugs found and fixed getting there

Each of these was confirmed by running it, not by reading:

1. **`SOFUU_UV_DIR` was a relative path.** `cargo` runs build scripts with the
   cwd set to the crate directory (`crates/sofuu-ffi/`), so
   `deps/libuv/build-ios-device` resolved to nothing and `build.rs` panicked
   with a misleading *"libuv not built"* while the archive sat right there.
2. **`libuv` was being built for macOS.** `-DCMAKE_OSX_SYSROOT` alone is not
   enough; without `-DCMAKE_SYSTEM_NAME=iOS` CMake configures for Darwin.
3. **`cargo build` also tried to emit a `.dylib`**, and that link is what fails
   on `-lcurl`. Now `cargo rustc --crate-type staticlib`.
4. **`source` inherits the caller's positional parameters.** `ios_env.sh`'s
   convenience auto-resolve saw `$1=deps` and died with *"unknown iOS slice:
   deps"*. It now matches only real slice names — the same trap would have
   broken `build_libcurl_ios.sh all`.
5. **Exporting `CC` silently builds OpenSSL for macOS.** OpenSSL's `Configure`
   honours an inherited `$CC`, and the resulting Makefile uses the raw Xcode
   `clang` with **no `-isysroot`** — which defaults to
   `Target: arm64-apple-darwin…`. The `-xcrun` targets exist precisely to
   avoid this, so the script now unsets `CC`/`CXX`/`AR`/`RANLIB` for Configure.
6. **`-mios-version-min` is the wrong flag for the simulator slice** — with an
   iPhoneSimulator sysroot it silently produces `platform IOS` (device!). Only
   `-mios-simulator-version-min` yields `platform IOSSIMULATOR`. Both are now
   per-slice in `ios_env.sh`.
7. **`unset SOFUU_QTSQ_DIR` does NOT disable QTSQ.** `crates/sofuu-ffi/build.rs`
   falls back to a hardcoded `/Users/priyanshuboruah/projects/black-hole-disk`
   when the variable is *absent* — contradicting the comment two lines above
   it, which says there is deliberately no developer-machine path. On this
   machine that checkout exists and holds an **arm64 macOS** build, so unsetting
   pulled 102 macOS `qtsq_*.o` objects into the iOS archive (10 MB → 20 MB).
   The repo-wide `dist/no-qtsq` marker directory is the correct idiom.
8. **A staticlib does not link, so the archive was not self-contained.**
   `cargo rustc --crate-type staticlib` archives Rust + QuickJS objects and
   drops every `rustc-link-lib`, leaving 371 `_uv_*` and 24 `_curl_*` symbols
   undefined. Android never sees this because its `.so` is a shared object. The
   merge step now folds libuv and libcurl into the engine archive with
   `libtool -static`, the same trick used one level down for OpenSSL.
9. **The app's Swift module collided with the engine's module.** `PRODUCT_NAME`
   is `Sofuu`, which made the target's Swift module `Sofuu` — the exact name of
   the Clang module the framework exports. Swift silently refuses to import a
   module with the current module's name, so `import Sofuu` was ignored with
   only a warning and every `sofuu_*` call failed. `PRODUCT_MODULE_NAME` is now
   `SofuuApp`. (The type-check harness had `-module-name SofuuApp` all along,
   which is why it passed while Xcode failed.)
10. **`-library … -headers …` xcframeworks cannot satisfy a Swift `import`.**
    Xcode copies the resolved slice's `.a` into `Build/Products` so the linker
    finds it, but there is no `.framework` bundle for Clang to resolve the
    module against. The xcframework now contains a real `Sofuu.framework` per
    slice, and its modulemap says `framework module` — inside a bundle a plain
    `module` is silently skipped ("lacks the 'framework' qualifier") and the
    module comes out empty.
11. **libc++ was not linked.** The engine contains a C++ translation unit
    (`src/simd/infer_kernels.cpp`); a Swift app does not link libc++ on its
    own, so the app compiled cleanly and died at link on ~15 `std::__1::…`
    symbols. `OTHER_LDFLAGS` now carries `-lc++`. `build.rs` emits `-lc++` only
    for `macos`, and adding it there would not help — a staticlib does not
    link, so the app link is Xcode's to configure.

`ios_env.sh` now also carries the checks that would have caught several of
these immediately: `ios_archive_platform` (whole-archive platform census) and
`ios_assert_no_macos`. These census **every** member, not the first one — a
single-member probe reported `platform: IOS ✓` for the archive that contained
102 macOS objects.

### Known limitations

* **No x86_64 simulator slice.** The engine ships arm64 only, so
  `EXCLUDED_ARCHS[sdk=iphonesimulator*] = x86_64` is set. Apple Silicon
  simulators are arm64, so this costs nothing here; an Intel Mac would need a
  third slice (OpenSSL + curl + libuv + Rust for `x86_64-apple-ios`).
* **Device builds are unexercised.** Everything above was verified against the
  simulator. The device slice is built and platform-verified, but no
  provisioning/signing run has happened.
* **`build.rs`'s hardcoded QTSQ fallback is still there.** It is an uncommitted
  local change from another workstream and contradicts its own comment; worth
  reverting.

## What the phone adds

The runtime is the same engine the CLI embeds; a phone cannot do less of
it, and in two places it can do more.

### Host tools

`sofuu_rt_register_host_tool` lets the host hand the agent a native
capability with no JS in the loop — the QuickJS sandbox cannot reach UIKit,
so anything platform-specific has to be answered from here. It was declared
in `include/sofuu_embed.h` and wired through `crates/sofuu-capi`, and
exposed by no binding at all until now.

Four ship, the same set Android ships, mapped to the platform:

| Tool | Does |
|---|---|
| `device_info` | model, iOS version, battery level, free storage, foreground state |
| `open_url` | opens a link in Safari |
| `share_text` | opens the iOS share sheet (`UIActivityViewController`) |
| `vibrate` | a haptic, through the UIKit feedback generators |

They register as ordinary agent tools, so `/tools` lists them, the
permission profile applies, and they land in the normal tool timeline.
`Settings → General → Phone tools` shows what actually registered at boot.

Two things the Swift side has to get right that Android's JNI shim also
had to: the C callback carries no tool name, so name + closure + context
pointer ride in `opaque` as one retained box; and the answer has to come
back through a buffer, because a Swift `String` has no address C can hold.
That buffer reallocates only when a *longer* result needs it, so a
long-running process cannot grow one per call.

To add one, call `bridge.registerHostTool(name:description:parametersJson:tool:)`
and return a JSON string, or nil and settle later with `resolveHostTool` —
safe from any thread.

### Surfaces the CLI reaches through a terminal

Settings carries panes for the runtime's own capabilities, each routed
through the existing eval path:

- **Agents** — register an agent from the phone (`sofuu.agent.define`) and
  run one headlessly (`sofuu.agent.run`). An agent added here is a
  first-class registry entry the chat turn, `/agents` and `delegate` all
  see, and its tools may include the host tools above.
- **Brain** — count, recall search with per-hit scores, forget,
  consolidate, decay, the embedding-space manifest with unavailable
  artifacts called out, text similarity, and the multimodal tower.
  Everything goes through `sofuu.agent.brainFor`, the ONE cached handle —
  a QTSQ flush rewrites the whole file, so a second handle would clobber
  the first on every write.
- **Context gates** — `sofuu.ml.info` and `ml.workset`, gold-label the most
  recent call wrong or wasted (`ml.feedback`), and ask
  `ml.supervisor.check` what a gate would say before the agent acts.

### Engine controls

The library configures itself through `std::env::var`. A CLI can
`export SOFUU_NO_SHELL=1`; a phone had no equivalent. General → Engine
controls now exposes:

`SOFUU_NO_TOOLS` · `SOFUU_NO_SHELL` · `SOFUU_NO_ML` ·
`SOFUU_MEMORY_BACKEND` · `SOFUU_WEB_ENGINE` ·
`SOFUU_WEB_ALLOW_INTERNAL` · `SOFUU_STALL_TIMEOUT_SECS` ·
`SOFUU_CHAT_MAX_STEPS`

Most are read inside the agent loop, so they apply to the next message.
`SOFUU_NO_ML` is the exception — it gates *module registration* in
sofuu-core, so it needs a restart and the UI says so.

### Local models

Brain → On this device imports a `.gguf`. The file is copied into the
container, because the engine `mmap`s a path and a Files URL is not one.
Selecting one mirrors the CLI's `/model x.gguf` branch: `provider` becomes
`local`, `base_url` is cleared, and the caps read out of the GGUF header
are applied rather than guessed. The model is probed with one short
generation before the config points at it.

## One deliberate improvement over Android

The cloud's **fallback tile is genuinely seamless here; Android's is not.**

Android's comment says *"sample noise on a torus so edges match"*, but the
implementation samples plain `valueFbm(a * 1.7 + 11.3, …)` over `a = u·2π`
— an aperiodic domain. Tiled 3×3, which is what a phone screen actually
shows, the repeat is plainly visible as a regular grid. It was baked and
inspected to confirm this before changing anything.

`CloudBackground.swift` wraps the lattice for real: every octave samples
`torusNoise`, whose hash indices are taken modulo that octave's period, and
each period is an exact integer multiple of the base (`f = 1 << o`, not
`freq *= 2.03`, which drifts 0.03 of a cell per octave and reopens the
seam). The span in `u` equals the period exactly. Measured wrap-column
difference is now within 1.3× of an ordinary interior column; it was 3.7×
before. The palette, weights and alpha curve are untouched, so it still
reads as the same cloud — only the grid is gone.

The Metal shader path (which is what actually runs on iOS) is an exact port
and was not modified.

## Status

| | Android | iOS |
|---|---|---|
| UI | Compose (Kotlin) | SwiftUI (Swift) — full port, 7 settings tabs |
| Host tools | ✅ 4 | ✅ 4 |
| Local models | ✅ | ✅ |
| Engine env knobs | ✅ 8 | ✅ 8 |
| Engine binding | JNI → `libsofuu.so` | direct C ABI → `libsofuu.xcframework` |
| Type-checks | ✅ | ✅ 23/23 clean |
| Shader | AGSL (API 33+) | Metal — compiles + renders verified |
| Engine packaging | ✅ | ✅ `Sofuu.framework` in an xcframework, both slices |
| App builds | ✅ | ✅ `xcodebuild` BUILD SUCCEEDED against the current engine |
| Runs on device | ✅ | ✅ in the Simulator (device build unprovisioned) |
| Engine links for iOS | ✅ | ✅ 160 symbols, incl. host tools |

## The engine must compile for iOS

The app links `dist/libsofuu.xcframework` statically, so it can only be
built against an engine that compiled for `aarch64-apple-ios`. That was
not true for a while: the inference engine's GPU paths were written
macOS-first and a handful of macOS-only constructs leaked into shared
code, so `sofuu-core` failed to compile for iOS and the xcframework went
stale. Typecheck alone will not catch this — `.devstub/sofuu_embed.h`
declares every symbol the app uses, so Swift resolves cleanly against a
library that cannot actually be produced.

When `sofuu-core` fails to build for iOS, the failure is in the engine, not
the app. Reproduce without the framework:

```bash
cargo check --release -p sofuu-core --target aarch64-apple-ios
```

Anything macOS-only referenced outside its `#[cfg]` will fail here. A new
Metal or MLX fast path has to keep its CPU fallback unconditional and gate
only the GPU branch — `block_graph.rs` and `lfm2.rs` both now compile for
macOS, iOS and the Android targets from the same source.
