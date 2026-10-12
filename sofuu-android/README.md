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
- `res/xml/network_security_config.xml` allows cleartext to `10.0.2.2`
  only (the emulator's host alias, for local-model testing); everything
  else is HTTPS-only.
- Min SDK 24, ABIs `arm64-v8a` + `x86_64`.

## Known gaps

- Workspace switching is app-internal for now (the drawer adds the
  app's `filesDir/workspace`); a SAF *tree* picker + path mapping for
  shared storage is the next step.
- The desktop's Shortcuts pane was deliberately **dropped** — macOS
  keyboard shortcuts have no phone equivalent, so it was removed rather
  than shipped as a list of bindings that cannot fire. Settings is
  General / Appearance / Models / Providers / Usage.
- No release signing config yet (add a `keystore.properties` +
  `signingConfigs` to `app/build.gradle.kts`). A release build also
  needs the engine cross-built with `SOFUU_ANDROID_PROFILE=release`,
  which currently does not compile.
- `armeabi-v7a` is not built, so devices older than ~2017 (32-bit only)
  cannot install.
- Switching to a session shows an empty page until the engine resumes it.
  This is deliberate and matches the desktop (`store.ts`: "empty if the
  session never had a transcript in this window"); read a past chat with
  the history sheet (read-only) or `/resume`.
- The embedding + voice ABI (`sofuu_embed_local/batch/image/info`,
  `sofuu_voice_transcribe/speak`) is not yet shimmed in the JNI bridge.
- The whole engine console stream was unreachable until the
  `sofuu_embed_set_log_cb` shim landed; it now appears in logcat under
  the `sofuu-js` tag alongside the app's own `sofuu-engine` tag.

## License

MIT for the app. A QTSQ-linked engine build additionally falls under
`LICENSES/QTSQ-FORMAT.txt` (see `docs/OPEN-CORE.md`). JetBrains Mono
(OFL-1.1) is bundled in `app/src/main/res/font` — its license is in
`licenses/JetBrainsMono-OFL.txt`.
