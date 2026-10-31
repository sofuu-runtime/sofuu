# Sofuu for Android

Offline, private AI in your app: an encrypted local brain, embeddings with no
model file, agents, MCP tools, and speech — on-device, with no cloud account,
no API key in your APK, and no telemetry.

## Install

```groovy
// app/build.gradle
repositories {
    maven { url = uri("https://sofuu-runtime.github.io/sofuu/maven") }
    mavenLocal()   // optional: after `make android-publish-local`
    mavenCentral()
}

dependencies {
    implementation 'com.sofuu:sofuu-android:0.3.0'
}
```

The same layout is available from the public GitHub Pages repository above and
from `mavenLocal()` after `make android-publish-local`. A Maven Central release
additionally requires ownership of the `com.sofuu` group and signing
credentials; those are operator-owned and are never committed.

That is the whole integration. The AAR bundles the per-ABI `libsofuu` native
runtime, the JNI bridge, and `libc++_shared.so`; you do **not** copy `.so`
files by hand.

## Use

```kotlin
import com.sofuu.runtime.SofuuBridge

// One runtime per thread; create it on the thread that will use it.
val sofuu = SofuuBridge()

// Embed text offline — no key, no network, no model file.
val vec = sofuu.embedLocal("the sky is blue")   // 768-dim unit vector
println("dims: ${vec?.size}")

// The brain: encrypted, local, yours.
sofuu.eval("""
  globalThis.brain = sofuu.memory.open("$filesDir/brain.qtsq", 768);
""".trimIndent())
sofuu.call("memory.remember", """{"vec": $vec, "text": "the sky is blue", "role": "user"}""")
val hits = sofuu.call("memory.recall", """{"vec": $vec, "k": 3}""")
```

See `src/main/java/com/sofuu/runtime/SofuuBridge.kt` for the full wrapper
(eval, funnel calls, embedding, image, voice) and `SofuuVoice.kt` for
on-device STT/TTS with no network.

## Crypto and JWT verification

`sofuu.crypto` is available through the same bridge. The host supplies the
JWKS; no identity vendor, API key, or network call is built into the SDK.

```kotlin
val digest = sofuu.sha256("abc")                  // lowercase hex
val encoded = sofuu.base64urlEncode("hello")      // "aGVsbG8"
val decoded = sofuu.base64urlDecode("aGVsbG8")    // "hello"

val claims = sofuu.verifyJWT(
    token = sessionJwt,
    jwksJson = jwksJson,
    optionsJson = """{"issuer":"https://id.example.com",
                       "authorizedParties":["https://app.example.com"]}""",
)
// {"ok":true,"payload":{...}} or {"ok":false,"error":{...}}
```

A missing `azp` is accepted because native SDK tokens have no browser origin;
a present `azp` must be in `authorizedParties`.

`checkCaps()` reports `has_crypto`, `has_storage`, `has_ai`, `has_agent`, and
`has_memory` for the loaded build.

## Local pipeline

A local-first host runs its JS pipeline in-process with `eval` (one bundled
script) and drives it through `call`/`callStream`. No HTTP server is needed.

```kotlin
val sofuu = SofuuBridge()

// Start a local model stream; the callback runs on this worker thread.
val stop = AtomicBoolean(false)
sofuu.callStream(
    method = "agent.run",
    argsJson = """{"target":"researcher","task":"summarize this"}""",
) { event ->
    render(event)
    !stop.get()
}
```

`callResult` unwraps the C ABI envelope when you want the method's value
without the outer `{"ok":true,"result":...}`. Use `call` when the error
envelope itself matters.

## Streaming, cancel, and host tools

`callStream` drives `ai.stream`, `ai.streamLocal` and `agent.run`. It
**blocks on the calling thread** — the runtime is thread-affine and the
events *are* the call — so run it on a worker, never the UI thread. Every
stream ends with exactly one terminal `{"kind":"done",…}` event.

```kotlin
val stop = AtomicBoolean(false)   // the UI's "Stop" button sets this

sofuu.streamLocal("/models/qwen3-0.6b.gguf", "Once upon a time") { evt ->
    if (evt.contains("\"kind\":\"delta\"")) render(evt)
    !stop.get()                   // false cancels the stream
}
```

`registerHostTool` hands the agent a native capability — camera, location,
the share sheet — that the QuickJS sandbox cannot reach. Return a JSON
string to answer synchronously, or `null` and settle the call later from
any thread (the agent waits at most its tool timeout, 30 s by default):

```kotlin
sofuu.registerHostTool("take_photo", "Takes a photo", """{"type":"object"}""") { _, callId ->
    camera.take { path -> sofuu.resolveHostTool(callId, """{"path":"$path"}""") }
    null
}
```

Call `sofuu.close()` when the screen (and its worker) goes away: it frees
the runtime and releases the registrations, which otherwise pin your
`HostTool` lambdas — and through them, the Activity.

This whole surface is exercised on the host JVM with no device attached:
`make sdk-jvm-check` compiles this module's `jni_bridge.c` for the host OS
and runs its assertions through the real wrapper and runtime.

## Building and publishing the AAR

The AAR wraps the prebuilt native runtime, so build that first:

```bash
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973"
make dist-android                       # QTSQ-linked libsofuu per ABI
bash scripts/mobile/stage_android_sdk.sh

cd bindings/android
gradle :sofuu:assembleRelease
gradle :sofuu:publishToMavenLocal
```

`stage_android_sdk.sh` copies `libsofuu.so`, the NDK's
`libc++_shared.so`, and the C header into the module. Both are required:
`libsofuu.so` declares `NEEDED libc++_shared.so`, and the AAR's CMake target
is pure C, so AGP would not package the C++ runtime on its own.

The release AAR contains, per ABI:

```text
jni/<abi>/libsofuu.so
jni/<abi>/libsofuu_jni.so
jni/<abi>/libc++_shared.so
```

`libsofuu.so` links QTSQ in every published AAR: the encrypted brain codec is
a core runtime component, not an optional add-on. The AAR's `classes.jar`
therefore also contains `META-INF/LICENSES/QTSQ-FORMAT.txt` and
`META-INF/SOFUU-NOTICE.txt`. QTSQ is open source under the Apache License,
Version 2.0 — use, modification, and redistribution (including commercial use)
are granted under its terms. Staging refuses a codec-less binary, and
`verifyAarContents` refuses an AAR missing the native libraries or either
legal notice.

### Remote publication

The module has `maven-publish` configured for `com.sofuu:sofuu-android` with a
POM, sources JAR, and Gradle module metadata. Supply the repository and
signing values in the operator's environment:

```bash
export MAVEN_REPO_URL=https://your-maven-repository
export MAVEN_REPO_USERNAME=...
export MAVEN_REPO_PASSWORD=...
export MAVEN_SIGNING_KEY="$(cat signing-key.asc)"
export MAVEN_SIGNING_PASSWORD=...
export REQUIRE_SIGNING=1

cd bindings/android
gradle :sofuu:publishReleasePublicationToRemoteRepository
```

Maven Central additionally requires ownership of the `com.sofuu` group and a
signed release; no credentials or private keys are stored in this repository.

## Manual (non-Gradle) integration

If you are not using the AAR — e.g. a host that already manages its own
`.so` layout — the original sample still works and is the reference for the
CMake wiring: `examples/headless/KotlinSample/` (`CMakeLists.txt`,
`jni_bridge.c`, `SofuuBridge.kt`). It must also package
`libc++_shared.so`.

## License

MIT for the runtime. QTSQ, the encrypted-store engine linked into every
published AAR, is open source under the Apache License, Version 2.0
(`LICENSES/QTSQ-FORMAT.txt`, https://github.com/haruhito2666/qtsq). See
`docs/EMBEDDING-DIST.md`.
