# Sofuu for Android

Offline, private AI in your app: an encrypted local brain, embeddings with no
model file, agents, MCP tools, and speech — on-device, with no cloud account,
no API key in your APK, and no telemetry.

## Install

```groovy
// app/build.gradle
dependencies {
    implementation 'com.sofuu:sofuu-android:0.2.0'
}
```

That is the whole integration. The AAR bundles the per-ABI `libsofuu` native
runtime and the JNI bridge; you do **not** copy `.so` files by hand.

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

## Building the AAR yourself

The AAR wraps the prebuilt native runtime, so build that first:

```bash
make dist-android            # → dist/libsofuu-android-{aarch64,x86_64}.so
# stage the .so into the library, then:
cd bindings/android
mkdir -p sofuu/src/main/jniLibs/{arm64-v8a,x86_64}
cp ../../dist/libsofuu-android-aarch64.so sofuu/src/main/jniLibs/arm64-v8a/libsofuu.so
cp ../../dist/libsofuu-android-x86_64.so  sofuu/src/main/jniLibs/x86_64/libsofuu.so
./gradlew :sofuu:assembleRelease
```

The `.so` files are QTSQ-free in a release build (pure MIT). A QTSQ-linked
build adds the encrypted-brain codec and falls under
`LICENSES/QTSQ-FORMAT.txt`.

## Manual (non-Gradle) integration

If you are not using the AAR — e.g. a host that already manages its own
`.so` layout — the original sample still works and is the reference for the
CMake wiring: `examples/headless/KotlinSample/` (`CMakeLists.txt`,
`jni_bridge.c`, `SofuuBridge.kt`).

## License

MIT for the runtime; a QTSQ-linked build additionally falls under
`LICENSES/QTSQ-FORMAT.txt`. See `docs/EMBEDDING-DIST.md`.
