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
