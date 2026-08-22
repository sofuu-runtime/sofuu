// examples/headless/KotlinSample/SofuuBridge.kt
//
// A minimal Kotlin/JNI wrapper around the libsofuu C ABI (PLAN-HEADLESS H5).
// QuickJS is a pure interpreter (no JIT) → Android compatible.
//
// Usage in a Compose screen:
//
//   @Composable
//   fun SofuuScreen() {
//       var output by remember { mutableStateOf("Ready.") }
//       val sofuu = remember { SofuuBridge() }
//
//       Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
//           Text(output)
//           Button(onClick = { output = sofuu.eval("1 + 1") ?: "failed" }) { Text("Eval 1+1") }
//           Button(onClick = { output = sofuu.checkCaps() }) { Text("Check caps") }
//       }
//   }
//
// To use in Android Studio:
//   1. Copy libsofuu-android-aarch64.so into app/src/main/jniLibs/arm64-v8a/
//   2. Copy sofuu_embed.h into app/src/main/cpp/include/
//   3. Add to CMakeLists.txt:
//        add_library(sofuu SHARED IMPORTED)
//        set_target_properties(sofuu PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsofuu.so)
//        target_include_directories(native-lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
//        target_link_libraries(native-lib sofuu log)
//

package com.sofuu.runtime

/// A thin Kotlin wrapper around the Sofuu C ABI via JNI.
/// One SofuuRuntime per thread; never share across threads.
class SofuuBridge(configJson: String? = null) {

    private val rtPtr: Long

    init {
        rtPtr = if (configJson != null) {
            nativeRtNew(configJson)
        } else {
            nativeRtNew(null)
        }
        if (rtPtr == 0L) {
            throw RuntimeException("sofuu_rt_new failed")
        }
    }

    protected fun finalize() {
        if (rtPtr != 0L) {
            nativeRtFree(rtPtr)
        }
    }

    /// The ABI version this library implements.
    val abiVersion: Int
        get() = nativeAbiVersion()

    /// Evaluate arbitrary JS — the eval escape hatch.
    /// Returns the JSON-stringified result, or null on failure.
    fun eval(source: String): String? {
        return nativeRtEval(rtPtr, source)
    }

    /// Call a funnel method (e.g. "ai.complete", "ai.embed", "rlm.query").
    fun call(method: String, argsJson: String? = null): String? {
        return nativeRtCall(rtPtr, method, argsJson)
    }

    /// Check which features are available in this build.
    fun checkCaps(): String {
        val js = """
        JSON.stringify({
            has_sofuu: typeof sofuu !== 'undefined',
            has_ai: typeof sofuu.ai !== 'undefined',
            has_agent: typeof sofuu.agent !== 'undefined',
            has_rlm: typeof sofuu.rlm !== 'undefined',
            has_web: typeof sofuu.web !== 'undefined',
            has_memory: typeof sofuu.memory !== 'undefined',
            has_mcp: typeof sofuu.mcp !== 'undefined',
            has_http: typeof sofuu.http !== 'undefined',
            has_fs: typeof sofuu.fs !== 'undefined'
        })
        """.trimIndent()
        return eval(js) ?: "{}"
    }

    /// Embed text using the bundled offline embedder (no network).
    fun embedLocal(text: String): FloatArray? {
        val escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        val js = "JSON.stringify(Array.from(sofuu.ai.embedLocal('$escaped')))"
        val result = eval(js) ?: return null
        // Parse "[0.1, 0.2, ...]" into FloatArray.
        val trimmed = result.trim().removeSurrounding("[", "]")
        if (trimmed.isEmpty()) return null
        return trimmed.split(",").mapNotNull { it.trim().toFloatOrNull() }.toFloatArray()
    }

    /// Compute cosine similarity between two vectors.
    fun similarity(a: FloatArray, b: FloatArray): Float {
        val aJS = a.joinToString(",")
        val bJS = b.joinToString(",")
        val js = "sofuu.ai.similarity(new Float32Array([$aJS]), new Float32Array([$bJS]))"
        val result = eval(js) ?: return 0f
        return result.trim().toFloatOrNull() ?: 0f
    }

    // ── JNI native declarations ──────────────────────────────────────

    companion object {
        init {
            System.loadLibrary("sofuu")
        }

        @JvmStatic private external fun nativeAbiVersion(): Int
        @JvmStatic private external fun nativeRtNew(config: String?): Long
        @JvmStatic private external fun nativeRtFree(rt: Long)
        @JvmStatic private external fun nativeRtEval(rt: Long, source: String): String?
        @JvmStatic private external fun nativeRtCall(rt: Long, method: String, args: String?): String?
    }
}
