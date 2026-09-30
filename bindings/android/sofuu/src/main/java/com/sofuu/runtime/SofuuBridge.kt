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

    /// Embed text using the bundled offline embedder (no network, no JSON).
    /// @param space "sem2-64" (default), "sem1-64", or "hash-768".
    fun embedLocal(text: String, space: String? = null): FloatArray? {
        return nativeEmbedLocal(rtPtr, text, space)
    }

    /// Embed many texts in one call. Returns one FloatArray per input.
    fun embedBatch(texts: Array<String>, space: String? = null): Array<FloatArray>? {
        if (texts.isEmpty()) return arrayOf()
        return nativeEmbedBatch(rtPtr, texts, space)
    }

    /// Embed image bytes (PNG/JPEG) into img1-64 (joint with sem2-64 text).
    fun embedImage(bytes: ByteArray): FloatArray? {
        if (bytes.isEmpty()) return null
        return nativeEmbedImage(rtPtr, bytes)
    }

    /// Provider transcription via the libsofuu funnel (BYO key in config).
    /// For offline use SofuuVoice (OS speech) instead. Returns transcript text.
    fun transcribeProvider(audio: ByteArray, optsJson: String? = null): String? {
        val b64 = android.util.Base64.encodeToString(audio, android.util.Base64.NO_WRAP)
        val env = nativeVoiceTranscribe(rtPtr, b64, optsJson) ?: return null
        return try {
            org.json.JSONObject(env).getJSONObject("result").getString("text")
        } catch (e: Exception) { null }
    }

    /// Provider speech via the libsofuu funnel. Returns (bytes, format).
    fun speakProvider(text: String, optsJson: String? = null): Pair<ByteArray, String>? {
        val env = nativeVoiceSpeak(rtPtr, text, optsJson) ?: return null
        return try {
            val result = org.json.JSONObject(env).getJSONObject("result")
            val indexed = result.getJSONObject("audio")
            val keys = indexed.keys().asSequence().mapNotNull { it.toIntOrNull() }.sorted().toList()
            if (keys.isEmpty()) return null
            val bytes = ByteArray(keys.size) { i -> indexed.getInt(keys[i].toString()).toByte() }
            Pair(bytes, result.getString("format"))
        } catch (e: Exception) { null }
    }

    /// Space manifest JSON (default space, model ids, dims, artifact hashes).
    fun embedInfo(): String? {
        return nativeEmbedInfo(rtPtr)
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
        @JvmStatic private external fun nativeEmbedLocal(rt: Long, text: String, space: String?): FloatArray?
        @JvmStatic private external fun nativeEmbedBatch(rt: Long, texts: Array<String>, space: String?): Array<FloatArray>?
        @JvmStatic private external fun nativeEmbedImage(rt: Long, bytes: ByteArray): FloatArray?
        @JvmStatic private external fun nativeVoiceTranscribe(rt: Long, audioB64: String, optsJson: String?): String?
        @JvmStatic private external fun nativeVoiceSpeak(rt: Long, text: String, optsJson: String?): String?
        @JvmStatic private external fun nativeEmbedInfo(rt: Long): String?
    }
}
