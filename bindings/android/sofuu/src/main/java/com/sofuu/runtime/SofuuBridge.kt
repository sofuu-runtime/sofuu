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

    /// Flipped to 0 by close(); every native entry point treats 0 as "gone".
    @Volatile
    private var rtPtr: Long

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

    /// true once close() has run; every call then returns null/false.
    val isClosed: Boolean
        get() = rtPtr == 0L

    /// Free the runtime and release every host-tool registration this bridge
    /// made — the C slot, its name, and the JNI global ref that pins the
    /// Kotlin HostTool receiver (and through it your Activity).
    ///
    /// Engine thread, after you stop using this wrapper (and never while a
    /// streaming call is in flight). Idempotent; finalize() calls it as a
    /// backstop, but a finalizer runs at the GC's convenience — close()
    /// explicitly when your screen/worker goes away.
    fun close() {
        val ptr = synchronized(this) {
            val p = rtPtr
            rtPtr = 0L
            p
        }
        if (ptr != 0L) {
            nativeRtFree(ptr)
            // Only now: while the runtime lived, the ABI still held the
            // registration pointers we are about to free.
            nativeFreeHostTools(ptr)
        }
    }

    protected fun finalize() {
        close()
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

    // ── Streaming ────────────────────────────────────────────────────

    /// One event of a streaming call, as a JSON string.
    ///
    /// Runs on the ENGINE thread, inside the runtime's event loop: keep it
    /// fast and never block. Event shapes: token/step events carry
    /// `kind:"delta"` (ai.stream / ai.streamLocal) or the agent's own step
    /// payloads (agent.run); every stream ends with exactly one terminal
    /// `{"kind":"done",…}` — `deltas`/`aborted` for ai.stream, the result
    /// for agent.run, an `error` field when the call failed.
    ///
    /// Returning false stops the stream (the terminal `done` then reports
    /// `aborted:true`); throwing from here also stops it. Cancellation has
    /// to come from this callback — the engine thread is inside the call,
    /// and the ABI's cancel is same-thread by design. For a UI Stop button
    /// keep a `@Volatile` flag the UI thread sets and return `!stop` here.
    fun interface StreamListener {
        fun onStreamEvent(eventJson: String): Boolean
    }

    /// Run a streaming funnel call — `ai.stream`, `ai.streamLocal`,
    /// `agent.run`.
    ///
    /// BLOCKS until the stream finishes: the runtime is thread-affine (one
    /// runtime per thread) and the events ARE the call. Drive it from the
    /// engine thread — a worker, not the UI thread.
    ///
    /// `argsJson` is one JSON object, or a JSON array to spread as
    /// POSITIONAL arguments: `["/models/a.gguf", "hi", {}]` reaches
    /// `ai.streamLocal(modelPath, prompt, options)`. null means `{}`.
    ///
    /// @return true when the call ran — its outcome is the terminal `done`
    ///         event, not this boolean. false means it could not be
    ///         dispatched at all.
    /// @throws IllegalArgumentException when argsJson is not valid JSON.
    fun callStream(
        method: String,
        argsJson: String? = null,
        listener: StreamListener,
    ): Boolean {
        require(method.isNotBlank()) { "method must not be blank" }
        return nativeCallStream(rtPtr, method, validatedArgs(argsJson), listener)
    }

    /// Generate from a LOCAL model, streaming tokens as they decode.
    ///
    /// `modelPath` is a `.gguf` file or a model directory (SafeTensors /
    /// PyTorch / NumPy with its `tokenizer.json`). The path alone routes to
    /// in-process inference — no provider, no network, entirely offline.
    ///
    ///     sofuu.streamLocal("/models/qwen3-0.6b.gguf", "Once upon a time") { evt ->
    ///         if (evt.contains("\"kind\":\"delta\"")) show(evt)
    ///         !stopRequested
    ///     }
    fun streamLocal(
        modelPath: String,
        prompt: String,
        maxTokens: Int? = null,
        temperature: Float? = null,
        listener: StreamListener,
    ): Boolean {
        require(maxTokens == null || maxTokens > 0) { "maxTokens must be > 0" }
        require(temperature == null || temperature.isFinite()) { "temperature must be finite" }
        val opts = StringBuilder("{")
        if (maxTokens != null) opts.append("\"maxTokens\":").append(maxTokens)
        if (temperature != null) {
            if (opts.length > 1) opts.append(',')
            opts.append("\"temperature\":").append(temperature)
        }
        opts.append('}')
        val args = "[${org.json.JSONObject.quote(modelPath)}, " +
                   "${org.json.JSONObject.quote(prompt)}, $opts]"
        return callStream("ai.streamLocal", args, listener)
    }

    // ── Host tools ───────────────────────────────────────────────────

    /// A native capability handed to the agent as a tool: the camera, GPS,
    /// the share sheet, your app's own data. This is the whole reason
    /// `sofuu_rt_register_host_tool` exists — the QuickJS sandbox inside
    /// the runtime cannot reach the Android SDK, so anything
    /// platform-specific has to be answered by the host.
    ///
    /// [invoke] runs on the ENGINE thread, inside the agent's tool
    /// round-trip: keep it fast. Return a JSON value as a string for a
    /// synchronous answer (the library copies it — no lifetime to manage);
    /// return null when the work isn't done yet and settle it later with
    /// [resolveHostTool] from any thread. Either way the agent waits at
    /// most its tool timeout (30 s per tool by default).
    fun interface HostTool {
        fun invoke(argsJson: String, callId: Long): String?
    }

    /// Expose a native capability to the agent as a tool.
    ///
    /// `parametersJson` is the JSON Schema the model sees, so an accurate
    /// one is the difference between the model using the tool correctly and
    /// guessing. It must be a JSON **object** — the runtime rejects
    /// anything else as an argument error rather than surfacing it later as
    /// a confusing model failure.
    ///
    /// Engine thread only, and only while this wrapper is alive.
    /// Re-registering a name replaces the earlier registration.
    fun registerHostTool(
        name: String,
        description: String,
        parametersJson: String,
        tool: HostTool,
    ): Boolean = nativeRegisterHostTool(rtPtr, name, description, parametersJson, tool)

    /// Settle a call whose [HostTool.invoke] returned null.
    ///
    /// Safe from ANY thread — the completion is marshalled onto the runtime
    /// thread through its event loop. Pass a message in `error` to fail the
    /// call instead; `resultJson` must itself be valid JSON. Returns false
    /// when the call id is unknown: already settled, or abandoned because
    /// the agent already timed out waiting.
    fun resolveHostTool(callId: Long, resultJson: String? = null, error: String? = null): Boolean =
        nativeHostToolResolve(rtPtr, callId, resultJson, error)

    /// JSON text the ABI accepts (object or array), or "{}" for null/blank.
    private fun validatedArgs(argsJson: String?): String {
        if (argsJson.isNullOrBlank()) return "{}"
        return try {
            when (org.json.JSONTokener(argsJson).nextValue()) {
                is org.json.JSONObject, is org.json.JSONArray -> argsJson
                else -> throw IllegalArgumentException(
                    "argsJson must be a JSON object or array, got: $argsJson")
            }
        } catch (e: org.json.JSONException) {
            throw IllegalArgumentException("argsJson is not valid JSON: $argsJson", e)
        }
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
            // Load the SHIM, not the runtime: the JNI entry points below
            // are compiled into libsofuu_jni.so, and that library's
            // DT_NEEDED pulls in libsofuu.so next to it. Loading "sofuu"
            // alone finds the runtime but none of these natives, and the
            // first call dies with UnsatisfiedLinkError.
            System.loadLibrary("sofuu_jni")
        }

        @JvmStatic private external fun nativeAbiVersion(): Int
        @JvmStatic private external fun nativeRtNew(config: String?): Long
        @JvmStatic private external fun nativeRtFree(rt: Long)
        @JvmStatic private external fun nativeRtEval(rt: Long, source: String): String?
        @JvmStatic private external fun nativeRtCall(rt: Long, method: String, args: String?): String?
        @JvmStatic private external fun nativeCallStream(
            rt: Long, method: String, args: String?, listener: StreamListener,
        ): Boolean
        @JvmStatic private external fun nativeRegisterHostTool(
            rt: Long, name: String, description: String, paramsJson: String, listener: HostTool,
        ): Boolean
        @JvmStatic private external fun nativeHostToolResolve(
            rt: Long, callId: Long, resultJson: String?, error: String?,
        ): Boolean
        @JvmStatic private external fun nativeFreeHostTools(rt: Long): Int
        @JvmStatic private external fun nativeEmbedLocal(rt: Long, text: String, space: String?): FloatArray?
        @JvmStatic private external fun nativeEmbedBatch(rt: Long, texts: Array<String>, space: String?): Array<FloatArray>?
        @JvmStatic private external fun nativeEmbedImage(rt: Long, bytes: ByteArray): FloatArray?
        @JvmStatic private external fun nativeVoiceTranscribe(rt: Long, audioB64: String, optsJson: String?): String?
        @JvmStatic private external fun nativeVoiceSpeak(rt: Long, text: String, optsJson: String?): String?
        @JvmStatic private external fun nativeEmbedInfo(rt: Long): String?
    }
}
