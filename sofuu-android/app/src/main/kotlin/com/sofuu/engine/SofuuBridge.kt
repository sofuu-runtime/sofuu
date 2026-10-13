package com.sofuu.engine

/**
 * Kotlin/JNI wrapper around the libsofuu C ABI (v2).
 *
 * One SofuuRuntime per thread; create and use it ONLY on the thread that
 * will drive it (the engine worker). The app funnels every call through
 * [SofuuEngine], which owns that thread — this class is intentionally
 * thin: raw calls, no state.
 *
 * Ported from bindings/android/sofuu/…/SofuuBridge.kt (the public AAR
 * wrapper), extended for the chat app's needs: HOME pinning, streaming
 * with a host listener, and cancel.
 */
class SofuuBridge private constructor(private val rtPtr: Long) {

    /** The ABI version this library implements. */
    val abiVersion: Int
        get() = nativeAbiVersion()

    /** Evaluate arbitrary JS — the eval escape hatch. Returns the
     *  JSON-stringified completion value, or null on failure. */
    fun eval(source: String): String? = nativeRtEval(rtPtr, source)

    /** Call a funnel method (e.g. "ai.complete", "agent.run").
     *  Returns the JSON envelope, or null on failure. */
    fun call(method: String, argsJson: String? = null): String? =
        nativeRtCall(rtPtr, method, argsJson)

    /**
     * Run a streamable funnel call (agent.run / ai.stream).
     *
     * The ABI hands events to a single process-global listener rather than
     * a per-call one, so [listener] is installed here rather than ignored.
     * The C side used to drop it on the floor, which made the promise below
     * untrue and the call silently returned 0 (no listener installed).
     *
     * Returns the cancel id; 0 means the dispatch failed.
     */
    fun callStream(method: String, argsJson: String?, listener: StreamListener): Long {
        nativeSetStreamListener(listener)
        return nativeRtCallStream(rtPtr, method, argsJson)
    }

    /** Cancel a streaming call by its id. */
    fun cancel(cancelId: Long): Boolean = nativeRtCancel(rtPtr, cancelId)

    /** Space manifest JSON (default space, model ids, dims, hashes). */
    fun embedInfo(): String? = nativeEmbedInfo()

    /** Transcribe base64 audio via the configured provider. Returns the
     *  JSON envelope (read `ok` / `result` / `error`). Engine thread only. */
    fun voiceTranscribe(audioBase64: String, optsJson: String? = null): String? =
        nativeVoiceTranscribe(rtPtr, audioBase64, optsJson)

/** Synthesize speech. Resolves {audio, format}; over JSON transport the
         *  audio bytes travel as an indexed-byte object the host reassembles.
         *  Engine thread only. */
    fun voiceSpeak(text: String, optsJson: String? = null): String? =
            nativeVoiceSpeak(rtPtr, text, optsJson)

    // ── Host tools ───────────────────────────────────────────────

    /**
     * Expose a native capability to the agent as a tool.
     *
     * [parametersJson] is the JSON Schema the model sees, so an accurate
     * one is the difference between the model using the tool correctly and
     * guessing. It must be a JSON **object** — the runtime rejects
     * anything else as an argument error rather than surfacing it later as
     * a confusing model failure.
     *
     * Engine thread only, and only while this bridge is alive. Re-using a
     * name replaces the earlier registration.
     */
    fun registerHostTool(
        name: String,
        description: String,
        parametersJson: String,
        tool: HostTool,
    ): Boolean = nativeRegisterHostTool(rtPtr, name, description, parametersJson, tool)

    /**
     * Settle a call whose [HostTool.invoke] returned null.
     *
     * Safe from ANY thread — the completion is marshalled onto the runtime
     * thread through its event loop. Pass a message in [error] to fail the
     * call instead; [resultJson] must itself be valid JSON.
     */
    fun resolveHostTool(callId: Long, resultJson: String? = null, error: String? = null): Boolean =
            nativeHostToolResolve(rtPtr, callId, resultJson, error)

    /** Called once from [close]; never call native methods after. */
    fun close() = nativeRtFree(rtPtr)

    /** Host for streaming events (runs on the engine thread). */
    interface StreamListener {
        fun onStreamEvent(eventJson: String)
    }

    /**
     * A native capability handed to the agent as a tool.
     *
     * This is the whole reason `sofuu_rt_register_host_tool` exists: the
     * QuickJS sandbox inside the runtime cannot reach the Android SDK, so
     * anything platform-specific — the share sheet, a vibration, the
     * device's own specs — has to be answered by the host.
     *
     * Runs on the ENGINE thread, inside the agent's tool round-trip, so
     * keep it fast and never block on it. Anything that needs a result
     * later returns null and calls [resolveHostTool] instead.
     */
    interface HostTool {
        /** @return a JSON value as a string, or null to settle asynchronously. */
        fun invoke(argsJson: String, callId: Long): String?
    }

    companion object {
        init {
            System.loadLibrary("sofuu_jni")
        }

        /** The ABI version this library implements (no runtime needed). */
        fun abiVersion(): Int = nativeAbiVersion()

        /** Pin $HOME for the process. MUST run before [create] — the
         *  runtime derives config/brain/session paths from it. */
        fun setHome(path: String) = nativeSetHome(path)

        /** Set the process working directory (tool-jail default + the
         *  cwd spawned MCP children inherit). Call before [create]. */
        fun chdir(path: String) = nativeChdir(path)

        /** Create the runtime on the calling thread. `configJson` is the
         *  sofuu_rt_new config (NULL-safe); the app passes the app files
         *  dir as config_root so ~/.sofuu lands in private storage. */
        fun create(configJson: String? = null): SofuuBridge {
            val ptr = nativeRtNew(configJson)
            require(ptr != 0L) { "sofuu_rt_new failed" }
            return SofuuBridge(ptr)
        }

        /** Route the engine's console output to logcat (tag `sofuu-js`).
         * Process-global per the ABI — call once, before any runtime is
         * created, so boot warnings are captured too. Pass false to send
         * output back to stdio. */
        fun setLogEnabled(enabled: Boolean) = nativeSetLogEnabled(enabled)

        fun setStreamListener(listener: StreamListener?) =
            nativeSetStreamListener(listener)

        // ── Vector embeddings ──────────────────────────────────────
        // Stateless pure functions in the ABI (sofuu_embed.h): these pass
        // a NULL rt, so they need no runtime and no thread affinity. The
        // default space is hash-768 — the same space the brain stores in,
        // so an embedded string can be remembered with no space argument.
        // Returns null when the space is unknown or the artifact is
        // unavailable, rather than a fake zero vector.

        /** Text → vector. `space` null/empty selects the default. */
        fun embedLocal(text: String, space: String? = null): FloatArray? =
            nativeEmbedLocal(text, space)

        /** Several texts at once → array of vectors. */
        fun embedBatch(texts: Array<String>, space: String? = null): Array<FloatArray>? =
            nativeEmbedBatch(texts, space)

        /** PNG/JPEG bytes → img1-64 vector (joint with sem2-64 geometry). */
        fun embedImage(bytes: ByteArray): FloatArray? = nativeEmbedImage(bytes)

        /** Space manifest: default space, per-space model ids/dims/hashes. */
        fun embedInfo(): String? = nativeEmbedInfo()

        @JvmStatic private external fun nativeSetHome(path: String)
        @JvmStatic private external fun nativeChdir(path: String)

        /**
         * Set (or, with a null value, unset) a process environment
         * variable.
         *
         * This is how the engine's own knobs get applied. The library reads
         * them with `std::env::var` — `SOFUU_NO_TOOLS`, `SOFUU_NO_SHELL`,
         * `SOFUU_MEMORY_BACKEND`, `SOFUU_WEB_ENGINE` and friends — so the
         * host has to put them in the environment before `sofuu_rt_new`.
         * Kotlin has no libc binding, hence the JNI hop.
         */
        @JvmStatic fun setEnv(name: String, value: String?) {
            nativeSetEnv(name, value)
        }

        @JvmStatic private external fun nativeSetEnv(name: String, value: String?)

        @JvmStatic private external fun nativeAbiVersion(): Int
        @JvmStatic private external fun nativeRtNew(config: String?): Long
        @JvmStatic private external fun nativeRtFree(rt: Long)
        @JvmStatic private external fun nativeRtEval(rt: Long, source: String): String?
        @JvmStatic private external fun nativeRtCall(rt: Long, method: String, args: String?): String?
        @JvmStatic private external fun nativeRtCallStream(rt: Long, method: String, args: String?): Long
        @JvmStatic private external fun nativeRtCancel(rt: Long, cancelId: Long): Boolean
        @JvmStatic private external fun nativeSetStreamListener(listener: StreamListener?)
        @JvmStatic private external fun nativeSetLogEnabled(enabled: Boolean)
        @JvmStatic private external fun nativeEmbedInfo(): String?
        @JvmStatic private external fun nativeEmbedLocal(text: String, space: String?): FloatArray?
        @JvmStatic private external fun nativeEmbedBatch(texts: Array<String>, space: String?): Array<FloatArray>?
        @JvmStatic private external fun nativeEmbedImage(bytes: ByteArray): FloatArray?
        @JvmStatic private external fun nativeVoiceTranscribe(rt: Long, audioBase64: String, opts: String?): String?
        @JvmStatic private external fun nativeVoiceSpeak(rt: Long, text: String, opts: String?): String?

        @JvmStatic private external fun nativeRegisterHostTool(
            rt: Long, name: String, description: String, paramsJson: String, listener: HostTool,
        ): Boolean

        @JvmStatic private external fun nativeHostToolResolve(
            rt: Long, callId: Long, resultJson: String?, error: String?,
        ): Boolean

        /**
         * Release every host-tool registration made in this process —
         * the C slot, the strdup'd name, and the JNI global ref pinning
         * the Kotlin receiver. Returns how many were freed.
         *
         * Must be called AFTER the runtime that used them has been freed.
         * Until then the ABI still holds its own pointers, so releasing
         * early would be a use-after-free.
         */
        @JvmStatic fun freeHostTools(): Int = nativeFreeHostTools()

        @JvmStatic private external fun nativeFreeHostTools(): Int
    }
}
