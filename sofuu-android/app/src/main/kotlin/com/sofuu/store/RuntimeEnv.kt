package com.sofuu.store

import android.content.Context

/**
 * The runtime's environment knobs, surfaced as settings.
 *
 * The engine reads these with plain `std::env::var` (see the env-var table
 * in docs/EMBEDDING.md and the call sites below), so the host applies them
 * with `setenv` before `sofuu_rt_new`. There is no ABI for them and no
 * reason for one: they are process configuration, not runtime state.
 *
 * **When they are read matters.** Most are consulted per turn — the turn
 * driver re-evaluates `chatToolDefs()`, the retry ladder resolves lazily on
 * the first transient error — so applying them at boot is both sufficient
 * and correct. `SOFUU_NO_ML` is the exception: it gates *module
 * registration* inside sofuu-core (`ml/mod.rs:441`), so it must be in the
 * environment before the runtime exists or the whole `sofuu.ml` namespace
 * is simply absent and the setting silently does nothing.
 */
class RuntimeEnv(private val context: Context) {

    private val prefs get() = context.getSharedPreferences("sofuu_env", Context.MODE_PRIVATE)

    // ── The knobs ────────────────────────────────────────────────

    /** Every 8 built-in coding tools off; web + MCP only. Read per turn
     *  at `chat.js:816`. */
    var noTools: Boolean
        get() = prefs.getBoolean(KEY_NO_TOOLS, false)
        set(v) = prefs.edit().putBoolean(KEY_NO_TOOLS, v).apply()

    /** `bash` stays in the tool list but every invocation refuses
     *  (`tools.js:540`). Note this does NOT stop MCP tools that shell
     *  out — it only closes the built-in. */
    var noShell: Boolean
        get() = prefs.getBoolean(KEY_NO_SHELL, false)
        set(v) = prefs.edit().putBoolean(KEY_NO_SHELL, v).apply()

    /** Drops the entire `sofuu.ml` context-economy gate layer.
     *  Compile-gated at registration — needs a restart to take effect. */
    var noMl: Boolean
        get() = prefs.getBoolean(KEY_NO_ML, false)
        set(v) = prefs.edit().putBoolean(KEY_NO_ML, v).apply()

    /** Memory backend: "" (auto), "hash", "semantic". Read per agent
     *  def/run (`agent.js:693`, `:865`). */
    var memoryBackend: String
        get() = prefs.getString(KEY_MEMORY_BACKEND, "") ?: ""
        set(v) = prefs.edit().putString(KEY_MEMORY_BACKEND, v).apply()

    /** Default web-search engine: "" (auto/duckduckgo), "brave",
     *  "tavily". Read per search (`web.js:369`). */
    var webEngine: String
        get() = prefs.getString(KEY_WEB_ENGINE, "") ?: ""
        set(v) = prefs.edit().putString(KEY_WEB_ENGINE, v).apply()

    /** Lets the agent reach private/loopback addresses through web.open.
     *  Off by default; the SSRF guard in `web.js:169` blocks them
     *  otherwise. */
    var webAllowInternal: Boolean
        get() = prefs.getBoolean(KEY_WEB_ALLOW_INTERNAL, false)
        set(v) = prefs.edit().putBoolean(KEY_WEB_ALLOW_INTERNAL, v).apply()

    /** Seconds a stalled transport waits before the turn gives up.
     *  0 keeps the runtime default (300). Read per stream (`ai.rs:1611`). */
    var stallTimeoutSecs: Int
        get() = prefs.getInt(KEY_STALL_TIMEOUT, 0)
        set(v) = prefs.edit().putInt(KEY_STALL_TIMEOUT, v.coerceIn(0, 3600)).apply()

    /** Tool-loop budget for one chat turn. 0 keeps the runtime default
     *  (200). Read per turn (`chat.js:1456`). */
    var chatMaxSteps: Int
        get() = prefs.getInt(KEY_MAX_STEPS, 0)
        set(v) = prefs.edit().putInt(KEY_MAX_STEPS, v.coerceIn(0, 2000)).apply()

    /** Inference thread count for the CPU kernel pool. 0 = auto (the pool
     *  picks 4 on mobile big.LITTLE). Read once, at pool creation — takes
     *  effect after an engine restart. The pool also refuses counts above
     *  the online core count. Also the DECODE fallback when no
     *  decode-specific override is present. */
    var numThreads: Int
        get() = prefs.getInt(KEY_NUM_THREADS, 0)
        set(v) = prefs.edit().putInt(KEY_NUM_THREADS, v.coerceIn(0, 8)).apply()

    /** Prefill-phase thread count (batched GEMM over the prompt —
     *  compute-bound, wants the whole performance cluster). 0 = auto
     *  (min(online, 8)). Read per scheduling iteration; takes effect
     *  after an engine restart. */
    var prefillThreads: Int
        get() = prefs.getInt(KEY_PREFILL_THREADS, 0)
        set(v) = prefs.edit().putInt(KEY_PREFILL_THREADS, v.coerceIn(0, 8)).apply()

    /** Decode-phase thread count override. 0 = the pool default
     *  (`numThreads`, else 4). Read per scheduling iteration. */
    var decodeThreads: Int
        get() = prefs.getInt(KEY_DECODE_THREADS, 0)
        set(v) = prefs.edit().putInt(KEY_DECODE_THREADS, v.coerceIn(0, 8)).apply()

    /** Per-phase decode instrumentation: appends a window line (forward /
     *  sample / glue per token + prefill aggregates) to
     *  files/phase-metrics.log every 256 tokens. Diagnostic; near-zero
     *  cost when off. */
    var phaseMetrics: Boolean
        get() = prefs.getBoolean(KEY_PHASE_METRICS, false)
        set(v) = prefs.edit().putBoolean(KEY_PHASE_METRICS, v).apply()

    /** Set before `sofuu_rt_new`. Returns what it actually applied, so
     *  the UI can show the effective values rather than the requested
     *  ones. */
    fun apply(): Map<String, String> {
        val applied = LinkedHashMap<String, String>()
        fun put(key: String, value: String?) {
            if (value.isNullOrEmpty()) {
                com.sofuu.engine.SofuuBridge.setEnv(key, null)
                applied[key] = ""
            } else {
                com.sofuu.engine.SofuuBridge.setEnv(key, value)
                applied[key] = value
            }
        }
        put(KEY_NO_TOOLS, if (noTools) "1" else null)
        put(KEY_NO_SHELL, if (noShell) "1" else null)
        // Registration-gated: without this the ml namespace is absent and
        // every gate degrades to a silent no-op.
        put(KEY_NO_ML, if (noMl) "1" else null)
        put(KEY_MEMORY_BACKEND, memoryBackend)
        put(KEY_WEB_ENGINE, webEngine)
        put(KEY_WEB_ALLOW_INTERNAL, if (webAllowInternal) "1" else null)
        put(KEY_STALL_TIMEOUT, if (stallTimeoutSecs > 0) stallTimeoutSecs.toString() else null)
        put(KEY_MAX_STEPS, if (chatMaxSteps > 0) chatMaxSteps.toString() else null)
        // Pool-creation-time read for the base count; the phase knobs are
        // read per scheduling iteration but are only meaningful with the
        // rebuilt pool — the engine restart keeps all three consistent.
        put(KEY_NUM_THREADS, if (numThreads > 0) numThreads.toString() else null)
        put(KEY_PREFILL_THREADS, if (prefillThreads > 0) prefillThreads.toString() else null)
        put(KEY_DECODE_THREADS, if (decodeThreads > 0) decodeThreads.toString() else null)
        put(
            KEY_PHASE_METRICS,
            if (phaseMetrics) context.getFileStreamPath("phase-metrics.log").absolutePath else null,
        )
        // The engine's static curl has no compiled-in trust store on
        // Android (see CaTrust), so every https:// call — provider model
        // lists, chat completions, web tools — needs this one.
        val caBundle = CaTrust.ensureBundle(context)
        put(KEY_CA_BUNDLE, caBundle)
        if (caBundle == null) {
            // Last resort: OpenSSL may still resolve the hash directory on
            // builds that support the legacy names. Better than no path.
            put(KEY_CA_PATH, "/system/etc/security/cacerts")
        } else {
            put(KEY_CA_PATH, null)
        }
        return applied
    }

    companion object {
        private const val KEY_NO_TOOLS = "SOFUU_NO_TOOLS"
        private const val KEY_NO_SHELL = "SOFUU_NO_SHELL"
        private const val KEY_NO_ML = "SOFUU_NO_ML"
        private const val KEY_MEMORY_BACKEND = "SOFUU_MEMORY_BACKEND"
        private const val KEY_WEB_ENGINE = "SOFUU_WEB_ENGINE"
        private const val KEY_WEB_ALLOW_INTERNAL = "SOFUU_WEB_ALLOW_INTERNAL"
        private const val KEY_STALL_TIMEOUT = "SOFUU_STALL_TIMEOUT_SECS"
        private const val KEY_MAX_STEPS = "SOFUU_CHAT_MAX_STEPS"
    private const val KEY_NUM_THREADS = "SOFUU_NUM_THREADS"
    private const val KEY_PREFILL_THREADS = "SOFUU_PREFILL_THREADS"
    private const val KEY_DECODE_THREADS = "SOFUU_DECODE_THREADS"
    private const val KEY_PHASE_METRICS = "SOFUU_PHASE_METRICS"
    const val KEY_CA_BUNDLE = "SOFUU_CA_BUNDLE"
    private const val KEY_CA_PATH = "SOFUU_CA_PATH"
    }
}