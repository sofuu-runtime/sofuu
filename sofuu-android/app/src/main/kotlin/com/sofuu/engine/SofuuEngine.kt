package com.sofuu.engine

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.File
import java.util.concurrent.CompletableFuture
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The in-process Sofuu runtime, driven from a dedicated worker thread.
 *
 * Architecture mirrors the desktop backend (sofuu-desktop/src-tauri/src/
 * engine.rs + host.rs), minus the WebView: the engine owns one thread,
 * the UI never touches the runtime, and everything crosses the boundary
 * as JSON.
 *
 * Turn driving is fire-and-forget + a JS event queue:
 *   1. `sofuu.chat.submit(text, {onEvent: globalThis.__sofuu_push})`
 *      starts the turn as a promise; the eval returns immediately.
 *   2. A pump drains `globalThis.__evq` every tick (≤50 ms), delivering
 *      events to [events] on the main thread. Each drain eval also runs
 *      the libuv loop, so the turn progresses between polls.
 *   3. Cancel + approvals ride the same engine thread (the thread is
 *      never blocked mid-turn, unlike the desktop's blocking eval), so
 *      they answer instantly through chat.js's `__host_poke`.
 *
 * Engine reads (sessions, state, model lists, …) come back through a
 * second queue, `globalThis.__rq`, resolved by the pump into the waiting
 * future. Replies are strictly sequential — one query in flight at a
 * time, matched by arrival order.
 */
class SofuuEngine(
    val homeDir: String,
    val projectDir: String,
    /** Needed to build the native host tools ([DeviceTools]); they need a
     *  Context for the vibrator, battery and share sheet. */
    private val appContext: android.content.Context,
) {
    /** Streaming/agent events + lifecycle events for the UI. */
    private val _events = MutableSharedFlow<EngineEvent>(replay = 0, extraBufferCapacity = 512)
    val events: SharedFlow<EngineEvent> = _events

    /**
     * Native tools registered on the engine thread at boot — the names the
     * agent can now call that JS alone could not. Surfaced so Settings can
     * show what the phone adds over a terminal.
     */
    val hostTools: MutableList<String> = java.util.Collections.synchronizedList(mutableListOf<String>())

    private val running = AtomicBoolean(false)
    private val ready = AtomicBoolean(false)
    private var thread: Thread? = null
    private var bridge: SofuuBridge? = null

    /** Pending engine replies, matched by arrival order. */
    private val pendingReplies = ArrayDeque<CompletableFuture<Any?>>()
    private val replyLock = Any()

    /** Commands the engine thread executes in order. */
    private val mailbox = LinkedBlockingQueue<Runnable>()

    val isReady: Boolean get() = ready.get()

    /** The C ABI version of the linked libsofuu. */
    val abiVersion: Int get() = SofuuBridge.abiVersion()

    // ── Boot ───────────────────────────────────────────────────────

    fun start() {
        if (running.getAndSet(true)) return
        thread = Thread({ engineMain() }, "sofuu-engine").also { it.start() }
    }

    fun shutdown() {
        if (!running.getAndSet(false)) return
        thread?.join(4000)
        thread = null
    }

    private fun log(msg: String) = android.util.Log.i("sofuu-engine", msg)

    private fun engineMain() {
        try {
            File(projectDir).mkdirs()
            // Engine console output → logcat (tag `sofuu-js`). Installed
            // BEFORE rt_new so boot warnings are not lost.
            SofuuBridge.setLogEnabled(true)
            log("boot: home=$homeDir project=$projectDir abi=${SofuuBridge.abiVersion()}")
            // Android has no $HOME — pin it before the runtime boots (the
            // engine derives config/brain/session paths from it).
            SofuuBridge.setHome(homeDir)
            SofuuBridge.chdir(projectDir)
            val cfg = JSONObject()
                .put("embedded", true)
                .put("config_root", File(homeDir, ".sofuu").absolutePath)
                .put("provider", "")
                .put("model", "")
            log("boot: rt_new")
            val rt = SofuuBridge.create(cfg.toString())
            bridge = rt
            log("boot: rt created")

            // Bootstrap (event + reply queues) and chat.init must land
            // before the pump starts, so no event is lost and the very
            // first drain sees an initialized engine.
            mailbox.put(Runnable {
                log("boot: bootstrap…")
                val boot = rt.eval(BOOTSTRAP_JS)
                log("boot: bootstrap=$boot")
                // Host tools BEFORE chat.init: the runtime publishes each
                // one into both __sofuu_host_tools (resolved when a turn
                // runs) and sofuu.tools.TOOLS (resolved when an agent is
                // defined), and registering after init would leave the
                // first turn's agent without them.
                val registered = try {
                    DeviceTools(appContext).registerAll(rt)
                } catch (e: Throwable) {
                    log("host tools failed: ${e.message ?: e.toString()}")
                    emptyList<String>()
                }
                hostTools.clear()
                hostTools.addAll(registered)
                log("boot: host tools = $registered")

                val init = JSONObject()
                    .put("host", "android")
                    .put("project", projectDir)
                    .put("permissions", "prompt")
                    .put("toolTimeoutMs", 600_000L) // approvals wait for a human
                    .put("pastTurns", 40)           // desktop resume depth
                val r = rt.eval("sofuu.chat.init(${jsonString(init.toString())})")
                log("boot: chat.init=$r")
                if (boot == null || r == null) {
                    emit(EngineEvent.BootError("engine boot failed (chat.js unavailable)"))
                } else {
                    ready.set(true)
                    emit(EngineEvent.EngineReady)
                }
            })

            // Pump loop: commands first, then a drain of both JS queues.
            while (running.get()) {
                val cmd = mailbox.poll(50, TimeUnit.MILLISECONDS)
                if (!running.get()) break
                try {
                    cmd?.run()
                } catch (e: Throwable) {
                    emit(EngineEvent.BootError(e.message ?: e.toString()))
                }
                drainEvents(rt)
            }
            log("pump exit")
            // Release the runtime HERE, while this is still the thread that
            // created it (docs/EMBEDDING.md: a runtime belongs to exactly
            // one thread). shutdown() used to only join the worker, so the
            // whole QuickJS + libuv engine leaked on every process death.
            try {
                bridge?.close()
                log("runtime freed")
            } catch (e: Throwable) {
                log("sofuu_rt_free failed: ${e.message ?: e.toString()}")
            }
            bridge = null
            ready.set(false)
            // Anything still queued will never be evaluated now, so release
            // its callers with null instead of leaving a future that hangs
            // whoever is blocking on it.
            synchronized(replyLock) {
                val stranded = pendingReplies.toList()
                pendingReplies.clear()
                stranded.forEach { if (!it.isDone) it.complete(null) }
            }
        } catch (e: Throwable) {
            emit(EngineEvent.BootError(e.message ?: e.toString()))
            ready.set(false)
        }
    }

    private fun onEngine(block: (SofuuBridge) -> Unit) {
        mailbox.put(Runnable {
            val rt = bridge ?: return@Runnable
            try {
                block(rt)
            } catch (e: Throwable) {
                emit(EngineEvent.BootError(e.message ?: e.toString()))
            }
        })
    }

    // ── Event pump ─────────────────────────────────────────────────

    /** Drain both JS queues; deliver events to the UI and complete
     *  pending reply futures. Runs on the engine thread. */
    private fun drainEvents(rt: SofuuBridge) {
        val raw = evalResult(rt, DRAIN_JS) ?: return
        val obj = try {
            JSONObject(raw)
        } catch (e: Throwable) {
            return
        }
        val rp = obj.optJSONArray("rp")
        if (rp != null) {
            for (i in 0 until rp.length()) {
                val d = synchronized(replyLock) {
                    if (pendingReplies.isNotEmpty()) pendingReplies.removeFirst() else null
                }
                if (d != null && !d.isDone) {
                    d.complete(parseReply(rp.optString(i)))
                }
            }
        }
        val ev = obj.optJSONArray("ev") ?: return
        for (i in 0 until ev.length()) {
            val e = ev.optJSONObject(i) ?: continue
            parseEvent(e)?.let { emit(it) }
        }
    }

    private fun emit(ev: EngineEvent) {
        _events.tryEmit(ev)
    }

    private fun parseEvent(e: JSONObject): EngineEvent? {
        val kind = e.optString("kind")
        val p = e.optJSONObject("payload") ?: JSONObject()
        return when (kind) {
            "ctx" -> EngineEvent.Ctx(
                used = p.optLong("used").coerceAtLeast(0),
                window = p.optLong("window").coerceAtLeast(0),
                source = p.optString("source"),
                model = p.optString("model").ifEmpty { null },
                sid = p.optString("sid").ifEmpty { null },
            )
            "start" -> EngineEvent.TurnStarted(sid = p.optString("sessionId").ifEmpty { null })
            "done" -> EngineEvent.TurnFinished(
                error = p.optJSONObject("error")?.optString("message"),
                usage = parseUsage(p),
            )
            "approval_request" -> EngineEvent.ApprovalRequested(
                id = p.optString("id"),
                tool = p.optString("tool").ifEmpty { "tool" },
                args = p.optJSONObject("args") ?: JSONObject(),
                risky = p.opt("risky") != false,
            )
            "approval_resolved" -> EngineEvent.ApprovalResolved(id = p.optString("id"))
            "warn" -> EngineEvent.Warn(message = p.optString("message"))
            else -> if (kind.isEmpty()) null else EngineEvent.Step(kind, e)
        }
    }

    /** A reply is JSON-encoded once by `__reply_put`. When the queried
     *  expression already produced a JSON STRING (every chat.js read is
     *  `JSON.stringify(...)`), the wire value is that string encoded
     *  again — so unwrap a String result one more time. */
    private fun parseReply(text: String): Any? {
        val first = runCatching { JSONTokener(text).nextValue() }.getOrNull()
        if (first is String) {
            return runCatching { JSONTokener(first).nextValue() }.getOrDefault(first)
        }
        return first
    }

    private fun parseUsage(p: JSONObject): Usage? {
        val u = p.optJSONObject("usage") ?: p
        val servedBy = u.optString("servedBy").ifEmpty { null }
        val prompt = u.optLong("promptTokens", -1)
        val completion = u.optLong("completionTokens", -1)
        val cost = u.optDouble("costUsd", -1.0)
        val steps = u.optInt("steps", -1)
        if (prompt < 0 && completion < 0 && cost < 0 && servedBy == null && steps < 0) return null
        return Usage(
            promptTokens = if (prompt >= 0) prompt else null,
            completionTokens = if (completion >= 0) completion else null,
            costUsd = if (cost >= 0) cost else null,
            steps = if (steps >= 0) steps else null,
            servedBy = servedBy,
        )
    }

    // ── Public surface (safe from any thread) ──────────────────────

    fun submitTurn(text: String, noTools: Boolean, images: List<String> = emptyList()) {
        onEngine { rt ->
            val textJs = jsonString(text)
            val noToolsJs = if (noTools) "true" else "false"
            val imagesJs = JSONArray(images).toString()
            rt.eval(
                "(function(){ try {" +
                    "sofuu.chat.submit($textJs, {onEvent: globalThis.__sofuu_push, noTools: $noToolsJs, images: $imagesJs})" +
                    ".then(function(){}, function(e){ globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); });" +
                    "return true;" +
                    "} catch(e) { globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); return false; } })()"
            )
        }
    }

    fun cancelTurn() {
        onEngine { rt -> rt.eval("__host_poke('{\"type\":\"cancel\"}')") }
    }

    fun resolveApproval(id: String, allow: Boolean, always: Boolean) {
        onEngine { rt ->
            val payload = JSONObject()
                .put("type", "approval")
                .put("id", id)
                .put("allow", allow)
                .put("always", always)
            rt.eval("__host_poke(${jsonString(payload.toString())})")
        }
    }

    fun compactSession(): CompletableFuture<JSONObject?> =
        query("Promise.resolve(sofuu.chat.compact())").thenCast()

    fun sessions(): CompletableFuture<JSONArray?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.sessions) ? sofuu.chat.sessions() : [])").thenArray()

    fun sessionTurns(id: String): CompletableFuture<JSONArray?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.sessionTurns) ? sofuu.chat.sessionTurns(${jsonString(id)}) : [])").thenArray()

    fun newSession(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.newSession) ? sofuu.chat.newSession() : {error:'chat engine unavailable'})").thenObject()

    fun resumeSession(id: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.resume) ? sofuu.chat.resume(${jsonString(id)}) : {error:'chat engine unavailable'})").thenObject()

    fun setProject(path: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.setProject) ? sofuu.chat.setProject(${jsonString(path)}) : {error:'chat engine unavailable'})").thenObject()

    fun setPermissions(profile: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.setPermissions) ? sofuu.chat.setPermissions(${jsonString(profile)}) : {error:'chat engine unavailable'})").thenObject()

    fun chatState(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {error:'chat engine unavailable'})").thenObject()

    fun deleteSession(id: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.deleteSession) ? sofuu.chat.deleteSession(${jsonString(id)}) : {ok:false, error:'chat engine unavailable'})").thenObject()

    fun clearSessions(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.clearSessions) ? sofuu.chat.clearSessions() : {ok:false, error:'chat engine unavailable'})").thenObject()

    fun usageReport(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.usage) ? sofuu.chat.usage() : {error:'chat engine unavailable'})").thenObject()

    fun brainRemember(fact: String): CompletableFuture<JSONObject?> =        query("JSON.stringify((function(){ try { if (sofuu.memory && sofuu.memory.remember) { sofuu.memory.remember(${jsonString(fact)}); return {ok:true}; } return {ok:false, error:'brain unavailable (enable /brain)'}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    fun brainWhy(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((function(){ try { var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; return {ok:true, lastRecall:(st && st.lastRecall) || null, recall:(st && st.lastRecallHits) || null}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    // ── Embeddings (stateless, NULL rt) ───────────────────────────

    /** Run a stateless ABI call off the UI thread and hand back a future.
     *  Embedders need no runtime, but they go through the same mailbox so
     *  they never race the engine thread for CPU during a turn. */
    private fun <T> onEngineValue(block: () -> T): CompletableFuture<T> {
        val f = CompletableFuture<T>()
        mailbox.put(Runnable {
            try {
                f.complete(block())
            } catch (e: Throwable) {
                if (!f.isDone) f.completeExceptionally(e)
            }
        })
        return f
    }

    fun embedLocal(text: String, space: String? = null): CompletableFuture<FloatArray?> =
        onEngineValue { SofuuBridge.embedLocal(text, space) }

    fun embedBatch(texts: Array<String>, space: String? = null): CompletableFuture<Array<FloatArray>?> =
        onEngineValue { SofuuBridge.embedBatch(texts, space) }

    fun embedImage(bytes: ByteArray): CompletableFuture<FloatArray?> =
        onEngineValue { SofuuBridge.embedImage(bytes) }

    fun embedInfo(): CompletableFuture<JSONObject?> =
        onEngineValue { SofuuBridge.embedInfo()?.let { try { JSONObject(it) } catch (e: Throwable) { null } } }

    // ── Voice (provider-backed, engine thread) ─────────────────────

    /** Base64 audio → transcript. Provider-agnostic: the active provider
     *  does the work, and an unsupported one comes back as an error
     *  envelope rather than a silent no-op. */
    fun voiceTranscribe(audioBase64: String, opts: String? = null): CompletableFuture<JSONObject?> =
        onEngineVoice { it.voiceTranscribe(audioBase64, opts) }

    fun voiceSpeak(text: String, opts: String? = null): CompletableFuture<JSONObject?> =
        onEngineVoice { it.voiceSpeak(text, opts) }

    private fun onEngineVoice(block: (SofuuBridge) -> String?): CompletableFuture<JSONObject?> {
        val f = CompletableFuture<JSONObject?>()
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                if (!f.isDone) f.complete(null)
                return@Runnable
            }
            try {
                val raw = block(rt)
                f.complete(if (raw == null) null else runCatching { JSONObject(raw) }.getOrNull())
            } catch (e: Throwable) {
                if (!f.isDone) f.completeExceptionally(e)
            }
        })
        return f
    }

    fun mlInfo(): CompletableFuture<JSONObject?> =        query("JSON.stringify((function(){ try { var info = (sofuu.ml && sofuu.ml.info) ? sofuu.ml.info() : null; return {ok:true, info:info}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    fun contextDump(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((function(){ try { var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; return {ok:true, state:st}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    fun listTools(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((function(){ try { var defs = []; if (sofuu.tools && sofuu.tools.TOOLS) { for (var k in sofuu.tools.TOOLS) defs.push(k); } var mcp = []; try { if (sofuu.chat && sofuu.chat.state) { var st = sofuu.chat.state(); if (st && st.mcpTools) mcp = st.mcpTools; } } catch(e){} return {ok:true, builtin:defs, mcp:mcp}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    fun listAgents(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((function(){ try { var list = (sofuu.agent && sofuu.agent.list) ? sofuu.agent.list() : []; return {ok:true, agents:list}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())").thenObject()

    // ── Local GGUF models ─────────────────────────────────────────

    /**
     * Read the real caps out of a GGUF header: `ai.modelCaps` opens the
     * file and returns its context window, max output, and whether it
     * carries a thinking channel. Mirrors what the CLI's `/model x.gguf`
     * does so a phone model is configured the same way a desktop one is.
     *
     * Returns null when the file is missing, truncated, or not a GGUF —
     * the caller keeps the previous limits rather than guessing.
     */
    fun localModelCaps(path: String): CompletableFuture<JSONObject?> =
        query(
            "JSON.stringify((function(){ try { var c = sofuu.ai.modelCaps(${jsonString(path)}); " +
                "return {ok:true, caps:c||{}}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())"
        ).thenObject()

    /**
     * Confirm the engine can actually open a model before the app points
     * its config at it. `modelCaps` only parses the header, so this also
     * runs one short generation — that is what surfaces a model whose
     * tensors are the wrong shape or unsupported.
     *
     * Synchronous on the engine thread by design; a local model has no
     * network round-trip to wait for.
     */
    fun probeLocalModel(path: String, prompt: String = "hi", maxTokens: Int = 8): CompletableFuture<JSONObject?> =
        query(
            "JSON.stringify((function(){ try { var t = sofuu.ai.inferLocal(${jsonString(path)}, " +
                jsonString(prompt) + ", {maxTokens:" + maxTokens + ", temperature:0}); " +
                "return {ok:true, text: String(t == null ? '' : t)}; } catch(e){ " +
                "return {ok:false, error:String(e&&e.message||e)}; } })())"
        ).thenObject()

    // ── Provider model lists (ported from commands.rs) ──────────────

    /** Fetch a provider's model list through the engine's own fetch
     *  (keys never leave the runtime). Returns {ok, models, note, error}. */
    fun fetchProviderModels(name: String, endpoint: String, profile: String, key: String): CompletableFuture<JSONObject?> {
        val url = modelsUrl(endpoint, profile)
            ?: return CompletableFuture.completedFuture(JSONObject().put("ok", false).put("error", "no models URL"))
        val urlJs = jsonString(url)
        val keyJs = jsonString(key)
        return query(
            "(function(){ var url = $urlJs; var key = $keyJs; " +
                "var headers = key ? { 'Authorization': 'Bearer ' + key } : {}; " +
                "var timer = setTimeout(function () { __reply_put(JSON.stringify({ok:false, error:'timeout', models:[]})); }, 12000); " +
                "try { " +
                "fetch(url, { headers: headers }).then(function (res) { " +
                "if (!res.ok) { clearTimeout(timer); __reply_put(JSON.stringify({ok:false, error:'HTTP ' + res.status, models:[]})); return; } " +
                "return res.json(); }).then(function (j) { " +
                "clearTimeout(timer); " +
                "var raw = (j && j.data && Array.isArray(j.data)) ? j.data : (j && j.models && Array.isArray(j.models)) ? j.models : []; " +
                "var arr = raw.map(function (m) { return (m && (m.id || m.name)) || m; }).filter(function (x) { return x && String(x).trim(); }); " +
                "__reply_put(JSON.stringify({ok:true, models: arr, raw: JSON.stringify(j)})); }).catch(function (e) { " +
                "clearTimeout(timer); __reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e), models:[]})); }); " +
                "} catch (e) { clearTimeout(timer); __reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e), models:[]})); } })()"
        ).thenObject()
    }

    // ── Query plumbing ─────────────────────────────────────────────

    /** Enqueue a JS expression whose *promise or value* is answered
     *  through `__reply_put`. Resolves with the parsed JSON value. */
    private fun query(expr: String): CompletableFuture<Any?> {
        val f = CompletableFuture<Any?>()
        val script = "(function(){ try { Promise.resolve(eval(${jsonString(expr)})).then(function (v) { " +
            "__reply_put(JSON.stringify(v)); }, function (e) { " +
            "__reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); }); } " +
            "catch(e) { __reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); } })()"
        synchronized(replyLock) { pendingReplies.addLast(f) }
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete(null) }
                return@Runnable
            }
            try {
                rt.eval(script)
            } catch (e: Throwable) {
                synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete(null) }
            }
        })
        return f
    }

    private fun CompletableFuture<Any?>.thenObject(): CompletableFuture<JSONObject?> =
        thenApply { v -> v as? JSONObject }

    private fun CompletableFuture<Any?>.thenArray(): CompletableFuture<JSONArray?> =
        thenApply { v -> v as? JSONArray }

    private fun CompletableFuture<Any?>.thenCast(): CompletableFuture<JSONObject?> =
        thenApply { v -> v as? JSONObject }

    private fun evalResult(rt: SofuuBridge, script: String): String? {
        val raw = rt.eval(script) ?: return null
        return try {
            val envelope = JSONObject(raw)
            when {
                envelope.optBoolean("ok") -> envelope.optString("result").ifEmpty { null }
                else -> null
            }
        } catch (e: Throwable) {
            null
        }
    }

    companion object {
        private val BOOTSTRAP_JS = """
            (function(){
              globalThis.__evq = [];
              globalThis.__sofuu_push = function (e) { try { globalThis.__evq.push(e); } catch (x) {} };
              globalThis.__rq = [];
              globalThis.__reply_put = function (j) { try { globalThis.__rq.push(j); } catch (x) {} };
              return true;
            })()
        """.trimIndent()

        private val DRAIN_JS = """
            (function(){
              var ev = globalThis.__evq.splice(0, globalThis.__evq.length);
              var rp = globalThis.__rq.splice(0, globalThis.__rq.length);
              return JSON.stringify({ev: ev, rp: rp});
            })()
        """.trimIndent()

        /** JSON-encode a string as a JS string literal (JSON is a subset). */
        fun jsonString(s: String): String = JSONObject.quote(s)

        // ── models_url / profile detection (ported from commands.rs) ──

        fun isAnthropic(endpoint: String, profile: String): Boolean {
            val prof = profile.trim().lowercase()
            val low = endpoint.lowercase()
            return prof == "anthropic" || (prof.isEmpty() && (low.contains("anthropic") || low.contains("claude")))
        }

        fun isLocal(endpoint: String, profile: String): Boolean {
            val prof = profile.trim().lowercase()
            val low = endpoint.lowercase()
            return prof == "local" || (prof.isEmpty() && (low.contains("11434") || low.contains("localhost") || low.contains("127.0.0.1")))
        }

        fun modelsUrl(endpoint: String, profile: String): String? {
            if (isAnthropic(endpoint, profile)) return null
            if (isLocal(endpoint, profile)) {
                val root = if (endpoint.trim().isEmpty()) "http://127.0.0.1:11434" else endpoint.trim().trimEnd('/')
                return "$root/api/tags"
            }
            val ep = endpoint.trim().trimEnd('/')
            if (ep.isEmpty()) return null
            val root = stripUntilCore(ep, listOf("/chat/completions", "/completions"))
            return "$root/models"
        }

        private fun stripUntilCore(endpoint: String, suffixes: List<String>): String {
            for (s in suffixes) {
                val idx = endpoint.indexOf(s)
                if (idx > 0) return endpoint.substring(0, idx)
            }
            return endpoint
        }
    }
}

/** Everything the UI can observe from the engine. */
sealed class EngineEvent {
    object EngineReady : EngineEvent()
    data class BootError(val message: String) : EngineEvent()
    data class TurnStarted(val sid: String?) : EngineEvent()
    data class TurnFinished(val error: String?, val usage: Usage?) : EngineEvent()
    data class Ctx(
        val used: Long, val window: Long, val source: String,
        val model: String?, val sid: String?,
    ) : EngineEvent()
    data class ApprovalRequested(
        val id: String, val tool: String, val args: JSONObject, val risky: Boolean,
    ) : EngineEvent()
    data class ApprovalResolved(val id: String) : EngineEvent()
    data class Warn(val message: String) : EngineEvent()
    data class Step(val kind: String, val raw: JSONObject) : EngineEvent()
}

data class Usage(
    val promptTokens: Long? = null,
    val completionTokens: Long? = null,
    val costUsd: Double? = null,
    val steps: Int? = null,
    val servedBy: String? = null,
)
