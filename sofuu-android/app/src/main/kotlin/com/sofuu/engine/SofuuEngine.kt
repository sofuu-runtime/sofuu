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
    projectDir: String,
    /** Needed to build the native host tools ([DeviceTools]); they need a
     *  Context for the vibrator, battery and share sheet. */
    private val appContext: android.content.Context,
) {
    /**
     * The engine's active workspace. Mutable on purpose: `chat.setProject`
     * moves the engine (it chdirs), so the app-side paths that copy
     * attachments and delete session files must move with it — a `val` here
     * kept them pointing at the boot workspace after a switch.
     */
    @Volatile
    var projectDir: String = projectDir
        private set

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

    /** Pending engine replies, matched by the query id the JS reply carries.
     *  Arrival-order matching let a stray or duplicate reply resolve the
     *  NEXT caller's future; an id makes that impossible. */
    private val pendingReplies = HashMap<Long, CompletableFuture<Any?>>()
    private val replyLock = Any()
    /** Monotonic id stamped into every queued query's reply. */
    private val querySeq = java.util.concurrent.atomic.AtomicLong(0)

    /** Daemon timer for query deadlines — `CompletableFuture.orTimeout` is
     *  API 31 and this app's minSdk is 24. */
    private val timeoutScheduler: java.util.concurrent.ScheduledExecutorService =
        java.util.concurrent.Executors.newSingleThreadScheduledExecutor { r ->
            Thread(r, "sofuu-query-deadline").apply { isDaemon = true }
        }

    /** Guards engine start/shutdown so a timed-out shutdown can never hand
     *  the runtime to a second thread. */
    private val ownerLock = Any()

    /** Mirrors SOFUU_NO_SHELL for host tools; set by AppStore.applyRuntimeEnv. */
    @Volatile var hostShellAllowed: Boolean = true

    /** Commands the engine thread executes in order. */
    private val mailbox = LinkedBlockingQueue<Runnable>()

    val isReady: Boolean get() = ready.get()

    /** The C ABI version of the linked libsofuu. */
    val abiVersion: Int get() = SofuuBridge.abiVersion()

    // ── Boot ───────────────────────────────────────────────────────

    /** @return false when the previous worker has not finished exiting —
     *  starting anyway would put two threads on one runtime. */
    fun start(): Boolean {
        synchronized(ownerLock) {
            if (running.get()) return true
            val prev = thread
            if (prev != null && prev.isAlive) {
                log("start refused: engine thread still alive (previous shutdown in progress)")
                return false
            }
            running.set(true)
            // Instance tag: two engines in one process would silently split the
            // event pipeline (the pump drains one runtime's queues while turns
            // push into the other's). The boot line pins each instance so a twin
            // is impossible to miss.
            log("start: engine#0x${Integer.toHexString(hashCode())} thread=${Thread.currentThread().name}")
            thread = Thread({ engineMain() }, "sofuu-engine").also { it.start() }
            return true
        }
    }

    fun shutdown() {
        val t = synchronized(ownerLock) {
            if (!running.getAndSet(false)) return
            thread
        }
        // JOIN OUTSIDE THE LOCK: engineMain's own finally takes ownerLock to
        // clear `thread`, so joining while holding it would deadlock until
        // the timeout on every single shutdown.
        t?.join(4000)
        synchronized(ownerLock) {
            if (t != null && !t.isAlive && thread === t) thread = null
            if (t != null && t.isAlive) {
                log("shutdown: engine thread still alive after 4s; start() refuses until it exits")
            }
        }
    }

    private fun log(msg: String) {
        android.util.Log.i("sofuu-engine", msg)
        // Persistent breadcrumb: logcat's ring buffer rotates away exactly
        // when we need what it held — after a crash, possibly with the
        // cable unplugged. This file survives all of that and stays
        // readable via `adb shell run-as com.sofuu cat files/engine-debug.log`.
        try {
            val f = java.io.File(homeDir, "engine-debug.log")
            synchronized(fileLock) {
                if (f.length() > 2_000_000L) {
                    val one = java.io.File(homeDir, "engine-debug.1.log")
                    one.delete()
                    f.renameTo(one)
                    f.createNewFile()
                }
                java.io.FileWriter(f, true).use { w ->
                    w.append(java.text.SimpleDateFormat("MM-dd HH:mm:ss.SSS", java.util.Locale.US).format(java.util.Date()))
                    w.append(' ').append(msg).append('\n')
                }
            }
        } catch (_: Throwable) {
        }
    }

    private val fileLock = Any()

    /** Turn tracking for the adaptive pump: `turnActive` means a chat turn
     *  is between its `start` and `done` events; `lastEventMs` is the last
     *  time ANY engine event was seen. Both only feed the pump interval. */
    @Volatile internal var turnActive = false
    @Volatile private var lastEventMs = 0L
    @Volatile private var lastDrainLogMs = 0L

    private fun engineMain() {
        try {
            File(projectDir).mkdirs()
            // Engine console output → logcat AND the console file (the
            // logcat tag is dead on some OEM builds — the file survives).
            // Installed BEFORE rt_new so boot warnings are not lost.
            // Create the console file immediately so its presence (or
            // absence after a crash) is never ambiguous mid-test.
            kotlin.runCatching {
                java.io.File(homeDir, "engine-console.log").createNewFile()
            }
            SofuuBridge.setEngineLogFile(java.io.File(homeDir, "engine-console.log").absolutePath)
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
                    DeviceTools(appContext, isShellAllowed = { hostShellAllowed }).registerAll(rt)
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
            // The poll interval is adaptive: 50 ms while a turn is in
            // flight (tokens arrive in bursts and cancel must answer fast)
            // and 250 ms when idle. Every idle drain is a JS eval plus a
            // libuv tick plus a JSON parse — waking at 20 Hz forever burns
            // real power for nothing and warms the phone.
            try {
                while (running.get()) {
                    val idle = !turnActive &&
                        android.os.SystemClock.elapsedRealtime() - lastEventMs > 4_000L
                    val cmd = mailbox.poll(if (idle) 250L else 50L, TimeUnit.MILLISECONDS)
                    if (!running.get()) break
                    try {
                        cmd?.run()
                    } catch (e: Throwable) {
                        emit(EngineEvent.BootError(e.message ?: e.toString()))
                    }
                    // Its own guard: drainEvents runs JS through the runtime,
                    // so a throw here must not skip the cleanup below. It
                    // used to be outside every catch, so one bad eval leaked
                    // the entire engine and wedged every future caller.
                    try {
                        drainEvents(rt)
                    } catch (e: Throwable) {
                        log("drain failed: ${e.message ?: e.toString()}")
                    }
                }
                log("pump exit")
            } finally {
                // Release the runtime HERE, while this is still the thread
                // that created it (docs/EMBEDDING.md: a runtime belongs to
                // exactly one thread).
                try {
                    bridge?.close()
                    log("runtime freed")
                } catch (e: Throwable) {
                    log("sofuu_rt_free failed: ${e.message ?: e.toString()}")
                }
                bridge = null
                // The runtime is gone, so nothing can reach these any more.
                // Dropping the JNI global refs here is what stops each
                // registration from pinning its receiver for the process
                // lifetime.
                val released = SofuuBridge.freeHostTools()
                if (released > 0) log("host tools released: $released")
                hostTools.clear()
                ready.set(false)
                // Anything still queued will never be evaluated now, so
                // release its callers with null instead of leaving a future
                // that hangs whoever is blocking on it.
                synchronized(replyLock) {
                    val stranded = pendingReplies.values.toList()
                    pendingReplies.clear()
                    stranded.forEach { if (!it.isDone) it.complete(null) }
                }
                // Clear the owner slot only if this thread still owns it, so
                // a later start() can run once the previous worker is gone.
                synchronized(ownerLock) {
                    if (thread === Thread.currentThread()) thread = null
                }
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
        // nb: the pump calls this every ≤50 ms; during an active turn the
        // queue splices MUST stay instant — a draining eval here is what
        // locked the thread until the turn finished.
        val raw = rt.evalNb(DRAIN_JS) ?: return
        // evalNb wraps EVERY result in {ok, result} — the actual drain
        // payload ({ev, rp, rc, pc}) sits inside `result` as a JSON STRING
        // and must be unwrapped first. Reading ev/pc off the OUTER envelope
        // made every field null/0 forever: turns streamed below while the
        // app discarded each one of their events at this line.
        val obj = try {
            JSONObject(raw)
        } catch (e: Throwable) {
            log("drain: unparsable envelope raw=${raw.take(120)}")
            return
        }
        if (!obj.optBoolean("ok", false)) {
            log("drain: envelope error ${obj.optString("error").take(160)}")
            return
        }
        val payload = obj.optString("result")
        if (payload.isEmpty()) return
        val obj2 = try {
            JSONObject(payload)
        } catch (e: Throwable) {
            log("drain: unparsable payload raw=${payload.take(120)}")
            return
        }
        val ev = obj2.optJSONArray("ev")
        val rp = obj2.optJSONArray("rp")
        // Deliver replies, then events. Logging here is deliberate: the
        // engine thread is on the token path during a turn, and a file
        // write per drain (20 Hz with a date format under a lock) cost
        // measurable power. Log holds: errors always; the first drain
        // with events 1/s at most.
        val errText = if (obj2.has("error")) obj2.optString("error").take(200) else null
        val now = android.os.SystemClock.elapsedRealtime()
        lastEventMs = now
        if (errText != null) {
            log("drain: error $errText")
        } else if ((ev?.length() ?: 0) > 0 && now - lastDrainLogMs > 1_000L) {
            lastDrainLogMs = now
            log("drain: delivering ${ev?.length() ?: 0} event(s)")
        }
        if (rp != null) {
            for (i in 0 until rp.length()) {
                // Tagged replies are {id, json}; an untagged entry is a stray
                // from an expression that answered itself and has no waiter.
                val entry = rp.optJSONObject(i)
                val id = entry?.optLong("id", -1L) ?: -1L
                val replyJson = entry?.optString("json") ?: rp.optString(i)
                if (replyJson.isEmpty()) continue
                val d = if (id >= 0) synchronized(replyLock) { pendingReplies.remove(id) } else null
                if (d != null && !d.isDone) {
                    d.complete(parseReply(replyJson))
                } else if (d == null) {
                    log("drain: reply id=$id has no waiter — dropped")
                }
            }
        }
        val ev2 = ev ?: return
        for (i in 0 until ev2.length()) {
            val e = ev2.optJSONObject(i) ?: continue
            parseEvent(e)?.let { emit(it) }
        }
    }

    private fun emit(ev: EngineEvent) {
        _events.tryEmit(ev)
    }

    private fun parseEvent(e: JSONObject): EngineEvent? {
        val kind = e.optString("kind")
        val p = e.optJSONObject("payload") ?: JSONObject()
        // Track the turn for the adaptive pump — no file logging here:
        // this runs on the engine thread once per token.
        when (kind) {
            "start" -> {
                turnActive = true
                lastEventMs = android.os.SystemClock.elapsedRealtime()
            }
            "done" -> {
                turnActive = false
                lastEventMs = android.os.SystemClock.elapsedRealtime()
            }
        }
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
            // evalNb, not eval: chat.submit starts an async turn and this
            // eval must return immediately — the draining eval would hold
            // the runtime thread (and the whole event pump) until the turn
            // finished, so no token event and no cancel could ever land.
            val r = rt.evalNb(
                "(function(){ try {" +
                    "sofuu.chat.submit($textJs, {onEvent: globalThis.__sofuu_push, noTools: $noToolsJs, images: $imagesJs})" +
                    ".then(function(){}, function(e){ globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); });" +
                    "return true;" +
                    "} catch(e) { globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); return false; } })()"
            )
            log("submit: evalNb → ${r?.take(180) ?: "NULL"}")
        }
    }

    fun cancelTurn() {
        // evalNb: a cancel must execute NOW — a draining eval would queue
        // behind the very turn it is meant to stop.
        //
        // TWO halts, on purpose: __host_poke cancels the chat loop
        // (approvals, RLM, agent steps), but chat.js's cancel() predates
        // local streaming and does not abort a model decoding mid-token —
        // __ai_abort() (the desktop's Esc hook) tears down the in-flight
        // local C stream by flipping its aborted flag.
        onEngine { rt ->
            val poke = rt.evalNb("__host_poke('{\"type\":\"cancel\"}')")
            val abort = rt.evalNb("(typeof __ai_abort === 'function') ? JSON.stringify(__ai_abort()) : '0'")
            log("cancel: poke=${poke?.take(60) ?: "NULL"} abort=${abort?.take(60) ?: "NULL"}")
        }
    }

    fun resolveApproval(id: String, allow: Boolean, always: Boolean) {
        onEngine { rt ->
            val payload = JSONObject()
                .put("type", "approval")
                .put("id", id)
                .put("allow", allow)
                .put("always", always)
            // evalNb — an approval answer must land mid-turn, where a
            // draining eval would wait for the turn it is being asked to
            // unblock.
            rt.evalNb("__host_poke(${jsonString(payload.toString())})")
        }
    }

    fun compactSession(): CompletableFuture<JSONObject?> =
        query("Promise.resolve(sofuu.chat.compact())").thenCast()

    fun sessions(): CompletableFuture<JSONArray?> =
        query(
            "JSON.stringify((function(){ " +
                "try { " +
                "  if (!sofuu.chat || !sofuu.chat.sessions) return []; " +
                "  var s = sofuu.chat.sessions(); " +
                "  if (Array.isArray(s)) return s; " +
                "  if (s && Array.isArray(s.sessions)) return s.sessions; " +
                "  return []; " +
                "} catch (e) { return []; } " +
                "})())"
        ).thenArray()

    fun sessionTurns(id: String): CompletableFuture<JSONArray?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.sessionTurns) ? sofuu.chat.sessionTurns(${jsonString(id)}) : [])").thenArray()

    fun newSession(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.newSession) ? sofuu.chat.newSession() : {error:'chat engine unavailable'})").thenObject()

    fun resumeSession(id: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.resume) ? sofuu.chat.resume(${jsonString(id)}) : {error:'chat engine unavailable'})").thenObject()

    fun setProject(path: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.setProject) ? sofuu.chat.setProject(${jsonString(path)}) : {error:'chat engine unavailable'})")
            .thenCast()
            // Follow the engine only when it accepted the switch: a rejected
            // path must not move the app's attachment/session roots.
            .thenApply { r ->
                if (r?.optBoolean("ok") == true) projectDir = path
                r
            }

    fun setPermissions(profile: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.setPermissions) ? sofuu.chat.setPermissions(${jsonString(profile)}) : {error:'chat engine unavailable'})").thenObject()

    fun chatState(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {error:'chat engine unavailable'})").thenObject()

    /** Re-harvest the active endpoint's model caps (async in JS). Called
     *  after a provider/model switch so the context meter and the output
     *  cap follow the NEW model instead of the one selected before. */
    fun detectCaps(): CompletableFuture<JSONObject?> =
        query("sofuu.chat.detectCaps()").thenObject()

    fun deleteSession(id: String): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.deleteSession) ? sofuu.chat.deleteSession(${jsonString(id)}) : {ok:false, error:'chat engine unavailable'})").thenObject()

    fun clearSessions(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.clearSessions) ? sofuu.chat.clearSessions() : {ok:false, error:'chat engine unavailable'})").thenObject()

    fun usageReport(): CompletableFuture<JSONObject?> =
        query("JSON.stringify((sofuu.chat && sofuu.chat.usage) ? sofuu.chat.usage() : {error:'chat engine unavailable'})").thenObject()

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
                // modelCaps returns a JSON STRING (rt/ai.rs); handing it to
                // Kotlin unparsed made optJSONObject("caps") null, so no
                // window/max-output was ever applied and every local model
                // was forced to effort off.
                "if (typeof c === 'string') { try { c = JSON.parse(c); } catch (eP) { c = null; } } " +
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
        // ONE returned promise, answered exactly once by the query wrapper.
        // The old shape started fetch(...) without returning it AND pushed
        // its own __reply_put: the future resolved immediately with null
        // (so the cache never populated) and the late reply could be
        // consumed by the next query. A bare `return fetch(...)` is not
        // enough either — the old settle() would then be a second reply.
        return query(
            """
            (function(){ var url = $urlJs; var key = $keyJs;
              var headers = key ? { 'Authorization': 'Bearer ' + key } : {};
              var timeout = new Promise(function (res) {
                setTimeout(function () { res({ok:false, error:'timeout', models:[]}); }, 12000);
              });
              var listing = fetch(url, { headers: headers }).then(function (res) {
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
              }).then(function (j) {
                var raw = (j && j.data && Array.isArray(j.data)) ? j.data
                        : (j && j.models && Array.isArray(j.models)) ? j.models : [];
                var arr = raw.map(function (m) { return (m && (m.id || m.name)) || m; })
                               .filter(function (x) { return x && String(x).trim(); });
                return { ok: true, models: arr };
              }).catch(function (e) {
                return { ok: false, error: String(e && e.message || e), models: [] };
              });
              return Promise.race([listing, timeout]);
            })()
            """.trimIndent()
        ).thenObject()
    }

    // ── Query plumbing ─────────────────────────────────────────────

    /** Enqueue a JS expression whose *promise or value* is answered
     *  through `__reply_put`. Resolves with the parsed JSON value.
     *
     *  [timeoutMs] is the caller's real budget, NOT a fixed cap: a headless
     *  agent run legitimately takes minutes, and the old fixed 30 s bound
     *  silently killed every `queryFuture(…, 300_000)` caller. */
    private fun query(expr: String, timeoutMs: Long = 30_000): CompletableFuture<Any?> {
        val qid = querySeq.incrementAndGet()
        val f = CompletableFuture<Any?>()
        // Release the id whenever the future settles — reply, timeout or
        // engine teardown. Leaving it registered let a late reply find a
        // future nobody was waiting on any more.
        f.whenComplete { _, _ -> synchronized(replyLock) { pendingReplies.remove(qid) } }
        submit(expr, f, qid)
        // Bound EVERY query at the source. A caller is usually a Main-
        // dispatcher coroutine; its bare .get() froze the whole UI the
        // moment a reply once failed to land (boot-time ANR, dropbox
        // trace at AppStore.kt refreshSessions → get). Timeout completes
        // the future with null so callers degrade instead of hanging.
        armTimeout(f, timeoutMs)
        return f.handle { v, e ->
            if (e != null) {
                log("query: bounded → null (${e.javaClass.simpleName}) expr=${expr.take(60)}")
                null
            } else {
                v
            }
        }
    }

    /** `CompletableFuture.orTimeout` is API 31 and minSdk is 24, so the
     *  deadline is a scheduled completion instead. */
    private fun armTimeout(f: CompletableFuture<Any?>, ms: Long) {
        val timer = timeoutScheduler.schedule({
            if (!f.isDone) {
                f.completeExceptionally(java.util.concurrent.TimeoutException("query timeout ${ms}ms"))
            }
        }, ms, TimeUnit.MILLISECONDS)
        f.whenComplete { _, _ -> timer.cancel(false) }
    }

    /**
     * Queue [expr] and resolve when the engine answers.
     *
     * Like [query] but bounded, so a caller that waits on the result off
     * the engine thread — a headless agent run takes tens of seconds —
     * cannot hang forever on an endpoint that never replies.
     */
    fun queryFuture(expr: String, timeoutMs: Long): CompletableFuture<JSONObject?> =
        query(expr, timeoutMs).thenApply { it as? JSONObject }

    private fun submit(expr: String, f: CompletableFuture<Any?>, qid: Long) {
        // SYNC FAST PATH: evaluate and complete the future straight from the
        // eval envelope — no promise, no reply queue, nothing that CAN be
        // lost. If the expression returns a promise (network fetches), fall
        // back to the id-tagged reply queue.
        val script = "(function(){ try { var v = eval(${jsonString(expr)});" +
                "if (v && typeof v.then === 'function') {" +
                "  globalThis.__rq = globalThis.__rq || [];" +
                "  globalThis.__reply_put = globalThis.__reply_put || function (j) { globalThis.__rq.push(j); };" +
                "  Promise.resolve(v).then(function (v2) { __reply_put({id:$qid, json: JSON.stringify(v2)}); }, " +
                "function (e) { __reply_put({id:$qid, json: JSON.stringify({__async_error:String(e&&e.message||e)})}); });" +
                "  return {ok:true, result:'__pending_promise'};" +
                "}" +
                "return {ok:true, result: v}; } catch(e) { return {ok:false, error:{message:String(e&&e.message||e)}}; } })()"
        synchronized(replyLock) { pendingReplies[qid] = f }
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                synchronized(replyLock) { pendingReplies.remove(qid) }
                if (!f.isDone) f.complete(null)
                return@Runnable
            }
            try {
                log("query: eval start, pending=${synchronized(replyLock) { pendingReplies.size }} expr=${expr.take(50)}")
                val raw = rt.evalNb(script)
                if (raw == null) {
                    log("query: evalNb null — completing null")
                    synchronized(replyLock) { pendingReplies.remove(qid) }
                    if (!f.isDone) f.complete(null)
                    return@Runnable
                }
                val obj = JSONObject(raw)
                if (!obj.optBoolean("ok", false)) {
                    val err = obj.optJSONObject("error")?.optString("message") ?: "engine error"
                    synchronized(replyLock) { pendingReplies.remove(qid) }
                    if (!f.isDone) f.complete("{\"__error\":$err}")
                    log("query: sync error $err")
                    return@Runnable
                }
                val result = obj.opt("result")
                if (result == "__pending_promise") {
                    // Async expression: DRAIN_JS + the tagged reply queue
                    // finish it (bounded by query()'s orTimeout).
                    log("query: promise pending — queue path")
                    return@Runnable
                }
                val payload = when (result) {
                    null, JSONObject.NULL -> "null"
                    else -> result.toString()
                }
                synchronized(replyLock) { pendingReplies.remove(qid) }
                if (!f.isDone) f.complete(parseReply(payload))
                log("query: sync ok len=${payload.length}")
            } catch (e: Throwable) {
                log("query: eval threw ${e.message ?: e.toString()}")
                synchronized(replyLock) { pendingReplies.remove(qid) }
                if (!f.isDone) f.complete(null)
            }
        })
    }

    private fun CompletableFuture<Any?>.thenObject(): CompletableFuture<JSONObject?> =
        thenApply { v -> v as? JSONObject }

    private fun CompletableFuture<Any?>.thenArray(): CompletableFuture<JSONArray?> =
        thenApply { v ->
            when (v) {
                is JSONArray -> v
                is JSONObject -> v.optJSONArray("sessions")
                else -> null
            }
        }

    private fun CompletableFuture<Any?>.thenCast(): CompletableFuture<JSONObject?> =
        thenApply { v -> v as? JSONObject }


    /**
     * Evaluate a JS expression on the engine thread and return the JSON
     * object the script produced.
     *
     * The ABI wraps every eval in its own `{ok, result}` transport
     * envelope; that envelope is unwrapped here so a snippet's own
     * `{ok:false, error:...}` is what the caller sees. Before the unwrap
     * every [RuntimeSurface] card read `count`/`dim`/`error` off the
     * transport object and rendered zeros or blanks.
     *
     * It MUST go through the mailbox. The runtime belongs to the engine
     * thread, and `sofuu_rt_eval` takes no lock, so calling it from a
     * caller thread while the pump is draining would run two JS contexts
     * on one QuickJS state — a native crash, not a wrong answer.
     *
     * Asynchronous on purpose: callers get a future and never touch the
     * runtime from their own thread.
     */
    fun evalJsonOnEngine(script: String): CompletableFuture<JSONObject?> {
        val f = CompletableFuture<JSONObject?>()
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                if (!f.isDone) f.complete(null)
                return@Runnable
            }
            try {
                // evalNb, not eval: every snippet on this channel is
                // synchronous, and `rt_eval` DRAINS the loop — calling it
                // from the pump during a streaming turn would hold the
                // engine thread until that turn finished.
                val raw = rt.evalNb(script)
                if (!f.isDone) f.complete(unwrapEval(raw))
            } catch (e: Throwable) {
                if (!f.isDone) f.complete(null)
            }
        })
        return f
    }

    /** Strip the ABI's `{ok,result}` envelope; synthesize an error object
     *  when the eval itself failed. */
    private fun unwrapEval(raw: String?): JSONObject? {
        if (raw == null) return null
        return try {
            val env = JSONObject(raw)
            if (!env.optBoolean("ok", false)) {
                JSONObject()
                    .put("ok", false)
                    .put("error", env.optJSONObject("error")?.optString("message") ?: "engine error")
            } else {
                when (val payload = env.opt("result")) {
                    is JSONObject -> payload
                    is String -> runCatching { JSONObject(payload) }.getOrNull()
                    null, JSONObject.NULL -> JSONObject()
                    else -> JSONObject().put("ok", true).put("value", payload)
                }
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
              if (typeof sofuu !== 'undefined' && typeof sofuu.spawn === 'function') {
                var _origSpawn = sofuu.spawn;
                sofuu.spawn = function(opts) {
                  if (opts && (opts.command === '/bin/sh' || opts.command === 'sh')) {
                    try {
                      if (typeof sofuu.fs !== 'undefined' && sofuu.fs.exists) {
                        if (!sofuu.fs.exists('/bin/sh') && sofuu.fs.exists('/system/bin/sh')) {
                          opts.command = '/system/bin/sh';
                        }
                      } else {
                        opts.command = '/system/bin/sh';
                      }
                    } catch (_) {
                      opts.command = '/system/bin/sh';
                    }
                  }
                  return _origSpawn.call(this, opts);
                };
              }
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

        /** Public alias — JS snippets built by [RuntimeSurface] embed
         *  user-supplied strings and need the same escaping. */
        @JvmStatic fun quote(s: String): String = JSONObject.quote(s)

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
