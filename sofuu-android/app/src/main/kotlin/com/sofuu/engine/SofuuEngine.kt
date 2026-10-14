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
        val t = thread
        thread = null
        t?.join(4000)
        // A thread that did not stop in time is still holding `bridge`, and
        // a later start() would spawn a second worker against the same
        // runtime. Refuse rather than hand one runtime to two threads —
        // that is the use-after-free this used to allow.
        if (t != null && t.isAlive) {
            log("shutdown: engine thread still alive after 4s; leaving it as the sole owner")
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

    private var lastSeenRc = 0
    private var lastSeenPc = 0
    @Volatile private var lastBeaconMs = 0L

    private val deltaCount = java.util.concurrent.atomic.AtomicLong(0)

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

            // TEMPORARY async-advance probe: does JS async work (timer +
            // promise + microtask) make progress under the pump's evalNb
            // ticks at all? Decides whether a silent turn is a dead JS job
            // queue or a stuck turn elsewhere. Remove once streaming is
            // verified.
            Thread({
                try {
                    Thread.sleep(1500)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        rt2.evalNb("globalThis.__probe_flag='sync_ok'")
                        val r1 = rt2.evalNb(
                            "(function(){ new Promise(function(r){ setTimeout(r, 150); }).then(function(){ globalThis.__probe_flag='async_ok'; }); return 'scheduled'; })()"
                        )
                        val r2 = rt2.evalNb(
                            "(async function(){ await Promise.resolve(); globalThis.__probe_pure='micro_ok'; })()"
                        )
                        log("probe: scheduled r1=${r1?.take(60) ?: "NULL"} pure=${r2?.take(60) ?: "NULL"}")
                    })
                    Thread.sleep(2500)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val v = rt2.evalNb("JSON.stringify({flag: globalThis.__probe_flag || 'unset', pure: globalThis.__probe_pure || 'unset'})")
                        log("probe: RESULT ${v?.take(140) ?: "NULL"}")
                    })
                    // Probe 3 — the engine's local stream in isolation, no
                    // chat.js/agent.js driver: if THIS stalls, the bug is in
                    // the engine stream path; if it completes, the turn
                    // driver stalls.
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val r = rt2.evalNb(
                            "(function(){ globalThis.__probe3='starting';" +
                                "(async function(){ try { var n=0, t0=Date.now(), last='';" +
                                "for await (var c of sofuu.ai.stream('Reply with exactly: ok', {model:'/data/user/0/com.sofuu/files/models/qwen3-0.6b-i1-q4_k_m.gguf'})) { n++; last=String(c&&c.text||''); if (n>500) break; }" +
                                "globalThis.__probe3='DONE n='+n+' last='+last+' ms='+(Date.now()-t0);" +
                                "} catch(e) { globalThis.__probe3='ERR '+String(e&&e&&e.message||e); } })();" +
                                "return 'launched'; })()"
                        )
                        log("probe3: ${r?.take(80) ?: "NULL"}")
                        // Probe 5 — cold-load stream with a 5-token cap:
                        // if THIS wedges too, the first load never reaches a
                        // single token and it is not a long-run scheduler
                        // issue.
                        val r5 = rt2.evalNb(
                            "(function(){ globalThis.__p5='starting';" +
                                "(async function(){ try { var n=0, t0=Date.now();" +
                                "for await (var c of sofuu.ai.stream('hi', {model:'/data/user/0/com.sofuu/files/models/qwen3-0.6b-i1-q4_k_m.gguf', maxTokens:5})) { n++; }" +
                                "globalThis.__p5='DONE n='+n+' ms='+(Date.now()-t0);" +
                                "} catch(e) { globalThis.__p5='ERR '+String(e&&e&&e.message||e); } })();" +
                                "return 'launched'; })()"
                        )
                        log("probe5: ${r5?.take(60) ?: "NULL"}")
                    })
                    Thread.sleep(35000)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val v = rt2.evalNb("JSON.stringify({p3: globalThis.__probe3 || 'unset', p5: globalThis.__p5 || 'unset'})")
                        log("probe3/5: T35 ${v?.take(200) ?: "NULL"}")
                    })
                    Thread.sleep(45000)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val v = rt2.evalNb("String(globalThis.__probe3 || 'unset')")
                        log("probe3: T80 ${v?.take(160) ?: "NULL"}")
                    })
                    // Probe 8 — the REAL chat turn, but noTools and under the
                    // probe's control: submit() + its resolution flag, then
                    // chat.state() polling. A tool-less turn that never
                    // resolves convicts chat.js's core; a resolved one
                    // convicts the tool dispatch path.
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val r = rt2.evalNb(
                            "(function(){ globalThis.__p8='submitting';" +
                                "globalThis.__p8t0 = Date.now();" +
                                "sofuu.chat.submit('Count to three in words', {onEvent: globalThis.__sofuu_push, noTools: true}).then(" +
                                "function(v){ globalThis.__p8='RESOLVED '+JSON.stringify(v)+' ms='+(Date.now()-globalThis.__p8t0); }," +
                                "function(e){ globalThis.__p8='REJECTED '+String(e&&e.message||e)+' ms='+(Date.now()-globalThis.__p8t0); });" +
                                "return 'k'; })()"
                        )
                        log("probe8: ${r?.take(60) ?: "NULL"}")
                    })
                    Thread.sleep(20000)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val st = rt2.evalNb("JSON.stringify((function(){ var s=(sofuu.chat&&sofuu.chat.state)?sofuu.chat.state():{}; return {p8: globalThis.__p8||'unset', streaming: s.streaming, steps: s.steps, tool: s.lastTool||null, err: s.lastError||null}; })())")
                        log("probe8: T20 ${st?.take(220) ?: "NULL"}")
                    })
                    Thread.sleep(30000)
                    mailbox.put(Runnable {
                        val rt2 = bridge ?: return@Runnable
                        val st = rt2.evalNb("JSON.stringify((function(){ var s=(sofuu.chat&&sofuu.chat.state)?sofuu.chat.state():{}; return {p8: globalThis.__p8||'unset', streaming: s.streaming, steps: s.steps, tool: s.lastTool||null, err: s.lastError||null}; })())")
                        log("probe8: T50 ${st?.take(220) ?: "NULL"}")
                    })
                } catch (_: InterruptedException) {
                }
            }, "sofuu-probe").start()

            // Pump loop: commands first, then a drain of both JS queues.
            var ticks = 0L
            try {
                while (running.get()) {
                    val cmd = mailbox.poll(50, TimeUnit.MILLISECONDS)
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
                    // Debug heartbeat: proves the pump thread is alive and
                    // ticking (a stall WITHOUT these lines is a dead or
                    // locked thread; a stall WITH them is the engine simply
                    // not producing events).
                    if (++ticks % 200L == 0L) log("pump: tick #$ticks")
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
                    val stranded = pendingReplies.toList()
                    pendingReplies.clear()
                    stranded.forEach { if (!it.isDone) it.complete(null) }
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
        val obj = try {
            JSONObject(raw)
        } catch (e: Throwable) {
            log("drain: unparsable raw=${raw.take(120)}")
            return
        }
        // TEMPORARY deep telemetry — unconditional: a DRAIN_JS exception
        // surfaces as an error envelope (no ev/rp/rc), and pushes surface
        // as pc drift; both printed verbatim until the reply-loss bug is
        // confirmed fixed. Remove once streaming is verified.
        val ev = obj.optJSONArray("ev")
        val rp = obj.optJSONArray("rp")
        val rc = obj.optInt("rc")
        val pc = obj.optInt("pc")
        val errText = if (obj.has("error")) obj.optString("error").take(160) else null
        val now = android.os.SystemClock.elapsedRealtime()
        // Beacon: log EVERY drain at least once per 2 s, even the all-zero
        // ones. A fully-silent channel used to be invisible by construction
        // (0 == lastSeen counters never differ), which is how a dead turn
        // looked identical to a healthy idle engine.
        if (errText != null || (ev?.length() ?: 0) > 0 || rc != lastSeenRc || pc != lastSeenPc ||
            now - lastBeaconMs > 2_000L
        ) {
            // Full kinds list + pc/rc: the anatomy of the turn. Events were
            // previously invisible behind the 120-char raw take, which is
            // exactly how a missing 'start'/'done' could not be distinguished
            // from a missing push.
            val kinds = (0 until (ev?.length() ?: 0)).mapNotNull { ev?.optJSONObject(it)?.optString("kind") }
            log("drain: n=${ev?.length() ?: 0} kinds=[${kinds.joinToString(",")}] rp=${rp?.length() ?: 0} rc=$rc pc=$pc err=$errText")
            if (errText != null) log("drain: raw=${raw.take(400)}")
            lastSeenRc = rc
            lastSeenPc = pc
            lastBeaconMs = now
        }
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
        // Turn breadcrumbs (low-noise: deltas throttled, rest verbatim).
        when {
            kind != "answer_delta" -> log("evt: $kind")
            deltaCount.getAndIncrement() % 25L == 0L -> log("evt: answer_delta #${deltaCount.get()}")
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
            """
            (function(){ var url = $urlJs; var key = $keyJs;
              var headers = key ? { 'Authorization': 'Bearer ' + key } : {};
              /* Exactly one answer per query, whichever arrives first. This
                 script answers ITSELF via __reply_put, and query() wraps
                 every expression in another __reply_put — so before this
                 guard a slow provider pushed two replies, the second of
                 which was consumed by the NEXT query and shifted every
                 later answer by one. That is why refreshModelCache never
                 populated: its future was resolved by the wrapper's
                 JSON.stringify(undefined). */
              var done = false, timer = null;
              function settle(o){ if (done) return; done = true; if (timer) clearTimeout(timer); __reply_put(JSON.stringify(o)); }
              timer = setTimeout(function () { settle({ok:false, error:'timeout', models:[]}); }, 12000);
              try {
                fetch(url, { headers: headers }).then(function (res) {
                  if (!res.ok) { settle({ok:false, error:'HTTP ' + res.status, models:[]}); return; }
                  return res.json();
                }).then(function (j) {
                  var raw = (j && j.data && Array.isArray(j.data)) ? j.data
                          : (j && j.models && Array.isArray(j.models)) ? j.models : [];
                  var arr = raw.map(function (m) { return (m && (m.id || m.name)) || m; })
                                 .filter(function (x) { return x && String(x).trim(); });
                  settle({ok:true, models: arr, raw: JSON.stringify(j)});
                }).catch(function (e) {
                  settle({ok:false, error:String(e&&e.message||e), models:[]});
                });
              } catch (e) {
                settle({ok:false, error:String(e&&e.message||e), models:[]});
              } })()
            """.trimIndent()
        ).thenObject()
    }

    // ── Query plumbing ─────────────────────────────────────────────

    /** Enqueue a JS expression whose *promise or value* is answered
     *  through `__reply_put`. Resolves with the parsed JSON value. */
    private fun query(expr: String): CompletableFuture<Any?> {
        val f = submit(expr, CompletableFuture<Any?>())
        // Bound EVERY query at the source. A caller is usually a Main-
        // dispatcher coroutine; its bare .get() froze the whole UI the
        // moment a reply once failed to land (boot-time ANR, dropbox
        // trace at AppStore.kt refreshSessions → get). Timeout completes
        // the future with null so callers degrade instead of hanging.
        return f.orTimeout(30_000, TimeUnit.MILLISECONDS)
            .handle { v, e ->
                if (e != null) {
                    log("query: bounded → null (${e.javaClass.simpleName}) expr=${expr.take(60)}")
                    null
                } else {
                    v
                }
            }
    }

    /**
     * Queue [expr] and resolve when the engine answers.
     *
     * Like [query] but bounded, so a caller that waits on the result off
     * the engine thread — a headless agent run takes tens of seconds —
     * cannot hang forever on an endpoint that never replies.
     */
    fun queryFuture(expr: String, timeoutMs: Long): CompletableFuture<JSONObject?> =
        query(expr).thenApply { it as? JSONObject }.orTimeout(timeoutMs, TimeUnit.MILLISECONDS)

    private fun submit(expr: String, f: CompletableFuture<Any?>): CompletableFuture<Any?> {
        // SYNC FAST PATH: evaluate and complete the future straight from the
        // eval envelope — no promise, no reply queue, nothing that CAN be
        // lost. (The old __rq/DRAIN_JS chain silently dropped replies on
        // device while working on host; every app query today is a
        // synchronous expression, so the queue is only needed for promises.)
        // If the expression returns a promise (network fetches), fall back
        // to the reply queue.
        val script = "(function(){ try { var v = eval(${jsonString(expr)});" +
                "if (v && typeof v.then === 'function') {" +
                "  globalThis.__rq = globalThis.__rq || [];" +
                "  globalThis.__reply_put = globalThis.__reply_put || function (j) { globalThis.__rq.push(j); };" +
                "  Promise.resolve(v).then(function (v2) { __reply_put(JSON.stringify(v2)); }, function (e) { __reply_put(JSON.stringify({__async_error:String(e&&e.message||e)})); });" +
                "  return {ok:true, result:'__pending_promise'};" +
                "}" +
                "return {ok:true, result: v}; } catch(e) { return {ok:false, error:{message:String(e&&e.message||e)}}; } })()"
        synchronized(replyLock) { pendingReplies.addLast(f) }
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete(null) }
                return@Runnable
            }
            try {
                log("query: eval start, pending=${synchronized(replyLock) { pendingReplies.size }} expr=${expr.take(50)}")
                val raw = rt.evalNb(script)
                if (raw == null) {
                    log("query: evalNb null — completing null")
                    synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete(null) }
                    return@Runnable
                }
                val obj = JSONObject(raw)
                if (!obj.optBoolean("ok", false)) {
                    val err = obj.optJSONObject("error")?.optString("message") ?: "engine error"
                    synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete("{\"__error\":$err}") }
                    log("query: sync error $err")
                    return@Runnable
                }
                val result = obj.opt("result")
                if (result == "__pending_promise") {
                    // Async expression: DRAIN_JS + the reply queue finish it
                    // (bounded by query()'s orTimeout).
                    log("query: promise pending — queue path")
                    return@Runnable
                }
                val payload = when (result) {
                    null, JSONObject.NULL -> "null"
                    else -> result.toString()
                }
                synchronized(replyLock) { if (pendingReplies.remove(f)) f.complete(parseReply(payload)) }
                log("query: sync ok len=${payload.length}")
            } catch (e: Throwable) {
                log("query: eval threw ${e.message ?: e.toString()}")
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

    /**
     * Evaluate a JS expression that already evaluates to a JSON object,
     * without the `{ok,result}` envelope the query channel wraps.
     *
     * [RuntimeSurface] uses this because those snippets build their own
     * `{ok:..., error:...}` object so a failure inside the brain or the
     * ML gates is reported rather than collapsing to null — which is the
     * whole difference between "the brain is off" and "the call threw".
     *
     * It MUST go through the mailbox. The runtime belongs to the engine
     * thread, and `sofuu_rt_eval` takes no lock, so calling it from a
     * caller thread while the pump is draining would run two JS contexts
     * on one QuickJS state — a native crash, not a wrong answer.
     *
     * Synchronous, so callers must be off the main thread.
     */
    @Synchronized
    fun evalJson(script: String): JSONObject? =
        try {
            JSONObject(bridge?.eval(script))
        } catch (e: Throwable) {
            null
        }

    /** The mailbox-routed form of [evalJson]: never touches the runtime
     *  from the caller's thread. Same contract, same returned object. */
    fun evalJsonOnEngine(script: String): CompletableFuture<JSONObject?> {
        val f = CompletableFuture<JSONObject?>()
        mailbox.put(Runnable {
            val rt = bridge
            if (rt == null) {
                if (!f.isDone) f.complete(null)
                return@Runnable
            }
            try {
                val raw = rt.eval(script)
                if (!f.isDone) f.complete(if (raw == null) null else try { JSONObject(raw) } catch (e: Throwable) { null })
            } catch (e: Throwable) {
                if (!f.isDone) f.complete(null)
            }
        })
        return f
    }

    companion object {
        private val BOOTSTRAP_JS = """
            (function(){
              globalThis.__evq = [];
              globalThis.__push_count = 0;
              globalThis.__sofuu_push = function (e) {
                globalThis.__push_count++;
                try { console.log('DBG push n=' + globalThis.__push_count + ' kind=' + (e && e.kind)); } catch (xL) {}
                try { globalThis.__evq.push(e); } catch (x) {}
              };
              globalThis.__rq = [];
              globalThis.__reply_count = 0;
              globalThis.__reply_put = function (j) {
                try {
                  globalThis.__reply_count++;
                  console.log("DBG reply_put n=" + globalThis.__reply_count + " len=" + String(j).length);
                  globalThis.__rq.push(j);
                } catch (x) { console.log("DBG reply_put fail " + x); }
              };
              return true;
            })()
        """.trimIndent()

        private val DRAIN_JS = """
            (function(){
              var ev = globalThis.__evq.splice(0, globalThis.__evq.length);
              var rp = globalThis.__rq.splice(0, globalThis.__rq.length);
              return JSON.stringify({ev: ev, rp: rp, rc: (globalThis.__reply_count || 0), pc: (globalThis.__push_count || 0)});
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
