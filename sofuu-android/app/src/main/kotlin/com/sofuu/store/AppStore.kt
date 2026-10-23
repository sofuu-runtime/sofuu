package com.sofuu.store

import android.app.Application
import android.content.Context
import android.os.Environment
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.sofuu.engine.EngineEvent
import com.sofuu.engine.SofuuEngine
import com.sofuu.engine.Usage as EngineUsage
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import com.sofuu.engine.Speech
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Full app state — the Kotlin mirror of the desktop's AppState. */
data class AppState(
    val engineReady: Boolean = false,
    val bootError: String? = null,
    val sessions: List<SessionInfo> = emptyList(),
    val activeSessionId: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val messageCache: Map<String, List<ChatMessage>> = emptyMap(),
    val streaming: Boolean = false,
    /** Live speed readout on the streaming bubble. */
    val liveMetrics: String? = null,
    val approvals: List<Approval> = emptyList(),
    val config: SofuuConfig? = null,
    val projectDir: String? = null,
    val workspaces: List<String> = emptyList(),
    val permissions: String = "prompt",
    val settingsOpen: Boolean = false,
    val settingsTab: String = "general",
    val chatFontSize: Int = 15,
    val mode: String = "agent", // agent | chat
    val preview: SessionPreview? = null,
    val theme: String = "system", // light | dark | system
    val cloudBg: Boolean = false,
    val pickedImages: List<String> = emptyList(),
    /** @mentions spliced into the composer by the file picker. */
    val pendingMentions: String = "",
    /** True while the mic is capturing for dictation. */
    val dictating: Boolean = false,
    val phase: String? = null,
    val ctx: CtxMeter? = null,
    val ctxCache: Map<String, CtxMeter> = emptyMap(),
    val drawerOpen: Boolean = false,
    val historyOpen: Boolean = false,
    val sheet: String? = null, // "model" | "effort" | "layers" | "mode" | "permissions"
    val modelsCache: Map<String, ModelCacheEntry> = emptyMap(),

    /** Absolute paths of the GGUF models imported into filesDir/models. */
    val localModels: List<String> = emptyList(),

    /** path → display label ("Llama 3.2 1B Instruct · Q8_0"), read from the
     *  GGUF header on refresh. Companion to [localModels]. */
    val localModelLabels: Map<String, String> = emptyMap(),
)

/** One provider's cached model listing (the desktop's model cache). */
data class ModelCacheEntry(val models: List<String>, val ok: Boolean, val error: String?, val note: String?, val cached: Boolean, val at: Long)

/** The slash-command registry — ONE list feeds the "/" palette and /help. */
data class SlashCommand(val name: String, val arg: String?, val desc: String) {
    val signature: String get() = "/" + name + (arg?.let { " $it" } ?: "")
}

val SLASH_COMMANDS = listOf(
    SlashCommand("new", null, "Fresh session — clears the chat"),
    SlashCommand("compact", null, "Fold old history into a summary"),
    SlashCommand("ctx", "[tokens]", "Show context usage, or set window"),
    SlashCommand("maxout", "[tokens]", "Show/set max output tokens"),
    SlashCommand("model", "[name]", "Show or set the model"),
    SlashCommand("provider", "[name]", "Show or set the provider"),
    SlashCommand("effort", "[low|medium|high|max|off]", "Show or set reasoning effort"),
    SlashCommand("brain", "[on|off]", "Show or toggle brain memory"),
    SlashCommand("remember", "<fact>", "Pin a fact to the brain"),
    SlashCommand("why", null, "Which memories shaped the last answer"),
    SlashCommand("cost", null, "Session tokens + spend"),
    SlashCommand("resume", "[id]", "List sessions, or resume one"),
    SlashCommand("permissions", "[profile]", "Show or set tool permissions"),
    SlashCommand("tools", null, "List built-in + MCP tools"),
    SlashCommand("agents", null, "List loaded agents"),
    SlashCommand("rlm", "[on|off|auto]", "Show or set long-context mode"),
    SlashCommand("ml", "[on|off|info]", "Context-economy gates"),
    SlashCommand("sessions", null, "List sessions in this workspace"),
    SlashCommand("context", null, "Show current engine context"),
    SlashCommand("help", null, "This list"),
)

const val MAX_SESSIONS = 7

/**
 * The app state holder: folds the engine event stream into UI state and
 * drives every engine call. A direct port of the desktop's store.ts —
 * same reducer semantics, same slash surface, same session discipline.
 */
class AppStore(app: Application) : AndroidViewModel(app) {

    val configs = ConfigStore(app)
    private val uiPrefs = UiPrefs(app)

/** The library's env-var knobs, applied before the runtime boots. */
val runtimeEnv = RuntimeEnv(app)

    /**
     * Re-write the process environment from the current settings.
     *
     * Safe without a restart for everything the engine re-reads per turn
     * or per request — the tool set, the shell gate, the memory backend,
     * the web engine, the SSRF guard, the stall deadline. Those call
     * sites run inside the agent loop, not once at load.
     */
    fun applyRuntimeEnv(): Map<String, String> {
        val applied = runtimeEnv.apply()
        // The engine gates its own bash on SOFUU_NO_SHELL; the host shell
        // tool must honour the same policy or it is a hole in it.
        engine.hostShellAllowed = !runtimeEnv.noShell
        return applied
    }

    /**
     * Tear the engine down and boot it again so a registration-gated knob
     * takes effect. Only SOFUU_NO_ML needs this — sofuu-core registers
     * the whole `sofuu.ml` namespace at init, so without a restart the
     * namespace is simply absent and every gate is a silent no-op.
     */
    fun restartEngineForEnv() {
        // Off-main: shutdown() joins the worker (up to 4 s) and this is
        // called from Settings toggles on the Main dispatcher.
        viewModelScope.launch(Dispatchers.Default) {
            if (state.streaming) {
                // Freeing the runtime mid-turn would strand the open tool
                // round-trip. Cancel first so the agent's abort path runs.
                engine.cancelTurn()
            }
            engine.shutdown()
            applyRuntimeEnv()
            var started = engine.start()
            var attempts = 0
            while (!started && attempts < 6) {
                kotlinx.coroutines.delay(500)
                started = engine.start()
                attempts++
            }
            if (!started) {
                withContext(Dispatchers.Main) {
                    systemRow(
                        "Engine restart deferred — the previous runtime is still " +
                            "shutting down. The setting applies on the next restart.",
                    )
                }
            }
        }
    }

    val engine: SofuuEngine = SofuuEngine(
        homeDir = app.filesDir.absolutePath,
        projectDir = (configs.projectDir() ?: File(app.filesDir, "workspace").absolutePath),
        appContext = app,
    )

    var state by mutableStateOf(
        AppState(
            permissions = uiPrefs.permissions,
            mode = uiPrefs.mode,
            theme = uiPrefs.theme,
            cloudBg = uiPrefs.cloudBg,
            chatFontSize = uiPrefs.chatFontSize,
            workspaces = configs.workspaces(),
            projectDir = configs.projectDir() ?: File(app.filesDir, "workspace").absolutePath,
        )
    )
        private set

    /** Direct engine surface for brain ops (/remember). The lambda reads
     *  the live Brain toggle so /remember refuses when memory is off. */
    private val runtimeSurface: com.sofuu.engine.RuntimeSurface by lazy {
        com.sofuu.engine.RuntimeSurface(engine) { state.config?.brain != false }
    }

    private val seq = AtomicLong(0)

    private fun nextId(prefix: String): String = "$prefix-${System.currentTimeMillis().toString(36)}-${seq.incrementAndGet()}"

    init {
        // Engine events → state.
        viewModelScope.launch {
            // collect, NOT collectLatest: handleEvent is a reducer, not a
            // suspending computation, and collectLatest cancels the PREVIOUS
            // handler when a new event arrives. An event landing between the
            // two statements of TurnFinished left `streaming=false` with the
            // assistant bubble still marked streaming and no notification.
            // handleEvent(TurnFinished) already refreshes sessions; a second
            // call here doubled the engine round-trips per turn.
            engine.events.collect { ev -> handleEvent(ev) }
        }
        // Load config + state immediately (cheap, no engine). Repair runs
        // first: an entry saved as a bare base URL before normalization
        // existed would otherwise fail every turn with "stream ended
        // without [DONE]".
        configs.repairEndpoints()
        state = state.copy(config = configs.redacted())
        // Engine environment knobs MUST be in place before the runtime is
        // created: SOFUU_NO_ML gates module registration inside the
        // library, so setting it afterwards silently does nothing.
        val appliedEnv = applyRuntimeEnv()
        engine.start()
        android.util.Log.i("sofuu-engine", "env applied: $appliedEnv")
        // The engine's curl has no built-in trust store on Android; without
        // this bundle every https:// call fails certificate verification.
        // Say so instead of leaving the user with a raw curl error.
        if (appliedEnv[RuntimeEnv.KEY_CA_BUNDLE].isNullOrEmpty()) {
            systemRow(
                "warning: no system CA certificates were readable — HTTPS " +
                    "providers will fail certificate verification.",
            )
        }
    }

    // ── Engine event reducer ──────────────────────────────────────

    private fun handleEvent(ev: EngineEvent) {
        when (ev) {
            is EngineEvent.EngineReady -> {
                state = state.copy(engineReady = true, bootError = null)
                onEngineReady()
            }
            is EngineEvent.BootError -> {
                // A failure mid-turn must return the composer to Send and
                // mark the bubble. The only writer of streaming=false used
                // to be TurnFinished, so a dead engine left Stop on screen
                // forever with no way back.
                if (state.streaming) {
                    patchAssistant { m ->
                        if (m.status == "streaming") m.copy(status = "error", error = ev.message) else m
                    }
                }
                state = state.copy(
                    engineReady = false,
                    bootError = ev.message,
                    streaming = false,
                    phase = null,
                    liveMetrics = null,
                )
            }
            is EngineEvent.TurnStarted -> {
                if (ev.sid != null && ev.sid != state.activeSessionId) {
                    state = state.copy(activeSessionId = ev.sid)
                    saveCurrentSessionMessages()
                }
            }
            is EngineEvent.TurnFinished -> {
                flushDelta()
                val speed = finalMetricsText()
                state = state.copy(streaming = false, phase = null, liveMetrics = null)
                if (!foreground && ev.error == null) notifier?.invoke("Sofuu finished", "The turn completed.")
                if (ev.error != null) {
                    patchAssistant { m -> m.copy(status = "error", error = ev.error) }
                } else {
                    patchAssistant { m -> m.copy(status = "done", usage = ev.usage?.toUi(), metrics = speed) }
                }
                saveCurrentSessionMessages()
                flushSessionSave()
                refreshSessions()
            }
            is EngineEvent.Ctx -> {
                val owner = ev.sid ?: state.activeSessionId ?: return
                val meter = CtxMeter(ev.used, ev.window, ev.source, ev.model, ev.sid)
                val cache = state.ctxCache + (owner to meter)
                state = if (owner == state.activeSessionId) state.copy(ctx = meter, ctxCache = cache) else state.copy(ctxCache = cache)
            }
            is EngineEvent.ApprovalRequested -> {
                state = state.copy(approvals = state.approvals + Approval(ev.id, ev.tool, ev.args, ev.risky))
                if (!foreground) notifier?.invoke("Sofuu needs your approval", "${ev.tool} is waiting for approval")
            }
            is EngineEvent.ApprovalResolved -> state = state.copy(approvals = state.approvals.filterNot { it.id == ev.id })
            is EngineEvent.Warn -> {
                // Engine warnings ride as system rows (dim, compact) — the
                // same surface the desktop's step reducer writes.
                systemRow(ev.message)
            }
            is EngineEvent.Step -> {
                if (ev.kind == "recall") noteRecall(ev.raw)
                applyStep(ev.kind, ev.raw)
            }
        }
    }

    private fun EngineUsage.toUi(): Usage = Usage(promptTokens, completionTokens, costUsd, steps, servedBy)

    /** Buffered answer text for the streaming assistant bubble (see the
     *  answer_delta case in applyStep). Must only run on the Main
     *  dispatcher — the event collector guarantees that. */
    private val pendingDelta = StringBuilder()
    private var deltaFlush: kotlinx.coroutines.Job? = null

    // ── Turn speed instrumentation (UI readout) ──
    // Deltas are token pieces, so the rate is piece-rate — within rounding
    // of the model's true tok/s. All timing via elapsedRealtime.
    private var turnStartMs = 0L
    private var firstTokenMs = 0L
    private var lastTokenMs = 0L
    private var tokenCount = 0

    // ── Last recall (feeds /why) ──
    // chat.state() has no lastRecall field, so the command used to print
    // "null". These track what the recall step itself reported.
    @Volatile private var lastRecallAt = 0L
    @Volatile private var lastRecallCount = 0L
    @Volatile private var lastRecallScope = ""
    @Volatile private var lastRecallDropped = 0L

    private fun noteRecall(raw: JSONObject) {
        val p = raw.optJSONObject("payload") ?: return
        lastRecallAt = android.os.SystemClock.elapsedRealtime()
        lastRecallCount = p.optLong("count")
        lastRecallScope = p.optString("scope")
        lastRecallDropped = p.optLong("dropped")
    }

    /** What the last recall step reported, for `/why`. */
    fun brainWhy(): JSONObject {
        if (lastRecallAt == 0L) return JSONObject().put("ok", true).put("none", true)
        return JSONObject()
            .put("ok", true)
            .put("count", lastRecallCount)
            .put("scope", lastRecallScope)
            .put("dropped", lastRecallDropped)
    }

    private fun resetTurnMetrics() {
        turnStartMs = android.os.SystemClock.elapsedRealtime()
        firstTokenMs = 0L
        lastTokenMs = 0L
        tokenCount = 0
    }

    private fun liveMetricsText(untilMs: Long): String? {
        if (firstTokenMs == 0L) return null
        val ttft = (firstTokenMs - turnStartMs) / 1000.0
        val window = (untilMs - firstTokenMs).coerceAtLeast(1L) / 1000.0
        val tps = tokenCount / window
        return "%.1f tok/s · first %.1fs".format(java.util.Locale.US, tps, ttft)
    }

    private fun finalMetricsText(): String? {
        if (firstTokenMs == 0L || tokenCount == 0) return null
        val ttft = (firstTokenMs - turnStartMs) / 1000.0
        val window = (lastTokenMs - firstTokenMs).coerceAtLeast(1L) / 1000.0
        val tps = tokenCount / window
        return "%.1f tok/s · first %.1fs · %d tok".format(java.util.Locale.US, tps, ttft, tokenCount)
    }

    private fun flushDelta() {
        deltaFlush?.cancel()
        deltaFlush = null
        if (pendingDelta.isEmpty()) return
        val text = pendingDelta.toString()
        pendingDelta.setLength(0)
        patchAssistant { m -> m.copy(content = m.content + text) }
        state = state.copy(
            liveMetrics = if (state.streaming) liveMetricsText(android.os.SystemClock.elapsedRealtime()) else null,
        )
    }

    private fun patchAssistant(fn: (ChatMessage) -> ChatMessage) {
        val messages = state.messages.toMutableList()
        for (i in messages.indices.reversed()) {
            if (messages[i].role == MsgRole.ASSISTANT) {
                messages[i] = fn(messages[i])
                break
            }
        }
        state = state.copy(messages = messages)
    }

    private fun currentAssistant(): ChatMessage? =
        state.messages.lastOrNull { it.role == MsgRole.ASSISTANT }

    /** The step reducer — the desktop's applyStep, in Kotlin. */
    private fun applyStep(kind: String, raw: JSONObject) {
        if (currentAssistant() == null) return
        val p = raw.optJSONObject("payload") ?: JSONObject()
        val runId = raw.optString("runId").ifEmpty { null }
        val t = if (raw.has("t")) raw.optLong("t") else null
        val phase = phaseForStep(kind, p)

        when (kind) {
            "answer_delta" -> {
                val text = p.optString("text")
                if (text.isEmpty()) {
                    state = state.copy(phase = phase ?: state.phase)
                    return
                }
                // Batch on a ~100 ms cadence instead of per token. At
                // 20-30 tok/s a per-token path copies the whole message
                // list and re-runs the markdown compositor for every
                // token — main-thread burn (heat) with zero visible
                // difference at that granularity.
                val now = android.os.SystemClock.elapsedRealtime()
                if (firstTokenMs == 0L) firstTokenMs = now
                lastTokenMs = now
                tokenCount++
                pendingDelta.append(text)
                if (deltaFlush == null) {
                    deltaFlush = viewModelScope.launch {
                        kotlinx.coroutines.delay(100)
                        flushDelta()
                    }
                }
            }
            "think" -> {
                val text = p.optString("text")
                if (text.isEmpty()) {
                    state = state.copy(phase = phase ?: state.phase)
                    return
                }
                patchAssistant { m ->
                    val tl = m.timeline.toMutableList()
                    val last = tl.lastOrNull()
                    if (last != null && last.kind == "think" && last.runId == runId) {
                        val started = last.t ?: t
                        val durationMs = if (t != null && started != null) Math.max(0, t - started) else null
                        tl[tl.size - 1] = last.copy(
                            detail = (last.detail ?: "") + text,
                            durationMs = durationMs,
                            badge = if (durationMs != null && durationMs > 0) "· ${Timeline.formatDur(durationMs)}" else last.badge,
                        )
                        m.copy(timeline = tl)
                    } else {
                        tl.add(TimelineEntry(nextId("tl"), "think", "Thought", detail = text, runId = runId, t = t))
                        m.copy(timeline = tl)
                    }
                }
            }
            "tool_result" -> {
                val isChecklist = p.optString("name") == "todo_write" && p.optJSONArray("todos") != null
                if (!isChecklist) {
                    patchAssistant { m ->
                        val ok = !p.has("error")
                        val result = p.optString("result").takeIf { it.isNotEmpty() }?.take(400)
                            ?: p.optString("summary").takeIf { it.isNotEmpty() }
                        val name = p.optString("name")
                        for (i in m.timeline.indices) {
                            val e = m.timeline[i]
                            val match = if (name == "delegate") {
                                e.kind == "delegate" && e.ok == null
                            } else {
                                (e.kind == "tool" || e.kind == "tool_result") && e.ok == null &&
                                    (if (e.tool != null) e.tool == name else e.title == name)
                            }
                            if (!match) continue
                            val tl = m.timeline.toMutableList()
                            tl[i] = e.copy(ok = ok, result = result ?: e.result)
                            return@patchAssistant m.copy(timeline = tl)
                        }
                        // No match (system notice, result after a reload) —
                        // the standalone result row still renders.
                        val fallback = stepToTimeline(kind, raw, p)
                        if (fallback != null) m.copy(timeline = m.timeline + fallback) else m
                    }
                }
            }
            "compact" -> {
                val stage = p.optString("stage")
                patchAssistant { m ->
                    val tl = m.timeline.toMutableList()
                    val idx = tl.indexOfFirst { it.kind == "compact" && it.live }
                    if (stage == "start") {
                        if (idx < 0) tl.add(TimelineEntry(nextId("tl"), "compact", "Compressing", live = true, t = t))
                    } else if (idx >= 0) {
                        if (p.optBoolean("ok", true) == false) {
                            tl.removeAt(idx)
                        } else {
                            val saved = Regex("([\\d,]+) tk saved").find(tl[idx].detail ?: "")?.groupValues?.get(1)
                            tl[idx] = tl[idx].copy(live = false, title = "Context compacted", badge = saved?.let { "· $it tk saved" })
                        }
                    }
                    m.copy(timeline = tl)
                }
            }
            else -> {
                // `warn` never reaches a STEP: SofuuEngine maps the event type
                // to EngineEvent.Warn first (system row), so the old
                // compact-notice patch here was unreachable.
                val entry = stepToTimeline(kind, raw, p)
                if (entry != null) {
                    patchAssistant { m -> m.copy(timeline = m.timeline + entry.copy(runId = runId)) }
                }
            }
        }
        state = state.copy(phase = phase ?: state.phase)
    }

    /** Step → timeline row (desktop stepToTimeline). */
    private fun stepToTimeline(kind: String, raw: JSONObject, p: JSONObject): TimelineEntry? {
        val t = if (raw.has("t")) raw.optLong("t") else null
        val runId = raw.optString("runId").ifEmpty { null }
        return when (kind) {
            "tool" -> {
                val name = p.optString("tool").ifEmpty { p.optString("name").ifEmpty { "tool" } }
                // delegate/todo_write have richer sibling events.
                if (name == "delegate" || name == "todo_write") null
                else {
                    val row = Timeline.toolRow(name, parseToolArgs(p.optString("args")))
                    TimelineEntry(nextId("tl"), "tool", row.title, tool = name, detail = row.detail, t = t)
                }
            }
            "tool_result" -> {
                if (p.optString("name") == "todo_write" && p.optJSONArray("todos") != null) {
                    val todos = p.optJSONArray("todos")!!
                    val total = todos.length()
                    var done = 0
                    for (i in 0 until total) if (todos.optJSONObject(i)?.optString("status") == "done") done++
                    TimelineEntry(
                        nextId("tl"), "todo", "Checklist",
                        detail = todos.toString(),
                        badge = "· $done/$total done", t = t,
                    )
                } else {
                    val name = p.optString("tool").ifEmpty { p.optString("name") }
                    // A policy-refused tool reports `blocked: <why>` and no
                    // `error`, so the row used to render as a success. Surface
                    // the refusal instead of a green result line.
                    val blocked = p.optString("blocked")
                    val resultText = p.optString("result").takeIf { it.isNotEmpty() }?.take(400)
                        ?: p.optString("summary").takeIf { it.isNotEmpty() }
                        ?: if (blocked.isNotEmpty()) "blocked by policy: $blocked" else null
                    val ok = !p.has("error") && blocked.isEmpty()
                    if (name == "system") {
                        TimelineEntry(nextId("tl"), "tool_result", "Notice", tool = "system", result = resultText, ok = ok, t = t)
                    } else {
                        TimelineEntry(nextId("tl"), "tool_result", name.ifEmpty { "result" }, tool = name.ifEmpty { null }, result = resultText, ok = ok, t = t)
                    }
                }
            }
            "delegate" -> TimelineEntry(nextId("tl"), "delegate", p.optString("agent").ifEmpty { "sub-agent" }, tool = "delegate", detail = p.optString("task").ifEmpty { null }, t = t)
            "recall" -> {
                val count = p.optLong("count")
                val bits = ArrayList<String>()
                if (count > 0) bits += "$count ${if (count == 1L) "memory" else "memories"}"
                when (p.optString("scope")) {
                    "agent" -> bits += "private"
                    "shared" -> bits += "shared brain"
                }
                if (p.optLong("dropped") > 0) bits += "${p.optLong("dropped")} dropped"
                TimelineEntry(nextId("tl"), "recall", "Recalled memories", badge = if (bits.isNotEmpty()) "· ${bits.joinToString(" · ")}" else null, t = t)
            }
            // The plan's only payload is "Step N" — dropped (desktop parity).
            "plan" -> null
            // No "warn" row: warnings arrive as EngineEvent.Warn (system row).
            "consolidated" -> {
                val clusters = p.optInt("clusters")
                TimelineEntry(nextId("tl"), "consolidated", "Memory consolidated", badge = if (clusters > 0) "· $clusters ${if (clusters == 1) "cluster" else "clusters"}" else null, t = t)
            }
            "allocgate" -> {
                val bits = ArrayList<String>()
                if (p.has("fit")) bits += p.opt("fit").toString()
                if (p.has("actions")) bits += p.opt("actions").toString()
                if (p.optString("action") == "learned_limit") bits += "learned ${p.optString("kind", "limit")} for ${p.optString("model", "model")}"
                TimelineEntry(nextId("tl"), "allocgate", "Context gate", detail = if (bits.isNotEmpty()) bits.joinToString(" — ") else null, t = t)
            }
            "mlgate" -> {
                val rule = p.optString("rule")
                TimelineEntry(nextId("tl"), "mlgate", Timeline.mlgateTitle(rule), tool = rule.ifEmpty { null }, detail = p.optString("nudge").ifEmpty { rule.ifEmpty { null } }, t = t)
            }
            else -> if (kind.startsWith("rlm")) {
                val o = p
                var detail: String? = null
                if (o.length() == 0) detail = null
                when (kind) {
                    "rlm:route" -> detail = "routed to RLM (${o.opt("ctxTokens") ?: "?"}/${o.opt("windowTokens") ?: "?"} tk)"
                    "rlm:done" -> {
                        val bits = ArrayList<String>()
                        if (o.optInt("calls") > 0) bits += "${o.optInt("calls")} LLM calls"
                        if (o.optInt("toolCalls") > 0) bits += "${o.optInt("toolCalls")} tool calls"
                        if (o.optInt("rounds") > 0) bits += "${o.optInt("rounds")} rounds"
                        detail = if (bits.isNotEmpty()) bits.joinToString(" · ") else null
                    }
                    "rlm:fallback" -> detail = o.optString("error").ifEmpty { null }
                }
                TimelineEntry(nextId("tl"), "rlm", "RLM run", tool = kind, detail = detail, t = t)
            } else null
        }
    }

    // ── Submit + slash commands ───────────────────────────────────

    fun submit(text: String, images: List<String> = emptyList()) {
        touchEngineUse()
        val trimmed = text.trim()
        if (trimmed.isEmpty() && images.isEmpty()) return
        if (trimmed.startsWith("/")) {
            // Slash commands run engine queries; they are bounded now but
            // even a bounded wait must not sit on the Main dispatcher.
            viewModelScope.launch(Dispatchers.Default) { handleCommand(trimmed) }
            return
        }
        val userText = if (trimmed.isEmpty() && images.isNotEmpty()) "[image]" else trimmed
        state = state.copy(
            messages = state.messages + ChatMessage(nextId("m"), MsgRole.USER, userText) +
                ChatMessage(nextId("m"), MsgRole.ASSISTANT, status = "streaming"),
            streaming = true,
            liveMetrics = null,
            // Truthful label: with effort off (all local non-thinking models)
            // nothing "thinks" — the model answers directly.
            phase = if (state.config?.effort.isNullOrEmpty()) "Generating" else "Thinking",
        )
        saveCurrentSessionMessages()
        resetTurnMetrics()
        // Off-main: a first send mints a session via engine.newSession().get()
        // — bounded, but a bounded wait is still a frozen UI on Main.
        viewModelScope.launch(Dispatchers.Default) {
            try {
                var activeId = state.activeSessionId
                if (activeId == null) {
                    val minted = engine.newSession().get()
                    val id = minted?.optString("sessionId").orEmpty()
                    if (id.isNotEmpty()) {
                        val sessionInfo = makeSessionInfo(id)
                        withContext(Dispatchers.Main) {
                            state = state.copy(sessions = listOf(sessionInfo) + state.sessions, activeSessionId = id)
                            saveCurrentSessionMessages()
                        }
                        activeId = id
                    }
                }
                withContext(Dispatchers.Main) {
                    saveCurrentSessionMessages()
                }
                engine.submitTurn(trimmed, noTools = state.mode == "chat", images = images)
            } catch (e: Exception) {
                patchAssistant { m -> m.copy(status = "error", error = e.message) }
                state = state.copy(streaming = false, phase = null)
            }
        }
    }

    fun systemRow(text: String) {
        state = state.copy(messages = state.messages + ChatMessage(nextId("m"), MsgRole.SYSTEM, text))
        saveCurrentSessionMessages()
    }

    private fun handleCommand(raw: String) {
        val m = Regex("^/(\\w+)(?:\\s+([\\s\\S]*))?$").find(raw.trim())
        val cmd = m?.groupValues?.get(1)?.lowercase() ?: "help"
        val arg = (m?.groupValues?.getOrNull(2) ?: "").trim()
        val help = SLASH_COMMANDS.joinToString("\n") {
            it.signature.padEnd(20) + "— " + it.desc
        }
        // Default, not Main.immediate: every branch below blocks on an
        // engine future (bounded at 30 s), and on Main that is an ANR.
        viewModelScope.launch(Dispatchers.Default) {
            when (cmd) {
                "help", "commands" -> systemRow(help)
                "new", "clear" -> newChat()
                "compact" -> {
                    systemRow("Compacting…")
                    val r = engine.compactSession().get()
                    systemRow(if (r?.optBoolean("ok") == true) "Compacted." else "Compact failed" + (r?.optString("error")?.let { ": $it" } ?: ""))
                }
                "ctx" -> {
                    if (arg.isNotEmpty()) {
                        val n = arg.filter { it.isDigit() }.toIntOrNull()
                        if (n != null && n in 1000..1000000) {
                            patchConfig(JSONObject().put("ctx_window", n))
                            systemRow("Context window → ${",".format(n)} tokens.")
                        } else systemRow("Usage: /ctx [tokens] (1000–1000000). Bare /ctx shows live usage.")
                    } else {
                        val c = state.ctx
                        if (c != null && c.window > 0) {
                            systemRow("Context ${",".format(c.used)} / ${",".format(c.window)} tokens (${c.pct}%) — ${c.source}")
                        } else {
                            // No live meter yet (fresh session or a model just
                            // switched): show the selected model's window from
                            // the config so the number is observable before the
                            // first turn, and say the usage side is unknown.
                            val cfgWin = state.config?.ctxWindow ?: 0
                            systemRow(
                                if (cfgWin > 0) "No live meter yet — window ${",".format(cfgWin)} tokens (from the selected model); the ring fills on the next turn."
                                else "No context meter yet — send a message first."
                            )
                        }
                    }
                }
                "maxout" -> {
                    if (arg.isEmpty()) {
                        val v = state.config?.maxOutput ?: 0
                        systemRow(if (v > 0) "Max output: ${",".format(v)} tokens." else "Max output: model default (set with /maxout <tokens>).")
                    } else {
                        val n = arg.filter { it.isDigit() }.toIntOrNull()
                        if (n != null && n in 256..384000) {
                            patchConfig(JSONObject().put("max_output", n))
                            systemRow("Max output → ${",".format(n)} tokens.")
                        } else systemRow("Usage: /maxout <tokens> (256–384000).")
                    }
                }
                "model" -> {
                    if (arg.isEmpty()) {
                        val c = state.config
                        systemRow(if (c != null) "${c.provider.ifEmpty { "(no provider)" }} · ${c.model.ifEmpty { "(no model)" }}" else "Config not loaded yet.")
                    } else {
                        // The engine reads the model off the ACTIVE provider
                        // entry; writing only the flat `model` key (what this
                        // used to do) left the turn running the old model and
                        // the command silently did nothing. Same contract as
                        // pickModel.
                        val cfg = state.config
                        val active = cfg?.active?.ifEmpty { cfg.provider }.orEmpty()
                        if (active.isEmpty()) {
                            systemRow("Model not set — no provider is active. Use /provider <name> first.")
                        } else {
                            pickModel(active, arg)
                            systemRow("Model → $arg ($active).")
                        }
                    }
                }
                "provider" -> {
                    if (arg.isEmpty()) {
                        val c = state.config
                        systemRow(if (c != null) "Provider: ${c.provider.ifEmpty { "(none)" }} · model ${c.model.ifEmpty { "(none)" }}" else "Config not loaded yet.")
                    } else {
                        patchConfig(
                            JSONObject().put("provider", arg).put("active", arg)
                                .put("ctx_window", 0).put("max_output", 0),
                        )
                        reharvestCaps()
                        systemRow("Provider → $arg. Pick a model in Settings → Models if turns fail.")
                    }
                }
                "effort" -> {
                    if (arg.isEmpty()) {
                        val cur = state.config?.effort?.takeIf { it.isNotEmpty() } ?: "off"
                        systemRow("Effort: $cur (low|medium|high|max|off).")
                    } else {
                        val lvl = arg.lowercase()
                        if (lvl !in listOf("low", "medium", "high", "max", "off")) systemRow("Usage: /effort low|medium|high|max|off.")
                        else {
                            // "off" is persisted as "" — that is the config
                            // contract (the effort sheet writes "" too).
                            // Writing the literal "off" shipped an unknown
                            // level to the engine.
                            patchConfig(JSONObject().put("effort", if (lvl == "off") "" else lvl))
                            systemRow("Effort → $lvl.")
                        }
                    }
                }
                "brain" -> {
                    if (arg.isEmpty()) {
                        systemRow("Brain: ${if (state.config?.brain != false) "on" else "off"} — memories persist across sessions.")
                    } else {
                        val on = Regex("^(on|1|true|yes)$", RegexOption.IGNORE_CASE).containsMatchIn(arg)
                        val off = Regex("^(off|0|false|no)$", RegexOption.IGNORE_CASE).containsMatchIn(arg)
                        if (!on && !off) systemRow("Usage: /brain [on|off].")
                        else {
                            patchConfig(JSONObject().put("brain", on))
                            systemRow(if (on) "Brain on — memories persist across sessions" else "Brain off.")
                        }
                    }
                }
                "remember" -> {
                    if (arg.isEmpty()) systemRow("Usage: /remember <fact> — pin a fact to the brain.")
                    else {
                        // The old call went through sofuu.memory.remember,
                        // which needs an OPEN handle and an options object;
                        // it always answered "no open brain". This routes
                        // through the same cached handle the chat uses.
                        val r = runtimeSurface.brainRemember(arg)
                        systemRow(
                            if (r?.optBoolean("ok") == true) {
                                "Remembered · ${r.optLong("count")} memories in this project's brain."
                            } else {
                                "Remember failed" + (r?.optString("error")?.let { ": $it" } ?: "")
                            }
                        )
                    }
                }
                "why" -> {
                    val r = brainWhy()
                    if (r.optBoolean("none", false)) {
                        systemRow("No recalled memories shaped the last answer.")
                    } else {
                        val scope = when (r.optString("scope")) {
                            "agent" -> "private scope"
                            "shared" -> "shared brain"
                            else -> "brain"
                        }
                        val dropped = if (r.optLong("dropped") > 0) " · ${r.optLong("dropped")} dropped" else ""
                        systemRow(
                            "Last recall: ${r.optLong("count")} " +
                                "${if (r.optLong("count") == 1L) "memory" else "memories"} from the $scope$dropped."
                        )
                    }
                }
                "tools" -> {
                    val r = engine.listTools().get()
                    if (r?.optBoolean("ok") != true) systemRow("tools: ${r?.optString("error") ?: "unavailable"}")
                    else {
                        val b = r.optJSONArray("builtin")?.let { arr -> (0 until arr.length()).joinToString(", ") { arr.optString(it) } } ?: "—"
                        systemRow("Built-in: $b\nMCP: ${r.optJSONArray("mcp") ?: JSONArray()}")
                    }
                }
                "agents" -> {
                    val r = engine.listAgents().get()
                    systemRow(if (r?.optBoolean("ok") == true) "Agents: ${r.opt("agents") ?: "[]"}" else "agents: ${r?.optString("error") ?: "unavailable"}")
                }
                "rlm" -> {
                    if (arg.isEmpty()) systemRow("RLM: ${state.config?.rlm ?: "off"} (on|off|auto).")
                    else {
                        val v = arg.lowercase()
                        if (v !in listOf("on", "off", "auto")) systemRow("Usage: /rlm on|off|auto.")
                        else {
                            patchConfig(JSONObject().put("rlm", v))
                            systemRow("RLM → $v.")
                        }
                    }
                }
                "ml" -> {
                    val sub = arg.lowercase()
                    if (arg.isEmpty() || sub == "info") {
                        val r = engine.mlInfo().get()
                        systemRow(if (r?.optBoolean("ok") == true) "ML gates: ${r.opt("info") ?: "{}"}" else "ml: ${r?.optString("error") ?: "unavailable"}")
                    } else if (sub == "on" || sub == "off") {
                        patchConfig(JSONObject().put("ml", sub == "on"))
                        systemRow("ML gates $sub.")
                    } else systemRow("Usage: /ml [on|off|info] — learn/adopt/discard/reset stay CLI-only for now.")
                }
                "sessions" -> {
                    val sessions = engine.sessions().get()?.let { SessionInfo.fromJsonArray(it) } ?: emptyList()
                    systemRow(
                        if (sessions.isNotEmpty()) sessions.joinToString("\n") { "${it.id} · ${it.model.ifEmpty { "?" }} · ${if (it.ended) "ended" else "live"}" }
                        else "No sessions in this workspace yet."
                    )
                }
                "context" -> {
                    val r = engine.contextDump().get()
                    systemRow(if (r?.optBoolean("ok") == true) "Context: ${(r.opt("state") ?: "{}").toString().take(2000)}" else "context: ${r?.optString("error") ?: "unavailable"}")
                }
                "cost" -> {
                    val st = engine.chatState().get()
                    systemRow("Session: ${"%04f".format(st?.optDouble("sessionSpendUsd") ?: 0.0)} USD · ${",".format((st?.optDouble("sessionTokens") ?: 0.0).toLong())} tokens")
                }
                "resume" -> {
                    if (arg.isEmpty()) {
                        val sessions = engine.sessions().get()?.let { SessionInfo.fromJsonArray(it) } ?: emptyList()
                        systemRow(
                            if (sessions.isNotEmpty()) sessions.take(10).joinToString("\n") { "${it.id} · ${it.model.ifEmpty { "?" }} · ${if (it.ended) "ended" else "live"}" } + "\n/resume <id> to open one"
                            else "No sessions in this workspace yet."
                        )
                    } else {
                        val r = engine.resumeSession(arg).get()
                        systemRow(if (r?.optBoolean("ok") == true) "Resumed $arg." else "Resume failed" + (r?.optString("error")?.let { ": $it" } ?: ""))
                    }
                }
                "permissions" -> {
                    if (arg.isNotEmpty()) {
                        val r = engine.setPermissions(arg).get()
                        systemRow(if (r?.optBoolean("ok") == true) "Permissions → ${r.optString("permissions")}" else "Failed${r?.optString("error")?.let { ": $it" } ?: ""}")
                        setPermissions(arg)
                    } else systemRow("Permissions: ${state.permissions} (full | edit | plan | prompt)")
                }
                // CLI-only surfaces: honest pointer instead of "unknown".
                "watch", "hooks", "ghost", "serve", "outputs", "verify", "share", "import", "work", "done", "note", "notify", "sync", "at" ->
                    systemRow("/$cmd is CLI-only in this version — use the terminal chat for it. Parity tracked for a later release.")
                else -> systemRow("Unknown command /$cmd — /help lists what exists.")
            }
        }
    }

    // ── Sessions ──────────────────────────────────────────────────

    private fun makeSessionInfo(id: String): SessionInfo = SessionInfo(
        id = id,
        cwd = state.projectDir ?: "",
        model = state.config?.model ?: "",
        provider = state.config?.provider ?: "",
        startedAt = System.currentTimeMillis() / 1000,
        lastSeen = System.currentTimeMillis() / 1000,
        task = null,
    )

    /** Engine-use timestamps for the idle unload. Touched on every send,
     *  turn event, and boot. */
    @Volatile private var lastEngineUse = android.os.SystemClock.elapsedRealtime()

    /** Unload the mapped model after this much idle time. */
    private val IDLE_UNLOAD_MS = 10 * 60_000L

    private fun touchEngineUse() {
        lastEngineUse = android.os.SystemClock.elapsedRealtime()
    }

    private fun logIdle(msg: String) = android.util.Log.i("sofuu-engine", msg)

    private fun onEngineReady() {
        touchEngineUse()
        refreshSessions()
        engine.setPermissions(state.permissions)
        // A tuned model keeps its measured decode thread count across
        // restarts. A manual decode override in Settings wins. An untuned
        // local model gets the Part-0 protocol (probe + receipts) run
        // automatically, so the measurement does not depend on re-selecting
        // the model in Settings.
        state.config?.model?.takeIf { it.endsWith(".gguf", ignoreCase = true) && it.contains("/files/models/") }?.let { model ->
            if (runtimeEnv.decodeThreads > 0) {
                applyTunedDecode(model)
            } else {
                viewModelScope.launch(Dispatchers.Default) { calibrateDecodeThreads(model) }
            }
        }
        // Prefetch every provider's model list in the background.
        refreshModelCache()
        // The on-device model list is just a directory read — no engine
        // round-trip, so it is correct to do before the first frame.
        refreshLocalModels()
        startIdleUnloadWatch()
    }

    /**
     * Load-on-demand + idle unload. Selecting a model only writes config
     * (a 20 s caps probe reads the header); the weights map in on the FIRST
     * turn. A model that stays mapped holds GBs of KV + weights, so after
     * 10 idle minutes the engine restarts — memory drops back to the
     * process baseline (~tens of MB for the runtime itself) and the next
     * send cold-loads again in a second or two. Chat history survives:
     * sessions are encrypted on disk, in-memory bubbles stay in the store.
     */
    private fun startIdleUnloadWatch() {
        viewModelScope.launch(Dispatchers.Default) {
            while (true) {
                kotlinx.coroutines.delay(60_000)
                val idle = android.os.SystemClock.elapsedRealtime() - lastEngineUse
                if (engine.isReady && !state.streaming && state.approvals.isEmpty() &&
                    idle > IDLE_UNLOAD_MS
                ) {
                    systemRow("Idle ${IDLE_UNLOAD_MS / 60_000} min — unloading the model (memory back to baseline).")
                    logIdle("idle unload: restarting engine (idle=${idle / 1000}s)")
                    engine.shutdown()
                    var restarted = engine.start()
                    if (!restarted) {
                        // The old worker was still exiting; one retry is
                        // enough (shutdown's join is 4 s, and the pump
                        // notices running=false within 250 ms).
                        kotlinx.coroutines.delay(1_000)
                        restarted = engine.start()
                    }
                    if (!restarted) {
                        systemRow("Idle unload deferred — the engine is still shutting down; it will retry next cycle.")
                    }
                    lastEngineUse = android.os.SystemClock.elapsedRealtime()
                }
            }
        }
    }

    /** Coalesced transcript writes.
     *
     *  `systemRow` used to rewrite the whole transcript to two files on the
     *  Main thread for EVERY warning row. Batching is keyed by session id,
     *  so a switch still persists the old chat — only the burst within one
     *  chat is collapsed. */
    private val pendingSaves = LinkedHashMap<String, Pair<String?, List<ChatMessage>>>()
    private var saveJob: kotlinx.coroutines.Job? = null

    fun saveCurrentSessionMessages() {
        val sid = state.activeSessionId ?: return
        val msgs = state.messages
        if (msgs.isEmpty()) return
        val cache = state.messageCache.toMutableMap()
        cache[sid] = msgs
        state = state.copy(messageCache = cache)
        val pDir = state.projectDir ?: engine.projectDir
        synchronized(pendingSaves) { pendingSaves[sid] = pDir to msgs }
        if (saveJob?.isActive == true) return
        saveJob = viewModelScope.launch(Dispatchers.IO) {
            kotlinx.coroutines.delay(250)
            drainPendingSaves()
        }
    }

    /** Write whatever is queued right now (turn end, pause, session switch). */
    fun flushSessionSave() {
        saveJob?.cancel()
        saveJob = null
        if (synchronized(pendingSaves) { pendingSaves.isEmpty() }) return
        viewModelScope.launch(Dispatchers.IO) { drainPendingSaves() }
    }

    private fun drainPendingSaves() {
        while (true) {
            val batch = synchronized(pendingSaves) {
                val copy = pendingSaves.toMap()
                pendingSaves.clear()
                copy
            }
            if (batch.isEmpty()) return
            for ((sid, v) in batch) {
                SessionMessageStore.save(getApplication(), v.first, sid, v.second)
            }
        }
    }

    suspend fun loadSessionMessages(sessionId: String): List<ChatMessage> {
        val mem = state.messageCache[sessionId]
        if (!mem.isNullOrEmpty()) return mem

        val pDir = state.projectDir ?: engine.projectDir
        val fromDisk = withContext(Dispatchers.IO) {
            SessionMessageStore.load(getApplication(), pDir, sessionId)
        }
        if (!fromDisk.isNullOrEmpty()) {
            val cache = state.messageCache.toMutableMap()
            cache[sessionId] = fromDisk
            state = state.copy(messageCache = cache)
            return fromDisk
        }

        val turns = withContext(Dispatchers.Default) {
            engine.sessionTurns(sessionId).get()
        }
        if (turns != null && turns.length() > 0) {
            val msgs = SessionMessageStore.turnsToMessages(turns)
            if (msgs.isNotEmpty()) {
                withContext(Dispatchers.IO) {
                    SessionMessageStore.save(getApplication(), pDir, sessionId, msgs)
                }
                val cache = state.messageCache.toMutableMap()
                cache[sessionId] = msgs
                state = state.copy(messageCache = cache)
                return msgs
            }
        }
        return emptyList()
    }

    fun refreshSessions(resumeNewest: Boolean = false) {
        val ws = state.projectDir ?: engine.projectDir
        viewModelScope.launch {
            val fetched = withContext(Dispatchers.Default) {
                engine.sessions().get()?.let { SessionInfo.fromJsonArray(it) }
            } ?: emptyList()
            // Merge, never clobber: the fetched list may be stale.
            val byId = LinkedHashMap<String, SessionInfo>()
            fetched.forEach { byId[it.id] = it }
            state.sessions.forEach { if (!byId.containsKey(it.id)) byId[it.id] = it }

            // Discover locally saved sessions on disk. getSessionTask parses
            // a whole transcript per id — disk work, so off Main.
            val local = withContext(Dispatchers.IO) {
                SessionMessageStore.getLocalSessionIds(getApplication(), ws).map { id ->
                    val file = File(ws, ".sofuu/sessions/$id")
                    val time = if (file.exists()) file.lastModified() / 1000 else System.currentTimeMillis() / 1000
                    Triple(id, time, SessionMessageStore.getSessionTask(getApplication(), ws, id))
                }
            }
            local.forEach { (id, time, taskName) ->
                if (!byId.containsKey(id)) {
                    byId[id] = SessionInfo(id = id, startedAt = time, lastSeen = time, task = taskName)
                } else if (byId[id]?.task.isNullOrBlank() && !taskName.isNullOrBlank()) {
                    byId[id] = byId[id]!!.copy(task = taskName)
                }
            }

            val list = byId.values.sortedByDescending { it.lastSeen }
            if (state.streaming) {
                state = state.copy(sessions = list)
                return@launch
            }
            val currentActive = state.activeSessionId
            val nextActive =
                if (currentActive != null && list.any { it.id == currentActive }) currentActive
                else if (resumeNewest) list.firstOrNull()?.id
                else null

            if (nextActive == null) {
                state = state.copy(
                    sessions = list,
                    activeSessionId = null,
                    messages = emptyList(),
                )
            } else if (nextActive == currentActive && state.messages.isNotEmpty()) {
                state = state.copy(sessions = list)
            } else {
                saveCurrentSessionMessages()
                val cache = state.messageCache.toMutableMap()
                val ctxCache = state.ctxCache.toMutableMap()
                currentActive?.let { state.ctx?.let { m -> ctxCache[it] = m } }
                val memMessages = cache[nextActive]
                state = state.copy(
                    sessions = list,
                    activeSessionId = nextActive,
                    messages = memMessages ?: emptyList(),
                    approvals = emptyList(),
                    preview = null,
                    messageCache = cache,
                    ctxCache = ctxCache,
                    ctx = ctxCache[nextActive],
                )
                if (memMessages == null || memMessages.isEmpty()) {
                    val loaded = loadSessionMessages(nextActive)
                    if (state.activeSessionId == nextActive) {
                        state = state.copy(messages = loaded)
                    }
                }
            }
            // Resume ONLY when explicitly asked:
            if (resumeNewest && nextActive != null) {
                engine.resumeSession(nextActive)
            }
        }
    }

    fun setActiveSession(id: String?) {
        closePreview()
        if (id == state.activeSessionId) return
        saveCurrentSessionMessages()
        if (state.streaming) {
            val sc = state.ctxCache.toMutableMap()
            state.activeSessionId?.let { state.ctx?.let { m -> sc[it] = m } }
            state = state.copy(activeSessionId = id, ctx = id?.let { sc[it] }, ctxCache = sc)
            return
        }
        val cache = state.messageCache.toMutableMap()
        val ctxCache = state.ctxCache.toMutableMap()
        state.activeSessionId?.let { state.ctx?.let { m -> ctxCache[it] = m } }

        val pDir = state.projectDir ?: engine.projectDir
        val memMessages = id?.let { cache[it] } ?: id?.let { SessionMessageStore.load(getApplication(), pDir, it) }
        if (id != null && memMessages != null) {
            cache[id] = memMessages
        }
        state = state.copy(
            activeSessionId = id,
            messages = memMessages ?: emptyList(),
            approvals = emptyList(),
            preview = null,
            messageCache = cache,
            ctxCache = ctxCache,
            ctx = id?.let { ctxCache[it] },
        )
        if (id != null) {
            viewModelScope.launch {
                engine.resumeSession(id)
                if (memMessages.isNullOrEmpty()) {
                    val loaded = loadSessionMessages(id)
                    withContext(Dispatchers.Main) {
                        if (state.activeSessionId == id) {
                            state = state.copy(messages = loaded)
                        }
                    }
                }
            }
        }
    }

    fun newChat() {
        if (state.sessions.size >= MAX_SESSIONS) {
            systemRow("Chat limit reached ($MAX_SESSIONS per workspace) — delete a chat from History (tap ×) to add a new one.")
            return
        }
        saveCurrentSessionMessages()
        val cache = state.messageCache.toMutableMap()
        state.activeSessionId?.let { cache[it] = state.messages }
        val ctxCache = state.ctxCache.toMutableMap()
        state.activeSessionId?.let { state.ctx?.let { m -> ctxCache[it] = m } }
        state = state.copy(
            activeSessionId = null,
            messages = emptyList(),
            approvals = emptyList(),
            preview = null,
            messageCache = cache,
            ctxCache = ctxCache,
            ctx = null,
        )
    }

    fun deleteSession(id: String) {
        viewModelScope.launch(Dispatchers.Default) {
            val r = engine.deleteSession(id).get()
            if (r?.optBoolean("ok") != true) return@launch
            // Remove the session's files from the ACTIVE project — the
            // engine's boot directory is the wrong one after a switch.
            val pDir = state.projectDir ?: engine.projectDir
            withContext(Dispatchers.IO) {
                val base = File(pDir, ".sofuu/sessions")
                File(base, id).deleteRecursively()
                File(base, "$id.qtsq").delete()
                SessionMessageStore.delete(getApplication(), pDir, id)
            }
            withContext(Dispatchers.Main) { applyDeleteSession(id) }
        }
    }

    private fun applyDeleteSession(id: String) {
        val wasActive = state.activeSessionId == id
        // The on-disk delete already ran on IO in deleteSession().
        val cache = state.messageCache.toMutableMap(); cache.remove(id)
        val ctxCache = state.ctxCache.toMutableMap(); ctxCache.remove(id)
        state = state.copy(
            sessions = state.sessions.filterNot { it.id == id },
            activeSessionId = if (wasActive) null else state.activeSessionId,
            messages = if (wasActive) emptyList() else state.messages,
            preview = if (wasActive) null else state.preview,
            messageCache = cache,
            ctxCache = ctxCache,
            ctx = if (wasActive) null else state.ctx,
        )
    }

    fun clearSessions() {
        viewModelScope.launch(Dispatchers.Default) {
            val r = engine.clearSessions().get()
            if (r?.optBoolean("ok") != true) return@launch
            val ids = r.optJSONArray("ids")
            val pDir = state.projectDir ?: engine.projectDir
            withContext(Dispatchers.IO) {
                val base = File(pDir, ".sofuu/sessions")
                ids?.let { arr ->
                    for (i in 0 until arr.length()) {
                        val sid = arr.optString(i)
                        // Ids are engine-generated; never let a registry
                        // entry turn into a traversal.
                        if (sid.isEmpty() || sid.contains('/') || sid.contains('\\') || sid.startsWith('.')) continue
                        File(base, sid).deleteRecursively()
                        File(base, "$sid.qtsq").delete()
                    }
                }
                SessionMessageStore.deleteAll(getApplication(), pDir)
            }
            withContext(Dispatchers.Main) { applyClearSessions() }
        }
    }

    private fun applyClearSessions() {
        // The on-disk delete already ran on IO in clearSessions().
        val cache = state.messageCache.toMutableMap(); state.sessions.forEach { cache.remove(it.id) }
        val ctxCache = state.ctxCache.toMutableMap(); state.sessions.forEach { ctxCache.remove(it.id) }
        state = state.copy(sessions = emptyList(), activeSessionId = null, messages = emptyList(), approvals = emptyList(), preview = null, messageCache = cache, ctxCache = ctxCache, ctx = null)
    }

    /** Load a past session's transcript for read-only viewing. */
    fun openPreview(session: SessionInfo) {
        viewModelScope.launch {
            val loaded = SessionMessageStore.load(getApplication(), state.projectDir, session.id)
            val turns = if (!loaded.isNullOrEmpty()) {
                val list = ArrayList<PreviewTurn>()
                var curPrompt = ""
                for (m in loaded) {
                    if (m.role == MsgRole.USER) {
                        curPrompt = m.content
                    } else if (m.role == MsgRole.ASSISTANT) {
                        list.add(PreviewTurn(curPrompt, m.content))
                        curPrompt = ""
                    }
                }
                list
            } else {
                withContext(Dispatchers.Default) {
                    engine.sessionTurns(session.id).get()?.let { arr ->
                        (0 until arr.length()).map { i ->
                            val o = arr.optJSONObject(i) ?: return@map PreviewTurn("", "")
                            PreviewTurn(o.optString("prompt"), o.optString("answer"))
                        }
                    }
                } ?: emptyList()
            }
            state = state.copy(preview = SessionPreview(session.id, session.labelOf(state.sessions.indexOf(session).coerceAtLeast(0)), turns))
        }
    }

    fun closePreview() {
        state = state.copy(preview = null)
    }

    // The chat-history sheet lives in AppState, not a local `remember`, so
    // the hardware-back handler in closeTopOverlay() can dismiss it. As a
    // local it was invisible to that path and Back quit the app instead.
    fun openHistory(open: Boolean = true) {
        state = state.copy(historyOpen = open)
    }

    /** Make a provider active (the desktop's per-provider "Use" button):
     *  writes `active` and re-syncs the flat provider/model mirror. */
    fun setActiveProvider(name: String) {
        val entry = state.config?.providers?.firstOrNull { it.name == name } ?: return
        if (state.config?.active == name) return
        patchConfig(
            JSONObject()
                .put("active", name)
                .put("provider", name)
                .put("model", entry.model)
                // Caps inherited from the PREVIOUS model (a local GGUF's
                // 128k window) must not follow us here; 0 is the engine's
                // "resolve from evidence" sentinel.
                .put("ctx_window", 0)
                .put("max_output", 0)
        )
        refreshModelCache()
        reharvestCaps()
    }

    /** Re-discover the active endpoint's caps after a model/provider switch.
     *  Fire-and-forget: the next turn and the ctx ring pick it up. */
    private fun reharvestCaps() {
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { engine.detectCaps().get(20, TimeUnit.SECONDS) }
        }
    }

    /** Edit one provider entry in place (the desktop's patchProvider).
     *  Only the named fields change and the array is merged by name, so the
     *  entry's stored API key survives an edit. */
    fun patchProvider(name: String, model: String? = null, endpoint: String? = null, profile: String? = null) {
        val current = state.config?.providers ?: return
        val providers = current.map { p ->
            if (p.name != name) p.toJson() else p.toJson().apply {
                model?.let { put("model", it.trim()) }
                endpoint?.let { put("endpoint", it.trim()) }
                profile?.let { put("profile", it.trim()) }
            }
        }
        if (providers == current.map { it.toJson() }) return
        val patch = JSONObject().put("providers", JSONArray(providers))
        // chat.js reads the model from the active provider entry, but the
        // flat mirror drives the composer pill — keep it in step.
        if (name == state.config?.active && model != null) patch.put("model", model.trim())
        patchConfig(patch)
        refreshModelCache()
    }

    /** Live chat.state() for the settings engine card. Blocking-safe: it
     *  never runs on the main thread — the caller owns the coroutine. */
    fun engineChatState(): JSONObject? {
        if (!engine.isReady) return null
        // A 2s deadline will expire whenever the engine is mid-turn, which
        // is the normal state while someone is chatting. That is a "no
        // update", not a failure — returning null lets the card keep its
        // last value instead of throwing out of the caller's LaunchedEffect.
        return try {
            engine.chatState().get(2, TimeUnit.SECONDS)
        } catch (e: Exception) {
            null
        }
    }

    /** Vector-embedding manifest (default space + per-space manifest).
     *  Blocking: callers must be off the main thread. */
    fun embedManifest(): JSONObject? {
        val env = engine.embedInfo().get(3, TimeUnit.SECONDS) ?: return null
        if (!env.optBoolean("ok", false)) return null
        return env.optJSONObject("result")
    }

    /** Embed one string locally (no provider, no network). Returns null if
     *  the space or the baked artifact is unavailable. */
    fun embedText(text: String, space: String? = null): FloatArray? =
        engine.embedLocal(text, space).get(5, TimeUnit.SECONDS)

    /**
     * Copy a picked image into app storage and return its path.
     *
     * The multimodal tower opens images by path, so a content:// URI is
     * no use to it — the same reason GGUF imports are copied rather than
     * referenced. Blocking; callers must be off the main thread.
     */
    fun importPickedImage(uri: android.net.Uri): String? {
        val dir = File(getApplication<Application>().filesDir, "images").apply { mkdirs() }
        val stamp = System.currentTimeMillis()
        val dest = File(dir, "img-$stamp.jpg")
        return try {
            getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { out -> input.copyTo(out, 64 * 1024) }
            } ?: return null
            dest.absolutePath
        } catch (e: Exception) {
            dest.delete()
            null
        }
    }

    // ── Workspaces ────────────────────────────────────────────────

    fun openWorkspace(path: String) {
        saveCurrentSessionMessages()
        // Folder change: the strip, pills and transcripts belong to the OLD
        // workspace — reset them so another folder's sessions can never
        // merge into this view.
        val ctxCache = state.ctxCache.toMutableMap()
        state.activeSessionId?.let { state.ctx?.let { m -> ctxCache[it] = m } }
        configs.setProjectDir(path)
        File(path).mkdirs()
        state = state.copy(
            projectDir = path,
            sessions = emptyList(),
            activeSessionId = null,
            messages = emptyList(),
            approvals = emptyList(),
            preview = null,
            ctxCache = ctxCache,
            ctx = null,
            drawerOpen = false,
        )
        viewModelScope.launch { engine.setProject(path) }
        refreshSessions()
    }

    fun addWorkspace(path: String) {
        val list = listOf(path) + state.workspaces.filterNot { it == path }
        configs.setWorkspaces(list)
        state = state.copy(workspaces = list)
        openWorkspace(path)
    }

    fun removeWorkspace(path: String) {
        val list = state.workspaces.filterNot { it == path }
        configs.setWorkspaces(list)
        state = state.copy(workspaces = list)
    }

    // ── Config ────────────────────────────────────────────────────

    fun patchConfig(patch: JSONObject) {
        val merged = configs.updateConfig(patch)
        state = state.copy(config = configs.redacted(merged))
        // Re-apply model-specific settings the UI can change live.
        refreshModelCache()
    }

    fun setProviderKey(provider: String, key: String) {
        configs.setKey(provider, key)
        if (key.isEmpty()) configs.deleteKey(provider)
        // Reflect into the config file so chat.js's per-turn read sees it.
        val providers = state.config?.providers?.map {
            if (it.name == provider) it.toJson().put("api_key", key) else it.toJson()
        } ?: return
        patchConfig(JSONObject().put("providers", JSONArray(providers)))
        refreshModelCache()
    }

    fun removeProvider(name: String): String? = try {
        val merged = configs.removeProvider(name)
        state = state.copy(config = configs.redacted(merged))
        refreshModelCache()
        null
    } catch (e: Exception) {
        e.message
    }

    /** Choose a model: writes it into the provider entry AND the flat
     *  mirror (the desktop pickModel semantics). */
    fun pickModel(provider: String, model: String) {
        val providers = state.config?.providers?.map {
            if (it.name == provider) it.toJson().put("model", model) else it.toJson()
        } ?: return
        val patch = JSONObject()
            .put("providers", JSONArray(providers))
            .put("active", provider)
            .put("provider", provider)
            .put("model", model)
            // A cloud model must not inherit the local one's window/output
            // override. 0 = resolve from evidence.
            .put("ctx_window", 0)
            .put("max_output", 0)
        patchConfig(patch)
        refreshModelCache()
        reharvestCaps()
    }

    // ── Local GGUF models ───────────────────────────────────────

    val localModels: LocalModels by lazy { LocalModels(getApplication()) }

    /**
     * Point the runtime at an on-device GGUF.
     *
     * This is the whole of the local-inference switch, and it mirrors the
     * CLI's `/model x.gguf` branch (`crates/sofuu-core/src/chat.rs:987`).
     * A `.gguf` path forces `provider="local"` and clears `base_url` and
     * `api_key`, because the engine routes on provider: `agent.js` calls
     * `sofuu.ai.stream`, and `rt/ai.rs:4228` sends it to the in-process
     * engine when the provider is Local or the model path ends `.gguf`.
     * Leaving the old endpoint in place would send a filesystem path to an
     * HTTP API.
     *
     * The harvested caps are applied only when the engine reports real
     * numbers — guessing a window for a model we have not opened is how a
     * turn ends in "maximum context length exceeded".
     */
    fun pickLocalModel(path: String) {
        // Default, not Main.immediate: the caps read is an engine round-trip
        // bounded at 20 s and tapping a model on Main froze the UI.
        viewModelScope.launch(Dispatchers.Default) {
            val caps = try {
                engine.localModelCaps(path).get(20, TimeUnit.SECONDS)
            } catch (e: Exception) {
                null
            }
            val patch = JSONObject()
                .put("provider", "local")
                .put("active", "local")
                .put("base_url", "")
                .put("model", path)
            if (caps?.optBoolean("ok") == true) {
                // `caps` is the unwrapped {ok, caps} object; accept a JSON
                // string too so an engine variant cannot silently skip the
                // window again.
                val c = when (val raw = caps.opt("caps")) {
                    is JSONObject -> raw
                    is String -> runCatching { JSONObject(raw) }.getOrNull() ?: JSONObject()
                    else -> JSONObject()
                }
                // Authoritative for this model. 0 = unknown → the engine
                // resolves it; a stale override from the previous model
                // must not survive the switch.
                //
                // The payload is `modelCaps`'s shape — `ctxWindow`/`maxOutput`
                // camelCase and `thinking` a KIND STRING ("none"|"effort"|
                // "budget"), not a boolean. Reading the snake_case/boolean
                // shape silently wrote 0/0 and forced effort off on every
                // local model.
                val win = c.optLong("ctxWindow", c.optLong("ctx_window", 0))
                val maxOut = c.optLong("maxOutput", c.optLong("max_output", 0))
                val thinking = when (val t = c.opt("thinking")) {
                    is Boolean -> t
                    is String -> t.isNotEmpty() && t != "none" && t != "unknown"
                    else -> false
                }
                patch.put("ctx_window", win)
                patch.put("max_output", maxOut)
                // A model with a thinking channel wants effort on; one
                // without it must not be sent "high", which some kernels
                // reject outright.
                patch.put("effort", if (thinking) "high" else "")
            }
            val merged = withContext(Dispatchers.IO) { configs.updateConfig(patch) }
            // No provider entry to write: "local" is not a configured
            // endpoint, so refreshModelCache would only 404 on it.
            withContext(Dispatchers.Main) {
                state = state.copy(config = configs.redacted(merged))
            }
            refreshLocalModels()
            refreshSessions()
            // Auto-tune: apply this model's stored decode thread count, or
            // measure it once on-device. The user's manual decode override
            // takes precedence over the auto result.
            if (!state.streaming) {
                if (runtimeEnv.decodeThreads > 0) {
                    applyTunedDecode(path)
                } else {
                    calibrateDecodeThreads(path)
                }
            }
        }
    }

    // ── Per-model decode auto-tuning ─────────────────────────────
    // Decode is memory-bandwidth-bound and the sweet-spot thread count
    // differs per model size; it cannot be computed from specs, so once
    // per model the app MEASURES it: a few short streaming generations at
    // each candidate count (through the real scheduler, so the phase pool
    // applies live), the highest tok/s wins and is persisted per model.

    private fun tunePrefs() = getApplication<Application>()
        .getSharedPreferences("sofuu_tune", android.content.Context.MODE_PRIVATE)

    private fun tunedDecodeFor(path: String): Int =
        tunePrefs().getInt("decode:" + path, 0)

    private fun applyTunedDecode(path: String) {
        val t = tunedDecodeFor(path)
        if (t > 0) com.sofuu.engine.SofuuBridge.setEnv("SOFUU_DECODE_THREADS", t.toString())
    }

    /** Part 0 receipts: one JSON line per measurement in
     *  files/bench-receipts.jsonl (pull via adb run-as). */
    private fun appendBenchReceipt(
        path: String,
        threads: Int,
        n: Int,
        ms: Long,
        ttft: Long,
        tps: Double,
        phase: String,
    ) {
        try {
            val f = File(getApplication<Application>().filesDir, "bench-receipts.jsonl")
            val rec = org.json.JSONObject()
                .put("ts", System.currentTimeMillis())
                .put("device", android.os.Build.MODEL)
                .put("phase", phase)
                .put("model", path.substringAfterLast('/'))
                .put("threads", threads)
                .put("n", n)
                .put("ms", ms)
                .put("ttft_ms", ttft)
                .put("tps", "%.2f".format(java.util.Locale.US, tps))
            f.appendText(rec.toString() + "\n")
        } catch (_: Throwable) {
        }
    }

    private suspend fun calibrateDecodeThreads(path: String) {
        if (state.streaming || tunedDecodeFor(path) > 0) return
        val cores = Runtime.getRuntime().availableProcessors()
        val candidates = listOf(4, 6, 8).filter { it <= cores }.distinct()
        if (candidates.size <= 1) return
        withContext(Dispatchers.Main) {
            systemRow("Auto-tuning decode threads — the tok/s winner is saved for this model.")
        }
        var best = -1
        var bestTps = 0.0
        var sawFailure = false
        var first = true
        for (t in candidates) {
            if (state.streaming) return  // a real turn takes priority
            withContext(Dispatchers.Main) {
                com.sofuu.engine.SofuuBridge.setEnv("SOFUU_DECODE_THREADS", t.toString())
                state = state.copy(liveMetrics = "tuning: " + t + " threads…")
            }
            val js = "(function(){ return new Promise(function(res){" + "\n" +
                "var t0 = Date.now(), n = 0, done = false, t1 = 0, err = '';" + "\n" +
                "function fin(){ if (!done) { done = true; res(JSON.stringify({n:n, ms:Date.now()-t0, ttft:(t1? (t1-t0) : (Date.now()-t0)), err:err})); } }" + "\n" +
                "var it; try { it = sofuu.ai.stream('Count from 1 to 12.', {model:" + com.sofuu.engine.SofuuEngine.quote(path) + ", maxTokens: 24}); } catch (e) { err = 'sync: ' + String(e && e.message || e); fin(); return; }" + "\n" +
                "(async function(){ try { for await (var c of it) { n++; if (!t1) t1 = Date.now(); } } catch (e) { err = 'stream: ' + String(e && e.message || e); } fin(); })();" + "\n" +
                "setTimeout(fin, 20000);" + "\n" +
                "}); })()"
            // First candidate pays the model load; give it headroom.
            val budget = if (first) 60_000L else 25_000L
            val r = try {
                engine.queryFuture(js, budget).get(budget + 1_000, TimeUnit.MILLISECONDS)
            } catch (e: Exception) {
                null
            }
            val n = r?.optInt("n", 0) ?: 0
            val ms = r?.optLong("ms", 0) ?: 0
            val ttft = r?.optLong("ttft", 0) ?: 0
            val errText = r?.optString("err", "").orEmpty()
            if (r == null || r.has("timeout") || n < 8 || ms <= 0) {
                sawFailure = true
                withContext(Dispatchers.Main) {
                    val why = when {
                        r == null -> "engine did not answer"
                        r.has("timeout") -> "timed out after ${ms}ms"
                        else -> "only $n tokens"
                    }
                    systemRow("tune $t threads: no data ($why)${if (errText.isNotEmpty()) " err=" + errText.take(140) else ""}")
                }
                continue
            }
            first = false
            val tps = n * 1000.0 / ms
            appendBenchReceipt(path, t, n, ms, ttft, tps, "decode-tune")
            withContext(Dispatchers.Main) {
                systemRow("tune $t threads: " + "%.1f".format(java.util.Locale.US, tps) + " tok/s (" + n + " tok / " + ms + "ms, first " + ttft + "ms)")
            }
            if (tps > bestTps) {
                bestTps = tps
                best = t
            }
        }
        if (best > 0 && !state.streaming && state.config?.model == path) {
            withContext(Dispatchers.Main) {
                com.sofuu.engine.SofuuBridge.setEnv("SOFUU_DECODE_THREADS", best.toString())
                systemRow("Auto-tuned: " + best + " threads best (" + "%.1f".format(java.util.Locale.US, bestTps) + " tok/s) — will be used for this model.")
            }
            tunePrefs().edit()
                .putInt("decode:" + path, best)
                .putFloat("tps:" + path, bestTps.toFloat())
                .apply()
            runtimeEnv.decodeThreads = 0  // the per-model value supersedes the global knob
            appendBenchReceipt(path, best, -1, -1, -1, bestTps, "decode-winner")
        } else if (sawFailure && !state.streaming) {
            withContext(Dispatchers.Main) {
                com.sofuu.engine.SofuuBridge.setEnv("SOFUU_DECODE_THREADS", null)
                systemRow("Auto-tune skipped (model busy or too slow to measure) — defaults kept.")
            }
        }
    }

    /** Re-read the on-device GGUF list with display labels derived from
     *  each header (general.name + quant class). Off the main thread — a
     *  4 MB header read per model is disk I/O. Called from engine-ready,
     *  after any import/delete/pick, and when the model sheet opens, since
     *  the private models dir can change underneath the cached state. */
    private fun refreshLocalModels() {
        viewModelScope.launch {
            val (paths, labels) = withContext(Dispatchers.IO) {
                val files = localModels.list()
                files.map { it.absolutePath } to
                    files.associate { it.absolutePath to localModels.label(it) }
            }
            state = state.copy(localModels = paths, localModelLabels = labels)
        }
    }

    /** Import a picked model and select it. Reports the outcome as a
     *  system row so a bad pick is visible in the transcript, not silent. */
    fun importLocalModel(uri: android.net.Uri) {
        // viewModelScope is Dispatchers.Main.immediate, so a blocking .get()
        // here is a main-thread block: a 180 s probe on a slow model is a
        // 180 s frozen UI and an ANR. Only the import touches the disk;
        // the engine query yields.
        viewModelScope.launch {
            val file = withContext(Dispatchers.IO) { localModels.import(uri) }
            if (file == null) {
                systemRow("Could not import that file — it needs to be a readable .gguf model.")
                return@launch
            }
            val sizeMb = file.length() / (1024.0 * 1024.0)
            systemRow("Imported ${localModels.label(file)} (${"%.0f".format(sizeMb)} MB) — checking it…")
            val probe = try {
                withContext(Dispatchers.IO) {
                    engine.probeLocalModel(file.absolutePath).get(180, TimeUnit.SECONDS)
                }
            } catch (e: Exception) {
                null
            }
            if (probe?.optBoolean("ok") != true) {
                systemRow("${localModels.label(file)} did not load: ${probe?.optString("error") ?: "engine timed out"}. The file is still in Settings → Models.")
                return@launch
            }
            pickLocalModel(file.absolutePath)
            systemRow("Model → ${localModels.label(file)} · running on this device, no network.")
        }
    }

    fun deleteLocalModel(path: String) {
        val f = File(path)
        if (localModels.delete(f)) {
            refreshLocalModels()
        }
    }

    /** Automatically create a dedicated directory on the phone for a new workspace. */
    fun createAndAddNewWorkspace(): String {
        val context = getApplication<Application>()
        val baseDir = getWorkspacesBaseDir(context)
        baseDir.mkdirs()

        var num = 1
        var target = File(baseDir, "Workspace $num")
        while (target.exists()) {
            num++
            target = File(baseDir, "Workspace $num")
        }
        target.mkdirs()
        val path = target.absolutePath
        addWorkspace(path)
        return path
    }

    private fun getWorkspacesBaseDir(context: Context): File {
        return try {
            val extDocs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val sofuuDocs = File(extDocs, "Sofuu/Workspaces")
            if (sofuuDocs.exists() || sofuuDocs.mkdirs() || extDocs.canWrite()) {
                sofuuDocs
            } else {
                val extFiles = context.getExternalFilesDir(null)
                File(extFiles ?: context.filesDir, "workspaces")
            }
        } catch (_: Exception) {
            val extFiles = context.getExternalFilesDir(null)
            File(extFiles ?: context.filesDir, "workspaces")
        }
    }

    // ── Model cache (prefetched, never on click) ──────────────────

    fun refreshModelCache() {
        // Off-main: per-provider fetches each hold an engine query; a slow
        // or hung endpoint must never sit on the UI thread.
        viewModelScope.launch(Dispatchers.Default) {
            for (p in state.config?.providers ?: emptyList()) {
                val key = configs.keyFor(p.name).orEmpty()
                val r = engine.fetchProviderModels(p.name, p.endpoint, p.profile, key).get()
                val entry = when {
                    r == null -> ModelCacheEntry(emptyList(), false, "no cache yet", null, false, now())
                    else -> {
                        val models = r.optJSONArray("models")?.let { arr -> (0 until arr.length()).map { arr.optString(it) } } ?: emptyList()
                        ModelCacheEntry(models, r.optBoolean("ok"), r.optString("error").ifEmpty { null }, r.optString("note").ifEmpty { null }, true, now())
                    }
                }
                state = state.copy(modelsCache = state.modelsCache + (p.name to entry))
            }
        }
    }

    private fun now(): Long = System.currentTimeMillis() / 1000

    // ── Approvals / cancel ────────────────────────────────────────

    fun approve(id: String, allow: Boolean, always: Boolean = false) {
        engine.resolveApproval(id, allow, always)
        state = state.copy(approvals = state.approvals.filterNot { it.id == id })
    }

    fun cancel() {
        engine.cancelTurn()
    }

    // ── UI toggles ────────────────────────────────────────────────

    fun setPermissions(p: String) {
        val v = if (p in listOf("full", "edit", "plan", "prompt")) p else "prompt"
        uiPrefs.permissions = v
        state = state.copy(permissions = v)
        viewModelScope.launch { engine.setPermissions(v) }
    }

    fun setMode(m: String) {
        uiPrefs.mode = m
        state = state.copy(mode = m)
    }

    fun setTheme(t: String) {
        uiPrefs.theme = t
        state = state.copy(theme = t)
    }

    fun removePickedImage(index: Int) {
        state = state.copy(pickedImages = state.pickedImages.filterIndexed { i, _ -> i != index })
    }

    fun clearPickedImages() {
        state = state.copy(pickedImages = emptyList())
    }

    /** Files chosen with the system picker: images ride the turn as
     *  attachments, everything else becomes an @mention of the copy the
     *  picker dropped into the workspace (chat.js expands @paths). */
    fun attachPicked(images: List<String>, mentions: List<String>) {
        val patch = JSONObject()
        var next = state
        if (images.isNotEmpty()) next = next.copy(pickedImages = (next.pickedImages + images).take(8))
        if (mentions.isNotEmpty()) {
            val text = mentions.joinToString(" ") { "@$it " }
            next = next.copy(pendingMentions = next.pendingMentions + text)
        }
        state = next
    }

    fun consumeMentions() {
        state = state.copy(pendingMentions = "")
    }

    // ── Voice (provider-backed; nothing is done on-device) ─────────

    val speech: Speech by lazy { Speech(getApplication()) }

    /** True when the active provider could actually serve audio.
     *  Anthropic has no audio API and a local Ollama has no bundled speech
     *  model (ai.rs:4923-4934), so the mic is hidden for those rather than
     *  offered and then failing on every tap. */
    fun voiceCapable(): Boolean {
        val cfg = state.config ?: return false
        val active = cfg.active.ifEmpty { cfg.provider }
        if (active.isEmpty()) return false
        val entry = cfg.providers.firstOrNull { it.name == active } ?: return false
        if (entry.model.isEmpty()) return false
        val profile = entry.profile.lowercase()
        return profile != "anthropic" && profile != "local"
    }

    fun toggleDictation() {
        if (speech.isRecording) {
            state = state.copy(dictating = false)
            viewModelScope.launch(Dispatchers.IO) {
                val b64 = speech.stopRecording()
                if (b64 == null) {
                    withContext(Dispatchers.Main) {
                        systemRow("Dictation cancelled — no audio captured.")
                    }
                    return@launch
                }
                val env = engine.voiceTranscribe(b64, Speech.TRANSCRIBE_OPTS).get(60, TimeUnit.SECONDS)
                val ok = env?.optBoolean("ok", false) == true
                val text = env?.optJSONObject("result")?.optString("text").orEmpty()
                val err = env?.optJSONObject("error")?.optString("message")
                    ?: env?.optString("error").orEmpty()
                withContext(Dispatchers.Main) {
                    when {
                        ok && text.isNotBlank() -> appendToComposer(text)
                        ok -> systemRow("Dictation returned no text.")
                        else -> systemRow("Dictation failed: ${err.ifBlank { "provider error" }}")
                    }
                }
            }
        } else {
            if (!voiceCapable()) {
                systemRow("Voice needs an OpenAI-compatible provider — Anthropic and local models have no audio API.")
                return
            }
            if (speech.startRecording()) state = state.copy(dictating = true)
            else systemRow("Could not start the microphone.")
        }
    }

    fun cancelDictation() {
        if (!speech.isRecording) return
        speech.cancelRecording()
        state = state.copy(dictating = false)
    }

    /** Read an assistant answer aloud. Provider errors surface inline
     *  instead of failing silently. */
    fun speak(text: String) {
        val body = text.trim()
        if (body.isEmpty()) return
        if (!voiceCapable()) {
            systemRow("Read-aloud needs an OpenAI-compatible provider.")
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            val env = engine.voiceSpeak(body, Speech.SPEAK_OPTS).get(90, TimeUnit.SECONDS)
            val ok = env?.optBoolean("ok", false) == true
            val result = env?.optJSONObject("result")
            val bytes = if (ok) Speech.audioBytesFrom(result) else null
            val err = env?.optJSONObject("error")?.optString("message")
                ?: env?.optString("error").orEmpty()
            withContext(Dispatchers.Main) {
                when {
                    bytes != null -> speech.play(bytes, result?.optString("format").orEmpty())
                    ok -> systemRow("Read-aloud returned no audio.")
                    else -> systemRow("Read-aloud failed: ${err.ifBlank { "provider error" }}")
                }
            }
        }
    }

    fun stopSpeaking() = speech.stopPlayback()

    /** Dictation/attachments land in the composer through the same
     *  pendingMentions splice the file picker uses. */
    private fun appendToComposer(text: String) {
        val cur = state.pendingMentions
        state = state.copy(pendingMentions = if (cur.isEmpty()) text else "$cur $text")
    }

    /** Post a system notification (approvals + finished turns) when the
     *  app is in the background. The activity installs the notifier. */
    var notifier: ((String, String) -> Unit)? = null

    /** True while the UI is foregrounded (the activity flips it). */
    @Volatile var foreground: Boolean = true

    fun setCloudBg(on: Boolean) {
        uiPrefs.cloudBg = on
        state = state.copy(cloudBg = on)
    }

    fun setChatFontSize(s: Int) {
        uiPrefs.chatFontSize = s.coerceIn(13, 22)
        state = state.copy(chatFontSize = uiPrefs.chatFontSize)
    }

    fun openSettings(open: Boolean, tab: String = "general") {
        state = state.copy(settingsOpen = open, settingsTab = tab)
    }

    fun setDrawer(open: Boolean) {
        state = state.copy(drawerOpen = open)
    }

    fun openSheet(which: String) {
        // The model sheet must never show a stale list — the models dir can
        // change outside the app entirely (adb, file manager, a re-import).
        if (which == "model") refreshLocalModels()
        state = state.copy(sheet = which)
    }

    fun closeSheet() {
        state = state.copy(sheet = null)
    }

    /** Hardware back: dismiss the top-most overlay, true if it consumed
     *  the press (drawer → sheet → preview → settings). */
    fun closeTopOverlay(): Boolean {
        val s = state
        return when {
            s.drawerOpen -> { state = s.copy(drawerOpen = false); true }
            s.historyOpen -> { state = s.copy(historyOpen = false); true }
            s.sheet != null -> { state = s.copy(sheet = null); true }
            s.preview != null -> { state = s.copy(preview = null); true }
            s.settingsOpen -> { state = s.copy(settingsOpen = false); true }
            else -> false
        }
    }

    fun resolvedDark(systemDark: Boolean): Boolean =
        state.theme == "dark" || (state.theme == "system" && systemDark)

    override fun onCleared() {
        super.onCleared()
        engine.shutdown()
    }
}
