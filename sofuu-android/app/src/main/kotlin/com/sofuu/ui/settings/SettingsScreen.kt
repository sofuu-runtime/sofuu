package com.sofuu.ui.settings

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Switch
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.concurrent.TimeUnit
import com.sofuu.store.AppStore
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.theme.LocalSofuuColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

private val TABS = listOf("general", "appearance", "models", "providers", "agents", "brain", "usage")

/** Full-screen settings — the same panes as the desktop modal, minus
 *  Shortcuts (deliberately dropped on mobile). */
@Composable
fun SettingsScreen(store: AppStore, onClose: () -> Unit) {
    val c = LocalSofuuColors.current
    var tab by remember { mutableStateOf(store.state.settingsTab) }
    // External tab requests (ModelSheet → "＋ Add provider", composer rows)
    // fire while Settings is already composed — a remember-once initial
    // value would silently ignore them. Local tab taps never write the
    // store, so this only fires on external change.
    LaunchedEffect(store.state.settingsTab) { tab = store.state.settingsTab }
    Box(Modifier.fillMaxSize().background(c.bgApp).statusBarsPadding()) {
        Column(Modifier.fillMaxSize()) {
            // Header + tab bar.
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Settings", style = MaterialTheme.typography.titleLarge, color = c.text, modifier = Modifier.weight(1f))
                Icon(Icons2.Close, "Close", tint = c.textMuted, modifier = Modifier.size(36.dp).clickable { onClose() }.padding(8.dp))
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                TABS.forEach { t ->
                    val active = t == tab
                    Text(
                        t.replaceFirstChar { it.uppercase() },
                        fontSize = 13.sp,
                        color = if (active) c.accentContrast else c.textMuted,
                        modifier = Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(if (active) c.accent else c.bgCard)
                            .clickable { tab = t }
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
                when (tab) {
                    "general" -> GeneralPane(store)
                    "appearance" -> AppearancePane(store)
                    "models" -> ModelsPane(store)
                    "providers" -> ProvidersPane(store)
                    "agents" -> AgentsPane(store)
                    "brain" -> BrainPane(store)
                    "usage" -> UsagePane(store)
                }
            }
        }
    }
}

@Composable
private fun Section(label: String) {
    Text(label, fontSize = 12.sp, color = LocalSofuuColors.current.textFaint, modifier = Modifier.padding(top = 22.dp, bottom = 6.dp))
}

@Composable
private fun Row_(label: String, desc: String?, content: @Composable () -> Unit) {
    val c = LocalSofuuColors.current
    Column(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Text(label, fontSize = 14.sp, color = c.text)
        if (desc != null) Text(desc, fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(bottom = 6.dp))
        content()
    }
}

@Composable
private fun Segmented(options: List<String>, selected: String, onSelect: (String) -> Unit) {
    val c = LocalSofuuColors.current
    Row(Modifier.clip(RoundedCornerShape(8.dp)).background(c.bgCard).padding(3.dp), horizontalArrangement = Arrangement.spacedBy(3.dp)) {
        options.forEach { o ->
            val active = o == selected
            Text(
                o.replaceFirstChar { it.uppercase() },
                fontSize = 12.sp,
                color = if (active) c.text else c.textMuted,
                modifier = Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .background(if (active) c.bgRaised else Color.Transparent)
                    .clickable { onSelect(o) }
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun TextFieldRow(
    label: String,
    value: String,
    placeholder: String,
    onChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = LocalSofuuColors.current
    // Local echo keeps the cursor stable while the caller's value makes a
    // round trip; every edit is still pushed to [onChange]. (This used to
    // type into local state and never call back at all, so every settings
    // field — including the provider form — read as empty.)
    var text by remember { mutableStateOf(value) }
    LaunchedEffect(value) { text = value }
    androidx.compose.foundation.text.BasicTextField(
        value = text,
        onValueChange = {
            text = it
            onChange(it)
        },
        textStyle = androidx.compose.ui.text.TextStyle(color = c.text, fontFamily = FontFamily.Monospace, fontSize = 13.sp),
        cursorBrush = androidx.compose.ui.graphics.SolidColor(c.accent),
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 44.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(c.monoFieldBg)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        decorationBox = { inner ->
            Box {
                if (text.isEmpty()) Text(placeholder, color = c.textFaint, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
                inner()
            }
        },
    )
}

/**
 * Text field whose value reaches [onApply] only when the user taps Apply.
 * For knobs that reboot the engine: applying per keystroke would restart
 * it while a multi-digit number is still being typed.
 */
@Composable
private fun AppliedFieldRow(
    label: String,
    value: String,
    placeholder: String,
    onApply: (String) -> Unit,
) {
    val c = LocalSofuuColors.current
    var text by remember { mutableStateOf(value) }
    LaunchedEffect(value) { text = value }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.weight(1f)) {
            TextFieldRow(label, text, placeholder, { text = it })
        }
        Text(
            "Apply",
            fontSize = 12.sp,
            color = c.accentContrast,
            modifier = Modifier
                .clip(RoundedCornerShape(999.dp))
                .background(c.accent)
                .clickable { onApply(text) }
                .padding(horizontal = 14.dp, vertical = 8.dp),
        )
    }
}

/** One key/value tile of the live engine card. */
@Composable
private fun EngineCell(key: String, value: String, mono: Boolean = false) {
    val c = LocalSofuuColors.current
    Row(Modifier.fillMaxWidth().padding(vertical = 5.dp), verticalAlignment = Alignment.Top) {
        Text(key, fontSize = 11.sp, color = c.textFaint, modifier = Modifier.width(96.dp))
        Text(
            value,
            fontSize = 12.sp,
            color = c.text,
            fontFamily = if (mono) FontFamily.Monospace else FontFamily.Default,
        )
    }
}

/**
 * The native vector engine (H-E1/M1) as a diagnostic card.
 *
 * This is deliberately a *manifest + verify box*, not a search product:
 * the four exported entry points exist and now work from Kotlin, and the
 * thing worth proving on-device is that the default space (`hash-768`)
 * is the same space the brain stores in — so an embedded string can be
 * remembered without a space argument.
 */
@Composable
private fun VectorEngineCard(store: AppStore) {
    val c = LocalSofuuColors.current
    var manifest by remember { mutableStateOf<JSONObject?>(null) }
    var probe by remember { mutableStateOf("Hello, Sofuu") }
    var probeResult by remember { mutableStateOf<String?>(null) }
    var probing by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        manifest = withContext(Dispatchers.IO) { runCatching { store.embedManifest() }.getOrNull() }
    }

    Section("Vector engine")
    if (manifest == null) {
        Text("Loading the embedding manifest…", fontSize = 12.sp, color = c.textFaint)
    } else {
        val m = manifest!!
        EngineCell(
            "Default space",
            m.optString("default_space").ifEmpty { "—" } + "  (the brain's space)",
            mono = true,
        )
        val spaces = m.optJSONArray("spaces")
        if (spaces != null) {
            for (i in 0 until spaces.length()) {
                val s = spaces.optJSONObject(i) ?: continue
                val id = s.optString("space")
                // The hash space is inlined in the manifest; the learned +
                // image spaces nest a `model` object. Read both shapes.
                val model = s.optJSONObject("model")
                val dim = s.optInt("dimension").takeIf { it > 0 }
                    ?: model?.optInt("dimension")?.takeIf { it > 0 }
                    ?: model?.optInt("dim")?.takeIf { it > 0 }
                    ?: 0
                val avail = s.optBoolean("available", true) &&
                    (model == null || model.optBoolean("available", true))
                val modelId = s.optString("id").ifEmpty { model?.optString("id").orEmpty() }
                Text(
                    "  $id" + if (dim > 0) "  ·  ${dim}-dim" else "" +
                        if (modelId.isNotEmpty()) "  ·  $modelId" else "",
                    fontSize = 11.sp,
                    color = if (avail) c.textMuted else c.err,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(top = 3.dp),
                )
                if (!avail) {
                    Text(
                        "     artifact unavailable",
                        fontSize = 10.sp,
                        color = c.err,
                        modifier = Modifier.padding(top = 1.dp),
                    )
                }
            }
        }
        Spacer(Modifier.height(10.dp))
        TextFieldRow("probe", probe, "text to embed locally", { probe = it })
        Text(
            if (probing) "Embedding…" else "Embed",
            fontSize = 12.sp,
            color = c.accentContrast,
            modifier = Modifier
                .padding(top = 8.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(if (probing) c.bgCardHover else c.accent)
                .clickable(enabled = !probing && probe.isNotBlank()) {
                    probing = true
                    probeResult = null
                }
                .padding(horizontal = 16.dp, vertical = 8.dp),
        )
        val res = probeResult
        if (res != null) {
            Text(res, fontSize = 11.sp, color = c.textMuted, fontFamily = FontFamily.Monospace, modifier = Modifier.padding(top = 8.dp))
        }
        // Launch the probe out here so the click handler stays cheap.
        LaunchedEffect(probing, probe) {
            if (!probing) return@LaunchedEffect
            val text = probe
            probeResult = withContext(Dispatchers.IO) {
                val v = runCatching { store.embedText(text) }.getOrNull()
                when {
                    v == null -> "Embed failed — space unknown or the baked artifact is unavailable."
                    else -> {
                        var sum = 0.0
                        for (x in v) sum += (x * x).toDouble()
                        val norm = Math.sqrt(sum)
                        val head = v.take(6).joinToString(" ") { String.format("%.4f", it) }
                        "dim ${v.size}  ·  ‖v‖ ${String.format("%.4f", norm)}\n$head …"
                    }
                }
            }
            probing = false
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Embeddings run entirely on-device — no provider, no network, no API key.",
            fontSize = 11.sp,
            color = c.textFaint,
        )
    }
}

// ── General ─────────────────────────────────────────────────────

@Composable
private fun GeneralPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    // Live engine card (desktop parity): chat.state() polled while the
    // pane is open, so Model/Provider/Session/History/Context/Spend are
    // real rather than a guess from config.json.
    var live by remember { mutableStateOf<JSONObject?>(null) }
    LaunchedEffect(Unit) {
        while (true) {
            // chat.state() is a blocking engine query — keep it off the
            // main thread or the card stalls the UI (and risks an ANR).
            live = withContext(Dispatchers.IO) { store.engineChatState() }
            kotlinx.coroutines.delay(2000)
        }
    }
    Column(Modifier.verticalScroll(rememberScrollState())) {
        Section("Engine")
        if (live == null) {
            Text("Connecting to the engine…", fontSize = 12.sp, color = c.textFaint)
        } else {
            val e = live!!
            EngineCell("Model", e.optString("model").ifEmpty { "None" }, mono = true)
            EngineCell("Provider", e.optString("provider").ifEmpty { "None" })
            EngineCell("Session", e.optString("sessionId").let { if (it.isEmpty()) "None yet" else it.take(10) + "…" }, mono = true)
            EngineCell("History", "${e.optInt("historyTurns")} turns")
            EngineCell("Context", "${e.optInt("contextTokens")} tk")
            EngineCell("Session spend", "$" + "%.4f".format(e.optDouble("sessionSpendUsd")))
            if (e.optBoolean("streaming") || e.optInt("pendingApprovals") > 0) {
                Text(
                    buildString {
                        if (e.optBoolean("streaming")) append("Generating…")
                        val n = e.optInt("pendingApprovals")
                        if (n > 0) { if (isNotEmpty()) append(" · "); append("$n awaiting approval") }
                    },
                    fontSize = 11.sp,
                    color = c.accent,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
        }
        Row_("Project folder", "Workspace root and per-project storage (.sofuu/ — sessions, memory, settings).", {
            Text(state.projectDir ?: "(none)", fontSize = 12.sp, color = c.textMuted, fontFamily = FontFamily.Monospace)
        })
        Row_("Tool permission profile", "Applied live in the engine and re-applied on launch.", {
            Segmented(listOf("full", "edit", "plan", "prompt"), state.permissions) { store.setPermissions(it) }
        })
        Row_("Memory (brain)", "Recall saved project memories into each turn (.sofuu/brain/).", {
            Segmented(listOf("on", "off"), if (state.config?.brain != false) "on" else "off") {
                store.patchConfig(JSONObject().put("brain", it == "on"))
            }
        })
        Row_("Smart context filters", "On-device neural filters (freshness, compaction, relevance, allocation).", {
            Segmented(listOf("on", "off"), if (state.config?.ml != false) "on" else "off") {
                store.patchConfig(JSONObject().put("ml", it == "on"))
            }
        })
        Row_("RLM sandbox", "Route very long-context turns through the local reasoning sandbox.", {
            Segmented(listOf("off", "on", "auto"), state.config?.rlm ?: "off") {
                store.patchConfig(JSONObject().put("rlm", it))
            }
        })
        Row_("Context window", "Override the model's window (0 = engine-resolved truth).", {
            AppliedFieldRow("ctx", state.config?.ctxWindow?.toString().orEmpty(), "auto", { v ->
                v.trim().toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("ctx_window", n)) }
            })
        })
        Row_("Max output tokens", "Override the per-request output cap (0 = model default).", {
            AppliedFieldRow("maxout", state.config?.maxOutput?.toString().orEmpty(), "model default", { v ->
                v.trim().toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("max_output", n)) }
            })
        })
        Row_("Session budget (USD)", "The engine refuses new turns past the cap (0 = uncapped).", {
            AppliedFieldRow("budget", state.config?.budgetUsd?.takeIf { it > 0 }?.toString().orEmpty(), "uncapped", { v ->
                v.trim().toDoubleOrNull()?.let { n -> store.patchConfig(JSONObject().put("budget_usd", n)) }
            })
        })
        Section("Memory & recall")
        Text(
            "How aggressively saved project memories are pulled into a turn. " +
                "Leave at 0 to use the engine defaults.",
            fontSize = 11.sp,
            color = c.textFaint,
            modifier = Modifier.padding(bottom = 4.dp),
        )
        Row_("Recall similarity floor", "Minimum match score for a memory to be recalled (0 = default 0.30).", {
            AppliedFieldRow("recall_min", state.config?.recallMin?.takeIf { it > 0 }?.toString().orEmpty(), "default 0.30", { v ->
                v.trim().toDoubleOrNull()?.let { n ->
                    store.patchConfig(JSONObject().put("recall_min", n.coerceIn(0.0, 1.0)))
                }
            })
        })
        Row_("Recall token budget", "Max tokens of memories injected per turn (0 = auto ≈2% of window, 1k–16k).", {
            AppliedFieldRow("recall_budget", state.config?.recallBudget?.takeIf { it > 0 }?.toString().orEmpty(), "auto", { v ->
                v.trim().toLongOrNull()?.let { n ->
                    store.patchConfig(JSONObject().put("recall_budget", n.coerceIn(0L, 16_000L)))
                }
            })
        })
        Row_("Embeddings — provider", "Optional OpenAI-compatible provider for memory embeddings. Blank = the bundled offline embedder.", {
            AppliedFieldRow("embed_provider", state.config?.embedProvider.orEmpty(), "bundled offline embedder", { v ->
                store.patchConfig(JSONObject().put("embed_provider", v.trim()))
            })
        })
        Row_("Embeddings — model", "Embedding model id used for memory vectors.", {
            AppliedFieldRow("embed_model", state.config?.embedModel.orEmpty(), "default", { v ->
                store.patchConfig(JSONObject().put("embed_model", v.trim()))
            })
        })
        VectorEngineCard(store)
        PhoneToolsCard(store)
        EngineEnvCard(store)
        Spacer(Modifier.height(24.dp))
    }
}

// ── Phone tools + engine knobs ──────────────────────────────────

/**
 * What the phone adds over a terminal.
 *
 * Host tools are registered on the engine thread at boot and appear in
 * /tools alongside the built-ins; listing them here is the only place a
 * user can see what the device can do that a CLI cannot.
 */
@Composable
private fun PhoneToolsCard(store: AppStore) {
    val c = LocalSofuuColors.current
    var tools by remember { mutableStateOf<List<String>>(emptyList()) }
    LaunchedEffect(Unit) {
        // The engine fills the list during boot, which may not be done the
        // first time this pane is drawn.
        while (true) {
            tools = store.engine.hostTools.toList()
            delay(500)
        }
    }

    Section("Phone tools")
    Text(
        "Native capabilities the agent can call. Registered on the engine at " +
            "boot, so they obey the permission profile and show up in /tools.",
        fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(bottom = 6.dp),
    )
    if (tools.isEmpty()) {
        Text("None registered yet.", fontSize = 12.sp, color = c.textFaint)
    } else {
        for (t in tools) {
            Text(
                "  ·  $t",
                fontSize = 12.sp, color = c.text,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(vertical = 2.dp),
            )
        }
    }
}

/**
 * The library's env-var switches.
 *
 * These are plain process configuration the engine reads through
 * std::env::var — there is no ABI call for them. Most are consulted per
 * turn, so a change applies to the next message; SOFUU_NO_ML gates module
 * registration and so needs a restart, which the UI says outright rather
 * than letting a user wonder why nothing moved.
 */
@Composable
private fun EngineEnvCard(store: AppStore) {
    val c = LocalSofuuColors.current
    val env = store.runtimeEnv

    Section("Engine controls")
    Text(
        "These are the runtime's own environment knobs, set before the engine " +
            "boots. Most take effect on the next message.",
        fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(bottom = 6.dp),
    )

    var noTools by remember { mutableStateOf(env.noTools) }
    var noShell by remember { mutableStateOf(env.noShell) }
    var noMl by remember { mutableStateOf(env.noMl) }
    var allowInternal by remember { mutableStateOf(env.webAllowInternal) }
    var backend by remember { mutableStateOf(env.memoryBackend) }
    var engine by remember { mutableStateOf(env.webEngine) }
    var phaseMetrics by remember { mutableStateOf(env.phaseMetrics) }

    ToggleRow("Coding tools", "All 8 built-ins (read/write/edit/grep/glob/dir/bash/todo) off. Web and MCP tools still work.", noTools)
    { noTools = it; env.noTools = it; store.applyRuntimeEnv() }
    ToggleRow("Shell", "Keep bash listed but refuse every invocation. Does not stop an MCP server that shells out itself.", noShell)
    { noShell = it; env.noShell = it; store.applyRuntimeEnv() }
    ToggleRow("Context gates (ML)", "Drop the whole sofuu.ml gate layer. This one needs a restart — it is decided when the engine loads.", noMl)
    { noMl = it; env.noMl = it; store.restartEngineForEnv() }
    ToggleRow("Allow private addresses", "Let web.open reach loopback and LAN addresses. The SSRF guard blocks them otherwise.", allowInternal)
    // M4: these three only wrote the SharedPreference, so the process
    // environment — and therefore the engine — kept the old value until
    // an unrelated restart. Every knob below re-applies.
    { allowInternal = it; env.webAllowInternal = it; store.applyRuntimeEnv() }

    Row_("Memory backend", "Auto uses the semantic projector when its artifact is present, else hash-768.") {
        Segmented(listOf("auto", "hash", "semantic"), backend.ifEmpty { "auto" }) {
            backend = it; env.memoryBackend = if (it == "auto") "" else it
            store.applyRuntimeEnv()
        }
    }
    Row_("Web search engine", "Default engine for web_search. Blank picks the keyless one.") {
        Segmented(listOf("auto", "brave", "tavily"), engine.ifEmpty { "auto" }) {
            engine = it; env.webEngine = if (it == "auto") "" else it
            store.applyRuntimeEnv()
        }
    }
    Row_("Turn step budget", "Tool rounds allowed per turn. 0 uses the engine default (200).") {
        TextFieldRow(
            "maxsteps",
            if (env.chatMaxSteps > 0) env.chatMaxSteps.toString() else "",
            "200 (engine default)",
            onChange = {
                val n = it.filter { c -> c.isDigit() }.toIntOrNull() ?: 0
                env.chatMaxSteps = n
                store.applyRuntimeEnv()
            },
        )
    }
    Row_("Stalled-request timeout", "Seconds before an unresponsive endpoint gives up. 0 uses the engine default (300).") {
        TextFieldRow(
            "stall",
            if (env.stallTimeoutSecs > 0) env.stallTimeoutSecs.toString() else "",
            "300 (engine default)",
            onChange = {
                val n = it.filter { c -> c.isDigit() }.toIntOrNull() ?: 0
                env.stallTimeoutSecs = n
                store.applyRuntimeEnv()
            },
        )
    }
    Row_("CPU threads for inference", "Kernel-pool threads decoding locally. 0 = auto (4 on this class of phone). Apply restarts the engine — watch the tok/s line to compare.") {
        AppliedFieldRow(
            "threads",
            if (env.numThreads > 0) env.numThreads.toString() else "",
            "auto (4)",
            onApply = {
                env.numThreads = it.filter { c -> c.isDigit() }.toIntOrNull() ?: 0
                store.restartEngineForEnv()
            },
        )
    }
    Row_("Prefill threads", "Threads for the batched prompt pass — compute-bound, scales with the big cluster. 0 = auto (8). Apply restarts; TTFT is the scoreboard.") {
        AppliedFieldRow(
            "prefill",
            if (env.prefillThreads > 0) env.prefillThreads.toString() else "",
            "auto (8)",
            onApply = {
                env.prefillThreads = it.filter { c -> c.isDigit() }.toIntOrNull() ?: 0
                store.restartEngineForEnv()
            },
        )
    }
    Row_("Decode threads", "Threads for token generation — bandwidth-bound, tune for the tok/s readout. 0 = auto (4). Apply restarts the engine.") {
        AppliedFieldRow(
            "decode",
            if (env.decodeThreads > 0) env.decodeThreads.toString() else "",
            "auto (4)",
            onApply = {
                env.decodeThreads = it.filter { c -> c.isDigit() }.toIntOrNull() ?: 0
                store.restartEngineForEnv()
            },
        )
    }
    ToggleRow("Phase metrics", "Write the per-token engine split (forward / sampler / glue) to files/phase-metrics.log every 256 tokens. Restart to apply. Readable over adb.", phaseMetrics)
    { phaseMetrics = it; env.phaseMetrics = it; store.restartEngineForEnv() }
    Text(
        "Context gates and CPU threads need a restart; everything else is read while a turn " +
            "runs, so it applies to the next message.",
        fontSize = 10.sp, color = c.textFaint, modifier = Modifier.padding(top = 6.dp),
    )
}

@Composable
private fun ToggleRow(label: String, desc: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier.fillMaxWidth().heightIn(min = 44.dp).clickable { onChange(!checked) }
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, fontSize = 14.sp, color = c.text)
            Text(desc, fontSize = 11.sp, color = c.textFaint)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

// ── Appearance ──────────────────────────────────────────────────

@Composable
private fun AppearancePane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    Column(Modifier.verticalScroll(rememberScrollState())) {
        Row_("Theme", "Follows the system unless you pick one.", {
            Segmented(listOf("light", "dark", "system"), state.theme) { store.setTheme(it) }
        })
        Row_("Cloud background", "Procedural gradient backdrop.", {
            Segmented(listOf("on", "off"), if (state.cloudBg) "on" else "off") { store.setCloudBg(it == "on") }
        })
        Row_("Chat font size", "Body text size in the chat pane.", {
            Segmented(listOf("13", "15", "17", "19"), state.chatFontSize.toString()) { store.setChatFontSize(it.toInt()) }
        })
        Spacer(Modifier.height(24.dp))
    }
}

// ── Models ──────────────────────────────────────────────────────

@Composable
private fun ModelsPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    Column(Modifier.verticalScroll(rememberScrollState())) {

        // ── On-device models ─────────────────────────────────────
        // Above the providers: a phone running its own model needs no
        // endpoint at all, and it is the only option that works offline.
        val pickModelFile = rememberLauncherForActivityResult(
            ActivityResultContracts.OpenDocument()
        ) { uri -> if (uri != null) store.importLocalModel(uri) }

        Section("On this device")
        if (state.localModels.isEmpty()) {
            Text(
                "No models installed. Import a .gguf file — it runs entirely on the phone, no network and no API key.",
                fontSize = 12.sp, color = c.textFaint, modifier = Modifier.padding(bottom = 8.dp),
            )
        } else {
            for (path in state.localModels) {
                // Header-derived label ("Llama 3.2 1B Instruct · Q8_0"),
                // identical to the composer pill; file-name fallback.
                val label = state.localModelLabels[path] ?: com.sofuu.store.fallbackModelLabel(path)
                val active = state.config?.model == path
                Row(
                    Modifier.fillMaxWidth().heightIn(min = 44.dp).padding(vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f).clickable { store.pickLocalModel(path) }) {
                        Text(
                            label,
                            fontSize = 13.sp,
                            color = if (active) c.accent else c.text,
                            fontFamily = FontFamily.Monospace,
                        )
                        if (active) {
                            Text("in use — runs locally", fontSize = 11.sp, color = c.textFaint)
                        }
                    }
                    if (!active) {
                        Icon(
                            Icons2.Close, "Remove", tint = c.textFaint,
                            modifier = Modifier.size(16.dp).clickable { store.deleteLocalModel(path) },
                        )
                    }
                }
            }
            if (state.config?.model?.endsWith(".gguf") == true) {
                Text(
                    "Switching back to a hosted provider: pick one below.",
                    fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
        Text(
            "Import a model…",
            fontSize = 13.sp,
            color = c.accentContrast,
            modifier = Modifier
                .padding(top = 8.dp, bottom = 4.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(c.accent)
                .clickable {
                    // .gguf has no registered MIME type on most devices,
                    // so an explicit allow-list beats a wildcard.
                    pickModelFile.launch(arrayOf("*/*"))
                }
                .padding(horizontal = 16.dp, vertical = 10.dp),
        )

        Spacer(Modifier.height(18.dp))
        Section("Hosted providers")
        val providers = state.config?.providers ?: emptyList()
        providers.forEach { p ->
            Row_("Provider: ${p.name}", p.endpoint, {
                val models = (state.modelsCache[p.name]?.models ?: emptyList())
                if (models.isEmpty()) {
                    Text("No model list yet — tap refresh.", fontSize = 12.sp, color = c.textFaint)
                } else {
                    models.take(30).forEach { m ->
                        Row(
                            Modifier.fillMaxWidth().clickable { store.pickModel(p.name, m) }.heightIn(min = 44.dp).padding(vertical = 8.dp),
                        ) {
                            Text(m, fontSize = 13.sp, color = if (m == p.model) c.accent else c.text, fontFamily = FontFamily.Monospace)
                        }
                    }
                }
            })
        }
        Text(
            "Refresh",
            fontSize = 13.sp,
            color = c.accentContrast,
            modifier = Modifier
                .padding(top = 12.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(c.accent)
                .clickable { store.refreshModelCache() }
                .padding(horizontal = 16.dp, vertical = 10.dp),
        )
        Spacer(Modifier.height(24.dp))
    }
}

// ── Providers ───────────────────────────────────────────────────

@Composable
private fun ProvidersPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    var newName by remember { mutableStateOf("") }
    var newEndpoint by remember { mutableStateOf("") }
    var newModel by remember { mutableStateOf("") }
    var newKey by remember { mutableStateOf("") }
    var addError by remember { mutableStateOf<String?>(null) }
    var addNote by remember { mutableStateOf<String?>(null) }
    var editing by remember { mutableStateOf(emptySet<String>()) }

    Column(Modifier.verticalScroll(rememberScrollState())) {
        Section("Add provider")
        val fieldGap = Modifier.padding(bottom = 8.dp)
        TextFieldRow("name", newName, "name (e.g. openai)", { newName = it }, fieldGap)
        TextFieldRow("endpoint", newEndpoint, "https://…/v1/chat/completions", { newEndpoint = it }, fieldGap)
        TextFieldRow("model", newModel, "model id", { newModel = it }, fieldGap)
        TextFieldRow("key", newKey, "API key (stored in the Android Keystore)", { newKey = it }, fieldGap)
        Text(
            "Add",
            fontSize = 13.sp,
            color = c.accentContrast,
            modifier = Modifier
                .padding(top = 10.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(c.accent)
                .clickable {
                    val name = newName.trim()
                    val endpoint = newEndpoint.trim()
                    val model = newModel.trim()
                    addNote = null
                    if (name.isEmpty() || endpoint.isEmpty()) {
                        // The button used to fail silently here, which read
                        // as "Add does nothing".
                        addError = "Name and endpoint are required."
                    } else {
                        val providers = (state.config?.providers ?: emptyList()).map { it.toJson() }.toMutableList()
                        providers.add(
                            JSONObject().put("name", name).put("endpoint", endpoint)
                                .put("model", model).put("profile", ""),
                        )
                        val patch = JSONObject()
                            .put("providers", org.json.JSONArray(providers))
                        if (model.isNotEmpty()) {
                            // chat.js resolves the model from the active
                            // provider entry, but the flat mirror drives the
                            // composer pill — keep both in step.
                            patch.put("active", name).put("provider", name).put("model", model)
                            addNote = null
                        } else {
                            // Switching to an entry with no model would keep
                            // sending the previous provider's model id to the
                            // new endpoint. Add it, leave it inactive, and say
                            // how to finish the switch.
                            addNote = "Added $name — open the model sheet in the composer, pick one of its models, and it becomes active."
                        }
                        store.patchConfig(patch)
                        if (newKey.isNotBlank()) store.setProviderKey(name, newKey.trim())
                        newName = ""; newEndpoint = ""; newModel = ""; newKey = ""
                        addError = null
                    }
                }
                .padding(horizontal = 18.dp, vertical = 10.dp),
        )
        addError?.let {
            Text(it, fontSize = 11.sp, color = c.errStrong, modifier = Modifier.padding(top = 6.dp))
        }
        addNote?.let {
            Text(it, fontSize = 11.sp, color = c.ok, modifier = Modifier.padding(top = 6.dp))
        }

        Section("Configured providers")
        for (p in state.config?.providers ?: emptyList()) {
            Column(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(p.name, fontSize = 14.sp, color = c.text)
                        Text(p.endpoint, fontSize = 11.sp, color = c.textFaint, maxLines = 1)
                        Text(
                            if (p.hasApiKey) "key saved · Keystore" else "no key",
                            fontSize = 11.sp,
                            color = if (p.hasApiKey) c.ok else c.warn,
                        )
                    }
                    if (p.name != state.config?.active) {
                        Text(
                            "Remove",
                            fontSize = 12.sp,
                            color = c.errStrong,
                            modifier = Modifier.clickable { store.removeProvider(p.name) }.padding(8.dp),
                        )
                    } else {
                        Text("active", fontSize = 11.sp, color = c.accent, modifier = Modifier.padding(8.dp))
                    }
                    Text(
                        if (editing.contains(p.name)) "Cancel" else "Edit",
                        fontSize = 12.sp,
                        color = c.accent,
                        modifier = Modifier.clickable {
                            editing = if (editing.contains(p.name)) editing - p.name else editing + p.name
                        }.padding(8.dp),
                    )
                }
                // In-place edit. Previously the only way to change a
                // provider's model or endpoint was Remove + re-add, which
                // also discards the stored key.
                if (editing.contains(p.name)) {
                    var ep by remember(p.name, p.endpoint) { mutableStateOf(p.endpoint) }
                    var md by remember(p.name, p.model) { mutableStateOf(p.model) }
                    var pf by remember(p.name, p.profile) { mutableStateOf(p.profile) }
                    TextFieldRow("endpoint", ep, "https://…/v1/chat/completions", { ep = it }, Modifier.padding(top = 8.dp))
                    TextFieldRow("model", md, "model id", { md = it }, Modifier.padding(top = 8.dp))
                    TextFieldRow("profile", pf, "openai | anthropic | local", { pf = it }, Modifier.padding(top = 8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
                        Text(
                            "Save",
                            fontSize = 12.sp,
                            color = c.accentContrast,
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(c.accent)
                                .clickable {
                                    if (ep.isNotBlank()) {
                                        store.patchProvider(p.name, model = md, endpoint = ep, profile = pf)
                                        editing = editing - p.name
                                    }
                                }
                                .padding(horizontal = 16.dp, vertical = 8.dp),
                        )
                        // Only offered for a non-active provider; the
                        // condition was inverted, so the one provider the
                        // button appeared on was the one it could not switch.
                        if (p.name != state.config?.active) {
                            Text(
                                "Set as active",
                                fontSize = 12.sp,
                                color = c.accent,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(999.dp))
                                    .background(c.bgCardHover)
                                    .clickable {
                                        store.patchProvider(p.name, model = md, endpoint = ep, profile = pf)
                                        store.setActiveProvider(p.name)
                                        editing = editing - p.name
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp),
                            )
                        }
                    }
                }
                // Key entry: explicit save (never a write per keystroke).
                var keyText by remember(p.name, p.hasApiKey) { mutableStateOf("") }
                if (keyText.isEmpty() && p.hasApiKey) {
                    Text(
                        "Replace key",
                        fontSize = 12.sp,
                        color = c.accent,
                        modifier = Modifier
                            .clickable { keyText = " " }
                            .padding(vertical = 6.dp),
                    )
                } else {
                    TextFieldRow("key", keyText.trim(), "paste API key", { keyText = it }, Modifier.padding(bottom = 10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            "Save key",
                            fontSize = 12.sp,
                            color = c.accentContrast,
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(c.accent)
                                .clickable {
                                    if (keyText.isNotBlank()) {
                                        store.setProviderKey(p.name, keyText.trim())
                                        keyText = ""
                                    }
                                }
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                        )
                        Text(
                            "Clear",
                            fontSize = 12.sp,
                            color = c.errStrong,
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(c.bgCardHover)
                                .clickable { store.setProviderKey(p.name, ""); keyText = "" }
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                        )
                    }
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

// ── Usage ───────────────────────────────────────────────────────

@Composable
private fun UsagePane(store: AppStore) {
    val c = LocalSofuuColors.current
    var usage by remember { mutableStateOf<JSONObject?>(null) }
    LaunchedEffect(Unit) {
        // Bounded and guarded: this had no deadline at all, so an engine
        // that never answered left the pane spinning for the whole visit.
        usage = runCatching {
            store.engine.usageReport().get(10, TimeUnit.SECONDS)
        }.getOrNull()
    }
    Column(Modifier.verticalScroll(rememberScrollState())) {
        val totals = usage?.optJSONObject("totals")
        Row_("Session spend", "Tokens + cost across this workspace's sessions.", {
            Text(
                "$" + "%.4f".format(totals?.optDouble("cost") ?: 0.0) + "  ·  " +
                    "%,.0f".format(totals?.optDouble("pt") ?: 0.0) + " in  ·  " +
                    "%,.0f".format(totals?.optDouble("ct") ?: 0.0) + " out",
                fontSize = 13.sp,
                color = c.text,
                fontFamily = FontFamily.Monospace,
            )
        })
        Section("Per day")
        val days = usage?.optJSONArray("days")
        if (days == null || days.length() == 0) {
            Text("No usage yet.", fontSize = 13.sp, color = c.textFaint)
        } else {
            for (i in 0 until days.length()) {
                val d = days.optJSONObject(i) ?: continue
                Text(
                    "${d.optString("day")}  ·  ${d.optInt("turns")} turns  ·  " +
                        "%,.0f".format(d.optDouble("pt")) + " in  ·  " + "%,.0f".format(d.optDouble("ct")) + " out",
                    fontSize = 12.sp,
                    color = c.textMuted,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(vertical = 3.dp),
                )
            }
        }
        Section("Per session")
        val sess = usage?.optJSONArray("sessions")
        if (sess == null || sess.length() == 0) {
            Text("No sessions yet.", fontSize = 13.sp, color = c.textFaint)
        } else {
            for (i in 0 until minOf(sess.length(), 20)) {
                val d = sess.optJSONObject(i) ?: continue
                Text(
                    "${d.optString("task").ifEmpty { d.optString("id") }.take(28)}  ·  ${d.optString("model")}  ·  $" + "%.4f".format(d.optDouble("cost")),
                    fontSize = 12.sp,
                    color = c.textMuted,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(vertical = 3.dp),
                )
            }
        }
        Section("Per model")
        val models = usage?.optJSONArray("models")
        if (models == null || models.length() == 0) {
            Text("No usage yet.", fontSize = 13.sp, color = c.textFaint)
        } else {
            for (i in 0 until models.length()) {
                val m = models.optJSONObject(i) ?: continue
                Text(
                    "${m.optString("model")}  ·  ${m.optInt("turns")} turns  ·  $" + "%.4f".format(m.optDouble("cost")),
                    fontSize = 12.sp,
                    color = c.textMuted,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(vertical = 3.dp),
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}


// ── Brain ───────────────────────────────────────────────────────

/**
 * The CMA brain and the vector spaces behind it.
 *
 * This is what `/remember`, `/why` and `/brain` sit on in the CLI, made
 * visible: what is stored, in which embedding space, and the operations
 * that change it. Every call routes through the ONE cached brain handle
 * on the engine thread, because two handles on one QTSQ file clobber each
 * other on flush.
 */
@Composable
private fun BrainPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val surface = remember { com.sofuu.engine.RuntimeSurface(store.engine) }

    var info by remember { mutableStateOf<JSONObject?>(null) }
    var err by remember { mutableStateOf<String?>(null) }
    var spaces by remember { mutableStateOf<JSONObject?>(null) }
    var reload by remember { mutableStateOf(0) }
    // Click handlers are not composable scope, so they cannot host a
    // LaunchedEffect; this scope is the same one the effects above run in.
    val scope = rememberCoroutineScope()
    var hits by remember { mutableStateOf<org.json.JSONArray?>(null) }
    var recalled by remember { mutableStateOf("") }
    var decayArmed by remember { mutableStateOf(false) }

    var ml by remember { mutableStateOf<JSONObject?>(null) }
    var m2 by remember { mutableStateOf<Boolean?>(null) }
    var probeTool by remember { mutableStateOf("read_file") }
    var probeArgs by remember { mutableStateOf("{\"path\":\"README.md\"}") }
    var verdict by remember { mutableStateOf<JSONObject?>(null) }
    var gcNote by remember { mutableStateOf<String?>(null) }
    var imgNote by remember { mutableStateOf<String?>(null) }
    var imgPath by remember { mutableStateOf<String?>(null) }
    var imgQuery by remember { mutableStateOf("a red background") }
    var imgSim by remember { mutableStateOf<JSONObject?>(null) }
    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            val img = withContext(Dispatchers.IO) { store.importPickedImage(uri) }
            if (img == null) {
                imgNote = "Could not import that image."
            } else {
                val r = withContext(Dispatchers.IO) {
                    runCatching { surface.embedImage(img).get(30, TimeUnit.SECONDS) }.getOrNull()
                }
                if (r?.optBoolean("ok") == true) {
                    imgNote = "${img.substringAfterLast('/')} · ${r.optInt("dim")}-dim img1 vector"
                    imgPath = img
                } else {
                    imgNote = "The image projector is not in this build: ${r?.optString("error") ?: "unavailable"}"
                    imgPath = null
                }
            }
        }
    }

    var simA by remember { mutableStateOf("the cat sat down") }
    var simB by remember { mutableStateOf("a feline rested") }
    var sim by remember { mutableStateOf<JSONObject?>(null) }

    // Reload is a counter, not a function: LaunchedEffect has to be called
    // from the composable's own scope, so the trigger is the key.
    LaunchedEffect(reload) {
        val r = withContext(Dispatchers.IO) { surface.brainInfo() }
        info = r?.takeIf { it.optBoolean("ok") }
        err = r?.optString("error")?.takeIf { it.isNotEmpty() }
        spaces = withContext(Dispatchers.IO) { surface.spaces() }
        ml = withContext(Dispatchers.IO) { surface.mlState() }
    }
    LaunchedEffect(recalled) {
        if (recalled.isBlank()) { hits = null; return@LaunchedEffect }
        val r = withContext(Dispatchers.IO) { surface.brainRecall(recalled, 8) }
        if (r?.optBoolean("ok") == true) hits = r.optJSONArray("hits") else hits = null
    }
    LaunchedEffect(simA, simB) {
        if (simA.isBlank() || simB.isBlank()) { sim = null; return@LaunchedEffect }
        sim = withContext(Dispatchers.IO) { surface.similarity(simA, simB) }
    }

    Column(Modifier.verticalScroll(rememberScrollState())) {

        Section("Stored")
        when {
            err != null -> Text(err!!, fontSize = 12.sp, color = c.err)
            info == null -> Text("Reading the brain…", fontSize = 12.sp, color = c.textFaint)
            else -> {
                val i = info!!
                Text(
                    "${i.optLong("count")} memories",
                    fontSize = 20.sp, color = c.text, fontFamily = FontFamily.Monospace,
                )
                Text(
                    "space ${i.optString("backend") ?: "?"} · ${i.optInt("dim")}-dim" +
                        if (i.optBoolean("semantic")) " · semantic channel on" else "",
                    fontSize = 11.sp, color = c.textFaint, fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }

        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            SmallButton(c.accent, c.accentContrast, "Refresh") { reload++ }
            SmallButton(c.accent, c.accentContrast, "Consolidate") {
                scope.launch {
                    val r = withContext(Dispatchers.IO) { surface.brainConsolidate() }
                    err = r?.optString("error")?.takeIf { it.isNotEmpty() }
                    reload++
                }
            }
            SmallButton(
                if (decayArmed) c.warn else c.accent,
                if (decayArmed) c.bgApp else c.accentContrast,
                if (decayArmed) "Confirm decay" else "Decay tick",
            ) {
                if (decayArmed) {
                    decayArmed = false
                    scope.launch {
                        val r = withContext(Dispatchers.IO) { surface.brainDecay() }
                        err = r?.optString("error")?.takeIf { it.isNotEmpty() }
                        reload++
                    }
                } else {
                    decayArmed = true
                }
            }
        }
        if (decayArmed) {
            Text(
                "Decay weakens and prunes old memories. Tap again to run it.",
                fontSize = 10.sp, color = c.warn, modifier = Modifier.padding(top = 4.dp),
            )
        }

        Section("Search")
        TextFieldRow("recall", recalled, "a fact to recall", onChange = { recalled = it })
        hits?.let { h ->
            Spacer(Modifier.height(8.dp))
            if (h.length() == 0) {
                Text("No memories matched.", fontSize = 12.sp, color = c.textFaint)
            }
            for (i in 0 until h.length()) {
                val o = h.optJSONObject(i) ?: continue
                val id = o.optInt("id", -1)
                Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            o.optString("text").ifEmpty { o.optString("kind") },
                            fontSize = 12.sp, color = c.text,
                        )
                        Text(
                            "score ${"%.3f".format(o.optDouble("score"))} · ${o.optString("tier")}",
                            fontSize = 10.sp, color = c.textFaint, fontFamily = FontFamily.Monospace,
                        )
                    }
                    if (id >= 0) {
                        Icon(
                            Icons2.Close, "Forget", tint = c.textFaint,
                            modifier = Modifier.size(16.dp).clickable {
                                scope.launch {
                                    val r = withContext(Dispatchers.IO) { surface.brainForget(id) }
                                    err = r?.optString("error")?.takeIf { it.isNotEmpty() }
                                    hits = null
                                    reload++
                                }
                            },
                        )
                    }
                }
            }
        }

        Section("Embedding spaces")
        val sp = spaces?.optJSONObject("info")
        if (sp == null) {
            Text("No embedding manifest yet.", fontSize = 12.sp, color = c.textFaint)
        } else {
            SpaceLine(c.text, c.err, c.textFaint, "hash-768 (default)", sp.optJSONObject("hash")?.optInt("dimension") ?: 0, true)
            val sem = sp.optJSONObject("semantic")
            SpaceLine(c.text, c.err, c.textFaint, "semantic", sem?.optInt("dimension") ?: 0, sem?.optBoolean("available", false) ?: false)
            val v2 = spaces?.optJSONObject("v2")
            if (v2 != null) {
                SpaceLine(c.text, c.err, c.textFaint, "semantic v2", v2.optInt("dimension"), v2.optBoolean("available", false))
            }
        }

        Section("Context gates")
        val m = ml
        if (m == null) {
            Text("The sofuu.ml layer is not loaded in this build.", fontSize = 12.sp, color = c.textFaint)
        } else if (m.optBoolean("ok") != true) {
            Text(m.optString("error").ifEmpty { "unavailable" }, fontSize = 12.sp, color = c.err)
        } else {
            val info = m.optJSONObject("info")
            val ws = m.optJSONObject("workset")
            Text(
                "${ws?.optLong("runs") ?: 0} runs · ${ws?.optLong("calls") ?: 0} calls · " +
                    "${ws?.optLong("segments") ?: 0} segments",
                fontSize = 12.sp, color = c.text, fontFamily = FontFamily.Monospace,
            )
            val on = info?.optJSONObject("on")
            if (on != null) {
                val keys = on.keys()
                while (keys.hasNext()) {
                    val k = keys.next()
                    val state = if (on.optBoolean(k)) "on" else "off"
                    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                        Text(
                            k, fontSize = 11.sp, color = c.textFaint,
                            fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
                        )
                        Text(
                            state, fontSize = 11.sp,
                            color = if (state == "on") c.accent else c.textMuted,
                        )
                    }
                }
            }
            val learner = info?.optJSONObject("online")
            if (learner != null && learner.optBoolean("enabled")) {
                Text(
                    "Online learning is on" + if (learner.optBoolean("pending")) " · a candidate is pending" else "",
                    fontSize = 11.sp, color = c.warn, modifier = Modifier.padding(top = 4.dp),
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 8.dp)) {
                SmallButton(c.accent, c.accentContrast, "Label wrong") {
                    scope.launch {
                        val r = withContext(Dispatchers.IO) { surface.mlFeedback("wrong") }
                        m2 = r?.optBoolean("ok") == true
                        reload++
                    }
                }
                SmallButton(c.accent, c.accentContrast, "Label wasted") {
                    scope.launch {
                        val r = withContext(Dispatchers.IO) { surface.mlFeedback("wasted") }
                        m2 = r?.optBoolean("ok") == true
                        reload++
                    }
                }
            }
            if (m2 != null) {
                Text(
                    if (m2 == true) "Label recorded." else "The engine declined that label — it needs an observed call to attach to.",
                    fontSize = 10.sp, color = if (m2 == true) c.textFaint else c.warn,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }

        Section("Gate check")
        Text(
            "Ask what the gates would say about a call before the agent makes it.",
            fontSize = 11.sp, color = c.textFaint,
        )
        TextFieldRow("tool", probeTool, "read_file", onChange = { probeTool = it })
        TextFieldRow("args", probeArgs, "{\"path\":\"README.md\"}", onChange = { probeArgs = it })
        Text(
            "Check",
            fontSize = 12.sp,
            color = c.accentContrast,
            modifier = Modifier
                .padding(top = 6.dp)
                .clip(RoundedCornerShape(999.dp))
                .background(c.accent)
                .clickable {
                    scope.launch {
                        val parsed = try { JSONObject(probeArgs) } catch (e: Exception) { JSONObject() }
                        verdict = withContext(Dispatchers.IO) { surface.mlCheck(probeTool, parsed) }
                    }
                }
                .padding(horizontal = 14.dp, vertical = 8.dp),
        )
        verdict?.let { v ->
            Spacer(Modifier.height(6.dp))
            Text(
                if (v.optBoolean("ok")) "pass · ${v.optString("source")} · ${"%.3f".format(v.optDouble("score"))}" +
                    (v.optString("nudge").takeIf { it.isNotEmpty() && it != "null" }?.let { " · nudge: $it" } ?: "")
                else "blocked · ${v.optString("reason")}",
                fontSize = 11.sp, color = if (v.optBoolean("ok")) c.textMuted else c.warn,
            )
        }

        Section("Memory")
        Text(
            "One runtime lives for the whole process on a phone, so nothing else " +
                "would run the collector at the right moment. Do this after a big " +
                "headless run or a batch embed.",
            fontSize = 11.sp, color = c.textFaint,
        )
        SmallButton(c.accent, c.accentContrast, "Collect garbage") {
            scope.launch {
                val okRun = withContext(Dispatchers.IO) { surface.collectGarbage() }
                gcNote = if (okRun) "Collected." else "The runtime did not expose __sofuu_gc in this build."
            }
        }
        gcNote?.let { Text(it, fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(top = 4.dp)) }

        Section("Image vectors")
        Text(
            "The multimodal tower embeds a picture into the same space a text " +
                "query lands in, so you can ask what an image looks like.",
            fontSize = 11.sp, color = c.textFaint,
        )
        SmallButton(c.accent, c.accentContrast, "Pick an image") { pickImage.launch("image/*") }
        imgNote?.let {
            Text(it, fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(top = 6.dp))
        }
        if (imgPath != null) {
            TextFieldRow("describe", imgQuery, "what to look for", onChange = { imgQuery = it })
            SmallButton(c.accent, c.accentContrast, "Compare to text") {
                scope.launch {
                    imgSim = withContext(Dispatchers.IO) { surface.imageTextSimilarity(imgPath!!, imgQuery) }
                }
            }
            imgSim?.let { r ->
                Spacer(Modifier.height(6.dp))
                Text(
                    if (r.optBoolean("ok")) "cosine ${"%.4f".format(r.optDouble("similarity"))}" +
                        " · ${r.optInt("dim")}-dim"
                    else r.optString("error").ifEmpty { "spaces do not line up" },
                    fontSize = 12.sp,
                    color = if (r.optBoolean("ok")) c.text else c.warn,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }

        Section("Similarity")
        TextFieldRow("a", simA, "first string", onChange = { simA = it })
        TextFieldRow("b", simB, "second string", onChange = { simB = it })
        sim?.let {
            Spacer(Modifier.height(6.dp))
            Text(
                "cosine ${"%.4f".format(it.optDouble("similarity"))} · ${it.optInt("dim")}-dim",
                fontSize = 12.sp, color = c.text, fontFamily = FontFamily.Monospace,
            )
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SpaceLine(text: Color, err: Color, faint: Color, label: String, dim: Int, available: Boolean) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(
            label, fontSize = 12.sp, color = if (available) text else err,
            fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
        )
        Text(
            if (available) "$dim-dim" else "artifact unavailable",
            fontSize = 11.sp, color = if (available) faint else err,
        )
    }
}

@Composable
private fun SmallButton(bg: Color, fg: Color, label: String, onClick: () -> Unit) {
    Text(
        label, fontSize = 12.sp, color = fg,
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(bg)
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 8.dp),
    )
}

// ── Agents ──────────────────────────────────────────────────────

/**
 * Registered agents, and the form to add one from a phone.
 *
 * `define` is the runtime's own registry — an agent added here is a
 * first-class entry that the chat turn, `/agents` and `delegate` all see,
 * not a local preset. Its tools are named the way `sofuu.tools.TOOLS`
 * names them, which after host-tool registration includes the phone's own
 * capabilities (device_info, share_text, …).
 */
@Composable
private fun AgentsPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val surface = remember { com.sofuu.engine.RuntimeSurface(store.engine) }
    val scope = rememberCoroutineScope()

    var list by remember { mutableStateOf<org.json.JSONArray?>(null) }
    var reload by remember { mutableStateOf(0) }
    var note by remember { mutableStateOf<String?>(null) }

    var name by remember { mutableStateOf("reviewer") }
    var system by remember { mutableStateOf("You are a careful code reviewer. Report only what you can point at.") }
    var maxSteps by remember { mutableStateOf("24") }
    var withWeb by remember { mutableStateOf(true) }
    var withCode by remember { mutableStateOf(true) }

    var task by remember { mutableStateOf("") }
    var answer by remember { mutableStateOf<JSONObject?>(null) }
    var running by remember { mutableStateOf(false) }

    LaunchedEffect(reload) {
        list = withContext(Dispatchers.IO) { surface.agents() }
    }

    Column(Modifier.verticalScroll(rememberScrollState())) {

        Section("Registered")
        val l = list
        if (l == null) {
            Text("Reading the registry…", fontSize = 12.sp, color = c.textFaint)
        } else if (l.length() == 0) {
            Text("No agents defined yet. Add one below.", fontSize = 12.sp, color = c.textFaint)
        } else {
            for (i in 0 until l.length()) {
                val a = l.optJSONObject(i) ?: continue
                Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    Text(
                        a.optString("name"),
                        fontSize = 14.sp, color = c.text, fontFamily = FontFamily.Monospace,
                    )
                    val tools = a.optJSONArray("tools")
                    val subs = a.optInt("agents")
                    Text(
                        buildString {
                            tools?.let { t ->
                                append(
                                    (0 until t.length()).joinToString(", ") { t.optString(it) }
                                        .ifEmpty { "no tools" }
                                )
                            }
                            if (subs > 0) append(" · $subs sub-agent slots")
                            a.optString("provider").takeIf { it.isNotEmpty() }?.let { append(" · $it") }
                        },
                        fontSize = 11.sp, color = c.textFaint,
                    )
                }
            }
        }

        Section("Define an agent")
        TextFieldRow("name", name, "reviewer", onChange = { name = it })
        TextFieldRow("system", system, "what this agent is for", onChange = { system = it })
        TextFieldRow("max steps", maxSteps, "24", onChange = { maxSteps = it })
        ToggleRow("Web tools", "web_search and web_open", withWeb) { withWeb = it }
        ToggleRow("Coding tools", "the 8 built-ins: read/write/edit/grep/glob/dir/bash/todo", withCode) { withCode = it }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 10.dp)) {
            SmallButton(c.accent, c.accentContrast, "Register") {
                scope.launch {
                    val tools = buildList {
                        if (withWeb) add("web")
                        if (withCode) add("code")
                    }
                    val budget = JSONObject().put("maxSteps", maxSteps.filter { it.isDigit() }.toIntOrNull() ?: 24)
                    val r = withContext(Dispatchers.IO) {
                        surface.defineAgent(name.trim(), system.trim(), tools, budget)
                    }
                    note = if (r?.optBoolean("ok") == true) {
                        "${r.optString("name")} registered — the agent can use it by name, and /agents lists it."
                    } else {
                        "Could not register: ${r?.optString("error") ?: "unknown error"}"
                    }
                    reload++
                }
            }
        }
        note?.let {
            Text(it, fontSize = 11.sp, color = c.textFaint, modifier = Modifier.padding(top = 6.dp))
        }

        Section("Run headlessly")
        TextFieldRow("task", task, "a task to hand the agent", onChange = { task = it })
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 8.dp)) {
            SmallButton(if (running) c.bgCardHover else c.accent, if (running) c.textFaint else c.accentContrast,
                if (running) "Running…" else "Run") {
                if (running || task.isBlank()) return@SmallButton
                running = true; answer = null
                scope.launch {
                    val r = withContext(Dispatchers.IO) {
                        runCatching {
                            surface.runAgent(name.trim(), task.trim(), 300_000).get(300, TimeUnit.SECONDS)
                        }.getOrNull()
                    }
                    answer = r ?: JSONObject().put("ok", false).put("error", "the run did not finish in 5 minutes")
                    running = false
                }
            }
        }
        Text(
            "A run goes through the same agent loop, tools, budget and brain as a " +
                "chat turn — the only difference is that the answer is collected " +
                "instead of streamed.",
            fontSize = 10.sp, color = c.textFaint, modifier = Modifier.padding(top = 6.dp),
        )
        answer?.let { a ->
            Spacer(Modifier.height(10.dp))
            if (a.optBoolean("ok") != true) {
                Text(a.optString("error").ifEmpty { "run failed" }, fontSize = 12.sp, color = c.err)
            } else {
                Text(
                    "${a.optInt("steps")} steps" + (a.optString("stopped").takeIf { it.isNotEmpty() && it != "null" }
                        ?.let { " · stopped: $it" } ?: ""),
                    fontSize = 11.sp, color = c.textFaint, fontFamily = FontFamily.Monospace,
                )
                Text(
                    a.optString("answer"),
                    fontSize = 13.sp, color = c.text, modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}
