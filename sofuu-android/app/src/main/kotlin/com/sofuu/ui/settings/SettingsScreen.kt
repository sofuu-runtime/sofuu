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
import com.sofuu.store.AppStore
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.theme.LocalSofuuColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

private val TABS = listOf("general", "appearance", "models", "providers", "usage")

/** Full-screen settings — the same panes as the desktop modal, minus
 *  Shortcuts (deliberately dropped on mobile). */
@Composable
fun SettingsScreen(store: AppStore, onClose: () -> Unit) {
    val c = LocalSofuuColors.current
    var tab by remember { mutableStateOf(store.state.settingsTab) }
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
    onSubmit: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = LocalSofuuColors.current
    var text by remember { mutableStateOf(value) }
    LaunchedEffect(value) { text = value }
    androidx.compose.foundation.text.BasicTextField(
        value = text,
        onValueChange = { text = it },
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
            TextFieldRow("ctx", state.config?.ctxWindow?.toString().orEmpty(), "auto", {
                it.toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("ctx_window", n)) }
            })
        })
        Row_("Max output tokens", "Override the per-request output cap (0 = model default).", {
            TextFieldRow("maxout", state.config?.maxOutput?.toString().orEmpty(), "model default", {
                it.toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("max_output", n)) }
            })
        })
        Row_("Session budget (USD)", "The engine refuses new turns past the cap (0 = uncapped).", {
            TextFieldRow("budget", state.config?.budgetUsd?.takeIf { it > 0 }?.toString().orEmpty(), "uncapped", {
                it.toDoubleOrNull()?.let { n -> store.patchConfig(JSONObject().put("budget_usd", n)) }
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
            TextFieldRow("recall_min", state.config?.recallMin?.takeIf { it > 0 }?.toString().orEmpty(), "default 0.30", {
                it.toDoubleOrNull()?.let { n ->
                    store.patchConfig(JSONObject().put("recall_min", n.coerceIn(0.0, 1.0)))
                }
            })
        })
        Row_("Recall token budget", "Max tokens of memories injected per turn (0 = auto ≈2% of window, 1k–16k).", {
            TextFieldRow("recall_budget", state.config?.recallBudget?.takeIf { it > 0 }?.toString().orEmpty(), "auto", {
                it.toLongOrNull()?.let { n ->
                    store.patchConfig(JSONObject().put("recall_budget", n.coerceIn(0L, 16_000L)))
                }
            })
        })
        Row_("Embeddings — provider", "Optional OpenAI-compatible provider for memory embeddings. Blank = the bundled offline embedder.", {
            TextFieldRow("embed_provider", state.config?.embedProvider.orEmpty(), "bundled offline embedder", {
                store.patchConfig(JSONObject().put("embed_provider", it.trim()))
            })
        })
        Row_("Embeddings — model", "Embedding model id used for memory vectors.", {
            TextFieldRow("embed_model", state.config?.embedModel.orEmpty(), "default", {
                store.patchConfig(JSONObject().put("embed_model", it.trim()))
            })
        })
        VectorEngineCard(store)
        Spacer(Modifier.height(24.dp))
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
                val label = path.substringAfterLast('/').removeSuffix(".gguf")
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
                    if (newName.isNotEmpty() && newEndpoint.isNotEmpty()) {
                        val providers = (state.config?.providers ?: emptyList()).map { it.toJson() }.toMutableList()
                        providers.add(JSONObject().put("name", newName).put("endpoint", newEndpoint).put("model", newModel).put("profile", ""))
                        store.patchConfig(JSONObject().put("providers", org.json.JSONArray(providers)).put("active", newName).put("provider", newName))
                        if (newKey.isNotEmpty()) store.setProviderKey(newName, newKey)
                        newName = ""; newEndpoint = ""; newModel = ""; newKey = ""
                    }
                }
                .padding(horizontal = 18.dp, vertical = 10.dp),
        )

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
                                    if (ep.isNotBlank()) store.patchProvider(p.name, model = md, endpoint = ep, profile = pf)
                                    editing = editing - p.name
                                }
                                .padding(horizontal = 16.dp, vertical = 8.dp),
                        )
                        if (p.name == state.config?.active) {
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
        usage = store.engine.usageReport().get()
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

