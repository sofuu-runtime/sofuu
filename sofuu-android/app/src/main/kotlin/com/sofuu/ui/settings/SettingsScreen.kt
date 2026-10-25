package com.sofuu.ui.settings

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.store.AppStore
import com.sofuu.ui.CloudBackground
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.theme.LocalSofuuColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private val TABS = listOf("general", "appearance", "models", "providers", "agents", "brain", "usage")

/** Full-screen settings — the same panes as the desktop modal, minus
 *  Shortcuts (deliberately dropped on mobile). */
@Composable
fun SettingsScreen(store: AppStore, onClose: () -> Unit) {
    val c = LocalSofuuColors.current
    val density = LocalDensity.current
    var tab by remember { mutableStateOf(store.state.settingsTab) }
    // External tab requests (ModelSheet → "＋ Add provider", composer rows)
    // fire while Settings is already composed — a remember-once initial
    // value would silently ignore them. Local tab taps never write the
    // store, so this only fires on external change.
    LaunchedEffect(store.state.settingsTab) { tab = store.state.settingsTab }

    val statusTop = WindowInsets.statusBars.asPaddingValues().calculateTopPadding()
    var headerPx by remember { mutableStateOf(0) }
    val headerHeight = with(density) { headerPx.toDp() }
    val effectiveTopBand = statusTop + if (headerPx > 0) headerHeight else 88.dp

    Box(Modifier.fillMaxSize().background(c.bgApp)) {
        if (store.state.cloudBg) CloudBackground()

        // ── Settings Content: starts close to buttons and scrolls UNDER top band ──
        Column(
            Modifier
                .fillMaxSize()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(
                    top = effectiveTopBand + 2.dp,
                    start = 16.dp,
                    end = 16.dp,
                    bottom = 28.dp,
                ),
        ) {
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

        // ── Top Transparency Band (Only top area for settings) ─────
        // Pinned to screen top: fades from opaque background (status bar)
        // down to fully transparent (alpha = 0f) at the bottom edge of
        // the category tabs.
        Box(
            Modifier
                .fillMaxWidth()
                .height(effectiveTopBand)
                .align(Alignment.TopCenter)
                .background(
                    Brush.verticalGradient(
                        0f to c.bgApp,
                        0.55f to c.bgApp.copy(alpha = 0.70f),
                        1f to c.bgApp.copy(alpha = 0f),
                    ),
                ),
        )

        // ── Header Controls: Title + Close + Category Pills ────────
        Column(
            Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .statusBarsPadding(),
        ) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .onSizeChanged { headerPx = it.height },
            ) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "Settings",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = c.text,
                        modifier = Modifier.weight(1f),
                    )
                    Icon(
                        Icons2.Close,
                        "Close",
                        tint = c.textMuted,
                        modifier = Modifier
                            .size(36.dp)
                            .clip(RoundedCornerShape(18.dp))
                            .clickable { onClose() }
                            .padding(8.dp),
                    )
                }
                Box(Modifier.padding(bottom = 6.dp)) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                            .padding(horizontal = 16.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        TABS.forEach { t ->
                            val active = t == tab
                            Text(
                                t.replaceFirstChar { it.uppercase() },
                                fontSize = 13.5.sp,
                                fontWeight = if (active) FontWeight.Bold else FontWeight.SemiBold,
                                color = if (active) c.accentContrast else c.textMuted,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(999.dp))
                                    .background(if (active) c.accent else c.bgCard.copy(alpha = 0.85f))
                                    .border(
                                        0.5.dp,
                                        if (active) c.accent else c.border.copy(alpha = 0.70f),
                                        RoundedCornerShape(999.dp),
                                    )
                                    .clickable { tab = t; store.setSettingsTab(t) }
                                    .padding(horizontal = 15.dp, vertical = 8.dp),
                            )
                        }
                    }
                    Box(
                        Modifier
                            .align(Alignment.CenterEnd)
                            .width(28.dp)
                            .height(38.dp)
                            .background(
                                Brush.horizontalGradient(listOf(Color.Transparent, c.bgApp.copy(alpha = 0.8f), c.bgApp)),
                            ),
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingsCard(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    val c = LocalSofuuColors.current
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(c.bgCard)
            .border(1.dp, c.border, RoundedCornerShape(14.dp))
            .padding(14.dp),
        content = content,
    )
}

@Composable
private fun CardDivider(modifier: Modifier = Modifier) {
    val c = LocalSofuuColors.current
    Box(
        modifier
            .fillMaxWidth()
            .padding(vertical = 10.dp)
            .height(0.8.dp)
            .background(c.border.copy(alpha = 0.6f)),
    )
}

@Composable
private fun Section(label: String, isFirst: Boolean = false) {
    Text(
        label.uppercase(),
        fontSize = 11.5.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 0.8.sp,
        color = LocalSofuuColors.current.textFaint,
        modifier = Modifier.padding(top = if (isFirst) 4.dp else 18.dp, bottom = 6.dp, start = 2.dp),
    )
}

@Composable
private fun Row_(label: String, desc: String?, content: @Composable () -> Unit) {
    val c = LocalSofuuColors.current
    Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Text(label, fontSize = 14.5.sp, fontWeight = FontWeight.Bold, color = c.text)
        if (desc != null) {
            Text(desc, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(top = 2.dp, bottom = 8.dp))
        } else {
            Spacer(Modifier.height(6.dp))
        }
        content()
    }
}

@Composable
private fun Segmented(options: List<String>, selected: String, onSelect: (String) -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier
            .clip(RoundedCornerShape(10.dp))
            .background(c.monoFieldBg)
            .border(1.dp, c.border, RoundedCornerShape(10.dp))
            .padding(3.dp),
        horizontalArrangement = Arrangement.spacedBy(3.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        options.forEach { o ->
            val active = o == selected
            val label = when (o) {
                "prompt" -> "Ask me"
                else -> o.replaceFirstChar { it.uppercase() }
            }
            Box(
                Modifier
                    .weight(1f, fill = false)
                    .clip(RoundedCornerShape(7.dp))
                    .background(if (active) c.accent else Color.Transparent)
                    .clickable { onSelect(o) }
                    .padding(horizontal = 14.dp, vertical = 7.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label,
                    fontSize = 13.sp,
                    fontWeight = if (active) FontWeight.Bold else FontWeight.SemiBold,
                    color = if (active) c.accentContrast else c.textMuted,
                )
            }
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
    var text by remember { mutableStateOf(value) }
    var focused by remember { mutableStateOf(false) }
    LaunchedEffect(value, focused) {
        if (!focused || value.isEmpty()) text = value
    }
    androidx.compose.foundation.text.BasicTextField(
        value = text,
        onValueChange = {
            text = it
            onChange(it)
        },
        singleLine = true,
        textStyle = androidx.compose.ui.text.TextStyle(
            color = c.text,
            fontFamily = FontFamily.Monospace,
            fontSize = 13.5.sp,
            fontWeight = FontWeight.Medium,
        ),
        cursorBrush = androidx.compose.ui.graphics.SolidColor(c.accent),
        modifier = modifier
            .onFocusChanged { focused = it.isFocused }
            .fillMaxWidth()
            .heightIn(min = 46.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(c.monoFieldBg)
            .border(1.dp, if (focused) c.accent else c.border, RoundedCornerShape(8.dp))
            .padding(horizontal = 12.dp, vertical = 12.dp),
        decorationBox = { inner ->
            Box {
                if (text.isEmpty()) {
                    Text(
                        placeholder,
                        color = c.textFaint,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.5.sp,
                        fontWeight = FontWeight.Medium,
                    )
                }
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
    var text by remember(value) { mutableStateOf(value) }
    LaunchedEffect(value) { text = value }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Box(Modifier.weight(1f)) {
            TextFieldRow(label, text, placeholder, { text = it })
        }
        Box(
            Modifier
                .height(46.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(c.accent)
                .clickable { onApply(text) }
                .padding(horizontal = 16.dp),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                "Apply",
                fontSize = 13.5.sp,
                fontWeight = FontWeight.Bold,
                color = c.accentContrast,
            )
        }
    }
}

/** One key/value tile of the live engine card. */
@Composable
private fun EngineCell(key: String, value: String, mono: Boolean = false) {
    val c = LocalSofuuColors.current
    Row(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(key, fontSize = 12.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.width(115.dp))
        Text(
            value,
            fontSize = 13.sp,
            color = c.text,
            fontWeight = if (mono) FontWeight.Medium else FontWeight.SemiBold,
            fontFamily = if (mono) FontFamily.Monospace else FontFamily.Default,
            maxLines = 1,
            overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
        )
    }
}

/**
 * The native vector engine (H-E1/M1) as a diagnostic card.
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
    SettingsCard {
        if (manifest == null) {
            Text("Loading the embedding manifest…", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
        } else {
            val m = manifest!!
            EngineCell(
                "Default space",
                m.optString("default_space").ifEmpty { "—" } + "  (the brain's space)",
                mono = true,
            )
            val spaces = m.optJSONArray("spaces")
            if (spaces != null) {
                CardDivider()
                Text("SPACES", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = c.textFaint, letterSpacing = 0.5.sp)
                Spacer(Modifier.height(4.dp))
                for (i in 0 until spaces.length()) {
                    val s = spaces.optJSONObject(i) ?: continue
                    val id = s.optString("space")
                    val model = s.optJSONObject("model")
                    val dim = s.optInt("dimension").takeIf { it > 0 }
                        ?: model?.optInt("dimension")?.takeIf { it > 0 }
                        ?: model?.optInt("dim")?.takeIf { it > 0 }
                        ?: 0
                    val avail = s.optBoolean("available", true) &&
                        (model == null || model.optBoolean("available", true))
                    val modelId = s.optString("id").ifEmpty { model?.optString("id").orEmpty() }
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            id,
                            fontSize = 12.5.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = if (avail) c.text else c.err,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.weight(1f),
                        )
                        if (dim > 0) {
                            Text(
                                "${dim}-dim",
                                fontSize = 11.5.sp,
                                fontWeight = FontWeight.Medium,
                                color = c.textMuted,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.padding(end = 6.dp),
                            )
                        }
                        Text(
                            if (avail) (if (modelId.isNotEmpty()) modelId else "ready") else "unavailable",
                            fontSize = 11.5.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = if (avail) c.ok else c.err,
                        )
                    }
                }
            }
            CardDivider()
            Text("Test Local Embedding", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = c.text)
            Text(
                "Embeddings run entirely on-device — no provider, no network, no API key.",
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                color = c.textFaint,
                modifier = Modifier.padding(top = 2.dp, bottom = 8.dp),
            )
            TextFieldRow("probe", probe, "text to embed locally", { probe = it })
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                SmallButton(
                    if (probing) c.bgCardHover else c.accent,
                    if (probing) c.textMuted else c.accentContrast,
                    if (probing) "Embedding…" else "Embed",
                ) {
                    if (!probing && probe.isNotBlank()) {
                        probing = true
                        probeResult = null
                    }
                }
            }
            val res = probeResult
            if (res != null) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .padding(top = 10.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(c.monoFieldBg)
                        .border(1.dp, c.border, RoundedCornerShape(8.dp))
                        .padding(10.dp),
                ) {
                    Text(res, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textMuted, fontFamily = FontFamily.Monospace)
                }
            }
            LaunchedEffect(probing, probe) {
                if (!probing) return@LaunchedEffect
                val text = probe
                probeResult = withContext(Dispatchers.IO) {
                    val v = runCatching { store.embedText(text) }.getOrNull()
                    when {
                        v == null -> "Embed failed — space unknown or artifact unavailable."
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
        }
    }
}

// ── General ─────────────────────────────────────────────────────

@Composable
private fun GeneralPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    var live by remember { mutableStateOf<JSONObject?>(null) }
    LaunchedEffect(Unit) {
        while (true) {
            // A null answer means "no update" (the engine is mid-turn, the 2 s
            // deadline expired) — keep the last card instead of flickering
            // back to "Connecting…" whenever a turn is streaming.
            val next = withContext(Dispatchers.IO) { store.engineChatState() }
            if (next != null) live = next
            delay(2000)
        }
    }
    Column {
        Section("Engine status", isFirst = true)
        if (live == null) {
            SettingsCard {
                Text("Connecting to the engine…", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            }
        } else {
            val e = live!!
            SettingsCard {
                EngineCell("Model", e.optString("model").ifEmpty { "None" }, mono = true)
                CardDivider()
                EngineCell("Provider", e.optString("provider").ifEmpty { "None" })
                CardDivider()
                EngineCell("Session", e.optString("sessionId").let { if (it.isEmpty()) "None yet" else it.take(10) + "…" }, mono = true)
                CardDivider()
                EngineCell("History", "${e.optInt("historyTurns")} turns")
                CardDivider()
                EngineCell("Context", "${e.optInt("contextTokens")} tk")
                CardDivider()
                val spend = e.optDouble("sessionSpendUsd")
                EngineCell("Session spend", if (spend.isNaN() || spend < 0) "$0.0000" else "$" + "%.4f".format(spend))
                if (e.optBoolean("streaming") || e.optInt("pendingApprovals") > 0) {
                    CardDivider()
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(6.dp).clip(RoundedCornerShape(3.dp)).background(c.accent))
                        Spacer(Modifier.width(6.dp))
                        Text(
                            buildString {
                                if (e.optBoolean("streaming")) append("Generating…")
                                val n = e.optInt("pendingApprovals")
                                if (n > 0) { if (isNotEmpty()) append(" · "); append("$n awaiting approval") }
                            },
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = c.accent,
                        )
                    }
                }
            }
        }

        Section("Workspace & permissions")
        SettingsCard {
            Row_("Project folder", "Workspace root and per-project storage (.sofuu/ — sessions, memory, settings).", {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(c.monoFieldBg)
                        .border(1.dp, c.border, RoundedCornerShape(8.dp))
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    Text(state.projectDir ?: "(none)", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.text, fontFamily = FontFamily.Monospace)
                }
            })
            CardDivider()
            Row_("Tool permission profile", "Applied live in the engine and re-applied on launch.", {
                Segmented(listOf("full", "edit", "plan", "prompt"), state.permissions) { store.setPermissions(it) }
            })
            CardDivider()
            val context = LocalContext.current
            val hasAllFiles = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else true
            Row_("Phone storage access", if (hasAllFiles) "Full phone filesystem access is active. The engine can access and edit any folder." else "Grant All Files Access so the engine can access and edit any phone directory.", {
                if (hasAllFiles) {
                    Text("Granted", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = c.ok)
                } else {
                    Row(
                        Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(c.accent)
                            .clickable {
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                                    try {
                                        val intent = Intent(android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                                            data = Uri.parse("package:" + context.packageName)
                                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                        }
                                        context.startActivity(intent)
                                    } catch (_: Exception) {
                                        val intent = Intent(android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION).apply {
                                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                        }
                                        context.startActivity(intent)
                                    }
                                }
                            }
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Grant access", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = c.accentContrast)
                    }
                }
            })
        }

        Section("Context & reasoning")
        SettingsCard {
            Row_("Memory (brain)", "Recall saved project memories into each turn (.sofuu/brain/).", {
                Segmented(listOf("on", "off"), if (state.config?.brain != false) "on" else "off") {
                    store.patchConfig(JSONObject().put("brain", it == "on"))
                }
            })
            CardDivider()
            Row_("Smart context filters", "On-device neural filters (freshness, compaction, relevance, allocation).", {
                Segmented(listOf("on", "off"), if (state.config?.ml != false) "on" else "off") {
                    store.patchConfig(JSONObject().put("ml", it == "on"))
                }
            })
            CardDivider()
            Row_("RLM sandbox", "Route very long-context turns through the local reasoning sandbox.", {
                Segmented(listOf("off", "on", "auto"), state.config?.rlm ?: "off") {
                    store.patchConfig(JSONObject().put("rlm", it))
                }
            })
        }

        Section("Budgets & limits")
        SettingsCard {
            Row_("Context window", "Override the model's window (0 = engine-resolved truth).", {
                AppliedFieldRow("ctx", state.config?.ctxWindow?.toString().orEmpty(), "auto", { v ->
                    v.trim().toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("ctx_window", n)) }
                })
            })
            CardDivider()
            Row_("Max output tokens", "Override the per-request output cap (0 = model default).", {
                AppliedFieldRow("maxout", state.config?.maxOutput?.toString().orEmpty(), "model default", { v ->
                    v.trim().toIntOrNull()?.let { n -> store.patchConfig(JSONObject().put("max_output", n)) }
                })
            })
            CardDivider()
            Row_("Session budget (USD)", "The engine refuses new turns past the cap (0 = uncapped).", {
                AppliedFieldRow("budget", state.config?.budgetUsd?.takeIf { it > 0 }?.toString().orEmpty(), "uncapped", { v ->
                    v.trim().toDoubleOrNull()?.let { n -> store.patchConfig(JSONObject().put("budget_usd", n)) }
                })
            })
        }

        Section("Memory & recall")
        SettingsCard {
            Text(
                "How aggressively saved project memories are pulled into a turn. Leave at 0 to use engine defaults.",
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                color = c.textFaint,
                modifier = Modifier.padding(bottom = 6.dp),
            )
            Row_("Recall similarity floor", "Minimum match score for a memory to be recalled (0 = default 0.30).", {
                AppliedFieldRow("recall_min", state.config?.recallMin?.takeIf { it > 0 }?.toString().orEmpty(), "default 0.30", { v ->
                    v.trim().toDoubleOrNull()?.let { n ->
                        store.patchConfig(JSONObject().put("recall_min", n.coerceIn(0.0, 1.0)))
                    }
                })
            })
            CardDivider()
            Row_("Recall token budget", "Max tokens of memories injected per turn (0 = auto ≈2% of window, 1k–16k).", {
                AppliedFieldRow("recall_budget", state.config?.recallBudget?.takeIf { it > 0 }?.toString().orEmpty(), "auto", { v ->
                    v.trim().toLongOrNull()?.let { n ->
                        store.patchConfig(JSONObject().put("recall_budget", n.coerceIn(0L, 16_000L)))
                    }
                })
            })
            CardDivider()
            Row_("Embeddings — provider", "Optional OpenAI-compatible provider for memory embeddings. Blank = bundled offline embedder.", {
                AppliedFieldRow("embed_provider", state.config?.embedProvider.orEmpty(), "bundled offline embedder", { v ->
                    store.patchConfig(JSONObject().put("embed_provider", v.trim()))
                })
            })
            CardDivider()
            Row_("Embeddings — model", "Embedding model id used for memory vectors.", {
                AppliedFieldRow("embed_model", state.config?.embedModel.orEmpty(), "default", { v ->
                    store.patchConfig(JSONObject().put("embed_model", v.trim()))
                })
            })
        }

        VectorEngineCard(store)
        PhoneToolsCard(store)
        EngineEnvCard(store)
        Spacer(Modifier.height(24.dp))
    }
}

// ── Phone tools + engine knobs ──────────────────────────────────

@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
private fun PhoneToolsCard(store: AppStore) {
    val c = LocalSofuuColors.current
    var tools by remember { mutableStateOf<List<String>>(emptyList()) }
    LaunchedEffect(Unit) {
        while (true) {
            tools = store.engine.hostTools.toList()
            delay(500)
        }
    }

    Section("Phone tools")
    SettingsCard {
        Text(
            "Native capabilities registered on the engine at boot. Obey the permission profile and appear in /tools.",
            fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 10.dp),
        )
        if (tools.isEmpty()) {
            Text("None registered yet.", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
        } else {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                for (t in tools) {
                    Box(
                        Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(c.monoFieldBg)
                            .border(0.5.dp, c.border, RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 5.dp),
                    ) {
                        Text(t, fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold, color = c.text, fontFamily = FontFamily.Monospace)
                    }
                }
            }
        }
    }
}

@Composable
private fun EngineEnvCard(store: AppStore) {
    val c = LocalSofuuColors.current
    val env = store.runtimeEnv

    Section("Engine controls")
    SettingsCard {
        Text(
            "Runtime environment switches set before the engine boots. Most take effect on the next message.",
            fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 8.dp),
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
        CardDivider()
        ToggleRow("Shell", "Keep bash listed but refuse every invocation. Does not stop an MCP server that shells out itself.", noShell)
        { noShell = it; env.noShell = it; store.applyRuntimeEnv() }
        CardDivider()
        ToggleRow("Context gates (ML)", "Drop the whole sofuu.ml gate layer. Needs a restart.", noMl)
        { noMl = it; env.noMl = it; store.restartEngineForEnv() }
        CardDivider()
        ToggleRow("Allow private addresses", "Let web.open reach loopback and LAN addresses.", allowInternal)
        { allowInternal = it; env.webAllowInternal = it; store.applyRuntimeEnv() }
        CardDivider()
        Row_("Memory backend", "Auto uses the semantic projector when present, else hash-768.") {
            Segmented(listOf("auto", "hash", "semantic"), backend.ifEmpty { "auto" }) {
                backend = it; env.memoryBackend = if (it == "auto") "" else it
                store.applyRuntimeEnv()
            }
        }
        CardDivider()
        Row_("Web search engine", "Default engine for web_search. Blank picks the keyless one.") {
            Segmented(listOf("auto", "brave", "tavily"), engine.ifEmpty { "auto" }) {
                engine = it; env.webEngine = if (it == "auto") "" else it
                store.applyRuntimeEnv()
            }
        }
        CardDivider()
        Row_("Turn step budget", "Tool rounds allowed per turn. 0 uses the engine default (200).") {
            TextFieldRow(
                "maxsteps",
                if (env.chatMaxSteps > 0) env.chatMaxSteps.toString() else "",
                "200 (engine default)",
                onChange = {
                    val n = it.filter { ch -> ch.isDigit() }.toIntOrNull() ?: 0
                    env.chatMaxSteps = n
                    store.applyRuntimeEnv()
                },
            )
        }
        CardDivider()
        Row_("Stalled-request timeout", "Seconds before an unresponsive endpoint gives up. 0 uses the engine default (300).") {
            TextFieldRow(
                "stall",
                if (env.stallTimeoutSecs > 0) env.stallTimeoutSecs.toString() else "",
                "300 (engine default)",
                onChange = {
                    val n = it.filter { ch -> ch.isDigit() }.toIntOrNull() ?: 0
                    env.stallTimeoutSecs = n
                    store.applyRuntimeEnv()
                },
            )
        }
        CardDivider()
        Row_("CPU threads for inference", "Kernel-pool threads decoding locally. 0 = auto (4). Apply restarts.") {
            AppliedFieldRow(
                "threads",
                if (env.numThreads > 0) env.numThreads.toString() else "",
                "auto (4)",
                onApply = {
                    env.numThreads = it.filter { ch -> ch.isDigit() }.toIntOrNull() ?: 0
                    store.restartEngineForEnv()
                },
            )
        }
        CardDivider()
        Row_("Prefill threads", "Threads for batched prompt pass. 0 = auto (8). Apply restarts.") {
            AppliedFieldRow(
                "prefill",
                if (env.prefillThreads > 0) env.prefillThreads.toString() else "",
                "auto (8)",
                onApply = {
                    env.prefillThreads = it.filter { ch -> ch.isDigit() }.toIntOrNull() ?: 0
                    store.restartEngineForEnv()
                },
            )
        }
        CardDivider()
        Row_("Decode threads", "Threads for token generation. 0 = auto (4). Apply restarts.") {
            AppliedFieldRow(
                "decode",
                if (env.decodeThreads > 0) env.decodeThreads.toString() else "",
                "auto (4)",
                onApply = {
                    env.decodeThreads = it.filter { ch -> ch.isDigit() }.toIntOrNull() ?: 0
                    store.restartEngineForEnv()
                },
            )
        }
        CardDivider()
        ToggleRow("Phase metrics", "Write per-token engine split to files/phase-metrics.log every 256 tokens. Restart to apply.", phaseMetrics)
        { phaseMetrics = it; env.phaseMetrics = it; store.restartEngineForEnv() }
    }
}

@Composable
private fun ToggleRow(label: String, desc: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .clip(RoundedCornerShape(8.dp))
            .clickable { onChange(!checked) }
            .padding(vertical = 8.dp, horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(label, fontSize = 14.5.sp, fontWeight = FontWeight.Bold, color = c.text)
            Text(desc, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(top = 2.dp))
        }
        Switch(
            checked = checked,
            onCheckedChange = onChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = c.accentContrast,
                checkedTrackColor = c.accent,
                checkedBorderColor = Color.Transparent,
                uncheckedThumbColor = c.textFaint,
                uncheckedTrackColor = c.toggleOff,
                uncheckedBorderColor = c.border,
            ),
        )
    }
}

// ── Appearance ──────────────────────────────────────────────────

@Composable
private fun AppearancePane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    Column {
        Section("Theme & Display", isFirst = true)
        SettingsCard {
            Row_("Theme", "Follows the system unless you pick one.", {
                Segmented(listOf("light", "dark", "system"), state.theme) { store.setTheme(it) }
            })
            CardDivider()
            Row_("Cloud background", "Procedural animated ambient gradient backdrop.", {
                Segmented(listOf("on", "off"), if (state.cloudBg) "on" else "off") { store.setCloudBg(it == "on") }
            })
            CardDivider()
            Row_("Chat font size", "Body text size in the chat pane.", {
                Segmented(listOf("13", "15", "17", "19"), state.chatFontSize.toString()) { store.setChatFontSize(it.toInt()) }
            })
            Spacer(Modifier.height(6.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(c.monoFieldBg)
                    .border(1.dp, c.border, RoundedCornerShape(8.dp))
                    .padding(10.dp),
            ) {
                Text(
                    "Sample chat preview at ${state.chatFontSize}sp",
                    fontSize = state.chatFontSize.sp,
                    fontWeight = FontWeight.Medium,
                    color = c.text,
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

// ── Models ──────────────────────────────────────────────────────

@Composable
private fun ModelsPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val state = store.state
    val pickModelFile = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> if (uri != null) store.importLocalModel(uri) }

    Column {
        Section("On this device", isFirst = true)
        SettingsCard {
            Text(
                "Local models (.gguf) run entirely on the phone — no network and no API key.",
                fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 10.dp),
            )
            if (state.localModels.isEmpty()) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(c.monoFieldBg)
                        .border(1.dp, c.border, RoundedCornerShape(8.dp))
                        .padding(14.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        "No local models installed yet.",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        color = c.textFaint,
                    )
                }
            } else {
                state.localModels.forEachIndexed { idx, path ->
                    if (idx > 0) CardDivider()
                    val label = state.localModelLabels[path] ?: com.sofuu.store.fallbackModelLabel(path)
                    val active = state.config?.model == path
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (active) c.bgCardHover else Color.Transparent)
                            .clickable { store.pickLocalModel(path) }
                            .padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                label,
                                fontSize = 13.5.sp,
                                fontWeight = if (active) FontWeight.Bold else FontWeight.SemiBold,
                                color = if (active) c.accent else c.text,
                                fontFamily = FontFamily.Monospace,
                            )
                            if (active) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.padding(top = 2.dp),
                                ) {
                                    Box(
                                        Modifier
                                            .size(6.dp)
                                            .clip(RoundedCornerShape(3.dp))
                                            .background(c.ok),
                                    )
                                    Spacer(Modifier.width(6.dp))
                                    Text("Active · running on device", fontSize = 11.5.sp, fontWeight = FontWeight.SemiBold, color = c.ok)
                                }
                            }
                        }
                        if (active) {
                            Text("✓", color = c.accent, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                        } else {
                            Icon(
                                Icons2.Delete,
                                "Remove",
                                tint = c.textFaint,
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(RoundedCornerShape(18.dp))
                                    .clickable { store.deleteLocalModel(path) }
                                    .padding(8.dp),
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(44.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(c.accent)
                    .clickable { pickModelFile.launch(arrayOf("*/*")) },
                contentAlignment = Alignment.Center,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons2.Folder, null, tint = c.accentContrast, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "Import a model (.gguf)",
                        fontSize = 13.5.sp,
                        fontWeight = FontWeight.Bold,
                        color = c.accentContrast,
                    )
                }
            }
        }

        Section("Hosted providers")
        val providers = state.config?.providers ?: emptyList()
        if (providers.isEmpty()) {
            SettingsCard {
                Text("No hosted providers configured. Add one in the Providers tab.", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            }
        } else {
            providers.forEach { p ->
                SettingsCard(modifier = Modifier.padding(bottom = 10.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.weight(1f)) {
                            Text(p.name, fontSize = 15.sp, fontWeight = FontWeight.Bold, color = c.text)
                            Text(p.endpoint, fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint, maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
                        }
                        if (p.name == state.config?.active) {
                            Box(
                                Modifier
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(c.accent.copy(alpha = 0.15f))
                                    .padding(horizontal = 6.dp, vertical = 3.dp),
                            ) {
                                Text("ACTIVE", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = c.accent)
                            }
                        }
                    }
                    CardDivider()
                    val models = (state.modelsCache[p.name]?.models ?: emptyList())
                    if (models.isEmpty()) {
                        Text("No models cached yet. Tap Refresh below to fetch.", fontSize = 12.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                    } else {
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .heightIn(max = 220.dp)
                                .verticalScroll(rememberScrollState()),
                        ) {
                            models.forEach { m ->
                                val active = m == p.model && p.name == state.config?.active
                                Row(
                                    Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(6.dp))
                                        .background(if (active) c.bgCardHover else Color.Transparent)
                                        .clickable { store.pickModel(p.name, m) }
                                        .padding(horizontal = 8.dp, vertical = 7.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        m,
                                        fontSize = 13.sp,
                                        fontWeight = if (active) FontWeight.Bold else FontWeight.SemiBold,
                                        color = if (active) c.accent else c.text,
                                        fontFamily = FontFamily.Monospace,
                                        modifier = Modifier.weight(1f),
                                    )
                                    if (active) {
                                        Text("✓", color = c.accent, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        SmallButton(c.accent, c.accentContrast, "Refresh Model Lists") {
            store.refreshModelCache()
        }
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
    // removeProvider returns the refusal reason (e.g. "cannot remove the
    // active provider"); the pane used to throw it away, so the × looked
    // broken when it was actually protecting you.
    var removeNote by remember { mutableStateOf<String?>(null) }
    var editing by remember { mutableStateOf(emptySet<String>()) }

    Column {
        Section("Add provider", isFirst = true)
        SettingsCard {
            val fieldGap = Modifier.padding(bottom = 8.dp)
            TextFieldRow("name", newName, "Name (e.g. openai, anthropic)", { newName = it }, fieldGap)
            TextFieldRow("endpoint", newEndpoint, "Endpoint (https://…/v1/chat/completions)", { newEndpoint = it }, fieldGap)
            TextFieldRow("model", newModel, "Default model ID (e.g. gpt-4o, claude-3-5-sonnet)", { newModel = it }, fieldGap)
            TextFieldRow("key", newKey, "API key (stored securely in Keystore)", { newKey = it }, fieldGap)

            Box(
                Modifier
                    .fillMaxWidth()
                    .height(44.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(c.accent)
                    .clickable {
                        val name = newName.trim()
                        val endpoint = newEndpoint.trim()
                        val model = newModel.trim()
                        addNote = null
                        if (name.isEmpty() || endpoint.isEmpty()) {
                            addError = "Name and endpoint are required."
                        } else {
                            val providers = (state.config?.providers ?: emptyList()).map { it.toJson() }.toMutableList()
                            providers.add(
                                JSONObject().put("name", name).put("endpoint", endpoint)
                                    .put("model", model).put("profile", ""),
                            )
                            val patch = JSONObject().put("providers", org.json.JSONArray(providers))
                            if (model.isNotEmpty()) {
                                patch.put("active", name).put("provider", name).put("model", model)
                                addNote = null
                            } else {
                                addNote = "Added $name — open the model sheet in the composer, pick one of its models, and it becomes active."
                            }
                            store.patchConfig(patch)
                            if (newKey.isNotBlank()) store.setProviderKey(name, newKey.trim())
                            newName = ""; newEndpoint = ""; newModel = ""; newKey = ""
                            addError = null
                        }
                    },
                contentAlignment = Alignment.Center,
            ) {
                Text("Add Provider", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = c.accentContrast)
            }
            addError?.let {
                Text(it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.errStrong, modifier = Modifier.padding(top = 8.dp))
            }
            addNote?.let {
                Text(it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.ok, modifier = Modifier.padding(top = 8.dp))
            }
        }

        Section("Configured providers")
        removeNote?.let {
            Text(
                it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.errStrong,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        val providers = state.config?.providers ?: emptyList()
        if (providers.isEmpty()) {
            SettingsCard {
                Text("No providers added yet.", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            }
        } else {
            providers.forEach { p ->
                SettingsCard(modifier = Modifier.padding(bottom = 10.dp)) {
                    val isActive = p.name == state.config?.active
                    val isEditing = editing.contains(p.name)
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.weight(1f)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    p.name,
                                    fontSize = 16.sp, fontWeight = FontWeight.Bold, color = c.text,
                                    maxLines = 1,
                                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f, fill = false),
                                )
                                if (isActive) {
                                    Spacer(Modifier.width(8.dp))
                                    Box(
                                        Modifier
                                            .clip(RoundedCornerShape(4.dp))
                                            .background(c.accent.copy(alpha = 0.15f))
                                            .padding(horizontal = 6.dp, vertical = 2.dp),
                                    ) {
                                        Text("ACTIVE", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = c.accent)
                                    }
                                }
                            }
                            Text(p.endpoint, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
                                Box(
                                    Modifier
                                        .size(6.dp)
                                        .clip(RoundedCornerShape(3.dp))
                                        .background(if (p.hasApiKey) c.ok else c.warn),
                                )
                                Spacer(Modifier.width(5.dp))
                                Text(
                                    if (p.hasApiKey) "key saved · Keystore" else "no API key",
                                    fontSize = 11.5.sp,
                                    fontWeight = FontWeight.Medium,
                                    color = if (p.hasApiKey) c.ok else c.warn,
                                )
                            }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            if (!isActive) {
                                Text(
                                    "Remove",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    color = c.errStrong,
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(6.dp))
                                        .clickable { removeNote = store.removeProvider(p.name) }
                                        .padding(horizontal = 8.dp, vertical = 6.dp),
                                )
                            }
                            Text(
                                if (isEditing) "Cancel" else "Edit",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = c.accent,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(6.dp))
                                    .clickable {
                                        editing = if (isEditing) editing - p.name else editing + p.name
                                    }
                                    .padding(horizontal = 8.dp, vertical = 6.dp),
                            )
                        }
                    }

                    if (isEditing) {
                        CardDivider()
                        var ep by remember(p.name, p.endpoint) { mutableStateOf(p.endpoint) }
                        var md by remember(p.name, p.model) { mutableStateOf(p.model) }
                        var pf by remember(p.name, p.profile) { mutableStateOf(p.profile) }
                        TextFieldRow("endpoint", ep, "https://…/v1/chat/completions", { ep = it }, Modifier.padding(bottom = 6.dp))
                        TextFieldRow("model", md, "model id", { md = it }, Modifier.padding(bottom = 6.dp))
                        TextFieldRow("profile", pf, "openai | anthropic | local", { pf = it }, Modifier.padding(bottom = 8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            SmallButton(c.accent, c.accentContrast, "Save Changes") {
                                if (ep.isNotBlank()) {
                                    store.patchProvider(p.name, model = md, endpoint = ep, profile = pf)
                                    editing = editing - p.name
                                }
                            }
                            if (!isActive) {
                                SmallButton(c.bgCardHover, c.accent, "Set as active") {
                                    store.patchProvider(p.name, model = md, endpoint = ep, profile = pf)
                                    store.setActiveProvider(p.name)
                                    editing = editing - p.name
                                }
                            }
                        }
                    }

                    // Key management
                    CardDivider()
                    var keyText by remember(p.name, p.hasApiKey) { mutableStateOf("") }
                    if (keyText.isEmpty() && p.hasApiKey) {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text("API Key", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = c.textMuted)
                            Text(
                                "Replace key",
                                fontSize = 13.sp,
                                color = c.accent,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(6.dp))
                                    .clickable { keyText = " " }
                                    .padding(horizontal = 6.dp, vertical = 4.dp),
                            )
                        }
                    } else {
                        TextFieldRow("key", keyText, "paste new API key", { keyText = it }, Modifier.padding(bottom = 6.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            SmallButton(c.accent, c.accentContrast, "Save key") {
                                if (keyText.isNotBlank()) {
                                    store.setProviderKey(p.name, keyText.trim())
                                    keyText = ""
                                }
                            }
                            SmallButton(c.bgCardHover, c.errStrong, "Clear") {
                                store.setProviderKey(p.name, "")
                                keyText = ""
                            }
                        }
                    }
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

// ── Usage ───────────────────────────────────────────────────────

/** A local GGUF is recorded by its on-device path; cloud ids are names. */
private fun isLocalModelId(id: String): Boolean =
    id.startsWith("/") || id.endsWith(".gguf", ignoreCase = true)

@Composable
private fun UsagePane(store: AppStore) {
    val c = LocalSofuuColors.current
    var usage by remember { mutableStateOf<JSONObject?>(null) }
    var reload by remember { mutableStateOf(0) }
    // Keyed on `reload`: the numbers move as turns land, and the pane used to
    // read once per composition with no way to ask again.
    LaunchedEffect(reload) {
        // LaunchedEffect runs on Main; the engine future can take 10 s.
        usage = withContext(Dispatchers.IO) { store.usageSummary() }
    }
    Column {
        Section("Workspace spend & tokens", isFirst = true)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(bottom = 8.dp)) {
            SmallButton(c.bgCardHover, c.text, "Refresh usage") { reload++ }
        }
        if (usage?.optString("source") == "app") {
            Text(
                "Recorded by the app: this engine build has no QTSQ codec, so its own per-turn usage history cannot persist. " +
                    "Tokens are captured from each finished turn on this install.",
                fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.warn,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        val totals = usage?.optJSONObject("totals")
        SettingsCard {
            val cost = totals?.optDouble("cost") ?: 0.0
            val pt = totals?.optDouble("pt") ?: 0.0
            val ct = totals?.optDouble("ct") ?: 0.0
            Text("TOTAL SPEND", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = c.textFaint, letterSpacing = 0.5.sp)
            Text(
                "$" + "%.4f".format(cost),
                fontSize = 24.sp,
                fontWeight = FontWeight.ExtraBold,
                color = c.text,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(vertical = 4.dp),
            )
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Column {
                    Text("Prompt tokens", fontSize = 11.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                    Text("%,.0f".format(pt), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = c.text, fontFamily = FontFamily.Monospace)
                }
                Column {
                    Text("Completion tokens", fontSize = 11.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                    Text("%,.0f".format(ct), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = c.text, fontFamily = FontFamily.Monospace)
                }
            }
            // The same totals, split the way they are actually spent. A local
            // GGUF records its model as an absolute path, so the split needs
            // no engine change and no pricing table.
            val models = usage?.optJSONArray("models")
            if (models != null && models.length() > 0) {
                var localPt = 0.0; var localCt = 0.0
                var cloudPt = 0.0; var cloudCt = 0.0
                for (i in 0 until models.length()) {
                    val m = models.optJSONObject(i) ?: continue
                    val local = isLocalModelId(m.optString("model"))
                    if (local) {
                        localPt += m.optDouble("pt"); localCt += m.optDouble("ct")
                    } else {
                        cloudPt += m.optDouble("pt"); cloudCt += m.optDouble("ct")
                    }
                }
                CardDivider()
                Text(
                    "Local  %,.0f in · %,.0f out".format(localPt, localCt),
                    fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.textMuted,
                    fontFamily = FontFamily.Monospace, modifier = Modifier.padding(top = 6.dp),
                )
                Text(
                    "Cloud  %,.0f in · %,.0f out".format(cloudPt, cloudCt),
                    fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.textMuted,
                    fontFamily = FontFamily.Monospace, modifier = Modifier.padding(top = 2.dp),
                )
            }
        }

        Section("Per model")
        SettingsCard {
            val models = usage?.optJSONArray("models")
            if (models == null || models.length() == 0) {
                Text(
                    "No per-model usage recorded yet. A turn records its tokens when the engine reports usage (local GGUFs and cloud providers both do).",
                    fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                )
            } else {
                for (i in 0 until models.length()) {
                    if (i > 0) CardDivider()
                    val m = models.optJSONObject(i) ?: continue
                    val id = m.optString("model").ifEmpty { "unknown" }
                    val local = isLocalModelId(id)
                    Row(
                        Modifier.fillMaxWidth().padding(vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                id.substringAfterLast('/'),
                                fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = c.text,
                                maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                            )
                            Text(
                                (if (local) "local" else "cloud") + " · ${m.optInt("turns")} turns",
                                fontSize = 11.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                            )
                        }
                        Text(
                            "%,.0f in · %,.0f out".format(m.optDouble("pt"), m.optDouble("ct")),
                            fontSize = 11.5.sp, fontWeight = FontWeight.SemiBold, color = c.textMuted,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                }
            }
        }

        Section("Per day")
        val days = usage?.optJSONArray("days")
        SettingsCard {
            if (days == null || days.length() == 0) {
                Text("No daily usage recorded yet.", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            } else {
                for (i in 0 until days.length()) {
                    if (i > 0) CardDivider()
                    val d = days.optJSONObject(i) ?: continue
                    Row(
                        Modifier.fillMaxWidth().padding(vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(d.optString("day"), fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.text)
                            Text("${d.optInt("turns")} turns", fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                        }
                        Text(
                            "%,.0f in · %,.0f out".format(d.optDouble("pt"), d.optDouble("ct")),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = c.textMuted,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                }
            }
        }

        Section("Per session")
        val sess = usage?.optJSONArray("sessions")
        SettingsCard {
            if (sess == null || sess.length() == 0) {
                Text("No session records found.", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            } else {
                for (i in 0 until minOf(sess.length(), 20)) {
                    if (i > 0) CardDivider()
                    val d = sess.optJSONObject(i) ?: continue
                    Row(
                        Modifier.fillMaxWidth().padding(vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                d.optString("task").ifEmpty { d.optString("id") }.take(28),
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = c.text,
                                maxLines = 1,
                                overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                            )
                            Text(d.optString("model"), fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint, fontFamily = FontFamily.Monospace)
                        }
                        Text(
                            "$" + "%.4f".format(d.optDouble("cost")),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = c.text,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(24.dp))
    }
}

// ── Brain ───────────────────────────────────────────────────────

@Composable
private fun BrainPane(store: AppStore) {
    val c = LocalSofuuColors.current
    val surface = remember {
        com.sofuu.engine.RuntimeSurface(store.engine) { store.state.config?.brain != false }
    }

    var info by remember { mutableStateOf<JSONObject?>(null) }
    var err by remember { mutableStateOf<String?>(null) }
    var spaces by remember { mutableStateOf<JSONObject?>(null) }
    var reload by remember { mutableStateOf(0) }
    val scope = rememberCoroutineScope()
    var hits by remember { mutableStateOf<org.json.JSONArray?>(null) }
    var recalled by remember { mutableStateOf("") }
    var decayArmed by remember { mutableStateOf(false) }
    var qtsqState by remember { mutableStateOf<JSONObject?>(null) }

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
                    runCatching { surface.embedImageFile(img).get(30, TimeUnit.SECONDS) }.getOrNull()
                }
                if (r?.optBoolean("ok") == true) {
                    imgNote = "${img.substringAfterLast('/')} · ${r.optInt("dim")}-dim img1 vector"
                    imgPath = img
                } else {
                    imgNote = "Image embedding failed: ${r?.optString("error") ?: "unavailable"}"
                    imgPath = null
                }
            }
        }
    }

    var simA by remember { mutableStateOf("the cat sat down") }
    var simB by remember { mutableStateOf("a feline rested") }
    var sim by remember { mutableStateOf<JSONObject?>(null) }

    LaunchedEffect(Unit) {
        qtsqState = withContext(Dispatchers.IO) {
            runCatching { surface.qtsqState() }.getOrNull()
        }
    }
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

    Column {
        Section("Stored memories", isFirst = true)
        // Where the data actually is: one QTSQ file per workspace, written by
        // the CMA and shared by every session of this workspace.
        (store.state.projectDir ?: store.engine.projectDir)?.let { ws ->
            Text(
                "Stored in ${ws.substringAfterLast('/')}/.sofuu/brain/brain.qtsq — one file per workspace, shared by all its chats.",
                fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        if (qtsqState?.optBoolean("canPersist") == false) {
            Text(
                "This engine build cannot persist memory (probe: ${qtsqState?.optString("reason").orEmpty().ifEmpty { "write failed" }}). " +
                    "Memories live in RAM for this run and the engine cannot store sessions.",
                fontSize = 11.5.sp, fontWeight = FontWeight.SemiBold, color = c.warn,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        SettingsCard {
            when {
                err != null -> Text(err!!, fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.err)
                info == null -> Text("Reading the brain…", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                else -> {
                    val i = info!!
                    Text(
                        "${i.optLong("count")} memories",
                        fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, color = c.text, fontFamily = FontFamily.Monospace,
                    )
                    Text(
                        "space ${i.optString("backend") ?: "?"} · ${i.optInt("dim")}-dim" +
                            if (i.optBoolean("semantic")) " · semantic channel on" else "",
                        fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, fontFamily = FontFamily.Monospace,
                        modifier = Modifier.padding(top = 2.dp, bottom = 10.dp),
                    )
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                    fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.warn, modifier = Modifier.padding(top = 6.dp),
                )
            }
        }

        Section("Search & recall")
        SettingsCard {
            TextFieldRow("recall", recalled, "Search memories (e.g. user preferences, rules)", onChange = { recalled = it })
            hits?.let { h ->
                CardDivider()
                if (h.length() == 0) {
                    Text("No memories matched.", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
                }
                for (i in 0 until h.length()) {
                    val o = h.optJSONObject(i) ?: continue
                    val id = o.optInt("id", -1)
                    if (i > 0) CardDivider()
                    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                o.optString("text").ifEmpty { o.optString("kind") },
                                fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.text,
                            )
                            Text(
                                "score ${"%.3f".format(o.optDouble("score"))} · ${o.optString("tier")}",
                                fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint, fontFamily = FontFamily.Monospace,
                            )
                        }
                        if (id >= 0) {
                            Icon(
                                Icons2.Delete, "Forget", tint = c.textFaint,
                                modifier = Modifier
                                    .size(32.dp)
                                    .clip(RoundedCornerShape(16.dp))
                                    .clickable {
                                        scope.launch {
                                            val r = withContext(Dispatchers.IO) { surface.brainForget(id) }
                                            err = r?.optString("error")?.takeIf { it.isNotEmpty() }
                                            hits = null
                                            reload++
                                        }
                                    }
                                    .padding(6.dp),
                            )
                        }
                    }
                }
            }
        }

        Section("Embedding spaces")
        SettingsCard {
            // embedInfo()/embedInfoV2() are two FLAT manifests (id, dimension,
            // available, …), not one object with "hash"/"semantic" children —
            // reading the old shape rendered 0 dims and "unavailable" against
            // a healthy engine.
            val sp = spaces?.optJSONObject("info")
            val v2 = spaces?.optJSONObject("v2")
            if (sp == null && v2 == null) {
                Text("No embedding manifest yet.", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            } else {
                if (sp != null) {
                    SpaceLine(
                        c.text, c.err, c.textFaint,
                        sp.optString("id").ifEmpty { "hash" },
                        sp.optInt("dimension"), sp.optBoolean("available", true),
                    )
                }
                if (v2 != null) {
                    CardDivider()
                    SpaceLine(
                        c.text, c.err, c.textFaint,
                        v2.optString("id").ifEmpty { "semantic v2" },
                        v2.optInt("dimension"), v2.optBoolean("available", false),
                    )
                }
            }
        }

        Section("Context gates (sofuu.ml)")
        SettingsCard {
            val m = ml
            if (m == null) {
                Text("The sofuu.ml layer is not loaded in this build.", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint)
            } else if (m.optBoolean("ok") != true) {
                Text(m.optString("error").ifEmpty { "unavailable" }, fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.err)
            } else {
                val info = m.optJSONObject("info")
                val ws = m.optJSONObject("workset")
                Text(
                    "${ws?.optLong("runs") ?: 0} runs · ${ws?.optLong("calls") ?: 0} calls · ${ws?.optLong("segments") ?: 0} segments",
                    fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.text, fontFamily = FontFamily.Monospace,
                )
                // The gate map is `gates` (name → "on"/"off" STRINGS); the
                // pane used to read a non-existent `on` object of booleans,
                // so no gate row ever rendered.
                val gates = info?.optJSONObject("gates")
                if (gates != null) {
                    CardDivider()
                    val keys = gates.keys()
                    while (keys.hasNext()) {
                        val k = keys.next()
                        val st = gates.optString(k)
                        Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                            Text(
                                k, fontSize = 12.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                                fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
                            )
                            Text(
                                st.ifEmpty { "off" }, fontSize = 12.5.sp, fontWeight = FontWeight.Bold,
                                color = if (st == "on") c.accent else c.textMuted,
                            )
                        }
                    }
                }
                val learner = info?.optJSONObject("online")
                if (learner != null && learner.optBoolean("enabled")) {
                    Text(
                        "Online learning is on" + if (learner.optBoolean("pending")) " · a candidate is pending" else "",
                        fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.warn, modifier = Modifier.padding(top = 4.dp),
                    )
                }
                CardDivider()
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                        fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = if (m2 == true) c.ok else c.warn,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
            }
        }

        Section("Gate check")
        SettingsCard {
            Text(
                "Ask what the gates would say about a call before the agent makes it.",
                fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 6.dp),
            )
            TextFieldRow("tool", probeTool, "tool name (e.g. read_file)", onChange = { probeTool = it })
            Spacer(Modifier.height(6.dp))
            TextFieldRow("args", probeArgs, "JSON arguments (e.g. {\"path\":\"README.md\"})", onChange = { probeArgs = it })
            Spacer(Modifier.height(8.dp))
            SmallButton(c.accent, c.accentContrast, "Check Verdict") {
                scope.launch {
                    val parsed = try { JSONObject(probeArgs) } catch (e: Exception) { JSONObject() }
                    verdict = withContext(Dispatchers.IO) { surface.mlCheck(probeTool, parsed) }
                }
            }
            verdict?.let { v ->
                CardDivider()
                val ok = v.optBoolean("ok")
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(6.dp).clip(RoundedCornerShape(3.dp)).background(if (ok) c.ok else c.warn))
                    Spacer(Modifier.width(6.dp))
                    Text(
                        if (ok) "PASS · ${v.optString("source")} · score ${"%.3f".format(v.optDouble("score"))}" +
                            (v.optString("nudge").takeIf { it.isNotEmpty() && it != "null" }?.let { " · nudge: $it" } ?: "")
                        else "BLOCKED · ${v.optString("reason")}",
                        fontSize = 12.5.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (ok) c.ok else c.warn,
                    )
                }
            }
        }

        Section("Garbage collection")
        SettingsCard {
            Text(
                "One runtime lives for the whole process on a phone. Collect memory after large runs or batch embeds.",
                fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 8.dp),
            )
            SmallButton(c.accent, c.accentContrast, "Collect garbage") {
                scope.launch {
                    val okRun = withContext(Dispatchers.IO) { surface.collectGarbage() }
                    gcNote = if (okRun) "Collected successfully." else "The runtime did not expose __sofuu_gc in this build."
                }
            }
            gcNote?.let { Text(it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.ok, modifier = Modifier.padding(top = 6.dp)) }
        }

        Section("Image vectors")
        SettingsCard {
            Text(
                "Embed an image into the multimodal space and compare it against a text query.",
                fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 8.dp),
            )
            SmallButton(c.accent, c.accentContrast, "Pick an image") { pickImage.launch("image/*") }
            imgNote?.let {
                Text(it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.textFaint, modifier = Modifier.padding(top = 6.dp))
            }
            if (imgPath != null) {
                CardDivider()
                TextFieldRow("describe", imgQuery, "what to look for", onChange = { imgQuery = it })
                Spacer(Modifier.height(8.dp))
                SmallButton(c.accent, c.accentContrast, "Compare to text") {
                    scope.launch {
                        imgSim = withContext(Dispatchers.IO) { surface.imageTextSimilarity(imgPath!!, imgQuery) }
                    }
                }
                imgSim?.let { r ->
                    CardDivider()
                    Text(
                        if (r.optBoolean("ok")) "cosine ${"%.4f".format(r.optDouble("similarity"))} · ${r.optInt("dim")}-dim"
                        else r.optString("error").ifEmpty { "spaces do not line up" },
                        fontSize = 13.5.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (r.optBoolean("ok")) c.text else c.warn,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }

        Section("String similarity")
        SettingsCard {
            TextFieldRow("a", simA, "first string", onChange = { simA = it })
            Spacer(Modifier.height(6.dp))
            TextFieldRow("b", simB, "second string", onChange = { simB = it })
            sim?.let {
                CardDivider()
                Text(
                    "cosine similarity: ${"%.4f".format(it.optDouble("similarity"))} · ${it.optInt("dim")}-dim",
                    fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.text, fontFamily = FontFamily.Monospace,
                )
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SpaceLine(text: Color, err: Color, faint: Color, label: String, dim: Int, available: Boolean) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(
            label, fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = if (available) text else err,
            fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
        )
        Text(
            if (available) "$dim-dim" else "artifact unavailable",
            fontSize = 12.sp, fontWeight = FontWeight.Medium, color = if (available) faint else err,
        )
    }
}

@Composable
private fun SmallButton(bg: Color, fg: Color, label: String, onClick: () -> Unit) {
    Box(
        Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bg)
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 9.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            label,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            color = fg,
        )
    }
}

// ── Agents ──────────────────────────────────────────────────────

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
    // 'none' | 'sensitive' | 'all' — the runtime's def.approve. It must be
    // set at define time; approveInteractive at run time only says HOW to
    // ask, never WHETHER to.
    var approveMode by remember { mutableStateOf("sensitive") }

    var task by remember { mutableStateOf("") }
    var answer by remember { mutableStateOf<JSONObject?>(null) }
    var running by remember { mutableStateOf(false) }
    // Run target selected from the registry (the editable name field is a
    // definition input, not a selection).
    var runAgentName by remember { mutableStateOf("") }
    var runSignal by remember { mutableStateOf<String?>(null) }
    /** Approval parked by the headless run, if any. */
    var pending by remember { mutableStateOf<JSONObject?>(null) }

    LaunchedEffect(reload) {
        list = withContext(Dispatchers.IO) { surface.agents() }
    }

    Column {
        Section("Registered agents", isFirst = true)
        val l = list
        if (l == null) {
            SettingsCard { Text("Reading the registry…", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint) }
        } else if (l.length() == 0) {
            SettingsCard { Text("No agents defined yet. Add one below.", fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint) }
        } else {
            for (i in 0 until l.length()) {
                val a = l.optJSONObject(i) ?: continue
                val agentName = a.optString("name")
                SettingsCard(modifier = Modifier.padding(bottom = 8.dp)) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clickable { runAgentName = agentName },
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            agentName,
                            fontSize = 15.sp, fontWeight = FontWeight.Bold, color = c.text,
                            fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f),
                        )
                        Text(
                            if (runAgentName == agentName) "run target" else "select",
                            fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                            color = if (runAgentName == agentName) c.accent else c.textFaint,
                        )
                    }
                    // A.list() reports `tools` as a COUNT and `agents` as an
                    // ARRAY; the old reader expected an array of tool names
                    // and an int, so both lines rendered empty.
                    val toolsVal = a.opt("tools")
                    val toolsText = when (toolsVal) {
                        is org.json.JSONArray ->
                            (0 until toolsVal.length()).joinToString(", ") { toolsVal.optString(it) }
                                .ifEmpty { "no tools" }
                        is Number -> "${toolsVal.toInt()} tools"
                        else -> null
                    }
                    val subsVal = a.opt("agents")
                    val subsText = when (subsVal) {
                        is org.json.JSONArray ->
                            if (subsVal.length() > 0) (0 until subsVal.length()).joinToString(", ") { subsVal.optString(it) } else null
                        is Number -> if (subsVal.toInt() > 0) "${subsVal.toInt()} sub-agent slots" else null
                        else -> null
                    }
                    Text(
                        buildString {
                            toolsText?.let { append(it) }
                            subsText?.let { if (isNotEmpty()) append(" · "); append("sub-agents: $it") }
                            a.optString("provider").takeIf { it.isNotEmpty() }?.let {
                                if (isNotEmpty()) append(" · "); append(it)
                            }
                            if (a.optInt("runs24h") > 0) {
                                if (isNotEmpty()) append(" · "); append("${a.optInt("runs24h")} runs/24h")
                            }
                        }.ifEmpty { "no tools" },
                        fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                }
            }
        }

        Section("Define an agent")
        SettingsCard {
            TextFieldRow("name", name, "Agent name (e.g. reviewer)", onChange = { name = it }, modifier = Modifier.padding(bottom = 6.dp))
            TextFieldRow("system", system, "System prompt (what this agent does)", onChange = { system = it }, modifier = Modifier.padding(bottom = 6.dp))
            TextFieldRow("max steps", maxSteps, "Max steps (e.g. 24)", onChange = { maxSteps = it }, modifier = Modifier.padding(bottom = 8.dp))
            ToggleRow("Web tools", "web_search and web_open", withWeb) { withWeb = it }
            CardDivider()
            ToggleRow("Coding tools", "the 8 built-ins: read/write/edit/grep/glob/dir/bash/todo", withCode) { withCode = it }
            Spacer(Modifier.height(10.dp))
            Text(
                "Approval gate — runs ask before this agent acts",
                fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.textMuted,
                modifier = Modifier.padding(bottom = 6.dp),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("sensitive" to "Writes + shell", "all" to "Every tool", "none" to "Never").forEach { (v, label) ->
                    SmallButton(
                        if (approveMode == v) c.accent else c.bgCardHover,
                        if (approveMode == v) c.accentContrast else c.textMuted,
                        label,
                    ) { approveMode = v }
                }
            }
            Text(
                "The answer is the card in Run headlessly below (Allow / Deny / Always).",
                fontSize = 11.5.sp, fontWeight = FontWeight.Medium, color = c.textFaint,
                modifier = Modifier.padding(top = 6.dp, bottom = 10.dp),
            )
            SmallButton(c.accent, c.accentContrast, "Register Agent") {
                scope.launch {
                    val tools = buildList {
                        if (withWeb) add("web")
                        if (withCode) add("code")
                    }
                    val budget = JSONObject().put("maxSteps", maxSteps.filter { it.isDigit() }.toIntOrNull() ?: 24)
                    val r = withContext(Dispatchers.IO) {
                        surface.defineAgent(name.trim(), system.trim(), tools, budget, approve = approveMode)
                    }
                    note = if (r?.optBoolean("ok") == true) {
                        "${r.optString("name")} registered — the agent can use it by name, and /agents lists it."
                    } else {
                        "Could not register: ${r?.optString("error") ?: "unknown error"}"
                    }
                    reload++
                }
            }
            note?.let {
                Text(it, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = c.ok, modifier = Modifier.padding(top = 8.dp))
            }
        }

        Section("Run headlessly")
        SettingsCard {
            Text(
                "Runs through the agent loop headlessly and returns the final collected answer.",
                fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, modifier = Modifier.padding(bottom = 8.dp),
            )
            val target = runAgentName.ifEmpty { name.trim() }
            Text(
                "Agent: ${target.ifEmpty { "(none — tap one above)" }}",
                fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                color = if (target.isEmpty()) c.warn else c.textMuted,
                modifier = Modifier.padding(bottom = 8.dp),
            )
            TextFieldRow("task", task, "a task to hand the agent", onChange = { task = it }, modifier = Modifier.padding(bottom = 8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SmallButton(
                    if (running) c.bgCardHover else c.accent,
                    if (running) c.textFaint else c.accentContrast,
                    if (running) "Running…" else "Run Task",
                ) {
                    if (running || task.isBlank()) return@SmallButton
                    if (target.isEmpty()) {
                        note = "Pick an agent above (or type a name) first."
                        return@SmallButton
                    }
                    running = true; answer = null; pending = null
                    val signal = "android-agent-" + System.currentTimeMillis()
                    runSignal = signal
                    scope.launch {
                        // approveInteractive parks an approval-gated tool
                        // instead of denying it; drain the queue while the
                        // run waits so the user can answer it.
                        val poller = launch {
                            while (true) {
                                val now = withContext(Dispatchers.IO) {
                                    runCatching { surface.agentPendingApprovals() }.getOrNull()
                                }
                                pending = if (now != null && now.length() > 0) now.optJSONObject(0) else null
                                kotlinx.coroutines.delay(800)
                            }
                        }
                        val r = withContext(Dispatchers.IO) {
                            runCatching {
                                surface.runAgent(target, task.trim(), 300_000, signal)
                                    .get(300, TimeUnit.SECONDS)
                            }.getOrNull()
                        }
                        poller.cancel()
                        pending = null
                        runSignal = null
                        answer = r ?: JSONObject().put("ok", false).put("error", "the run did not finish in 5 minutes")
                        running = false
                    }
                }
                if (running) {
                    SmallButton(c.bgCardHover, c.errStrong, "Cancel") {
                        val sig = runSignal ?: return@SmallButton
                        scope.launch(Dispatchers.IO) { runCatching { surface.cancelAgent(sig) } }
                    }
                }
            }
            pending?.let { p ->
                val pid = p.optString("id")
                CardDivider()
                Text(
                    "Approval needed · ${p.optString("tool").ifEmpty { "tool" }}",
                    fontSize = 13.sp, fontWeight = FontWeight.Bold, color = c.warn,
                )
                Spacer(Modifier.height(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SmallButton(c.bgCardHover, c.text, "Deny") {
                        scope.launch(Dispatchers.IO) { runCatching { surface.resolveAgentApproval(pid, false, false) } }
                        pending = null
                    }
                    SmallButton(c.bgCardHover, c.text, "Always") {
                        scope.launch(Dispatchers.IO) { runCatching { surface.resolveAgentApproval(pid, true, true) } }
                        pending = null
                    }
                    SmallButton(c.accent, c.accentContrast, "Allow once") {
                        scope.launch(Dispatchers.IO) { runCatching { surface.resolveAgentApproval(pid, true, false) } }
                        pending = null
                    }
                }
            }
            answer?.let { a ->
                CardDivider()
                if (a.optBoolean("ok") != true) {
                    Text(a.optString("error").ifEmpty { "run failed" }, fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, color = c.err)
                } else {
                    Text(
                        "${a.optInt("steps")} steps" + (a.optString("stopped").takeIf { it.isNotEmpty() && it != "null" }
                            ?.let { " · stopped: $it" } ?: ""),
                        fontSize = 12.sp, fontWeight = FontWeight.Medium, color = c.textFaint, fontFamily = FontFamily.Monospace,
                    )
                    Text(
                        a.optString("answer"),
                        fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = c.text, modifier = Modifier.padding(top = 4.dp),
                    )
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}
