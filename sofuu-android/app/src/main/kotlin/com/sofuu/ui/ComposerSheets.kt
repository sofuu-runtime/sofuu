package com.sofuu.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.store.AppStore
import com.sofuu.store.fallbackModelLabel
import com.sofuu.ui.theme.LocalSofuuColors
import org.json.JSONArray
import org.json.JSONObject

/** Bottom sheets backing the composer's pills: model, effort, context
 *  layers, turn mode, permissions. Same options as the desktop popovers. */
@Composable
fun ComposerSheets(store: AppStore, which: String, onClose: () -> Unit) {
    val c = LocalSofuuColors.current
    val state = store.state
    Box(Modifier.fillMaxSize().background(c.backdropVeil).clickable { onClose() }) {
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(c.bgRaised, RoundedCornerShape(topStart = 18.dp, topEnd = 18.dp))
                .clickable { }
                .padding(top = 18.dp, bottom = 12.dp)
                .padding(horizontal = 16.dp),
        ) {
            when (which) {
                "model" -> ModelSheet(store)
                "effort" -> EffortSheet(store)
                "layers" -> LayersSheet(store)
                "mode" -> ModeSheet(store)
                "permissions" -> PermissionsSheet(store)
            }
        }
    }
}

@Composable
private fun SheetTitle(t: String) {
    val c = LocalSofuuColors.current
    Text(t, fontSize = 13.sp, color = c.textMuted, modifier = Modifier.padding(bottom = 8.dp))
}

@Composable
private fun SheetButton(label: String, bg: Color, fg: Color, onClick: () -> Unit) {
    Box(
        Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(bg)
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 12.dp),
    ) {
        Text(label, fontSize = 13.sp, color = fg)
    }
}

@Composable
private fun SheetRow(label: String, hint: String?, selected: Boolean, onClick: () -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(if (selected) c.bgCardHover else Color.Transparent)
            .clickable { onClick() }
            .heightIn(min = 48.dp)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, fontSize = 14.sp, color = c.text)
            if (hint != null) Text(hint, fontSize = 11.sp, color = c.textFaint)
        }
        if (selected) Text("✓", color = c.accent, fontSize = 14.sp)
    }
}

@Composable
private fun ModelSheet(store: AppStore) {
    val state = store.state
    val c = LocalSofuuColors.current
    var provider by remember { mutableStateOf<String?>(null) }
    var search by remember { mutableStateOf("") }
    var manual by remember { mutableStateOf("") }
    val providers = state.config?.providers ?: emptyList()

    if (provider == null) {
        SheetTitle("Select a model")
        // On-device GGUF models route by absolute path (rt/ai.rs sends local
        // models to the in-process engine), so they need no provider entry —
        // but not being listed here meant the model sheet could never reach
        // them, and a local-only setup saw just "＋ Add provider".
        if (state.localModels.isNotEmpty()) {
            state.localModels.forEach { path ->
                val label = state.localModelLabels[path] ?: fallbackModelLabel(path)
                SheetRow(label, null, path == state.config?.model) {
                    store.pickLocalModel(path)
                    store.closeSheet()
                }
            }
            Spacer(Modifier.height(4.dp))
        }
        if (providers.isNotEmpty()) SheetTitle("Cloud providers")
        providers.forEach { p ->
            SheetRow(p.name, p.model.ifEmpty { null }, p.name == state.config?.active) { provider = p.name }
        }
        Spacer(Modifier.height(4.dp))
        // Always offered, not only when the list is empty — otherwise a
        // configured provider leaves no way to add a second one from here.
        SheetRow("＋ Add provider", null, false) { onCloseSheet(store) }
        // Add a model FILE (the on-device list above is import/delete); a
        // picked .gguf/.safetensors lands in filesDir/models and appears
        // here on the next open.
        SheetRow("＋ Add a model file from this device", null, false) {
            store.closeSheet()
            store.openSettings(true, "models")
        }
    } else {
        val prov = provider!!
        val entry = providers.firstOrNull { it.name == prov }
        SheetRow("‹ Providers", null, false) { provider = null }
        Spacer(Modifier.height(4.dp))
        SheetTitle(prov)
        val cached = state.modelsCache[prov]
        val saved = entry?.model.orEmpty()
        val rows = ArrayList<Pair<String, String?>>()
        if (saved.isNotEmpty()) rows.add(saved to if (prov == state.config?.active) "current" else "saved")
        cached?.models?.forEach { m -> if (rows.none { it.first == m }) rows.add(m to null) }
        val filtered = rows.filter { search.isEmpty() || it.first.contains(search, ignoreCase = true) }
        if (rows.size > 1) {
            Spacer(Modifier.height(6.dp))
            androidx.compose.foundation.text.BasicTextField(
                value = search,
                onValueChange = { search = it },
                singleLine = true,
                textStyle = androidx.compose.ui.text.TextStyle(color = c.text, fontSize = 13.sp),
                cursorBrush = androidx.compose.ui.graphics.SolidColor(c.accent),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 44.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(c.monoFieldBg)
                    .padding(horizontal = 12.dp, vertical = 12.dp),
                decorationBox = { inner ->
                    androidx.compose.foundation.layout.Box {
                        if (search.isEmpty()) {
                            Text("Search models…", color = c.textFaint, fontSize = 13.sp)
                        }
                        inner()
                    }
                },
            )
        }
        if (cached?.error != null) {
            Text(cached.error, fontSize = 12.sp, color = c.err, modifier = Modifier.padding(top = 8.dp, start = 4.dp))
        }
        if (rows.isEmpty() || (filtered.isEmpty() && search.isNotEmpty())) {
            Text(
                if (rows.isEmpty())
                    "This provider publishes no model list — type a model name below."
                else
                    "No model matches \"${search.trim()}\" — type a model name below.",
                fontSize = 13.sp,
                color = c.textFaint,
                modifier = Modifier.padding(12.dp),
            )
        }
        if (rows.isNotEmpty()) {
            LazyColumn(Modifier.heightIn(max = 320.dp).padding(top = 4.dp)) {
                items(filtered) { (model, hint) ->
                    SheetRow(model, hint, model == saved && prov == state.config?.active) {
                        store.pickModel(prov, model)
                    }
                }
            }
        }
        // The manual form is the ONLY way to set a model on an endpoint with
        // no listing (Anthropic has no /models route; a local Ollama may be
        // down). It used to be gated on `rows.isEmpty()` — but `rows` already
        // contains the saved model, so any provider that had one showed
        // rows.size == 1 and the picker became a dead end. Gate on whether a
        // LIST exists, and also offer it whenever a search matches nothing,
        // so a typo can never strand the user.
        if (rows.size <= 1 || (filtered.isEmpty() && search.isNotEmpty())) {
            Spacer(Modifier.height(8.dp))
            SheetTitle("Set a model by hand")
            androidx.compose.foundation.text.BasicTextField(
                value = manual,
                onValueChange = { manual = it },
                singleLine = true,
                textStyle = androidx.compose.ui.text.TextStyle(color = c.text, fontSize = 13.sp),
                cursorBrush = androidx.compose.ui.graphics.SolidColor(c.accent),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 44.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(c.monoFieldBg)
                    .padding(horizontal = 12.dp, vertical = 12.dp),
                decorationBox = { inner ->
                    androidx.compose.foundation.layout.Box {
                        if (manual.isEmpty()) Text("Type a model name…", color = c.textFaint, fontSize = 13.sp)
                        inner()
                    }
                },
            )
            Spacer(Modifier.height(8.dp))
            SheetButton("Use this model", c.accent, Color.White) {
                if (manual.isNotBlank()) store.pickModel(prov, manual.trim())
            }
        }
    }
}

private fun onCloseSheet(store: AppStore) {
    // The composer sheet renders ABOVE Settings in the z-order — opening
    // Settings without first dropping the sheet buried the providers pane
    // under the still-visible sheet.
    store.closeSheet()
    store.openSettings(true, "providers")
}

@Composable
private fun EffortSheet(store: AppStore) {
    SheetTitle("Thinking effort")
    listOf("off", "low", "medium", "high", "max").forEach { e ->
        SheetRow(
            if (e == "off") "Off" else e.replaceFirstChar { it.uppercase() },
            null,
            store.state.config?.effort == e || (store.state.config?.effort == "" && e == "off"),
        ) { store.patchConfig(JSONObject().put("effort", if (e == "off") "" else e)) }
    }
}

@Composable
private fun LayersSheet(store: AppStore) {
    val state = store.state
    val cfg = state.config
    SheetTitle("Context layers")
    ToggleRow("Project memory", "Recall relevant saved memories", cfg?.brain != false) {
        store.patchConfig(JSONObject().put("brain", !(cfg?.brain != false)))
    }
    ToggleRow("Smart context filters", "Adaptive context relevance & compaction", cfg?.ml != false) {
        store.patchConfig(JSONObject().put("ml", !(cfg?.ml != false)))
    }
    val rlm = cfg?.rlm ?: "off"
    ToggleRow("Reasoning sandbox", rlm, rlm == "on") {
        store.patchConfig(JSONObject().put("rlm", if (rlm == "on") "off" else "on"))
    }
}

@Composable
private fun ToggleRow(label: String, hint: String, on: Boolean, onToggle: () -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier
            .fillMaxWidth()
            .clickable { onToggle() }
            .heightIn(min = 52.dp)
            .padding(horizontal = 4.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, fontSize = 14.sp, color = c.text)
            Text(hint, fontSize = 11.sp, color = c.textFaint)
        }
        Box(
            Modifier
                .size(width = 40.dp, height = 22.dp)
                .clip(RoundedCornerShape(11.dp))
                .background(if (on) c.accent else c.toggleOff),
            contentAlignment = if (on) Alignment.CenterEnd else Alignment.CenterStart,
        ) {
            Box(
                Modifier
                    .padding(2.dp)
                    .size(18.dp)
                    .clip(RoundedCornerShape(9.dp))
                    .background(Color.White),
            )
        }
    }
}

@Composable
private fun ModeSheet(store: AppStore) {
    val state = store.state
    SheetTitle("Turn mode")
    SheetRow("Agent", "Autonomous tools & file edits", state.mode == "agent") { store.setMode("agent") }
    SheetRow("Chat only", "Conversation only, no tools", state.mode == "chat") { store.setMode("chat") }
}

@Composable
private fun PermissionsSheet(store: AppStore) {
    val state = store.state
    SheetTitle("Tool permissions")
    SheetRow("Full access", "All tools run automatically", state.permissions == "full") { store.setPermissions("full") }
    SheetRow("Edit only", "Read files & apply edits", state.permissions == "edit") { store.setPermissions("edit") }
    SheetRow("Plan mode", "Plan & analyze, no changes", state.permissions == "plan") { store.setPermissions("plan") }
    SheetRow("Ask me", "Approve each tool execution", state.permissions == "prompt") { store.setPermissions("prompt") }
}
