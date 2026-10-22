package com.sofuu.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.store.AppStore
import com.sofuu.store.Approval
import com.sofuu.ui.chat.ChatScreen
import com.sofuu.ui.chat.DiffView
import com.sofuu.ui.chat.PhaseLine
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.composer.Composer
import com.sofuu.ui.settings.SettingsScreen
import com.sofuu.ui.theme.LocalSofuuColors

/** The app shell: drawer + topbar + chat + composer + overlays — the
 *  mobile mapping of the desktop's Sidebar/TopBar/App composition. */
@Composable
fun SofuuApp(store: AppStore, onPickFiles: () -> Unit, onRequestMic: () -> Boolean = { false }) {
    val state = store.state
    val c = LocalSofuuColors.current
    val density = LocalDensity.current
    val historyOpen = state.historyOpen

    val statusTop = WindowInsets.statusBars.asPaddingValues().calculateTopPadding()
    val bottomPadding = WindowInsets.safeDrawing.asPaddingValues().calculateBottomPadding()

    var topBarPx by remember { mutableStateOf(0) }
    var textFieldPx by remember { mutableStateOf(0) }
    var composerTotalPx by remember { mutableStateOf(0) }

    val topBarHeight = with(density) { topBarPx.toDp() }
    val textFieldHeight = with(density) { textFieldPx.toDp() }
    val composerTotalHeight = with(density) { composerTotalPx.toDp() }

    val effectiveTopBand = if (topBarPx > 0) (statusTop + topBarHeight) else (statusTop + 48.dp)
    // 6.dp is composer's bottom vertical padding inside Composer.kt
    val effectiveBottomBand = if (textFieldPx > 0) (bottomPadding + 6.dp + textFieldHeight) else (bottomPadding + 110.dp)
    val effectiveComposerTotal = if (composerTotalPx > 0) (bottomPadding + composerTotalHeight) else (bottomPadding + 160.dp)

    Box(Modifier.fillMaxSize().background(c.bgApp)) {
        LaunchedEffect(state.cloudBg) {
            android.util.Log.i("sofuu-cloud", "cloudBg enabled=${state.cloudBg}")
        }
        if (state.cloudBg) CloudBackground()

        // ── Main Content Viewport (chat / starter prompt / preview) ──
        // The chat fills the full screen and scrolls UNDER the top bar
        // and composer gradients, so content flows seamlessly edge-to-edge.
        Box(Modifier.fillMaxSize()) {
            if (state.bootError != null) {
                Box(Modifier.fillMaxWidth().statusBarsPadding().padding(top = topBarHeight).align(Alignment.TopCenter)) {
                    Banner(state.bootError!!, isError = true)
                }
            }
            if (state.preview != null) {
                Box(Modifier.fillMaxWidth().statusBarsPadding().padding(top = topBarHeight).align(Alignment.TopCenter)) {
                    PreviewBar(state.preview!!.label) { store.closePreview() }
                }
            }
            if (state.messages.isEmpty() && state.preview == null) {
                Box(
                    Modifier
                        .fillMaxSize()
                        .padding(top = effectiveTopBand, bottom = effectiveComposerTotal),
                ) {
                    HomePrompt(
                        state.projectDir?.let { "Workspace: ${it.substringAfterLast('/')}" },
                        onPromptSelect = { store.submit(it, emptyList()) },
                    )
                }
            } else {
                ChatScreen(
                    messages = state.messages,
                    phase = state.phase,
                    streaming = state.streaming,
                    chatFontSize = state.chatFontSize,
                    preview = state.preview,
                    topInset = effectiveTopBand + 8.dp,
                    bottomInset = effectiveComposerTotal + 8.dp,
                    liveMetrics = state.liveMetrics,
                    voiceCapable = store.voiceCapable(),
                    onSpeak = { store.speak(it) },
                )
            }
        }

        // ── Bottom Transparency Band ───────────────────────────────
        // Pinned to the bottom of the screen with height extending exactly
        // to the top edge of the text field box ("at the end of text field
        // box not a bit more then that"). The gradient runs from opaque app
        // background at the screen bottom (covering nav bar / gesture bar)
        // to fully transparent (alpha = 0f) at the top of the text field box.
        if (state.preview == null) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(effectiveBottomBand)
                    .align(Alignment.BottomCenter)
                    .background(
                        Brush.verticalGradient(
                            0f to c.bgApp.copy(alpha = 0f),
                            0.45f to c.bgApp.copy(alpha = 0.70f),
                            1f to c.bgApp,
                        ),
                    ),
            )
        }

        // ── Top Transparency Band ──────────────────────────────────
        // Pinned to the top of the screen with height extending to the
        // bottom edge of the "New chat" button row. The gradient runs from
        // opaque app background at the top edge (covering status bar) to
        // fully transparent (alpha = 0f) at the end of the top bar.
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

        // ── TopBar: hamburger + session pills + overflow ───────────
        Row(
            Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(horizontal = 6.dp, vertical = 4.dp)
                .onSizeChanged { topBarPx = it.height },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            BarIcon(Icons2.Menu, "Workspaces") { store.setDrawer(true) }
            // Session pills strip (scrollable, active highlighted).
            Row(
                Modifier
                    .weight(1f)
                    .padding(horizontal = 4.dp)
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Pills follow creation order (desktop: task || "Chat N"),
                // newest first, capped with overflow in the history sheet.
                val ordered = state.sessions.sortedBy { it.startedAt }
                val pills = if (ordered.isEmpty()) listOf<String?>(null) else ordered.reversed().take(6).map { it.id }
                for (sid in pills) {
                    val active = sid == state.activeSessionId
                    val s = ordered.firstOrNull { it.id == sid }
                    val label = if (sid == null) "New chat" else (s?.labelOf(ordered.indexOf(s)) ?: "Chat")
                    Row(
                        Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(if (active) c.bgCardHover.copy(alpha = 0.85f) else Color.Transparent)
                            .border(
                                if (active) 0.5.dp else 0.dp,
                                if (active) c.border.copy(alpha = 0.60f) else Color.Transparent,
                                RoundedCornerShape(999.dp),
                            )
                            .clickable {
                                if (sid == null) store.newChat()
                                else store.setActiveSession(sid)
                            }
                            .heightIn(min = 36.dp)
                            .padding(start = 12.dp, end = if (active && sid != null) 6.dp else 12.dp, top = 6.dp, bottom = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(label, fontSize = 13.sp, color = if (active) c.text else c.textMuted, maxLines = 1)
                        if (active && sid != null) {
                            Spacer(Modifier.width(4.dp))
                            Icon(
                                Icons2.Close,
                                "Delete chat",
                                tint = c.textFaint,
                                modifier = Modifier
                                    .size(24.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .clickable { store.deleteSession(sid) }
                                    .padding(5.dp),
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.width(4.dp))
            BarIcon(Icons2.Plus, "New chat") { store.newChat() }
            Spacer(Modifier.width(2.dp))
            BarIcon(Icons2.MoreVert, "More") { store.openHistory() }
        }

        // ── Composer overlay ───────────────────────────────────────
        // Aligned to the bottom of the screen, with navigation bars and IME
        // padding. Pending tool approvals cleanly stack directly above.
        if (state.preview == null) {
            Column(
                Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .imePadding()
                    .onSizeChanged { composerTotalPx = it.height },
            ) {
                if (state.approvals.isNotEmpty()) {
                    ApprovalSheet(
                        state.approvals.first(),
                        queued = state.approvals.size - 1,
                    ) { allow, always ->
                        store.approve(state.approvals.first().id, allow, always)
                    }
                }
                Composer(
                    streaming = state.streaming,
                    mode = state.mode,
                    permissions = state.permissions,
                    config = state.config,
                    localModelLabels = state.localModelLabels,
                    ctx = state.ctx,
                    images = state.pickedImages,
                    pendingMentions = state.pendingMentions,
                    onSubmit = { t, imgs ->
                        store.submit(t, imgs)
                        store.clearPickedImages()
                    },
                    onCancel = { store.cancel() },
                    onModeChange = { store.setMode(it) },
                    onPermissionsChange = { store.setPermissions(it) },
                    onConfigPatch = { patch ->
                        store.patchConfig(org.json.JSONObject().apply { patch.forEach { (k, v) -> put(k, v) } })
                    },
                    onOpenSettings = { tab -> store.openSettings(true, tab) },
                    onAttach = { onPickFiles() },
                    dictating = state.dictating,
                    voiceCapable = store.voiceCapable(),
                    onMic = {
                        // Ask first, then record — the launcher restarts
                        // toggleDictation() once permission lands.
                        if (onRequestMic()) store.toggleDictation()
                    },
                    onRemoveImage = { i -> store.removePickedImage(i) },
                    onMentionsConsumed = { store.consumeMentions() },
                    onSheet = { store.openSheet(it.name.lowercase()) },
                    onTextFieldHeightChanged = { textFieldPx = it },
                )
            }
        }

        // ── Overlays ───────────────────────────────────────────────
        if (state.drawerOpen) {
            WorkspaceDrawer(store) { store.setDrawer(false) }
        }
        if (historyOpen) {
            HistorySheet(
                store = store,
                onClose = { store.openHistory(false) },
            )
        }
        if (state.settingsOpen) {
            SettingsScreen(store) { store.openSettings(false) }
        }
        state.sheet?.let { which ->
            ComposerSheets(store, which) { store.closeSheet() }
        }
    }
}

/** A top-bar icon button: one target, one glyph size — the same rule the
 *  composer's bar uses, so both bars read at one scale. */
@Composable
private fun BarIcon(icon: androidx.compose.ui.graphics.vector.ImageVector, description: String, onClick: () -> Unit) {
    val c = LocalSofuuColors.current
    Box(
        Modifier
            .size(44.dp)
            .clip(RoundedCornerShape(22.dp))
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, description, tint = c.textMuted, modifier = Modifier.size(20.dp))
    }
}

@Composable
private fun Banner(text: String, isError: Boolean) {
    val c = LocalSofuuColors.current
    Text(
        text,
        color = if (isError) c.err else c.text,
        fontSize = 13.sp,
        modifier = Modifier
            .fillMaxWidth()
            .background(if (isError) c.errBg else c.bgCard)
            .padding(12.dp),
    )
}

@Composable
private fun HomePrompt(subtitle: String?, onPromptSelect: (String) -> Unit = {}) {
    val c = LocalSofuuColors.current
    val starters = listOf(
        "Explain this project" to "Walk through architecture and entry points",
        "Review git diff" to "Inspect uncommitted code changes",
        "Find bugs & tests" to "Analyze test coverage and failure modes",
        "Health check" to "Run doctor diagnostics on the runtime",
    )
    Column(
        Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("What should we work on?", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Medium, color = c.text)
        if (subtitle != null) {
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = c.textFaint, modifier = Modifier.padding(top = 6.dp, bottom = 24.dp))
        } else {
            Spacer(Modifier.height(24.dp))
        }
        Column(
            Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            starters.forEach { (title, desc) ->
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(c.bgCard)
                        .border(1.dp, c.border, RoundedCornerShape(12.dp))
                        .clickable { onPromptSelect(title) }
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(title, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = c.text)
                        Text(desc, fontSize = 11.sp, color = c.textFaint)
                    }
                    Icon(Icons2.ChevronRight, null, tint = c.textFaint, modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

@Composable
private fun PreviewBar(label: String, onClose: () -> Unit) {
    val c = LocalSofuuColors.current
    Row(
        Modifier.fillMaxWidth().background(c.bgCard).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons2.Clock, null, tint = c.accent, modifier = Modifier.size(15.dp))
        Spacer(Modifier.width(6.dp))
        Text("Read-only · $label", fontSize = 13.sp, color = c.textMuted, modifier = Modifier.weight(1f))
        Icon(Icons2.Close, "Close", tint = c.textFaint, modifier = Modifier.size(18.dp).clickable { onClose() })
    }
}

// ── Workspace drawer (the desktop sidebar, as a slide-over) ──────

@Composable
private fun WorkspaceDrawer(store: AppStore, onClose: () -> Unit) {
    val state = store.state
    val c = LocalSofuuColors.current
    Box(Modifier.fillMaxSize().background(c.backdropVeil).clickable { onClose() }) {
        Column(
            Modifier
                .fillMaxWidth(0.82f)
                .fillMaxSize()
                .background(c.bgSidebar)
                .statusBarsPadding()
                .navigationBarsPadding()
                .clickable { }
                .padding(16.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Workspaces", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, color = c.text, modifier = Modifier.weight(1f))
                Icon(
                    Icons2.Gear,
                    "Settings",
                    tint = c.textMuted,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(22.dp))
                        .clickable { onClose(); store.openSettings(true) }
                        .padding(11.dp),
                )
            }
            Spacer(Modifier.height(8.dp))
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { onClose(); store.addDefaultWorkspace() }
                    .padding(horizontal = 10.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons2.Plus, null, tint = c.accent, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Add workspace", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = c.accent)
            }
            Spacer(Modifier.height(8.dp))
            Column(Modifier.verticalScroll(rememberScrollState())) {
                if (state.workspaces.isEmpty()) {
                    Text("No open workspaces", color = c.textFaint, fontSize = 13.sp, modifier = Modifier.padding(12.dp))
                }
                for (ws in state.workspaces) {
                    val active = ws == state.projectDir
                    Spacer(Modifier.height(2.dp))
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (active) c.bgSidebarActive else Color.Transparent)
                            .clickable { onClose(); store.openWorkspace(ws) }
                            .heightIn(min = 44.dp)
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons2.Folder, null, tint = if (active) c.accent else c.textFaint, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(ws.substringAfterLast('/'), fontSize = 14.sp, color = c.text, modifier = Modifier.weight(1f))
                        if (state.workspaces.size > 1) {
                            Icon(
                                Icons2.Close,
                                "Remove",
                                tint = c.textFaint,
                                modifier = Modifier
                                    .size(28.dp)
                                    .clip(RoundedCornerShape(14.dp))
                                    .clickable { store.removeWorkspace(ws) }
                                    .padding(6.dp),
                            )
                        }
                    }
                }
            }
        }
    }
}

// ── History sheet (the desktop's clock popover) ──────────────────

@Composable
private fun HistorySheet(store: AppStore, onClose: () -> Unit) {
    val state = store.state
    val c = LocalSofuuColors.current
    var clearArmed by remember { mutableStateOf(false) }
    Box(Modifier.fillMaxSize().background(c.backdropVeil).clickable { onClose() }) {
        Column(
            Modifier
                .align(Alignment.TopEnd)
                .statusBarsPadding()
                .padding(top = 16.dp, end = 12.dp)
                .width(290.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(c.bgRaised)
                .border(1.dp, c.border, RoundedCornerShape(14.dp))
                .padding(10.dp)
                .clickable { },
        ) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Chat history", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = c.text, modifier = Modifier.weight(1f))
                Icon(
                    Icons2.Close,
                    "Close",
                    tint = c.textMuted,
                    modifier = Modifier
                        .size(28.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .clickable { onClose() }
                        .padding(5.dp),
                )
            }
            if (state.sessions.isEmpty()) {
                Text("No chats yet", fontSize = 13.sp, color = c.textFaint, modifier = Modifier.padding(12.dp))
            }
            val ordered = state.sessions.sortedBy { it.startedAt }
            Column(
                Modifier
                    .fillMaxWidth()
                    .heightIn(max = 340.dp)
                    .verticalScroll(rememberScrollState()),
            ) {
                state.sessions.sortedByDescending { it.lastSeen }.forEach { s ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { onClose(); store.openPreview(s) }
                            .heightIn(min = 44.dp)
                            .padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(s.labelOf(ordered.indexOf(s)), fontSize = 13.sp, color = c.text, modifier = Modifier.weight(1f), maxLines = 1)
                        Spacer(Modifier.width(8.dp))
                        Text(ageOf(s.lastSeen), fontSize = 11.sp, color = c.textFaint)
                    }
                }
            }
            if (state.sessions.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Box(Modifier.fillMaxWidth().height(1.dp).background(c.separator))
                Text(
                    if (clearArmed) "Click again to confirm" else "Delete all chats in workspace",
                    fontSize = 12.sp,
                    color = c.errStrong,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { if (clearArmed) { onClose(); store.clearSessions() } else clearArmed = true }
                        .padding(10.dp),
                )
            }
        }
    }
}

private fun ageOf(lastSeen: Long): String {
    val mins = ((System.currentTimeMillis() / 1000 - lastSeen) / 60).coerceAtLeast(0)
    return when {
        mins < 1 -> "now"
        mins < 60 -> "${mins}m ago"
        mins < 1440 -> "${mins / 60}h ago"
        else -> "${mins / 1440}d ago"
    }
}

// ── Approval sheet ──────────────────────────────────────────────

@Composable
private fun ApprovalSheet(approval: Approval, queued: Int = 0, onResolve: (Boolean, Boolean) -> Unit) {
    val c = LocalSofuuColors.current
    Column(
        Modifier
            .fillMaxWidth()
            .padding(10.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(c.bgRaised)
            .border(1.dp, c.border, RoundedCornerShape(16.dp))
            .padding(14.dp),
    ) {
        Text("Tool approval", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = c.accent)
        Text(approval.tool, fontSize = 14.sp, fontWeight = FontWeight.Medium, color = c.text, modifier = Modifier.padding(top = 4.dp))
        // write/edit approvals get a real diff (desktop parity); anything
        // else shows the plain args.
        val args = approval.args
        when {
            approval.tool == "edit_file" && args.has("old_string") && args.has("new_string") -> {
                Text(
                    args.optString("path"),
                    fontSize = 11.sp,
                    color = c.textFaint,
                    modifier = Modifier.padding(top = 2.dp),
                )
                DiffView(args.optString("old_string"), args.optString("new_string"), Modifier.padding(top = 6.dp))
            }
            approval.tool == "write_file" && args.has("content") -> {
                Text(
                    args.optString("path"),
                    fontSize = 11.sp,
                    color = c.textFaint,
                    modifier = Modifier.padding(top = 2.dp),
                )
                DiffView("", args.optString("content"), Modifier.padding(top = 6.dp))
            }
            else -> {
                Text(
                    args.toString().take(600),
                    fontSize = 12.sp,
                    color = c.textMuted,
                    modifier = Modifier
                        .padding(top = 4.dp)
                        .heightIn(max = 140.dp)
                        .verticalScroll(rememberScrollState()),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
            SheetButton("Deny", c.bgCardHover, c.text) { onResolve(false, false) }
            SheetButton("Always", c.bgCardHover, c.text) { onResolve(true, true) }
            SheetButton("Allow once", c.accent, Color.White) { onResolve(true, false) }
        }
        if (queued > 0) {
            Text(
                "$queued more waiting",
                fontSize = 11.sp,
                color = c.textFaint,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}

@Composable
private fun SheetButton(label: String, bg: Color, fg: Color, onClick: () -> Unit) {
    Box(
        Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(bg)
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 10.dp),
    ) {
        Text(label, fontSize = 13.sp, color = fg)
    }
}
