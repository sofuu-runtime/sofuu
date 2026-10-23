package com.sofuu.ui.composer

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.store.CtxMeter
import com.sofuu.store.SLASH_COMMANDS
import com.sofuu.store.SofuuConfig
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.theme.LocalSofuuColors

/**
 * The composer card: text input on top, bottom bar with model/effort
 * pills + icons left and the send pill right; the mode + permissions
 * pills sit below. Same controls as the desktop composer, laid out for
 * touch (44px targets, sheets for the pickers).
 */
@Composable
fun Composer(
    streaming: Boolean,
    mode: String,
    permissions: String,
    config: SofuuConfig?,
    localModelLabels: Map<String, String> = emptyMap(),
    ctx: CtxMeter?,
    images: List<String>,
    pendingMentions: String,
    onSubmit: (String, List<String>) -> Unit,
    onCancel: () -> Unit,
    onModeChange: (String) -> Unit,
    onPermissionsChange: (String) -> Unit,
    onConfigPatch: (Map<String, Any>) -> Unit,
    onOpenSettings: (String) -> Unit,
    onAttach: () -> Unit,
    dictating: Boolean = false,
    voiceCapable: Boolean = false,
    onMic: () -> Unit = {},
    onRemoveImage: (Int) -> Unit,
    onMentionsConsumed: () -> Unit,
    onSheet: (ComposerSheet) -> Unit,
    onTextFieldHeightChanged: (Int) -> Unit = {},
) {
    val c = LocalSofuuColors.current
    var text by remember { mutableStateOf("") }
    val slashOpen = text.startsWith("/")
    var slashQuery by remember(text) { mutableStateOf(if (slashOpen) text.substring(1).trim().lowercase() else "") }
    var slashIdx by remember { mutableStateOf(0) }
    val slashMatches = remember(slashQuery) {
        if (!slashOpen) emptyList()
        else SLASH_COMMANDS.filter { slashQuery.isEmpty() || it.name.contains(slashQuery) || it.desc.lowercase().contains(slashQuery) }
    }

    val model = config?.model?.takeIf { it.isNotEmpty() }?.let { modelLabel(it, localModelLabels) }
        ?: (if (config?.providers?.isNotEmpty() == true) "Choose model" else "No model")
    val effortValue = config?.effort.orEmpty()
    // Mirror the Effort sheet's truth: an empty effort IS off (pickLocalModel
    // clears it for non-thinking models), so the pill must not read "High"
    // while the sheet highlights "Off" and the engine sends nothing.
    val effortLabel = when {
        effortValue.isEmpty() || effortValue == "off" -> "Off"
        else -> effortValue.replaceFirstChar { it.uppercase() }
    }
    val rlmState = config?.rlm ?: "off"

    // @mentions spliced by the file picker land in the input.
    LaunchedEffect(pendingMentions) {
        if (pendingMentions.isNotEmpty()) {
            text += pendingMentions
            onMentionsConsumed()
        }
    }

    val submit = {
        val t = text.trim()
        if (!streaming && (t.isNotEmpty() || images.isNotEmpty())) {
            onSubmit(text, images)
            text = ""
        }
    }

    Column(Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
        if (images.isNotEmpty()) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(bottom = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                images.forEachIndexed { i, url ->
                    Box {
                        androidx.compose.foundation.Image(
                            rememberBitmapPainter(url),
                            contentDescription = null,
                            modifier = Modifier.size(44.dp).clip(RoundedCornerShape(8.dp)),
                        )
                        Box(
                            Modifier
                                .size(18.dp)
                                .align(Alignment.TopEnd)
                                .background(c.errStrong, RoundedCornerShape(9.dp))
                                .clickable { onRemoveImage(i) },
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("×", color = androidx.compose.ui.graphics.Color.White, fontSize = 11.sp)
                        }
                    }
                }
            }
        }

        // Mode + permissions pills — their own row, just ABOVE the input
        // card (never inside the text field).
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Pill(if (mode == "chat") "Chat only" else "Agent", Icons2.Chat) { onSheet(ComposerSheet.Mode) }
            Pill(permissionsLabel(permissions), Icons2.Shield) { onSheet(ComposerSheet.Permissions) }
        }
        Spacer(Modifier.height(6.dp))

        Box(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(c.bgCard)
                .border(1.dp, c.border, RoundedCornerShape(16.dp))
                .padding(12.dp)
                .onSizeChanged { onTextFieldHeightChanged(it.height) },
        ) {
            Column {
                if (dictating) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(c.accent.copy(alpha = 0.12f))
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons2.Mic, null, tint = c.accent, modifier = Modifier.size(15.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(
                            "Listening… speak now or tap Stop",
                            color = c.accent,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                        )
                    }
                }
                BasicTextField(
                    value = text,
                    onValueChange = {
                        text = it
                        slashIdx = 0
                    },
                    textStyle = TextStyle(color = c.text, fontSize = 15.sp, fontWeight = FontWeight.Medium, fontFamily = FontFamily.Monospace),
                    cursorBrush = SolidColor(c.accent),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 44.dp, max = 160.dp),
                    decorationBox = { inner ->
                        if (text.isEmpty()) {
                            Text(
                                "Ask, @mention files, or type / for commands…",
                                color = c.textFaint,
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Medium,
                                fontFamily = FontFamily.Monospace,
                            )
                        }
                        inner()
                    },
                )

                if (slashOpen && slashMatches.isNotEmpty()) {
                    SlashPalette(slashMatches, slashIdx) { cmd ->
                        onSubmit("/" + cmd.name, emptyList())
                        text = ""
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 8.dp),
                ) {
                    // weight(1f, fill=false): a local-model label derives from a
                    // long file path and must never push the send button off
                    // the right edge — cap the pill, ellipsize its text.
                    Pill(
                        model, Icons2.Spark,
                        modifier = Modifier.weight(1f, fill = false),
                        flexibleLabel = true,
                    ) { onSheet(ComposerSheet.Model) }
                    Spacer(Modifier.width(6.dp))
                    Pill(effortLabel, null) { onSheet(ComposerSheet.Effort) }
                    Spacer(Modifier.weight(1f))
                    IconAction(Icons2.Attach, "Attach") { onAttach() }
                    Spacer(Modifier.width(2.dp))
                    IconAction(Icons2.Layers, "Context layers") { onSheet(ComposerSheet.Layers) }
                    Spacer(Modifier.width(2.dp))
                    // Dictation. Shown only when the active provider could
                    // actually serve audio — Anthropic and local models have
                    // no audio API, so offering the mic there would be a
                    // button that fails 100% of the time. While recording it
                    // turns into a stop affordance and stops sending.
                    if (voiceCapable && !streaming) {
                        IconAction(
                            if (dictating) Icons2.Stop else Icons2.Mic,
                            if (dictating) "Stop dictation" else "Dictate",
                            active = dictating,
                        ) { onMic() }
                        Spacer(Modifier.width(2.dp))
                    }
                    Spacer(Modifier.width(8.dp))
                    CtxRing(ctx)
                    Spacer(Modifier.width(10.dp))
                    if (streaming) {
                        SendButton(Icons2.Stop, c.errStrong) { onCancel() }
                    } else {
                        SendButton(Icons2.Send, c.accent) { submit() }
                    }
                }
            }
        }
    }
}

/** A composer bar icon: ONE touch target and ONE glyph size, so every
 *  action in the bar is pixel-identical (a per-icon padding drift is
 *  exactly what made the paperclip and layers icons render at different
 *  sizes). */
@Composable
private fun IconAction(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    description: String,
    active: Boolean = false,
    onClick: () -> Unit,
) {
    val c = LocalSofuuColors.current
    Box(
        Modifier
            .size(40.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(if (active) c.accent.copy(alpha = 0.16f) else androidx.compose.ui.graphics.Color.Transparent)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, description, tint = if (active) c.accent else c.textMuted, modifier = Modifier.size(20.dp))
    }
}

enum class ComposerSheet { Model, Effort, Layers, Mode, Permissions }

private fun permissionsLabel(p: String): String = when (p) {
    "full" -> "Full access"
    "edit" -> "Edit only"
    "plan" -> "Plan mode"
    else -> "Ask me"
}

/** Composer pill label for the active model. Local models keep an
 *  absolute path in config.model (the engine opens the file by path), so
 *  show the header-derived label from the store's cache when present
 *  ("Llama 3.2 1B Instruct · Q8_0"); fall back to a name-only split of the
 *  file name. Cloud model ids pass through untouched. */
private fun modelLabel(model: String?, labels: Map<String, String>): String? {
    if (model.isNullOrEmpty()) return null
    labels[model]?.let { return it }
    return if (model.endsWith(".gguf")) com.sofuu.store.fallbackModelLabel(model) else model
}

@Composable
private fun Pill(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector?,
    modifier: Modifier = Modifier,
    flexibleLabel: Boolean = false,
    onClick: () -> Unit,
) {
    val c = LocalSofuuColors.current
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
            .clip(RoundedCornerShape(999.dp))
            .background(c.bgCardHover.copy(alpha = 0.85f))
            .border(0.5.dp, c.border.copy(alpha = 0.70f), RoundedCornerShape(999.dp))
            .clickable { onClick() }
            .heightIn(min = 40.dp)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        if (icon != null) {
            Icon(icon, null, tint = c.accent, modifier = Modifier.size(14.dp))
            Spacer(Modifier.width(5.dp))
        }
        Text(
            label,
            fontSize = 13.sp,
            fontWeight = FontWeight.SemiBold,
            color = c.text,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = if (flexibleLabel) Modifier.weight(1f, fill = false) else Modifier,
        )
        Spacer(Modifier.width(3.dp))
        Icon(Icons2.Caret, null, tint = c.textFaint, modifier = Modifier.size(13.dp))
    }
}

@Composable
private fun SendButton(icon: androidx.compose.ui.graphics.vector.ImageVector, tint: androidx.compose.ui.graphics.Color, onClick: () -> Unit) {
    Box(
        Modifier
            .size(42.dp)
            .clip(RoundedCornerShape(21.dp))
            .background(tint)
            .clickable { onClick() },
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, null, tint = androidx.compose.ui.graphics.Color.White, modifier = Modifier.size(20.dp))
    }
}

/** Live context-window fill ring (the desktop's ContextRing). */
@Composable
fun CtxRing(ctx: CtxMeter?) {
    val c = LocalSofuuColors.current
    if (ctx == null || ctx.window <= 0) return
    Box(Modifier.size(24.dp), contentAlignment = Alignment.Center) {
        androidx.compose.foundation.Canvas(Modifier.size(22.dp)) {
            val stroke = 2.2.dp.toPx()
            drawArc(
                color = c.border,
                startAngle = -90f,
                sweepAngle = 360f,
                useCenter = false,
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = stroke),
            )
            drawArc(
                color = when {
                    ctx.fraction > 0.85f -> c.err
                    ctx.fraction > 0.7f -> c.warn
                    else -> c.accent
                },
                startAngle = -90f,
                sweepAngle = 360f * ctx.fraction,
                useCenter = false,
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = stroke),
            )
        }
        Text("${ctx.pct}", fontSize = 8.sp, color = c.textFaint)
    }
}

/** The "/" autocomplete card. */
@Composable
private fun SlashPalette(commands: List<com.sofuu.store.SlashCommand>, activeIdx: Int, onPick: (com.sofuu.store.SlashCommand) -> Unit) {
    val c = LocalSofuuColors.current
    Column(
        Modifier
            .fillMaxWidth()
            .heightIn(max = 220.dp)
            .padding(top = 8.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(c.bgRaised)
            .border(1.dp, c.border, RoundedCornerShape(10.dp))
            .padding(vertical = 4.dp),
    ) {
        LazyColumn(Modifier.heightIn(max = 190.dp)) {
            items(commands) { cmd ->
                val active = commands.indexOf(cmd) == activeIdx
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(6.dp))
                        .clickable { onPick(cmd) }
                        .background(if (active) c.bgCardHover else androidx.compose.ui.graphics.Color.Transparent)
                        .heightIn(min = 40.dp)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("/${cmd.name}", color = c.accent, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace)
                    if (cmd.arg != null) {
                        Spacer(Modifier.width(4.dp))
                        Text(cmd.arg, color = c.textFaint, fontSize = 13.sp, fontFamily = FontFamily.Monospace)
                    }
                    Spacer(Modifier.width(8.dp))
                    Text(cmd.desc, color = c.textMuted, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

// ── image bitmap helper ────────────────────────────────────────

@Composable
private fun rememberBitmapPainter(dataUrl: String): androidx.compose.ui.graphics.painter.Painter {
    val bmp = remember(dataUrl) { decodeDataUrl(dataUrl) }
    return androidx.compose.ui.graphics.painter.BitmapPainter(
        (bmp ?: android.graphics.Bitmap.createBitmap(1, 1, android.graphics.Bitmap.Config.ARGB_8888)).asImageBitmap(),
    )
}

private fun decodeDataUrl(dataUrl: String): android.graphics.Bitmap? {
    val idx = dataUrl.indexOf(',')
    if (idx < 0) return null
    val bytes = try {
        android.util.Base64.decode(dataUrl.substring(idx + 1), android.util.Base64.DEFAULT)
    } catch (e: Exception) {
        return null
    }
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
