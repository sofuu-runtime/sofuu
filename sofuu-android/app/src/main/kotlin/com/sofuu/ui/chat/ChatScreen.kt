package com.sofuu.ui.chat

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.text.selection.SelectionContainer
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.store.ChatMessage
import com.sofuu.store.MsgRole
import com.sofuu.store.SessionPreview
import com.sofuu.store.Timeline
import com.sofuu.store.TimelineEntry
import com.sofuu.ui.components.Icons2
import com.sofuu.ui.components.iconFor
import com.sofuu.ui.theme.LocalSofuuColors

/** The message list: user bubbles, assistant markdown + activity
 *  timeline + thinking + usage, and the live phase line. */
@Composable
fun ChatScreen(
    messages: List<ChatMessage>,
    phase: String?,
    streaming: Boolean,
    chatFontSize: Int,
    preview: SessionPreview?,
    /** Extra bottom padding for content that scrolls under the translucent
     *  composer overlay — when pinned to the bottom, the newest line rests
     *  just above it; scrolled-up text shows through to the screen edge. */
    bottomInset: androidx.compose.ui.unit.Dp = 0.dp,
    liveMetrics: String? = null,
    voiceCapable: Boolean = false,
    onSpeak: (String) -> Unit = {},
) {
    val c = LocalSofuuColors.current
    val listState = rememberLazyListState()

    // Stick to the bottom while new content streams — but only when the
    // user is already there, and without animating. During streaming the
    // content length changes ~10×/s; an animateScrollToItem per change
    // cancelled the previous scroll mid-flight, which read as constant
    // stutter. An instant jump when pinned, nothing when the user scrolled
    // up to read, is the standard chat-scroll contract.
    LaunchedEffect(messages.size, messages.lastOrNull()?.content?.length) {
        if (messages.isNotEmpty() && listState.canScrollForward == false) {
            listState.scrollToItem((messages.size - 1).coerceAtLeast(0))
        }
    }

    if (preview != null) {
        PreviewPane(preview, chatFontSize)
        return
    }

    Box(Modifier.fillMaxSize()) {
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(top = 12.dp, bottom = bottomInset),
        ) {
            // Keys: without them, item identity is positional — every
            // content flush re-attached per-row remember state (expanded
            // tool rows, selection context) to whatever bubble now sat at
            // that index, and all visible rows recomposed wholesale.
            items(messages, key = { it.id }, contentType = { it.role }) { m ->
                MessageRow(m, chatFontSize, phase, voiceCapable, onSpeak, liveMetrics)
            }
        }
    }
}

@Composable
private fun MessageRow(
    m: ChatMessage,
    fontSize: Int,
    phase: String?,
    voiceCapable: Boolean,
    onSpeak: (String) -> Unit,
    liveMetrics: String? = null,
) {
    val c = LocalSofuuColors.current
    when (m.role) {
        MsgRole.USER -> Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
            Text(
                m.content,
                fontSize = fontSize.sp,
                lineHeight = (fontSize + 7).sp,
                color = c.text,
                modifier = Modifier
                    .padding(vertical = 6.dp, horizontal = 4.dp)
                    .background(c.bgCard, RoundedCornerShape(14.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
            )
        }
        MsgRole.SYSTEM -> Text(
            m.content,
            fontSize = (fontSize - 3).sp,
            lineHeight = (fontSize + 4).sp,
            color = c.textMuted,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
        )
        MsgRole.ASSISTANT -> Column(Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 10.dp)) {
            // Live phase label (Thinking → Coding → Web searching …).
            // It belongs to the whole turn, not just the empty pre-first-
            // delta window: this used to pass a literal `null` AND hide
            // itself the moment content arrived, so the app sat on
            // "Thinking…" for every turn regardless of what it was doing.
            if (m.status == "streaming" && phase != null) {
                PhaseLine(phase, streaming = true)
            }
            if (m.status == "streaming" && liveMetrics != null) {
                Text(
                    liveMetrics,
                    color = c.textFaint,
                    fontSize = (fontSize - 3).sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(bottom = 2.dp),
                )
            }
            ToolTimelineView(m.timeline)
            if (m.content.isNotEmpty()) {
                Row(verticalAlignment = Alignment.Bottom) {
                    // SelectionContainer: long-press to select + the
                    // system Copy action. Without it, no chat text on
                    // Android can be copied at all — and links only became
                    // tappable (P5) in a container that handles touches.
                    Box(Modifier.weight(1f)) {
                        SelectionContainer { MarkdownBody(m.content, fontSize) }
                    }
                    if (m.status == "streaming") {
                        Spacer(Modifier.width(2.dp))
                        StreamingCursor()
                    }
                }
            }
            if (m.status == "error" && m.error != null) {
                Text(
                    m.error,
                    color = c.err,
                    fontSize = (fontSize - 2).sp,
                    modifier = Modifier
                        .padding(top = 6.dp)
                        .background(c.errBg, RoundedCornerShape(8.dp))
                        .padding(10.dp),
                )
            }
            if (m.status == "done" && m.usage != null) {
                m.usage.line()?.let {
                    Text(it, color = c.textFaint, fontSize = (fontSize - 4).sp, modifier = Modifier.padding(top = 4.dp))
                }
            }
            if (m.status == "done" && m.metrics != null) {
                Text(
                    "⌁ ${m.metrics}",
                    color = c.textFaint,
                    fontSize = (fontSize - 4).sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
            // Read aloud. Provider-backed, and only offered when the active
            // provider can actually synthesize — same gate as the mic.
            if (m.status == "done" && voiceCapable && m.content.isNotBlank()) {
                Row(
                    Modifier.padding(top = 6.dp).clickable { onSpeak(m.content) },
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons2.Volume, "Read aloud", tint = c.textFaint, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Read aloud", fontSize = 11.sp, color = c.textFaint)
                }
            }
        }
    }
}

/** The live phase line: an animated "Thinking…" / "Coding…" sweep while
 *  the turn streams. */
@Composable
fun PhaseLine(phase: String?, streaming: Boolean) {
    if (!streaming) return
    val c = LocalSofuuColors.current
    val transition = rememberInfiniteTransition(label = "phase")
    val alpha by transition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(700, easing = LinearEasing), RepeatMode.Reverse),
        label = "phaseAlpha",
    )
    Text(
        "${phase ?: "Thinking"}…",
        color = c.textMuted,
        fontSize = 13.sp,
        modifier = Modifier.padding(vertical = 4.dp).alpha(alpha),
    )
}

@Composable
private fun StreamingCursor() {
    val c = LocalSofuuColors.current
    val transition = rememberInfiniteTransition(label = "cursor")
    val alpha by transition.animateFloat(0f, 1f, infiniteRepeatable(tween(600)), label = "cursorAlpha")
    Box(
        Modifier
            .size(width = 8.dp, height = 15.dp)
            .background(c.accent, RoundedCornerShape(2.dp))
            .alpha(alpha),
    )
}

/** The tool timeline under an assistant message. */
@Composable
fun ToolTimelineView(entries: List<TimelineEntry>) {
    if (entries.isEmpty()) return
    val c = LocalSofuuColors.current
    var expandedId by remember { mutableStateOf<String?>(null) }
    Column(Modifier.padding(top = 6.dp, bottom = 4.dp)) {
        // Consecutive same-tool rows collapse into one summary row.
        var i = 0
        while (i < entries.size) {
            val e = entries[i]
            val runKind = e.kind
            var j = i + 1
            if (runKind == "tool") {
                while (j < entries.size && entries[j].kind == "tool" && (entries[j].tool ?: entries[j].title) == (e.tool ?: e.title)) j++
            }
            if (j - i >= 3) {
                TimelineRow(
                    icon = Timeline.iconIdOf(e),
                    title = Timeline.composeSummary(entries.subList(i, j)),
                    badge = null,
                    ok = null,
                    expanded = false,
                    detail = null,
                    onToggle = null,
                )
                i = j
                continue
            }
            val expanded = expandedId == e.id
            TimelineRow(
                icon = Timeline.iconIdOf(e),
                title = e.title,
                badge = e.badge,
                ok = e.ok,
                expanded = expanded,
                detail = when {
                    expanded -> (e.result ?: e.detail)
                    else -> null
                },
                onToggle = {
                    expandedId = if (expanded) null else e.id
                },
            )
            i++
        }
    }
}

@Composable
private fun TimelineRow(
    icon: String,
    title: String,
    badge: String?,
    ok: Boolean?,
    expanded: Boolean,
    detail: String?,
    onToggle: (() -> Unit)?,
) {
    val c = LocalSofuuColors.current
    Column(Modifier.fillMaxWidth().clickable(enabled = onToggle != null) { onToggle?.invoke() }.padding(vertical = 3.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                iconFor(icon),
                contentDescription = null,
                tint = when (icon) {
                    Timeline.ICON_WARN -> c.warn
                    else -> c.textFaint
                },
                modifier = Modifier.size(14.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                title,
                fontSize = 13.sp,
                color = c.textMuted,
                fontWeight = FontWeight.Medium,
            )
            if (badge != null) {
                Spacer(Modifier.width(6.dp))
                Text(badge, fontSize = 12.sp, color = c.textFaint)
            }
            if (ok != null) {
                Spacer(Modifier.width(6.dp))
                Text(
                    if (ok) "✓" else "✗",
                    fontSize = 12.sp,
                    color = if (ok) c.ok else c.err,
                )
            }
            if (onToggle != null && (detail != null || expanded)) {
                Spacer(Modifier.width(4.dp))
                Icon(Icons2.ChevronDown, null, tint = c.textFaint, modifier = Modifier.size(14.dp).alpha(if (expanded) 1f else 0.5f))
            }
        }
        if (expanded && detail != null) {
            Text(
                detail.take(1500),
                fontSize = 12.sp,
                lineHeight = 17.sp,
                color = c.textFaint,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier
                    .padding(start = 22.dp, top = 2.dp, bottom = 4.dp)
                    .fillMaxWidth()
                    .background(c.codeBlock, RoundedCornerShape(6.dp))
                    .padding(8.dp),
            )
        }
    }
}

/** Read-only transcript of a past session. */
@Composable
private fun PreviewPane(preview: SessionPreview, fontSize: Int) {
    val c = LocalSofuuColors.current
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(top = 12.dp, bottom = 16.dp),
    ) {
        item {
            Text(
                "Read-only transcript · ${preview.label} · ${preview.turns.size} turns",
                color = c.textFaint,
                fontSize = 12.sp,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        items(preview.turns) { t ->
            Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
                Text(
                    t.prompt,
                    fontSize = fontSize.sp,
                    color = c.text,
                    modifier = Modifier
                        .padding(vertical = 6.dp)
                        .background(c.bgCard, RoundedCornerShape(14.dp))
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                )
            }
            if (t.answer.isNotEmpty()) {
                Column(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
                    MarkdownBody(t.answer, fontSize)
                }
            }
        }
    }
}
