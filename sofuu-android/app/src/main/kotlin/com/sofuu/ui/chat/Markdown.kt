package com.sofuu.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.ui.theme.LocalSofuuColors
import com.sofuu.ui.theme.SofuuColors
import kotlin.math.max

/**
 * A compact markdown renderer — chat-answer parity with the desktop's
 * react-markdown + remark-gfm: headings, bold/italic/strikethrough, inline
 * code, fenced code blocks, bulleted + numbered lists, blockquotes, links
 * (labelled and bare-autolinked), GFM tables and horizontal rules.
 *
 * Link safety is the same XSS discipline as the desktop's safeUrl
 * (ChatPane.tsx): only https/http/mailto become live links. A model-rendered
 * `javascript:` or `data:` URL renders as inert text showing the target, so
 * it can never be tapped into.
 */
@Composable
fun MarkdownBody(text: String, fontSize: Int = 15) {
    val c = LocalSofuuColors.current
    val blocks = remember(text) { parseBlocks(text) }
    Column {
        for (b in blocks) {
            when (b) {
                is MdBlock.Code -> CodeBlock(b.code, b.lang, fontSize, c.codeBlock, c.text)
                is MdBlock.Heading -> Text(
                    inlineMd(b.text),
                    fontSize = (fontSize + 4).sp,
                    lineHeight = (fontSize + 10).sp,
                    fontWeight = FontWeight.Bold,
                    color = c.text,
                    modifier = Modifier.padding(top = 10.dp, bottom = 4.dp),
                )
                is MdBlock.Paragraph -> Text(
                    inlineMd(b.text),
                    fontSize = fontSize.sp,
                    lineHeight = (fontSize + 7).sp,
                    color = c.text,
                    modifier = Modifier.padding(vertical = 2.dp),
                )
                is MdBlock.Bullet -> Row(modifier = Modifier.padding(vertical = 1.dp)) {
                    Text("·  ", color = c.textMuted, fontSize = fontSize.sp, lineHeight = (fontSize + 7).sp)
                    Text(inlineMd(b.text), fontSize = fontSize.sp, lineHeight = (fontSize + 7).sp, color = c.text)
                }
                is MdBlock.Numbered -> Row(modifier = Modifier.padding(vertical = 1.dp)) {
                    Text("${b.index}.  ", color = c.textMuted, fontSize = fontSize.sp, lineHeight = (fontSize + 7).sp)
                    Text(inlineMd(b.text), fontSize = fontSize.sp, lineHeight = (fontSize + 7).sp, color = c.text)
                }
                is MdBlock.Quote -> Row(modifier = Modifier.padding(vertical = 2.dp)) {
                    Spacer(Modifier.width(10.dp).height(0.dp))
                    Text(
                        inlineMd(b.text),
                        fontSize = fontSize.sp,
                        lineHeight = (fontSize + 7).sp,
                        color = c.textMuted,
                        modifier = Modifier
                            .padding(start = 8.dp)
                            .background(c.hoverVeil, RoundedCornerShape(4.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                    )
                }
                is MdBlock.Table -> TableBlock(b, fontSize)
                is MdBlock.Rule -> Spacer(
                    Modifier
                        .padding(vertical = 8.dp)
                        .fillMaxWidth()
                        .height(1.dp)
                        .background(c.separator),
                )
            }
        }
    }
}

@Composable
private fun CodeBlock(code: String, lang: String, fontSize: Int, bg: Color, fg: Color) {
    val c = LocalSofuuColors.current
    Column(
        Modifier
            .padding(vertical = 6.dp)
            .fillMaxWidth()
            .background(bg, RoundedCornerShape(8.dp))
            .padding(10.dp),
    ) {
        if (lang.isNotEmpty()) {
            Text(lang, color = c.textFaint, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
        }
        Text(
            code.trimEnd(),
            color = fg,
            fontSize = (fontSize - 1).sp,
            lineHeight = (fontSize + 5).sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.horizontalScroll(rememberScrollState()),
        )
    }
}

/** GFM table. Column widths come from the widest cell so a table scrolls
 *  horizontally instead of crushing every column to fit the screen. */
@Composable
private fun TableBlock(t: MdBlock.Table, fontSize: Int) {
    val c = LocalSofuuColors.current
    val cols = max(t.headers.size, t.rows.maxOfOrNull { it.size } ?: 0)
    val widths = List(cols) { idx ->
        val longest = (t.headers.getOrNull(idx)?.length ?: 0).coerceAtLeast(
            t.rows.maxOfOrNull { it.getOrNull(idx)?.length ?: 0 } ?: 0
        )
        (longest * 7 + 24).coerceIn(72, 260).dp
    }
    val align = t.alignRight
    Column(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .horizontalScroll(rememberScrollState()),
    ) {
        Row {
            t.headers.forEachIndexed { i, h ->
                Text(
                    inlineMd(h),
                    fontSize = (fontSize - 1).sp,
                    fontWeight = FontWeight.Bold,
                    color = c.text,
                    textAlign = if (align.getOrElse(i) { false }) TextAlign.End else TextAlign.Start,
                    modifier = Modifier.width(widths[i]).padding(horizontal = 8.dp, vertical = 6.dp),
                )
            }
        }
        Box(Modifier.height(1.dp).background(c.separator))
        t.rows.forEach { row ->
            Row {
                repeat(cols) { i ->
                    Text(
                        inlineMd(row.getOrElse(i) { "" }),
                        fontSize = (fontSize - 1).sp,
                        lineHeight = (fontSize + 5).sp,
                        color = c.text,
                        textAlign = if (align.getOrElse(i) { false }) TextAlign.End else TextAlign.Start,
                        modifier = Modifier.width(widths[i]).padding(horizontal = 8.dp, vertical = 6.dp),
                    )
                }
            }
        }
    }
}

// ── Inline markdown ─────────────────────────────────────────────

@Composable
private fun inlineMd(s: String): AnnotatedString {
    val c = LocalSofuuColors.current
    return buildInline(s, c)
}

/**
 * Single-pass inline scanner over [s] in `[from, to)`.
 *
 * This used to hand `plain()` ONE CHARACTER AT A TIME, so every
 * multi-character construct inside it could never match: `**bold**`,
 * `*italic*` and `[label](url)` all rendered as literal source. Only
 * backticks worked. Scanning the real span is what makes emphasis, links
 * and strikethrough actually parse.
 */
@OptIn(ExperimentalTextApi::class)
private fun buildInline(s: String, c: SofuuColors, from: Int = 0, to: Int = s.length): AnnotatedString =
    buildAnnotatedString { scanInline(s, c, from, to) }

@OptIn(ExperimentalTextApi::class)
private fun AnnotatedString.Builder.scanInline(s: String, c: SofuuColors, from: Int, to: Int) {
    var i = from
    val mono = SpanStyle(fontFamily = FontFamily.Monospace, background = c.codeChip)

    while (i < to) {
        when {
            // **bold**
            s.startsWith("**", i) -> {
                val end = s.indexOf("**", i + 2)
                if (end in (i + 2) until to) {
                    pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                    scanInline(s, c, i + 2, end)
                    pop()
                    i = end + 2
                } else {
                    append(s[i]); i++
                }
            }
            // ~~strikethrough~~ (GFM)
            s.startsWith("~~", i) -> {
                val end = s.indexOf("~~", i + 2)
                if (end in (i + 2) until to) {
                    pushStyle(SpanStyle(textDecoration = TextDecoration.LineThrough))
                    scanInline(s, c, i + 2, end)
                    pop()
                    i = end + 2
                } else {
                    append(s[i]); i++
                }
            }
            // `inline code`
            s[i] == '`' -> {
                val end = s.indexOf('`', i + 1)
                if (end in (i + 1) until to) {
                    pushStyle(mono)
                    append(s, i + 1, end)
                    pop()
                    i = end + 1
                } else {
                    append(s[i]); i++
                }
            }
            // *italic* / _italic_
            s[i] == '*' || s[i] == '_' -> {
                val marker = s[i]
                val end = s.indexOf(marker, i + 1)
                if (end > i + 1 && end < to && !s.substring(i + 1, end).contains('\n')) {
                    pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                    append(s, i + 1, end)
                    pop()
                    i = end + 1
                } else {
                    append(s[i]); i++
                }
            }
            // [label](url)
            s[i] == '[' -> {
                val close = s.indexOf(']', i)
                if (close in (i + 1) until to && close + 1 < to && s[close + 1] == '(') {
                    val paren = s.indexOf(')', close + 2)
                    if (paren in (close + 2) until to) {
                        emitLink(
                            s, c,
                            label = s.substring(i + 1, close),
                            url = s.substring(close + 2, paren),
                        )
                        i = paren + 1
                    } else {
                        append(s[i]); i++
                    }
                } else {
                    append(s[i]); i++
                }
            }
            // bare URL autolink (GFM)
            else -> {
                val end = bareUrlEnd(s, i, to)
                if (end > i) {
                    emitLink(s, c, label = s.substring(i, end), url = s.substring(i, end))
                    i = end
                } else {
                    append(s[i]); i++
                }
            }
        }
    }
}

/** End offset of a bare http(s):// URL starting at [i], or [i] if none. */
private fun bareUrlEnd(s: String, i: Int, to: Int): Int {
    val starts = s.startsWith("https://", i, ignoreCase = true) || s.startsWith("http://", i, ignoreCase = true)
    if (!starts) return i
    var j = i
    while (j < to) {
        val ch = s[j]
        if (ch.isWhitespace() || ch == '<' || ch == ')' || ch == ']') break
        j++
    }
    // Trailing sentence punctuation is not part of the link.
    while (j > i && s[j - 1] in ".,;:!?") j--
    return max(i, j)
}

/** Same allowlist as the desktop (ChatPane.tsx safeUrl). Anything else is
 *  rendered as inert text that still SHOWS the target, so the reader can
 *  see where it pointed without the tap ever becoming live. */
private fun isSafeUrl(url: String): Boolean {
    val u = url.trim()
    return u.startsWith("https://", ignoreCase = true) ||
        u.startsWith("http://", ignoreCase = true) ||
        u.startsWith("mailto:", ignoreCase = true)
}

@OptIn(ExperimentalTextApi::class)
private fun AnnotatedString.Builder.emitLink(
    s: String,
    c: SofuuColors,
    label: String,
    url: String,
) {
    if (isSafeUrl(url)) {
        pushLink(LinkAnnotation.Url(url.trim()))
        pushStyle(SpanStyle(color = c.accent, textDecoration = TextDecoration.Underline))
        // The label may itself carry emphasis: [**bold**](url)
        if (label.contains('*') || label.contains('`')) {
            scanInline(label, c, 0, label.length)
        } else {
            append(label)
        }
        pop()
        pop()
    } else {
        pushStyle(SpanStyle(color = c.textMuted))
        append(label)
        append(" ($url)")
        pop()
    }
}

// ── Block parsing ───────────────────────────────────────────────

private sealed class MdBlock {
    data class Paragraph(val text: String) : MdBlock()
    data class Heading(val text: String) : MdBlock()
    data class Code(val code: String, val lang: String) : MdBlock()
    data class Bullet(val text: String) : MdBlock()
    data class Numbered(val index: Int, val text: String) : MdBlock()
    data class Quote(val text: String) : MdBlock()
    data class Table(
        val headers: List<String>,
        val rows: List<List<String>>,
        val alignRight: List<Boolean>,
    ) : MdBlock()
    data object Rule : MdBlock()
}

/** | --- | :---: | ---: | */
private val TABLE_DELIM = Regex("^\\|?\\s*:?-{2,}:?\\s*(\\|\\s*:?-{2,}:?\\s*)*\\|?$")

private fun splitRow(line: String): List<String> =
    line.trim().trim('|').split('|').map { it.trim() }

private fun parseBlocks(src: String): List<MdBlock> {
    val blocks = ArrayList<MdBlock>()
    val lines = src.replace("\r\n", "\n").split("\n")
    var i = 0
    var para = StringBuilder()
    var numbered = 0

    fun flushPara() {
        if (para.isNotEmpty()) {
            blocks.add(MdBlock.Paragraph(para.toString()))
            para = StringBuilder()
        }
        numbered = 0
    }

    while (i < lines.size) {
        val line = lines[i]
        val trimmed = line.trim()

        // GFM table: a pipe row whose NEXT line is the ---/:---: delimiter.
        if (trimmed.contains('|') && i + 1 < lines.size && TABLE_DELIM.matches(lines[i + 1].trim())) {
            flushPara()
            val headers = splitRow(trimmed)
            val alignRight = splitRow(lines[i + 1]).map { it.endsWith(":") && !it.startsWith(":") }
            val rows = ArrayList<List<String>>()
            var k = i + 2
            while (k < lines.size && lines[k].contains('|') && lines[k].isNotBlank()) {
                rows.add(splitRow(lines[k]))
                k++
            }
            blocks.add(MdBlock.Table(headers, rows, alignRight))
            i = if (k > i + 2) k - 1 else i + 2
            continue
        }

        when {
            trimmed.startsWith("```") -> {
                flushPara()
                val lang = trimmed.removePrefix("```").trim()
                val code = StringBuilder()
                var k = i + 1
                while (k < lines.size && !lines[k].trim().startsWith("```")) {
                    code.append(lines[k]).append("\n")
                    k++
                }
                blocks.add(MdBlock.Code(code.toString(), lang))
                // Land on the closing fence (or the last line). The old
                // loop advanced past it AND the loop's own i++, which ate
                // the first line of any paragraph after a code block.
                i = if (k < lines.size) k else lines.size - 1
            }
            trimmed.matches(Regex("^(-{3,}|\\*{3,}|_{3,})$")) -> {
                flushPara()
                blocks.add(MdBlock.Rule)
            }
            trimmed.startsWith("#") -> {
                flushPara()
                blocks.add(MdBlock.Heading(trimmed.trimStart('#').trim()))
            }
            trimmed.startsWith("> ") || trimmed == ">" -> {
                flushPara()
                val parts = ArrayList<String>()
                var k = i
                while (k < lines.size && (lines[k].trim().startsWith("> ") || lines[k].trim() == ">")) {
                    parts.add(lines[k].trim().removePrefix(">").trim())
                    k++
                }
                blocks.add(MdBlock.Quote(parts.joinToString("\n")))
                i = k - 1
            }
            trimmed.matches(Regex("^[-*+] .+")) -> {
                flushPara()
                blocks.add(MdBlock.Bullet(trimmed.drop(2).trim()))
            }
            trimmed.matches(Regex("^\\d+\\. .+")) -> {
                flushPara()
                numbered++
                blocks.add(MdBlock.Numbered(numbered, trimmed.dropWhile { it != ' ' }.trim()))
            }
            trimmed.isEmpty() -> flushPara()
            else -> {
                if (para.isNotEmpty()) para.append(" ")
                para.append(trimmed)
            }
        }
        i++
    }
    flushPara()
    return blocks
}