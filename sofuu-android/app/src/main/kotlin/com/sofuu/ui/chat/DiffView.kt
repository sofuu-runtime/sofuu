package com.sofuu.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sofuu.ui.theme.LocalSofuuColors

private const val MAX_LINES = 400

data class DiffRow(val kind: String, val text: String) // ctx | del | add

/** LCS line diff (the desktop's DiffView algorithm, ported). */
fun diffLines(oldText: String, newText: String): List<DiffRow> {
    val a = oldText.split("\n")
    val b = newText.split("\n")
    if (a.size > MAX_LINES || b.size > MAX_LINES) {
        return listOf(
            DiffRow("del", "… ${a.size} old lines (too large to diff inline) …"),
            DiffRow("add", "… ${b.size} new lines (too large to diff inline) …"),
        )
    }
    // dp[i][j] = LCS length of a[i..] and b[j..]
    val dp = Array(a.size + 1) { IntArray(b.size + 1) }
    for (i in a.indices.reversed()) {
        for (j in b.indices.reversed()) {
            dp[i][j] = if (a[i] == b[j]) dp[i + 1][j + 1] + 1 else maxOf(dp[i + 1][j], dp[i][j + 1])
        }
    }
    val rows = ArrayList<DiffRow>()
    var i = 0
    var j = 0
    while (i < a.size && j < b.size) {
        when {
            a[i] == b[j] -> {
                rows.add(DiffRow("ctx", a[i])); i++; j++
            }
            dp[i + 1][j] >= dp[i][j + 1] -> rows.add(DiffRow("del", a[i++]))
            else -> rows.add(DiffRow("add", b[j++]))
        }
    }
    while (i < a.size) rows.add(DiffRow("del", a[i++]))
    while (j < b.size) rows.add(DiffRow("add", b[j++]))
    return rows
}

/** Red/green line diff for write/edit tool results (desktop parity). */
@Composable
fun DiffView(oldText: String, newText: String, modifier: Modifier = Modifier) {
    val c = LocalSofuuColors.current
    val rows = remember(oldText, newText) { diffLines(oldText, newText) }
    val hScroll = rememberScrollState()
    val vScroll = rememberScrollState()

    Column(
        modifier
            .fillMaxWidth()
            .heightIn(max = 320.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(c.codeBlock)
            .border(1.dp, c.border, RoundedCornerShape(8.dp))
            .verticalScroll(vScroll)
            .horizontalScroll(hScroll)
            .padding(vertical = 4.dp),
    ) {
        var line = 1
        for (r in rows) {
            val n = if (r.kind == "add") null else line++
            val bg = when (r.kind) {
                "add" -> c.diffAddBg
                "del" -> c.diffDelBg
                else -> androidx.compose.ui.graphics.Color.Transparent
            }
            val fg = when (r.kind) {
                "add" -> c.diffAddFg
                "del" -> c.diffDelFg
                else -> c.textMuted
            }
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(bg)
                    .padding(horizontal = 8.dp, vertical = 2.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Text(
                    when (r.kind) {
                        "add" -> "+"
                        "del" -> "−"
                        else -> " "
                    },
                    color = fg,
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.width(16.dp),
                )
                Text(
                    n?.toString().orEmpty(),
                    color = c.textFaint,
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.width(36.dp),
                )
                Text(r.text.ifEmpty { " " }, color = fg, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }
    }
}
