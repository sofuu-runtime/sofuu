package com.sofuu.store

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * App-side token accounting, one line per finished turn.
 *
 * The engine already writes a `usage` record per turn, and `chat.usage()`
 * aggregates those — but they live in the workspace's QTSQ session store.
 * On an engine build without the QTSQ codec (allowed by
 * `SOFUU_ALLOW_NO_QTSQ=1`, and the reason the Makefile calls such a build
 * "INVISIBLY broken") that aggregation reads nothing and every Usage pane is
 * zero while the transcript still shows "1,385 in · 718 out".
 *
 * Those numbers reach the app on `TurnFinished`, so this ledger keeps them
 * independently: append-only JSONL in filesDir, capped, no engine call.
 */
class UsageLedger(context: Context) {

    private val file = File(context.filesDir, "usage-ledger.jsonl")
    private val lock = Any()

    data class Entry(
        val t: Long,
        val model: String,
        val provider: String,
        val pt: Long,
        val ct: Long,
        val cost: Double,
        val sid: String?,
    )

    fun append(e: Entry) {
        synchronized(lock) {
            runCatching {
                if (file.length() > MAX_BYTES) trim()
                file.appendText(JSONObject().apply {
                    put("t", e.t)
                    put("model", e.model)
                    put("provider", e.provider)
                    put("pt", e.pt)
                    put("ct", e.ct)
                    put("cost", e.cost)
                    put("sid", e.sid ?: "")
                }.toString() + "\n")
            }
        }
    }

    private fun trim() {
        val keep = readAll().takeLast(MAX_ROWS)
        file.writeText(keep.joinToString("\n") { it.toString() } + if (keep.isEmpty()) "" else "\n")
    }

    private fun readAll(): List<JSONObject> {
        if (!file.exists()) return emptyList()
        return runCatching {
            file.readLines().mapNotNull { line ->
                if (line.isBlank()) null else runCatching { JSONObject(line) }.getOrNull()
            }
        }.getOrDefault(emptyList())
    }

    /** Same shape `sofuu.chat.usage()` returns, so the pane renders either. */
    fun snapshot(): JSONObject {
        val rows = synchronized(lock) { readAll() }
        val dayFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val byModel = LinkedHashMap<String, JSONObject>()
        val byDay = LinkedHashMap<String, JSONObject>()
        val bySession = LinkedHashMap<String, JSONObject>()
        var totPt = 0L; var totCt = 0L; var totCost = 0.0; var totTurns = 0L

        for (r in rows) {
            val pt = r.optLong("pt"); val ct = r.optLong("ct")
            val cost = r.optDouble("cost", 0.0)
            if (pt <= 0 && ct <= 0) continue
            val model = r.optString("model").ifEmpty { "unknown" }
            val sid = r.optString("sid")
            val t = r.optLong("t") * 1000L

            val m = byModel.getOrPut(model) {
                JSONObject().put("model", model).put("pt", 0L).put("ct", 0L)
                    .put("cr", 0L).put("cw", 0L).put("turns", 0L).put("cost", 0.0)
            }
            m.put("pt", m.optLong("pt") + pt)
            m.put("ct", m.optLong("ct") + ct)
            m.put("turns", m.optLong("turns") + 1)
            m.put("cost", m.optDouble("cost", 0.0) + cost)

            val day = dayFmt.format(Date(t))
            val d = byDay.getOrPut(day) {
                JSONObject().put("day", day).put("pt", 0L).put("ct", 0L).put("turns", 0L)
            }
            d.put("pt", d.optLong("pt") + pt)
            d.put("ct", d.optLong("ct") + ct)
            d.put("turns", d.optLong("turns") + 1)

            if (sid.isNotEmpty()) {
                val s = bySession.getOrPut(sid) {
                    JSONObject().put("id", sid).put("model", model)
                        .put("started_at", r.optLong("t")).put("pt", 0L).put("ct", 0L)
                        .put("turns", 0L).put("cost", 0.0)
                }
                s.put("pt", s.optLong("pt") + pt)
                s.put("ct", s.optLong("ct") + ct)
                s.put("turns", s.optLong("turns") + 1)
                s.put("cost", s.optDouble("cost", 0.0) + cost)
                s.put("last_seen", r.optLong("t"))
            }

            totPt += pt; totCt += ct; totCost += cost; totTurns++
        }

        return JSONObject()
            .put("ok", true)
            .put("source", "app")
            .put("version", 1)
            .put("models", JSONArray(byModel.values.toList()))
            .put("days", JSONArray(byDay.values.sortedBy { it.optString("day") }))
            .put("sessions", JSONArray(bySession.values.sortedByDescending { it.optLong("last_seen") }))
            .put(
                "totals", JSONObject()
                    .put("pt", totPt).put("ct", totCt).put("cr", 0L).put("cw", 0L)
                    .put("turns", totTurns)
                    .put("cost", Math.round(totCost * 1e6) / 1e6)
                    .put("sessions", bySession.size),
            )
    }

    companion object {
        private const val MAX_ROWS = 20_000
        private const val MAX_BYTES = 2L * 1024 * 1024
    }
}
