package com.sofuu.store

import org.json.JSONArray
import org.json.JSONObject

/** One entry in the tool timeline under an assistant message. */
data class TimelineEntry(
    val id: String,
    val kind: String, // tool tool_result delegate recall plan think warn consolidated compact rlm allocgate mlgate todo
    val title: String,
    val tool: String? = null,
    val detail: String? = null,
    val badge: String? = null,
    val result: String? = null,
    val ok: Boolean? = null,
    val runId: String? = null,
    val durationMs: Long? = null,
    val live: Boolean = false,
    val t: Long? = null,
) {
    companion object {
        fun raw(id: String, kind: String, title: String) = TimelineEntry(id, kind, title)
    }
}

data class Usage(
    val promptTokens: Long? = null,
    val completionTokens: Long? = null,
    val costUsd: Double? = null,
    val steps: Int? = null,
    val servedBy: String? = null,
) {
    /** "gpt-4o · 1,204 in · 87 out · $0.0021 · 3 steps" (mirrors the
     *  desktop's usageLine). */
    fun line(): String? {
        val bits = ArrayList<String>()
        servedBy?.let { bits += it }
        promptTokens?.let { bits += "${fmt(it)} in" }
        completionTokens?.let { bits += "${fmt(it)} out" }
        costUsd?.takeIf { it > 0 }?.let { bits += "$${"%.4f".format(it)}" }
        steps?.takeIf { it > 0 }?.let { bits += "$it steps" }
        return if (bits.isEmpty()) null else bits.joinToString(" · ")
    }

    private fun fmt(n: Long): String = "%,d".format(n)
}

enum class MsgRole { USER, ASSISTANT, SYSTEM }

data class ChatMessage(
    val id: String,
    val role: MsgRole,
    val content: String = "",
    val thinking: String? = null,
    val timeline: List<TimelineEntry> = emptyList(),
    val usage: Usage? = null,
    val status: String = "done", // streaming | done | error
    val error: String? = null,
    /** Final speed line, set when the turn finishes:
     *  "22.4 tok/s · first 0.38s · 117 tok". */
    val metrics: String? = null,
)

data class Approval(
    val id: String,
    val tool: String,
    val args: JSONObject,
    val risky: Boolean,
)

/** A past session's transcript loaded for read-only viewing. */
data class SessionPreview(
    val id: String,
    val label: String,
    val turns: List<PreviewTurn>,
)

data class PreviewTurn(val prompt: String, val answer: String)

data class SessionInfo(
    val id: String,
    val pid: Int = 0,
    val host: String = "local",
    val cwd: String = "",
    val model: String = "",
    val provider: String = "",
    val startedAt: Long = 0,
    val lastSeen: Long = 0,
    val task: String? = null,
    val ended: Boolean = false,
) {
    /** "Chat N" numbered by creation order, or the session's task line. */
    fun labelOf(index: Int): String = task?.takeIf { it.isNotBlank() } ?: "Chat ${index + 1}"

    companion object {
        fun fromJson(o: JSONObject): SessionInfo = SessionInfo(
            id = o.optString("id"),
            pid = o.optInt("pid"),
            host = o.optString("host", "local"),
            cwd = o.optString("cwd"),
            model = o.optString("model"),
            provider = o.optString("provider"),
            startedAt = o.optLong("started_at"),
            lastSeen = o.optLong("last_seen"),
            task = o.optString("task").ifBlank { null },
            ended = o.optBoolean("ended"),
        )

        fun fromJsonArray(a: JSONArray): List<SessionInfo> =
            (0 until a.length()).mapNotNull { i -> a.optJSONObject(i)?.let { fromJson(it) } }
    }
}

/** Live context-window meter for the composer ring. */
data class CtxMeter(
    val used: Long = 0,
    val window: Long = 0,
    val source: String = "",
    val model: String? = null,
    val sid: String? = null,
) {
    val fraction: Float get() = if (window > 0) (used.toFloat() / window).coerceIn(0f, 1f) else 0f
    val pct: Int get() = if (window > 0) Math.round(used * 100.0 / window).toInt() else 0
}

/** One provider entry in the shared config (Settings → Providers). */
data class ProviderEntry(
    val name: String,
    val endpoint: String,
    val model: String,
    val profile: String = "",
    val hasApiKey: Boolean = false,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("name", name)
        .put("endpoint", endpoint)
        .put("model", model)
        .put("profile", profile)

    companion object {
        fun fromJson(o: JSONObject): ProviderEntry = ProviderEntry(
            name = o.optString("name"),
            endpoint = o.optString("endpoint"),
            model = o.optString("model"),
            profile = o.optString("profile"),
            hasApiKey = !o.optString("api_key").isNullOrEmpty() || o.optBoolean("has_api_key"),
        )
    }
}

/** The shared ~/.sofuu/config.json shape (api keys redacted for the UI). */
data class SofuuConfig(
    val provider: String = "",
    val model: String = "",
    val effort: String = "high",
    val brain: Boolean = true,
    val ml: Boolean = true,
    val rlm: String = "off",
    val ctxWindow: Long = 0,
    val maxOutput: Long = 0,
    val budgetUsd: Double = 0.0,
    val embedProvider: String = "",
    val embedModel: String = "",
    val recallMin: Double = 0.0,
    val recallBudget: Long = 0,
    val active: String = "",
    val providers: List<ProviderEntry> = emptyList(),
    val raw: JSONObject = JSONObject(),
) {
    companion object {
        fun fromJson(o: JSONObject): SofuuConfig = SofuuConfig(
            provider = o.optString("provider"),
            model = o.optString("model"),
            effort = o.optString("effort", "high"),
            brain = o.opt("brain") != false,
            ml = o.opt("ml") != false,
            rlm = o.optString("rlm", "off"),
            ctxWindow = o.optLong("ctx_window"),
            maxOutput = o.optLong("max_output"),
            budgetUsd = o.optDouble("budget_usd"),
            embedProvider = o.optString("embed_provider"),
            embedModel = o.optString("embed_model"),
            recallMin = o.optDouble("recall_min"),
            recallBudget = o.optLong("recall_budget"),
            active = o.optString("active"),
            providers = o.optJSONArray("providers")?.let { arr ->
                (0 until arr.length()).mapNotNull { i -> arr.optJSONObject(i)?.let { ProviderEntry.fromJson(it) } }
            } ?: emptyList(),
            raw = o,
        )
    }

    /** Write the config back as a JSON patch, preserving unknown keys. */
    fun mergedWith(patch: JSONObject): JSONObject {
        val out = JSONObject(raw.toString())
        for (k in patch.keys()) out.put(k, patch.get(k))
        return out
    }
}
