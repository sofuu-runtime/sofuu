package com.sofuu.engine.remote

import com.sofuu.engine.EngineEvent
import com.sofuu.engine.Usage
import org.json.JSONObject

/**
 * Maps the host's chat-event envelope to the SAME [EngineEvent] values the
 * in-process engine emits. The UI reducer is therefore identical in local
 * and remote mode — no second timeline implementation to drift.
 */
object RemoteEventMapper {

    fun map(e: JSONObject): EngineEvent? {
        val kind = e.optString("kind")
        if (kind.isEmpty()) return null
        val p = e.optJSONObject("payload") ?: JSONObject()
        return when (kind) {
            "ctx" -> EngineEvent.Ctx(
                used = p.optLong("used").coerceAtLeast(0),
                window = p.optLong("window").coerceAtLeast(0),
                source = p.optString("source"),
                model = p.optString("model").ifEmpty { null },
                sid = p.optString("sid").ifEmpty { e.optString("sid").ifEmpty { null } },
            )
            "start" -> EngineEvent.TurnStarted(
                sid = p.optString("sessionId").ifEmpty { e.optString("sid").ifEmpty { null } },
            )
            "done" -> EngineEvent.TurnFinished(
                error = p.optJSONObject("error")?.optString("message")?.ifEmpty { null },
                usage = parseUsage(p),
            )
            "approval_request" -> EngineEvent.ApprovalRequested(
                id = p.optString("id"),
                tool = p.optString("tool").ifEmpty { "tool" },
                args = p.optJSONObject("args") ?: JSONObject(),
                risky = p.opt("risky") != false,
            )
            "approval_resolved" -> EngineEvent.ApprovalResolved(id = p.optString("id"))
            "warn" -> EngineEvent.Warn(message = p.optString("message"))
            else -> EngineEvent.Step(kind, e)
        }
    }

    private fun parseUsage(p: JSONObject): Usage? {
        val u = p.optJSONObject("usage") ?: p
        val servedBy = u.optString("servedBy").ifEmpty { p.optString("servedBy") }.ifEmpty { null }
        val prompt = u.optLong("promptTokens", -1)
        val completion = u.optLong("completionTokens", -1)
        val cost = u.optDouble("costUsd", -1.0)
        val steps = u.optInt("steps", -1)
        if (prompt < 0 && completion < 0 && cost < 0 && servedBy == null && steps < 0) return null
        return Usage(
            promptTokens = if (prompt >= 0) prompt else null,
            completionTokens = if (completion >= 0) completion else null,
            costUsd = if (cost >= 0) cost else null,
            steps = if (steps >= 0) steps else null,
            servedBy = servedBy,
        )
    }
}
