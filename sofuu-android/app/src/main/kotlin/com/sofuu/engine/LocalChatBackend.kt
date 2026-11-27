package com.sofuu.engine

import org.json.JSONArray
import org.json.JSONObject

/**
 * The in-process engine behind [ChatBackend]. This is the exact surface the
 * app used before remote hosts existed — the wrapper only turns the engine's
 * futures into blocking calls so both backends share one shape.
 */
class LocalChatBackend(private val engine: SofuuEngine) : ChatBackend {
    override val isRemote: Boolean get() = false
    override val hostId: String? get() = null

    override fun sessions(): JSONArray? = engine.sessions().get()

    override fun sessionTurns(id: String): JSONArray? = engine.sessionTurns(id).get()

    override fun newSession(): JSONObject? = engine.newSession().get()

    override fun resumeSession(id: String): JSONObject? = engine.resumeSession(id).get()

    override fun deleteSession(id: String): JSONObject? = engine.deleteSession(id).get()

    override fun clearSessions(): JSONObject? = engine.clearSessions().get()

    override fun submitTurn(text: String, noTools: Boolean, images: List<String>) {
        engine.submitTurn(text, noTools = noTools, images = images)
    }

    override fun cancelTurn() = engine.cancelTurn()

    override fun resolveApproval(id: String, allow: Boolean, always: Boolean) {
        engine.resolveApproval(id, allow, always)
    }

    override fun setPermissions(profile: String): JSONObject? = engine.setPermissions(profile).get()

    override fun chatState(): JSONObject? = engine.chatState().get()

    override fun compactSession(): JSONObject? = engine.compactSession().get()
}
