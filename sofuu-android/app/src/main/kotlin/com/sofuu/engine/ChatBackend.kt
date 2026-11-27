package com.sofuu.engine

import org.json.JSONArray
import org.json.JSONObject

/**
 * The chat operations the UI drives, independent of where the engine runs.
 *
 * [SofuuEngine] is the local (in-process) implementation; a paired
 * `sofuu host` on the LAN is driven through `RemoteChatBackend`. Methods
 * BLOCK on I/O for the remote case, so callers must stay off the Main
 * dispatcher — the same rule the store already follows for engine futures.
 */
interface ChatBackend {
    val isRemote: Boolean
    val hostId: String?

    fun sessions(): JSONArray?
    fun sessionTurns(id: String): JSONArray?
    fun newSession(): JSONObject?
    fun resumeSession(id: String): JSONObject?
    fun deleteSession(id: String): JSONObject?
    fun clearSessions(): JSONObject?

    fun submitTurn(text: String, noTools: Boolean, images: List<String> = emptyList())
    fun cancelTurn()
    fun resolveApproval(id: String, allow: Boolean, always: Boolean = false)

    fun setPermissions(profile: String): JSONObject?
    fun chatState(): JSONObject?
    fun compactSession(): JSONObject?

    /** Release sockets/jobs. The local backend owns nothing. */
    fun close() {}

    /** Stop following the current session without cancelling its turn.
     *  `newChat` uses this so a later snapshot from the old session cannot
     *  re-open it under a fresh chat. */
    fun leaveSession() {}
}
