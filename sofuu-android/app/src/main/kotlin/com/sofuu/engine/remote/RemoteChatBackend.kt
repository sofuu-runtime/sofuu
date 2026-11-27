package com.sofuu.engine.remote

import com.sofuu.engine.ChatBackend
import com.sofuu.engine.EngineEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLEncoder

/**
 * A chat backend whose engine runs in `sofuu host` on the LAN.
 *
 * Transport: blocking HTTP + SSE on Dispatchers.IO. Turns are detached on
 * the host, so killing this socket never kills the turn; [streamLoop]
 * reconnects from the persisted cursor and the replay ring supplies every
 * event the client missed. When the cursor is outside the ring (host
 * restart, or the ring evicted it) the host answers `sync/reset` and a
 * snapshot rebuilds the transcript.
 */
class RemoteChatBackend(
    val profile: HostProfile,
    private val cursors: RemoteCursorStore,
    private val emit: (EngineEvent) -> Unit,
) : ChatBackend {

    private val client = HostHttpClient(profile.baseUrl(), profile.token)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val streamLock = Any()
    private val cursorLock = Any()
    private val reconnectLock = Any()

    /** In-memory view of the replay cursor. Writing the encrypted store on
     *  every `answer_delta` would be one Keystore round-trip per token; the
     *  store is flushed on a short debounce and at every stream boundary.
     *  Losing the last sub-second of cursor is safe: a process restart
     *  resumes from the snapshot, not from the persisted cursor. */
    private val cursorCache = HashMap<String, Long>()
    /** Ring epoch that owns each cached cursor. Read with the cursor so a
     *  host restart invalidates the seq before the first reconnect. */
    private val epochCache = HashMap<String, String>()
    private var cursorJob: Job? = null

    @Volatile private var activeSession: String? = null
    @Volatile private var streamJob: Job? = null
    @Volatile private var streamSession: String? = null
    @Volatile private var currentRunId: String? = null
    @Volatile private var closed = false

    override val isRemote: Boolean get() = true
    override val hostId: String get() = profile.id

    /** GET /v1/host — used by the connect flow before the backend is adopted. */
    fun checkHost(): JSONObject? = try {
        client.request("GET", "/v1/host")
    } catch (_: Exception) {
        null
    }

    fun sessionId(): String? = activeSession

    /** Re-probe and re-attach after a network blip. Idempotent: tapping
     *  Connect while already on this host used to build a second backend.
     *  Serialized because onResume and the network callback can fire
     *  together. */
    fun reconnect() {
        synchronized(reconnectLock) {
            if (closed) return
            val info = checkHost()
            if (info == null) {
                emit(EngineEvent.RemoteStatus(connected = false, detail = "host unreachable"))
                return
            }
            emit(EngineEvent.RemoteStatus(connected = true, detail = null))
            val sid = activeSession ?: return
            // Break a half-open SSE read NOW. Android can keep a Wi-Fi
            // socket "alive" long after the network is gone; waiting for
            // the 45 s read timeout would blow the foreground-reconnect
            // budget. Restart the loop so a pending 8 s backoff does not
            // delay the first try.
            synchronized(streamLock) {
                streamJob?.cancel()
                streamJob = null
                streamSession = null
            }
            client.close()
            ensureStream(sid)
        }
    }

    // ── reads ─────────────────────────────────────────────────────

    override fun sessions(): JSONArray? = try {
        client.request("GET", "/v1/sessions").optJSONArray("sessions")
    } catch (e: Exception) {
        emit(EngineEvent.RemoteStatus(connected = false, detail = e.message))
        null
    }

    override fun sessionTurns(id: String): JSONArray? = try {
        client.request("GET", "/v1/sessions/${enc(id)}/snapshot").optJSONArray("turns")
    } catch (_: Exception) {
        null
    }

    override fun chatState(): JSONObject? {
        val sid = activeSession
        val host = try {
            client.request("GET", "/v1/host")
        } catch (e: Exception) {
            return JSONObject().put("ok", false).put("error", e.message)
        }
        val cfg = if (sid != null) {
            try {
                client.request("GET", "/v1/sessions/${enc(sid)}/snapshot").optJSONObject("config")
            } catch (_: Exception) {
                null
            }
        } else null
        return JSONObject()
            .put("ok", true)
            .put("host", profile.name)
            .put("sessionId", sid ?: JSONObject.NULL)
            .put("model", host.optString("model"))
            .put("provider", host.optString("provider"))
            .put("permissions", host.optString("permissions", "prompt"))
            .put("streaming", host.optBoolean("streaming"))
            .put("sessionTokens", cfg?.optLong("sessionTokens") ?: 0L)
            .put("contextTokens", cfg?.optLong("contextTokens") ?: 0L)
            .put("sessionSpendUsd", cfg?.optDouble("sessionSpendUsd") ?: 0.0)
    }

    // ── sessions ──────────────────────────────────────────────────

    override fun newSession(): JSONObject? {
        val r = try {
            client.request("POST", "/v1/sessions", JSONObject())
        } catch (e: Exception) {
            return JSONObject().put("ok", false).put("error", e.message)
        }
        val sid = r.optString("sessionId")
        if (sid.isBlank()) return JSONObject().put("ok", false).put("error", "host returned no session id")
        activeSession = sid
        currentRunId = null
        ensureStream(sid)
        return JSONObject().put("ok", true).put("sessionId", sid)
    }

    override fun resumeSession(id: String): JSONObject? {
        activeSession = id
        currentRunId = null
        val snap = try {
            client.request("GET", "/v1/sessions/${enc(id)}/snapshot")
        } catch (e: HostApiException) {
            return JSONObject().put("ok", false).put("error", e.message)
        } catch (e: Exception) {
            return JSONObject().put("ok", false).put("error", e.message)
        }
        applySnapshot(id, snap)
        ensureStream(id)
        return JSONObject().put("ok", true).put("sessionId", id)
    }

    override fun deleteSession(id: String): JSONObject? = try {
        val r = client.request("DELETE", "/v1/sessions/${enc(id)}")
        dropCursor(id)
        if (activeSession == id) {
            activeSession = null
            currentRunId = null
            streamJob?.cancel()
            streamJob = null
            streamSession = null
        }
        JSONObject().put("ok", true).put("ids", JSONArray().put(id)).also {
            it.put("wasActive", r.optBoolean("wasActive"))
        }
    } catch (e: Exception) {
        JSONObject().put("ok", false).put("error", e.message)
    }

    override fun clearSessions(): JSONObject? {
        val ids = JSONArray()
        val failed = JSONArray()
        val list = try {
            client.request("GET", "/v1/sessions").optJSONArray("sessions") ?: JSONArray()
        } catch (e: Exception) {
            return JSONObject().put("ok", false).put("error", e.message)
        }
        for (i in 0 until list.length()) {
            val sid = list.optJSONObject(i)?.optString("id").orEmpty()
            if (sid.isBlank()) continue
            try {
                client.request("DELETE", "/v1/sessions/${enc(sid)}")
                ids.put(sid)
                dropCursor(sid)
            } catch (_: Exception) {
                failed.put(sid)
            }
        }
        activeSession = null
        currentRunId = null
        streamJob?.cancel()
        streamJob = null
        streamSession = null
        return JSONObject().put("ok", true).put("ids", ids).put("failed", failed)
    }

    // ── turn control ──────────────────────────────────────────────

    override fun submitTurn(text: String, noTools: Boolean, images: List<String>) {
        val sid = activeSession
        if (sid == null) {
            emit(EngineEvent.TurnFinished(error = "no active host session", usage = null))
            return
        }
        val body = JSONObject().put("prompt", text).put("noTools", noTools)
        if (images.isNotEmpty()) body.put("images", JSONArray(images))
        try {
            val r = client.request("POST", "/v1/sessions/${enc(sid)}/runs", body)
            currentRunId = r.optString("runId").ifBlank { null }
            /* Deliberately do NOT jump the cursor to r.seq: the start/ctx
             * events are already in the ring and the stream must apply
             * them in order. */
            ensureStream(sid)
        } catch (e: HostApiException) {
            emit(EngineEvent.TurnFinished(error = e.message, usage = null))
            if (e.status == 409) emit(EngineEvent.Warn("A turn is already running on the host."))
            else emit(EngineEvent.RemoteStatus(connected = e.status > 0, detail = e.message))
        } catch (e: Exception) {
            emit(EngineEvent.TurnFinished(error = e.message, usage = null))
        }
    }

    override fun cancelTurn() {
        val sid = activeSession ?: return
        val run = currentRunId ?: "current"
        try {
            client.request("POST", "/v1/sessions/${enc(sid)}/runs/${enc(run)}/cancel", JSONObject())
        } catch (e: Exception) {
            emit(EngineEvent.Warn("Cancel failed: ${e.message}"))
        }
    }

    override fun resolveApproval(id: String, allow: Boolean, always: Boolean) {
        val sid = activeSession ?: return
        try {
            client.request(
                "POST",
                "/v1/sessions/${enc(sid)}/approve",
                JSONObject().put("id", id).put("allow", allow).put("always", always),
            )
        } catch (e: Exception) {
            emit(EngineEvent.Warn("Approval failed: ${e.message}"))
        }
    }

    override fun setPermissions(profile: String): JSONObject? {
        val sid = activeSession
        if (sid == null) return JSONObject().put("ok", true).put("permissions", profile)
        return try {
            client.request(
                "POST",
                "/v1/sessions/${enc(sid)}/permissions",
                JSONObject().put("profile", profile),
            )
        } catch (e: Exception) {
            JSONObject().put("ok", false).put("error", e.message)
        }
    }

    override fun compactSession(): JSONObject? {
        val sid = activeSession
            ?: return JSONObject().put("ok", false).put("error", "no active host session")
        return try {
            client.request("POST", "/v1/sessions/${enc(sid)}/compact", JSONObject())
        } catch (e: Exception) {
            JSONObject().put("ok", false).put("error", e.message)
        }
    }

    override fun close() {
        closed = true
        flushCursors()
        streamJob?.cancel()
        streamJob = null
        client.close()
        scope.cancel()
    }

    override fun leaveSession() {
        synchronized(streamLock) {
            streamJob?.cancel()
            streamJob = null
            streamSession = null
        }
        flushCursors()
        activeSession = null
        currentRunId = null
        client.close()
    }

    // ── snapshot + stream ─────────────────────────────────────────

    private fun applySnapshot(sid: String, snap: JSONObject) {
        if (closed || activeSession != sid) return
        val latest = snap.optLong("latestSeq", 0L)
        val epoch = snap.optString("epoch")
        if (epoch.isNotEmpty()) noteEpoch(sid, epoch)
        resetCursor(sid, latest)
        flushCursors()
        val run = snap.optJSONObject("activeRun")
        currentRunId = run?.optString("runId")?.ifBlank { null }
        val cfg = snap.optJSONObject("config")
        emit(EngineEvent.SessionSnapshot(
            sessionId = sid,
            turns = snap.optJSONArray("turns") ?: JSONArray(),
            streaming = snap.optBoolean("streaming"),
            partialAnswer = snap.optString("partialAnswer"),
            pendingApprovals = snap.optJSONArray("pendingApprovals") ?: JSONArray(),
            latestSeq = latest,
            model = cfg?.optString("model").orEmpty(),
            provider = cfg?.optString("provider").orEmpty(),
        ))
        // The ring's `ctx` event predates the cursor, so a reconnecting
        // client would otherwise show a stale (or empty) context meter
        // until the next turn. The snapshot carries the live numbers.
        val win = cfg?.optLong("contextWindow") ?: 0L
        if (win > 0) {
            emit(EngineEvent.Ctx(
                used = cfg?.optLong("contextTokens")?.coerceAtLeast(0L) ?: 0L,
                window = win,
                source = "host",
                model = cfg?.optString("model")?.ifEmpty { null },
                sid = sid,
            ))
        }
        emit(EngineEvent.RemoteStatus(connected = true, detail = null))
    }

    private fun loadSnapshot(sid: String) {
        try {
            val snap = client.request("GET", "/v1/sessions/${enc(sid)}/snapshot")
            applySnapshot(sid, snap)
        } catch (e: Exception) {
            emit(EngineEvent.RemoteStatus(connected = false, detail = e.message))
        }
    }

    private fun ensureStream(sid: String) {
        if (closed) return
        synchronized(streamLock) {
            if (streamSession == sid && streamJob?.isActive == true) return
            streamJob?.cancel()
            streamSession = sid
            streamJob = scope.launch { streamLoop(sid) }
        }
    }

    private suspend fun streamLoop(sid: String) {
        var backoff = 500L
        while (!closed && activeSession == sid) {
            val after = cursorFor(sid)
            val epoch = epochFor(sid)
            var evPath = "/v1/sessions/${enc(sid)}/events?after_seq=$after&follow=1"
            if (epoch.isNotEmpty()) evPath += "&epoch=${enc(epoch)}"
            try {
                client.stream(
                    evPath,
                    onOpen = {
                        if (!closed && activeSession == sid) {
                            emit(EngineEvent.RemoteStatus(connected = true, detail = null))
                        }
                        backoff = 500L
                    },
                ) { ev ->
                    handleEnvelope(sid, ev)
                    backoff = 500L
                }
            } catch (e: Exception) {
                if (!closed && activeSession == sid) {
                    emit(EngineEvent.RemoteStatus(connected = false, detail = e.message))
                }
            }
            if (closed || activeSession != sid) break
            delay(backoff)
            backoff = (backoff * 2).coerceAtMost(8_000L)
        }
        flushCursors()
    }

    private fun handleEnvelope(sid: String, ev: JSONObject) {
        val kind = ev.optString("kind")
        if (kind == "sync") {
            val payload = ev.optJSONObject("payload")
            val epoch = payload?.optString("epoch").orEmpty()
            if (epoch.isNotEmpty()) noteEpoch(sid, epoch)
            if (payload?.optString("type") == "reset") loadSnapshot(sid)
            return
        }
        val seq = ev.optLong("seq", 0L)
        noteCursor(sid, seq)
        if (activeSession != sid) return
        val mapped = RemoteEventMapper.map(ev) ?: return
        if (mapped is EngineEvent.TurnFinished) {
            currentRunId = null
            flushCursors()
        }
        emit(mapped)
    }

    private fun cursorFor(sid: String): Long = synchronized(cursorLock) {
        cursorCache[sid] ?: cursors.getCursor(profile.id, sid).also { cursorCache[sid] = it }
    }

    private fun epochFor(sid: String): String = synchronized(cursorLock) {
        epochCache[sid] ?: cursors.getEpoch(profile.id, sid).also { epochCache[sid] = it }
    }

    private fun noteEpoch(sid: String, epoch: String) {
        if (epoch.isBlank()) return
        synchronized(cursorLock) {
            if (epochCache[sid] == epoch) return
            epochCache[sid] = epoch
        }
        cursors.setEpoch(profile.id, sid, epoch)
    }

    private fun noteCursor(sid: String, seq: Long) {
        if (seq <= 0) return
        synchronized(cursorLock) {
            val cur = cursorCache[sid] ?: 0L
            if (seq <= cur) return
            cursorCache[sid] = seq
            scheduleCursorFlushLocked()
        }
    }

    /** A snapshot is a resynchronization point: the host's sequence space
     *  may have restarted (a new host process starts the ring at 1). The
     *  cursor is therefore allowed to move BACKWARDS here; a monotonic
     *  update would keep replaying from a position the new ring can never
     *  reach and reset on every reconnect. */
    private fun resetCursor(sid: String, seq: Long) {
        val next = seq.coerceAtLeast(0L)
        synchronized(cursorLock) {
            cursorCache[sid] = next
            scheduleCursorFlushLocked()
        }
    }

    private fun scheduleCursorFlushLocked() {
        if (cursorJob?.isActive == true) return
        cursorJob = scope.launch {
            delay(750)
            flushCursors()
        }
    }

    /** Write every cached cursor. Called on a debounce and when a stream
     *  ends, so a host socket drop cannot lose the position it needs. */
    private fun flushCursors() {
        val batch: List<Pair<String, Long>> = synchronized(cursorLock) {
            cursorCache.map { it.key to it.value }
        }
        for ((sid, seq) in batch) {
            if (seq <= 0) cursors.clearCursor(profile.id, sid)
            else cursors.setCursor(profile.id, sid, seq)
        }
    }

    private fun dropCursor(sid: String) {
        synchronized(cursorLock) {
            cursorCache.remove(sid)
            epochCache.remove(sid)
        }
        cursors.clearCursor(profile.id, sid)
    }

    private fun enc(s: String): String =
        URLEncoder.encode(s, "UTF-8").replace("+", "%20")
}
