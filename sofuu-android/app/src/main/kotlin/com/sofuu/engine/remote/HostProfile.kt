package com.sofuu.engine.remote

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** A paired `sofuu host` on the LAN. */
data class HostProfile(
    val id: String,
    val name: String,
    val address: String,
    val token: String,
    val pairedAt: Long = 0L,
    val lastSeen: Long = 0L,
) {
    /** Base URL with a scheme and no trailing slash. Never guesses a port:
     *  the host prints its LAN URL and the user pastes it. */
    fun baseUrl(): String {
        var a = address.trim()
        if (a.isEmpty()) return a
        if (!a.startsWith("http://") && !a.startsWith("https://")) a = "http://$a"
        return a.trimEnd('/')
    }

    fun displayAddress(): String = baseUrl().removePrefix("http://").removePrefix("https://")

    fun toJson(): JSONObject = JSONObject()
        .put("id", id)
        .put("name", name)
        .put("address", address)
        .put("token", token)
        .put("paired_at", pairedAt)
        .put("last_seen", lastSeen)

    companion object {
        fun fromJson(o: JSONObject): HostProfile = HostProfile(
            id = o.optString("id"),
            name = o.optString("name").ifBlank { "Sofuu host" },
            address = o.optString("address"),
            token = o.optString("token"),
            pairedAt = o.optLong("paired_at"),
            lastSeen = o.optLong("last_seen"),
        )

        fun create(name: String, address: String, token: String): HostProfile =
            HostProfile(
                id = UUID.randomUUID().toString(),
                name = name.ifBlank { "Sofuu host" },
                address = address.trim(),
                token = token.trim(),
                pairedAt = System.currentTimeMillis() / 1000,
            )
    }
}

/** Cursor + host persistence. Kept as an interface so the remote backend
 *  can be unit-tested without Android's SharedPreferences. */
interface RemoteCursorStore {
    fun getCursor(hostId: String, sessionId: String): Long
    fun setCursor(hostId: String, sessionId: String, seq: Long)
    fun clearCursor(hostId: String, sessionId: String)

    /** Epoch of the ring that issued the stored cursor. A host restart
     *  reuses seq numbers from 1, so the cursor is only trustworthy while
     *  the epoch matches. Blank means "unknown" (older store). */
    fun getEpoch(hostId: String, sessionId: String): String = ""
    fun setEpoch(hostId: String, sessionId: String, epoch: String) {}
}

/**
 * Paired hosts, the active selection, and per-(host, session) replay
 * cursors. Device tokens are bearer credentials, so this file lives in
 * EncryptedSharedPreferences (Keystore-wrapped master key) — the same
 * store the app already uses for provider API keys, never plaintext.
 */
class HostStore(context: Context) : RemoteCursorStore {
    private val prefs = context.applicationContext
        .getSharedPreferences(LEGACY_PREFS, Context.MODE_PRIVATE)

    private val secure by lazy {
        val sp = try {
            EncryptedSharedPreferences.create(
                context.applicationContext,
                "sofuu_hosts_secure",
                MasterKey.Builder(context.applicationContext)
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                    .build(),
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
            )
        } catch (e: Exception) {
            // A broken Keystore must not brick chat: fall back to the
            // app-private (unencrypted) store rather than crash on boot.
            android.util.Log.w("HostStore", "secure host store unavailable: ${e.message}")
            prefs
        }
        migrateLegacyPlaintext(sp)
        sp
    }

    /** v1 never shipped with plaintext hosts, but a development build may
     *  have written one. Move it, then erase it. */
    private fun migrateLegacyPlaintext(target: SharedPreferences) {
        if (target === prefs) return
        try {
            val raw = prefs.getString(KEY_HOSTS, null) ?: return
            if (!target.contains(KEY_HOSTS)) target.edit().putString(KEY_HOSTS, raw).commit()
            prefs.edit().remove(KEY_HOSTS).remove(KEY_ACTIVE).commit()
        } catch (_: Exception) { /* leave it; the secure store is the target */ }
    }

    fun hosts(): List<HostProfile> {
        val raw = secure.getString(KEY_HOSTS, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            (0 until arr.length()).mapNotNull { i ->
                arr.optJSONObject(i)?.let { HostProfile.fromJson(it) }
            }.filter { it.address.isNotBlank() && it.token.isNotBlank() }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun save(profile: HostProfile) {
        val next = hosts().toMutableList()
        val idx = next.indexOfFirst { it.id == profile.id }
        if (idx >= 0) next[idx] = profile else next.add(profile)
        writeHosts(next)
    }

    fun remove(id: String) {
        writeHosts(hosts().filterNot { it.id == id })
        if (activeId() == id) setActiveId(null)
    }

    fun activeId(): String? = secure.getString(KEY_ACTIVE, null)?.takeIf { it.isNotBlank() }

    fun setActiveId(id: String?) {
        secure.edit().apply {
            if (id.isNullOrBlank()) remove(KEY_ACTIVE) else putString(KEY_ACTIVE, id)
        }.apply()
    }

    fun touch(profile: HostProfile) {
        save(profile.copy(lastSeen = System.currentTimeMillis() / 1000))
    }

    private fun writeHosts(list: List<HostProfile>) {
        val arr = JSONArray()
        list.forEach { arr.put(it.toJson()) }
        secure.edit().putString(KEY_HOSTS, arr.toString()).apply()
    }

    override fun getCursor(hostId: String, sessionId: String): Long =
        secure.getLong("cursor:$hostId:$sessionId", 0L)

    override fun setCursor(hostId: String, sessionId: String, seq: Long) {
        if (seq <= 0) return
        secure.edit().putLong("cursor:$hostId:$sessionId", seq).apply()
    }

    override fun clearCursor(hostId: String, sessionId: String) {
        secure.edit()
            .remove("cursor:$hostId:$sessionId")
            .remove("epoch:$hostId:$sessionId")
            .apply()
    }

    override fun getEpoch(hostId: String, sessionId: String): String =
        secure.getString("epoch:$hostId:$sessionId", null).orEmpty()

    override fun setEpoch(hostId: String, sessionId: String, epoch: String) {
        if (epoch.isBlank()) return
        secure.edit().putString("epoch:$hostId:$sessionId", epoch).apply()
    }

    private companion object {
        const val KEY_HOSTS = "hosts"
        const val KEY_ACTIVE = "active_host"
        const val LEGACY_PREFS = "sofuu_hosts"
    }
}
