package com.sofuu.store

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import org.json.JSONObject
import java.io.File

/**
 * The shared ~/.sofuu/config.json + the secure key store — the Kotlin
 * port of the desktop backend's config.rs / keychain.rs.
 *
 * API keys never live only in config.json: the desktop keeps them in the
 * OS credential store and injects them at boot. Android does the same
 * with EncryptedSharedPreferences (Keystore-wrapped master key) and
 * folds them into the providers' api_key fields when writing config,
 * which is what chat.js reads per turn.
 */
class ConfigStore(private val context: Context) {

    private val sofuuDir: File get() = File(context.filesDir, ".sofuu")

    private val configFile: File get() = File(sofuuDir, "config.json")

    private val stateFile: File get() = File(sofuuDir, "mobile.json")

    private val prefs: SharedPreferences by lazy {
        EncryptedSharedPreferences.create(
            context,
            "sofuu_keys",
            MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    // ── config.json ───────────────────────────────────────────────

    fun readConfig(): JSONObject {
        return try {
            val text = configFile.readText()
            JSONObject(text)
        } catch (e: Exception) {
            JSONObject()
        }
    }

    fun writeConfig(cfg: JSONObject) {
        sofuuDir.mkdirs()
        // Inject the secure keys into the provider entries so the engine's
        // per-turn config read sees them (same role the desktop's boot-time
        // keychain injection plays).
        val providers = cfg.optJSONArray("providers")
        if (providers != null) {
            for (i in 0 until providers.length()) {
                val p = providers.optJSONObject(i) ?: continue
                val name = p.optString("name")
                if (name.isNotEmpty()) {
                    val key = keyFor(name)
                    if (!key.isNullOrEmpty()) p.put("api_key", key)
                }
            }
        }
        // Keep the flat mirror in sync with the ACTIVE entry — the same
        // merge chat.js's loadConfig performs, so a UI patch that changes
        // one field never clobbers the entry-derived truth.
        configFile.writeText(cfg.toString(2))
    }

    /** Patch the config (desktop update_config semantics: providers merge
     *  by name, unknown keys preserved). Returns the merged config. */
    fun updateConfig(patch: JSONObject): JSONObject {
        val cur = readConfig()
        val merged = JSONObject(cur.toString())
        for (k in patch.keys()) {
            val v = patch.get(k)
            if (k == "providers" && v is org.json.JSONArray) {
                merged.put("providers", mergeProviders(cur.optJSONArray("providers"), v))
            } else {
                merged.put(k, v)
            }
        }
        normalizeEndpoints(merged)
        syncFlatMirror(merged)
        writeConfig(merged)
        return merged
    }

    /**
     * OpenAI-wire providers need the FULL chat-completions URL — the
     * engine (chat.js) uses the entry's endpoint verbatim as the request
     * URL. A bare base like `https://api.river.ai/v1` answers
     * HTTP 200 with an empty body and a clean close, which the engine
     * reports as "stream ended without [DONE] (transfer closed) —
     * possible truncation" and burns the retry ladder. Same rule as the
     * desktop's normalizeEndpoints; anthropic/local endpoints carry their
     * own paths and are left alone. Idempotent.
     */
    private fun normalizeEndpoints(cfg: JSONObject) {
        val providers = cfg.optJSONArray("providers") ?: return
        for (i in 0 until providers.length()) {
            val p = providers.optJSONObject(i) ?: continue
            val profile = p.optString("profile").lowercase()
            if (profile == "anthropic" || profile == "local") continue
            val endpoint = p.optString("endpoint").trim()
            if (endpoint.isEmpty()) continue
            if (endpoint.lowercase().contains("/chat/completions")) continue
            p.put(
                "endpoint",
                endpoint + if (endpoint.endsWith("/")) "chat/completions" else "/chat/completions",
            )
        }
    }

    /**
     * Keep the legacy flat mirror on the active provider's entry — the
     * shape chat.js's loadConfig builds for itself, and what the composer
     * label and session rows read. An empty entry model clears the mirror
     * ("no model selected" until picked), matching the desktop.
     */
    private fun syncFlatMirror(cfg: JSONObject) {
        val active = cfg.optString("active")
        if (active.isEmpty()) return
        val providers = cfg.optJSONArray("providers") ?: return
        var entry: JSONObject? = null
        for (i in 0 until providers.length()) {
            val p = providers.optJSONObject(i) ?: continue
            if (p.optString("name") == active) { entry = p; break }
        }
        entry ?: return
        cfg.put("provider", entry.optString("name"))
        cfg.put("model", entry.optString("model"))
        cfg.put("base_url", entry.optString("endpoint"))
    }

    /**
     * Repair a config written before normalization existed (or by a tool
     * that stored a base URL). Runs once at boot; writes only when the
     * file actually changed.
     */
    fun repairEndpoints() {
        val cur = readConfig()
        if (cur.length() == 0) return
        val before = cur.toString()
        normalizeEndpoints(cur)
        syncFlatMirror(cur)
        if (cur.toString() != before) writeConfig(cur)
    }

    private fun mergeProviders(existing: org.json.JSONArray?, incoming: org.json.JSONArray): org.json.JSONArray {
        // Union by name: incoming wins, existing extras preserved (same
        // rule as the desktop's merge_providers).
        val byName = LinkedHashMap<String, JSONObject>()
        existing?.let { arr -> for (i in 0 until arr.length()) arr.optJSONObject(i)?.let { byName[it.optString("name")] = it } }
        for (i in 0 until incoming.length()) {
            val p = incoming.optJSONObject(i) ?: continue
            val name = p.optString("name")
            val prev = byName[name]
            byName[name] = if (prev != null) JSONObject(prev.toString()).also { out ->
                for (k in p.keys()) out.put(k, p.get(k))
            } else p
        }
        return org.json.JSONArray(byName.values.toList())
    }

    /** Delete a provider end-to-end: config entry + stored key. The
     *  active provider is refused (switch first) — desktop parity. */
    fun removeProvider(name: String): JSONObject {
        val cur = readConfig()
        if (cur.optString("active") == name || cur.optString("provider") == name) {
            throw IllegalStateException("cannot remove the active provider — switch first")
        }
        val providers = cur.optJSONArray("providers")
        val kept = org.json.JSONArray()
        providers?.let { arr ->
            for (i in 0 until arr.length()) {
                val p = arr.optJSONObject(i) ?: continue
                if (p.optString("name") != name) kept.put(p)
            }
        }
        val merged = JSONObject(cur.toString())
        merged.put("providers", kept)
        writeConfig(merged)
        deleteKey(name)
        return merged
    }

    /** Redacted view for the UI: has_api_key flags, no raw keys. */
    fun redacted(cfg: JSONObject = readConfig()): SofuuConfig {
        val out = JSONObject(cfg.toString())
        val providers = out.optJSONArray("providers")
        if (providers != null) {
            for (i in 0 until providers.length()) {
                val p = providers.optJSONObject(i) ?: continue
                val key = p.optString("api_key")
                if (key.isNotEmpty()) p.put("has_api_key", true)
                p.put("api_key", "")
            }
        }
        if (out.optString("api_key").isNotEmpty()) {
            out.put("has_api_key", true)
            out.put("api_key", "")
        }
        return SofuuConfig.fromJson(out)
    }

    // ── mobile.json (app state: project dir, workspaces) ──────────

    fun readState(): JSONObject = try {
        JSONObject(stateFile.readText())
    } catch (e: Exception) {
        JSONObject()
    }

    fun writeState(s: JSONObject) {
        sofuuDir.mkdirs()
        stateFile.writeText(s.toString(2))
    }

    fun projectDir(): String? =
        readState().optString("project_dir").takeIf { it?.isNotEmpty() == true }

    fun setProjectDir(path: String) {
        val s = readState()
        s.put("project_dir", path)
        writeState(s)
    }

    fun workspaces(): List<String> =
        readState().optJSONArray("workspaces")?.let { arr ->
            (0 until arr.length()).mapNotNull { arr.optString(it).takeIf { s -> s.isNotEmpty() } }
        } ?: emptyList()

    fun setWorkspaces(list: List<String>) {
        val s = readState()
        s.put("workspaces", org.json.JSONArray(list))
        writeState(s)
    }

    // ── The Keystore-backed key store ────────────────────────────

    fun keyFor(provider: String): String? = prefs.getString("key:$provider", null)

    fun setKey(provider: String, key: String) {
        prefs.edit().putString("key:$provider", key).apply()
    }

    fun deleteKey(provider: String) {
        prefs.edit().remove("key:$provider").apply()
    }

}

/** UI preferences that persist across launches (the desktop's
 *  localStorage equivalents). */
class UiPrefs(context: Context) {
    private val prefs = context.getSharedPreferences("sofuu.ui", Context.MODE_PRIVATE)

    var theme: String
        get() = prefs.getString("theme", "system") ?: "system"
        set(v) = prefs.edit().putString("theme", v).apply()

    var mode: String
        get() = prefs.getString("mode", "agent") ?: "agent"
        set(v) = prefs.edit().putString("mode", v).apply()

    var permissions: String
        get() = prefs.getString("permissions", "prompt") ?: "prompt"
        set(v) = prefs.edit().putString("permissions", v).apply()

    // Key bumped to "cloudBg2": installs that stored the old default
    // (true) must land on the NEW default (false) — the API-clean way.
    var cloudBg: Boolean
        get() = prefs.getBoolean("cloudBg2", false)
        set(v) = prefs.edit().putBoolean("cloudBg2", v).apply()

    var chatFontSize: Int
        get() = prefs.getInt("chatFontSize", 15)
        set(v) = prefs.edit().putInt("chatFontSize", v.coerceIn(13, 22)).apply()

    /** The pane Settings reopens on — a local `remember` lost it every time. */
    var settingsTab: String
        get() = prefs.getString("settingsTab", "general") ?: "general"
        set(v) = prefs.edit().putString("settingsTab", v).apply()
}
