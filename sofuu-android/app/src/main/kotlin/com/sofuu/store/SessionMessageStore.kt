package com.sofuu.store

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * UI fast-path cache for chat conversations, activity timelines, tool calls
 * and edits.
 *
 * The SOURCE OF TRUTH is the owning engine's session document — the local
 * `<project>/.sofuu/sessions/<id>/session.json` (R4 / §5.2), or the paired
 * host's `/v1/sessions/:id/snapshot`. This store never writes into either;
 * an app-private cache lives under `filesDir/session_ui/<namespace>/` and is
 * always rebuildable from the engine's turns.
 *
 * [namespace] separates otherwise-colliding session ids across backends:
 * `"local"` for the in-process engine, `"host-<uuid>"` for a paired Mac.
 * Older app builds wrote `ui_messages.json` INSIDE the engine's session
 * folder; that file is migrated into the local cache on first load.
 */
object SessionMessageStore {

    private const val FILE_NAME = "ui_messages.json"
    private const val CACHE_DIR = "session_ui"
    private const val LOCAL_NS = "local"

    private fun safeNs(namespace: String): String {
        val ns = namespace.ifBlank { LOCAL_NS }
        if (ns.contains('/') || ns.contains('\\') || ns.startsWith(".") ||
            !ns.all { it.isLetterOrDigit() || it == '-' || it == '_' || it == '.' }
        ) return LOCAL_NS
        return ns
    }

    private fun cacheDir(context: Context, namespace: String): File =
        File(File(context.filesDir, CACHE_DIR), safeNs(namespace))

    private fun cacheFile(context: Context, namespace: String, sessionId: String): File =
        File(cacheDir(context, namespace), "$sessionId.json")

    fun save(
        context: Context,
        sessionId: String,
        messages: List<ChatMessage>,
        namespace: String = LOCAL_NS,
    ) {
        if (sessionId.isBlank() || messages.isEmpty()) return
        try {
            val jsonArr = JSONArray()
            for (m in messages) {
                jsonArr.put(m.toJson())
            }
            val dir = cacheDir(context, namespace)
            dir.mkdirs()
            File(dir, "$sessionId.json").writeText(jsonArr.toString())
        } catch (e: Exception) {
            android.util.Log.w("SessionMessageStore", "Failed to save messages for $sessionId: ${e.message}")
        }
    }

    fun load(
        context: Context,
        projectDir: String?,
        sessionId: String,
        namespace: String = LOCAL_NS,
    ): List<ChatMessage>? {
        if (sessionId.isBlank()) return null
        val ns = safeNs(namespace)
        try {
            // 1. The app-private cache for this backend.
            val file = cacheFile(context, ns, sessionId)
            if (file.exists() && file.length() > 0) {
                val cached = parseJsonMessages(file.readText())
                if (cached.isNotEmpty()) return cached
            }

            // 1b. Local cache written by an older build at the root.
            if (ns == LOCAL_NS) {
                val legacyRoot = File(File(context.filesDir, CACHE_DIR), "$sessionId.json")
                if (legacyRoot.exists() && legacyRoot.length() > 0) {
                    val list = parseJsonMessages(legacyRoot.readText())
                    if (list.isNotEmpty()) {
                        save(context, sessionId, list, LOCAL_NS)
                        legacyRoot.delete()
                        return list
                    }
                    legacyRoot.delete()
                }
            }

            // 2. One-time migration of the legacy in-workspace file (local
            //    engine only — a remote host never wrote into this folder).
            if (ns == LOCAL_NS) {
                val legacy = legacyFile(projectDir, sessionId)
                if (legacy != null && legacy.exists() && legacy.length() > 0) {
                    val list = parseJsonMessages(legacy.readText())
                    if (list.isNotEmpty()) {
                        save(context, sessionId, list, LOCAL_NS)
                        legacy.delete()
                        return list
                    }
                    legacy.delete()
                }
            }
        } catch (e: Exception) {
            android.util.Log.w("SessionMessageStore", "Failed to load messages for $sessionId: ${e.message}")
        }
        return null
    }

    fun delete(
        context: Context,
        projectDir: String?,
        sessionId: String,
        namespace: String = LOCAL_NS,
    ) {
        if (sessionId.isBlank()) return
        val ns = safeNs(namespace)
        try {
            cacheFile(context, ns, sessionId).delete()
            if (ns == LOCAL_NS) {
                File(File(context.filesDir, CACHE_DIR), "$sessionId.json").delete()
                legacyFile(projectDir, sessionId)?.delete()
            }
        } catch (_: Exception) {}
    }

    fun deleteAll(context: Context, projectDir: String?, namespace: String = LOCAL_NS) {
        val ns = safeNs(namespace)
        try {
            cacheDir(context, ns).deleteRecursively()
            if (ns == LOCAL_NS) {
                // Clean up only OUR deprecated artifact; the session folders,
                // session.json and registry files belong to the engine.
                if (!projectDir.isNullOrBlank()) {
                    File(projectDir, ".sofuu/sessions").listFiles()?.forEach { dir ->
                        if (dir.isDirectory) File(dir, FILE_NAME).delete()
                    }
                }
            }
        } catch (_: Exception) {}
    }

    /** Returns all locally-owned session IDs (never a remote host's). */
    fun getLocalSessionIds(context: Context, projectDir: String?): List<String> {
        val ids = LinkedHashSet<String>()
        try {
            // Read-only discovery: a session folder the engine created (by
            // the CLI or an older app build) should still appear in the list.
            if (!projectDir.isNullOrBlank()) {
                val sDir = File(projectDir, ".sofuu/sessions")
                if (sDir.exists()) {
                    sDir.listFiles()?.forEach { f ->
                        if (f.isDirectory && !f.name.startsWith(".")) {
                            ids.add(f.name)
                        }
                    }
                }
            }
            val localDir = cacheDir(context, LOCAL_NS)
            if (localDir.exists()) {
                localDir.listFiles()?.forEach { f ->
                    if (f.isFile && f.name.endsWith(".json")) {
                        ids.add(f.name.removeSuffix(".json"))
                    }
                }
            }
            // Pre-namespace builds wrote straight into session_ui/.
            val root = File(context.filesDir, CACHE_DIR)
            if (root.exists()) {
                root.listFiles()?.forEach { f ->
                    if (f.isFile && f.name.endsWith(".json")) {
                        ids.add(f.name.removeSuffix(".json"))
                    }
                }
            }
        } catch (_: Exception) {}
        return ids.toList()
    }

    /** Returns the first user message text to use as session title, or null. */
    fun getSessionTask(
        context: Context,
        projectDir: String?,
        sessionId: String,
        namespace: String = LOCAL_NS,
    ): String? {
        val msgs = load(context, projectDir, sessionId, namespace) ?: return null
        return msgs.firstOrNull { it.role == MsgRole.USER }?.content?.take(40)?.replace("\n", " ")
    }

    private fun legacyFile(projectDir: String?, sessionId: String): File? {
        if (projectDir.isNullOrBlank() || sessionId.isBlank()) return null
        if (sessionId.contains('/') || sessionId.contains('\\') || sessionId.startsWith(".")) return null
        return File(projectDir, ".sofuu/sessions/$sessionId/$FILE_NAME")
    }

    private fun parseJsonMessages(jsonText: String): List<ChatMessage> {
        val arr = JSONArray(jsonText)
        val list = ArrayList<ChatMessage>(arr.length())
        for (i in 0 until arr.length()) {
            val obj = arr.optJSONObject(i) ?: continue
            list.add(ChatMessage.fromJson(obj))
        }
        return list
    }

    /**
     * Converts past turns from the engine (`sessionTurns` / host snapshot)
     * into rich ChatMessage items with reconstructed user bubbles and
     * assistant activity timelines.
     */
    fun turnsToMessages(turns: JSONArray): List<ChatMessage> {
        val list = ArrayList<ChatMessage>()
        for (i in 0 until turns.length()) {
            val turn = turns.optJSONObject(i) ?: continue
            val prompt = turn.optString("prompt", "").trim()
            val answer = turn.optString("answer", "").trim()
            val tools = turn.optJSONArray("tools")
            val usageObj = turn.optJSONObject("usage")
            val servedBy = turn.optString("servedBy").ifEmpty { turn.optString("model") }
                .takeIf { it.isNotEmpty() }

            if (prompt.isNotEmpty()) {
                list.add(ChatMessage("h_u_${i}_${System.currentTimeMillis()}", MsgRole.USER, prompt))
            }

            val timeline = ArrayList<TimelineEntry>()
            if (tools != null) {
                for (j in 0 until tools.length()) {
                    val tObj = tools.optJSONObject(j) ?: continue
                    val name = tObj.optString("n", "tool")
                    val argsStr = tObj.optString("a", "")
                    val resStr = tObj.optString("r").takeIf { it.isNotEmpty() }

                    val parsedArgs = try {
                        if (argsStr.startsWith("{") && argsStr.endsWith("}")) JSONObject(argsStr) else null
                    } catch (_: Exception) { null }

                    val row = Timeline.toolRow(name, parsedArgs)
                    timeline.add(
                        TimelineEntry(
                            id = "tl_h_${i}_${j}",
                            kind = "tool",
                            title = row.title,
                            tool = name,
                            detail = row.detail ?: (if (argsStr.isNotEmpty()) argsStr else null),
                            result = resStr,
                            ok = true,
                        )
                    )
                }
            }

            val usage = if (usageObj != null) {
                Usage.fromJson(usageObj)
            } else if (servedBy != null) {
                Usage(servedBy = servedBy)
            } else null

            if (answer.isNotEmpty() || timeline.isNotEmpty()) {
                list.add(
                    ChatMessage(
                        id = "h_a_${i}_${System.currentTimeMillis()}",
                        role = MsgRole.ASSISTANT,
                        content = answer,
                        timeline = timeline,
                        usage = usage,
                        status = "done",
                    )
                )
            }
        }
        return list
    }
}
