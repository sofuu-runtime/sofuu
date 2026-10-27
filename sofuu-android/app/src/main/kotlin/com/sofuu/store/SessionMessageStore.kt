package com.sofuu.store

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * UI fast-path cache for chat conversations, activity timelines, tool calls
 * and edits.
 *
 * The SOURCE OF TRUTH is the engine's session document,
 * `<project>/.sofuu/sessions/<id>/session.json` (R4 / §5.2): turns, per-turn
 * ids, usage and redacted events. This store never writes into that folder —
 * an app-private cache lives under `filesDir/session_ui/` and is always
 * rebuildable from the engine (`sofuu.chat.sessionTurns` → turnsToMessages).
 * It exists so a session switch does not pay a JS round-trip, and it can
 * carry UI rows (activity entries, edits) that the engine document does not
 * model.
 *
 * Older app builds wrote `ui_messages.json` INSIDE the engine's session
 * folder. That file is a second authoring format in the canonical store; it
 * is migrated into the cache on first load and then deleted.
 */
object SessionMessageStore {

    private const val FILE_NAME = "ui_messages.json"
    private const val CACHE_DIR = "session_ui"

    fun save(context: Context, sessionId: String, messages: List<ChatMessage>) {
        if (sessionId.isBlank() || messages.isEmpty()) return
        try {
            val jsonArr = JSONArray()
            for (m in messages) {
                jsonArr.put(m.toJson())
            }
            val cacheDir = File(context.filesDir, CACHE_DIR)
            cacheDir.mkdirs()
            File(cacheDir, "$sessionId.json").writeText(jsonArr.toString())
        } catch (e: Exception) {
            android.util.Log.w("SessionMessageStore", "Failed to save messages for $sessionId: ${e.message}")
        }
    }

    fun load(context: Context, projectDir: String?, sessionId: String): List<ChatMessage>? {
        if (sessionId.isBlank()) return null
        try {
            // 1. The app-private cache.
            val cacheFile = File(context.filesDir, "$CACHE_DIR/$sessionId.json")
            if (cacheFile.exists() && cacheFile.length() > 0) {
                val cached = parseJsonMessages(cacheFile.readText())
                if (cached.isNotEmpty()) return cached
            }

            // 2. One-time migration of the legacy in-workspace file. Copy it
            //    into the cache, then remove it so the engine's session folder
            //    holds only engine-owned files.
            val legacy = legacyFile(projectDir, sessionId)
            if (legacy != null && legacy.exists() && legacy.length() > 0) {
                val list = parseJsonMessages(legacy.readText())
                if (list.isNotEmpty()) {
                    save(context, sessionId, list)
                    legacy.delete()
                    return list
                }
                legacy.delete()
            }
        } catch (e: Exception) {
            android.util.Log.w("SessionMessageStore", "Failed to load messages for $sessionId: ${e.message}")
        }
        return null
    }

    fun delete(context: Context, projectDir: String?, sessionId: String) {
        if (sessionId.isBlank()) return
        try {
            File(context.filesDir, "$CACHE_DIR/$sessionId.json").delete()
            legacyFile(projectDir, sessionId)?.delete()
        } catch (_: Exception) {}
    }

    fun deleteAll(context: Context, projectDir: String?) {
        try {
            File(context.filesDir, CACHE_DIR).deleteRecursively()
            // Clean up only OUR deprecated artifact; the session folders,
            // session.json and registry files belong to the engine.
            if (!projectDir.isNullOrBlank()) {
                File(projectDir, ".sofuu/sessions").listFiles()?.forEach { dir ->
                    if (dir.isDirectory) File(dir, FILE_NAME).delete()
                }
            }
        } catch (_: Exception) {}
    }

    /** Returns all session IDs discovered locally on disk. */
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
            val cacheDir = File(context.filesDir, CACHE_DIR)
            if (cacheDir.exists()) {
                cacheDir.listFiles()?.forEach { f ->
                    if (f.isFile && f.name.endsWith(".json")) {
                        ids.add(f.name.removeSuffix(".json"))
                    }
                }
            }
        } catch (_: Exception) {}
        return ids.toList()
    }

    /** Returns the first user message text to use as session title, or null. */
    fun getSessionTask(context: Context, projectDir: String?, sessionId: String): String? {
        val msgs = load(context, projectDir, sessionId) ?: return null
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
     * Converts past turns from engine.sessionTurns(id) into rich ChatMessage
     * items with reconstructed user bubbles and assistant activity timelines.
     */
    fun turnsToMessages(turns: JSONArray): List<ChatMessage> {
        val list = ArrayList<ChatMessage>()
        for (i in 0 until turns.length()) {
            val turn = turns.optJSONObject(i) ?: continue
            val prompt = turn.optString("prompt", "").trim()
            val answer = turn.optString("answer", "").trim()
            val tools = turn.optJSONArray("tools")
            val usageObj = turn.optJSONObject("usage")
            val servedBy = turn.optString("servedBy").takeIf { it.isNotEmpty() }

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
