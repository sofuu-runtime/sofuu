package com.sofuu.store

import org.json.JSONObject

/** Timeline presentation helpers — a port of the desktop's timeline.ts:
 *  verb-first tool row titles, icon ids, phase labels, durations. */
object Timeline {

    // ── Icons (Compose material symbols map; the desktop's custom SVGs
    // become these Material icon ids) ──────────────────────────────
    const val ICON_READ = "read"
    const val ICON_WRITE = "write"
    const val ICON_EDIT = "edit"
    const val ICON_GREP = "grep"
    const val ICON_GLOB = "glob"
    const val ICON_FOLDER = "folder"
    const val ICON_TERMINAL = "terminal"
    const val ICON_WEB = "web"
    const val ICON_LINK = "link"
    const val ICON_DELEGATE = "delegate"
    const val ICON_CHECKLIST = "checklist"
    const val ICON_BRAIN = "brain"
    const val ICON_COMPRESS = "compress"
    const val ICON_GAUGE = "gauge"
    const val ICON_FRESHNESS = "freshness"
    const val ICON_RELEVANCE = "relevance"
    const val ICON_SUPERVISOR = "supervisor"
    const val ICON_LOOP = "loop"
    const val ICON_RLM = "rlm"
    const val ICON_WARN = "warn"
    const val ICON_NOTICE = "notice"
    const val ICON_PLAN = "plan"
    const val ICON_DOT = "dot"
    const val ICON_THINK = "think"

    private val TOOL_ICON = mapOf(
        "read_file" to ICON_READ,
        "write_file" to ICON_WRITE,
        "edit_file" to ICON_EDIT,
        "grep" to ICON_GREP,
        "glob" to ICON_GLOB,
        "list_dir" to ICON_FOLDER,
        "bash" to ICON_TERMINAL,
        "web_search" to ICON_WEB,
        "web_open" to ICON_LINK,
        "delegate" to ICON_DELEGATE,
        "todo_write" to ICON_CHECKLIST,
    )

    fun toolIcon(name: String): String = TOOL_ICON[name] ?: ICON_DOT

    fun iconIdOf(e: TimelineEntry): String = when (e.kind) {
        "tool" -> toolIcon(e.tool ?: e.title)
        "tool_result" -> if (e.title == "Notice") ICON_NOTICE else toolIcon(e.tool ?: e.title)
        "delegate" -> ICON_DELEGATE
        "recall" -> ICON_BRAIN
        "consolidated" -> ICON_COMPRESS
        "allocgate" -> ICON_GAUGE
        "mlgate" -> when (e.tool) {
            "freshness" -> ICON_FRESHNESS
            "relevance" -> ICON_RELEVANCE
            "loop" -> ICON_LOOP
            else -> ICON_SUPERVISOR
        }
        "rlm" -> ICON_RLM
        "todo" -> ICON_CHECKLIST
        "warn" -> if (Regex("compact|prun", RegexOption.IGNORE_CASE).containsMatchIn(e.title)) ICON_COMPRESS else ICON_WARN
        "think" -> ICON_THINK
        "plan" -> ICON_PLAN
        else -> ICON_DOT
    }

    fun isMonoIcon(icon: String): Boolean =
        icon == ICON_TERMINAL || icon == ICON_GREP || icon == ICON_GLOB

    // ── Verb-first row titles ─────────────────────────────────────

    fun basename(path: String): String {
        val parts = path.split("/")
        return parts.lastOrNull()?.takeIf { it.isNotEmpty() } ?: path
    }

    fun clip(s: String, n: Int): String = if (s.length > n) s.take(n) + "…" else s

    fun firstLine(s: String): String {
        val nl = s.indexOf("\n")
        return if (nl >= 0) s.substring(0, nl) else s
    }

    fun hostOf(url: String): String =
        Regex("^[a-z]+://([^/]+)").find(url)?.groupValues?.get(1) ?: clip(url, 40)

    data class ToolRow(val title: String, val detail: String? = null)

    /** {title, detail} for one tool call. `args` is the JSON-parsed args
     *  the engine clipped to 200 chars — may be null when the clip broke
     *  the JSON. */
    fun toolRow(name: String, args: JSONObject?): ToolRow {
        val a = args ?: JSONObject()
        fun str(k: String): String = a.optString(k).takeIf { it.isNotEmpty() } ?: ""
        fun opt(k: String): String = if (a.has(k) && !a.isNull(k)) a.get(k).toString() else ""
        when (name) {
            "read_file" -> {
                val path = str("path")
                if (path.isNotEmpty()) {
                    val range = if (a.has("start") || a.has("end")) {
                        "lines ${if (a.has("start")) a.opt("start") else 1}–${if (a.has("end")) a.opt("end") else "end"}"
                    } else null
                    return ToolRow("Read ${basename(path)}", range)
                }
            }
            "write_file" -> {
                val path = str("path")
                if (path.isNotEmpty()) return ToolRow("Wrote ${basename(path)}")
            }
            "edit_file" -> {
                val path = str("path")
                if (path.isNotEmpty()) return ToolRow("Edited ${basename(path)}")
            }
            "grep" -> {
                val pattern = str("pattern")
                if (pattern.isNotEmpty()) {
                    val path = str("path")
                    return ToolRow(
                        "Searched “${clip(pattern, 48)}”",
                        if (path.isNotEmpty()) "${clip(pattern, 200)}\nin $path" else clip(pattern, 200),
                    )
                }
            }
            "glob" -> {
                val pattern = str("pattern")
                if (pattern.isNotEmpty()) {
                    val path = str("path")
                    return ToolRow(
                        "Matched “${clip(pattern, 48)}”",
                        if (path.isNotEmpty()) "${clip(pattern, 200)}\nin $path" else clip(pattern, 200),
                    )
                }
            }
            "list_dir" -> {
                val path = str("path")
                return ToolRow("Listed ${if (path.isNotEmpty()) basename(path) else "project"}")
            }
            "bash" -> {
                val cmd = str("command")
                if (cmd.isNotEmpty()) return ToolRow("Ran ${clip(firstLine(cmd), 72)}", cmd)
            }
            "web_search" -> {
                val query = str("query")
                if (query.isNotEmpty()) return ToolRow("Searched the web for “${clip(query, 48)}”")
            }
            "web_open" -> {
                val url = str("url")
                if (url.isNotEmpty()) return ToolRow("Opened ${hostOf(url)}", url)
            }
        }
        // Built-in tool with unparseable args, or a dynamic (MCP) tool:
        // the raw name is the honest row.
        return ToolRow(name)
    }

    // ── Run summaries ────────────────────────────────────────────

    private val TOOL_RUN = mapOf(
        "read_file" to Triple("Read", "file", "files"),
        "write_file" to Triple("Wrote", "file", "files"),
        "edit_file" to Triple("Edited", "file", "files"),
        "grep" to Triple("Searched", "pattern", "patterns"),
        "glob" to Triple("Matched", "pattern", "patterns"),
        "list_dir" to Triple("Listed", "folder", "folders"),
        "bash" to Triple("Ran", "command", "commands"),
        "web_search" to Triple("Searched the web for", "query", "queries"),
        "web_open" to Triple("Opened", "page", "pages"),
        "delegate" to Triple("Delegated", "task", "tasks"),
    )

    private fun runClause(tool: String, count: Int): String {
        val run = TOOL_RUN[tool]
            ?: return if (count > 1) "Ran $tool ×$count" else "Ran $tool"
        return if (count > 1) "${run.first} $count ${run.third}" else "${run.first} 1 ${run.second}"
    }

    fun composeSummary(items: List<TimelineEntry>): String {
        val order = ArrayList<String>()
        val counts = LinkedHashMap<String, Int>()
        for (e in items) {
            val key = e.tool ?: e.title
            if (!counts.containsKey(key)) order.add(key)
            counts[key] = (counts[key] ?: 0) + 1
        }
        val clauses = order.map { runClause(it, counts[it] ?: 1) }
        val shown = clauses.take(3)
        val rest = clauses.size - shown.size
        return shown.joinToString(", ") + if (rest > 0) ", +$rest more" else ""
    }

    // ── Durations & gate titles ──────────────────────────────────

    fun formatDur(ms: Long): String =
        if (ms < 1000) "%.1fs".format(ms / 1000.0) else "${Math.round(ms / 1000.0)}s"

    fun mlgateTitle(rule: String): String = when (rule) {
        "freshness" -> "Freshness gate"
        "relevance" -> "Relevance gate"
        "loop" -> "Loop gate"
        else -> "Supervisor"
    }
}

/** Live phase label, in lockstep with the TUI's phaseForStep. Returns
 *  null for "keep the current phase". */
fun phaseForStep(kind: String, payload: JSONObject?): String? {
    val p = payload ?: JSONObject()
    if (kind == "plan" || kind == "think" || kind == "recall") return "Thinking"
    if (kind == "tool" || kind == "delegate") {
        val name = p.optString("name").ifEmpty { p.optString("agent") }.ifEmpty { p.optString("tool") }.ifEmpty { p.optString("action") }
        return when {
            Regex("web_search|web_open|search", RegexOption.IGNORE_CASE).containsMatchIn(name) -> "Web searching"
            Regex("debug|trace|error|fix|diagnos", RegexOption.IGNORE_CASE).containsMatchIn(name) -> "Debugging"
            Regex("write|create|edit|implement|refactor|generate", RegexOption.IGNORE_CASE).containsMatchIn(name) &&
                !Regex("debug|test", RegexOption.IGNORE_CASE).containsMatchIn(name) -> "Coding"
            Regex("read|list|scan|grep|inspect|review|audit|test|lint|check|verify", RegexOption.IGNORE_CASE).containsMatchIn(name) -> "Auditing"
            else -> "Working"
        }
    }
    if (kind == "tool_result") return null
    if (kind == "compact") return "Compressing"
    if (kind == "answer_delta") return "Working"
    if (kind.startsWith("rlm")) return "Working"
    return null
}

/** The engine clips tool args to 200 chars, so the string often isn't
 *  valid JSON — parse defensively, null on failure. */
fun parseToolArgs(raw: String?): JSONObject? {
    if (raw.isNullOrEmpty()) return null
    return try {
        val v = JSONObject(raw)
        v
    } catch (e: Exception) {
        null
    }
}
