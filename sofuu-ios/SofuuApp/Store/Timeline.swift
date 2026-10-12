import Foundation

/// Timeline presentation helpers — a port of the Android app's `Timeline.kt`
/// (itself a port of the desktop's `timeline.ts`): verb-first tool row titles,
/// icon ids, phase labels, durations.
enum Timeline {

    // ── Icons (SF Symbol names; the desktop's custom SVGs become these) ──
    static let iconRead = "read"
    static let iconWrite = "write"
    static let iconEdit = "edit"
    static let iconGrep = "grep"
    static let iconGlob = "glob"
    static let iconFolder = "folder"
    static let iconTerminal = "terminal"
    static let iconWeb = "web"
    static let iconLink = "link"
    static let iconDelegate = "delegate"
    static let iconChecklist = "checklist"
    static let iconBrain = "brain"
    static let iconCompress = "compress"
    static let iconGauge = "gauge"
    static let iconFreshness = "freshness"
    static let iconRelevance = "relevance"
    static let iconSupervisor = "supervisor"
    static let iconLoop = "loop"
    static let iconRLM = "rlm"
    static let iconWarn = "warn"
    static let iconNotice = "notice"
    static let iconPlan = "plan"
    static let iconDot = "dot"
    static let iconThink = "think"

    private static let toolIconMap: [String: String] = [
        "read_file": iconRead,
        "write_file": iconWrite,
        "edit_file": iconEdit,
        "grep": iconGrep,
        "glob": iconGlob,
        "list_dir": iconFolder,
        "bash": iconTerminal,
        "web_search": iconWeb,
        "web_open": iconLink,
        "delegate": iconDelegate,
        "todo_write": iconChecklist,
    ]

    static func toolIcon(_ name: String) -> String { toolIconMap[name] ?? iconDot }

    static func iconId(of e: TimelineEntry) -> String {
        switch e.kind {
        case "tool":
            return toolIcon(e.tool ?? e.title)
        case "tool_result":
            return e.title == "Notice" ? iconNotice : toolIcon(e.tool ?? e.title)
        case "delegate": return iconDelegate
        case "recall": return iconBrain
        case "consolidated": return iconCompress
        case "allocgate": return iconGauge
        case "mlgate":
            switch e.tool {
            case "freshness": return iconFreshness
            case "relevance": return iconRelevance
            case "loop": return iconLoop
            default: return iconSupervisor
            }
        case "rlm": return iconRLM
        case "todo": return iconChecklist
        case "warn":
            return e.title.range(of: "compact|prun", options: [.regularExpression, .caseInsensitive]) != nil
                ? iconCompress : iconWarn
        case "think": return iconThink
        case "plan": return iconPlan
        default: return iconDot
        }
    }

    static func isMonoIcon(_ icon: String) -> Bool {
        icon == iconTerminal || icon == iconGrep || icon == iconGlob
    }

    // ── Verb-first row titles ────────────────────────────────────────

    static func basename(_ path: String) -> String {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        if let last = parts.last, !last.isEmpty { return String(last) }
        return path
    }

    static func clip(_ s: String, _ n: Int) -> String {
        s.count > n ? String(s.prefix(n)) + "…" : s
    }

    static func firstLine(_ s: String) -> String {
        if let nl = s.firstIndex(of: "\n") { return String(s[s.startIndex..<nl]) }
        return s
    }

    static func hostOf(_ url: String) -> String {
        if let r = url.range(of: "^[a-z]+://([^/]+)", options: .regularExpression) {
            let m = String(url[r])
            if let schemeEnd = m.range(of: "://") {
                return String(m[schemeEnd.upperBound...])
            }
        }
        return clip(url, 40)
    }

    struct ToolRow {
        var title: String
        var detail: String?
    }

    /// {title, detail} for one tool call. `args` is the JSON-parsed args the
    /// engine clipped to 200 chars — may be nil when the clip broke the JSON.
    static func toolRow(_ name: String, _ args: JSONObject?) -> ToolRow {
        let a = args ?? [:]
        func str(_ k: String) -> String { a.optString(k) }

        switch name {
        case "read_file":
            let path = str("path")
            if !path.isEmpty {
                var range: String?
                if a.has("start") || a.has("end") {
                    let start = a.has("start") ? "\(a.optInt("start"))" : "1"
                    let end = a.has("end") ? "\(a.optInt("end"))" : "end"
                    range = "lines \(start)–\(end)"
                }
                return ToolRow(title: "Read \(basename(path))", detail: range)
            }
        case "write_file":
            let path = str("path")
            if !path.isEmpty { return ToolRow(title: "Wrote \(basename(path))", detail: nil) }
        case "edit_file":
            let path = str("path")
            if !path.isEmpty { return ToolRow(title: "Edited \(basename(path))", detail: nil) }
        case "grep":
            let pattern = str("pattern")
            if !pattern.isEmpty {
                let path = str("path")
                let detail = path.isEmpty ? clip(pattern, 200) : "\(clip(pattern, 200))\nin \(path)"
                return ToolRow(title: "Searched “\(clip(pattern, 48))”", detail: detail)
            }
        case "glob":
            let pattern = str("pattern")
            if !pattern.isEmpty {
                let path = str("path")
                let detail = path.isEmpty ? clip(pattern, 200) : "\(clip(pattern, 200))\nin \(path)"
                return ToolRow(title: "Matched “\(clip(pattern, 48))”", detail: detail)
            }
        case "list_dir":
            let path = str("path")
            return ToolRow(title: "Listed \(path.isEmpty ? "project" : basename(path))", detail: nil)
        case "bash":
            let cmd = str("command")
            if !cmd.isEmpty { return ToolRow(title: "Ran \(clip(firstLine(cmd), 72))", detail: cmd) }
        case "web_search":
            let query = str("query")
            if !query.isEmpty { return ToolRow(title: "Searched the web for “\(clip(query, 48))”", detail: nil) }
        case "web_open":
            let url = str("url")
            if !url.isEmpty { return ToolRow(title: "Opened \(hostOf(url))", detail: url) }
        default:
            break
        }
        // Built-in tool with unparseable args, or a dynamic (MCP) tool: the
        // raw name is the honest row.
        return ToolRow(title: name, detail: nil)
    }

    // ── Run summaries ────────────────────────────────────────────────

    private static let toolRun: [String: (String, String, String)] = [
        "read_file": ("Read", "file", "files"),
        "write_file": ("Wrote", "file", "files"),
        "edit_file": ("Edited", "file", "files"),
        "grep": ("Searched", "pattern", "patterns"),
        "glob": ("Matched", "pattern", "patterns"),
        "list_dir": ("Listed", "folder", "folders"),
        "bash": ("Ran", "command", "commands"),
        "web_search": ("Searched the web for", "query", "queries"),
        "web_open": ("Opened", "page", "pages"),
        "delegate": ("Delegated", "task", "tasks"),
    ]

    private static func runClause(_ tool: String, _ count: Int) -> String {
        guard let run = toolRun[tool] else {
            return count > 1 ? "Ran \(tool) ×\(count)" : "Ran \(tool)"
        }
        return count > 1 ? "\(run.0) \(count) \(run.2)" : "\(run.0) 1 \(run.1)"
    }

    static func composeSummary(_ items: [TimelineEntry]) -> String {
        var order: [String] = []
        var counts: [String: Int] = [:]
        for e in items {
            let key = e.tool ?? e.title
            if counts[key] == nil { order.append(key) }
            counts[key, default: 0] += 1
        }
        let clauses = order.map { runClause($0, counts[$0] ?? 1) }
        let shown = Array(clauses.prefix(3))
        let rest = clauses.count - shown.count
        return shown.joined(separator: ", ") + (rest > 0 ? ", +\(rest) more" : "")
    }

    // ── Durations & gate titles ──────────────────────────────────────

    static func formatDur(_ ms: Int) -> String {
        ms < 1000 ? String(format: "%.1fs", Double(ms) / 1000.0) : "\(Int((Double(ms) / 1000.0).rounded()))s"
    }

    static func mlgateTitle(_ rule: String) -> String {
        switch rule {
        case "freshness": return "Freshness gate"
        case "relevance": return "Relevance gate"
        case "loop": return "Loop gate"
        default: return "Supervisor"
        }
    }
}

/// Live phase label, in lockstep with the TUI's phaseForStep. Returns nil for
/// "keep the current phase".
func phaseForStep(_ kind: String, _ payload: JSONObject?) -> String? {
    let p = payload ?? [:]
    if kind == "plan" || kind == "think" || kind == "recall" { return "Thinking" }
    if kind == "tool" || kind == "delegate" {
        let name = p.optString("name")
            .nilIfEmpty ?? p.optString("agent").nilIfEmpty ?? p.optString("tool").nilIfEmpty ?? p.optString("action")
        func matches(_ pattern: String) -> Bool {
            name.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }
        if matches("web_search|web_open|search") { return "Web searching" }
        if matches("debug|trace|error|fix|diagnos") { return "Debugging" }
        if matches("write|create|edit|implement|refactor|generate") && !matches("debug|test") { return "Coding" }
        if matches("read|list|scan|grep|inspect|review|audit|test|lint|check|verify") { return "Auditing" }
        return "Working"
    }
    if kind == "tool_result" { return nil }
    if kind == "compact" { return "Compressing" }
    if kind == "answer_delta" { return "Working" }
    if kind.hasPrefix("rlm") { return "Working" }
    return nil
}

/// The engine clips tool args to 200 chars, so the string often isn't valid
/// JSON — parse defensively, nil on failure.
func parseToolArgs(_ raw: String?) -> JSONObject? {
    guard let raw, !raw.isEmpty else { return nil }
    return parseJSONObject(raw)
}
