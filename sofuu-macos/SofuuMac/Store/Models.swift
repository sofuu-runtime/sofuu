import Foundation

/// Full app state — the Swift mirror of the Android app's `AppState`
/// (which mirrors the desktop's `AppState`).
struct AppState {
    var engineReady = false
    var bootError: String?
    var sessions: [SessionInfo] = []
    var activeSessionId: String?
    var messages: [ChatMessage] = []
    var messageCache: [String: [ChatMessage]] = [:]
    var streaming = false
    var approvals: [Approval] = []
    var config: SofuuConfig?
    var projectDir: String?
    var workspaces: [String] = []
    var permissions = "prompt"
    var settingsOpen = false
    var settingsTab = "general"
    var chatFontSize = 15
    var mode = "agent" // agent | chat
    var preview: SessionPreview?
    var theme = "system" // light | dark | system
    var cloudBg = true
    var pickedImages: [String] = []
    /// @mentions spliced into the composer by the file picker.
    var pendingMentions = ""
    /// Text handed over from outside the app (macOS Services "Send to
    /// Sofuu", sofuu:// routes); the visible Composer consumes and clears it.
    var composerDraft: String?
    /// First-run welcome card (UserDefaults-gated, shown once).
    var onboardingVisible = false
    /// ⇧⌘/ keyboard-shortcut cheat sheet.
    var shortcutHelpVisible = false
    /// Chat display-name overrides (session id → name), persisted in
    /// UiPrefs; rendered by pills and the history popover.
    var sessionNames: [String: String] = [:]
    /// ⇧⌘S chat search bar + its jump handshake (target message id; the
    /// tick changes alongside so re-jumping the same row still scrolls).
    var chatSearchOpen = false
    var chatSearchTarget: String?
    var chatSearchTick = 0
    /// Engine surface probe says sofuu.ai.transcribe exists in this build.
    /// The full gate also checks the active provider (Composer param) —
    /// the engine rejects Local/Anthropic audio at call time as well.
    var voiceSurfaceOK = false
    var phase: String?
    /// TUI-style live streaming state (chat.rs's answer spinner + think
    /// line + PHASE sweep, ported 1:1).
    var streamFrame = 0        // spinner + sweep cursor, advances every 120ms
    var liveThink = ""         // joined think deltas (the "▸ thinking …" line)
    var ctx: CtxMeter?
    var ctxCache: [String: CtxMeter] = [:]
    var sheet: String? // "model" | "effort" | "layers" | "mode" | "permissions"
    var modelsCache: [String: ModelCacheEntry] = [:]
    /// Absolute paths of the GGUF models imported into ~/.sofuu/models.
    var localModels: [String] = []

    // ── Mac-only preferences (desktop localStorage parity) ──────────
    /// Interface zoom, 0.5–2.0 in 0.1 steps (localStorage sofuu.zoom).
    var zoom: Double = 1.0
    /// ⌘B — the workspaces sidebar collapses to a rail-less hidden state.
    var sidebarCollapsed = false
    // ── IDE mode: file tree | tabbed code viewport | agent panel ────
    var ideMode = false
    var openTabs: [String] = []          // absolute file paths, in tab order
    var activeTab: String?
    var tabDirty: [String: Bool] = [:]   // edited vs on-disk per open path
    // IDE panel widths (the splitters); AppStore seeds these from UiPrefs.
    var ideTreeWidth: CGFloat = 230
    var ideAgentWidth: CGFloat = 380
    /// File-watcher pulse: bumped (debounced) whenever the workspace
    /// changed on disk — the tree re-lists on any state change, so a
    /// nudge is all it takes.
    var externalTick = 0
    /// Dirty tabs whose file changed on disk while we hold unsaved work.
    var externalEdits: Set<String> = []
    /// Read-only git working-tree status (path → kind), driven by the
    /// watcher + manual refresh.
    var gitStatus: [String: GitStatusKind] = [:]
    /// Tabs opened beyond the 1 MB edit cap (≤ 20 MB) — view-only.
    var readOnlyTabs: Set<String> = []
}

/// One provider's cached model listing (the desktop's model cache).
struct ModelCacheEntry {
    var models: [String] = []
    var ok = false
    var error: String?
    var note: String?
    var cached = false
    var at: Int = 0
}

/// The slash-command registry — ONE list feeds the "/" palette and /help.
struct SlashCommand: Identifiable {
    let name: String
    let arg: String?
    let desc: String
    var id: String { name }
    var signature: String { "/" + name + (arg.map { " \($0)" } ?? "") }
}

let SLASH_COMMANDS: [SlashCommand] = [
    SlashCommand(name: "new", arg: nil, desc: "Fresh session — clears the chat"),
    SlashCommand(name: "compact", arg: nil, desc: "Fold old history into a summary"),
    SlashCommand(name: "ctx", arg: "[tokens]", desc: "Show context usage, or set window"),
    SlashCommand(name: "maxout", arg: "[tokens]", desc: "Show/set max output tokens"),
    SlashCommand(name: "model", arg: "[name]", desc: "Show or set the model"),
    SlashCommand(name: "provider", arg: "[name]", desc: "Show or set the provider"),
    SlashCommand(name: "effort", arg: "[low|medium|high|max|off]", desc: "Show or set reasoning effort"),
    SlashCommand(name: "brain", arg: "[on|off]", desc: "Show or toggle brain memory"),
    SlashCommand(name: "remember", arg: "<fact>", desc: "Pin a fact to the brain"),
    SlashCommand(name: "why", arg: nil, desc: "Which memories shaped the last answer"),
    SlashCommand(name: "cost", arg: nil, desc: "Session tokens + spend"),
    SlashCommand(name: "resume", arg: "[id]", desc: "List sessions, or resume one"),
    SlashCommand(name: "permissions", arg: "[profile]", desc: "Show or set tool permissions"),
    SlashCommand(name: "tools", arg: nil, desc: "List built-in + MCP tools"),
    SlashCommand(name: "agents", arg: nil, desc: "List loaded agents"),
    SlashCommand(name: "rlm", arg: "[on|off|auto]", desc: "Show or set long-context mode"),
    SlashCommand(name: "ml", arg: "[on|off|info]", desc: "Context-economy gates"),
    SlashCommand(name: "sessions", arg: nil, desc: "List sessions in this workspace"),
    SlashCommand(name: "context", arg: nil, desc: "Show current engine context"),
    SlashCommand(name: "help", arg: nil, desc: "This list"),
]

let MAX_SESSIONS = 7

// ── Messages ────────────────────────────────────────────────────────

/// One entry in the tool timeline under an assistant message.
struct TimelineEntry: Identifiable {
    let id: String
    var kind: String // tool tool_result delegate recall plan think warn consolidated compact rlm allocgate mlgate todo
    var title: String
    var tool: String?
    var detail: String?
    var badge: String?
    var result: String?
    var ok: Bool?
    var runId: String?
    var durationMs: Int?
    var live = false
    var t: Int?
}

struct Usage {
    var promptTokens: Int?
    var completionTokens: Int?
    var costUsd: Double?
    var steps: Int?
    var servedBy: String?

    /// "gpt-4o · 1,204 in · 87 out · $0.0021 · 3 steps" (mirrors the
    /// desktop's usageLine).
    func line() -> String? {
        var bits: [String] = []
        if let servedBy { bits.append(servedBy) }
        if let promptTokens { bits.append("\(fmt(promptTokens)) in") }
        if let completionTokens { bits.append("\(fmt(completionTokens)) out") }
        if let costUsd, costUsd > 0 { bits.append("$" + String(format: "%.4f", costUsd)) }
        if let steps, steps > 0 { bits.append("\(steps) steps") }
        return bits.isEmpty ? nil : bits.joined(separator: " · ")
    }

    private func fmt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}

enum MsgRole { case user, assistant, system }

struct ChatMessage: Identifiable {
    let id: String
    var role: MsgRole
    var content = ""
    var thinking: String?
    var timeline: [TimelineEntry] = []
    var usage: Usage?
    var status = "done" // streaming | done | error
    var error: String?
}

struct Approval: Identifiable {
    let id: String
    let tool: String
    let args: JSONObject
    let risky: Bool
}

/// A past session's transcript loaded for read-only viewing.
struct SessionPreview {
    let id: String
    let label: String
    let turns: [PreviewTurn]
}

struct PreviewTurn: Identifiable {
    let id = UUID()
    let prompt: String
    let answer: String
}

struct SessionInfo: Identifiable {
    let id: String
    var pid = 0
    var host = "local"
    var cwd = ""
    var model = ""
    var provider = ""
    var startedAt = 0
    var lastSeen = 0
    var task: String?
    var ended = false

    /// "Chat N" numbered by creation order, or the session's task line.
    func labelOf(_ index: Int) -> String {
        if let task, !task.trimmingCharacters(in: .whitespaces).isEmpty { return task }
        return "Chat \(index + 1)"
    }

    static func fromJson(_ o: JSONObject) -> SessionInfo {
        SessionInfo(
            id: o.optString("id"),
            pid: o.optInt("pid"),
            host: o.optString("host", "local"),
            cwd: o.optString("cwd"),
            model: o.optString("model"),
            provider: o.optString("provider"),
            startedAt: o.optInt("started_at"),
            lastSeen: o.optInt("last_seen"),
            task: o.optString("task").nilIfEmpty,
            ended: o.optBool("ended"))
    }

    static func fromJsonArray(_ a: [Any]) -> [SessionInfo] {
        a.compactMap { $0 as? JSONObject }.map { fromJson($0) }
    }
}

/// Live context-window meter for the composer ring.
struct CtxMeter {
    var used = 0
    var window = 0
    var source = ""
    var model: String?
    var sid: String?
    /// Desktop hover-card extras: the engine's estimate of the NEXT
    /// request's shape (ctx event payload, chat.js).
    var clampedConfig = false
    var cacheHit: Double?
    var sections: JSONObject?

    var fraction: Double {
        window > 0 ? min(1, max(0, Double(used) / Double(window))) : 0
    }
    var pct: Int {
        window > 0 ? Int((Double(used) * 100 / Double(window)).rounded()) : 0
    }
}

/// One provider entry in the shared config (Settings → Providers).
struct ProviderEntry: Identifiable {
    var name: String
    var endpoint: String
    var model: String
    var profile = ""
    var hasApiKey = false
    var id: String { name }

    func toJson() -> JSONObject {
        ["name": name, "endpoint": endpoint, "model": model, "profile": profile]
    }

    static func fromJson(_ o: JSONObject) -> ProviderEntry {
        ProviderEntry(
            name: o.optString("name"),
            endpoint: o.optString("endpoint"),
            model: o.optString("model"),
            profile: o.optString("profile"),
            hasApiKey: !o.optString("api_key").isEmpty || o.optBool("has_api_key"))
    }
}

/// The shared ~/.sofuu/config.json shape (api keys redacted for the UI).
struct SofuuConfig {
    var provider = ""
    var model = ""
    var effort = "high"
    var brain = false
    var ml = true
    var rlm = "off"
    var ctxWindow = 0
    var maxOutput = 0
    var budgetUsd = 0.0
    var embedProvider = ""
    var embedModel = ""
    var recallMin = 0.0
    var recallBudget = 0
    var active = ""
    var providers: [ProviderEntry] = []
    var raw: JSONObject = [:]

    static func fromJson(_ o: JSONObject) -> SofuuConfig {
        SofuuConfig(
            provider: o.optString("provider"),
            model: o.optString("model"),
            effort: o.optString("effort", "high"),
            brain: o.optBool("brain", false),
            ml: o.optBool("ml", true),
            rlm: o.optString("rlm", "off"),
            ctxWindow: o.optLong("ctx_window"),
            maxOutput: o.optLong("max_output"),
            budgetUsd: o.optDouble("budget_usd"),
            embedProvider: o.optString("embed_provider"),
            embedModel: o.optString("embed_model"),
            recallMin: o.optDouble("recall_min"),
            recallBudget: o.optLong("recall_budget"),
            active: o.optString("active"),
            providers: (o.optArray("providers") ?? []).compactMap { $0 as? JSONObject }.map { ProviderEntry.fromJson($0) },
            raw: o)
    }

    /// Write the config back as a JSON patch, preserving unknown keys.
    func mergedWith(_ patch: JSONObject) -> JSONObject {
        var out = raw
        for (k, v) in patch { out[k] = v }
        return out
    }
}
