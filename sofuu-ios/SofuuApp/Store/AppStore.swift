import Foundation
import SwiftUI

/// The app state holder: folds the engine event stream into UI state and
/// drives every engine call. A direct port of the Android app's `AppStore.kt`
/// (itself a port of the desktop's `store.ts`) — same reducer semantics, same
/// slash surface, same session discipline.
///
/// Everything here is `@MainActor`: the engine delivers events on the main
/// queue, and every `Task` created inside a method inherits that isolation,
/// so state is only ever touched on the main thread and streaming deltas
/// never interleave.
@MainActor
final class AppStore: ObservableObject {

    let configs = ConfigStore()
    private let uiPrefs = UiPrefs()

    /// The library's env-var knobs, applied before the runtime boots.
    let runtimeEnv = RuntimeEnv()

    let engine: SofuuEngine

    @Published private(set) var state: AppState

    /// Posted to the system notification centre when the app is backgrounded.
    var notifier: ((String, String) -> Void)?

    /// True while the UI is foregrounded (the App flips it via scenePhase).
    var foreground = true

    private var seq = 0
    private func nextId(_ prefix: String) -> String {
        seq += 1
        return "\(prefix)-\(String(Int(Date().timeIntervalSince1970 * 1000), radix: 36))-\(seq)"
    }

    init() {
        let savedProject = configs.projectDir() ?? AppPaths.defaultWorkspace.path
        try? FileManager.default.createDirectory(atPath: savedProject, withIntermediateDirectories: true)

        engine = SofuuEngine(homeDir: AppPaths.home, projectDir: savedProject)

        state = AppState(
            projectDir: configs.projectDir(),
            workspaces: configs.workspaces(),
            permissions: uiPrefs.permissions,
            chatFontSize: uiPrefs.chatFontSize,
            mode: uiPrefs.mode,
            theme: uiPrefs.theme,
            cloudBg: uiPrefs.cloudBg)

        // Engine events → state.
        engine.observe { [weak self] ev in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.handleEvent(ev)
                if case .turnFinished = ev { self.refreshSessions() }
            }
        }

        // Load config + state immediately (cheap, no engine).
        state.config = configs.redacted()
        // Engine environment knobs MUST be in place before the runtime is
        // created: SOFUU_NO_ML gates module registration inside the
        // library, so setting it afterwards silently does nothing.
        let appliedEnv = runtimeEnv.apply()
        NSLog("[sofuu-engine] env applied: \(appliedEnv)")
        engine.start()
        state.localModels = LocalModels.list().map(\.path)
    }

    func shutdown() { engine.shutdown() }

    /// Re-write the process environment from the current settings.
    ///
    /// Safe without a restart for everything the engine re-reads per turn
    /// or per request — the tool set, the shell gate, the memory backend,
    /// the web engine, the SSRF guard, the stall deadline. Those call
    /// sites run inside the agent loop, not once at load.
    func applyRuntimeEnv() { runtimeEnv.apply() }

    /// Tear the engine down and boot it again so a registration-gated knob
    /// takes effect. Only SOFUU_NO_ML needs this — sofuu-core registers
    /// the whole `sofuu.ml` namespace at init, so without a restart the
    /// namespace is simply absent and every gate is a silent no-op.
    func restartEngineForEnv() {
        if state.streaming {
            // Freeing the runtime mid-turn would strand the open tool
            // round-trip. Cancel first so the agent's abort path runs.
            Task { await engine.cancelTurn() }
        }
        engine.shutdown()
        runtimeEnv.apply()
        state.engineReady = false
        engine.start()
    }

    // ── Engine event reducer ─────────────────────────────────────────

    private func handleEvent(_ ev: EngineEvent) {
        switch ev {
        case .engineReady:
            state.engineReady = true
            state.bootError = nil
            onEngineReady()
        case .bootError(let message):
            state.engineReady = false
            state.bootError = message
        case .turnStarted:
            break // the assistant bubble is created on submit
        case .turnFinished(let error, let usage):
            state.streaming = false
            state.phase = nil
            if !foreground && error == nil { notifier?("Sofuu finished", "The turn completed.") }
            if let error {
                patchAssistant { $0.status = "error"; $0.error = error }
            } else {
                patchAssistant { $0.status = "done"; $0.usage = usage }
            }
        case .ctx(let used, let window, let source, let model, let sid):
            guard let owner = sid ?? state.activeSessionId else { return }
            let meter = CtxMeter(used: used, window: window, source: source, model: model, sid: sid)
            state.ctxCache[owner] = meter
            if owner == state.activeSessionId { state.ctx = meter }
        case .approvalRequested(let id, let tool, let args, let risky):
            state.approvals.append(Approval(id: id, tool: tool, args: args, risky: risky))
            if !foreground { notifier?("Sofuu needs your approval", "\(tool) is waiting for approval") }
        case .approvalResolved(let id):
            state.approvals.removeAll { $0.id == id }
        case .warn(let message):
            // Engine warnings ride as system rows (dim, compact) — the same
            // surface the desktop's step reducer writes.
            systemRow(message)
        case .step(let kind, let raw):
            applyStep(kind, raw)
        }
    }

    private func patchAssistant(_ fn: (inout ChatMessage) -> Void) {
        for i in state.messages.indices.reversed() where state.messages[i].role == .assistant {
            fn(&state.messages[i])
            break
        }
    }

    private var currentAssistant: ChatMessage? {
        state.messages.last { $0.role == .assistant }
    }

    /// The step reducer — the desktop's `applyStep`, in Swift.
    private func applyStep(_ kind: String, _ raw: JSONObject) {
        guard currentAssistant != nil else { return }
        let p = raw.optObject("payload") ?? [:]
        let runId = raw.optString("runId").nilIfEmpty
        let t = raw.has("t") ? raw.optInt("t") : nil
        let phase = phaseForStep(kind, p)

        switch kind {
        case "answer_delta":
            let text = p.optString("text")
            if text.isEmpty {
                state.phase = phase ?? state.phase
                return
            }
            patchAssistant { $0.content += text }

        case "think":
            let text = p.optString("text")
            if text.isEmpty {
                state.phase = phase ?? state.phase
                return
            }
            patchAssistant { m in
                if var last = m.timeline.last, last.kind == "think", last.runId == runId {
                    let started = last.t ?? t
                    var durationMs: Int?
                    if let t, let started { durationMs = max(0, t - started) }
                    last.detail = (last.detail ?? "") + text
                    last.durationMs = durationMs
                    if let durationMs, durationMs > 0 { last.badge = "· \(Timeline.formatDur(durationMs))" }
                    m.timeline[m.timeline.count - 1] = last
                } else {
                    m.timeline.append(TimelineEntry(
                        id: nextId("tl"), kind: "think", title: "Thought",
                        detail: text, runId: runId, t: t))
                }
            }

        case "tool_result":
            let isChecklist = p.optString("name") == "todo_write" && p.optArray("todos") != nil
            if !isChecklist {
                patchAssistant { m in
                    let ok = !p.has("error")
                    let result = p.optString("result").nilIfEmpty.map { Timeline.clip($0, 400) }
                        ?? p.optString("summary").nilIfEmpty
                    let name = p.optString("name")
                    for i in m.timeline.indices {
                        let e = m.timeline[i]
                        let match: Bool
                        if name == "delegate" {
                            match = e.kind == "delegate" && e.ok == nil
                        } else {
                            match = (e.kind == "tool" || e.kind == "tool_result") && e.ok == nil
                                && (e.tool != nil ? e.tool == name : e.title == name)
                        }
                        if !match { continue }
                        m.timeline[i].ok = ok
                        if let result { m.timeline[i].result = result }
                        return
                    }
                    // No match (system notice, result after a reload) — the
                    // standalone result row still renders.
                    if let fallback = stepToTimeline(kind, raw, p) {
                        m.timeline.append(fallback)
                    }
                }
            }

        case "compact":
            let stage = p.optString("stage")
            patchAssistant { m in
                let idx = m.timeline.firstIndex { $0.kind == "compact" && $0.live }
                if stage == "start" {
                    if idx == nil {
                        m.timeline.append(TimelineEntry(
                            id: nextId("tl"), kind: "compact", title: "Compressing", live: true, t: t))
                    }
                } else if let idx {
                    if !p.optBool("ok", true) {
                        m.timeline.remove(at: idx)
                    } else {
                        let saved = firstMatch("([\\d,]+) tk saved", m.timeline[idx].detail ?? "")
                        m.timeline[idx].live = false
                        m.timeline[idx].title = "Context compacted"
                        if let saved { m.timeline[idx].badge = "· \(saved) tk saved" }
                    }
                }
            }

        default:
            if kind == "warn",
               p.optString("message").range(of: "compact", options: [.regularExpression, .caseInsensitive]) != nil {
                let pending = currentAssistant?.timeline.contains { $0.kind == "compact" && $0.live } == true
                if pending {
                    let msg = p.optString("message")
                    patchAssistant { m in
                        if let idx = m.timeline.firstIndex(where: { $0.kind == "compact" && $0.live }) {
                            m.timeline[idx].detail = msg
                        }
                    }
                    return
                }
            }
            if let entry = stepToTimeline(kind, raw, p) {
                patchAssistant { m in
                    var e = entry
                    e.runId = runId
                    m.timeline.append(e)
                }
            }
        }
        state.phase = phase ?? state.phase
    }

    /// Step → timeline row (desktop `stepToTimeline`).
    private func stepToTimeline(_ kind: String, _ raw: JSONObject, _ p: JSONObject) -> TimelineEntry? {
        let t = raw.has("t") ? raw.optInt("t") : nil
        switch kind {
        case "tool":
            let name = p.optString("tool").nilIfEmpty ?? p.optString("name").nilIfEmpty ?? "tool"
            // delegate/todo_write have richer sibling events.
            if name == "delegate" || name == "todo_write" { return nil }
            let row = Timeline.toolRow(name, parseToolArgs(p.optString("args")))
            return TimelineEntry(id: nextId("tl"), kind: "tool", title: row.title,
                                 tool: name, detail: row.detail, t: t)

        case "tool_result":
            if p.optString("name") == "todo_write", let todos = p.optArray("todos") {
                let total = todos.count
                let done = todos.compactMap { $0 as? JSONObject }.filter { $0.optString("status") == "done" }.count
                return TimelineEntry(id: nextId("tl"), kind: "todo", title: "Checklist",
                                     detail: jsonEncode(todos), badge: "· \(done)/\(total) done", t: t)
            }
            let name = p.optString("tool").nilIfEmpty ?? p.optString("name")
            let resultText = p.optString("result").nilIfEmpty.map { Timeline.clip($0, 400) }
                ?? p.optString("summary").nilIfEmpty
            let ok = !p.has("error")
            if name == "system" {
                return TimelineEntry(id: nextId("tl"), kind: "tool_result", title: "Notice",
                                     tool: "system", result: resultText, ok: ok, t: t)
            }
            return TimelineEntry(id: nextId("tl"), kind: "tool_result",
                                 title: name.isEmpty ? "result" : name,
                                 tool: name.nilIfEmpty, result: resultText, ok: ok, t: t)

        case "delegate":
            return TimelineEntry(id: nextId("tl"), kind: "delegate",
                                 title: p.optString("agent").nilIfEmpty ?? "sub-agent",
                                 tool: "delegate", detail: p.optString("task").nilIfEmpty, t: t)

        case "recall":
            let count = p.optLong("count")
            var bits: [String] = []
            if count > 0 { bits.append("\(count) \(count == 1 ? "memory" : "memories")") }
            switch p.optString("scope") {
            case "agent": bits.append("private")
            case "shared": bits.append("shared brain")
            default: break
            }
            if p.optLong("dropped") > 0 { bits.append("\(p.optLong("dropped")) dropped") }
            return TimelineEntry(id: nextId("tl"), kind: "recall", title: "Recalled memories",
                                 badge: bits.isEmpty ? nil : "· " + bits.joined(separator: " · "), t: t)

        // The plan's only payload is "Step N" — dropped (desktop parity).
        case "plan":
            return nil

        case "warn":
            let msg = p.optString("message")
            if msg.hasPrefix("ml-compaction") {
                let freed = firstMatch("freed ~([\\d,]+) tk", msg)
                return TimelineEntry(id: nextId("tl"), kind: "warn", title: "Context pruned",
                                     detail: msg, badge: freed.map { "· ~\($0) tk freed" }, t: t)
            }
            if msg.range(of: "compact", options: [.regularExpression, .caseInsensitive]) != nil {
                return TimelineEntry(id: nextId("tl"), kind: "warn", title: "Context compacted", detail: msg, t: t)
            }
            return TimelineEntry(id: nextId("tl"), kind: "warn", title: "Warning",
                                 detail: msg.nilIfEmpty, t: t)

        case "consolidated":
            let clusters = p.optInt("clusters")
            return TimelineEntry(id: nextId("tl"), kind: "consolidated", title: "Memory consolidated",
                                 badge: clusters > 0 ? "· \(clusters) \(clusters == 1 ? "cluster" : "clusters")" : nil,
                                 t: t)

        case "allocgate":
            var bits: [String] = []
            if let fit = p.optStringy("fit") { bits.append(fit) }
            if let actions = p.optStringy("actions") { bits.append(actions) }
            if p.optString("action") == "learned_limit" {
                bits.append("learned \(p.optString("kind", "limit")) for \(p.optString("model", "model"))")
            }
            return TimelineEntry(id: nextId("tl"), kind: "allocgate", title: "Context gate",
                                 detail: bits.isEmpty ? nil : bits.joined(separator: " — "), t: t)

        case "mlgate":
            let rule = p.optString("rule")
            return TimelineEntry(id: nextId("tl"), kind: "mlgate", title: Timeline.mlgateTitle(rule),
                                 tool: rule.nilIfEmpty,
                                 detail: p.optString("nudge").nilIfEmpty ?? rule.nilIfEmpty, t: t)

        default:
            guard kind.hasPrefix("rlm") else { return nil }
            var detail: String?
            switch kind {
            case "rlm:route":
                let ctx = p.optStringy("ctxTokens") ?? "?"
                let win = p.optStringy("windowTokens") ?? "?"
                detail = "routed to RLM (\(ctx)/\(win) tk)"
            case "rlm:done":
                var bits: [String] = []
                if p.optInt("calls") > 0 { bits.append("\(p.optInt("calls")) LLM calls") }
                if p.optInt("toolCalls") > 0 { bits.append("\(p.optInt("toolCalls")) tool calls") }
                if p.optInt("rounds") > 0 { bits.append("\(p.optInt("rounds")) rounds") }
                detail = bits.isEmpty ? nil : bits.joined(separator: " · ")
            case "rlm:fallback":
                detail = p.optString("error").nilIfEmpty
            default:
                detail = nil
            }
            return TimelineEntry(id: nextId("tl"), kind: "rlm", title: "RLM run", tool: kind, detail: detail, t: t)
        }
    }

    // ── Submit + slash commands ──────────────────────────────────────

    func submit(_ text: String, images: [String] = []) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty && images.isEmpty { return }
        if trimmed.hasPrefix("/") {
            handleCommand(trimmed)
            return
        }
        let userText = (trimmed.isEmpty && !images.isEmpty) ? "[image]" : trimmed
        state.messages.append(ChatMessage(id: nextId("m"), role: .user, content: userText))
        state.messages.append(ChatMessage(id: nextId("m"), role: .assistant, status: "streaming"))
        state.streaming = true
        state.phase = "Thinking"

        Task {
            // Mint or re-attach a session before the first turn. The engine
            // attaches the newest session itself on boot, so a first send
            // appends to it rather than creating a duplicate.
            if state.activeSessionId == nil {
                if let newest = state.sessions.first {
                    _ = await engine.resumeSession(newest.id)
                    setActiveSession(newest.id)
                } else if let minted = await engine.newSession(),
                          let id = minted.optString("sessionId").nilIfEmpty {
                    state.sessions.insert(makeSessionInfo(id), at: 0)
                    setActiveSession(id)
                }
            }
            engine.submitTurn(trimmed, noTools: state.mode == "chat", images: images)
        }
    }

    func systemRow(_ text: String) {
        state.messages.append(ChatMessage(id: nextId("m"), role: .system, content: text))
    }

    private func handleCommand(_ raw: String) {
        let (cmd, arg) = parseCommand(raw)
        let help = SLASH_COMMANDS.map { $0.signature.padding(toLength: max(20, $0.signature.count), withPad: " ", startingAt: 0) + "— " + $0.desc }
            .joined(separator: "\n")

        Task {
            switch cmd {
            case "help", "commands":
                systemRow(help)

            case "new", "clear":
                newChat()

            case "compact":
                systemRow("Compacting…")
                let r = await engine.compactSession()
                if r?.optBool("ok") == true { systemRow("Compacted.") }
                else { systemRow("Compact failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? "")) }

            case "ctx":
                if !arg.isEmpty {
                    if let n = Int(arg.filter(\.isNumber)), (1000...1_000_000).contains(n) {
                        patchConfig(["ctx_window": n])
                        systemRow("Context window → \(grouped(n)) tokens.")
                    } else {
                        systemRow("Usage: /ctx [tokens] (1000–1000000). Bare /ctx shows live usage.")
                    }
                } else if let c = state.ctx, c.window > 0 {
                    systemRow("Context \(grouped(c.used)) / \(grouped(c.window)) tokens (\(c.pct)%) — \(c.source)")
                } else {
                    systemRow("No context meter yet — send a message first.")
                }

            case "maxout":
                if arg.isEmpty {
                    let v = state.config?.maxOutput ?? 0
                    systemRow(v > 0 ? "Max output: \(grouped(v)) tokens." : "Max output: model default (set with /maxout <tokens>).")
                } else if let n = Int(arg.filter(\.isNumber)), (256...384_000).contains(n) {
                    patchConfig(["max_output": n])
                    systemRow("Max output → \(grouped(n)) tokens.")
                } else {
                    systemRow("Usage: /maxout <tokens> (256–384000).")
                }

            case "model":
                if arg.isEmpty {
                    if let c = state.config {
                        systemRow("\(c.provider.isEmpty ? "(no provider)" : c.provider) · \(c.model.isEmpty ? "(no model)" : c.model)")
                    } else { systemRow("Config not loaded yet.") }
                } else {
                    patchConfig(["model": arg])
                    systemRow("Model → \(arg).")
                }

            case "provider":
                if arg.isEmpty {
                    if let c = state.config {
                        systemRow("Provider: \(c.provider.isEmpty ? "(none)" : c.provider) · model \(c.model.isEmpty ? "(none)" : c.model)")
                    } else { systemRow("Config not loaded yet.") }
                } else {
                    patchConfig(["provider": arg, "active": arg])
                    systemRow("Provider → \(arg). Pick a model in Settings → Models if turns fail.")
                }

            case "effort":
                if arg.isEmpty {
                    systemRow("Effort: \(state.config?.effort ?? "off") (low|medium|high|max|off).")
                } else {
                    let lvl = arg.lowercased()
                    if !["low", "medium", "high", "max", "off"].contains(lvl) {
                        systemRow("Usage: /effort low|medium|high|max|off.")
                    } else {
                        patchConfig(["effort": lvl])
                        systemRow("Effort → \(lvl).")
                    }
                }

            case "brain":
                if arg.isEmpty {
                    systemRow("Brain: \(state.config?.brain != false ? "on" : "off") — memories persist across sessions.")
                } else {
                    let on = arg.range(of: "^(on|1|true|yes)$", options: [.regularExpression, .caseInsensitive]) != nil
                    let off = arg.range(of: "^(off|0|false|no)$", options: [.regularExpression, .caseInsensitive]) != nil
                    if !on && !off { systemRow("Usage: /brain [on|off].") }
                    else {
                        patchConfig(["brain": on])
                        systemRow(on ? "Brain on — memories persist across sessions" : "Brain off.")
                    }
                }

            case "remember":
                if arg.isEmpty { systemRow("Usage: /remember <fact> — pin a fact to the brain.") }
                else {
                    let r = await engine.brainRemember(arg)
                    if r?.optBool("ok") == true { systemRow("Remembered.") }
                    else { systemRow("Remember failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? "")) }
                }

            case "why":
                let r = await engine.brainWhy()
                if r?.optBool("ok") != true {
                    systemRow("why: \(r?.optString("error").nilIfEmpty ?? "unavailable")")
                } else if r?.has("lastRecall") != true && r?.has("recall") != true {
                    systemRow("No recalled memories shaped the last answer.")
                } else {
                    systemRow("Last recall: \(r?.optStringy("lastRecall") ?? r?.optStringy("recall") ?? "—")")
                }

            case "tools":
                let r = await engine.listTools()
                if r?.optBool("ok") != true {
                    systemRow("tools: \(r?.optString("error").nilIfEmpty ?? "unavailable")")
                } else {
                    let b = (r?.optArray("builtin") as? [String])?.joined(separator: ", ") ?? "—"
                    let mcp = r?.optArray("mcp") ?? []
                    systemRow("Built-in: \(b)\nMCP: \(jsonEncode(mcp))")
                }

            case "agents":
                let r = await engine.listAgents()
                if r?.optBool("ok") == true {
                    systemRow("Agents: \(r?.optStringy("agents") ?? "[]")")
                } else { systemRow("agents: \(r?.optString("error").nilIfEmpty ?? "unavailable")") }

            case "rlm":
                if arg.isEmpty { systemRow("RLM: \(state.config?.rlm ?? "off") (on|off|auto).") }
                else {
                    let v = arg.lowercased()
                    if !["on", "off", "auto"].contains(v) { systemRow("Usage: /rlm on|off|auto.") }
                    else { patchConfig(["rlm": v]); systemRow("RLM → \(v).") }
                }

            case "ml":
                let sub = arg.lowercased()
                if arg.isEmpty || sub == "info" {
                    let r = await engine.mlInfo()
                    if r?.optBool("ok") == true { systemRow("ML gates: \(r?.optStringy("info") ?? "{}")") }
                    else { systemRow("ml: \(r?.optString("error").nilIfEmpty ?? "unavailable")") }
                } else if sub == "on" || sub == "off" {
                    patchConfig(["ml": sub == "on"])
                    systemRow("ML gates \(sub).")
                } else {
                    systemRow("Usage: /ml [on|off|info] — learn/adopt/discard/reset stay CLI-only for now.")
                }

            case "sessions":
                let sessions = await fetchSessions()
                systemRow(sessions.isEmpty
                    ? "No sessions in this workspace yet."
                    : sessions.map { "\($0.id) · \($0.model.isEmpty ? "?" : $0.model) · \($0.ended ? "ended" : "live")" }.joined(separator: "\n"))

            case "context":
                let r = await engine.contextDump()
                if r?.optBool("ok") == true {
                    systemRow("Context: \(String((r?.optStringy("state") ?? "{}").prefix(2000)))")
                } else { systemRow("context: \(r?.optString("error").nilIfEmpty ?? "unavailable")") }

            case "cost":
                let st = await engine.chatState()
                let spend = st?.optDouble("sessionSpendUsd") ?? 0
                let tokens = Int(st?.optDouble("sessionTokens") ?? 0)
                systemRow("Session: \(String(format: "%.4f", spend)) USD · \(grouped(tokens)) tokens")

            case "resume":
                if arg.isEmpty {
                    let sessions = await fetchSessions()
                    systemRow(sessions.isEmpty
                        ? "No sessions in this workspace yet."
                        : sessions.prefix(10).map { "\($0.id) · \($0.model.isEmpty ? "?" : $0.model) · \($0.ended ? "ended" : "live")" }.joined(separator: "\n") + "\n/resume <id> to open one")
                } else {
                    let r = await engine.resumeSession(arg)
                    if r?.optBool("ok") == true { systemRow("Resumed \(arg).") }
                    else { systemRow("Resume failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? "")) }
                }

            case "permissions":
                if !arg.isEmpty {
                    let r = await engine.setPermissions(arg)
                    if r?.optBool("ok") == true { systemRow("Permissions → \(r?.optString("permissions") ?? "")") }
                    else { systemRow("Failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? "")) }
                    setPermissions(arg)
                } else {
                    systemRow("Permissions: \(state.permissions) (full | edit | plan | prompt)")
                }

            // CLI-only surfaces: honest pointer instead of "unknown".
            case "watch", "hooks", "ghost", "serve", "outputs", "verify", "share",
                 "import", "work", "done", "note", "notify", "sync", "at":
                systemRow("/\(cmd) is CLI-only in this version — use the terminal chat for it. Parity tracked for a later release.")

            default:
                systemRow("Unknown command /\(cmd) — /help lists what exists.")
            }
        }
    }

    // ── Sessions ─────────────────────────────────────────────────────

    private func makeSessionInfo(_ id: String) -> SessionInfo {
        let now = Int(Date().timeIntervalSince1970)
        return SessionInfo(id: id, cwd: state.projectDir ?? "",
                           model: state.config?.model ?? "",
                           provider: state.config?.provider ?? "",
                           startedAt: now, lastSeen: now, task: nil)
    }

    private func onEngineReady() {
        refreshSessions(resumeNewest: true)
        let p = state.permissions
        Task { _ = await engine.setPermissions(p) }
        // Prefetch every provider's model list in the background.
        refreshModelCache()
    }

    private func fetchSessions() async -> [SessionInfo] {
        guard let arr = await engine.sessions() else { return [] }
        return SessionInfo.fromJsonArray(arr)
    }

    func refreshSessions(resumeNewest: Bool = false) {
        Task {
            let fetched = await fetchSessions()
            guard !fetched.isEmpty || state.sessions.isEmpty else { return }
            // Merge, never clobber: the fetched list may be stale.
            var byId: [String: SessionInfo] = [:]
            var order: [String] = []
            for s in fetched { if byId[s.id] == nil { order.append(s.id) }; byId[s.id] = s }
            for s in state.sessions where byId[s.id] == nil { order.append(s.id); byId[s.id] = s }
            let list = byId.values.sorted { $0.lastSeen > $1.lastSeen }

            if state.streaming {
                state.sessions = list
                return
            }
            let nextActive: String?
            if let cur = state.activeSessionId, list.contains(where: { $0.id == cur }) { nextActive = cur }
            else { nextActive = list.first?.id }

            if nextActive == state.activeSessionId {
                state.sessions = list
            } else {
                if let cur = state.activeSessionId {
                    state.messageCache[cur] = state.messages
                    if let m = state.ctx { state.ctxCache[cur] = m }
                }
                state.sessions = list
                state.activeSessionId = nextActive
                state.messages = nextActive.flatMap { state.messageCache[$0] } ?? []
                state.approvals = []
                state.preview = nil
                state.ctx = nextActive.flatMap { state.ctxCache[$0] }
            }
            // Resume ONLY on the boot pass: the engine attached the newest
            // session itself, so a first send appends to it instead of minting
            // a duplicate. Re-resuming on every refresh would hijack whatever
            // session the user is viewing.
            if resumeNewest, let first = fetched.first {
                _ = await engine.resumeSession(first.id)
            }
        }
    }

    func setActiveSession(_ id: String?) {
        guard id != state.activeSessionId else { return }
        if state.streaming {
            if let cur = state.activeSessionId, let m = state.ctx { state.ctxCache[cur] = m }
            state.activeSessionId = id
            state.ctx = id.flatMap { state.ctxCache[$0] }
            return
        }
        if let cur = state.activeSessionId {
            state.messageCache[cur] = state.messages
            if let m = state.ctx { state.ctxCache[cur] = m }
        }
        state.activeSessionId = id
        state.messages = id.flatMap { state.messageCache[$0] } ?? []
        state.approvals = []
        state.preview = nil
        state.ctx = id.flatMap { state.ctxCache[$0] }
        if let id { Task { _ = await engine.resumeSession(id) } }
    }

    func newChat() {
        if state.sessions.count >= MAX_SESSIONS {
            systemRow("Chat limit reached (\(MAX_SESSIONS) per workspace) — delete a chat (the × on a pill) to add a new one.")
            return
        }
        // Park the outgoing page and clear the active id FIRST (the desktop's
        // new_chat dispatch) so the mint that lands afterwards activates
        // cleanly instead of re-parking an empty page.
        if let cur = state.activeSessionId {
            state.messageCache[cur] = state.messages
            if let m = state.ctx { state.ctxCache[cur] = m }
        }
        state.activeSessionId = nil
        state.messages = []
        state.approvals = []
        state.preview = nil
        state.ctx = nil

        Task {
            guard let res = await engine.newSession(),
                  let sid = res.optString("sessionId").nilIfEmpty else { return }
            state.sessions.insert(makeSessionInfo(sid), at: 0)
            setActiveSession(sid)
        }
    }

    func deleteSession(_ id: String) {
        Task {
            guard let r = await engine.deleteSession(id), r.optBool("ok") else { return }
            // Remove the session's files (project store) — the engine dropped
            // the registry entry, the bytes go with it.
            let base = URL(fileURLWithPath: engine.projectDir).appendingPathComponent(".sofuu/sessions")
            try? FileManager.default.removeItem(at: base.appendingPathComponent(id))
            try? FileManager.default.removeItem(at: base.appendingPathComponent("\(id).qtsq"))
            applyDeleteSession(id)
        }
    }

    private func applyDeleteSession(_ id: String) {
        let wasActive = state.activeSessionId == id
        state.messageCache.removeValue(forKey: id)
        state.ctxCache.removeValue(forKey: id)
        state.sessions.removeAll { $0.id == id }
        if wasActive {
            state.activeSessionId = nil
            state.messages = []
            state.preview = nil
            state.ctx = nil
        }
    }

    func clearSessions() {
        Task {
            guard let r = await engine.clearSessions(), r.optBool("ok") else { return }
            let base = URL(fileURLWithPath: engine.projectDir).appendingPathComponent(".sofuu/sessions")
            for sid in (r.optArray("ids") as? [String]) ?? [] {
                // Ids are engine-generated; never let a registry entry turn
                // into a traversal.
                if sid.isEmpty || sid.contains("/") || sid.contains("\\") || sid.hasPrefix(".") { continue }
                try? FileManager.default.removeItem(at: base.appendingPathComponent(sid))
                try? FileManager.default.removeItem(at: base.appendingPathComponent("\(sid).qtsq"))
            }
            for s in state.sessions {
                state.messageCache.removeValue(forKey: s.id)
                state.ctxCache.removeValue(forKey: s.id)
            }
            state.sessions = []
            state.activeSessionId = nil
            state.messages = []
            state.approvals = []
            state.preview = nil
            state.ctx = nil
        }
    }

    /// Load a past session's transcript for read-only viewing.
    func openPreview(_ session: SessionInfo) {
        Task {
            let arr = await engine.sessionTurns(session.id) ?? []
            let turns = arr.compactMap { $0 as? JSONObject }.map {
                PreviewTurn(prompt: $0.optString("prompt"), answer: $0.optString("answer"))
            }
            let idx = state.sessions.firstIndex { $0.id == session.id } ?? 0
            state.preview = SessionPreview(id: session.id, label: session.labelOf(max(0, idx)), turns: turns)
        }
    }

    func closePreview() { state.preview = nil }

    // ── Workspaces ───────────────────────────────────────────────────

    func openWorkspace(_ path: String) {
        // Folder change: the strip, pills and transcripts belong to the OLD
        // workspace — reset them so another folder's sessions can never merge
        // into this view.
        if let cur = state.activeSessionId, let m = state.ctx { state.ctxCache[cur] = m }
        configs.setProjectDir(path)
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        state.projectDir = path
        state.sessions = []
        state.activeSessionId = nil
        state.messages = []
        state.approvals = []
        state.preview = nil
        state.ctx = nil
        state.drawerOpen = false
        Task { _ = await engine.setProject(path) }
        refreshSessions()
    }

    func addWorkspace(_ path: String) {
        let list = [path] + state.workspaces.filter { $0 != path }
        configs.setWorkspaces(list)
        state.workspaces = list
        openWorkspace(path)
    }

    func removeWorkspace(_ path: String) {
        let list = state.workspaces.filter { $0 != path }
        configs.setWorkspaces(list)
        state.workspaces = list
    }

    // ── Config ───────────────────────────────────────────────────────

    func patchConfig(_ patch: JSONObject) {
        let merged = configs.updateConfig(patch)
        state.config = configs.redacted(merged)
        // Re-apply model-specific settings the UI can change live.
        refreshModelCache()
    }

    func setProviderKey(_ provider: String, _ key: String) {
        configs.setKey(provider, key)
        if key.isEmpty { configs.deleteKey(provider) }
        // Reflect into the config file so chat.js's per-turn read sees it.
        guard let providers = state.config?.providers else { return }
        let updated = providers.map { p -> JSONObject in
            var j = p.toJson()
            if p.name == provider { j["api_key"] = key }
            return j
        }
        patchConfig(["providers": updated])
        refreshModelCache()
    }

    @discardableResult
    func removeProvider(_ name: String) -> String? {
        do {
            let merged = try configs.removeProvider(name)
            state.config = configs.redacted(merged)
            refreshModelCache()
            return nil
        } catch {
            return (error as? SofuuError)?.message ?? "\(error)"
        }
    }

    /// Choose a model: writes it into the provider entry AND the flat mirror
    /// (the desktop `pickModel` semantics).
    func pickModel(_ provider: String, _ model: String) {
        guard let providers = state.config?.providers else { return }
        let updated = providers.map { p -> JSONObject in
            var j = p.toJson()
            if p.name == provider { j["model"] = model }
            return j
        }
        patchConfig(["providers": updated, "active": provider, "provider": provider, "model": model])
        refreshModelCache()
    }

    // ── Local GGUF models ─────────────────────────────────────────────

    var localModelURLs: [URL] { LocalModels.list() }

    /// Point the runtime at an on-device GGUF.
    ///
    /// This is the whole of the local-inference switch, and it mirrors the
    /// CLI's `/model x.gguf` branch (`crates/sofuu-core/src/chat.rs:987`).
    /// A `.gguf` path forces `provider="local"` and clears `base_url`, so
    /// the engine routes correctly: `agent.js` calls `sofuu.ai.stream`,
    /// and `rt/ai.rs:4228` sends it to the in-process engine when the
    /// provider is Local or the model path ends `.gguf`. Leaving the old
    /// endpoint in place would send a filesystem path to an HTTP API.
    ///
    /// The harvested caps are applied only when the engine reports real
    /// numbers — guessing a window for a model we have not opened is how a
    /// turn ends in "maximum context length exceeded".
    func pickLocalModel(_ path: String) async {
        var patch: JSONObject = [
            "provider": "local",
            "active": "local",
            "base_url": "",
            "model": path,
        ]
        if let caps = await engine.localModelCaps(path),
           let ok = caps["ok"] as? Bool, ok,
           let c = caps["caps"] as? JSONObject {
            let win = (c["ctx_window"] as? NSNumber)?.int64Value ?? 0
            let maxOut = (c["max_output"] as? NSNumber)?.int64Value ?? 0
            if win > 0 { patch["ctx_window"] = win }
            if maxOut > 0 { patch["max_output"] = maxOut }
            // A model with a thinking channel wants effort on; one without
            // it must not be sent "high", which some kernels reject.
            let thinking = (c["thinking"] as? Bool) ?? (c["thinking"] as? NSNumber)?.boolValue ?? false
            patch["effort"] = thinking ? "high" : ""
        }
        do {
            let merged = try configs.updateConfig(patch)
            state.config = configs.redacted(merged)
            // No provider entry to write: "local" is not a configured
            // endpoint, so refreshModelCache would only fail on it.
            state.localModels = LocalModels.list().map(\.path)
        } catch {
            systemRow("Could not switch to that model: \((error as? SofuuError)?.message ?? "\(error)")")
        }
    }

    /// Import a picked model and select it, reporting the outcome as a
    /// system row so a bad pick is visible rather than silent.
    func importLocalModel(_ url: URL) async {
        guard let file = LocalModels.importModel(from: url) else {
            systemRow("Could not import that file — it needs to be a readable .gguf model.")
            return
        }
        let sizeMB = Double(LocalModels.size(file) ?? 0) / 1_048_576.0
        systemRow(String(format: "Imported %@ (%.0f MB) — checking it…", LocalModels.label(file), sizeMB))
        if let probe = await engine.probeLocalModel(file.path),
           let ok = probe["ok"] as? Bool, !ok {
            let why = probe["error"] as? String ?? "the engine timed out"
            systemRow("\(LocalModels.label(file)) did not load: \(why). The file is still in Settings → Models.")
            return
        }
        await pickLocalModel(file.path)
        systemRow("Model → \(LocalModels.label(file)) · running on this device, no network.")
    }

    func deleteLocalModel(_ path: String) {
        if LocalModels.delete(URL(fileURLWithPath: path)) {
            state.localModels = LocalModels.list().map(\.path)
        }
    }

    /// The drawer "+": add the default app-internal workspace (a real folder
    /// picker can replace it later via the settings pane).
    func addDefaultWorkspace() {
        let dir = AppPaths.defaultWorkspace.path
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addWorkspace(dir)
    }

    // ── Model cache (prefetched, never on click) ─────────────────────

    func refreshModelCache() {
        Task {
            for p in state.config?.providers ?? [] {
                let key = configs.keyFor(p.name) ?? ""
                let r = await engine.fetchProviderModels(p.name, endpoint: p.endpoint, profile: p.profile, key: key)
                let entry: ModelCacheEntry
                if let r {
                    let models = (r.optArray("models") as? [String]) ?? []
                    entry = ModelCacheEntry(models: models, ok: r.optBool("ok"),
                                            error: r.optString("error").nilIfEmpty,
                                            note: r.optString("note").nilIfEmpty,
                                            cached: true, at: now())
                } else {
                    entry = ModelCacheEntry(models: [], ok: false, error: "no cache yet",
                                            note: nil, cached: false, at: now())
                }
                state.modelsCache[p.name] = entry
            }
        }
    }

    private func now() -> Int { Int(Date().timeIntervalSince1970) }

    // ── Approvals / cancel ───────────────────────────────────────────

    func approve(_ id: String, allow: Bool, always: Bool = false) {
        engine.resolveApproval(id, allow: allow, always: always)
        state.approvals.removeAll { $0.id == id }
    }

    func cancel() { engine.cancelTurn() }

    // ── UI toggles ───────────────────────────────────────────────────

    func setPermissions(_ p: String) {
        let v = ["full", "edit", "plan", "prompt"].contains(p) ? p : "prompt"
        uiPrefs.permissions = v
        state.permissions = v
        Task { _ = await engine.setPermissions(v) }
    }

    func setMode(_ m: String) {
        uiPrefs.mode = m
        state.mode = m
    }

    func setTheme(_ t: String) {
        uiPrefs.theme = t
        state.theme = t
    }

    /// Images chosen with the system photo picker, riding the next turn.
    func consumePickedImages(_ urls: [String]) {
        state.pickedImages = Array((state.pickedImages + urls).suffix(8))
    }

    func removePickedImage(_ index: Int) {
        guard state.pickedImages.indices.contains(index) else { return }
        state.pickedImages.remove(at: index)
    }

    func clearPickedImages() { state.pickedImages = [] }

    /// Files chosen with the system picker: images ride the turn as
    /// attachments, everything else becomes an @mention of the copy the
    /// picker dropped into the workspace (chat.js expands @paths).
    func attachPicked(images: [String], mentions: [String]) {
        if !images.isEmpty {
            state.pickedImages = Array((state.pickedImages + images).suffix(8))
        }
        if !mentions.isEmpty {
            let text = mentions.map { "@\($0) " }.joined()
            state.pendingMentions += text
        }
    }

    func consumeMentions() { state.pendingMentions = "" }

    func setCloudBg(_ on: Bool) {
        uiPrefs.cloudBg = on
        state.cloudBg = on
    }

    func setChatFontSize(_ s: Int) {
        uiPrefs.chatFontSize = s
        state.chatFontSize = uiPrefs.chatFontSize
    }

    func openSettings(_ open: Bool, tab: String = "general") {
        state.settingsOpen = open
        state.settingsTab = tab
    }

    func setDrawer(_ open: Bool) { state.drawerOpen = open }

    func openSheet(_ which: String) { state.sheet = which }

    func closeSheet() { state.sheet = nil }

    /// Dismiss the top-most overlay, true if it consumed the action
    /// (drawer → sheet → preview → settings). The iOS counterpart to the
    /// Android hardware-back handler.
    func closeTopOverlay() -> Bool {
        if state.drawerOpen { state.drawerOpen = false; return true }
        if state.sheet != nil { state.sheet = nil; return true }
        if state.preview != nil { state.preview = nil; return true }
        if state.settingsOpen { state.settingsOpen = false; return true }
        return false
    }

    func resolvedDark(_ systemDark: Bool) -> Bool {
        state.theme == "dark" || (state.theme == "system" && systemDark)
    }
}

// MARK: - Small helpers

private func parseCommand(_ raw: String) -> (String, String) {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let r = trimmed.range(of: "^/(\\w+)(?:\\s+([\\s\\S]*))?$", options: .regularExpression) else {
        return ("help", "")
    }
    let matched = String(trimmed[r])
    let body = String(matched.dropFirst()) // drop "/"
    let parts = body.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
    let cmd = parts.first.map { String($0).lowercased() } ?? "help"
    let arg = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
    return (cmd, arg)
}

private func firstMatch(_ pattern: String, _ text: String) -> String? {
    guard let re = try? NSRegularExpression(pattern: pattern),
          let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
          m.numberOfRanges > 1,
          let r = Range(m.range(at: 1), in: text) else { return nil }
    return String(text[r])
}

func grouped(_ n: Int) -> String {
    let f = NumberFormatter()
    f.numberStyle = .decimal
    return f.string(from: NSNumber(value: n)) ?? "\(n)"
}
