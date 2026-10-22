import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// The app state holder: folds the engine event stream into UI state and
/// drives every engine call. The mac twin of the iOS `AppStore` (itself a
/// port of the desktop's `store.ts`) — same reducer semantics, same slash
/// surface, same session discipline, plus the desktop-only preferences
/// (zoom, sidebar) and the Tauri app's background side effects (dock
/// bounce on approval, notify-on-finish only when unfocused).
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

    /// Posted when the app is unfocused and a turn or approval matters.
    var notifier: ((String, String) -> Void)?

    /// True while a Sofuu window has keyboard focus (wired from
    /// NSApplication activation notifications, not scenePhase — the iOS
    /// signal means nothing on the desktop).
    var foreground = true

    // ── TUI-style streaming state (chat.rs parity) ───────────────────
    /// Spinner frames — chat.js SPINNER, verbatim.
    private let spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    /// Unrevealed answer text: deltas land here and a 60 fps timer
    /// releases characters letter-by-letter, so a burst chunk from the
    /// provider still reads as smooth typing.
    private var streamBuffer = ""
    private var streamMessageId: String?
    private var typeTimer: Timer?
    /// Set when `turnFinished` arrives while typing is still draining —
    /// finalize the message (usage/status) the moment the buffer empties.
    private var deferredFinish: (error: String?, usage: Usage?)?
    /// The 120 ms animation clock: spinner + phase-sweep cursor
    /// (chat.js tick2/tickPhase at 120ms, verbatim).
    private var animTimer: Timer?

    private var seq = 0
    private func nextId(_ prefix: String) -> String {
        seq += 1
        return "\(prefix)-\(String(Int(Date().timeIntervalSince1970 * 1000), radix: 36))-\(seq)"
    }

    init() {
        let savedProject = configs.projectDir() ?? MacPaths.defaultWorkspace
        try? FileManager.default.createDirectory(atPath: savedProject, withIntermediateDirectories: true)

        engine = SofuuEngine(homeDir: MacPaths.home, projectDir: savedProject)

        // First launch: the sidebar starts from the active workspace (the
        // Tauri app kept this list in WebView localStorage, which does not
        // migrate — re-add the current project so the sidebar is never
        // empty while sessions exist).
        var workspaceList = configs.workspaces()
        if workspaceList.isEmpty && !savedProject.isEmpty {
            workspaceList = [savedProject]
            configs.setWorkspaces(workspaceList)
        }

        state = AppState(
            projectDir: configs.projectDir(),
            workspaces: workspaceList,
            permissions: uiPrefs.permissions,
            chatFontSize: uiPrefs.chatFontSize,
            mode: uiPrefs.mode,
            theme: uiPrefs.theme,
            cloudBg: uiPrefs.cloudBg)
        state.zoom = uiPrefs.zoom
        state.sidebarCollapsed = uiPrefs.sidebarCollapsed
        state.ideMode = uiPrefs.ideMode
        state.ideTreeWidth = CGFloat(uiPrefs.ideTreeWidth)
        state.ideAgentWidth = CGFloat(uiPrefs.ideAgentWidth)
        state.sessionNames = uiPrefs.sessionNames

        // Engine events → state.
        engine.observe { [weak self] ev in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.handleEvent(ev)
                if case .turnFinished = ev { self.refreshSessions() }
            }
        }

        // Load config + state immediately (cheap, no engine). Keys are
        // folded from the Keychain before the runtime boots, mirroring the
        // Tauri app's step 5a.
        state.config = configs.redacted(configs.readConfigWithKeys())
        updateVoiceGate()
        loadEngineVersion()
        // Engine environment knobs MUST be in place before the runtime is
        // created: SOFUU_NO_ML gates module registration inside the
        // library, so setting it afterwards silently does nothing.
        let appliedEnv = runtimeEnv.apply()
        NSLog("[sofuu-engine] env applied: \(appliedEnv)")
        engine.start()
        state.localModels = LocalModels.list().map(\.path)

        restoreIdeStateIfNeeded()
        restartWorkspaceWatcher()
        refreshGitStatus()
        NotificationCenter.default.addObserver(
            forName: Notification.Name("IDEReloadPath"), object: nil, queue: .main
        ) { [weak self] note in
            guard let self, let path = note.object as? String,
                  self.state.openTabs.contains(path), !self.isDirty(path) else { return }
            self.reloadIfChanged(path)
        }
    }

    func shutdown() { engine.shutdown() }

    /// Re-write the process environment from the current settings.
    func applyRuntimeEnv() { runtimeEnv.apply() }

    /// Tear the engine down and boot it again so a registration-gated knob
    /// takes effect. Only SOFUU_NO_ML needs this.
    func restartEngineForEnv() {
        if state.streaming {
            // Freeing the runtime mid-turn would strand the open tool
            // round-trip, so poke the agent's abort path first.
            engine.cancelTurn()
        }
        engine.shutdown()
        runtimeEnv.apply()
        // The old runtime can emit no further `done`, so nothing else would
        // ever clear this — the composer would stay on the stop button for
        // the rest of the session.
        clearStream()
        stopAnim()
        state.streaming = false
        state.phase = nil
        state.liveThink = ""
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
            if !streamBuffer.isEmpty && error == nil {
                // Typewriter still draining — finish when the buffer empties.
                deferredFinish = (error, usage)
                return
            }
            finalizeTurn(error: error, usage: usage)
            if !foreground && error == nil { notifier?("Sofuu finished", "The turn completed.") }
        case .ctx(let used, let window, let source, let model, let sid, let clampedConfig, let cacheHit, let sections):
            guard let owner = sid ?? state.activeSessionId else { return }
            var meter = CtxMeter(used: used, window: window, source: source, model: model, sid: sid)
            meter.clampedConfig = clampedConfig
            meter.cacheHit = cacheHit
            meter.sections = sections
            state.ctxCache[owner] = meter
            if owner == state.activeSessionId { state.ctx = meter }
        case .approvalRequested(let id, let tool, let args, let risky):
            state.approvals.append(Approval(id: id, tool: tool, args: args, risky: risky))
            if !foreground { notifier?("Sofuu needs your approval", "\(tool) is waiting for approval") }
            // The Tauri app's dock bounce (host.rs:149-163) — CRITICAL
            // attention no matter where the window is, because a queued
            // approval stalls the whole turn.
            Notifications.requestUserAttention()
        case .approvalResolved(let id):
            state.approvals.removeAll { $0.id == id }
        case .warn(let message):
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

    // ── Typewriter + animation plumbing ──────────────────────────────

    /// Advance the 120 ms clock: spinner glyph + phase-sweep cursor.
    private func startAnim() {
        guard animTimer == nil else { return }
        state.streamFrame = 0
        animTimer = Timer.scheduledTimer(withTimeInterval: 0.12, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.state.streaming else { return }
                self.state.streamFrame += 1
            }
        }
    }

    private func stopAnim() {
        animTimer?.invalidate()
        animTimer = nil
        state.liveThink = ""
    }

    /// Release characters from the buffer into the message at typing pace.
    private func startTypeTimer() {
        guard typeTimer == nil else { return }
        typeTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.typeTick() }
        }
    }

    private func stopTypeTimer() {
        typeTimer?.invalidate()
        typeTimer = nil
    }

    private func typeTick() {
        if streamBuffer.isEmpty {
            if let f = deferredFinish {
                clearStream()
                finalizeTurn(error: f.error, usage: f.usage)
                if f.error == nil && !foreground { notifier?("Sofuu finished", "The turn completed.") }
            }
            stopTypeTimer()
            return
        }
        // Adaptive: catch up on provider bursts (≥6 chars/frame), else one
        // letter at a time — always reads as typing, never falls behind.
        let n = max(1, streamBuffer.count / 6)
        let reveal = String(streamBuffer.prefix(n))
        streamBuffer.removeFirst(n)
        patchStream { $0.content += reveal }
    }

    /// Drop everything unrevealed: cancel/failure shows what was typed.
    private func clearStream() {
        if !streamBuffer.isEmpty {
            patchStream { $0.content += streamBuffer }
        }
        streamBuffer = ""
        deferredFinish = nil
        streamMessageId = nil
        stopTypeTimer()
    }

    /// Patch the streaming message by id — it may already be parked in a
    /// session's cache after a mid-stream view switch.
    private func patchStream(_ fn: (inout ChatMessage) -> Void) {
        guard let id = streamMessageId else { return }
        if let i = state.messages.firstIndex(where: { $0.id == id }) {
            fn(&state.messages[i])
            return
        }
        if let cid = state.activeSessionId, var parked = state.messageCache[cid],
           let i = parked.firstIndex(where: { $0.id == id }) {
            fn(&parked[i])
            state.messageCache[cid] = parked
        }
    }

    /// The turn is really over (either the done event, or the buffer just
    /// finished draining). Usage/status land, streaming state resets.
    /// Patches by the streaming message's id first — after a mid-stream
    /// view switch the bubble lives in a session cache, not `messages`.
    private func finalizeTurn(error: String?, usage: Usage?) {
        if let id = streamMessageId {
            patchStream { $0.status = error == nil ? "done" : "error" }
            if let error {
                patchStream { $0.error = error }
            } else {
                patchStream { $0.usage = usage }
            }
        } else if error != nil {
            patchAssistant { $0.status = "error"; $0.error = error }
        } else {
            patchAssistant { $0.status = "done"; $0.usage = usage }
        }
        streamBuffer = ""
        streamMessageId = nil
        deferredFinish = nil
        stopTypeTimer()
        stopAnim()
        state.streaming = false
        state.phase = nil
        state.liveThink = ""
        refreshTabsAfterTurn()
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
            state.liveThink = ""  // TUI: the think line is cleared the moment answer text starts
            streamBuffer += text
            if streamMessageId == nil { streamMessageId = currentAssistant?.id }
            startTypeTimer()

        case "think":
            let text = p.optString("text")
            if text.isEmpty {
                state.phase = phase ?? state.phase
                return
            }
            // TUI: `▸ thinking ` + all think chunks joined, live.
            state.liveThink += text
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
        // Boot race: the mailbox would eval against an engine that has not
        // run chat.init yet (~1-2s after launch) and the turn dies with a
        // "submit is not a function" error row.
        guard state.engineReady else {
            systemRow("Engine is still booting — try again in a second.")
            return
        }
        if trimmed.hasPrefix("/") {
            handleCommand(trimmed)
            return
        }
        let userText = (trimmed.isEmpty && !images.isEmpty) ? "[image]" : trimmed
        state.messages.append(ChatMessage(id: nextId("m"), role: .user, content: userText))
        state.messages.append(ChatMessage(id: nextId("m"), role: .assistant, status: "streaming"))
        state.streaming = true
        state.phase = "Thinking"
        startAnim()

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
                    // Validate BEFORE touching the engine: a bogus profile
                    // must not silently reset the stored state to "prompt".
                    guard ["full", "edit", "plan", "prompt"].contains(arg.lowercased()) else {
                        systemRow("Usage: /permissions full | edit | plan | prompt")
                        return
                    }
                    let r = await engine.setPermissions(arg.lowercased())
                    if r?.optBool("ok") == true {
                        systemRow("Permissions → \(r?.optString("permissions") ?? "")")
                        setPermissions(arg.lowercased())
                    } else {
                        systemRow("Failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? ""))
                    }
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
        // Prefetch every provider's model list in the background (the
        // desktop app re-runs this every 5 minutes — see SofuuMacApp).
        refreshModelCache()
    }

    private func fetchSessions() async -> [SessionInfo] {
        guard let raw = await engine.sessions() else { return [] }
        // The engine's chat.sessions() wraps in {sessions:[…]} (newer
        // chat.js); older builds (and the desktop Tauri command) returned a
        // bare array — normalize BOTH before the .map (the desktop's
        // "malformed shape must never blank the app" rule).
        let arr: [Any]
        if let list = raw as? [Any] {
            arr = list
        } else if let obj = raw as? JSONObject, let list = obj["sessions"] as? [Any] {
            arr = list
        } else {
            arr = []
        }
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
            // session itself, so a first send appends to it instead of
            // minting a duplicate. Re-resuming on every refresh would
            // hijack whatever session the user is viewing.
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
            // Engine dropped its registry entry — the bytes go with it, at
            // the MESH root (git toplevel when the project sits inside a
            // repo, else the folder — MacPaths, desktop sessions.rs rule).
            //Disk work + the git-tooplevel spawn run OFF the main thread.
            let projectDir = engine.projectDir
            await Task.detached(priority: .utility) {
                let base = MacPaths.sessionsDir(projectDir)
                try? FileManager.default.removeItem(at: base.appendingPathComponent(id))
                try? FileManager.default.removeItem(at: base.appendingPathComponent("\(id).qtsq"))
            }.value
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
            let ids = (r.optArray("ids") as? [String]) ?? []
            // Ids are engine-generated; never let a registry entry turn
            // into a traversal. Disk work + git-tooplevel spawn off main.
            let projectDir = engine.projectDir
            await Task.detached(priority: .utility) {
                let base = MacPaths.sessionsDir(projectDir)
                for sid in ids where !sid.isEmpty && !sid.contains("/")
                    && !sid.contains("\\") && !sid.hasPrefix(".") {
                    try? FileManager.default.removeItem(at: base.appendingPathComponent(sid))
                    try? FileManager.default.removeItem(at: base.appendingPathComponent("\(sid).qtsq"))
                }
            }.value
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
            systemRow("Deleted all chats in this workspace.")
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
        // workspace — reset them so another folder's sessions can never
        // merge into this view.
        if let cur = state.activeSessionId, let m = state.ctx { state.ctxCache[cur] = m }
        configs.setProjectDir(path)
        // Language servers are (language, root)-keyed: servers from the old
        // root have nothing left to serve here.
        LSPRegistry.shared.stopAll(exceptRoot: path)
        state.externalEdits = []
        state.gitStatus = [:]
        restartWorkspaceWatcher()
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        state.projectDir = path
        state.sessions = []
        state.activeSessionId = nil
        state.messages = []
        state.approvals = []
        state.preview = nil
        state.ctx = nil
        closeAllTabs()
        Task { _ = await engine.setProject(path) }
        refreshSessions()
        // Make sure the switched-to folder is in the sidebar list too.
        if !state.workspaces.contains(path) {
            let list = [path] + state.workspaces
            configs.setWorkspaces(list)
            state.workspaces = list
        }
        restoreIdeStateIfNeeded()
        refreshGitStatus(force: true)
    }

    func addWorkspace(_ path: String) {
        let list = [path] + state.workspaces.filter { $0 != path }
        configs.setWorkspaces(list)
        state.workspaces = list
        openWorkspace(path)
    }

    /// Sidebar-list removal only — the folder on disk is untouched (desktop
    /// removeWorkspace parity).
    func removeWorkspace(_ path: String) {
        let list = state.workspaces.filter { $0 != path }
        configs.setWorkspaces(list)
        state.workspaces = list
        if state.projectDir == path, let next = list.first {
            openWorkspace(next)
        }
    }

    // ── Config ───────────────────────────────────────────────────────

    func patchConfig(_ patch: JSONObject) {
        let merged = configs.updateConfig(patch)
        state.config = configs.redacted(merged)
        updateVoiceGate()
        // Re-apply model-specific settings the UI can change live.
        refreshModelCache()
    }

    func setProviderKey(_ provider: String, _ key: String) {
        // The 0600 config file IS the key store (the Tauri keychain's
        // per-item ACLs re-prompted the login password on every dev
        // rebuild — the keychain layer is gone by design).
        if key.isEmpty {
            configs.clearKey(provider)
            let merged = configs.readConfig()
            state.config = configs.redacted(merged)
            updateVoiceGate()
        } else {
            configs.setKey(provider, key)
            let merged = configs.readConfig()
            state.config = configs.redacted(merged)
            updateVoiceGate()
        }
        refreshModelCache()
    }

    @discardableResult
    func removeProvider(_ name: String) -> String? {
        do {
            let merged = try configs.removeProvider(name)
            state.config = configs.redacted(merged)
            updateVoiceGate()
            state.modelsCache.removeValue(forKey: name)
            refreshModelCache()
            return nil
        } catch {
            return (error as? SofuuError)?.message ?? "\(error)"
        }
    }

    /// Choose a model: writes it into the provider entry AND the flat
    /// mirror (ConfigStore.updateConfig performs the sync — the desktop
    /// pickModel covenant).
    func pickModel(_ provider: String, _ model: String) {
        guard let providers = state.config?.providers else { return }
        let updated = providers.map { p -> JSONObject in
            var j = p.toJson()
            if p.name == provider { j["model"] = model }
            return j
        }
        patchConfig(["providers": updated, "active": provider, "model": model])
    }

    // ── Local GGUF models ─────────────────────────────────────────────

    var localModelURLs: [URL] { LocalModels.list() }

    /// Point the runtime at an on-device GGUF — the CLI's `/model x.gguf`
    /// branch: a `.gguf` path forces `provider="local"` and clears
    /// `base_url`, so the engine routes to the in-process infer engine.
    /// Harvested caps apply only when the engine reports real numbers.
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
        let merged = configs.updateConfig(patch)
        state.config = configs.redacted(merged)
        state.localModels = LocalModels.list().map(\.path)
        if state.config?.model != path {
            systemRow("Could not switch to that model — the config did not take. Try again, or pick it from Models.")
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
        let probe = await engine.probeLocalModel(file.path)
        guard let probe, let ok = probe["ok"] as? Bool, ok else {
            let why = probe?["error"] as? String ?? "the engine did not answer"
            systemRow("\(LocalModels.label(file)) did not load: \(why). The file is still in Settings → Models.")
            return
        }
        await pickLocalModel(file.path)
        systemRow("Model → \(LocalModels.label(file)) · running locally, no network.")
    }

    func deleteLocalModel(_ path: String) {
        if LocalModels.delete(URL(fileURLWithPath: path)) {
            state.localModels = LocalModels.list().map(\.path)
            if state.config?.model == path {
                systemRow("That model was in use — pick a provider model in Models for the next turn.")
            }
        }
    }

    // ── Model cache (prefetched, never fetched at click time) ────────

    func refreshModelCache() {
        Task {
            for p in state.config?.providers ?? [] {
                // Anthropic has no public model list — the desktop answers
                // with a note and the picker falls back to manual entry.
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

    /// ⌘K — fold old history into a summary (menu + shortcut entry point).
    func compactSession() {
        Task {
            let r = await engine.compactSession()
            if r?.optBool("ok") == true { systemRow("Compacted.") }
            else { systemRow("Compact failed" + (r?.optString("error").nilIfEmpty.map { ": \($0)" } ?? "")) }
        }
    }

    // ── UI toggles ───────────────────────────────────────────────────

    func setPermissions(_ p: String) {
        let v = ["full", "edit", "plan", "prompt"].contains(p) ? p : "prompt"
        uiPrefs.permissions = v
        state.permissions = v
        Task {
            if let r = await engine.setPermissions(v), r.optBool("ok") != true {
                let why = r.optString("error").nilIfEmpty ?? "unknown error"
                systemRow("Applying the permission profile failed: \(why) — the engine keeps the previous profile.")
            }
        }
    }

    func setMode(_ m: String) {
        uiPrefs.mode = m
        state.mode = m
    }

    func setTheme(_ t: String) {
        uiPrefs.theme = t
        state.theme = t
    }

    // ── Desktop-only preferences ─────────────────────────────────────

    func zoomBy(_ delta: Double) {
        uiPrefs.zoom = uiPrefs.zoom + delta
        state.zoom = uiPrefs.zoom
    }

    func zoomReset() {
        uiPrefs.zoom = 1.0
        state.zoom = 1.0
    }

    func toggleSidebar() {
        uiPrefs.sidebarCollapsed.toggle()
        state.sidebarCollapsed = uiPrefs.sidebarCollapsed
    }

    // ── IDE mode ────────────────────────────────────────────────────────

    /// The code gutter's observable state — owned here (stable class) so
    /// it survives tab switches; only the gutter Canvas observes it, so a
    /// 60 Hz scroll never re-renders the shell.
    let ideGutter = GutterModel()
    /// The live editor coordinator — the gutter Canvas reads the text
    /// view's fresh geometry from it at draw time.
    weak var ideCoordinator: CodeEditor.Coordinator?

    /// path → current editor text (live binding for the code viewport).
    var fileBuffers: [String: String] = [:]
    /// path → on-disk text at open/reload time (dirty detection + revert).
    private var diskContents: [String: String] = [:]

    func toggleIdeMode() {
        uiPrefs.ideMode.toggle()
        state.ideMode = uiPrefs.ideMode
    }

    /// Switch into IDE mode holding nothing (open-with a file when no
    /// project is loaded yet — the tab still opens; the tree fills once a
    /// workspace is chosen).
    func enterIdeMode() {
        if state.ideMode { return }
        uiPrefs.ideMode = true
        state.ideMode = true
    }

    // ── Platform intake (Finder/Open-with • sofuu:// • Services) ──────

    /// Everything macOS hands the app from outside routes through here:
    /// file:// URLs (Finder "Open With", dock drops) and sofuu:// deep
    /// links. The app is frontmost by definition for onOpenURL, but
    /// activation is harmless and makes Services hand-offs deterministic.
    func openIncoming(_ url: URL) {
        let activate = { NSApp.activate(ignoringOtherApps: true) }
        if url.isFileURL {
            var isDir: ObjCBool = false
            let path = url.path
            guard !path.isEmpty,
                  FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else { return }
            activate()
            if isDir.boolValue {
                openWorkspace(path)
            } else {
                enterIdeMode()
                openFile(path)
            }
            return
        }
        guard url.scheme?.lowercased() == "sofuu" else { return }
        let host = (url.host ?? "").lowercased()
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        let path = items.first { $0.name == "path" }?.value
        switch host {
        case "workspace":
            guard let path, !path.isEmpty else { return }
            activate()
            openWorkspace(path)
        case "open":
            guard let path, !path.isEmpty else { return }
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else { return }
            activate()
            if isDir.boolValue { openWorkspace(path) }
            else { enterIdeMode(); openFile(path) }
        case "new-chat":
            activate()
            newChat()
        default:
            break
        }
    }

    /// macOS Services "Send to Sofuu": park the text in the draft slot;
    /// whichever Composer is on screen (chat or IDE agent panel) seeds
    /// itself from it and consumes the slot.
    func receiveSharedText(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        state.composerDraft = trimmed
        NSApp.activate(ignoringOtherApps: true)
    }

    func consumeComposerDraft() {
        state.composerDraft = nil
    }

    // ── Onboarding + shortcut cheat sheet ────────────────────────────

    static let onboardingShownKey = "sofuu.onboardingShown"

    func maybeShowOnboarding() {
        guard !UserDefaults.standard.bool(forKey: Self.onboardingShownKey) else { return }
        state.onboardingVisible = true
    }

    func dismissOnboarding() {
        state.onboardingVisible = false
        UserDefaults.standard.set(true, forKey: Self.onboardingShownKey)
    }

    func showShortcutHelp() { state.shortcutHelpVisible = true }

    func dismissShortcutHelp() { state.shortcutHelpVisible = false }

    static let maxCodeFileBytes = 1 << 20  // 1 MB — larger files open read-only-empty with a notice

    /// Open a file into a tab: read from disk, mark clean. Binary content
    /// or oversize files refuse with a system row instead of garbling the
    /// editor.
    /// Files beyond the 1 MB edit cap open view-only up to this size.
    static let maxPreviewBytes = 20 << 20

    func openFile(_ path: String) {
        if openTabsContains(path) { state.activeTab = path; return }
        recordRecent(path)
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else {
            systemRow("Cannot open \((path as NSString).lastPathComponent) in the editor (binary or non-UTF-8).")
            return
        }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        let readOnly = size > Self.maxCodeFileBytes
        guard size <= Self.maxPreviewBytes else {
            systemRow("Too large to open (over 20 MB): \((path as NSString).lastPathComponent)")
            return
        }
        diskContents[path] = text
        fileBuffers[path] = text
        state.openTabs.append(path)
        state.tabDirty[path] = false
        if readOnly {
            state.readOnlyTabs.insert(path)
            systemRow("Opened read-only — larger than the 1 MB edit cap.")
        }
        state.activeTab = path
    }

    func isReadOnly(_ path: String) -> Bool { state.readOnlyTabs.contains(path) }

    /// Most-recently-opened files (MRU, persisted in the IDE state).
    private(set) var recentFiles: [String] = []

    private func recordRecent(_ path: String) {
        recentFiles.removeAll { $0 == path }
        recentFiles.insert(path, at: 0)
        if recentFiles.count > 10 { recentFiles.removeLast(recentFiles.count - 10) }
        persistIdeStateSoon()
    }

    private func openTabsContains(_ path: String) -> Bool { state.openTabs.contains(path) }

    func closeTab(_ path: String, force: Bool = false) {
        if state.tabDirty[path] == true && !force {
            // Two-step close for unsaved work (the sidebar's armed ✕ rule).
            state.tabDirty[path] = nil // sentinel: the ✕ now shows "again?"
            systemRow("Unsaved changes in \(URL(fileURLWithPath: path).lastPathComponent) — click × again to discard.")
            return
        }
        state.openTabs.removeAll { $0 == path }
        state.tabDirty.removeValue(forKey: path)
        fileBuffers.removeValue(forKey: path)
        diskContents.removeValue(forKey: path)
        // Drop any pending auto-save for this path (its buffer is gone).
        autoSaveWork[path]?.cancel()
        autoSaveWork.removeValue(forKey: path)
        // The server needs to drop the document too (no stale diagnostics
        // for a closed file).
        NotificationCenter.default.post(name: .init("IDEDocumentClosed"), object: path)
        if state.activeTab == path {
            state.activeTab = state.openTabs.last
        }
    }

    /// The close button's first tap arms; second tap discards.
    func setActiveTab(_ path: String) {
        state.activeTab = path
        // Re-read from disk if it changed since it was opened (the agent
        // may have edited it in another turn).
        reloadIfChanged(path)
    }

    func updateBuffer(_ path: String, _ text: String) {
        fileBuffers[path] = text
        // Only publish when the dirty FLAG flips — a per-keystroke write
        // would re-render the whole shell on every character.
        let dirty = text != diskContents[path]
        if state.tabDirty[path] != dirty {
            state.tabDirty[path] = dirty
        }
    }

    func buffer(_ path: String) -> String { fileBuffers[path] ?? "" }

    func isDirty(_ path: String) -> Bool { state.tabDirty[path] == true }

    /// Write the buffer back to disk. Returns nil on success, an error
    /// line on failure.
    private func writeThrough(_ path: String) -> String? {
        guard let text = fileBuffers[path] else { return "nothing to save" }
        // A buffer whose file was deleted on disk (by the agent, or a
        // terminal) must NOT resurrect it through an auto-save write.
        guard FileManager.default.fileExists(atPath: path) else {
            return "file is gone from disk — close the tab and reopen it from the tree"
        }
        do {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
            diskContents[path] = text
            state.tabDirty[path] = false
            NotificationCenter.default.post(name: .init("IDEFileContentSaved"), object: path)
            return nil
        } catch {
            return "save failed: \(error.localizedDescription)"
        }
    }

    @discardableResult
    func saveFile(_ path: String) -> String? {
        let err = writeThrough(path)
        if err == nil { scheduleFormat(path) }
        return err
    }

    func scheduleFormat(_ path: String) {
        guard uiPrefs.ideFormatOnSave else { return }
        runFormatter(path, savingFirst: false)
    }

    /// ⌥⇧F / Format on save: run the language's installed formatter on
    /// this file off-main, then reload the tab from disk when it changes.
    func formatFile(_ path: String) {
        guard CodeFormatters.spec(for: Languages.language(forPath: path).id) != nil else {
            systemRow("No formatter configured for \(Languages.language(forPath: path).name).")
            return
        }
        if isReadOnly(path) {
            systemRow("Read-only file — nothing to format.")
            return
        }
        // save-without-hook: format-on-save must not re-fire mid-format.
        if isDirty(path) && writeThrough(path) != nil { return }
        runFormatter(path, savingFirst: false)
    }

    func runFormatter(_ path: String, savingFirst: Bool) {
        let lang = Languages.language(forPath: path)
        guard let specs = CodeFormatters.spec(for: lang.id) else { return }
        if savingFirst, isDirty(path) {
            if let err = saveFile(path) {
                systemRow(err)
                return
            }
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            for (bin, args) in specs {
                guard let binary = CodeFormatters.resolve(bin) else { continue }
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: binary)
                proc.arguments = args.map { $0 == "%PATH%" ? path : $0 }
                proc.standardOutput = FileHandle.nullDevice
                let errPipe = Pipe()
                proc.standardError = errPipe
                do {
                    try proc.run()
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    proc.waitUntilExit()
                    _ = errData
                    guard proc.terminationStatus == 0 else { continue }
                    DispatchQueue.main.async {
                        self?.reloadIfChanged(path)
                        self?.state.externalTick += 1
                        self?.systemRow("Formatted with \(bin).")
                    }
                    return
                } catch { continue }
            }
            DispatchQueue.main.async {
                self?.systemRow("No formatter available for \(lang.name) (checked: \(specs.map(\.bin).joined(separator: ", "))).")
            }
        }
    }

    /// Formatter per language: first binary found on PATH wins. %PATH% in
    /// args is the file.
    enum CodeFormatters {
        private static let cacheLock = NSLock()
        private static var cacheBacking: [String: String?] = [:]

        static var whichCache: [String: String?] {
            get { cacheLock.lock(); defer { cacheLock.unlock() }; return cacheBacking }
            set { cacheLock.lock(); defer { cacheLock.unlock() }; cacheBacking = newValue }
        }

        static func spec(for id: LangID) -> [(bin: String, args: [String])]? {
            let prettier = [("prettier", ["--write", "%PATH%"])]
            switch id {
            case .swift: return [("swift-format", ["format", "--in-place", "%PATH%"])]
            case .rust: return [("rustfmt", ["%PATH%"])]
            case .ts, .json, .css, .html, .markdown, .yaml: return prettier
            case .python: return [("black", ["-q", "%PATH%"]), prettier[0]]
            case .cLike: return [("clang-format", ["-i", "%PATH%"])]
            case .go: return [("gofmt", ["-w", "%PATH%"])]
            default: return nil
            }
        }

        static func resolve(_ bin: String) -> String? {
            if let cached = whichCache[bin] { return cached }
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
            proc.arguments = [bin]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice
            var found: String?
            if (try? proc.run()) != nil {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                let path = String(decoding: data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                found = path.isEmpty ? nil : path
            }
            whichCache[bin] = found
            return found
        }
    }

    func closeAllTabs() {
        // Never silently discard unsaved work: with auto-save ON every
        // dirty buffer gets written; with it OFF the dirty tabs stay open
        // and only the clean ones close.
        let dirtyTabs = state.openTabs.filter { isDirty($0) }
        if !dirtyTabs.isEmpty {
            if ideAutoSaveOn {
                for p in dirtyTabs {
                    if let err = saveFile(p) {
                        systemRow(err)
                        return
                    }
                }
            } else {
                systemRow("Unsaved changes — save (⌘S) or turn auto-save on before closing all tabs.")
                return
            }
        }
        for path in state.openTabs {
            NotificationCenter.default.post(name: .init("IDEDocumentClosed"), object: path)
        }
        state.openTabs = []
        state.activeTab = nil
        state.tabDirty = [:]
        fileBuffers = [:]
        diskContents = [:]
    }

    /// ⌘S — save the active tab; failures surface as a system row.
    func saveActiveTab() {
        guard let path = state.activeTab else { return }
        if let err = saveFile(path) { systemRow(err) }
    }

    // ── IDE file operations (create / rename / delete / auto-save) ──────

    var ideAutoSaveOn: Bool { uiPrefs.ideAutoSave }
    var editorFontFamily: String? { uiPrefs.editorFontFamily }
    var indentGuidesOn: Bool { uiPrefs.indentGuides }
    var showWhitespace: Bool { uiPrefs.showWhitespace }
    var editorFontSize: Int { uiPrefs.editorFontSize }
    var formatOnSave: Bool { uiPrefs.ideFormatOnSave }

    func setIdeAutoSave(_ on: Bool) {
        uiPrefs.ideAutoSave = on
        if !on {
            for (_, work) in autoSaveWork { work.cancel() }
            autoSaveWork = [:]
        }
    }

    private var autoSaveWork: [String: DispatchWorkItem] = [:]
    private var autoSaveErrorNotified = false

    /// Debounced auto-write of the buffer (real-IDE auto save). Silent —
    /// only a failure surfaces, and only once until a save succeeds.
    /// Per-path work items: typing in file B must not cancel file A's
    /// pending save — each path's debounce runs to completion on its own.
    func scheduleAutoSave(_ path: String) {
        guard uiPrefs.ideAutoSave, fileBuffers[path] != nil else { return }
        autoSaveWork[path]?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, let text = self.fileBuffers[path] else { return }
            _ = text // buffer is the source of truth; saveFile reads it
            if let err = self.saveFile(path) {
                if !self.autoSaveErrorNotified {
                    self.autoSaveErrorNotified = true
                    self.systemRow("Auto save failed for \(URL(fileURLWithPath: path).lastPathComponent): \(err)")
                }
            } else {
                self.autoSaveErrorNotified = false
            }
        }
        autoSaveWork[path] = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8, execute: work)
    }

    /// Sanitize an entry name from the tree composer.
    private func validateEntryName(_ raw: String) -> String? {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/") else { return nil }
        return name
    }

    /// Create + open a new empty file in `dir`. Returns an error line.
    @discardableResult
    func createFile(at dir: String, name: String) -> String? {
        guard let clean = validateEntryName(name) else { return "That file name is not usable." }
        let path = dir + "/" + clean
        if FileManager.default.fileExists(atPath: path) {
            return "\(clean) already exists."
        }
        do {
            try Data().write(to: URL(fileURLWithPath: path))
        } catch {
            return "Could not create \(clean): \(error.localizedDescription)"
        }
        openFile(path)
        return nil
    }

    /// Create a new folder in `dir`. Returns an error line.
    @discardableResult
    func createFolder(at dir: String, name: String) -> String? {
        guard let clean = validateEntryName(name) else { return "That folder name is not usable." }
        let path = dir + "/" + clean
        if FileManager.default.fileExists(atPath: path) {
            return "\(clean) already exists."
        }
        do {
            try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: false)
        } catch {
            return "Could not create \(clean): \(error.localizedDescription)"
        }
        return nil
    }

    /// Rename / move a file or folder; open tabs follow if the path (or a
    /// folder prefix) matches. Returns an error line.
    @discardableResult
    func renameFile(_ path: String, newName: String) -> String? {
        guard let clean = validateEntryName(newName) else { return "That name is not usable." }
        let dir = (path as NSString).deletingLastPathComponent
        let target = dir + "/" + clean
        guard target != path else { return nil }
        guard !FileManager.default.fileExists(atPath: target) else {
            return "\(clean) already exists."
        }
        do {
            try FileManager.default.moveItem(atPath: path, toPath: target)
        } catch {
            return "Rename failed: \(error.localizedDescription)"
        }
        remapTabPaths(path, to: target)
        return nil
    }

    /// Trash a file or folder and close matching tabs. Returns an error line.
    @discardableResult
    func deleteFile(_ path: String) -> String? {
        guard path != state.projectDir else { return "That is the workspace folder." }
        let url = URL(fileURLWithPath: path)
        do {
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        } catch {
            // Fallback for volumes without a Trash.
            do { try FileManager.default.removeItem(at: url) }
            catch { return "Delete failed: \(error.localizedDescription)" }
        }
        let prefix = path.hasSuffix("/") ? path : path + "/"
        for open in state.openTabs where open == path || open.hasPrefix(prefix) {
            closeTab(open, force: true)
        }
        return nil
    }

    /// Move open-tab bookkeeping when a file/folder path changes on disk.
    private func remapTabPaths(_ old: String, to new: String) {
        var isDir: ObjCBool = false
        _ = FileManager.default.fileExists(atPath: old, isDirectory: &isDir)
        let folder = isDir.boolValue
        let prefix = old.hasSuffix("/") ? old : old + "/"
        let remap: (String) -> String = { p in
            if p == old { return new }
            if folder && p.hasPrefix(prefix) { return new + p.dropFirst(prefix.count - 1) }
            return p
        }
        state.openTabs = state.openTabs.map(remap)
        if let active = state.activeTab {
            state.activeTab = remap(active)
        }
        for (p, v) in fileBuffers where remap(p) != p {
            fileBuffers[remap(p)] = v
            fileBuffers.removeValue(forKey: p)
        }
        for (p, v) in diskContents where remap(p) != p {
            diskContents[remap(p)] = v
            diskContents.removeValue(forKey: p)
        }
        let dirtyKeys = state.tabDirty.keys
        var newDirty: [String: Bool] = state.tabDirty
        for p in dirtyKeys where remap(p) != p {
            newDirty[remap(p)] = newDirty[p]
            newDirty.removeValue(forKey: p)
        }
        state.tabDirty = newDirty
    }

    /// IDE panel widths (the splitters) — read by IdeLayout from state so
    /// the AppKit drag handles can call the store DIRECTLY (a class ref
    /// that never goes stale, unlike view closures), with the layout
    /// change flowing back through the normal published pipeline. dx is a
    /// PER-EVENT delta, so it accumulates onto the live width — applying
    /// it to a fixed base made every event compute the same value (the
    /// "drag only moves one step" bug).
    enum IdeSplitter { case tree, agent }

    func ideSplitterDrag(_ which: IdeSplitter, dx: CGFloat) {
        switch which {
        case .tree:
            state.ideTreeWidth = min(460, max(170, state.ideTreeWidth + dx))
        case .agent:
            state.ideAgentWidth = min(640, max(280, state.ideAgentWidth - dx))
        }
    }

    func ideSplitterEnd(_ which: IdeSplitter) {
        uiPrefs.ideTreeWidth = Double(state.ideTreeWidth)
        uiPrefs.ideAgentWidth = Double(state.ideAgentWidth)
        _ = which
    }

    private func reloadIfChanged(_ path: String) {
        guard !isDirty(path),
              let text = try? String(contentsOfFile: path, encoding: .utf8),
              text != diskContents[path] else { return }
        diskContents[path] = text
        fileBuffers[path] = text
    }

    /// After an agent turn, quietly refresh clean open tabs whose files
    /// changed on disk (the agent edits files the editor may show).
    func refreshTabsAfterTurn() {
        for path in state.openTabs where !isDirty(path) {
            reloadIfChanged(path)
        }
    }

    /// Images picked from the system panel / paste / drag, riding the next
    /// turn as `data:` URLs (≤8 chips — the desktop chip cap).
    func addPickedImages(_ urls: [String]) {
        state.pickedImages = Array((state.pickedImages + urls).suffix(8))
    }

    func removePickedImage(_ index: Int) {
        guard state.pickedImages.indices.contains(index) else { return }
        state.pickedImages.remove(at: index)
    }

    func clearPickedImages() { state.pickedImages = [] }

    // ── Overlay plumbing ─────────────────────────────────────────────

    func setCloudBg(_ on: Bool) {
        uiPrefs.cloudBg = on
        state.cloudBg = on
    }

    func setEditorFont(size: Int) {
        uiPrefs.editorFontSize = size
        // Live font swap happens in the editor's update pass (fontKey).
        objectWillChange.send()
    }

    func setEditorFontFamily(_ family: String?) {
        uiPrefs.editorFontFamily = family
        objectWillChange.send()
    }

    func setIndentGuides(_ on: Bool) {
        uiPrefs.indentGuides = on
        objectWillChange.send()
    }

    func setFormatOnSave(_ on: Bool) {
        uiPrefs.ideFormatOnSave = on
    }

    func setChatFontSize(_ s: Int) {
        uiPrefs.chatFontSize = s
        state.chatFontSize = uiPrefs.chatFontSize
    }

    func openSettings(_ open: Bool, tab: String = "general") {
        state.settingsOpen = open
        state.settingsTab = tab
    }

    func openSheet(_ which: String) { state.sheet = which }

    func closeSheet() { state.sheet = nil }

    /// Dismiss the top-most overlay, true if it consumed the action (Esc).
    func closeTopOverlay() -> Bool {
        if state.sheet != nil { state.sheet = nil; return true }
        if state.preview != nil { state.preview = nil; return true }
        if state.settingsOpen { state.settingsOpen = false; return true }
        if state.shortcutHelpVisible { state.shortcutHelpVisible = false; return true }
        if state.chatSearchOpen { closeChatSearch(); return true }
        if state.gitPanelOpen { setGitPanelOpen(false); return true }
        // Dismissing the welcome card also retires the first-run gate —
        // Esc here means "got it".
        if state.onboardingVisible { dismissOnboarding(); return true }
        return false
    }

    // ── Chat utilities (rename / search / export / voice) ─────────────

    /// Pill/history display title: rename override wins, then the
    /// engine's task line, then "Chat N" by creation order.
    func sessionTitle(_ s: SessionInfo, index: Int) -> String {
        if let n = state.sessionNames[s.id], !n.isEmpty { return n }
        return s.labelOf(index)
    }

    /// Display-name override only — the engine's session record (task
    /// line, mesh id) is intentionally untouched. Empty clears the
    /// override, back to the engine title.
    func renameSession(_ id: String, to raw: String) {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        var map = uiPrefs.sessionNames
        if name.isEmpty { map.removeValue(forKey: id) } else { map[id] = name }
        uiPrefs.sessionNames = map
        state.sessionNames = map
    }

    func setChatSearchOpen(_ open: Bool) {
        state.chatSearchOpen = open
        if !open { state.chatSearchTarget = nil }
    }

    func closeChatSearch() { setChatSearchOpen(false) }

    /// Scroll the transcript to a message row. The tick changes with the
    /// id so re-picking the SAME row still fires ChatScreen's onChange.
    func jumpToMessage(_ id: String) {
        state.chatSearchTarget = id
        state.chatSearchTick += 1
    }

    /// Write the live transcript to a .md file the user picks. Refuses
    /// mid-turn — a streaming message's frame churn would produce a
    /// point-in-time file that contradicts what's on screen.
    func exportSessionMarkdown() {
        guard !state.streaming else {
            systemRow("Stop the running turn before exporting.")
            return
        }
        let messages = state.messages
        guard !messages.isEmpty else { return }

        let title = sessionTitle(
            state.sessions.first { $0.id == state.activeSessionId }
                ?? SessionInfo(id: "active"),
            index: state.sessions.firstIndex { $0.id == state.activeSessionId } ?? 0)
        let md = markdownTranscript(messages: messages, title: title)

        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [UTType.plainText]
        let stamp = Self.exportDate.string(from: Date())
        let safe = title.map { ch in
            ch.isLetter || ch.isNumber ? ch : "-"
        }
        panel.nameFieldStringValue = "sofuu-\(String(safe).lowercased())-\(stamp).md"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try md.write(to: url, atomically: true, encoding: .utf8)
                systemRow("Chat exported → \(url.path)")
            } catch {
                systemRow("Export failed: \(error.localizedDescription)")
            }
        }
    }

    private static let exportDate: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmm"
        return f
    }()

    private func markdownTranscript(messages: [ChatMessage], title: String) -> String {
        var out = "# \(title)\n\n"
        out += "- Session: `\(state.activeSessionId ?? "active")`\n"
        out += "- Exported: \(DateFormatter.localizedString(from: Date(), dateStyle: .medium, timeStyle: .short))\n"
        if let ws = state.projectDir { out += "- Workspace: \(ws)\n" }
        if let cfg = state.config, !cfg.model.isEmpty { out += "- Model: \(cfg.model)\n" }
        out += "\n---\n"
        for m in messages {
            switch m.role {
            case .user:
                out += "\n## You\n\n\(m.content.isEmpty ? "[image]" : m.content)\n"
            case .assistant:
                out += "\n## Sofuu\n"
                if let think = m.thinking, !think.isEmpty {
                    out += "\n<details><summary>thinking</summary>\n\n\(think)\n\n</details>\n"
                }
                let tools = m.timeline.filter { $0.kind == "tool" }
                if !tools.isEmpty {
                    out += "\n"
                    for e in tools {
                        let mark = (e.ok == false) ? "✗" : "✓"
                        out += "- \(mark) `\(e.tool ?? e.title)`"
                        if let d = e.detail, !d.isEmpty { out += " — \(d)" }
                        if let ms = e.durationMs, ms > 0 { out += " (\(String(format: "%.1f", Double(ms) / 1000))s)" }
                        out += "\n"
                    }
                }
                if !m.content.isEmpty { out += "\n\(m.content)\n" }
                if m.status == "done", let u = m.usage, let line = u.line() {
                    out += "\n> \(line)\n"
                }
                if m.status == "error", let err = m.error {
                    out += "\n> ⚠️ \(err)\n"
                }
            case .system:
                out += "\n> ℹ️ \(m.content)\n"
            }
        }
        return out
    }

    // ── Voice input (engine-gated) ─────────────────────────────────────

    /// The full gate shown to the Composer: engine surface (probed once,
    /// retried whenever the config changes until it answers) AND the
    /// active provider shape. The engine rejects Local/Anthropic/no-model
    /// calls itself — this mirror only decides whether the mic first
    /// appears, so a wrong provider never shows a button that must fail.
    func voiceAvailable() -> Bool {
        state.voiceSurfaceOK && voiceEligibleFromConfig()
    }

    private func voiceEligibleFromConfig() -> Bool {
        guard let cfg = state.config else { return false }
        let wanted = cfg.active.isEmpty ? cfg.provider : cfg.active
        let p = cfg.providers.first { $0.name == wanted }
            ?? cfg.providers.first { $0.name == cfg.provider }
        guard let p, !p.endpoint.isEmpty else { return false }
        if SofuuEngine.isLocal(p.endpoint, p.profile) { return false }
        if SofuuEngine.isAnthropic(p.endpoint, p.profile) { return false }
        return !(p.model.isEmpty && cfg.model.isEmpty)
    }

    /// Re-run whenever the redacted config lands (boot + every patch).
    /// The surface probe stays a no-op once true; if the engine boots
    /// voice later than the first config load, later patches heal it.
    func updateVoiceGate() {
        guard voiceEligibleFromConfig() else {
            state.voiceSurfaceOK = false
            return
        }
        guard !state.voiceSurfaceOK else { return }
        Task { [weak self] in
            let ok = await self?.engine.voiceSurfaceProbe() ?? false
            self?.state.voiceSurfaceOK = ok
        }
    }

    /// Engine transcription for the composer's mic. `onDone(text, error)`
    /// — exactly one is non-nil — lands on main.
    func transcribeVoice(_ audioData: Data,
                         onDone: @escaping (String?, String?) -> Void) {
        Task { [weak self] in
            let b64 = audioData.base64EncodedString()
            guard let self else {
                onDone(nil, "Engine went away mid-transcription")
                return
            }
            guard let r = await self.engine.voiceTranscribe(b64) else {
                self.announceTranscriptionFailure(onDone: onDone)
                return
            }
            if r.optBool("ok") == true {
                let text = r.optString("text")
                DispatchQueue.main.async {
                    onDone(text.nilIfEmpty, text.nilIfEmpty == nil ? "Empty transcript" : nil)
                }
            } else {
                DispatchQueue.main.async {
                    onDone(nil, r.optString("error").nilIfEmpty ?? "Transcription failed")
                }
            }
        }
    }

    private func announceTranscriptionFailure(onDone: @escaping (String?, String?) -> Void) {
        DispatchQueue.main.async { onDone(nil, "Transcription failed") }
    }

    func resolvedDark(_ systemDark: Bool) -> Bool {
        state.theme == "dark" || (state.theme == "system" && systemDark)
    }

    // ── Workspace file watching (external changes) ───────────────────

    private var fileWatcher: FileWatcher?
    private var externalChangeWork: DispatchWorkItem?
    private var pendingExternalPaths: Set<String> = []

    func restartWorkspaceWatcher() {
        fileWatcher?.stop()
        fileWatcher = nil
        guard let root = state.projectDir else { return }
        fileWatcher = FileWatcher(root: root) { [weak self] paths in
            self?.coalesceExternalChanges(paths)
        }
    }

    private func coalesceExternalChanges(_ paths: [String]) {
        for p in paths { pendingExternalPaths.insert(p) }
        externalChangeWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            let paths = self.pendingExternalPaths
            self.pendingExternalPaths = []
            self.applyExternalChanges(paths: paths)
        }
        externalChangeWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4, execute: work)
    }

    private func applyExternalChanges(paths: Set<String>) {
        guard !paths.isEmpty else { return }
        // The tree re-lists from disk on every render — a published nudge
        // is all the tree needs to follow the disk.
        state.externalTick += 1
        refreshGitStatus()
        var externallyEdited: Set<String> = []
        for path in state.openTabs where paths.contains(path) {
            if isDirty(path) {
                externallyEdited.insert(path)
            } else if let text = try? String(contentsOfFile: path, encoding: .utf8),
                      text != diskContents[path] {
                diskContents[path] = text
                fileBuffers[path] = text
            }
        }
        if !externallyEdited.isEmpty {
            state.externalEdits.formUnion(externallyEdited)
        }
    }

    func reloadExternalEdit(_ path: String) {
        if let text = try? String(contentsOfFile: path, encoding: .utf8) {
            diskContents[path] = text
            fileBuffers[path] = text
            state.tabDirty[path] = false
        }
        state.externalEdits.remove(path)
    }

    func dismissExternalEdit(_ path: String) {
        state.externalEdits.remove(path)
    }

    /// Editor-side nudge: re-list the tree (any state publish does that).
    func nudgeExternal() {
        state.externalTick += 1
    }

    // ── Git working-tree badge (read-only) ───────────────────────────

    private var gitRefreshAt = Date.distantPast

    func refreshGitStatus(force: Bool = false) {
        guard let root = state.projectDir else { return }
        let now = Date()
        guard force || now.timeIntervalSince(gitRefreshAt) > 0.8 else { return }
        gitRefreshAt = now
        GitMonitor.status(root: root) { [weak self] map in
            Task { @MainActor in self?.state.gitStatus = map }
        }
        GitMonitor.detailedStatus(root: root) { [weak self] files, branch in
            Task { @MainActor in
                self?.state.gitFiles = files
                self?.state.gitBranch = branch
            }
        }
    }

    // ── Git staging / commit panel ─────────────────────────────────────

    func setGitPanelOpen(_ open: Bool) {
        state.gitPanelOpen = open
        if open { refreshGitStatus(force: true) }
    }

    private func runGitAction(_ args: [String], failurePrefix: String) {
        guard let root = state.projectDir else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let (ok, output) = GitMonitor.mutate(root, args)
            DispatchQueue.main.async {
                if !ok { self?.systemRow("\(failurePrefix): \(output)") }
                self?.refreshGitStatus(force: true)
            }
        }
    }

    /// Stage relative paths. Untracked files included — the panel only
    /// offers whole-file staging, the TUI's `git add` level.
    func gitStage(_ relatives: [String]) {
        runGitAction(["add", "--"] + relatives, failurePrefix: "git add failed")
    }

    func gitUnstage(_ relatives: [String]) {
        runGitAction(["restore", "--staged", "--"] + relatives, failurePrefix: "git restore failed")
    }

    /// Commit what the index holds right now. Refuses an empty message
    /// and an empty index instead of shelling out to fail opaquely.
    func gitCommit(message: String, onDone: @escaping (Bool, String) -> Void) {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { onDone(false, "Write a commit message first"); return }
        let staged = state.gitFiles.values.filter { $0.staged != nil }
        guard !staged.isEmpty else { onDone(false, "Nothing is staged"); return }
        guard let root = state.projectDir else { onDone(false, "No workspace open"); return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let (ok, output) = GitMonitor.mutate(root, ["commit", "-m", text])
            DispatchQueue.main.async {
                if ok {
                    self?.systemRow("Committed to \(self?.state.gitBranch ?? "") — \(staged.count) file(s).")
                }
                self?.refreshGitStatus(force: true)
                onDone(ok, ok ? "" : output)
            }
        }
    }

    // ── Engine watch / version ─────────────────────────────────────────

    /// 10 s watchdog: surface a wedged or exited engine pump instead of
    /// dying silently. Auto-clears when the pump ticks again (or a fresh
    /// restartEngineForEnv() boots a new one).
    func checkPumpHealth() {
        let age = engine.pumpAge
        if age > 8 {
            if state.enginePumpStall == nil {
                let detail = age.isFinite ? "\(Int(age))s idle" : "thread exited"
                state.enginePumpStall = "Engine pump stalled (\(detail)) — turns may not respond. Session ▸ Restart Engine reboots it."
                NSLog("[sofuu-engine] pump stalled: \(detail)")
            }
        } else if state.enginePumpStall != nil {
            state.enginePumpStall = nil
        }
    }

    /// Settings engine-card line: from the bundled dylib's own attributes
    /// (works in Debug AND installed builds); ABI version straight from
    /// the linked bridge.
    func loadEngineVersion() {
        var line = "libsofuu ABI v\(engine.abiVersion)"
        let dylib = Bundle.main.bundleURL.appendingPathComponent("Frameworks/libsofuu.dylib")
        if let values = try? dylib.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]) {
            let mb = Double(values.fileSize ?? 0) / 1_048_576
            if let date = values.contentModificationDate {
                let f = DateFormatter()
                f.dateFormat = "yyyy-MM-dd HH:mm"
                line += String(format: " · %.1f MB · built %@", mb, f.string(from: date))
            } else {
                line += String(format: " · %.1f MB", mb)
            }
        }
        state.engineVersion = line
    }

    /// Working tree vs HEAD in the floating diff panel.
    func showGitDiff(_ path: String) {
        guard let root = state.projectDir else { return }
        let rel = path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path
        let kind = state.gitStatus[path]
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let old = (kind != .untracked) ? GitMonitor.headText(root: root, relativePath: rel) : ""
            let new = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
            DispatchQueue.main.async {
                IDEHUD.showDiff(
                    title: (path as NSString).lastPathComponent + " — working tree vs HEAD",
                    oldText: old, newText: new)
            }
        }
    }

    // ── IDE state persistence (tabs / tree expansion / carets) ───────

    private var ideSaveWork: DispatchWorkItem?
    private(set) var ideExpanded: Set<String> = []
    private var ideCursors: [String: (line: Int, column: Int)] = [:]
    private var ideScrolls: [String: CGFloat] = [:]
    private var ideRestoredFor: String?

    func restoreIdeStateIfNeeded() {
        guard ideRestoredFor != state.projectDir, let root = state.projectDir else { return }
        ideRestoredFor = root
        guard state.openTabs.isEmpty else { return }   // a live session wins
        let ide = configs.readIdeState()
        guard ide.optString("project") == root else { return }
        let tabs = (ide.optArray("tabs") as? [String]) ?? []
        for p in tabs.prefix(12) where FileManager.default.fileExists(atPath: p) {
            openFile(p)
        }
        let wantedActive = ide.optString("active")
        if state.openTabs.contains(wantedActive) {
            state.activeTab = wantedActive
        }
        recentFiles = (ide.optArray("recent") as? [String]) ?? []
        ideExpanded = Set((ide.optArray("expanded") as? [String]) ?? [])
        if let cursors = ide.optObject("cursors") as? [String: String] {
            for (path, pair) in cursors {
                let parts = pair.split(separator: ":").compactMap { Int($0) }
                if parts.count == 2 { ideCursors[path] = (parts[0], parts[1]) }
            }
        }
        if let scrolls = ide.optObject("scrolls") as? [String: Double] {
            ideScrolls = scrolls.mapValues { CGFloat($0) }
        }
    }

    func ideRecordExpanded(_ set: Set<String>) {
        ideExpanded = set
        persistIdeStateSoon()
    }

    func ideCursor(for path: String) -> (line: Int, column: Int)? { ideCursors[path] }
    func ideScrollFraction(for path: String) -> CGFloat? { ideScrolls[path] }

    func ideRecordCursor(_ path: String?, line: Int, column: Int) {
        guard let path else { return }
        if let existing = ideCursors[path], existing == (line: line, column: column) { return }
        ideCursors[path] = (line, column)
        persistIdeStateSoon()
    }

    func ideRecordScroll(_ path: String, fraction: CGFloat) {
        let clamped = max(0, min(1, fraction))
        guard abs((ideScrolls[path] ?? -1) - clamped) > 0.005 else { return }
        ideScrolls[path] = clamped
        persistIdeStateSoon()
    }

    func persistIdeStateSoon() {
        ideSaveWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.persistIdeState() }
        ideSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6, execute: work)
    }

    private func persistIdeState() {
        guard let root = state.projectDir else { return }
        let obj: JSONObject = [
            "project": root,
            "tabs": state.openTabs,
            "active": state.activeTab ?? "",
            "recent": recentFiles,
            "expanded": Array(ideExpanded).sorted(),
            "cursors": ideCursors.mapValues { "\($0.line):\($0.column)" },
            "scrolls": ideScrolls.mapValues { Double($0) },
        ]
        configs.writeIdeState(obj)
    }

    /// Pending scroll target for a just-opened file (search results).
    var pendingScroll: (path: String, line: Int, column: Int)?

    // ── Tree drag & drop: internal moves + Finder ingest ─────────────

    func movePath(_ src: String, toFolder folder: String) {
        let name = (src as NSString).lastPathComponent
        let target = folder + "/" + name
        guard target != src else { return }
        guard !target.hasPrefix(src + "/") else {
            systemRow("Cannot move a folder into itself.")
            return
        }
        guard !FileManager.default.fileExists(atPath: target) else {
            systemRow("\u{201C}\(name)\u{201D} already exists there.")
            return
        }
        do {
            try FileManager.default.moveItem(atPath: src, toPath: target)
        } catch {
            systemRow("Move failed: \(error.localizedDescription)")
            return
        }
        remapTabPaths(src, to: target)
        refreshGitStatus(force: true)
    }

    func ingestExternalFile(_ url: URL, toFolder folder: String) {
        let name = url.lastPathComponent
        var target = folder + "/" + name
        var n = 2
        let stem = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        while FileManager.default.fileExists(atPath: target) {
            target = folder + "/" + (ext.isEmpty ? "\(stem) (\(n))" : "\(stem) (\(n)).\(ext)")
            n += 1
        }
        do {
            try FileManager.default.copyItem(at: url, to: URL(fileURLWithPath: target))
            systemRow("Copied \(name) into the workspace.")
        } catch {
            systemRow("Copy failed: \(error.localizedDescription)")
        }
        refreshGitStatus(force: true)
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
