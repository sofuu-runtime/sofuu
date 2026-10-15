import SwiftUI

private let settingsTabs = ["general", "appearance", "models", "providers", "agents", "brain", "usage", "shortcuts"]

/// Page margin for the settings content (cards run window-width minus this).
private let settingsMargin: CGFloat = 28

// MARK: - Settings design system
//
// The page is ONE centered column. Sections are CARDS (soft bgCard panel
// with an internal header); rows come in exactly two shapes so nothing
// drifts:
//
//   ControlRow — label + description left, control RIGHT, vertically
//                centered (toggles, segmented, pickers)
//   StackRow   — label + description, control directly BELOW (fields that
//                want width — urls, paths, monospace inputs)

struct SettingsScreen: View {
    @ObservedObject var store: AppStore
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c
    @State private var tab: String

    init(store: AppStore, onClose: @escaping () -> Void) {
        self.store = store
        self.onClose = onClose
        _tab = State(initialValue: store.state.settingsTab)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Title-bar band: the traffic lights live in the top 28pt —
            // never let the header crowd them.
            Color.clear.frame(height: 34)

            header
            tabBar

            Divider().overlay(c.separator)

            ScrollView {
                VStack(spacing: 14) {
                    switch tab {
                    case "general": GeneralPane(store: store)
                    case "appearance": AppearancePane(store: store)
                    case "models": ModelsPane(store: store)
                    case "providers": ProvidersPane(store: store)
                    case "agents": AgentsPane(store: store)
                    case "brain": BrainPane(store: store)
                    case "usage": UsagePane(store: store)
                    case "shortcuts": ShortcutsPane()
                    default: EmptyView()
                    }
                }
                // Cards respect the window's full width — page margins only,
                // no centered column.
                .padding(.horizontal, settingsMargin)
                .padding(.vertical, 18)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(c.bgApp.ignoresSafeArea())
    }

    // ── Header + tabs, aligned to the same centered column ───────────

    private var header: some View {
        HStack(spacing: 12) {
            Button(action: onClose) {
                HStack(spacing: 5) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Back")
                        .font(.system(size: 13))
                }
                .foregroundStyle(c.accent)
                .padding(.horizontal, 12).padding(.vertical, 7)
                .background(c.bgCard, in: Capsule())
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
            .help("Back to the chat (Esc)")

            Text("Settings")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(c.text)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, settingsMargin)
        .padding(.bottom, 10)
    }

    private var tabBar: some View {
        HStack(spacing: 4) {
            ForEach(settingsTabs, id: \.self) { t in
                let active = t == tab
                Button { tab = t } label: {
                    Text(t.prefix(1).uppercased() + t.dropFirst())
                        .font(.system(size: 12))
                        .foregroundStyle(active ? c.accentContrast : c.textMuted)
                        .padding(.horizontal, 13).padding(.vertical, 7)
                        .background(active ? c.accent : c.bgCard, in: Capsule())
                }
                .buttonStyle(.plain)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, settingsMargin)
        .padding(.bottom, 12)
    }
}

// MARK: - Cards & rows

/// A section card: soft panel, internal header, consistent padding.
struct SectionCard<Content: View>: View {
    let label: String
    var footer: String?
    @ViewBuilder var content: Content

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(c.textFaint)
            content
            if let footer {
                Text(footer)
                    .font(.system(size: 10))
                    .foregroundStyle(c.textFaint)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(c.bgCard, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Label + description left, control right, vertically centered.
struct ControlRow<Control: View>: View {
    let label: String
    var desc: String? = nil
    @ViewBuilder var control: Control

    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.system(size: 13)).foregroundStyle(c.text)
                if let desc {
                    Text(desc).font(.system(size: 11)).foregroundStyle(c.textFaint)
                }
            }
            Spacer(minLength: 12)
            control
        }
    }
}

/// Label + description with the control directly BELOW (fields that want
/// width — urls, paths, monospace inputs).
struct StackRow<Control: View>: View {
    let label: String
    var desc: String? = nil
    var controlWidth: CGFloat? = 220
    @ViewBuilder var control: Control

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.system(size: 13)).foregroundStyle(c.text)
            if let desc {
                Text(desc).font(.system(size: 11)).foregroundStyle(c.textFaint)
            }
            control
                .frame(maxWidth: controlWidth, alignment: .leading)
        }
    }
}

/// Label + description left, toggle right, vertically centered.
struct ToggleRow: View {
    let label: String
    var desc: String? = nil
    let on: Bool
    let onToggle: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onToggle) {
            HStack(alignment: .center, spacing: 16) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(label).font(.system(size: 13)).foregroundStyle(c.text)
                    if let desc {
                        Text(desc).font(.system(size: 11)).foregroundStyle(c.textFaint)
                    }
                }
                Spacer(minLength: 12)
                ZStack(alignment: on ? .trailing : .leading) {
                    Capsule().fill(on ? c.accent : c.toggleOff).frame(width: 38, height: 21)
                    Circle().fill(.white).frame(width: 17, height: 17).padding(2)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A horizontal segmented picker sized to its labels.
struct Segmented: View {
    let options: [String]
    let selected: String
    var onSelect: (String) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack(spacing: 3) {
            ForEach(options, id: \.self) { o in
                let active = o == selected
                Button { onSelect(o) } label: {
                    Text(display(o))
                        .font(.system(size: 12))
                        .foregroundStyle(active ? c.text : c.textMuted)
                        .padding(.horizontal, 11).padding(.vertical, 6)
                        .background(active ? c.bgRaised : .clear, in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(active ? c.border : .clear, lineWidth: 1))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(c.bgCard, in: RoundedRectangle(cornerRadius: 8))
    }

    private func display(_ raw: String) -> String {
        Int(raw) != nil ? raw : (raw.prefix(1).uppercased() + raw.dropFirst())
    }
}

/// Monospace field, commit-on-submit; used for urls, paths, numbers, keys.
struct TextFieldRow: View {
    let placeholder: String
    let value: String
    var secure = false
    let onSubmit: (String) -> Void

    @Environment(\.sofuuColors) private var c
    @State private var text = ""

    var body: some View {
        Group {
            if secure {
                SecureField(placeholder, text: $text)
            } else {
                TextField(placeholder, text: $text)
            }
        }
        .font(.system(size: 12, design: .monospaced))
        .foregroundStyle(c.text)
        .tint(c.accent)
        .textFieldStyle(.plain)
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(c.monoFieldBg, in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(c.border, lineWidth: 1))
        // Commit live on every keystroke: a commit-on-Enter field loses
        // typed text when the user clicks a button instead of pressing
        // return (the "endpoint must start with https://" ghost — the
        // field showed the text but the form state never saw it).
        .onChange(of: text) { _ in onSubmit(text) }
        .onSubmit { onSubmit(text) }
        .onAppear { text = value }
        .onChange(of: value) { text = $0 }
    }
}

/// Filled pill button for a card's primary action.
struct AccentButton: View {
    let label: String
    let onClick: () -> Void
    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            Text(label).font(.system(size: 12)).foregroundStyle(c.accentContrast)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(c.accent, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Smaller pill button for secondary actions.
struct SmallButton: View {
    let label: String
    let bg: Color
    let fg: Color
    var action: () -> Void

    var body: some View {
        Button { action() } label: {
            Text(label).font(.system(size: 12)).foregroundStyle(fg)
                .padding(.horizontal, 12).padding(.vertical, 7)
                .background(bg, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Screen

private struct GeneralPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var engineCard: JSONObject?
    @State private var cardTimer: Timer?

    var body: some View {
        let state = store.state
        VStack(spacing: 14) {
            SectionCard(label: "ENGINE") {
                if let e = engineCard {
                    engineLine("Model", e.optString("model").nilIfEmpty ?? "(none)")
                    engineLine("Provider", e.optString("provider").nilIfEmpty ?? "(none)")
                    engineLine("Session", String(e.optString("sessionId").prefix(10)).nilIfEmpty ?? "—")
                    engineLine("History", "\(e.optInt("historyTurns")) turns")
                    engineLine("Context", "\(grouped(e.optInt("contextTokens"))) tk")
                    engineLine("Spend", "$\(String(format: "%.4f", e.optDouble("sessionSpendUsd")))")
                    if e.optBool("streaming") {
                        Text("Generating…").font(.system(size: 11)).foregroundStyle(c.accent)
                    }
                    if e.optInt("pendingApprovals") > 0 {
                        Text("\(e.optInt("pendingApprovals")) awaiting approval")
                            .font(.system(size: 11)).foregroundStyle(c.warn)
                    }
                } else {
                    Text(state.engineReady ? "Connected — waiting for engine state…" : "Engine not connected.")
                        .font(.system(size: 12)).foregroundStyle(c.textFaint)
                }
            }

            SectionCard(label: "WORKSPACE") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Project folder").font(.system(size: 13)).foregroundStyle(c.text)
                    Text("Workspace root; sessions, memory and settings live in its .sofuu/")
                        .font(.system(size: 11)).foregroundStyle(c.textFaint)
                    HStack(alignment: .center, spacing: 10) {
                        Text(state.projectDir ?? "(none)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(c.textMuted)
                            .lineLimit(1)
                            .truncationMode(.head)
                        SmallButton(label: "Choose…", bg: c.bgRaised, fg: c.text) {
                            if let picked = pickFolder(), !picked.isEmpty {
                                store.openWorkspace(picked)
                            }
                        }
                    }
                }
            }

            SectionCard(label: "PERMISSIONS") {
                ControlRow(label: "Tool permission profile",
                           desc: "full = everything incl. shell · edit = reads + edits · plan = read-only · prompt = ask first") {
                    Segmented(options: ["full", "edit", "plan", "prompt"], selected: state.permissions) {
                        store.setPermissions($0)
                    }
                }
            }

            SectionCard(label: "MEMORY") {
                ToggleRow(label: "Brain memory",
                          desc: "Recall saved project memories into each turn",
                          on: state.config?.brain != false) {
                    store.patchConfig(["brain": !(state.config?.brain != false)])
                }
                Divider().overlay(c.separator)
                StackRow(label: "Recall minimum",
                         desc: "Lowest similarity for a memory to surface (0 = built-in 0.30)") {
                    TextFieldRow(placeholder: "0.30",
                                 value: (state.config?.recallMin ?? 0) > 0 ? String(state.config?.recallMin ?? 0) : "") {
                        if let n = Double($0) { store.patchConfig(["recall_min": n]) }
                    }
                }
                StackRow(label: "Recall budget (tokens)",
                         desc: "Cap on recalled text per turn (0 = auto ≈2% of the window)") {
                    TextFieldRow(placeholder: "auto",
                                 value: (state.config?.recallBudget ?? 0) > 0 ? String(state.config?.recallBudget ?? 0) : "") {
                        if let n = Int($0) { store.patchConfig(["recall_budget": n]) }
                    }
                }
            }

            SectionCard(label: "REASONING") {
                ToggleRow(label: "Smart context filters (ML)",
                          desc: "Freshness, compaction, relevance and allocation gates",
                          on: state.config?.ml != false) {
                    store.patchConfig(["ml": !(state.config?.ml != false)])
                }
                Divider().overlay(c.separator)
                ControlRow(label: "RLM sandbox",
                           desc: "Route long-context turns through the reasoning sandbox") {
                    Segmented(options: ["off", "on", "auto"], selected: state.config?.rlm ?? "off") {
                        store.patchConfig(["rlm": $0])
                    }
                }
            }

            SectionCard(label: "DESKTOP TOOLS",
                        footer: "Native capabilities the agent can call. Obey the permission profile; listed in /tools.") {
                MacToolsList(store: store)
            }

            SectionCard(label: "ENGINE CONTROLS") {
                EngineEnvControls(store: store)
            }
        }
        .onAppear { refreshCard() }
        .onDisappear { cardTimer?.invalidate() }
    }

    private func engineLine(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label).font(.system(size: 12)).foregroundStyle(c.textFaint)
                .frame(width: 72, alignment: .leading)
            Text(value).font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text)
            Spacer(minLength: 0)
        }
    }

    private func refreshCard() {
        cardTimer?.invalidate()
        Task { engineCard = await store.engine.chatState() }
        cardTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in
            Task { @MainActor in
                engineCard = await store.engine.chatState()
            }
        }
    }

    private func pickFolder() -> String? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.message = "Choose a project folder"
        return panel.runModal() == .OK ? panel.urls.first?.path : nil
    }
}

/// The native host tools the engine registered at boot.
struct MacToolsList: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c
    @State private var tools: [String] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if tools.isEmpty {
                Text("None registered yet.")
                    .font(.system(size: 12)).foregroundStyle(c.textFaint)
            } else {
                ForEach(tools, id: \.self) { t in
                    HStack(spacing: 6) {
                        Image(systemName: Icons2.check)
                            .font(.system(size: 9)).foregroundStyle(c.ok)
                        Text(t).font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text)
                    }
                }
            }
        }
        .task {
            // The engine fills the list during boot.
            for _ in 0..<20 {
                tools = store.engine.hostTools
                if !tools.isEmpty { break }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
        }
    }
}

/// The library's env-var knobs, restacked as plain rows inside the
/// Engine Controls card.
struct EngineEnvControls: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c
    @State private var noTools = false
    @State private var noShell = false
    @State private var noMl = false
    @State private var allowInternal = false
    @State private var backend = ""
    @State private var webEngine = ""
    @State private var maxSteps = ""
    @State private var stall = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ToggleRow(label: "Coding tools",
                      desc: "All 8 built-ins off; web + MCP unaffected",
                      on: noTools) {
                noTools.toggle(); store.runtimeEnv.noTools = noTools; store.applyRuntimeEnv()
            }
            ToggleRow(label: "Shell",
                      desc: "Keep bash listed but refuse every invocation",
                      on: noShell) {
                noShell.toggle(); store.runtimeEnv.noShell = noShell; store.applyRuntimeEnv()
            }
            ToggleRow(label: "Context gates (ML)",
                      desc: "Drop the whole sofuu.ml layer — needs a restart",
                      on: noMl) {
                noMl.toggle(); store.runtimeEnv.noMl = noMl; store.restartEngineForEnv()
            }
            ToggleRow(label: "Allow private addresses",
                      desc: "web.open may reach loopback/LAN instead of the SSRF guard",
                      on: allowInternal) {
                allowInternal.toggle(); store.runtimeEnv.webAllowInternal = allowInternal; store.applyRuntimeEnv()
            }
            Divider().overlay(c.separator)
            StackRow(label: "Memory backend",
                     desc: "Auto = semantic projector when its artifact exists, else hash-768") {
                Segmented(options: ["auto", "hash", "semantic"], selected: backend.isEmpty ? "auto" : backend) {
                    backend = ($0 == "auto") ? "" : $0
                    store.runtimeEnv.memoryBackend = backend
                    store.applyRuntimeEnv()
                }
            }
            StackRow(label: "Web search engine",
                     desc: "Blank picks the keyless default") {
                Segmented(options: ["auto", "brave", "tavily"], selected: webEngine.isEmpty ? "auto" : webEngine) {
                    webEngine = ($0 == "auto") ? "" : $0
                    store.runtimeEnv.webEngine = webEngine
                    store.applyRuntimeEnv()
                }
            }
            StackRow(label: "Turn step budget",
                     desc: "Tool rounds per turn (0 = engine default 200)",
                     controlWidth: 160) {
                TextFieldRow(placeholder: "200", value: maxSteps) {
                    maxSteps = $0
                    store.runtimeEnv.chatMaxSteps = Int($0.filter(\.isNumber)) ?? 0
                    store.applyRuntimeEnv()
                }
            }
            StackRow(label: "Stalled-request timeout",
                     desc: "Seconds before an unresponsive endpoint gives up (0 = default 300)",
                     controlWidth: 160) {
                TextFieldRow(placeholder: "300", value: stall) {
                    stall = $0
                    store.runtimeEnv.stallTimeoutSecs = Int($0.filter(\.isNumber)) ?? 0
                    store.applyRuntimeEnv()
                }
            }
        }
        .onAppear { load() }
    }

    func load() {
        let env = store.runtimeEnv
        noTools = env.noTools
        noShell = env.noShell
        noMl = env.noMl
        allowInternal = env.webAllowInternal
        backend = env.memoryBackend
        webEngine = env.webEngine
        maxSteps = env.chatMaxSteps > 0 ? String(env.chatMaxSteps) : ""
        stall = env.stallTimeoutSecs > 0 ? String(env.stallTimeoutSecs) : ""
    }
}

// MARK: - Appearance

private struct AppearancePane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        let state = store.state
        VStack(spacing: 14) {
            SectionCard(label: "THEME") {
                ControlRow(label: "Appearance",
                           desc: "The native window follows the same choice") {
                    Segmented(options: ["light", "dark", "system"], selected: state.theme) { store.setTheme($0) }
                }
                Divider().overlay(c.separator)
                ToggleRow(label: "Cloud background",
                          desc: "The procedural gradient backdrop",
                          on: state.cloudBg) { store.setCloudBg(!state.cloudBg) }
            }

            SectionCard(label: "SCALING") {
                ControlRow(label: "Interface zoom",
                           desc: "Scales the whole UI (⌘+ / ⌘− / ⌘0)") {
                    stepper(value: "\(Int((state.zoom * 100).rounded()))%",
                            onMinus: { store.zoomBy(-0.1) },
                            onPlus: { store.zoomBy(0.1) },
                            reset: abs(state.zoom - 1.0) > 0.001 ? { store.zoomReset() } : nil)
                }
                Divider().overlay(c.separator)
                ControlRow(label: "Chat font size",
                           desc: "Body text in the chat pane (14–24px)") {
                    stepper(value: "\(state.chatFontSize)px",
                            onMinus: { store.setChatFontSize(state.chatFontSize - 1) },
                            onPlus: { store.setChatFontSize(state.chatFontSize + 1) },
                            reset: nil)
                }
            }
        }
    }

    private func stepper(value: String,
                         onMinus: @escaping () -> Void,
                         onPlus: @escaping () -> Void,
                         reset: (() -> Void)?) -> some View {
        HStack(spacing: 8) {
            SmallButton(label: "−", bg: c.bgRaised, fg: c.text, action: onMinus)
            Text(value)
                .font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text)
                .frame(minWidth: 52)
            SmallButton(label: "+", bg: c.bgRaised, fg: c.text, action: onPlus)
            if let reset {
                SmallButton(label: "Reset", bg: c.bgCard, fg: c.textMuted, action: reset)
            }
        }
    }
}

// MARK: - Models

private struct ModelsPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        let state = store.state
        VStack(spacing: 14) {
            SectionCard(label: "ACTIVE MODEL") {
                StackRow(label: "Model", desc: "Sent verbatim to the provider", controlWidth: nil) {
                    TextFieldRow(placeholder: "model id", value: state.config?.model ?? "") {
                        if !$0.isEmpty { store.patchConfig(["model": $0]) }
                    }
                }
                ControlRow(label: "Thinking effort",
                           desc: "Off omits the parameter; rejecting models are remembered") {
                    Segmented(options: ["off", "low", "medium", "high", "max"],
                              selected: (state.config?.effort.nilIfEmpty) ?? "off") {
                        store.patchConfig(["effort": $0 == "off" ? "" : $0])
                    }
                }
            }

            SectionCard(label: "LIMITS") {
                StackRow(label: "Context window override",
                         desc: "0 = engine-resolved truth; manual values are clamped to the model's real limit",
                         controlWidth: 160) {
                    TextFieldRow(placeholder: "auto",
                                 value: (state.config?.ctxWindow ?? 0) == 0 ? "" : String(state.config?.ctxWindow ?? 0)) {
                        if let n = Int($0) { store.patchConfig(["ctx_window": n]) }
                    }
                }
                StackRow(label: "Max output tokens",
                         desc: "0 = provider default — small overrides get 400'd by big models",
                         controlWidth: 160) {
                    TextFieldRow(placeholder: "model default",
                                 value: (state.config?.maxOutput ?? 0) == 0 ? "" : String(state.config?.maxOutput ?? 0)) {
                        if let n = Int($0) { store.patchConfig(["max_output": n]) }
                    }
                }
                StackRow(label: "Session budget (USD)",
                         desc: "The engine refuses new turns past the cap (0 = uncapped)",
                         controlWidth: 160) {
                    TextFieldRow(placeholder: "uncapped",
                                 value: (state.config?.budgetUsd ?? 0) > 0 ? String(state.config?.budgetUsd ?? 0) : "") {
                        if let n = Double($0) { store.patchConfig(["budget_usd": n]) }
                    }
                }
            }

            SectionCard(label: "PROVIDER MODEL LISTS",
                        footer: "Prefetched every 5 minutes — pick a model to switch instantly.") {
                ForEach(state.config?.providers ?? []) { p in
                    let entry = state.modelsCache[p.name]
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text(p.name).font(.system(size: 12, weight: .medium)).foregroundStyle(c.text)
                            Spacer()
                            Text(entry?.error ?? entry?.note ?? "\(entry?.models.count ?? 0) cached")
                                .font(.system(size: 10)).foregroundStyle(c.textFaint)
                        }
                        if let models = entry?.models, !models.isEmpty {
                            VStack(alignment: .leading, spacing: 0) {
                                ForEach(models.prefix(12), id: \.self) { m in
                                    Button { store.pickModel(p.name, m) } label: {
                                        HStack(spacing: 6) {
                                            Text(m)
                                                .font(.system(size: 11, design: .monospaced))
                                                .foregroundStyle(m == p.model ? c.accent : c.textMuted)
                                            if m == p.model {
                                                Text("current").font(.system(size: 9)).foregroundStyle(c.accent)
                                            }
                                            Spacer(minLength: 0)
                                        }
                                        .padding(.vertical, 4)
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    if p.name != state.config?.providers.last?.name {
                        Divider().overlay(c.separator)
                    }
                }
                SmallButton(label: "Refresh lists", bg: c.accent, fg: c.accentContrast) {
                    store.refreshModelCache()
                }
            }

            SectionCard(label: "ON-DEVICE GGUF MODELS",
                        footer: "Runs locally, no network. Imports land in ~/.sofuu/models.") {
                if store.localModelURLs.isEmpty {
                    Text("No models imported.").font(.system(size: 12)).foregroundStyle(c.textFaint)
                }
                ForEach(store.localModelURLs, id: \.path) { u in
                    let active = state.config?.model == u.path
                    HStack {
                        Button { Task { await store.pickLocalModel(u.path) } } label: {
                            HStack(spacing: 8) {
                                Text(LocalModels.label(u))
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundStyle(active ? c.accent : c.text)
                                if active {
                                    Text("in use").font(.system(size: 10)).foregroundStyle(c.accent)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        Spacer()
                        if !active {
                            SmallButton(label: "Remove", bg: c.bgCard, fg: c.errStrong) {
                                store.deleteLocalModel(u.path)
                            }
                        }
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }
}

// MARK: - Providers

private struct ProvidersPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var newName = ""
    @State private var newEndpoint = ""
    @State private var newProfile = "openai"
    @State private var newModel = ""
    @State private var newKey = ""
    @State private var notice: String?

    var body: some View {
        let state = store.state
        VStack(spacing: 14) {
            SectionCard(label: "CONFIGURED PROVIDERS") {
                if (state.config?.providers ?? []).isEmpty {
                    Text("No providers yet — add one below.")
                        .font(.system(size: 12)).foregroundStyle(c.textFaint)
                }
                ForEach(state.config?.providers ?? []) { p in
                    ProviderRow(store: store, p: p, isActive: p.name == state.config?.active)
                    if p.name != state.config?.providers.last?.name {
                        Divider().overlay(c.separator)
                    }
                }
            }

            SectionCard(label: "ADD PROVIDER",
                        footer: notice ?? "The new provider becomes active immediately. Keys are stored in ~/.sofuu/config.json (0600).") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 10) {
                        TextFieldRow(placeholder: "name (e.g. openai)", value: newName) { newName = $0 }
                        Segmented(options: ["openai", "anthropic", "local"], selected: newProfile) { newProfile = $0 }
                    }
                    TextFieldRow(placeholder: "https://…/v1/chat/completions", value: newEndpoint) { newEndpoint = $0 }
                    HStack(spacing: 10) {
                        TextFieldRow(placeholder: "model id (optional)", value: newModel) { newModel = $0 }
                        TextFieldRow(placeholder: "API key (optional)", value: newKey, secure: true) { newKey = $0 }
                    }
                    AccentButton(label: "Add and set active") { addProvider() }
                }
            }
        }
    }

    private func addProvider() {
        let name = newName.trimmingCharacters(in: .whitespaces)
        let endpoint = newEndpoint.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else {
            notice = "A provider needs a name."
            return
        }
        guard endpoint.hasPrefix("http://") || endpoint.hasPrefix("https://") else {
            notice = "The endpoint must start with http:// or https://."
            return
        }
        var entry: JSONObject = ["name": name, "endpoint": endpoint, "profile": newProfile]
        let model = newModel.trimmingCharacters(in: .whitespaces)
        if !model.isEmpty { entry["model"] = model }
        store.patchConfig([
            "providers": [entry],
            "active": name,
            "provider": name,
        ])
        let key = newKey.trimmingCharacters(in: .whitespaces)
        if !key.isEmpty { store.setProviderKey(name, key) }
        notice = "Provider \(name) added and set active."
        newName = ""; newEndpoint = ""; newModel = ""; newKey = ""
    }
}

private struct ProviderRow: View {
    @ObservedObject var store: AppStore
    let p: ProviderEntry
    let isActive: Bool

    @Environment(\.sofuuColors) private var c
    @State private var keyText = ""
    @State private var editingKey = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(p.name).font(.system(size: 13, weight: .medium)).foregroundStyle(c.text)
                        if isActive {
                            Text("active").font(.system(size: 10)).foregroundStyle(c.accentContrast)
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .background(c.accent, in: Capsule())
                        }
                    }
                    Text(p.endpoint.isEmpty ? p.profile : p.endpoint)
                        .font(.system(size: 11)).foregroundStyle(c.textFaint).lineLimit(1)
                }
                Spacer(minLength: 12)
                if isActive {
                    Text(p.hasApiKey ? "Keychain" : "no key")
                        .font(.system(size: 11))
                        .foregroundStyle(p.hasApiKey ? c.ok : c.warn)
                } else {
                    SmallButton(label: "Use", bg: c.bgRaised, fg: c.text) {
                        store.patchConfig(["provider": p.name, "active": p.name, "model": p.model])
                    }
                    SmallButton(label: "Remove", bg: c.bgCard, fg: c.errStrong) {
                        if let err = store.removeProvider(p.name) {
                            store.systemRow(err)
                        }
                    }
                }
            }

            if editingKey {
                TextFieldRow(placeholder: "paste API key", value: keyText, secure: true) { keyText = $0 }
                HStack(spacing: 8) {
                    AccentButton(label: "Save key") {
                        let t = keyText.trimmingCharacters(in: .whitespaces)
                        if !t.isEmpty { store.setProviderKey(p.name, t) }
                        keyText = ""
                        editingKey = false
                    }
                    if p.hasApiKey {
                        SmallButton(label: "Remove key", bg: c.bgCard, fg: c.errStrong) {
                            store.setProviderKey(p.name, "")
                            editingKey = false
                        }
                    }
                    SmallButton(label: "Cancel", bg: c.bgCard, fg: c.textMuted) {
                        keyText = ""
                        editingKey = false
                    }
                }
            } else {
                SmallButton(label: p.hasApiKey ? "Replace key" : "Set key",
                            bg: c.bgCard, fg: c.accent) {
                    editingKey = true
                }
            }
        }
    }
}
