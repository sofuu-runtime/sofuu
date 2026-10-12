import SwiftUI

private let settingsTabs = ["general", "appearance", "models", "providers", "agents", "brain", "usage"]

/// Full-screen settings with the same five panes as the desktop modal /
/// Android app.
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
        ZStack {
            c.bgApp.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("Settings").font(.system(size: 20, weight: .medium)).foregroundStyle(c.text)
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: Icons2.close)
                            .font(.system(size: 16)).foregroundStyle(c.textMuted)
                            .frame(width: 36, height: 36).contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 16).padding(.vertical, 12)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(settingsTabs, id: \.self) { t in
                            let active = t == tab
                            Button { tab = t } label: {
                                Text(t.prefix(1).uppercased() + t.dropFirst())
                                    .font(.system(size: 13))
                                    .foregroundStyle(active ? c.accentContrast : c.textMuted)
                                    .padding(.horizontal, 14).padding(.vertical, 8)
                                    .background(active ? c.accent : c.bgCard, in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 12)
                }

                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        switch tab {
                        case "general": GeneralPane(store: store)
                        case "appearance": AppearancePane(store: store)
                        case "models": ModelsPane(store: store)
                        case "providers": ProvidersPane(store: store)
                        case "agents": AgentsPane(store: store)
                        case "brain": BrainPane(store: store)
                        case "usage": UsagePane(store: store)
                        default: EmptyView()
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                }
            }
        }
    }
}

// MARK: - Shared bits

private struct Section: View {
    let label: String
    @Environment(\.sofuuColors) private var c
    var body: some View {
        Text(label).font(.system(size: 12)).foregroundStyle(c.textFaint)
            .padding(.top, 22).padding(.bottom, 6)
    }
}

private struct RowBox<Content: View>: View {
    let label: String
    let desc: String?
    @ViewBuilder var content: Content
    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label).font(.system(size: 14)).foregroundStyle(c.text)
            if let desc {
                Text(desc).font(.system(size: 11)).foregroundStyle(c.textFaint).padding(.bottom, 6)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
    }
}

private struct Segmented: View {
    let options: [String]
    let selected: String
    let onSelect: (String) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack(spacing: 3) {
            ForEach(options, id: \.self) { o in
                let active = o == selected
                Button { onSelect(o) } label: {
                    Text(o.prefix(1).uppercased() + o.dropFirst())
                        .font(.system(size: 12))
                        .foregroundStyle(active ? c.text : c.textMuted)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(active ? c.bgRaised : .clear, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(c.bgCard, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct TextFieldRow: View {
    let placeholder: String
    let value: String
    let onSubmit: (String) -> Void

    @Environment(\.sofuuColors) private var c
    @State private var text = ""

    var body: some View {
        TextField(placeholder, text: $text)
            .font(.system(size: 13, design: .monospaced))
            .foregroundStyle(c.text)
            .tint(c.accent)
            .textFieldStyle(.plain)
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)
            .padding(.horizontal, 12).padding(.vertical, 10)
            .frame(minHeight: 44)
            .background(c.monoFieldBg, in: RoundedRectangle(cornerRadius: 8))
            .onSubmit { onSubmit(text) }
            .onAppear { text = value }
            .onChange(of: value) { text = $0 }
    }
}

private struct AccentButton: View {
    let label: String
    let onClick: () -> Void
    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            Text(label).font(.system(size: 13)).foregroundStyle(c.accentContrast)
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(c.accent, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - General

private struct GeneralPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        let state = store.state
        VStack(alignment: .leading, spacing: 0) {
            RowBox(label: "Project folder",
                   desc: "Workspace root and per-project storage (.sofuu/ — sessions, memory, settings).") {
                Text(state.projectDir ?? "(none)")
                    .font(.system(size: 12, design: .monospaced)).foregroundStyle(c.textMuted)
            }
            RowBox(label: "Tool permission profile", desc: "Applied live in the engine and re-applied on launch.") {
                Segmented(options: ["full", "edit", "plan", "prompt"], selected: state.permissions) {
                    store.setPermissions($0)
                }
            }
            RowBox(label: "Memory (brain)", desc: "Recall saved project memories into each turn (.sofuu/brain/).") {
                Segmented(options: ["on", "off"], selected: state.config?.brain != false ? "on" : "off") {
                    store.patchConfig(["brain": $0 == "on"])
                }
            }
            RowBox(label: "Smart context filters", desc: "On-device neural filters (freshness, compaction, relevance, allocation).") {
                Segmented(options: ["on", "off"], selected: state.config?.ml != false ? "on" : "off") {
                    store.patchConfig(["ml": $0 == "on"])
                }
            }
            RowBox(label: "RLM sandbox", desc: "Route very long-context turns through the local reasoning sandbox.") {
                Segmented(options: ["off", "on", "auto"], selected: state.config?.rlm ?? "off") {
                    store.patchConfig(["rlm": $0])
                }
            }
            RowBox(label: "Context window", desc: "Override the model's window (0 = engine-resolved truth).") {
                TextFieldRow(placeholder: "auto",
                             value: (state.config?.ctxWindow ?? 0) == 0 ? "" : String(state.config?.ctxWindow ?? 0)) {
                    if let n = Int($0) { store.patchConfig(["ctx_window": n]) }
                }
            }
            RowBox(label: "Max output tokens", desc: "Override the per-request output cap (0 = model default).") {
                TextFieldRow(placeholder: "model default",
                             value: (state.config?.maxOutput ?? 0) == 0 ? "" : String(state.config?.maxOutput ?? 0)) {
                    if let n = Int($0) { store.patchConfig(["max_output": n]) }
                }
            }
            RowBox(label: "Session budget (USD)", desc: "The engine refuses new turns past the cap (0 = uncapped).") {
                TextFieldRow(placeholder: "uncapped",
                             value: (state.config?.budgetUsd ?? 0) > 0 ? String(state.config?.budgetUsd ?? 0) : "") {
                    if let n = Double($0) { store.patchConfig(["budget_usd": n]) }
                }
            }
            Spacer(minLength: 24)
        }
    }
}

// MARK: - Appearance

private struct AppearancePane: View {
    @ObservedObject var store: AppStore

    var body: some View {
        let state = store.state
        VStack(alignment: .leading, spacing: 0) {
            RowBox(label: "Theme", desc: "Follows the system unless you pick one.") {
                Segmented(options: ["light", "dark", "system"], selected: state.theme) { store.setTheme($0) }
            }
            RowBox(label: "Cloud background", desc: "Procedural gradient backdrop.") {
                Segmented(options: ["on", "off"], selected: state.cloudBg ? "on" : "off") { store.setCloudBg($0 == "on") }
            }
            RowBox(label: "Chat font size", desc: "Body text size in the chat pane.") {
                Segmented(options: ["13", "15", "17", "19"], selected: String(state.chatFontSize)) {
                    if let n = Int($0) { store.setChatFontSize(n) }
                }
            }
            Spacer(minLength: 24)
        }
    }
}

// MARK: - Models

private struct ModelsPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        let state = store.state
        VStack(alignment: .leading, spacing: 0) {
            ForEach(state.config?.providers ?? []) { p in
                RowBox(label: "Provider: \(p.name)", desc: p.endpoint) {
                    let models = state.modelsCache[p.name]?.models ?? []
                    if models.isEmpty {
                        Text("No model list yet — tap refresh.")
                            .font(.system(size: 12)).foregroundStyle(c.textFaint)
                    } else {
                        ForEach(models.prefix(30), id: \.self) { m in
                            Button { store.pickModel(p.name, m) } label: {
                                Text(m)
                                    .font(.system(size: 13, design: .monospaced))
                                    .foregroundStyle(m == p.model ? c.accent : c.text)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .frame(minHeight: 44)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            AccentButton(label: "Refresh") { store.refreshModelCache() }
                .padding(.top, 12)
            Spacer(minLength: 24)
        }
    }
}

// MARK: - Providers

private struct ProvidersPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var newName = ""
    @State private var newEndpoint = ""
    @State private var newModel = ""
    @State private var newKey = ""

    var body: some View {
        let state = store.state
        VStack(alignment: .leading, spacing: 0) {
            Section(label: "Add provider")
            TextFieldRow(placeholder: "name (e.g. openai)", value: newName) { newName = $0 }
                .padding(.bottom, 8)
            TextFieldRow(placeholder: "https://…/v1/chat/completions", value: newEndpoint) { newEndpoint = $0 }
                .padding(.bottom, 8)
            TextFieldRow(placeholder: "model id", value: newModel) { newModel = $0 }
                .padding(.bottom, 8)
            TextFieldRow(placeholder: "API key (stored in the iOS Keychain)", value: newKey) { newKey = $0 }
                .padding(.bottom, 8)

            AccentButton(label: "Add") {
                guard !newName.isEmpty, !newEndpoint.isEmpty else { return }
                var providers = (state.config?.providers ?? []).map { $0.toJson() }
                providers.append(["name": newName, "endpoint": newEndpoint,
                                  "model": newModel, "profile": ""])
                store.patchConfig(["providers": providers, "active": newName, "provider": newName])
                if !newKey.isEmpty { store.setProviderKey(newName, newKey) }
                newName = ""; newEndpoint = ""; newModel = ""; newKey = ""
            }
            .padding(.top, 10)

            Section(label: "Configured providers")
            ForEach(state.config?.providers ?? []) { p in
                ProviderRow(store: store, p: p, isActive: p.name == state.config?.active)
            }
            Spacer(minLength: 24)
        }
    }
}

private struct ProviderRow: View {
    @ObservedObject var store: AppStore
    let p: ProviderEntry
    let isActive: Bool

    @Environment(\.sofuuColors) private var c
    @State private var keyText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(p.name).font(.system(size: 14)).foregroundStyle(c.text)
                    Text(p.endpoint).font(.system(size: 11)).foregroundStyle(c.textFaint).lineLimit(1)
                    Text(p.hasApiKey ? "key saved · Keychain" : "no key")
                        .font(.system(size: 11))
                        .foregroundStyle(p.hasApiKey ? c.ok : c.warn)
                }
                Spacer()
                if !isActive {
                    Button { _ = store.removeProvider(p.name) } label: {
                        Text("Remove").font(.system(size: 12)).foregroundStyle(c.errStrong).padding(8)
                    }
                    .buttonStyle(.plain)
                } else {
                    Text("active").font(.system(size: 11)).foregroundStyle(c.accent).padding(8)
                }
            }
            // Key entry: explicit save (never a write per keystroke).
            if keyText.isEmpty && p.hasApiKey {
                Button { keyText = " " } label: {
                    Text("Replace key").font(.system(size: 12)).foregroundStyle(c.accent).padding(.vertical, 6)
                }
                .buttonStyle(.plain)
            } else {
                TextFieldRow(placeholder: "paste API key",
                             value: keyText.trimmingCharacters(in: .whitespaces)) { keyText = $0 }
                    .padding(.bottom, 10)
                HStack(spacing: 8) {
                    AccentButton(label: "Save key") {
                        let t = keyText.trimmingCharacters(in: .whitespaces)
                        if !t.isEmpty { store.setProviderKey(p.name, t); keyText = "" }
                    }
                    Button { store.setProviderKey(p.name, ""); keyText = "" } label: {
                        Text("Clear").font(.system(size: 12)).foregroundStyle(c.errStrong)
                            .padding(.horizontal, 14).padding(.vertical, 8)
                            .background(c.bgCardHover, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
    }
}

// MARK: - Usage

private struct UsagePane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var usage: JSONObject?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            let totals = usage?.optObject("totals")
            RowBox(label: "Session spend", desc: "Tokens + cost across this workspace's sessions.") {
                Text(String(format: "$%.4f  ·  %@ in  ·  %@ out",
                            totals?.optDouble("cost") ?? 0,
                            grouped(Int(totals?.optDouble("pt") ?? 0)),
                            grouped(Int(totals?.optDouble("ct") ?? 0))))
                    .font(.system(size: 13, design: .monospaced)).foregroundStyle(c.text)
            }

            Section(label: "Per day")
            let days = usage?.optArray("days") ?? []
            if days.isEmpty {
                Text("No usage yet.").font(.system(size: 13)).foregroundStyle(c.textFaint)
            } else {
                ForEach(Array(days.enumerated()), id: \.offset) { _, d in
                    if let d = d as? JSONObject {
                        mono("\(d.optString("day"))  ·  \(d.optInt("turns")) turns  ·  \(grouped(Int(d.optDouble("pt")))) in  ·  \(grouped(Int(d.optDouble("ct")))) out")
                    }
                }
            }

            Section(label: "Per session")
            let sess = usage?.optArray("sessions") ?? []
            if sess.isEmpty {
                Text("No sessions yet.").font(.system(size: 13)).foregroundStyle(c.textFaint)
            } else {
                ForEach(Array(sess.prefix(20).enumerated()), id: \.offset) { _, s in
                    if let s = s as? JSONObject {
                        let name = s.optString("task").nilIfEmpty ?? s.optString("id")
                        mono("\(String(name.prefix(28)))  ·  \(s.optString("model"))  ·  $\(String(format: "%.4f", s.optDouble("cost")))")
                    }
                }
            }

            Section(label: "Per model")
            let models = usage?.optArray("models") ?? []
            if models.isEmpty {
                Text("No usage yet.").font(.system(size: 13)).foregroundStyle(c.textFaint)
            } else {
                ForEach(Array(models.enumerated()), id: \.offset) { _, m in
                    if let m = m as? JSONObject {
                        mono("\(m.optString("model"))  ·  \(m.optInt("turns")) turns  ·  $\(String(format: "%.4f", m.optDouble("cost")))")
                    }
                }
            }
            Spacer(minLength: 24)
        }
        .task { usage = await store.engine.usageReport() }
    }

    private func mono(_ s: String) -> some View {
        Text(s).font(.system(size: 12, design: .monospaced))
            .foregroundStyle(c.textMuted)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 3)
    }
}

// ── Agents ─────────────────────────────────────────────────────────

/// Registered agents, and the form to add one from a phone.
///
/// `define` is the runtime's own registry — an agent added here is a
/// first-class entry that the chat turn, `/agents` and `delegate` all see,
/// not a local preset. Its tools are named the way `sofuu.tools.TOOLS`
/// names them, which after host-tool registration includes the phone's own
/// capabilities (device_info, share_text, …).
private struct AgentsPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c
    @State private var agents: [JSONObject] = []
    @State private var note: String?

    @State private var name = "reviewer"
    @State private var system = "You are a careful code reviewer. Report only what you can point at."
    @State private var maxSteps = "24"
    @State private var withWeb = true
    @State private var withCode = true

    @State private var task = ""
    @State private var answer: JSONObject?
    @State private var running = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Section(label: "Registered")
            if agents.isEmpty {
                Text("No agents defined yet. Add one below.")
                    .font(.system(size: 12)).foregroundStyle(c.textFaint)
            } else {
                ForEach(Array(agents.enumerated()), id: \.offset) { _, a in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(a.optString("name"))
                            .font(.system(size: 14, design: .monospaced)).foregroundStyle(c.text)
                        Text(describe(a))
                            .font(.system(size: 11)).foregroundStyle(c.textFaint)
                    }
                    .padding(.vertical, 4)
                }
            }

            Section(label: "Define an agent")
            TextFieldRow(placeholder: "reviewer", value: "") { name = $0 }
            TextFieldRow(placeholder: "what this agent is for", value: "") { system = $0 }
            TextFieldRow(placeholder: "24", value: "") { maxSteps = $0 }
            Toggle(isOn: $withWeb) {
    Text("Web tools").font(.system(size: 14)).foregroundStyle(c.text)
}
.font(.system(size: 12))
.tint(c.accent)
.padding(.vertical, 4)
.overlay(alignment: .bottomLeading) {
    Text("web_search and web_open").font(.system(size: 11)).foregroundStyle(c.textFaint)
    .offset(y: 16)
}
            Toggle(isOn: $withCode) {
    Text("Coding tools").font(.system(size: 14)).foregroundStyle(c.text)
}
.font(.system(size: 12))
.tint(c.accent)
.padding(.vertical, 4)
.overlay(alignment: .bottomLeading) {
    Text("the 8 built-ins: read/write/edit/grep/glob/dir/bash/todo").font(.system(size: 11)).foregroundStyle(c.textFaint)
    .offset(y: 16)
}
            HStack(spacing: 6) {
                SmallButton(label: "Register", bg: c.accent, fg: c.accentContrast) {
                    Task { @MainActor in
                        var tools: [String] = []
                        if withWeb { tools.append("web") }
                        if withCode { tools.append("code") }
                        let budget: JSONObject = ["maxSteps": Int(maxSteps.filter(\.isNumber)) ?? 24]
                        let r = await RuntimeSurface(engine: store.engine)
                            .defineAgent(name: name.trimmingCharacters(in: .whitespaces),
                                         system: system.trimmingCharacters(in: .whitespaces),
                                         tools: tools, budget: budget)
                        if (r?["ok"] as? Bool) == true {
                            note = "\(r?.optString("name") ?? name) registered — the agent can use it by name, and /agents lists it."
                        } else {
                            note = "Could not register: \(r?.optString("error") ?? "unknown error")"
                        }
                        await reload()
                    }
                }
            }
            .padding(.top, 10)
            if let note {
                Text(note).font(.system(size: 11)).foregroundStyle(c.textFaint).padding(.top, 6)
            }

            Section(label: "Run headlessly")
            TextFieldRow(placeholder: "a task to hand the agent", value: "") { task = $0 }
            SmallButton(
                label: running ? "Running…" : "Run",
                bg: running ? c.bgCard : c.accent,
                fg: running ? c.textFaint : c.accentContrast
            ) {
                guard !running, !task.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                running = true
                answer = nil
                Task { @MainActor in
                    answer = await RuntimeSurface(engine: store.engine)
                        .runAgent(name: name.trimmingCharacters(in: .whitespaces),
                                  task: task.trimmingCharacters(in: .whitespaces))
                    running = false
                }
            }
            Text("A run goes through the same agent loop, tools, budget and brain as a chat turn — the only difference is that the answer is collected instead of streamed.")
                .font(.system(size: 10)).foregroundStyle(c.textFaint).padding(.top, 6)
            if let a = answer {
                if (a["ok"] as? Bool) != true {
                    Text(a.optString("error").isEmpty ? "run failed" : a.optString("error"))
                        .font(.system(size: 12)).foregroundStyle(c.err).padding(.top, 8)
                } else {
                    let stopped = a.optString("stopped")
                    let head = "\(a.optInt("steps")) steps"
                        + (stopped.isEmpty || stopped == "null" ? "" : " · stopped: \(stopped)")
                    Text(head).font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(c.textFaint).padding(.top, 8)
                    Text(a.optString("answer")).font(.system(size: 13))
                        .foregroundStyle(c.text).padding(.top, 4)
                }
            }
        }
        .task { await reload() }
    }

    private func describe(_ a: JSONObject) -> String {
        var parts: [String] = []
        if let tools = a["tools"] as? [Any], !tools.isEmpty {
            parts.append(tools.compactMap { $0 as? String }.joined(separator: ", "))
        } else {
            parts.append("no tools")
        }
        let subs = a.optInt("agents")
        if subs > 0 { parts.append("\(subs) sub-agent slots") }
        let provider = a.optString("provider")
        if !provider.isEmpty { parts.append(provider) }
        return parts.joined(separator: " · ")
    }

    private func reload() async {
        agents = await RuntimeSurface(engine: store.engine).agents()
    }
}

// ── Brain ──────────────────────────────────────────────────────────

/// The CMA brain and the vector spaces behind it.
///
/// This is what `/remember`, `/why` and `/brain` sit on in the CLI, made
/// visible: what is stored, in which embedding space, and the operations
/// that change it. Every call routes through the ONE cached brain handle
/// on the engine thread, because two handles on one QTSQ file clobber each
/// other on flush.
private struct BrainPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var info: JSONObject?
    @State private var spaces: JSONObject?
    @State private var ml: JSONObject?
    @State private var err: String?
    @State private var reload = 0

    @State private var recall = ""
    @State private var hits: [JSONObject] = []
    @State private var decayArmed = false
    @State private var labelNote: String?

    @State private var simA = "the cat sat down"
    @State private var simB = "a feline rested"
    @State private var sim: JSONObject?

    @State private var gcNote: String?
    @State private var probeTool = "read_file"
    @State private var probeArgs = #"{"path":"README.md"}"#
    @State private var verdict: JSONObject?

    @State private var importingModel = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            StoredSection
            SearchSection
            ContextGatesSection
            GateCheckSection
            MemorySection
            SpacesSection
            SimilaritySection
            LocalModelsSection
        }
        .task(id: reload) { await load() }
        .task(id: recall) {
            if !recall.isEmpty { await runRecall() }
        }
        .fileImporter(
            isPresented: $importingModel,
            allowedContentTypes: [.item],
            allowsMultipleSelection: false
        ) { result in
            importingModel = false
            if case .success(let urls) = result, let u = urls.first {
                Task { @MainActor in await store.importLocalModel(u) }
            }
        }
        .task(id: simA + "\u{0}" + simB) {
            if !simA.isEmpty && !simB.isEmpty {
                sim = RuntimeSurface(engine: store.engine).similarity(simA, simB)
            }
        }
    }

    private func load() async {
        let s = RuntimeSurface(engine: store.engine)
        let (i, sp, m) = await (s.brainInfo(), s.spaces(), s.mlState())
        info = i
        spaces = sp
        ml = m
        err = (i?["ok"] as? Bool) == true ? nil : (i?.optString("error"))
    }

    // MARK: - Stored
    @ViewBuilder
    private var StoredSection: some View {
        Section(label: "Stored")
        if let err {
            Text(err).font(.system(size: 12)).foregroundStyle(c.err)
        } else if let i = info {
            Text("\(i.optInt("count")) memories")
                .font(.system(size: 20, design: .monospaced)).foregroundStyle(c.text)
            let sub = "space \(i.optString("backend")) · \(i.optInt("dim"))-dim"
                + ((i["semantic"] as? Bool) == true ? " · semantic channel on" : "")
            Text(sub).font(.system(size: 11, design: .monospaced))
                .foregroundStyle(c.textFaint).padding(.top, 2)
        } else {
            Text("Reading the brain…").font(.system(size: 12)).foregroundStyle(c.textFaint)
        }

        HStack(spacing: 6) {
            SmallButton(label: "Refresh", bg: c.accent, fg: c.accentContrast) { reload += 1 }
            SmallButton(label: "Consolidate", bg: c.accent, fg: c.accentContrast) {
                Task { @MainActor in
                    _ = await RuntimeSurface(engine: store.engine).brainConsolidate()
                    reload += 1
                }
            }
            SmallButton(
                label: decayArmed ? "Confirm decay" : "Decay tick",
                bg: decayArmed ? c.warn : c.accent,
                fg: decayArmed ? c.bgApp : c.accentContrast
            ) {
                if decayArmed {
                    decayArmed = false
                    Task { @MainActor in
                        _ = await RuntimeSurface(engine: store.engine).brainDecay()
                        reload += 1
                    }
                } else { decayArmed = true }
            }
        }
        .padding(.top, 10)
        if decayArmed {
            Text("Decay weakens and prunes old memories. Tap again to run it.")
                .font(.system(size: 10)).foregroundStyle(c.warn).padding(.top, 4)
        }
    }

    // MARK: - Search
    @ViewBuilder
    private var SearchSection: some View {
        Section(label: "Search")
        TextFieldRow(placeholder: "a fact to recall", value: "") { recall = $0 }
        if !recall.isEmpty {
            Button { Task { await runRecall() } } label: {
                Text("Recall").font(.system(size: 12)).foregroundStyle(c.accentContrast)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(c.accent, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        ForEach(Array(hits.enumerated()), id: \.offset) { _, h in
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(h.optString("text").isEmpty ? h.optString("kind") : h.optString("text"))
                        .font(.system(size: 12)).foregroundStyle(c.text)
                    Text(String(format: "score %.3f · %@", h.optDouble("score"), h.optString("tier")))
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(c.textFaint)
                }
                Spacer()
                let id = h.optInt("id", -1)
                if id >= 0 {
                    Button {
                        Task { @MainActor in
                            _ = await RuntimeSurface(engine: store.engine).brainForget(id)
                            hits = []
                            reload += 1
                        }
                    } label: {
                        Image(systemName: "xmark").font(.system(size: 10)).foregroundStyle(c.textFaint)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 3)
        }
    }

    // MARK: - Context gates
    @ViewBuilder
    private var ContextGatesSection: some View {
        Section(label: "Context gates")
        if let m = ml, (m["ok"] as? Bool) == true {
            let ws = m["workset"] as? JSONObject
            Text("\(ws?.optInt("runs") ?? 0) runs · \(ws?.optInt("calls") ?? 0) calls · \(ws?.optInt("segments") ?? 0) segments")
                .font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text)
            if let on = (m["info"] as? JSONObject)?["on"] as? JSONObject {
                ForEach(on.keys.sorted(), id: \.self) { k in
                    HStack {
                        Text(k).font(.system(size: 11, design: .monospaced)).foregroundStyle(c.textFaint)
                        Spacer()
                        Text((on[k] as? Bool) == true ? "on" : "off")
                            .font(.system(size: 11))
                            .foregroundStyle((on[k] as? Bool) == true ? c.accent : c.textMuted)
                    }
                }
            }
            HStack(spacing: 6) {
                SmallButton(label: "Label wrong", bg: c.accent, fg: c.accentContrast) {
                    Task { @MainActor in
                        let r = await RuntimeSurface(engine: store.engine).mlFeedback(kind: "wrong")
                        labelNote = (r?["ok"] as? Bool) == true
                            ? "Label recorded."
                            : "The engine declined that label — it needs an observed call to attach to."
                        reload += 1
                    }
                }
                SmallButton(label: "Label wasted", bg: c.accent, fg: c.accentContrast) {
                    Task { @MainActor in
                        let r = await RuntimeSurface(engine: store.engine).mlFeedback(kind: "wasted")
                        labelNote = (r?["ok"] as? Bool) == true
                            ? "Label recorded."
                            : "The engine declined that label — it needs an observed call to attach to."
                        reload += 1
                    }
                }
            }
            .padding(.top, 8)
            if let labelNote {
                Text(labelNote).font(.system(size: 10))
                    .foregroundStyle(labelNote.hasPrefix("Label recorded") ? c.textFaint : c.warn)
                    .padding(.top, 4)
            }
        } else if let m = ml {
            Text(m.optString("error").isEmpty ? "The sofuu.ml layer is not loaded in this build." : m.optString("error"))
                .font(.system(size: 12)).foregroundStyle(c.err)
        } else {
            Text("The sofuu.ml layer is not loaded in this build.")
                .font(.system(size: 12)).foregroundStyle(c.textFaint)
        }
    }

    // MARK: - Gate check
    @ViewBuilder
    private var GateCheckSection: some View {
        Section(label: "Gate check")
        Text("Ask what the gates would say about a call before the agent makes it.")
            .font(.system(size: 11)).foregroundStyle(c.textFaint)
        TextFieldRow(placeholder: "read_file", value: "") { probeTool = $0 }
        TextFieldRow(placeholder: "{\"path\":\"README.md\"}", value: "") { probeArgs = $0 }
        SmallButton(label: "Check", bg: c.accent, fg: c.accentContrast) {
            Task { @MainActor in
                var args: JSONObject = [:]
                if let d = probeArgs.data(using: .utf8),
                   let p = try? JSONSerialization.jsonObject(with: d) as? JSONObject { args = p }
                verdict = await RuntimeSurface(engine: store.engine).mlCheck(tool: probeTool, args: args)
            }
        }
        .padding(.top, 6)
        if let v = verdict {
            if (v["ok"] as? Bool) == true {
                let nudge = v.optString("nudge")
                let s = String(format: "pass · %@ · %.3f", v.optString("source"), v.optDouble("score"))
                    + (nudge.isEmpty || nudge == "null" ? "" : " · nudge: \(nudge)")
                Text(s).font(.system(size: 11)).foregroundStyle(c.textMuted).padding(.top, 6)
            } else {
                Text("blocked · \(v.optString("reason"))")
                    .font(.system(size: 11)).foregroundStyle(c.warn).padding(.top, 6)
            }
        }
    }

    // MARK: - Memory
    @ViewBuilder
    private var MemorySection: some View {
        Section(label: "Memory")
        Text("One runtime lives for the whole process on a phone, so nothing else would run the collector at the moment a big fan-out wants it.")
            .font(.system(size: 11)).foregroundStyle(c.textFaint)
        SmallButton(label: "Collect garbage", bg: c.accent, fg: c.accentContrast) {
            Task { @MainActor in
                let ok = await RuntimeSurface(engine: store.engine).collectGarbage()
                gcNote = ok ? "Collected." : "The runtime did not expose __sofuu_gc in this build."
            }
        }
        if let gcNote {
            Text(gcNote).font(.system(size: 11)).foregroundStyle(c.textFaint).padding(.top, 4)
        }
    }

    // MARK: - Spaces
    @ViewBuilder
    private var SpacesSection: some View {
        Section(label: "Embedding spaces")
        if let sp = spaces?["info"] as? JSONObject {
            SpaceLine(label: "hash-768 (default)", dim: (sp["hash"] as? JSONObject)?.optInt("dimension") ?? 0, available: true)
            let sem = sp["semantic"] as? JSONObject
            SpaceLine(label: "semantic", dim: sem?.optInt("dimension") ?? 0, available: (sem?["available"] as? Bool) ?? false)
            if let v2 = spaces?["v2"] as? JSONObject {
                SpaceLine(label: "semantic v2", dim: v2.optInt("dimension"), available: (v2["available"] as? Bool) ?? false)
            }
        } else {
            Text("No embedding manifest yet.").font(.system(size: 12)).foregroundStyle(c.textFaint)
        }
    }

    // MARK: - Similarity
    @ViewBuilder
    private var SimilaritySection: some View {
        Section(label: "Similarity")
        TextFieldRow(placeholder: "first string", value: "") { simA = $0 }
        TextFieldRow(placeholder: "second string", value: "") { simB = $0 }
        SmallButton(label: "Compare", bg: c.accent, fg: c.accentContrast) {
            sim = RuntimeSurface(engine: store.engine).similarity(simA, simB)
        }
        if let s = sim {
            Text(String(format: "cosine %.4f · %d-dim", s.optDouble("similarity"), s.optInt("dim")))
                .font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text).padding(.top, 6)
        }
    }

    // MARK: - Local models
    @ViewBuilder
    private var LocalModelsSection: some View {
        Section(label: "On this device")
        Text("A .gguf runs entirely on the phone — no network, no API key. It must live in the app's container, because the engine opens models by absolute path.")
            .font(.system(size: 11)).foregroundStyle(c.textFaint)
        SmallButton(label: importingModel ? "Choose a file…" : "Import a model", bg: c.accent, fg: c.accentContrast) {
            importingModel = true
        }
        .disabled(importingModel)
        ForEach(store.localModelURLs, id: \.self) { u in
            let active = store.state.config?.model == u.path
            HStack(alignment: .center) {
                Button { Task { @MainActor in await store.pickLocalModel(u.path) } } label: {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(LocalModels.label(u))
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundStyle(active ? c.accent : c.text)
                        if active {
                            Text("in use — runs locally").font(.system(size: 11)).foregroundStyle(c.textFaint)
                        }
                    }
                }
                .buttonStyle(.plain)
                Spacer()
                if !active {
                    Button { store.deleteLocalModel(u.path) } label: {
                        Image(systemName: "xmark").font(.system(size: 10)).foregroundStyle(c.textFaint)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
        }
        if store.state.config?.model.hasSuffix(".gguf") == true {
            Text("Switching back to a hosted provider: pick one in Models.")
                .font(.system(size: 11)).foregroundStyle(c.textFaint).padding(.top, 4)
        }
    }

    private func runRecall() async {
        let r = await RuntimeSurface(engine: store.engine).brainRecall(recall, k: 8)
        guard (r?["ok"] as? Bool) == true, let arr = r?["hits"] as? [JSONObject] else {
            hits = []
            return
        }
        hits = arr
    }
}

// ── Shared bits ───────────────────────────────────────────────────

private struct SpaceLine: View {
    let label: String
    let dim: Int
    let available: Bool
    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack {
            Text(label).font(.system(size: 12, design: .monospaced))
                .foregroundStyle(available ? c.text : c.err)
            Spacer()
            Text(available ? "\(dim)-dim" : "artifact unavailable")
                .font(.system(size: 11))
                .foregroundStyle(available ? c.textFaint : c.err)
        }
        .padding(.vertical, 3)
    }
}

private struct SmallButton: View {
    let label: String
    let bg: Color
    let fg: Color
    var action: () -> Void

    var body: some View {
        Button { action() } label: {
            Text(label).font(.system(size: 12)).foregroundStyle(fg)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(bg, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}
