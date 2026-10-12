import SwiftUI

private let settingsTabs = ["general", "appearance", "models", "providers", "usage"]

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
