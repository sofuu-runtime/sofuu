import SwiftUI

/// The composer's pill popovers: model, effort, context layers, turn mode,
/// permissions. Same panes as the iOS bottom sheets, presented as a
/// floating card (a popover wants an anchor view we don't own here, and a
/// centered card matches the desktop's popover feel).
struct ComposerSheets: View {
    @ObservedObject var store: AppStore
    let which: String
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        ZStack(alignment: .bottom) {
            c.backdropVeil
                .ignoresSafeArea()
                .onTapGesture { onClose() }

            VStack(alignment: .leading, spacing: 0) {
                switch which {
                case "model": ModelSheet(store: store, onClose: onClose)
                case "effort": EffortSheet(store: store)
                case "layers": LayersSheet(store: store)
                case "mode": ModeSheet(store: store)
                case "permissions": PermissionsSheet(store: store)
                default: EmptyView()
                }
            }
            .padding(.top, 18)
            .padding(.horizontal, 16)
            .padding(.bottom, 18)
            .frame(width: 380)
            .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(c.border, lineWidth: 1))
            .shadow(color: .black.opacity(0.18), radius: 18, y: 6)
            .padding(.bottom, 96)
        }
    }
}

// MARK: - Shared rows

private struct SheetTitle: View {
    let t: String
    @Environment(\.sofuuColors) private var c
    var body: some View {
        Text(t).font(.system(size: 13)).foregroundStyle(c.textMuted).padding(.bottom, 8)
    }
}

private struct SheetRow: View {
    let label: String
    let hint: String?
    let selected: Bool
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            HStack {
                VStack(alignment: .leading, spacing: 1) {
                    Text(label).font(.system(size: 14)).foregroundStyle(c.text)
                    if let hint { Text(hint).font(.system(size: 11)).foregroundStyle(c.textFaint) }
                }
                Spacer()
                if selected { Text("✓").font(.system(size: 14)).foregroundStyle(c.accent) }
            }
            .frame(minHeight: 40)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(selected ? c.bgCardHover : .clear, in: RoundedRectangle(cornerRadius: 10))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct SheetToggleRow: View {
    let label: String
    let hint: String
    let on: Bool
    let onToggle: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onToggle) {
            HStack {
                VStack(alignment: .leading, spacing: 1) {
                    Text(label).font(.system(size: 14)).foregroundStyle(c.text)
                    Text(hint).font(.system(size: 11)).foregroundStyle(c.textFaint)
                }
                Spacer()
                ZStack(alignment: on ? .trailing : .leading) {
                    Capsule().fill(on ? c.accent : c.toggleOff).frame(width: 40, height: 22)
                    Circle().fill(.white).frame(width: 18, height: 18).padding(2)
                }
            }
            .frame(minHeight: 44)
            .padding(.horizontal, 4).padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct SheetButton: View {
    let label: String
    let bg: Color
    let fg: Color
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            Text(label).font(.system(size: 13)).foregroundStyle(fg)
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(bg, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Panes

private struct ModelSheet: View {
    @ObservedObject var store: AppStore
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c
    @State private var provider: String?
    @State private var search = ""
    @State private var manual = ""

    private var providers: [ProviderEntry] { store.state.config?.providers ?? [] }

    var body: some View {
        if provider == nil {
            providerList
        } else if let prov = provider {
            modelList(prov)
        }
    }

    @ViewBuilder private var providerList: some View {
        SheetTitle(t: "Select a provider")
        if providers.isEmpty {
            Text("No model selected — connect a provider first.")
                .font(.system(size: 13)).foregroundStyle(c.textFaint).padding(12)
            SheetRow(label: "＋ Add provider", hint: nil, selected: false) {
                store.openSettings(true, tab: "providers")
                onClose()
            }
        }
        ForEach(providers) { p in
            SheetRow(label: p.name, hint: p.model.isEmpty ? nil : p.model,
                     selected: p.name == store.state.config?.active) { provider = p.name }
        }
    }

    /// The provider's saved model first, then whatever the prefetched cache
    /// holds — de-duplicated, in that order.
    private func modelRows(_ prov: String) -> [ModelRow] {
        var out: [ModelRow] = []
        let saved = providers.first { $0.name == prov }?.model ?? ""
        if !saved.isEmpty {
            out.append(ModelRow(model: saved,
                                hint: prov == store.state.config?.active ? "current" : "saved"))
        }
        for m in store.state.modelsCache[prov]?.models ?? [] where !out.contains(where: { $0.model == m }) {
            out.append(ModelRow(model: m, hint: nil))
        }
        return out
    }

    @ViewBuilder private func modelList(_ prov: String) -> some View {
        let all = modelRows(prov)
        let saved = providers.first { $0.name == prov }?.model ?? ""
        let filtered = all.filter { search.isEmpty || $0.model.localizedCaseInsensitiveContains(search) }
        let cached = store.state.modelsCache[prov]

        SheetRow(label: "‹ Providers", hint: nil, selected: false) { provider = nil }
        SheetTitle(t: prov).padding(.top, 4)

        if all.count > 1 {
            TextField("Search models…", text: $search)
                .font(.system(size: 13))
                .textFieldStyle(.plain)
                .padding(.horizontal, 12).padding(.vertical, 10)
                .frame(minHeight: 36)
                .background(c.monoFieldBg, in: RoundedRectangle(cornerRadius: 8))
                .padding(.top, 6)
        }
        if let err = cached?.error {
            Text(err).font(.system(size: 12)).foregroundStyle(c.err).padding(.top, 8)
        }
        if let note = cached?.note, !note.isEmpty {
            Text(note).font(.system(size: 12)).foregroundStyle(c.textFaint).padding(.top, 4)
        }

        if all.isEmpty {
            Text("No model list for this provider — set a model name by hand below, or reconnect the provider.")
                .font(.system(size: 13)).foregroundStyle(c.textFaint).padding(12)
            TextField("Type a model name…", text: $manual)
                .font(.system(size: 13))
                .textFieldStyle(.plain)
                .padding(.horizontal, 12).padding(.vertical, 10)
                .frame(minHeight: 36)
                .background(c.monoFieldBg, in: RoundedRectangle(cornerRadius: 8))
            SheetButton(label: "Use this model", bg: c.accent, fg: .white) {
                let m = manual.trimmingCharacters(in: .whitespaces)
                if !m.isEmpty {
                    store.pickModel(prov, m)
                    onClose()
                }
            }
            .padding(.top, 8)
        } else {
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(filtered) { row in
                        SheetRow(label: row.model, hint: row.hint,
                                 selected: row.model == saved && prov == store.state.config?.active) {
                            store.pickModel(prov, row.model)
                            onClose()
                        }
                    }
                }
            }
            .frame(maxHeight: 300)
        }
    }
}

private struct ModelRow: Identifiable {
    let model: String
    let hint: String?
    var id: String { model }
}

private struct EffortSheet: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        SheetTitle(t: "Thinking effort")
        ForEach(["off", "low", "medium", "high", "max"], id: \.self) { e in
            let cur = store.state.config?.effort ?? ""
            let selected = cur == e || (cur.isEmpty && e == "off")
            SheetRow(label: e == "off" ? "Off" : e.prefix(1).uppercased() + e.dropFirst(),
                     hint: nil, selected: selected) {
                store.patchConfig(["effort": e == "off" ? "" : e])
            }
        }
    }
}

private struct LayersSheet: View {
    @ObservedObject var store: AppStore

    var body: some View {
        let cfg = store.state.config
        SheetTitle(t: "Context layers")
        SheetToggleRow(label: "Project memory", hint: "Recall relevant saved memories",
                  on: cfg?.brain != false) {
            store.patchConfig(["brain": !(cfg?.brain != false)])
        }
        SheetToggleRow(label: "Smart context filters", hint: "Adaptive context relevance & compaction",
                  on: cfg?.ml != false) {
            store.patchConfig(["ml": !(cfg?.ml != false)])
        }
        let rlm = cfg?.rlm ?? "off"
        SheetToggleRow(label: "Reasoning sandbox", hint: rlm, on: rlm == "on") {
            store.patchConfig(["rlm": rlm == "on" ? "off" : "on"])
        }
    }
}

private struct ModeSheet: View {
    @ObservedObject var store: AppStore

    var body: some View {
        SheetTitle(t: "Turn mode")
        SheetRow(label: "Agent", hint: "Autonomous tools & file edits",
                 selected: store.state.mode == "agent") { store.setMode("agent") }
        SheetRow(label: "Chat only", hint: "Conversation only, no tools",
                 selected: store.state.mode == "chat") { store.setMode("chat") }
    }
}

private struct PermissionsSheet: View {
    @ObservedObject var store: AppStore

    var body: some View {
        SheetTitle(t: "Tool permissions")
        SheetRow(label: "Full access", hint: "All tools run automatically",
                 selected: store.state.permissions == "full") { store.setPermissions("full") }
        SheetRow(label: "Edit only", hint: "Read files & apply edits",
                 selected: store.state.permissions == "edit") { store.setPermissions("edit") }
        SheetRow(label: "Plan mode", hint: "Plan & analyze, no changes",
                 selected: store.state.permissions == "plan") { store.setPermissions("plan") }
        SheetRow(label: "Ask me", hint: "Approve each tool execution",
                 selected: store.state.permissions == "prompt") { store.setPermissions("prompt") }
    }
}
