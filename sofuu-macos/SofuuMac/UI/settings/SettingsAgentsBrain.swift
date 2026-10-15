import SwiftUI

// MARK: - Agents (superset pane, ported from the iOS app)

/// Registered agents, and the form to add one from the desktop.
///
/// `define` is the runtime's own registry — an agent added here is a
/// first-class entry that the chat turn, `/agents` and `delegate` all see,
/// not a local preset.
struct AgentsPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c
    @State private var agents: [JSONObject] = []
    @State private var note: String?

    @State private var name = "reviewer"
    // `value:` inputs commit on Enter; these mirrors keep the fields live.
    @State private var nameField = "reviewer"
    @State private var system = "You are a careful code reviewer. Report only what you can point at."
    @State private var systemField = "You are a careful code reviewer. Report only what you can point at."
    @State private var maxSteps = "24"
    @State private var maxStepsField = "24"
    @State private var withWeb = true
    @State private var withCode = true

    @State private var task = ""
    @State private var answer: JSONObject?
    @State private var running = false

    var body: some View {
        VStack(spacing: 14) {
            SectionCard(label: "REGISTERED") {
                if agents.isEmpty {
                    Text("No agents defined yet. Add one below.")
                        .font(.system(size: 12)).foregroundStyle(c.textFaint)
                } else {
                    ForEach(Array(agents.enumerated()), id: \.offset) { _, a in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(a.optString("name"))
                                .font(.system(size: 13, design: .monospaced)).foregroundStyle(c.text)
                            Text(describe(a))
                                .font(.system(size: 11)).foregroundStyle(c.textFaint)
                        }
                        .padding(.vertical, 3)
                    }
                }
            }

            SectionCard(label: "DEFINE AN AGENT",
                        footer: "web = web_search + web_open · code = the 8 built-ins (read/write/edit/grep/glob/dir/bash/todo)") {
                TextFieldRow(placeholder: "reviewer", value: nameField) { name = $0 }
                TextFieldRow(placeholder: "what this agent is for", value: systemField) { system = $0 }
                HStack(spacing: 12) {
                    StackRow(label: "Step budget", controlWidth: 90) {
                        TextFieldRow(placeholder: "24", value: maxStepsField) { maxSteps = $0 }
                    }
                    ToggleRow(label: "Web tools", on: withWeb) { withWeb.toggle() }
                    ToggleRow(label: "Coding tools", on: withCode) { withCode.toggle() }
                }
                HStack(spacing: 10) {
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
                                note = "\(r?.optString("name") ?? name) registered — /agents lists it."
                            } else {
                                note = "Could not register: \(r?.optString("error") ?? "unknown error")"
                            }
                            await reload()
                        }
                    }
                    if let note {
                        Text(note).font(.system(size: 11))
                            .foregroundStyle(note.hasPrefix("Could not") ? c.err : c.textFaint)
                    }
                }
            }

            SectionCard(label: "RUN HEADLESSLY",
                        footer: "A run goes through the same agent loop, tools, budget and brain as a chat turn — the answer is collected instead of streamed.") {
                TextFieldRow(placeholder: "a task to hand the agent", value: task) { task = $0 }
                HStack(spacing: 10) {
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
                    if let a = answer {
                        if (a["ok"] as? Bool) != true {
                            Text(a.optString("error").isEmpty ? "run failed" : a.optString("error"))
                                .font(.system(size: 12)).foregroundStyle(c.err)
                        } else {
                            let stopped = a.optString("stopped")
                            Text("\(a.optInt("steps")) steps"
                                + (stopped.isEmpty || stopped == "null" ? "" : " · stopped: \(stopped)"))
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(c.textFaint)
                        }
                    }
                }
                if let a = answer, (a["ok"] as? Bool) == true {
                    Text(a.optString("answer")).font(.system(size: 12))
                        .foregroundStyle(c.text)
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

// MARK: - Brain (superset pane, ported from the iOS app)

/// The CMA brain and the vector spaces behind it — what `/remember`, `/why`
/// and `/brain` sit on in the CLI, made visible. Every call routes through
/// the ONE cached brain handle on the engine thread.
struct BrainPane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var info: JSONObject?
    @State private var spaces: JSONObject?
    @State private var ml: JSONObject?
    @State private var brainErr: String?
    @State private var reload = 0

    @State private var recall = ""
    @State private var hits: [JSONObject] = []
    @State private var decayArmed = false
    @State private var labelNote: String?

    @State private var simA = "the cat sat down"
    @State private var simAField = "the cat sat down"
    @State private var simB = "a feline rested"
    @State private var simBField = "a feline rested"
    @State private var sim: JSONObject?

    @State private var gcNote: String?
    @State private var probeTool = "read_file"
    @State private var probeToolField = "read_file"
    @State private var probeArgs = #"{"path":"README.md"}"#
    @State private var probeArgsField = #"{"path":"README.md"}"#
    @State private var verdict: JSONObject?

    @State private var showImporter = false

    var body: some View {
        VStack(spacing: 14) {
            storedSection
            searchSection
            contextGatesSection
            gateCheckSection
            gcSection
            spacesSection
            similaritySection
            localModelsSection
        }
        .task(id: reload) { await load() }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            showImporter = false
            if case .success(let urls) = result, let u = urls.first {
                Task { @MainActor in await store.importLocalModel(u) }
            }
        }
    }

    private func load() async {
        let s = RuntimeSurface(engine: store.engine)
        let (i, sp, m) = await (s.brainInfo(), s.spaces(), s.mlState())
        info = i
        spaces = sp
        ml = m
        brainErr = (i?["ok"] as? Bool) == true ? nil : (i?.optString("error"))
    }

    // MARK: Stored
    private var storedSection: some View {
        SectionCard(label: "STORED") {
            if let brainErr {
                Text(brainErr).font(.system(size: 12)).foregroundStyle(c.err)
            } else if let i = info {
                Text("\(i.optInt("count")) memories")
                    .font(.system(size: 18, design: .monospaced)).foregroundStyle(c.text)
                Text("space \(i.optString("backend")) · \(i.optInt("dim"))-dim"
                    + ((i["semantic"] as? Bool) == true ? " · semantic channel on" : ""))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(c.textFaint)
            } else {
                Text("Reading the brain…").font(.system(size: 12)).foregroundStyle(c.textFaint)
            }

            HStack(spacing: 8) {
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
            if decayArmed {
                Text("Decay weakens and prunes old memories. Click again to run it.")
                    .font(.system(size: 10)).foregroundStyle(c.warn)
            }
        }
    }

    // MARK: Search
    private var searchSection: some View {
        SectionCard(label: "SEARCH") {
            TextFieldRow(placeholder: "a fact to recall", value: recall) { recall = $0 }
            HStack(spacing: 10) {
                SmallButton(label: "Recall", bg: c.accent, fg: c.accentContrast) {
                    Task { await runRecall() }
                }
                if hits.isEmpty {
                    Text("Scored nearest memories — forget with the ✕.")
                        .font(.system(size: 10)).foregroundStyle(c.textFaint)
                }
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
                .padding(.vertical, 2)
            }
        }
    }

    // MARK: Context gates
    private var contextGatesSection: some View {
        SectionCard(label: "CONTEXT GATES") {
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
                HStack(spacing: 8) {
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
                if let labelNote {
                    Text(labelNote).font(.system(size: 10))
                        .foregroundStyle(labelNote.hasPrefix("Label recorded") ? c.textFaint : c.warn)
                }
            } else if let m = ml {
                Text(m.optString("error").isEmpty ? "The sofuu.ml layer is not loaded in this build." : m.optString("error"))
                    .font(.system(size: 12)).foregroundStyle(c.err)
            } else {
                Text("The sofuu.ml layer is not loaded in this build.")
                    .font(.system(size: 12)).foregroundStyle(c.textFaint)
            }
        }
    }

    // MARK: Gate check
    private var gateCheckSection: some View {
        SectionCard(label: "GATE CHECK",
                    footer: "Ask what the gates would say about a call before the agent makes it.") {
            TextFieldRow(placeholder: "read_file", value: probeToolField) { probeTool = $0 }
            TextFieldRow(placeholder: #"{"path":"README.md"}"#, value: probeArgsField) { probeArgs = $0 }
            HStack(spacing: 10) {
                SmallButton(label: "Check", bg: c.accent, fg: c.accentContrast) {
                    Task { @MainActor in
                        var args: JSONObject = [:]
                        if let d = probeArgs.data(using: .utf8),
                           let p = try? JSONSerialization.jsonObject(with: d) as? JSONObject { args = p }
                        verdict = await RuntimeSurface(engine: store.engine).mlCheck(tool: probeTool, args: args)
                    }
                }
                if let v = verdict {
                    if (v["ok"] as? Bool) == true {
                        let nudge = v.optString("nudge")
                        Text(String(format: "pass · %@ · %.3f", v.optString("source"), v.optDouble("score"))
                            + (nudge.isEmpty || nudge == "null" ? "" : " · nudge: \(nudge)"))
                            .font(.system(size: 11)).foregroundStyle(c.textMuted)
                    } else {
                        Text("blocked · \(v.optString("reason"))")
                            .font(.system(size: 11)).foregroundStyle(c.warn)
                    }
                }
            }
        }
    }

    // MARK: GC
    private var gcSection: some View {
        SectionCard(label: "MEMORY",
                    footer: "One runtime lives for the whole process — nothing else would run the collector when a big fan-out wants it.") {
            HStack(spacing: 10) {
                SmallButton(label: "Collect garbage", bg: c.accent, fg: c.accentContrast) {
                    Task { @MainActor in
                        let ok = await RuntimeSurface(engine: store.engine).collectGarbage()
                        gcNote = ok ? "Collected." : "The runtime did not expose __sofuu_gc in this build."
                    }
                }
                if let gcNote {
                    Text(gcNote).font(.system(size: 11)).foregroundStyle(c.textFaint)
                }
            }
        }
    }

    // MARK: Spaces
    private var spacesSection: some View {
        SectionCard(label: "EMBEDDING SPACES") {
            if let sp = spaces?["info"] as? JSONObject {
                spaceLine("hash-768 (default)", dim: (sp["hash"] as? JSONObject)?.optInt("dimension") ?? 0, available: true)
                let sem = sp["semantic"] as? JSONObject
                spaceLine("semantic", dim: sem?.optInt("dimension") ?? 0, available: (sem?["available"] as? Bool) ?? false)
                if let v2 = spaces?["v2"] as? JSONObject {
                    spaceLine("semantic v2", dim: v2.optInt("dimension"), available: (v2["available"] as? Bool) ?? false)
                }
            } else {
                Text("No embedding manifest yet.").font(.system(size: 12)).foregroundStyle(c.textFaint)
            }
        }
    }

    private func spaceLine(_ label: String, dim: Int, available: Bool) -> some View {
        HStack {
            Text(label).font(.system(size: 12, design: .monospaced))
                .foregroundStyle(available ? c.text : c.err)
            Spacer()
            Text(available ? "\(dim)-dim" : "artifact unavailable")
                .font(.system(size: 11))
                .foregroundStyle(available ? c.textFaint : c.err)
        }
    }

    // MARK: Similarity
    private var similaritySection: some View {
        SectionCard(label: "SIMILARITY") {
            TextFieldRow(placeholder: "first string", value: simAField) { simA = $0 }
            TextFieldRow(placeholder: "second string", value: simBField) { simB = $0 }
            HStack(spacing: 10) {
                SmallButton(label: "Compare", bg: c.accent, fg: c.accentContrast) {
                    sim = RuntimeSurface(engine: store.engine).similarity(simA, simB)
                }
                if let s = sim {
                    Text(String(format: "cosine %.4f · %d-dim", s.optDouble("similarity"), s.optInt("dim")))
                        .font(.system(size: 12, design: .monospaced)).foregroundStyle(c.text)
                }
            }
        }
    }

    // MARK: Local models
    private var localModelsSection: some View {
        SectionCard(label: "ON THIS MAC",
                    footer: "A .gguf runs entirely on this machine — no network, no API key. Imports land in ~/.sofuu/models.") {
            SmallButton(label: "Import a model…", bg: c.accent, fg: c.accentContrast) {
                showImporter = true
            }
            ForEach(store.localModelURLs, id: \.path) { u in
                let active = store.state.config?.model == u.path
                HStack(alignment: .center) {
                    Button { Task { @MainActor in await store.pickLocalModel(u.path) } } label: {
                        HStack(spacing: 8) {
                            Text(LocalModels.label(u))
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(active ? c.accent : c.text)
                            if active {
                                Text("in use — runs locally").font(.system(size: 10)).foregroundStyle(c.textFaint)
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
                .padding(.vertical, 2)
            }
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
