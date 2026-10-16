import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The app shell — the desktop mapping: workspace sidebar (vibrancy) on the
/// left, session-pills strip + chat + composer on the right, approval
/// float over the composer, history popover, settings overlay. The iOS
/// Shell's donor tree with the desktop chrome (hidden titlebar, traffic
/// light gutter, ⌘B collapse).
struct Shell: View {
    @ObservedObject var store: AppStore

    @Environment(\.sofuuColors) private var c
    @State private var historyPopover = false
    @State private var removeArm: String?

    var body: some View {
        // Settings is a full dedicated PAGE: it takes the whole window
        // (back button returns to the main UI) rather than a modal.
        ZStack {
            // Procedural cloud backdrop (the desktop's CloudBackground) —
            // behind everything, gated by the Appearance toggle.
            if store.state.cloudBg {
                CloudBackground().ignoresSafeArea()
            }
            if store.state.settingsOpen {
                SettingsScreen(store: store) { store.openSettings(false) }
                    .transition(.opacity)
            } else {
                mainLayout
                    .overlay(alignment: .top) {
                        // Sheets ride over whichever layout is up (chat or
                        // IDE) — settings is a page, not an overlay.
                        if let which = store.state.sheet {
                            ComposerSheets(store: store, which: which) { store.closeSheet() }
                        }
                    }
            }
        }
        .background(c.bgApp.ignoresSafeArea())
        // Edge to edge: the shell flows under the (invisible) title bar and
        // to the window borders — no dead band at the top. The sidebar
        // carries the traffic-light gutter itself.
        .ignoresSafeArea()
        .background(WindowChrome())
        .animation(.easeOut(duration: 0.15), value: store.state.sidebarCollapsed)
        .animation(.easeOut(duration: 0.15), value: store.state.settingsOpen)
    }

    private var mainLayout: some View {
        HStack(spacing: 0) {
            if store.state.ideMode {
                IdeLayout(store: store)
            } else if !store.state.sidebarCollapsed {
                sidebar
                    .transition(.move(edge: .leading).combined(with: .opacity))
            } else {
                collapsedRail
            }

            if !store.state.ideMode {
                Rectangle().fill(c.separator).frame(width: 1)
            }

            if !store.state.ideMode {
                mainColumn
            }
        }
    }

    // ── Sidebar (workspaces) ─────────────────────────────────────────

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Traffic-light gutter: hiddenTitleBar keeps the buttons over
            // the sidebar, so reserve their band.
            Color.clear.frame(height: 28)

            HStack {
                Text("Workspaces")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(c.textMuted)
                Spacer(minLength: 0)
                Button { addWorkspaceFolder() } label: {
                    Image(systemName: Icons2.plus)
                        .font(.system(size: 13))
                        .foregroundStyle(c.textMuted)
                        .frame(width: 26, height: 26)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Add a workspace folder")
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 8)

            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(store.state.workspaces, id: \.self) { ws in
                        workspaceRow(ws)
                    }
                    if store.state.workspaces.isEmpty {
                        Text("No open workspaces")
                            .font(.system(size: 12))
                            .foregroundStyle(c.textFaint)
                            .padding(12)
                    }
                }
                .padding(.horizontal, 8)
            }

            Divider().overlay(c.separator)
            HStack {
                Text("Sofuu")
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                Spacer()
                Button { store.openSettings(true) } label: {
                    Image(systemName: Icons2.gear)
                        .font(.system(size: 12))
                        .foregroundStyle(c.textMuted)
                        .frame(width: 26, height: 26)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Settings (⌘,)")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
        }
        .frame(width: 200)
        .background(
            ZStack {
                Vibrancy()
                c.bgSidebar.opacity(0.55)
            }
            .ignoresSafeArea()
        )
    }

    private func workspaceRow(_ ws: String) -> some View {
        let active = ws == store.state.projectDir
        let name = (ws as NSString).lastPathComponent
        return HStack(spacing: 7) {
            Image(systemName: Icons2.folder)
                .font(.system(size: 12))
                .foregroundStyle(active ? c.accent : c.textFaint)
            Text(name.isEmpty ? ws : name)
                .font(.system(size: 13))
                .foregroundStyle(c.text)
                .lineLimit(1)
                .help(ws)
            Spacer(minLength: 2)
            if removeArm == ws {
                Text("again ✓")
                    .font(.system(size: 10))
                    .foregroundStyle(c.errStrong)
                    .onTapGesture {
                        removeArm = nil
                        store.removeWorkspace(ws)
                    }
            } else if store.state.workspaces.count > 1 {
                Image(systemName: Icons2.close)
                    .font(.system(size: 9))
                    .foregroundStyle(c.textFaint.opacity(0.7))
                    .onTapGesture {
                        removeArm = ws
                        // Two-step confirm, re-armed after 2.5 s (desktop rule).
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                            if removeArm == ws { removeArm = nil }
                        }
                    }
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 7)
        .background(active ? c.bgSidebarActive : Color.clear, in: RoundedRectangle(cornerRadius: 7))
        .contentShape(Rectangle())
        .onTapGesture {
            removeArm = nil
            store.openWorkspace(ws)
        }
    }

    /// ⌘B collapse: a slim rail so the main column still has a way back.
    /// The traffic lights draw over its empty top band.
    private var collapsedRail: some View {
        VStack {
            Color.clear.frame(height: 28)
            Button { store.toggleSidebar() } label: {
                Image(systemName: Icons2.sidebar)
                    .font(.system(size: 12))
                    .foregroundStyle(c.textMuted)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Show sidebar (⌘B)")
            Spacer()
        }
        .frame(width: 34)
    }

    // ── Main column ──────────────────────────────────────────────────

    private var mainColumn: some View {
        VStack(spacing: 0) {
            pillsBar
            Divider().overlay(c.separator)

            ZStack(alignment: .bottom) {
                VStack(spacing: 0) {
                    if let boot = store.state.bootError {
                        Banner(text: boot, isError: true)
                    }
                    if let preview = store.state.preview {
                        PreviewBar(label: preview.label) { store.closePreview() }
                    }
                    if store.state.messages.isEmpty && store.state.preview == nil {
                        HomePrompt(subtitle: store.state.projectDir.map {
                            "Workspace: \((($0 as NSString).lastPathComponent).isEmpty ? $0 : ($0 as NSString).lastPathComponent)"
                        })
                    } else {
                        ChatScreen(
                            messages: store.state.messages,
                            phase: store.state.phase,
                            streaming: store.state.streaming,
                            chatFontSize: store.state.chatFontSize,
                            preview: store.state.preview,
                            streamFrame: store.state.streamFrame,
                            liveThink: store.state.liveThink)
                    }
                }

                // Approval float — rides the composer's top edge.
                if let approval = store.state.approvals.first {
                    ApprovalCard(approval: approval) { allow, always in
                        store.approve(approval.id, allow: allow, always: always)
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 4)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .animation(.easeOut(duration: 0.18), value: store.state.approvals.count)

            if store.state.preview == nil {
                Composer(
                    streaming: store.state.streaming,
                    mode: store.state.mode,
                    permissions: store.state.permissions,
                    config: store.state.config,
                    ctx: store.state.ctx,
                    images: store.state.pickedImages,
                    onSubmit: { text, images in
                        store.submit(text, images: images)
                        store.clearPickedImages()
                    },
                    onCancel: { store.cancel() },
                    onModeChange: { store.setMode($0) },
                    onPermissionsChange: { store.setPermissions($0) },
                    onOpenSettings: { _ in },
                    onRemoveImage: { store.removePickedImage($0) },
                    onAddImages: { store.addPickedImages($0) },
                    onSheet: { store.openSheet($0.rawValue) })
            }
        }
    }

    // ── Session pills strip (desktop TopBar) ─────────────────────────

    private var pillsBar: some View {
        HStack(spacing: 0) {
            if let preview = store.state.preview {
                HStack(spacing: 6) {
                    Image(systemName: Icons2.clock)
                        .font(.system(size: 11))
                        .foregroundStyle(c.accent)
                    Text("Read-only · \(preview.label)")
                        .font(.system(size: 12))
                        .foregroundStyle(c.textMuted)
                        .lineLimit(1)
                }
                .padding(.horizontal, 12)
                .frame(minHeight: 36)
                .background(c.bgCardHover, in: Capsule())
                .padding(.horizontal, 6)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(pillIds, id: \.self) { sid in
                            pill(sid)
                        }
                    }
                    .padding(.horizontal, 6)
                }
            }

            Spacer(minLength: 8)

            Button { store.newChat() } label: {
                Image(systemName: Icons2.plus)
                    .font(.system(size: 14))
                    .foregroundStyle(c.textMuted)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("New chat (⌘N)")

            Button { historyPopover = true } label: {
                Image(systemName: Icons2.clock)
                    .font(.system(size: 13))
                    .foregroundStyle(c.textMuted)
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Chat history")
            .popover(isPresented: $historyPopover, arrowEdge: .bottom) {
                HistoryList(
                    store: store,
                    onPick: { historyPopover = false; store.openPreview($0) },
                    onResume: { historyPopover = false; store.setActiveSession($0.id) })
            }
        }
        .padding(.leading, store.state.sidebarCollapsed ? 82 : 8)
        .padding(.trailing, 8)
        .padding(.vertical, 5)
    }

    /// Newest-7 pills by last_seen, with the ACTIVE pill always shown
    /// (swaps in for the oldest shown — desktop TopBar rule).
    private var pillIds: [String?] {
        var ordered = store.state.sessions.sorted { $0.lastSeen > $1.lastSeen }
        if let cur = store.state.activeSessionId,
           !ordered.prefix(7).contains(where: { $0.id == cur }) {
            if let i = ordered.firstIndex(where: { $0.id == cur }) {
                let s = ordered.remove(at: i)
                ordered = [s] + ordered
            }
        }
        if ordered.isEmpty { return [nil] }
        return ordered.prefix(7).map { Optional($0.id) }
    }

    private func pill(_ sid: String?) -> some View {
        let active = sid == store.state.activeSessionId
        let session = store.state.sessions.first { $0.id == sid }
        let creationOrder = store.state.sessions.sorted { $0.startedAt < $1.startedAt }
        let index = session.flatMap { s in creationOrder.firstIndex { $0.id == s.id } } ?? 0
        let label = sid == nil ? "New chat" : (session?.labelOf(index) ?? "Chat")

        return HStack(spacing: 6) {
            if active {
                Circle().fill(c.accent).frame(width: 5, height: 5)
            }
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(active ? c.text : c.textMuted)
                .lineLimit(1)
            if active, let sid {
                Image(systemName: Icons2.close)
                    .font(.system(size: 8, weight: .medium))
                    .foregroundStyle(c.textFaint)
                    .onTapGesture {
                        if removeArm == sid || labelIsUntitled(label) {
                            store.deleteSession(sid)
                            removeArm = nil
                        } else {
                            removeArm = sid
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                                if removeArm == sid { removeArm = nil }
                            }
                        }
                    }
                    .help("Delete chat")
            }
        }
        .padding(.horizontal, 11)
        .frame(minHeight: 30)
        .background(active ? c.bgCardHover : Color.clear, in: Capsule())
        .contentShape(Capsule())
        .onTapGesture {
            if sid == nil { store.newChat() }
            else if let sid { store.setActiveSession(sid) }
        }
    }

    /// Untitled chats delete without the arm (desktop TopBar rule).
    private func labelIsUntitled(_ label: String) -> Bool {
        label.hasPrefix("Chat ")
    }

    private func addWorkspaceFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.message = "Add a project folder"
        if panel.runModal() == .OK, let url = panel.urls.first {
            store.addWorkspace(url.path)
        }
    }
}

// MARK: - Small pieces

private struct Banner: View {
    let text: String
    let isError: Bool

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Text(text)
            .font(.system(size: 12))
            .foregroundStyle(isError ? c.err : c.text)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12).padding(.vertical, 9)
            .background(isError ? c.errBg : c.bgCard)
    }
}

private struct PreviewBar: View {
    let label: String
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: Icons2.clock)
                .font(.system(size: 12))
                .foregroundStyle(c.accent)
            Text("Read-only · \(label)")
                .font(.system(size: 12))
                .foregroundStyle(c.textMuted)
                .lineLimit(1)
            Spacer(minLength: 8)
            Button(action: onClose) {
                Image(systemName: Icons2.close)
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(c.bgCard)
    }
}

private struct HomePrompt: View {
    let subtitle: String?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(spacing: 6) {
            Text("What should we work on?")
                .font(.system(size: 24, weight: .medium))
                .foregroundStyle(c.text)
            if let subtitle {
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundStyle(c.textFaint)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
}

// MARK: - History popover

/// The clock popover: every session with age hints; click previews it
/// read-only, double-activates.
private struct HistoryList: View {
    @ObservedObject var store: AppStore
    let onPick: (SessionInfo) -> Void
    let onResume: (SessionInfo) -> Void

    @Environment(\.sofuuColors) private var c
    @State private var clearArmed = false

    private var byRecency: [SessionInfo] { store.state.sessions.sorted { $0.lastSeen > $1.lastSeen } }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Chat history")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(c.textMuted)
                .padding(10)

            if store.state.sessions.isEmpty {
                Text("No chats yet")
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .padding(12)
            }

            ForEach(byRecency) { s in
                Button { onResume(s) } label: {
                    HStack(spacing: 8) {
                        Text(s.task.flatMap { $0.nilIfEmpty } ?? "Chat")
                            .font(.system(size: 12))
                            .foregroundStyle(c.text)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Text(ageOf(s.lastSeen))
                            .font(.system(size: 10))
                            .foregroundStyle(c.textFaint)
                    }
                    .padding(.horizontal, 10)
                    .frame(minHeight: 28)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .onTapGesture(count: 2) { onPick(s) }
            }

            if !store.state.sessions.isEmpty {
                Divider().overlay(c.separator).padding(.top, 4)
                Button {
                    if clearArmed {
                        store.clearSessions()
                        clearArmed = false
                    } else {
                        clearArmed = true
                    }
                } label: {
                    Text(clearArmed ? "Click again to confirm" : "Delete all chats in workspace")
                        .font(.system(size: 11))
                        .foregroundStyle(c.errStrong)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(width: 260)
        .padding(.vertical, 4)
        .background(c.bgRaised)
    }
}

private func ageOf(_ lastSeen: Int) -> String {
    let mins = max(0, (Int(Date().timeIntervalSince1970) - lastSeen) / 60)
    switch mins {
    case ..<1: return "now"
    case ..<60: return "\(mins)m ago"
    case ..<1440: return "\(mins / 60)h ago"
    default: return "\(mins / 1440)d ago"
    }
}

// MARK: - Approval card

/// Floating card for the first pending approval: diff for edit/write,
/// command preview for bash, JSON otherwise. Buttons follow the desktop's
/// ApprovalDialog: Deny / Always-allow-this-command (bash) or
/// Always-allow-this-session / Allow.
private struct ApprovalCard: View {
    let approval: Approval
    let onResolve: (_ allow: Bool, _ always: Bool) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Approve \(approval.tool)?")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(c.text)

            let args = approval.args
            if approval.tool == "edit_file", args.has("old_string"), args.has("new_string") {
                Text(args.optString("path"))
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                    .padding(.top, 3)
                DiffView(oldText: args.optString("old_string"), newText: args.optString("new_string"))
                    .padding(.top, 6)
            } else if approval.tool == "write_file", args.has("content") {
                Text(args.optString("path"))
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                    .padding(.top, 3)
                DiffView(oldText: "", newText: args.optString("content"))
                    .padding(.top, 6)
            } else if approval.tool == "bash", let cmd = args["command"] as? String {
                ScrollView {
                    Text(cmd)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(c.text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
                .padding(8)
                .background(c.monoFieldBg, in: RoundedRectangle(cornerRadius: 7))
                .padding(.top, 6)
            } else {
                ScrollView {
                    Text(String(jsonEncode(args).prefix(600)))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(c.textMuted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
                .padding(.top, 6)
            }

            HStack(spacing: 8) {
                ApprovalButton(label: "Deny", bg: c.bgCardHover, fg: c.text) { onResolve(false, false) }
                ApprovalButton(
                    label: approval.tool == "bash" ? "Always allow this command" : "Always allow this session",
                    bg: c.bgCardHover,
                    fg: c.text) { onResolve(true, true) }
                ApprovalButton(label: "Allow", bg: c.accent, fg: .white) { onResolve(true, false) }
                Spacer()
            }
            .padding(.top, 10)
        }
        .padding(14)
        .frame(maxWidth: 520, alignment: .leading)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 13))
        .overlay(RoundedRectangle(cornerRadius: 13).stroke(c.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.16), radius: 16, y: 5)
    }
}

private struct ApprovalButton: View {
    let label: String
    let bg: Color
    let fg: Color
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(fg)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(bg, in: Capsule())
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

