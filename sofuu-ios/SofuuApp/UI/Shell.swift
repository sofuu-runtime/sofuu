import SwiftUI
import UniformTypeIdentifiers

/// The app shell — the iOS mapping of the desktop's Sidebar/TopBar/App
/// composition, ported from the Android app's `Shell.kt`.
///
/// Layout strategy — the ONE place iOS genuinely differs from Android:
///
///   * **Compact width** (iPhone, iPad in a narrow split): the workspace
///     drawer is a slide-over over a dimmed veil, byte-for-byte the Android
///     drawer (82 % width, drag-to-dismiss, veil tap to close).
///   * **Regular width** (iPad full-screen, iPhone landscape on Max-class
///     devices): the same drawer content is pinned as a permanent sidebar.
///     A slide-over on a 12.9" screen wastes the entire point of the size
///     class; the rows, actions and order are identical, only the chrome
///     changes and the hamburger disappears.
///
/// Everything below the top bar — banner, preview bar, chat, approvals,
/// composer — is the same view tree on both.
struct Shell: View {
    @ObservedObject var store: AppStore

    @Environment(\.horizontalSizeClass) private var hSize
    @Environment(\.sofuuColors) private var c

    @State private var historyOpen = false
    @State private var importerOpen = false

    private var regular: Bool { hSize == .regular }

    var body: some View {
        ZStack {
            c.bgApp.ignoresSafeArea()
            if store.state.cloudBg { CloudBackground().ignoresSafeArea() }

            if regular { regularLayout } else { compactLayout }

            overlays
        }
        .fileImporter(
            isPresented: $importerOpen,
            allowedContentTypes: [.item],
            allowsMultipleSelection: true,
            onCompletion: handleImport)
    }

    // ── Layouts ──────────────────────────────────────────────────────

    private var compactLayout: some View {
        VStack(spacing: 0) {
            TopBar(store: store, historyOpen: $historyOpen, showMenu: true)
            chatArea
            composerArea
        }
    }

    private var regularLayout: some View {
        HStack(spacing: 0) {
            WorkspaceSidebar(store: store, onClose: nil)
                .frame(width: 300)
                .overlay(alignment: .trailing) {
                    Rectangle().fill(c.separator).frame(width: 1)
                }
            VStack(spacing: 0) {
                TopBar(store: store, historyOpen: $historyOpen, showMenu: false)
                chatArea
                composerArea
            }
        }
    }

    // ── Chat column ──────────────────────────────────────────────────

    private var chatArea: some View {
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
                        preview: store.state.preview)
                }
            }

            // Approval float — rides the bottom of the chat area, exactly
            // where the Android overlay lands.
            if let approval = store.state.approvals.first {
                ApprovalSheet(approval: approval) { allow, always in
                    store.approve(approval.id, allow: allow, always: always)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeOut(duration: 0.18), value: store.state.approvals.count)
    }

    @ViewBuilder private var composerArea: some View {
        if store.state.preview == nil {
            Composer(
                streaming: store.state.streaming,
                mode: store.state.mode,
                permissions: store.state.permissions,
                config: store.state.config,
                ctx: store.state.ctx,
                images: store.state.pickedImages,
                pendingMentions: store.state.pendingMentions,
                onSubmit: { text, images in
                    store.submit(text, images: images)
                    store.clearPickedImages()
                },
                onCancel: { store.cancel() },
                onModeChange: { store.setMode($0) },
                onPermissionsChange: { store.setPermissions($0) },
                onConfigPatch: { store.patchConfig($0) },
                onOpenSettings: { tab in store.openSettings(true, tab: tab) },
                onAttach: { importerOpen = true },
                onRemoveImage: { store.removePickedImage($0) },
                onMentionsConsumed: { store.consumeMentions() },
                onSheet: { store.openSheet($0.rawValue) })
        }
    }

    // ── Overlays ─────────────────────────────────────────────────────

    @ViewBuilder private var overlays: some View {
        if !regular && store.state.drawerOpen {
            WorkspaceDrawer(store: store) { store.setDrawer(false) }
                .transition(.move(edge: .leading))
        }
        if historyOpen {
            HistorySheet(store: store) { historyOpen = false }
        }
        if store.state.settingsOpen {
            SettingsScreen(store: store) { store.openSettings(false) }
                .transition(.opacity)
        }
        if let which = store.state.sheet {
            ComposerSheets(store: store, which: which) { store.closeSheet() }
                .transition(.opacity)
        }
    }

    // ── File picking (the SAF picker, iOS-side) ──────────────────────
    //
    // The Android app funnels every pick through one handler: images become
    // `data:` URLs that ride the next turn as attachments, everything else
    // is copied into the workspace and spliced in as an `@mention` (chat.js
    // expands those). Same contract here, with two iOS-specific guards:
    //
    //   * security-scoped access around the read (the Android SAF
    //     equivalent — the URL is only readable inside the scope), and
    //   * a collision-proof destination, so picking a file named like an
    //     existing workspace file can never clobber it.
    private func handleImport(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result else { return }
        var images: [String] = []
        var mentions: [String] = []

        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }

            let name = url.lastPathComponent
            guard !name.isEmpty, name != ".", name != "..",
                  !name.contains("/"), !name.contains("\\"), !name.hasPrefix(".") else { continue }

            let ext = url.pathExtension.lowercased()
            if shellImageExts.contains(ext),
               let data = try? Data(contentsOf: url), data.count <= 12 << 20 {
                images.append("data:\(shellMime(ext));base64,\(data.base64EncodedString())")
                continue
            }
            guard let dir = store.state.projectDir,
                  let dest = uniqueDestination(dir, name),
                  (try? FileManager.default.copyItem(at: url, to: dest)) != nil else { continue }
            mentions.append(dest.lastPathComponent)
        }

        store.attachPicked(images: images, mentions: mentions)
    }

    private func uniqueDestination(_ dir: String, _ name: String) -> URL? {
        let base = URL(fileURLWithPath: dir, isDirectory: true)
        let direct = base.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: direct.path) { return direct }
        let stem = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        for n in 2...999 {
            let candidate = base.appendingPathComponent(ext.isEmpty ? "\(stem)-\(n)" : "\(stem)-\(n).\(ext)")
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }
}

// MARK: - Top bar

/// hamburger (compact only) + session pills strip + new-chat + overflow.
private struct TopBar: View {
    @ObservedObject var store: AppStore
    @Binding var historyOpen: Bool
    let showMenu: Bool

    @Environment(\.sofuuColors) private var c

    /// Pills follow creation order (desktop: task || "Chat N"), newest
    /// first, capped at 6 — the overflow lives in the history sheet.
    private var pills: [String?] {
        let ordered = store.state.sessions.sorted { $0.startedAt < $1.startedAt }
        if ordered.isEmpty { return [nil] }
        return ordered.reversed().prefix(6).map { Optional($0.id) }
    }

    var body: some View {
        HStack(spacing: 0) {
            if showMenu {
                BarIcon(icon: Icons2.menu, description: "Workspaces") { store.setDrawer(true) }
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(pills, id: \.self) { sid in
                        pill(sid)
                    }
                }
                .padding(.horizontal, 4)
            }
            .frame(maxWidth: .infinity)

            BarIcon(icon: Icons2.plus, description: "New chat") { store.newChat() }
            BarIcon(icon: Icons2.moreVert, description: "More") { historyOpen = true }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
    }

    private func pill(_ sid: String?) -> some View {
        let ordered = store.state.sessions.sorted { $0.startedAt < $1.startedAt }
        let active = sid == store.state.activeSessionId
        let session = ordered.first { $0.id == sid }
        let index = session.flatMap { s in ordered.firstIndex { $0.id == s.id } } ?? 0
        let label = sid == nil ? "New chat" : (session?.labelOf(index) ?? "Chat")

        return HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 13))
                .foregroundStyle(active ? c.text : c.textMuted)
                .lineLimit(1)
            if active, let sid {
                Button { store.deleteSession(sid) } label: {
                    Image(systemName: Icons2.close)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(c.textFaint)
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Delete chat")
            }
        }
        .frame(minHeight: 36)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(active ? c.bgCardHover : Color.clear, in: Capsule())
        .contentShape(Capsule())
        .onTapGesture {
            if sid == nil { store.newChat() }
            else if let sid { store.setActiveSession(sid) }
        }
    }
}

/// A top-bar icon button: one target, one glyph size — the same rule the
/// composer's bar uses, so both bars read at one scale.
private struct BarIcon: View {
    let icon: String
    let description: String
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            Image(systemName: icon)
                .font(.system(size: 17))
                .foregroundStyle(c.textMuted)
                .frame(width: 40, height: 40)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(description)
    }
}

// MARK: - Small pieces

private struct Banner: View {
    let text: String
    let isError: Bool

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Text(text)
            .font(.system(size: 13))
            .foregroundStyle(isError ? c.err : c.text)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(isError ? c.errBg : c.bgCard)
    }
}

private struct HomePrompt: View {
    let subtitle: String?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(spacing: 6) {
            Text("What should we work on?")
                .font(.system(size: 22))
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

private struct PreviewBar: View {
    let label: String
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: Icons2.clock)
                .font(.system(size: 13))
                .foregroundStyle(c.accent)
            Text("Read-only · \(label)")
                .font(.system(size: 13))
                .foregroundStyle(c.textMuted)
                .lineLimit(1)
            Spacer(minLength: 8)
            Button(action: onClose) {
                Image(systemName: Icons2.close)
                    .font(.system(size: 14))
                    .foregroundStyle(c.textFaint)
                    .frame(width: 32, height: 32)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close preview")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(c.bgCard)
    }
}

// MARK: - Workspaces (drawer on compact, pinned sidebar on regular)

private struct WorkspaceSidebar: View {
    @ObservedObject var store: AppStore
    /// nil = pinned sidebar — there is nothing to close.
    let onClose: (() -> Void)?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Workspaces")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(c.text)
                Spacer(minLength: 0)
                Button {
                    onClose?()
                    store.openSettings(true)
                } label: {
                    Image(systemName: Icons2.gear)
                        .font(.system(size: 16))
                        .foregroundStyle(c.textMuted)
                        .frame(width: 36, height: 36)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Settings")
            }
            .padding(.bottom, 8)

            Button {
                onClose?()
                store.addDefaultWorkspace()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: Icons2.plus)
                        .font(.system(size: 15))
                        .foregroundStyle(c.accent)
                    Text("Add workspace")
                        .font(.system(size: 14))
                        .foregroundStyle(c.accent)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.bottom, 8)

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    if store.state.workspaces.isEmpty {
                        Text("No open workspaces")
                            .font(.system(size: 13))
                            .foregroundStyle(c.textFaint)
                            .padding(12)
                    }
                    ForEach(store.state.workspaces, id: \.self) { ws in
                        row(ws)
                    }
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(c.bgSidebar)
    }

    private func row(_ ws: String) -> some View {
        let active = ws == store.state.projectDir
        let name = (ws as NSString).lastPathComponent
        return HStack(spacing: 8) {
            Image(systemName: Icons2.folder)
                .font(.system(size: 14))
                .foregroundStyle(active ? c.accent : c.textFaint)
            Text(name.isEmpty ? ws : name)
                .font(.system(size: 14))
                .foregroundStyle(c.text)
                .lineLimit(1)
            Spacer(minLength: 0)
            if store.state.workspaces.count > 1 {
                Button { store.removeWorkspace(ws) } label: {
                    Image(systemName: Icons2.close)
                        .font(.system(size: 12))
                        .foregroundStyle(c.textFaint)
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove workspace")
            }
        }
        .frame(minHeight: 44)
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(active ? c.bgSidebarActive : Color.clear, in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onTapGesture {
            onClose?()
            store.openWorkspace(ws)
        }
    }
}

private struct WorkspaceDrawer: View {
    @ObservedObject var store: AppStore
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c
    /// Live horizontal drag, clamped to ≤ 0 so the panel can only slide out.
    @GestureState private var dragX: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                c.backdropVeil
                    .ignoresSafeArea()
                    .onTapGesture { onClose() }

                WorkspaceSidebar(store: store, onClose: onClose)
                    .frame(width: geo.size.width * 0.82)
                    .offset(x: min(0, dragX))
                    .gesture(
                        DragGesture()
                            .updating($dragX) { value, state, _ in
                                state = min(0, value.translation.width)
                            }
                            .onEnded { value in
                                if value.translation.width < -60 { onClose() }
                            })
            }
        }
        .ignoresSafeArea(edges: .bottom)
    }
}

// MARK: - History sheet (the desktop's clock popover)

private struct HistorySheet: View {
    @ObservedObject var store: AppStore
    let onClose: () -> Void

    @Environment(\.sofuuColors) private var c
    @State private var clearArmed = false

    /// Numbering is by creation order, but the list reads newest-first.
    private var ordered: [SessionInfo] { store.state.sessions.sorted { $0.startedAt < $1.startedAt } }
    private var byRecency: [SessionInfo] { store.state.sessions.sorted { $0.lastSeen > $1.lastSeen } }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            c.backdropVeil
                .ignoresSafeArea()
                .onTapGesture { onClose() }

            VStack(alignment: .leading, spacing: 0) {
                Text("Chat history")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(c.textMuted)
                    .padding(8)

                if store.state.sessions.isEmpty {
                    Text("No chats yet")
                        .font(.system(size: 13))
                        .foregroundStyle(c.textFaint)
                        .padding(12)
                }

                ForEach(byRecency) { s in
                    let index = ordered.firstIndex { $0.id == s.id } ?? 0
                    Button {
                        onClose()
                        store.openPreview(s)
                    } label: {
                        HStack(spacing: 8) {
                            Text(s.labelOf(index))
                                .font(.system(size: 13))
                                .foregroundStyle(c.text)
                                .lineLimit(1)
                            Spacer(minLength: 8)
                            Text(ageOf(s.lastSeen))
                                .font(.system(size: 11))
                                .foregroundStyle(c.textFaint)
                        }
                        .frame(minHeight: 44)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }

                if !store.state.sessions.isEmpty {
                    Rectangle()
                        .fill(c.separator)
                        .frame(height: 1)
                        .padding(.top, 4)
                    Button {
                        if clearArmed {
                            onClose()
                            store.clearSessions()
                        } else {
                            clearArmed = true
                        }
                    } label: {
                        Text(clearArmed ? "Tap again to confirm" : "Delete all chats in workspace")
                            .font(.system(size: 12))
                            .foregroundStyle(c.errStrong)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .frame(width: 280)
            .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 12))
            .padding(8)
            .padding(.top, 52)
            .padding(.trailing, 12)
        }
        .transition(.opacity)
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

// MARK: - Approval sheet

private struct ApprovalSheet: View {
    let approval: Approval
    let onResolve: (_ allow: Bool, _ always: Bool) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Tool approval")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(c.accent)
            Text(approval.tool)
                .font(.system(size: 14))
                .foregroundStyle(c.text)
                .padding(.top, 4)

            // write/edit approvals get a real diff (desktop parity);
            // anything else shows the plain args.
            let args = approval.args
            if approval.tool == "edit_file", args.has("old_string"), args.has("new_string") {
                Text(args.optString("path"))
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                    .padding(.top, 2)
                DiffView(oldText: args.optString("old_string"), newText: args.optString("new_string"))
                    .padding(.top, 6)
            } else if approval.tool == "write_file", args.has("content") {
                Text(args.optString("path"))
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                    .padding(.top, 2)
                DiffView(oldText: "", newText: args.optString("content"))
                    .padding(.top, 6)
            } else {
                ScrollView {
                    Text(String(jsonEncode(args).prefix(600)))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(c.textMuted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 140)
                .padding(.top, 4)
            }

            HStack(spacing: 8) {
                ApprovalButton(label: "Deny", bg: c.bgCardHover, fg: c.text) { onResolve(false, false) }
                ApprovalButton(label: "Always", bg: c.bgCardHover, fg: c.text) { onResolve(true, true) }
                ApprovalButton(label: "Allow once", bg: c.accent, fg: .white) { onResolve(true, false) }
            }
            .padding(.top, 10)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 14))
        .padding(12)
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
                .font(.system(size: 13))
                .foregroundStyle(fg)
                .padding(.horizontal, 16)
                .padding(.vertical, 11)
                .background(bg, in: Capsule())
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Picker helpers

private let shellImageExts: Set<String> = ["png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "bmp", "tiff"]

private func shellMime(_ ext: String) -> String {
    switch ext {
    case "jpg", "jpeg": return "image/jpeg"
    case "gif": return "image/gif"
    case "webp": return "image/webp"
    case "heic", "heif": return "image/heic"
    case "bmp": return "image/bmp"
    case "tiff": return "image/tiff"
    default: return "image/png"
    }
}
