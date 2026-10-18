import SwiftUI
import AppKit

/// IDE mode layout: workspace tree on the left, tabbed code viewport in
/// the middle, the agent panel (chat + composer) on the right. Minimal on
/// purpose — no search field, no status/metrics bar. Both boundaries are
/// resizable drag handles; widths persist across launches.
struct IdeLayout: View {
    @ObservedObject var store: AppStore

    @Environment(\.sofuuColors) private var c
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 0) {
            FileTree(store: store)
                .frame(width: store.state.ideTreeWidth)
                .background(c.bgSidebar.ignoresSafeArea())

            VerticalSplitter(
                store: store,
                which: .tree,
                canvasAt: .trailing,
                lineColor: NSColor(c.border),
                hoverLineColor: NSColor(c.borderStrong))
                .frame(width: 11)
                .background(
                    // Depth: the raised canvas casts a soft shadow onto
                    // the panel beside it.
                    LinearGradient(
                        colors: [Color.black.opacity(0), Color.black.opacity(0.05)],
                        startPoint: .leading, endPoint: .trailing)
                )

            editorColumn
                .frame(maxWidth: .infinity)
                .background(c.bgRaised.ignoresSafeArea())

            VerticalSplitter(
                store: store,
                which: .agent,
                canvasAt: .leading,
                lineColor: NSColor(c.border),
                hoverLineColor: NSColor(c.borderStrong))
                .frame(width: 11)
                .background(
                    LinearGradient(
                        colors: [Color.black.opacity(0.05), Color.black.opacity(0)],
                        startPoint: .leading, endPoint: .trailing)
                )

            agentPanel
                .frame(width: store.state.ideAgentWidth)
                .background(c.bgSidebar.ignoresSafeArea())
        }
        .background(c.bgApp.ignoresSafeArea())
    }

    // ── Middle: tabs + code viewport ─────────────────────────────────

    private var editorColumn: some View {
        VStack(spacing: 0) {
            tabStrip
            Divider().overlay(c.separator)
            editorBody
        }
    }

    private var editorBody: some View {
        ZStack(alignment: .center) {
            c.bgRaised.ignoresSafeArea()
            if let path = store.state.activeTab, store.state.openTabs.contains(path) {
                HStack(spacing: 0) {
                    // The gutter renders even before the coordinator exists
                    // (first-pass ordering) and fills in on its first pulse.
                    LineGutter(pulse: store.ideGutter, coordinator: store.ideCoordinator)
                        .frame(width: CodeEditor.gutterWidth)
                    CodeEditor(
                        text: Binding(
                            get: { store.buffer(path) },
                            set: { store.updateBuffer(path, $0) }),
                        path: path,
                        dark: scheme == .dark,
                        store: store,
                        onEdited: { text in
                            store.updateBuffer(path, text)
                            store.scheduleAutoSave(path)
                        })
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                Text(store.state.openTabs.isEmpty
                        ? "Open a file from the tree"
                        : "Select a tab")
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var tabStrip: some View {
        HStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 1) {
                    ForEach(store.state.openTabs, id: \.self) { path in
                        EditorTab(store: store, path: path)
                    }
                }
                // Start the tabs ALIGNED WITH THE CODE TEXT (past the
                // line-number gutter) — clear of the separator.
                .padding(.leading, 52)
                .padding(.trailing, 4)
            }
            .frame(maxWidth: .infinity)

            if !store.state.openTabs.isEmpty {
                openFilesMenu
            }

            Spacer(minLength: 4)

            // Auto save: writes the buffer 0.8 s after the last edit.
            Button {
                store.setIdeAutoSave(!store.ideAutoSaveOn)
            } label: {
                Image(systemName: "arrow.down.doc")
                    .font(.system(size: 12))
                    .foregroundStyle(store.ideAutoSaveOn ? c.accent : c.textFaint)
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(store.ideAutoSaveOn ? "Auto save is ON" : "Auto save is OFF")
            .padding(.trailing, 2)

            if let path = store.state.activeTab, store.isDirty(path) {
                Button {
                    if let err = store.saveFile(path) { store.systemRow(err) }
                } label: {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 12))
                        .foregroundStyle(c.textMuted)
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Save (⌘S)")
                .padding(.trailing, 6)
            }
        }
        .frame(minHeight: 34)
    }

    /// The ▾ menu: every open file (dirty dot included) for quick
    /// switching, plus Close-all — "see and manage all the files".
    private var openFilesMenu: some View {
        Menu {
            ForEach(store.state.openTabs, id: \.self) { path in
                Button {
                    store.setActiveTab(path)
                } label: {
                    if path == store.state.activeTab {
                        Label((path as NSString).lastPathComponent, systemImage: "checkmark")
                    } else if store.isDirty(path) {
                        Label((path as NSString).lastPathComponent, systemImage: "circle.fill")
                    } else {
                        Text((path as NSString).lastPathComponent)
                    }
                }
            }
            Divider()
            Button("Close all tabs") { store.closeAllTabs() }
        } label: {
            Image(systemName: "chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(c.textMuted)
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .menuIndicator(.hidden)
        .fixedSize()
        .help("All open files")
    }

    // ── Right: agent panel ───────────────────────────────────────────

    private var agentPanel: some View {
        VStack(spacing: 0) {
            agentHeader
            Divider().overlay(c.separator)

            ZStack(alignment: .bottom) {
                VStack(spacing: 0) {
                    if let boot = store.state.bootError {
                        IdeBanner(text: boot, isError: true)
                    }
                    if store.state.messages.isEmpty {
                        VStack(spacing: 5) {
                            Text("Agent")
                                .font(.system(size: 17, weight: .medium))
                                .foregroundStyle(c.text)
                            Text("Ask it to build or change what you have open.")
                                .font(.system(size: 11))
                                .foregroundStyle(c.textFaint)
                                .multilineTextAlignment(.center)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(20)
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

                // Approval float — same card as chat mode, riding the
                // composer's top edge of the agent panel.
                if let approval = store.state.approvals.first {
                    IdeApprovalCard(approval: approval) { allow, always in
                        store.approve(approval.id, allow: allow, always: always)
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .padding(.horizontal, 10)
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
        .background(c.bgApp.ignoresSafeArea())
    }

    private var agentHeader: some View {
        HStack(spacing: 2) {
            Text("AGENT")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(c.textFaint)
            Spacer(minLength: 0)
            agentButton("plus", "New chat (⌘N)") { store.newChat() }
            agentButton("square.and.rectangle.stack", "Exit IDE mode (⌘⇧I)") { store.toggleIdeMode() }
            agentButton("gear", "Settings (⌘,)") { store.openSettings(true) }
        }
        .padding(.leading, 22)
        .padding(.trailing, 8)
        .frame(height: 34)
    }

    private func agentButton(_ icon: String, _ tip: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundStyle(c.textMuted)
                .frame(width: 26, height: 26)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(tip)
    }
}

// MARK: - Tab

/// One editor tab: file name, dirty dot, two-step close (armed ✕ shows
/// "again ✓", mirroring the sidebar's remove rule).
private struct EditorTab: View {
    @ObservedObject var store: AppStore
    let path: String

    @Environment(\.sofuuColors) private var c
    @State private var closeArm = false

    private var active: Bool { store.state.activeTab == path }
    private var dirty: Bool { store.isDirty(path) }
    private var name: String { (path as NSString).lastPathComponent }

    var body: some View {
        HStack(spacing: 6) {
            Text(name)
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(active ? c.text : c.textMuted)
                .lineLimit(1)
                .fixedSize()
            if dirty {
                Circle().fill(c.accent).frame(width: 5, height: 5)
            }
            if active {
                if closeArm {
                    Text("again ✓")
                        .font(.system(size: 9))
                        .foregroundStyle(c.errStrong)
                        .onTapGesture {
                            closeArm = false
                            store.closeTab(path, force: true)
                        }
                } else {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(c.textFaint)
                        .onTapGesture {
                            if !dirty {
                                store.closeTab(path)
                            } else {
                                closeArm = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                                    if closeArm { closeArm = false }
                                }
                            }
                        }
                }
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 26)
        .background(active ? c.bgCardHover : Color.clear, in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
        .onTapGesture {
            closeArm = false
            store.setActiveTab(path)
        }
        .help(path)
    }
}

// MARK: - Shared card banners (Shell's are private there)

private struct IdeBanner: View {
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

/// The approval card again — Shell's is file-private, so the IDE panel
/// carries its own copy of the same design (diff for edit/write, command
/// preview for bash, JSON otherwise; same button ladder).
private struct IdeApprovalCard: View {
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
                .frame(maxHeight: 90)
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
                .frame(maxHeight: 90)
                .padding(.top, 6)
            }

            HStack(spacing: 6) {
                IdeApprovalButton(label: "Deny", bg: c.bgCardHover, fg: c.text) { onResolve(false, false) }
                IdeApprovalButton(
                    label: approval.tool == "bash" ? "Always allow this command" : "Always allow this session",
                    bg: c.bgCardHover,
                    fg: c.text) { onResolve(true, true) }
                IdeApprovalButton(label: "Allow", bg: c.accent, fg: .white) { onResolve(true, false) }
                Spacer()
            }
            .padding(.top, 10)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(c.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.16), radius: 14, y: 5)
    }
}

private struct IdeApprovalButton: View {
    let label: String
    let bg: Color
    let fg: Color
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(fg)
                .padding(.horizontal, 11)
                .padding(.vertical, 7)
                .background(bg, in: Capsule())
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Splitter

/// The resizable boundary between panels: an 11pt-wide AppKit drag band
/// with a themed 1pt hairline centered in it (content on both sides
/// keeps its distance from the line), brightening on hover, resize
/// cursor, plus a soft gradient where the raised canvas casts its shadow
/// onto the adjacent panel.
///
/// AppKit, not a SwiftUI DragGesture: with `isMovableByWindowBackground`
/// on the window, SwiftUI gestures on non-control views lose the race and
/// the "splitter drag" moves the whole window. And the handle talks to
/// the STORE directly (a stable class reference) instead of view
/// closures — closure writes went stale after the first re-render and
/// dropped every drag event but the first. mouseDownCanMoveWindow=false
/// claims the drag for the handle; deltas use window coordinates because
/// the handle itself moves as the panel it resizes grows (a local
/// reference oscillates).
private final class SplitHandleHost: NSView {
    weak var store: AppStore?
    var which: AppStore.IdeSplitter = .tree
    var lineColor: NSColor = .separatorColor
    var hoverLineColor: NSColor = .separatorColor

    private var lastX: CGFloat?
    private let line = NSView()

    override var mouseDownCanMoveWindow: Bool { false }
    override var isFlipped: Bool { true }

    override func viewDidMoveToWindow() {
        wantsLayer = true
        line.wantsLayer = true
        addSubview(line)
        addTrackingArea(NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil))
        tint(hover: false)
    }

    private func tint(hover: Bool) {
        line.layer?.backgroundColor = (hover ? hoverLineColor : lineColor).cgColor
    }

    override func layout() {
        super.layout()
        line.frame = NSRect(x: bounds.midX - 0.5, y: 0, width: 1, height: bounds.height)
    }

    override func mouseEntered(with event: NSEvent) { tint(hover: true) }
    override func mouseExited(with event: NSEvent) { tint(hover: false) }

    override func mouseDown(with event: NSEvent) {
        lastX = event.locationInWindow.x
    }

    override func mouseDragged(with event: NSEvent) {
        let x = event.locationInWindow.x
        if let last = lastX { store?.ideSplitterDrag(which, dx: x - last) }
        lastX = x
    }

    override func mouseUp(with event: NSEvent) {
        store?.ideSplitterEnd(which)
        lastX = nil
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .resizeLeftRight)
    }
}

/// Thin AppKit handle plus the depth shadow the raised canvas casts onto
/// the panel beside it. `canvasAt` says which side of the band the canvas
/// is on (tree splitter: trailing; agent splitter: leading).
private struct VerticalSplitter: NSViewRepresentable {
    let store: AppStore
    let which: AppStore.IdeSplitter
    let canvasAt: HorizontalEdge
    let lineColor: NSColor
    let hoverLineColor: NSColor

    func makeNSView(context: Context) -> SplitHandleHost {
        let v = SplitHandleHost()
        v.store = store
        v.which = which
        v.lineColor = lineColor
        v.hoverLineColor = hoverLineColor
        return v
    }

    func updateNSView(_ v: SplitHandleHost, context: Context) {
        v.store = store
        v.which = which
        v.lineColor = lineColor
        v.hoverLineColor = hoverLineColor
    }
}

enum HorizontalEdge { case leading, trailing }

