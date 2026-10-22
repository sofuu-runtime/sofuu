import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The composer card: input on top, bar with model/effort pills + icons
/// left and send/stop right; mode + permissions pills below. The desktop
/// Composer's interaction set, native:
///
///   * slash palette with FULL keyboard navigation (↑↓ move, ↵ runs,
///     Tab tab-completes into the input, Esc dismisses) via an NSEvent
///     local monitor — the arrow keys never reach the field editor while
///     the palette is open;
///   * Enter sends, Shift+Enter inserts a newline;
///   * attach opens NSOpenPanel: images (≤6 MB) become data-URL chips,
///     everything else splices `@/real/path` AT THE CURSOR (the desktop's
///     semantics — paths pass through untouched, nothing is copied);
///   * ⌘V with an image on the pasteboard rides the next turn as an
///     attachment, like the desktop's paste-image handling;
///   * drag-and-dropped images do the same (`.onDrop`).
struct Composer: View {
    let streaming: Bool
    let mode: String
    let permissions: String
    let config: SofuuConfig?
    let ctx: CtxMeter?
    let images: [String]

    let onSubmit: (String, [String]) -> Void
    let onCancel: () -> Void
    let onModeChange: (String) -> Void
    let onPermissionsChange: (String) -> Void
    let onOpenSettings: (String) -> Void
    let onRemoveImage: (Int) -> Void
    let onAddImages: ([String]) -> Void
    let onSheet: (ComposerSheet) -> Void

    /// External text hand-off (Services "Send to Sofuu", sofuu:// deep
    /// links). Seeded once on first render and again whenever the store
    /// parks new draft text; consumed (cleared) by the store so the same
    /// text never re-applies after the user edits it.
    var pendingDraft: String? = nil
    var onConsumeDraft: () -> Void = {}

    @Environment(\.sofuuColors) private var c
    @State private var text = ""
    @State private var slashIndex = 0
    @State private var slashDismissed = false
    @FocusState private var focused: Bool

    // ── Slash palette state ──────────────────────────────────────────
    private var slashOpen: Bool { !slashDismissed && text.hasPrefix("/") && !text.contains(" ") }
    private var slashQuery: String {
        slashOpen ? String(text.dropFirst()).trimmingCharacters(in: .whitespaces).lowercased() : ""
    }
    private var slashMatches: [SlashCommand] {
        guard slashOpen else { return [] }
        return SLASH_COMMANDS.filter {
            slashQuery.isEmpty
                || $0.name.contains(slashQuery)
                || $0.desc.lowercased().contains(slashQuery)
        }
    }

    private var modelLabel: String {
        if let m = config?.model, !m.isEmpty { return m }
        return (config?.providers.isEmpty == false) ? "Choose model" : "No model"
    }
    private var effortLabel: String {
        let e = (config?.effort).flatMap { $0.isEmpty ? nil : $0 } ?? "high"
        return e == "off" ? "Off" : e.prefix(1).uppercased() + e.dropFirst()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !images.isEmpty { imageStrip }

            // Mode + permission switching is always visible (desktop/iOS
            // parity): Agent ↔ Chat-only, and Full/Edit-only/Plan/Ask me.
            HStack(spacing: 6) {
                Pill(label: mode == "chat" ? "Chat only" : "Agent", icon: Icons2.chat) { onSheet(.mode) }
                Pill(label: permissionsLabel(permissions), icon: Icons2.shield) { onSheet(.permissions) }
            }
            .padding(.bottom, 6)

            VStack(alignment: .leading, spacing: 0) {
                // Borderless: macOS TextFields default to a bezeled box;
                // the desktop composer's input sits straight on the card.
                TextField("Ask, @mention files, or type / for commands…", text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundStyle(c.text)
                    .tint(c.accent)
                    .lineLimit(2...10)
                    .focused($focused)
                    .onSubmit { submit() }
                    .onChange(of: text) { _ in resetSlash() }

                if slashOpen && !slashMatches.isEmpty {
                    SlashPalette(
                        commands: slashMatches,
                        selectedIndex: $slashIndex
                    ) { cmd in
                        runCommand(cmd)
                    }
                    .padding(.top, 8)
                }

                HStack(spacing: 6) {
                    Pill(label: modelLabel, icon: Icons2.spark) { onSheet(.model) }
                    Pill(label: effortLabel, icon: nil) { onSheet(.effort) }
                    if mode == "agent" {
                        IconAction(icon: Icons2.layers, description: "Context layers") { onSheet(.layers) }
                    }
                    Spacer(minLength: 8)
                    IconAction(icon: Icons2.attach, description: "Attach files") { attachFiles() }
                    CtxRing(ctx: ctx)
                    if streaming {
                        SendButton(icon: Icons2.stop, tint: c.errStrong) { onCancel() }
                    } else {
                        SendButton(icon: Icons2.send, tint: c.accent) { submit() }
                    }
                }
                .padding(.top, 8)
            }
            .padding(12)
            .background(c.bgCard, in: RoundedRectangle(cornerRadius: 16))
            .onDrop(of: [UTType.image.identifier], isTargeted: nil) { providers in
                handleDrop(providers)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .onAppear {
            installKeyMonitor()
            if let draft = pendingDraft, !draft.isEmpty {
                text = draft
                onConsumeDraft()
            }
        }
        .onDisappear { removeKeyMonitor() }
        .onChange(of: pendingDraft) { draft in
            guard let draft, !draft.isEmpty else { return }
            text = draft
            onConsumeDraft()
        }
    }

    private var imageStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.offset) { i, url in
                    ZStack(alignment: .topTrailing) {
                        if let img = decodeDataUrl(url) {
                            Image(nsImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 44, height: 44)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        } else {
                            RoundedRectangle(cornerRadius: 8).fill(c.bgCardHover)
                                .frame(width: 44, height: 44)
                        }
                        Button { onRemoveImage(i) } label: {
                            Text("×")
                                .font(.system(size: 11))
                                .foregroundStyle(.white)
                                .frame(width: 18, height: 18)
                                .background(c.errStrong, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .offset(x: 4, y: -4)
                    }
                }
            }
            .padding(.bottom, 8)
        }
    }

    // ── Submit / slash commands ──────────────────────────────────────

    private func submit() {
        guard !streaming else { return }
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty || !images.isEmpty else { return }
        onSubmit(text, images)
        text = ""
        resetSlash()
    }

    private func runCommand(_ cmd: SlashCommand) {
        onSubmit("/" + cmd.name, [])
        text = ""
        resetSlash()
    }

    private func resetSlash() {
        slashDismissed = false
        slashIndex = 0
    }

    private func permissionsLabel(_ p: String) -> String {
        switch p {
        case "full": return "Full access"
        case "edit": return "Edit only"
        case "plan": return "Plan mode"
        default: return "Ask me"
        }
    }

    // ── Attach: NSOpenPanel → images as data URLs, files @ cursor ────

    private func attachFiles() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.message = "Attach files — images ride the turn; text files become @mentions"
        guard panel.runModal() == .OK else { return }

        var imageDataUrls: [String] = []
        var mentionPaths: [String] = []
        for url in panel.urls {
            let ext = url.pathExtension.lowercased()
            if composerImageExts.contains(ext),
               let data = try? Data(contentsOf: url), data.count <= 6 << 20 {
                imageDataUrls.append("data:\(composerMime(ext));base64,\(data.base64EncodedString())")
            } else {
                mentionPaths.append(url.path)
            }
        }
        if !imageDataUrls.isEmpty { onAddImages(imageDataUrls) }
        for path in mentionPaths {
            insertAtCursor("@\(path) ")
        }
    }

    /// Splice text at the cursor of the focused field editor — the
    /// desktop's textarea `selectionStart` behaviour. Falls back to
    /// appending when the composer is not focused.
    private func insertAtCursor(_ insert: String) {
        let fieldEditor = NSApp.keyWindow?.firstResponder as? NSTextView
        if let tv = fieldEditor, focused {
            let range = tv.selectedRange()
            let before = (text as NSString).substring(to: min(range.location, (text as NSString).length))
            let after = range.location + range.length <= (text as NSString).length
                ? (text as NSString).substring(from: range.location + range.length)
                : ""
            text = before + insert + after
            // Put the caret after the insert once SwiftUI refreshes
            // the field editor from the binding.
            DispatchQueue.main.async {
                guard (NSApp.keyWindow?.firstResponder as? NSTextView) === tv else { return }
                let loc = min(before.count + insert.count, (text as NSString).length)
                tv.setSelectedRange(NSRange(location: loc, length: 0))
            }
        } else {
            text += (text.isEmpty ? "" : " ") + insert
            focused = true
        }
    }

    // ── Paste / drop images ──────────────────────────────────────────

    private func pasteImageIfPresent() -> Bool {
        // Only when there is no text on the pasteboard — regular ⌘V of a
        // string must keep pasting as text.
        guard NSPasteboard.general.data(forType: .string) == nil,
              let img = NSPasteboard.general.data(forType: .png)
                  ?? NSPasteboard.general.data(forType: .tiff),
              img.count <= 6 << 20 else { return false }
        addImageBytes(img, mime: NSPasteboard.general.data(forType: .png) != nil ? "png" : "tiff")
        return true
    }

    private func addImageBytes(_ data: Data, mime: String) {
        onAddImages(["data:image/\(mime);base64,\(data.base64EncodedString())"])
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        var handledAny = false
        for p in providers where p.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            handledAny = true
            p.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data, data.count <= 6 << 20 else { return }
                DispatchQueue.main.async {
                    let ext = UTType.image.preferredFilenameExtension ?? "png"
                    addImageBytes(data, mime: providerMime(ext))
                }
            }
        }
        return handledAny
    }

    // ── The key monitor (palette nav, shift-enter, paste) ────────────

    @State private var monitorAny: Any?
    private func installKeyMonitor() {
        guard monitorAny == nil else { return }
        monitorAny = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            handleKey(event)
        }
    }
    private func removeKeyMonitor() {
        if let m = monitorAny { NSEvent.removeMonitor(m) }
        monitorAny = nil
    }

    /// Consume (return nil) or pass through (return the event). Runs on
    /// the main thread before the field editor sees the key.
    private func handleKey(_ event: NSEvent) -> NSEvent? {
        guard let window = NSApp.keyWindow, window.firstResponder is NSTextView else { return event }
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        let plainShift = flags == .shift
        let onlyCommand = flags == .command

        switch event.keyCode {
        case 125, 126: // down / up
            guard slashOpen, !slashMatches.isEmpty else { return event }
            if event.keyCode == 125 {
                slashIndex = (slashIndex + 1) % slashMatches.count
            } else {
                slashIndex = (slashIndex - 1 + slashMatches.count) % slashMatches.count
            }
            return nil
        case 48: // tab
            guard slashOpen, slashMatches.indices.contains(slashIndex) else { return event }
            text = "/" + slashMatches[slashIndex].name + " "
            return nil
        case 36: // return
            if onlyCommand { return event }
            if plainShift {
                // Shift+Enter = newline in the composer (desktop rule).
                insertNewlineAtCursor()
                return nil
            }
            // Plain return with an open palette runs the highlighted
            // command; the field editor's onSubmit handles the rest.
            if slashOpen && !slashMatches.isEmpty {
                runCommand(slashMatches[slashIndex])
                return nil
            }
            return event
        case 53: // esc
            if slashOpen {
                slashDismissed = true
                return nil
            }
            return event
        case 9: // v — image paste (when the pasteboard carries no text)
            if onlyCommand { return pasteImageIfPresent() ? nil : event }
            return event
        default:
            return event
        }
    }

    private func insertNewlineAtCursor() {
        if let tv = NSApp.keyWindow?.firstResponder as? NSTextView {
            tv.insertText("\n")
        }
    }
}

enum ComposerSheet: String {
    case model, effort, layers, mode, permissions
}

// MARK: - Small controls

/// A composer bar icon: one target, one glyph size.
struct IconAction: View {
    let icon: String
    let description: String
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            Image(systemName: icon)
                .font(.system(size: 15))
                .foregroundStyle(c.textMuted)
                .frame(width: 34, height: 34)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(description)
    }
}

struct Pill: View {
    let label: String
    let icon: String?
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            HStack(spacing: 5) {
                if let icon {
                    Image(systemName: icon).font(.system(size: 11)).foregroundStyle(c.accent)
                }
                Text(label).font(.system(size: 13)).foregroundStyle(c.text).lineLimit(1)
                Image(systemName: Icons2.caret).font(.system(size: 9)).foregroundStyle(c.textFaint)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 32)
            .background(c.bgCardHover, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

private struct SendButton: View {
    let icon: String
    let tint: Color
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 36, height: 36)
                .background(tint, in: Circle())
        }
        .buttonStyle(.plain)
    }
}

/// Live context-window fill ring with the desktop's hover card: used /
/// window, the section breakdown, cache hit and the honesty line.
struct CtxRing: View {
    let ctx: CtxMeter?

    @Environment(\.sofuuColors) private var c
    @State private var hovering = false

    var body: some View {
        Group {
            if let ctx, ctx.window > 0 {
                ZStack {
                    Circle()
                        .stroke(c.border, lineWidth: 2.2)
                        .frame(width: 22, height: 22)
                    Circle()
                        .trim(from: 0, to: CGFloat(ctx.fraction))
                        .stroke(ringColor(ctx.fraction), style: StrokeStyle(lineWidth: 2.2, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 22, height: 22)
                }
                .frame(width: 24, height: 24)
                .onHover { hovering = $0 }
                .popover(isPresented: $hovering, arrowEdge: .bottom) {
                    CtxHoverCard(ctx: ctx)
                }
            }
        }
    }

    private func ringColor(_ f: Double) -> Color {
        if f > 0.85 { return c.err }
        if f > 0.7 { return c.warn }
        return c.accent
    }
}

/// The desktop ContextRing's hover payload: real usage out of the true
/// window, the engine's section estimate (after the first turn only —
/// until then the engine sends nothing to estimate), and where the window
/// number came from.
private struct CtxHoverCard: View {
    let ctx: CtxMeter
    @Environment(\.sofuuColors) private var c

    private var sourceLine: String {
        switch ctx.source {
        case "learned": return "learned from this model's refusals"
        case "discovered": return "discovered from the endpoint"
        case "registry": return "known model"
        case "config": return "your context-window override"
        default: return "default estimate"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Context window").font(.system(size: 12, weight: .medium)).foregroundStyle(c.text)
            Text("\(grouped(ctx.used)) / \(grouped(ctx.window)) tk (\(ctx.pct)%)")
                .font(.system(size: 13, design: .monospaced)).foregroundStyle(c.textMuted)

            if let s = ctx.sections {
                sectionBar(s)
            }
            if let hit = ctx.cacheHit {
                Text("Cache hit (this turn) \(Int(hit * 100))%")
                    .font(.system(size: 11)).foregroundStyle(c.textFaint)
            }
            Text("window: \(sourceLine)")
                .font(.system(size: 11)).foregroundStyle(c.textFaint)
            if ctx.clampedConfig {
                Text("config override clamped to the model's real limit")
                    .font(.system(size: 11)).foregroundStyle(c.warn)
            }
        }
        .padding(14)
        .frame(width: 260)
    }

    private func sectionBar(_ s: JSONObject) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            let totals = Double(ctx.used)
            let rows: [(String, Double, Color)] = [
                ("Messages", s.optDouble("messages"), Color(hex: 0x7C2D8A)),
                ("System tools", s.optDouble("systemTools"), Color(hex: 0xA44F9E)),
                ("MCP tools", s.optDouble("mcpTools"), Color(hex: 0xC07AB3)),
                ("System prompt", s.optDouble("system"), Color(hex: 0xD9A6C9)),
                ("Meta", s.optDouble("meta"), Color(hex: 0x9AA7B8)),
            ].compactMap { (label, v, color) in
                v > 0 ? (label, v, color) : nil
            }.sorted { $0.1 > $1.1 }
            if !rows.isEmpty, totals > 0 {
                GeometryReader { geo in
                    HStack(spacing: 0) {
                        ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                            Rectangle().fill(r.2)
                                .frame(width: geo.size.width * CGFloat(r.1 / totals))
                        }
                    }
                }
                .frame(height: 6)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                    HStack {
                        Circle().fill(r.2).frame(width: 6, height: 6)
                        Text(r.0).font(.system(size: 11)).foregroundStyle(c.textMuted)
                        Spacer()
                        Text(grouped(Int(r.1))).font(.system(size: 11, design: .monospaced)).foregroundStyle(c.textFaint)
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }
}

/// The "/" autocomplete card — selection follows the keyboard (index), the
/// mouse can pick too.
private struct SlashPalette: View {
    let commands: [SlashCommand]
    @Binding var selectedIndex: Int
    let onPick: (SlashCommand) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(commands.enumerated()), id: \.element.id) { i, cmd in
                        Button { onPick(cmd) } label: {
                            HStack(spacing: 4) {
                                Text("/\(cmd.name)").foregroundStyle(c.accent)
                                if let arg = cmd.arg { Text(arg).foregroundStyle(c.textFaint) }
                                Spacer(minLength: 8)
                                Text(cmd.desc).foregroundStyle(c.textMuted).lineLimit(1)
                            }
                            .font(.system(size: 13, design: .monospaced))
                            .padding(.horizontal, 10).padding(.vertical, 7)
                            .background(i == selectedIndex ? c.hoverVeil : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .id(cmd.id)
                        .onHover { hovering in if hovering { selectedIndex = i } }
                    }
                }
                .padding(6)
            }
            .frame(maxHeight: 190)
            .onChange(of: selectedIndex) { i in
                if i < commands.count { proxy.scrollTo(commands[i].id, anchor: .center) }
            }
        }
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .bottom) {
            Text("↑↓ navigate   ↵ select   tab complete   esc dismiss")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(c.textFaint)
                .padding(.vertical, 4)
        }
        .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
    }
}

// MARK: - image helpers

/// Decode a `data:image/...;base64,` URL into an NSImage.
func decodeDataUrl(_ dataUrl: String) -> NSImage? {
    guard let comma = dataUrl.firstIndex(of: ",") else { return nil }
    let b64 = String(dataUrl[dataUrl.index(after: comma)...])
    guard let data = Data(base64Encoded: b64) else { return nil }
    return NSImage(data: data)
}

private let composerImageExts: Set<String> = ["png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "bmp", "tiff"]

private func composerMime(_ ext: String) -> String {
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

private func providerMime(_ ext: String) -> String {
    // Drop-data comes stampless most of the time — png is the safe bet
    // for clipboard/drag imagery.
    ext == "tiff" || ext == "tif" ? "tiff" : "png"
}
