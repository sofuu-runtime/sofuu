import SwiftUI
import AppKit

/// HUD state for the IDE editor leaf: gutter, hover/completion panels,
/// cursor line, LSP status, diagnostics. Only editor-leaf views observe
/// this, so high-frequency updates never re-render the shell.
final class GutterModel: ObservableObject {
    /// Set by the coordinator: any model-driven UI restyle.
    var onModelChange: (() -> Void)?
    /// Set by the coordinator: the visible completion row was committed.
    var onCompletionCommit: (() -> Void)?
    @Published var diagnostics: [String: [LSPWire.Diagnostic]] = [:]
    @Published var hoverText: String?
    @Published var hoverRect: NSRect = .zero
    @Published var completions: [LSPWire.CompletionItem]?
    @Published var completionIndex = 0
    @Published var cursor: (line: Int, column: Int) = (1, 1)
    @Published var lspStatus: String = "LSP off"
    @Published var outline: [LSPWire.OutlineNode] = []
}

/// The IDE code viewport: a hand-written NSTextView (SofuuCodeView) with
/// a separate number gutter (Canvas), a from-scratch lexer for syntax,
/// and a from-scratch LSP client feeding diagnostics / hover /
/// completion / symbols / definition.
struct CodeEditor: NSViewRepresentable {
    @Binding var text: String
    let path: String
    let dark: Bool
    let store: AppStore
    var editable: Bool = true
    var onEdited: (String) -> Void

    static let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
    static let gutterTopInset: CGFloat = 6
    static let gutterWidth: CGFloat = 46

    private var language: Language { Languages.language(forPath: path) }

    func makeCoordinator() -> Coordinator {
        Coordinator(store: store)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = SofuuCodeView()
        tv.font = Self.font
        tv.isEditable = editable
        tv.isRichText = false
        tv.allowsUndo = true
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = true
        tv.autoresizingMask = []
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        tv.drawsBackground = false
        tv.textContainerInset = NSSize(width: 4, height: Self.gutterTopInset)
        tv.insertionPointColor = .secondaryLabelColor
        tv.delegate = context.coordinator
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.isContinuousSpellCheckingEnabled = false

        let scrollView = NSScrollView()
        scrollView.documentView = tv
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false

        context.coordinator.attach(scrollView)
        context.coordinator.onEdited = onEdited
        context.coordinator.path = path
        context.coordinator.language = language
        context.coordinator.textView = tv
        context.coordinator.lexerHighlight(text: text, language: language)
        context.coordinator.syncLSPDocument(path: path, text: text, language: language)
        context.coordinator.push()

        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let tv = scrollView.documentView as? SofuuCodeView else { return }
        let c = context.coordinator
        c.onEdited = onEdited
        c.path = path
        c.language = language
        c.textView = tv
        tv.editor = c
        tv.delegate = c
        tv.hud = store.ideGutter

        if tv.string != text {
            tv.string = text
            c.rebuildLineIndex()
            c.lexerHighlight(text: text, language: language)
            c.syncLSPDocument(path: path, text: text, language: language)
            c.push()
        }
        if c.isDark != dark {
            c.isDark = dark
            c.lexerHighlight(text: tv.string, language: language)
        }
        tv.isEditable = editable
        c.refreshCursorStatus()
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        let store: AppStore
        let gutter: GutterModel
        weak var textView: SofuuCodeView?
        weak var scrollView: NSScrollView?
        private var scrollObserver: NSObjectProtocol?
        private var saveObserver: NSObjectProtocol?
        private var closeObserver: NSObjectProtocol?

        var isDark = true
        var path: String?
        var language: Language = Languages.plain
        var lsp: LSPClient?
        var lspLang: LangID?
        var openedPaths: Set<String> = []
        var onEdited: ((String) -> Void)?

        var lineStarts: [Int] = [0]
        var lineStep: CGFloat?
        private var lspChangeWork: DispatchWorkItem?

        init(store: AppStore) {
            self.store = store
            self.gutter = store.ideGutter
            super.init()
            store.ideCoordinator = self
            gutter.onModelChange = { [weak self] in
                guard let self else { return }
                self.applyDiagnosticUnderlines(text: self.textView?.string ?? "")
                if self.gutter.completions == nil { IDEHUD.hideCompletions() }
                if self.gutter.hoverText == nil { IDEHUD.hideHover() }
            }
            gutter.onCompletionCommit = { [weak self] in
                guard let self,
                      let items = self.gutter.completions,
                      self.gutter.completionIndex < items.count else { return }
                self.textView?.insertCompletionText(items[self.gutter.completionIndex].insert)
                IDEHUD.hideCompletions()
            }
            observeFileNotifications()
        }

        deinit {
            if let scrollObserver { NotificationCenter.default.removeObserver(scrollObserver) }
            if let saveObserver { NotificationCenter.default.removeObserver(saveObserver) }
            if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        }

        // MARK: - Attach / text pipeline

        func attach(_ sv: NSScrollView) {
            scrollView = sv
            scrollObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification,
                object: sv.contentView,
                queue: .main
            ) { [weak self] _ in
                self?.push()
            }
        }

        func textDidChange(_ notification: Notification) {
            rebuildLineIndex()
            lexerHighlight(text: textView?.string ?? "", language: language)
            scheduleLSPChange()
            push()
            onEdited?(textView?.string ?? "")
            (textView as? SofuuCodeView)?.triggerCompletion()
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            refreshCursorStatus()
        }

        func rebuildLineIndex() {
            let text = textView?.string ?? ""
            var starts: [Int] = [0]
            let ns = text as NSString
            var i = 0
            while i < ns.length {
                let nl = ns.rangeOfCharacter(from: .newlines, options: [], range: NSRange(location: i, length: ns.length - i))
                guard nl.location != NSNotFound else { break }
                starts.append(nl.location + nl.length)
                i = nl.location + nl.length
            }
            lineStarts = starts
        }

        /// Pulse the editor leaf (gutter position + HUD re-read).
        func push() {
            if let tv = textView, let lm = tv.layoutManager {
                calibrateStep(ns: tv.string as NSString, lm: lm)
            }
            gutter.objectWillChange.send()
        }

        func calibrateStep(ns: NSString, lm: NSLayoutManager) {
            guard lineStep == nil, ns.length > 0 else { return }
            var tops: [CGFloat] = []
            lm.enumerateLineFragments(
                forGlyphRange: NSRange(location: 0, length: min(4, ns.length))
            ) { r, _, _, _, _ in
                if tops.last.map({ abs(r.minY - $0) > 0.5 }) ?? true { tops.append(r.minY) }
            }
            if tops.count >= 2 { lineStep = tops[1] - tops[0] }
        }

        // MARK: - Syntax (our lexer, debounced, background scan)

        private var highlightWork: DispatchWorkItem?

        func lexerHighlight(text: String, language: Language) {
            highlightWork?.cancel()
            let dark = isDark
            let diags = gutter.diagnostics[path ?? ""] ?? []
            let work = DispatchWorkItem { [weak self] in
                let lexed = Lexer.lex(text, language: language)
                DispatchQueue.main.async { [weak self] in
                    guard let self, let tv = self.textView, tv.string == text,
                          let storage = tv.textStorage else { return }
                    let length = (text as NSString).length
                    let full = NSRange(location: 0, length: length)
                    storage.beginEditing()
                    storage.setAttributes([
                        .font: CodeEditor.font,
                        .foregroundColor: SyntaxPalette.plain(dark: dark),
                    ], range: full)
                    for token in lexed.tokens
                    where token.range.location + token.range.length <= length {
                        storage.addAttribute(
                            .foregroundColor,
                            value: SyntaxPalette.color(token.kind, dark: dark),
                            range: token.range)
                    }
                    storage.endEditing()
                    self.applyDiagnosticUnderlines(text: text)
                    if !self.runningLSP {
                        self.gutter.outline = lexed.outline.map {
                            LSPWire.OutlineNode(name: $0.name, kind: 0, line: $0.line, children: [])
                        }
                    }
                }
            }
            highlightWork = work
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.15, execute: work)
        }

        // MARK: - LSP lifecycle

        func syncLSPDocument(path: String, text: String, language: Language) {
            let root = store.state.projectDir
                ?? URL(fileURLWithPath: path).deletingLastPathComponent().path
            // One server per language: a tab switch between languages must
            // not feed a foreign file to the previous language's server.
            if lspLang != language.id {
                lsp = nil
                lspLang = nil
                openedPaths.removeAll()
            }
            if lsp == nil || lsp?.isRunning != true {
                // Async launch: makeNSView/updateNSView must never wait on
                // a subprocess (main-thread task waits pump the run loop
                // and re-enter SwiftUI — hard deadlock).
                gutter.lspStatus = "\(language.name) LSP: starting…"
                LSPRegistry.shared.clientAsync(for: language, rootPath: root) { [weak self] client in
                    guard let self else { return }
                    guard let client else {
                        if self.lsp == nil {
                            self.gutter.lspStatus = "\(language.name): no language server on PATH"
                        }
                        return
                    }
                    // The composer may have moved on to another language
                    // (or another client won the race); keep the winner.
                    if self.lspLang == language.id && self.lsp?.isRunning == true { return }
                    self.lsp = client
                    self.lspLang = language.id
                    client.onDiagnostics = { [weak self] diagPath, diags in
                        guard let self else { return }
                        self.gutter.diagnostics[diagPath] = diags
                        self.applyDiagnosticUnderlines(text: self.textView?.string ?? "")
                        self.push()
                    }
                    client.onServerExit = { [weak self] in
                        self?.lsp = nil
                        self?.lspLang = nil
                        self?.gutter.lspStatus = "LSP exited"
                    }
                    self.gutter.lspStatus = "\(language.name) LSP: running"
                    if self.path == path {
                        client.didOpen(path: path, text: text)
                        self.openedPaths.insert(path)
                    }
                    // Any other file synced meanwhile didOpens on its next
                    // syncLSPDocument (lsp is non-nil by then).
                }
                return
            }
            guard let lsp else { return }
            if !openedPaths.contains(path) {
                lsp.didOpen(path: path, text: text)
                openedPaths.insert(path)
            } else {
                lsp.didChange(path: path, text: text)
            }
        }

        func scheduleLSPChange() {
            lspChangeWork?.cancel()
            guard let path, let tv = textView else { return }
            let text = tv.string
            let work = DispatchWorkItem { [weak self] in
                self?.lspChangeWork = nil
                self?.lsp?.didChange(path: path, text: text)
            }
            lspChangeWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35, execute: work)
        }

        /// Send any pending document change NOW. Completion/targeted
        /// requests run this first, or the server answers from stale text.
        func flushLSPChangeNow() {
            guard let path, let tv = textView, lspChangeWork != nil else { return }
            lspChangeWork?.cancel()
            lspChangeWork = nil
            lsp?.didChange(path: path, text: tv.string)
        }

        private func observeFileNotifications() {
            saveObserver = NotificationCenter.default.addObserver(
                forName: Notification.Name("IDEFileContentSaved"),
                object: nil, queue: .main
            ) { [weak self] note in
                guard let savedPath = note.object as? String, savedPath == self?.path else { return }
                self?.lsp?.didSave(path: savedPath)
            }
            closeObserver = NotificationCenter.default.addObserver(
                forName: Notification.Name("IDEDocumentClosed"),
                object: nil, queue: .main
            ) { [weak self] note in
                guard let closedPath = note.object as? String else { return }
                self?.lsp?.didClose(path: closedPath)
                self?.gutter.diagnostics[closedPath] = nil
            }
        }

        private var runningLSP: Bool {
            guard let path else { return false }
            let language = Languages.language(forPath: path)
            let root = store.state.projectDir
                ?? URL(fileURLWithPath: path).deletingLastPathComponent().path
            return LSPRegistry.shared.runningClient(langID: language.id.rawValue, rootPath: root) != nil
        }

        // MARK: - Diagnostics underlines

        func applyDiagnosticUnderlines(text: String) {
            guard let tv = textView, let storage = tv.textStorage, let path else { return }
            let diags = gutter.diagnostics[path] ?? []
            let length = (text as NSString).length
            let full = NSRange(location: 0, length: length)
            storage.removeAttribute(.underlineStyle, range: full)
            for diag in diags where diag.line < lineStarts.count {
                let startOffset = min(lineStarts[diag.line] + diag.character, length)
                let endLineIndex = min(diag.endLine, lineStarts.count - 1)
                let endOffset = min(lineStarts[endLineIndex] + diag.endCharacter, length)
                let range = NSRange(location: startOffset, length: max(1, endOffset - startOffset))
                storage.addAttribute(.underlineStyle, value: NSUnderlineStyle.single.rawValue, range: range)
                storage.addAttribute(.underlineColor, value: diag.isError ? NSColor.systemRed : NSColor.systemYellow, range: range)
            }
        }

        // MARK: - Cursor

        func refreshCursorStatus() {
            let (line, column) = lineColumn(ofOffset: Int(textView?.selectedRange().location ?? 0))
            gutter.cursor = (line: line, column: column)
        }

        func selectedOffset() -> Int {
            Int(textView?.selectedRange().location ?? 0)
        }

        func lineColumn(ofOffset offset: Int) -> (line: Int, column: Int) {
            let clamped = max(0, min(offset, lineStarts.count - 1))
            var lo = 0, hi = lineStarts.count - 1
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if lineStarts[mid] <= clamped { lo = mid } else { hi = mid - 1 }
            }
            return (lo + 1, clamped - lineStarts[lo] + 1)
        }

        // MARK: - HUD actions

        func requestHover(line: Int, column: Int, screenRect: NSRect) {
            guard let path, let lsp else { return }
            lsp.hover(path: path, line: line - 1, character: column - 1, text: textView?.string) { text in
                guard let text, !text.isEmpty else { return }
                IDEHUD.showHover(String(text.prefix(2000)), at: screenRect)
            }
        }

        private var completionWork: DispatchWorkItem?

        /// Debounced (0.08 s) — one keystroke must not equal one server
        /// request + one HUD rebuild; the anchor rect is recomputed at
        /// fire time from the captured caret location.
        func requestCompletion(prefix: String, line: Int, column: Int, location: Int) {
            completionWork?.cancel()
            let work = DispatchWorkItem { [weak self] in
                guard let self, let tv = self.textView else { return }
                let rect = (tv as? SofuuCodeView)?.screenRect(at: location) ?? .zero
                if let lsp = self.lsp, let path = self.path {
                    self.flushLSPChangeNow()
                    lsp.completion(path: path, line: line - 1, character: column - 1, text: tv.string) { items in
                        // Empty prefix (after '.') = member completion — the
                        // server already ordered against the receiver.
                        let filtered = prefix.isEmpty
                            ? items
                            : items.filter { $0.label.lowercased().hasPrefix(prefix.lowercased()) }
                        guard !filtered.isEmpty else { return }
                        self.gutter.completions = Array(filtered.prefix(50))
                        self.gutter.completionIndex = 0
                        IDEHUD.showCompletions(self.gutter, anchor: rect)
                    }
                } else {
                    self.offerDocumentWords(prefix: prefix, screenRect: rect)
                }
            }
            completionWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.08, execute: work)
        }

        private func offerDocumentWords(prefix: String, screenRect: NSRect) {
            let text = textView?.string ?? ""
            let words = Set(text.split(whereSeparator: { !($0.isLetter || $0.isNumber || $0 == "_") }).map(String.init))
            let items = words
                .filter { $0.lowercased().hasPrefix(prefix.lowercased()) && $0 != prefix }
                .sorted()
                .prefix(30)
                .map { LSPWire.CompletionItem(label: $0, insert: $0) }
            guard !items.isEmpty else { return }
            gutter.completions = Array(items)
            gutter.completionIndex = 0
            IDEHUD.showCompletions(gutter, anchor: screenRect)
        }

        func goToDefinition(at offset: Int) {
            guard let path else { return }
            let (line, column) = lineColumn(ofOffset: offset)
            guard let lsp else {
                store.systemRow("No language server for go-to-definition (install sourcekit-lsp / rust-analyzer / pyright / clangd).")
                return
            }
            lsp.definition(path: path, line: line - 1, character: column - 1, text: textView?.string) { locations in
                guard let target = locations.first else { return }
                let root = self.store.state.projectDir
                if target.path == path {
                    self.scrollTo(characterLine: target.line + 1, column: target.character + 1)
                } else if root == nil || target.path.hasPrefix(root ?? "#nobody") {
                    self.store.openFile(target.path)
                } else {
                    self.store.systemRow("Definition is outside the workspace: \(target.path)")
                }
            }
        }

        func scrollTo(characterLine line: Int, column: Int) {
            guard line >= 1, !lineStarts.isEmpty else { return }
            let startIndex = lineStarts[min(line - 1, lineStarts.count - 1)]
            let range = NSRange(location: min(startIndex + max(0, column - 1), (textView?.string.utf16.count ?? 0)), length: 0)
            textView?.select(range: range)
        }

        func insertCompletion(_ item: LSPWire.CompletionItem) {
            textView?.insertCompletionText(item.insert)
        }

        // MARK: - Find (⌘F)

        func findMatches(query: String, caseSensitive: Bool) -> Int {
            guard !query.isEmpty, let text = textView?.string else { return 0 }
            let needle = caseSensitive ? query : query.lowercased()
            let haystack = caseSensitive ? text : text.lowercased()
            var count = 0
            var probe = 0
            let length = (haystack as NSString).length
            while probe <= length {
                let found = (haystack as NSString).range(
                    of: needle, options: [],
                    range: NSRange(location: probe, length: length - probe))
                guard found.location != NSNotFound else { break }
                count += 1
                probe = found.location + max(1, found.length)
            }
            return count
        }

        func selectMatch(query: String, caseSensitive: Bool, forward: Bool) {
            guard !query.isEmpty, let view = textView else { return }
            let length = (view.string as NSString).length
            let needle = caseSensitive ? query : query.lowercased()
            let options: NSString.CompareOptions = caseSensitive ? [] : [.caseInsensitive]
            let caret = selectedOffset()
            let found: NSRange
            if forward {
                let from = min(caret + 1, length)
                let hit = (view.string as NSString).range(
                    of: needle, options: options,
                    range: NSRange(location: from, length: max(0, length - from)))
                found = hit.location != NSNotFound
                    ? hit
                    : (view.string as NSString).range(of: needle, options: options, range: NSRange(location: 0, length: length))
            } else {
                let upTo = min(caret, length)
                let searchOptions = options.union(.backwards)
                let hit = (view.string as NSString).range(
                    of: needle, options: searchOptions,
                    range: NSRange(location: 0, length: upTo))
                found = hit.location != NSNotFound
                    ? hit
                    : (view.string as NSString).range(of: needle, options: options, range: NSRange(location: 0, length: length))
            }
            guard found.location != NSNotFound, found.location + found.length <= length else { return }
            view.select(range: found)
        }

        func endFind() {
            // Hook reserved for find teardown (selection retention is fine).
        }

        // MARK: - Outline (⇧⌘O)

        func requestOutline() {
            guard let path else { return }
            if let lsp {
                lsp.symbols(path: path, text: textView?.string) { [weak self] nodes in
                    guard let self else { return }
                    if !nodes.isEmpty {
                        self.gutter.outline = nodes
                    }
                    IDEHUD.showOutline(self.gutter.outline) { [weak self] node in
                        self?.scrollTo(characterLine: node.line + 1, column: 1)
                    }
                }
            } else {
                IDEHUD.showOutline(gutter.outline) { [weak self] node in
                    self?.scrollTo(characterLine: node.line + 1, column: 1)
                }
            }
        }
    }
}

// MARK: - Palette

enum SyntaxPalette {
    static func plain(dark: Bool) -> NSColor {
        color(dark ? 0xABB2BF : 0x383A42)
    }

    static func color(_ kind: Lexer.Kind, dark: Bool) -> NSColor {
        switch kind {
        case .comment: return color(dark ? 0x7F848E : 0x9DA0A6)
        case .string: return color(dark ? 0x98C379 : 0x50A14F)
        case .number: return color(dark ? 0xD19A66 : 0x986801)
        case .keyword: return color(dark ? 0xC678DD : 0xA626A4)
        case .type: return color(dark ? 0xE5C07B : 0xC18401)
        case .decorator: return color(dark ? 0x56B6C2 : 0x0184BC)
        case .op: return color(dark ? 0x56B6C2 : 0x4078F2)
        case .plain: return plain(dark: dark)
        }
    }

    private static func color(_ hex: UInt32) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255.0,
                green: CGFloat((hex >> 8) & 0xFF) / 255.0,
                blue: CGFloat(hex & 0xFF) / 255.0,
                alpha: 1)
    }
}

// MARK: - Gutter

struct LineGutter: View {
    @ObservedObject var pulse: GutterModel
    let coordinator: CodeEditor.Coordinator?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .background(c.bgRaised)
        .onAppear { pulse.objectWillChange.send() }
    }

    private func draw(ctx: GraphicsContext, size: CGSize) {
        guard let coordinator, let tv = coordinator.textView, let lm = tv.layoutManager else { return }
        let nsText = tv.string as NSString
        guard nsText.length > 0 else { return }
        if coordinator.lineStep == nil {
            coordinator.calibrateStep(ns: nsText, lm: lm)
        }
        let step = coordinator.lineStep ?? lm.defaultLineHeight(for: tv.font ?? CodeEditor.font)
        guard step > 0 else { return }

        let scrollY = coordinator.scrollView?.contentView.bounds.origin.y ?? 0
        let inset = CodeEditor.gutterTopInset
        let total = coordinator.lineStarts.count
        let diags = pulse.diagnostics[coordinator.path ?? "#"] ?? []
        // Several diagnostics can share one line; prefer the error (min).
        let diagLines = Dictionary(diags.map { ($0.line, $0.severity) },
                                   uniquingKeysWith: { a, b in min(a, b) })

        let font = Font.system(size: 11, design: .monospaced).monospacedDigit()
        let lineColor = Color.primary.opacity(0.45)

        var i = max(0, Int(((scrollY - inset) / step).rounded(.down)))
        while i < total {
            let y = inset + CGFloat(i) * step - scrollY
            if y > size.height { break }
            if y >= -step {
                let text = ctx.resolve(
                    Text(String(i + 1)).font(font).foregroundColor(lineColor))
                let w = text.measure(in: CGSize(width: CodeEditor.gutterWidth, height: .infinity)).width
                ctx.draw(text, at: CGPoint(x: CodeEditor.gutterWidth - 12 - w, y: y + (step - 13) / 2),
                         anchor: .topLeading)

                if let severity = diagLines[i] {
                    let markColor: Color = severity <= 1 ? Color.red.opacity(0.85) : Color.yellow.opacity(0.9)
                    ctx.fill(
                        Path(roundedRect: CGRect(x: CodeEditor.gutterWidth - 4, y: y + 4, width: 3, height: step - 8), cornerRadius: 1.5),
                        with: .color(markColor))
                }
            }
            i += 1
        }
    }
}
