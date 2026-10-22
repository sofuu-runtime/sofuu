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
    /// line (0-based) → folded? for foldable blocks (drives chevrons).
    @Published var foldLines: [Int: Bool] = [:]
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

    static let gutterTopInset: CGFloat = 6
    static let gutterWidth: CGFloat = 46

    /// The editor typeface resolves from prefs (Settings  Appearance);
    /// unknown families fall back to the system monospaced.
    static func makeFont(family: String?, size: Int) -> NSFont {
        if let family, let f = NSFont(name: family, size: CGFloat(size)) {
            return f
        }
        return NSFont.monospacedSystemFont(ofSize: CGFloat(size), weight: .regular)
    }
    static var font: NSFont { makeFont(family: nil, size: 13) }

    private var language: Language { Languages.language(forPath: path) }

    func makeCoordinator() -> Coordinator {
        Coordinator(store: store)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = SofuuCodeView()
        tv.font = Self.makeFont(family: store.editorFontFamily, size: store.editorFontSize)
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

        let whitespace = WhitespaceOverlay()
        whitespace.frame = tv.bounds
        whitespace.autoresizingMask = [.width, .height]
        whitespace.isHidden = !store.showWhitespace
        tv.addSubview(whitespace)

        let scrollView = NSScrollView()
        scrollView.documentView = tv
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false

        context.coordinator.attach(scrollView)
        context.coordinator.whitespaceOverlay = whitespace
        tv.layoutManager?.delegate = context.coordinator
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
        // Per-tab restore (cursor + scroll fraction) and a pending search
        // target  each applied exactly once per path switch.
        if c.restoredPath != path {
            c.restoredPath = path
            if let scroll = store.pendingScroll, scroll.path == path {
                store.pendingScroll = nil
                c.scrollTo(characterLine: scroll.line, column: max(1, scroll.column + 1))
            } else if let cursor = store.ideCursor(for: path) {
                c.scrollTo(characterLine: cursor.line, column: cursor.column)
            } else if let fraction = store.ideScrollFraction(for: path), fraction > 0.005,
                      let sv = c.scrollView {
                let clip = sv.contentView
                let docH = max(tv.frame.height, clip.bounds.height)
                clip.scroll(to: NSPoint(x: 0, y: fraction * max(0, docH - clip.bounds.height)))
                sv.reflectScrolledClipView(clip)
            }
        }
        if c.isDark != dark {
            c.isDark = dark
            c.lexerHighlight(text: tv.string, language: language)
        }
        tv.isEditable = editable
        let fontKey = (store.editorFontFamily, store.editorFontSize)
        if c.fontKey != fontKey {
            c.fontKey = fontKey
            c.fontSize = fontKey.1
            let newFont = CodeEditor.makeFont(family: fontKey.0, size: fontKey.1)
            tv.font = newFont
            c.currentFont = newFont
            c.lexerHighlight(text: tv.string, language: language)
        }
        c.whitespaceOverlay?.isHidden = !store.showWhitespace
        c.refreshCursorStatus()
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate, NSLayoutManagerDelegate {
        let store: AppStore
        let gutter: GutterModel
        weak var textView: SofuuCodeView?
        weak var scrollView: NSScrollView?
        private var scrollObserver: NSObjectProtocol?
        private var saveObserver: NSObjectProtocol?
        private var closeObserver: NSObjectProtocol?

        var isDark = true
        var path: String?
        var fontKey: (family: String?, size: Int) = (nil, 13)
        var currentFont: NSFont = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        var restoredPath: String?
        var fontSize: Int = 13
        var indentUnitWidth: CGFloat?
        /// Latest server semantic spans (LSP line/char space) for the
        /// current path  applied over the lexer in the highlight pass.
        var semanticSpans: [LSPWire.SemanticSpan] = []
        private var semanticWork: DispatchWorkItem?
        weak var whitespaceOverlay: WhitespaceOverlay?
        // Folding (glyph-nulling; document offsets stay intact for LSP).
        var foldablePairs: [(start: Int, end: Int)] = []   // char offsets
        var foldedStarts: Set<Int> = []
        private(set) var hiddenRanges: [NSRange] = []
        var language: Language = Languages.plain
        var lsp: LSPClient?
        var lspLang: LangID?
        var openedPaths: Set<String> = []
        var onEdited: ((String) -> Void)?
        var lspMenuReady: Bool { lsp?.isRunning == true }

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

        /// Hide folded glyphs: mark them .null; the layout (and every LSP
        /// offset) stays intact  only drawing skips.
        func layoutManager(
            _ layoutManager: NSLayoutManager,
            shouldGenerateGlyphs glyphs: UnsafePointer<CGGlyph>,
            properties: UnsafeMutablePointer<NSLayoutManager.GlyphProperty>,
            characterIndexes: UnsafePointer<Int>,
            font aFont: NSFont,
            forGlyphRange glyphRange: NSRange
        ) -> Int {
            guard !hiddenRanges.isEmpty else { return 0 }
            var touched = false
            for i in 0..<glyphRange.length {
                if hiddenRanges.contains(where: { NSLocationInRange(characterIndexes[i], $0) }) {
                    properties[i].insert(.null)
                    touched = true
                }
            }
            return touched ? glyphRange.length : 0
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
            if indentUnitWidth == nil, let tvNow = textView, let font = tvNow.font {
                let unit = language.indentUnit
                indentUnitWidth = (unit as NSString).size(withAttributes: [.font: font]).width
            }
            // Remember the scroll position for tab restore.
            if let tv = textView, let sv = scrollView, let scrollPath = path {
                let clip = sv.contentView
                let docH = max(tv.frame.height, clip.bounds.height)
                if docH > clip.bounds.height + 1 {
                    store.ideRecordScroll(scrollPath, fraction: clip.bounds.origin.y / (docH - clip.bounds.height))
                }
            }
            whitespaceOverlay?.needsDisplay = true
            gutter.objectWillChange.send()
        }

        /// From the lexer's bracket pairs: multi-line blocks become fold
        /// affordances; folded ones turn into hidden glyph ranges.
        func applyFoldables(pairs: [(NSRange, NSRange)], textLength: Int) {
            var foldable: [(Int, Int)] = []
            for (open, close) in pairs where open.location < close.location {
                if lineOf(charOffset: open.location) != lineOf(charOffset: close.location) {
                    foldable.append((open.location, close.location + 1))
                }
            }
            foldablePairs = foldable
            hiddenRanges = foldablePairs.compactMap { pair in
                foldedStarts.contains(pair.start)
                    ? NSRange(location: pair.start, length: max(1, pair.end - pair.start))
                    : nil
            }
            if let tv = textView, let lm = tv.layoutManager {
                lm.invalidateGlyphs(forCharacterRange: NSRange(location: 0, length: textLength),
                                    changeInLength: 0, actualCharacterRange: nil)
            }
            publishFoldLines()
    }

        private func lineOf(charOffset: Int) -> Int {
            var lo = 0, hi = lineStarts.count - 1
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if lineStarts[mid] <= charOffset { lo = mid } else { hi = mid - 1 }
            }
            return lo
        }

        func toggleFold(atLine line: Int) {
            let offset = lineStarts.indices.contains(line) ? lineStarts[line] : nil
            guard let offset else { return }
            guard let pair = foldablePairs.first(where: { $0.start == offset }) else { return }
            if foldedStarts.contains(pair.start) {
                foldedStarts.remove(pair.start)
            } else {
                foldedStarts.insert(pair.start)
            }
            if let tv = textView, let lm = tv.layoutManager {
                lm.invalidateGlyphs(forCharacterRange: NSRange(location: 0, length: tv.string.utf16.count),
                                    changeInLength: 0, actualCharacterRange: nil)
            }
            hiddenRanges = foldablePairs.compactMap { p in
                foldedStarts.contains(p.start)
                    ? NSRange(location: p.start, length: max(1, p.end - p.start))
                    : nil
            }
            publishFoldLines()
            push()
        }

        func collapseAllFolds() {
            foldedStarts = Set(foldablePairs.map(\.0))
            if let tv = textView, let lm = tv.layoutManager {
                lm.invalidateGlyphs(forCharacterRange: NSRange(location: 0, length: tv.string.utf16.count),
                                    changeInLength: 0, actualCharacterRange: nil)
            }
            hiddenRanges = foldablePairs.compactMap { p in
                foldedStarts.contains(p.start)
                    ? NSRange(location: p.start, length: max(1, p.end - p.start))
                    : nil
            }
            publishFoldLines()
            push()
        }

        func expandAllFolds() {
            foldedStarts = []
            hiddenRanges = []
            if let tv = textView, let lm = tv.layoutManager {
                lm.invalidateGlyphs(forCharacterRange: NSRange(location: 0, length: tv.string.utf16.count),
                                    changeInLength: 0, actualCharacterRange: nil)
            }
            publishFoldLines()
            push()
        }

        var foldableLines: Set<Int> {
            Set(foldablePairs.map { lineOf(charOffset: $0.start) })
        }

        private func publishFoldLines() {
            var map: [Int: Bool] = [:]
            for pair in foldablePairs {
                let line = lineOf(charOffset: pair.start)
                map[line] = foldedStarts.contains(pair.start)
            }
            gutter.foldLines = map
            push()
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
                        .font: self.currentFont,
                        .foregroundColor: SyntaxPalette.plain(dark: dark),
                    ], range: full)
                    for token in lexed.tokens
                    where token.range.location + token.range.length <= length {
                        storage.addAttribute(
                            .foregroundColor,
                            value: SyntaxPalette.color(token.kind, dark: dark),
                            range: token.range)
                    }
                    // Semantic overlay: where the server identified a token
                    // kind we care about, its color wins over the lexer.
                    if !self.semanticSpans.isEmpty {
                        self.applySemanticOverlay(storage: storage, text: text as NSString, dark: dark)
                    }
                    storage.endEditing()
                    self.applyFoldables(pairs: lexed.pairs, textLength: length)
                    self.whitespaceOverlay?.needsDisplay = true
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
                // and re-enter SwiftUI  hard deadlock).
                gutter.lspStatus = "\(language.name) LSP: starting"
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
                self?.scheduleSemanticTokens(path: path, text: text)
            }
            lspChangeWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35, execute: work)
        }

        private func scheduleSemanticTokens(path: String, text: String) {
            semanticWork?.cancel()
            guard let lsp = lsp, lsp.semanticLegend != nil else { return }
            let work = DispatchWorkItem { [weak self] in
                self?.semanticWork = nil
                lsp.semanticTokens(path: path) { [weak self] data in
                    guard let self else { return }
                    // Drop stale replies: the doc moved on meanwhile.
                    guard self.path == path, self.textView?.string == text else { return }
                    self.semanticSpans = LSPWire.decodeSemanticTokens(data, legend: lsp.semanticLegend ?? [])
                    self.lexerHighlight(text: text, language: self.language)
                }
            }
            semanticWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2, execute: work)
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
            ideRecordCursor(line: line, column: column)
        }

        private func ideRecordCursor(line: Int, column: Int) {
            guard let path, let view = textView, view.window?.firstResponder === view else { return }
            store.ideRecordCursor(path, line: line, column: column)
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

        /// Debounced (0.08 s)  one keystroke must not equal one server
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
                        // Empty prefix (after '.') = member completion  the
                        // server already ordered against the receiver.
                        // Otherwise FUZZY subsequence ranking (prefix bonus,
                        // contiguity bonus), ties broken by label length.
                        let filtered: [LSPWire.CompletionItem]
                        if prefix.isEmpty {
                            filtered = items
                        } else {
                            filtered = items.compactMap { item -> (LSPWire.CompletionItem, Int)? in
                                self.fuzzyRank(label: item.label, query: prefix).map { (item, $0) }
                            }
                            .sorted { left, right in
                                left.1 != right.1 ? left.1 > right.1 : left.0.label.count < right.0.label.count
                            }
                            .map(\.0)
                        }
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

        private func fuzzyRank(label: String, query: String) -> Int? {
            let l = label.lowercased(), q = query.lowercased()
            guard !q.isEmpty else { return 0 }
            var score = 0
            var prev = -1
            var searchStart = l.startIndex
            for ch in q {
                guard let r = l.range(of: String(ch), range: searchStart..<l.endIndex) else { return nil }
                let pos = l.distance(from: l.startIndex, to: r.lowerBound)
                score += 1
                if pos == prev + 1 { score += 3 }
                if pos == 0 { score += 5 }
                prev = pos
                searchStart = r.upperBound
            }
            return score
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

        // MARK: - Find (F)

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

        // MARK: - Replace

        /// Replace the current selection when it matches, else find and
        /// replace the next match, then select the following one.
        func replaceOne(query: String, replacement: String, caseSensitive: Bool) {
            guard let view = textView, let storage = view.textStorage else { return }
            let length = (view.string as NSString).length
            let options: NSString.CompareOptions = caseSensitive ? [] : [.caseInsensitive]
            let viewRange = view.selectedRange()

            let target: NSRange
            let selectedLength = viewRange.length
            let selectedMatches = selectedLength > 0
                && viewRange.location + viewRange.length <= length
                && (view.string as NSString).substring(with: viewRange)
                    .compare(query, options: options) == .orderedSame
            if selectedMatches {
                target = viewRange
            } else {
                guard let hit = nextMatch(query: query, caseSensitive: caseSensitive, from: viewRange.location) else { return }
                target = hit
            }
            view.shouldChangeText(in: target, replacementString: replacement)
            storage.replaceCharacters(in: target, with: replacement)
            view.didChangeText()
        }

        /// Replace every occurrence in one undoable edit. Returns the count.
        @discardableResult
        func replaceAll(query: String, replacement: String, caseSensitive: Bool) -> Int {
            guard let view = textView, let storage = view.textStorage else { return 0 }
            let text = view.string as NSString
            let length = text.length
            let options: NSString.CompareOptions = caseSensitive ? [] : [.caseInsensitive]
            var ranges: [NSRange] = []
            var probe = 0
            while probe <= length {
                let found = text.range(
                    of: query, options: options,
                    range: NSRange(location: probe, length: length - probe))
                guard found.location != NSNotFound, found.length > 0 else { break }
                ranges.append(found)
                probe = NSMaxRange(found)
            }
            guard !ranges.isEmpty else { return 0 }

            let replaced = NSMutableString(string: text)
            for range in ranges.reversed() {
                replaced.replaceCharacters(in: range, with: replacement)
            }
            let full = NSRange(location: 0, length: length)
            view.shouldChangeText(in: full, replacementString: replaced as String)
            storage.replaceCharacters(in: full, with: replaced as String)
            view.didChangeText()
            return ranges.count
        }

        /// The next match strictly after `from`, wrapping once.
        private func nextMatch(query: String, caseSensitive: Bool, from: Int) -> NSRange? {
            guard let view = textView else { return nil }
            let text = view.string as NSString
            let length = text.length
            let options: NSString.CompareOptions = caseSensitive ? [] : [.caseInsensitive]
            if length == 0 { return nil }
            let start = min(max(0, from), length - 1)
            let hit = text.range(
                of: query, options: options,
                range: NSRange(location: start, length: length - start))
            if hit.location != NSNotFound { return hit }
            let wrapped = text.range(of: query, options: options, range: NSRange(location: 0, length: max(0, start)))
            return wrapped.location != NSNotFound ? wrapped : nil
        }

        /// Map server token types onto our palette. Kinds our lexer already
        /// colors equally are skipped so servers disagree rarely visible.
        private static let semanticKindMap: [String: Lexer.Kind] = [
            "comment": .comment,
            "keyword": .keyword,
            "string": .string,
            "number": .number,
            "operator": .op,
            "decorator": .decorator,
            "macro": .decorator,
            "attribute": .decorator,
            "type": .type,
            "class": .type,
            "enum": .type,
            "interface": .type,
            "struct": .type,
            "namespace": .type,
            "enumMember": .type,
        ]

        private func applySemanticOverlay(storage: NSTextStorage, text ns: NSString, dark: Bool) {
            let length = ns.length
            for span in semanticSpans {
                guard span.line + 1 < lineStarts.count || (span.line < lineStarts.count && lineStarts[span.line] < length) else { continue }
                guard span.line < lineStarts.count else { continue }
                let start = min(lineStarts[span.line] + span.startCharacter, length)
                let end = min(start + span.length, length)
                guard end > start else { continue }
                guard let kind = CodeEditor.Coordinator.semanticKindMap[span.tokenType] else { continue }
                storage.addAttribute(
                    .foregroundColor,
                    value: SyntaxPalette.color(kind, dark: dark),
                    range: NSRange(location: start, length: end - start))
            }
        }

        // MARK: - Rename / references / code actions / workspace symbols

        func renameAtCaret() {
            guard let path, let lsp else {
                store.systemRow("No language server for rename.")
                return
            }
            let text = textView?.string as NSString? ?? ""
            let caret = selectedOffset()
            let (line, column) = lineColumn(ofOffset: caret)
            var wordStart = min(caret, text.length)
            while wordStart > 0, isWordByte(text.character(at: wordStart - 1)) { wordStart -= 1 }
            var wordEnd = wordStart
            while wordEnd < text.length, isWordByte(text.character(at: wordEnd)) { wordEnd += 1 }
            let initial = text.substring(with: NSRange(location: wordStart, length: wordEnd - wordStart))
            IDEHUD.showPrompt(title: "Rename symbol", placeholder: "New name", initial: initial) { [weak self] newName in
                guard let self, !newName.isEmpty else { return }
                lsp.rename(path: path, line: line - 1, character: column - 1, newName: newName) { edits in
                    self.applyWorkspaceEdits(edits)
                    if edits.isEmpty {
                        self.store.systemRow("Rename: the server returned no edits.")
                    }
                }
            }
        }

        private func isWordByte(_ u: unichar?) -> Bool {
            guard let u else { return false }
            return (u >= 65 && u <= 90) || (u >= 97 && u <= 122) || (u >= 48 && u <= 57) || u == 0x5F
        }

        /// Apply an LSP WorkspaceEdit: the CURRENT document gets one
        /// in-place undoable batch; other files are edited on disk (their
        /// line starts computed fresh), open tabs reloaded, and the server
        /// notified about any document it has open.
        func applyWorkspaceEdits(_ editsByPath: [String: [LSPWire.RawEdit]]) {
            for (editPath, edits) in editsByPath where !edits.isEmpty {
                if editPath == path {
                    applyInDoc(edits)
                } else {
                    applyOnDisk(path: editPath, edits: edits)
                }
            }
        }

        private func applyInDoc(_ edits: [LSPWire.RawEdit]) {
            guard let textView, let storage = textView.textStorage else { return }
            let nsText = textView.string as NSString
            let length = nsText.length
            let starts = lineStarts
            var ranges: [(NSRange, String)] = []
            for e in edits {
                let startLine = min(e.line, starts.count - 1)
                let endLine = min(e.endLine, starts.count - 1)
                let start = min(starts[startLine] + e.character, length)
                let end = min(starts[endLine] + e.endCharacter, length)
                guard start <= end else { continue }
                ranges.append((NSRange(location: start, length: end - start), e.newText))
            }
            guard !ranges.isEmpty else { return }
            let new = NSMutableString(string: textView.string)
            for (range, replacement) in ranges.sorted(by: { $0.0.location > $1.0.location }) {
                new.replaceCharacters(in: range, with: replacement)
            }
            let full = NSRange(location: 0, length: length)
            textView.shouldChangeText(in: full, replacementString: new as String)
            storage.replaceCharacters(in: full, with: new as String)
            textView.didChangeText()
        }

        private func applyOnDisk(path: String, edits: [LSPWire.RawEdit]) {
            guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else {
                store.systemRow("Rename: could not read \((path as NSString).lastPathComponent).")
                return
            }
            let ns = raw as NSString
            let length = ns.length
            var starts: [Int] = [0]
            var i = 0
            while i < length {
                let nl = ns.rangeOfCharacter(from: .newlines, options: [], range: NSRange(location: i, length: length - i))
                guard nl.location != NSNotFound else { break }
                starts.append(nl.location + nl.length)
                i = nl.location + nl.length
            }
            let locate: (Int, Int) -> Int = { line, col in
                let l = min(max(0, line), starts.count - 1)
                return min(starts[l] + col, length)
            }
            let new = NSMutableString(string: raw)
            for e in edits.sorted(by: { ($0.line, $0.character) > ($1.line, $1.character) }) {
                let start = locate(e.line, e.character)
                let end = locate(e.endLine, e.endCharacter)
                guard start <= end else { continue }
                new.replaceCharacters(in: NSRange(location: start, length: end - start), with: e.newText)
            }
            do {
                try (new as String).write(toFile: path, atomically: true, encoding: .utf8)
            } catch {
                store.systemRow("Rename write failed: \(error.localizedDescription)")
                return
            }
            refreshExternalView(path: path)
        }

        /// Post-disk-edit freshness: re-list/tick + notify the server when
        /// it holds that document. Clean open tabs reload via the IDE
        /// reload notification; dirty tabs keep the user's buffer.
        private func refreshExternalView(path: String) {
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: .init("IDEReloadPath"), object: path)
                self.store.nudgeExternal()
                if let lsp = self.lsp, lsp.hasOpenDocs,
                   let text = try? String(contentsOfFile: path, encoding: .utf8) {
                    lsp.didChange(path: path, text: text)
                }
            }
        }

        func findReferencesAtCaret() {
            guard let path, let lsp else {
                store.systemRow("No language server for find references.")
                return
            }
            let (line, column) = lineColumn(ofOffset: selectedOffset())
            lsp.references(path: path, line: line - 1, character: column - 1, includeDeclaration: true) { locations in
                guard !locations.isEmpty else {
                    self.store.systemRow("No references found.")
                    return
                }
                let nodes = locations.map { loc in
                    LSPWire.OutlineNode(
                        name: "\((loc.path as NSString).lastPathComponent):\(loc.line + 1)",
                        kind: 0, line: loc.line, children: [])
                }
                IDEHUD.showOutline(nodes) { [weak self] picked in
                    guard let self, let index = nodes.firstIndex(where: { $0.name == picked.name }) else { return }
                    let loc = locations[index]
                    if loc.path == self.path {
                        self.scrollTo(characterLine: loc.line + 1, column: loc.character + 1)
                    } else {
                        self.store.pendingScroll = (path: loc.path, line: loc.line + 1, column: loc.character)
                        self.store.openFile(loc.path)
                    }
                }
            }
        }

        func showWorkspaceSymbols() {
            guard let lsp else {
                store.systemRow("No language server for workspace symbols.")
                return
            }
            IDEHUD.showPrompt(title: "Find symbols in workspace", placeholder: "Name", initial: "") { [weak self] query in
                guard let self else { return }
                lsp.workspaceSymbols(query: query) { symbols in
                    guard !symbols.isEmpty else {
                        self.store.systemRow("No workspace symbols matched \(query).")
                        return
                    }
                    let displays = symbols.map {
                        "\($0.name)  \(($0.path as NSString).lastPathComponent):\($0.line + 1)"
                    }
                    IDEHUD.showList(title: "Workspace symbols", items: displays) { index in
                        let sym = symbols[index]
                        if sym.path == self.path {
                            self.scrollTo(characterLine: sym.line + 1, column: sym.character + 1)
                        } else {
                            self.store.pendingScroll = (path: sym.path, line: sym.line + 1, column: sym.character)
                            self.store.openFile(sym.path)
                        }
                    }
                }
            }
        }

        func showCodeActions() {
            guard let path, let lsp else {
                store.systemRow("No language server for code actions.")
                return
            }
            let sel = textView?.selectedRange() ?? NSRange(location: 0, length: 0)
            let (sl, sc) = lineColumn(ofOffset: sel.location)
            let endOffset = min(NSMaxRange(sel), max(0, (textView?.string.utf16.count ?? 1) - 1))
            let (el, ec) = lineColumn(ofOffset: endOffset)
            let diags = (store.ideGutter.diagnostics[path] ?? []).map { d -> [String: Any] in
                [
                    "range": [
                        "start": ["line": d.line, "character": d.character],
                        "end": ["line": d.endLine, "character": d.endCharacter],
                    ],
                    "message": d.message,
                    "severity": d.severity,
                ]
            }
            lsp.codeActions(
                path: path, startLine: sl - 1, startChar: sc - 1,
                endLine: el - 1, endChar: max(0, ec - 1), diagnostics: diags) { actions in
                let usable = actions.filter { $0.isEditOnly && !$0.edits.isEmpty }
                guard !usable.isEmpty else {
                    self.store.systemRow("No quick actions available here.")
                    return
                }
                IDEHUD.showList(title: "Quick actions", items: usable.map(\.title)) { [weak self] index in
                    self?.applyWorkspaceEdits(usable[index].edits)
                }
            }
        }

        // MARK: - Outline (O)

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
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0).onEnded { gesture in
            handleTap(at: gesture.location
                .applying(.identity))
        })
        .onAppear { pulse.objectWillChange.send() }
    }

    private func handleTap(at point: CGPoint) {
        guard let coordinator else { return }
        guard !pulse.foldLines.isEmpty, let tv = coordinator.textView else { return }
        let scrollY = coordinator.scrollView?.contentView.bounds.origin.y ?? 0
        let step = coordinator.lineStep ?? 13.0
        guard step > 0 else { return }
        let line = Int(((point.y + scrollY - CodeEditor.gutterTopInset) / step).rounded(.down))
        guard coordinator.foldableLines.contains(line) else { return }
        coordinator.toggleFold(atLine: line)
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

        let font = Font.system(size: max(9, Double(coordinator.fontSize - 2)), design: .monospaced).monospacedDigit()
        let lineColor = Color.primary.opacity(0.45)

        var i = max(0, Int(((scrollY - inset) / step).rounded(.down)))
        while i < total {
            let y = inset + CGFloat(i) * step - scrollY
            if y > size.height { break }
            if y >= -step {
                // Folding chevron on foldable lines: ▾ open / ▸ folded.
                if let folded = pulse.foldLines[i] {
                    let cx = CodeEditor.gutterWidth - 38
                    var arrow = Path()
                    let cy = y + step / 2
                    if folded {
                        arrow.move(to: CGPoint(x: cx, y: cy - 3))
                        arrow.addLine(to: CGPoint(x: cx + 4, y: cy))
                        arrow.addLine(to: CGPoint(x: cx, y: cy + 3))
                    } else {
                        arrow.move(to: CGPoint(x: cx - 3, y: cy - 2))
                        arrow.addLine(to: CGPoint(x: cx + 3, y: cy - 2))
                        arrow.addLine(to: CGPoint(x: cx - 3, y: cy + 2))
                    }
                    ctx.stroke(arrow, with: .color(.secondary), lineWidth: 1.2)
                }
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


// MARK: - Indent guides (cosmetic vertical columns, scroll-synced)

/// Draws one faint vertical hairline per indent level, scrolling with the
/// text (fresh clip reads at draw time  the gutter's pattern). Cheap and
/// decorative; per-scope guides need a parse tree, this is column math.
struct IndentGuidesCanvas: View {
    @ObservedObject var pulse: GutterModel
    let coordinator: CodeEditor.Coordinator?
    let store: AppStore

    var body: some View {
        Canvas { ctx, size in
            guard let coordinator, let tv = coordinator.textView else { return }
            guard coordinator.fontSize > 4 else { return }
            guard let unit = coordinator.indentUnitWidth, unit > 2 else { return }
            let textX = tv.textContainerInset.width
            for level in 1...12 {
                let x = textX + CGFloat(level) * unit - unit / 2
                guard x < size.width else { break }
                var path = Path()
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))
                ctx.stroke(path, with: .color(Color.primary.opacity(0.07)), lineWidth: 1)
            }
            _ = pulse
        }
    }
}


// MARK: - Visible whitespace (dot pass over the live glyphs)

/// A subview OF the text view, so it scrolls with the content for free.
/// Draws a faint tick for every space and a longer dash for every tab in
/// the visible character range  nothing is inserted into the text.
final class WhitespaceOverlay: NSView {
    weak var textView: NSTextView?

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func draw(_ dirtyRect: NSRect) {
        guard let tv = textView, let lm = tv.layoutManager,
              let storage = tv.textStorage, let container = tv.textContainer,
              let clip = tv.enclosingScrollView?.contentView else { return }
        let visible = NSRect(x: 0, y: clip.bounds.origin.y,
                             width: max(1, tv.bounds.width),
                             height: max(1, clip.bounds.height))
        let glyphRange = lm.glyphRange(forBoundingRect: visible, in: container)
        let charRange = lm.characterRange(forGlyphRange: glyphRange, actualGlyphRange: nil)
        NSColor.labelColor.withAlphaComponent(0.14).setFill()
        let ns = storage.string as NSString
        var index = charRange.location
        let end = charRange.location + charRange.length
        while index < end {
            let ch = ns.character(at: index)
            let isTab = ch == 0x09
            let isSpace = ch == 0x20
            if isSpace || isTab {
                let gr = lm.glyphRange(forCharacterRange: NSRange(location: index, length: 1),
                                       actualCharacterRange: nil)
                if gr.length > 0 {
                    var rect = lm.boundingRect(forGlyphRange: gr, in: container)
                    rect.origin.x += tv.textContainerInset.width
                    rect.origin.y += tv.textContainerInset.height
                    if isSpace {
                        let dot = CGRect(x: rect.midX - 0.75, y: rect.maxY - 3.5, width: 1.5, height: 1.5)
                        NSBezierPath(ovalIn: dot).fill()
                    } else {
                        let dash = CGRect(x: rect.minX + 1, y: rect.maxY - 3.0,
                                          width: max(2, rect.width - 2), height: 1.2)
                        NSBezierPath(roundedRect: dash, xRadius: 0.6, yRadius: 0.6).fill()
                    }
                }
            }
            index += 1
        }
    }
}
