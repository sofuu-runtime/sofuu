import AppKit
import SwiftUI

/// The code editor text view: typing behaviors (auto-close pairs with
/// overtype, smart indent, brace expansion, tab/shift-tab indent, ⌘/
/// comment toggle), completion keyboard routing, ⌘-click go to
/// definition, and hover tracking. All hand-written.
final class SofuuCodeView: NSTextView {

    weak var editor: CodeEditor.Coordinator?
    weak var hud: GutterModel?

    private var trackingArea: NSTrackingArea?
    private var hoverTimer: Timer?
    private var lastHoverPoint: CGPoint?

    // MARK: - Lifecycle

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let old = trackingArea { removeTrackingArea(old) }
        trackingArea = nil
        guard window != nil else { return }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    // MARK: - Small helpers

    private func character(at index: Int) -> Character? {
        guard index >= 0, index < string.utf16.count,
              let idx = string.index(
                string.startIndex, offsetBy: index,
                limitedBy: string.endIndex) else { return nil }
        return string[idx]
    }

    private func collapsed() -> Bool { selectedRange().length == 0 }

    private func moveSelectionBy(_ delta: Int) {
        let location = max(0, min(string.utf16.count, selectedRange().location + delta))
        setSelectedRange(NSRange(location: location, length: 0))
    }

    // MARK: - Typing (auto-close + overtype)

    private func closingPartner(_ ch: Character) -> Character? {
        switch ch {
        case "(": return ")"
        case "[": return "]"
        case "{": return "}"
        case "\"": return "\""
        case "'": return "'"
        case "`": return "`"
        default: return nil
        }
    }

    private func isOpenBracket(_ ch: Character) -> Bool {
        ch == "(" || ch == "[" || ch == "{"
    }

    override func insertText(_ insert: Any, replacementRange: NSRange) {
        guard let typed = insert as? String, typed.count == 1, collapsed(),
              let ch = typed.first else {
            return super.insertText(insert, replacementRange: replacementRange)
        }
        let closers: Set<Character> = [")", "]", "}", "\"", "'", "`"]

        // Overtype: the closer already sits under the caret → step over it.
        if closers.contains(ch), let under = character(at: selectedRange().location), under == ch {
            moveSelectionBy(1)
            return
        }

        // Auto-close: brackets always; quotes only when the next char is
        // not part of a word (typing inside `foo|bar` must not invent them).
        if let close = closingPartner(ch) {
            let applies = isOpenBracket(ch) ? true : !nextCharIsWordy()
            if applies {
                super.insertText("\(ch)\(close)", replacementRange: replacementRange)
                moveSelectionBy(-1)
                return
            }
        }

        super.insertText(insert, replacementRange: replacementRange)
    }

    private func nextCharIsWordy() -> Bool {
        guard let next = character(at: selectedRange().location) else { return false }
        return next.isLetter || next.isNumber
    }

    // MARK: - Newline: smart indent + brace expansion

    override func insertNewline(_ sender: Any?) {
        let text = string as NSString
        let caret = selectedRange().location
        let lineRange = text.lineRange(for: NSRange(location: caret, length: 0))
        let prefixText = text.substring(with: NSRange(location: lineRange.lowerBound, length: caret - lineRange.lowerBound))
        let indent = String(prefixText.prefix(while: { $0 == " " || $0 == "\t" }))
        let unit = editor?.language.indentUnit ?? "    "

        let before = caret > 0 ? character(at: caret - 1) : nil
        let after = caret < text.length ? character(at: caret) : nil

        func partners(_ a: Character, _ b: Character) -> Bool {
            (a == "{" && b == "}") || (a == "(" && b == ")") || (a == "[" && b == "]")
        }

        // { | } → newline, indented body, newline; caret between them.
        if let b = before, let a = after, partners(b, a), collapsed(),
           b == "{" || b == "(" || b == "[" {
            let bodyIndent = indent + unit
            insertText("\n\(bodyIndent)\n\(indent)", replacementRange: NSRange(location: caret, length: 0))
            moveSelectionBy(-(indent.count + 1))
            return
        }

        // Line ends with an opener → next line indents one more level.
        if let b = before, b == "{" || b == "(" || b == "[" {
            insertText("\n\(indent)\(unit)", replacementRange: NSRange(location: caret, length: 0))
            return
        }

        insertText("\n\(indent)", replacementRange: NSRange(location: caret, length: 0))
    }

    // MARK: - Tab / shift-tab indent

    override func insertTab(_ sender: Any?) {
        if selectedRange().length > 0 {
            shiftSelectedLines(direction: 1)
        } else {
            insertText(editor?.language.indentUnit ?? "    ", replacementRange: selectedRange())
        }
    }

    override func insertBacktab(_ sender: Any?) {
        shiftSelectedLines(direction: -1)
    }

    private func shiftSelectedLines(direction: Int) {
        guard let storage = textStorage else { return }
        let unit = editor?.language.indentUnit ?? "    "
        let range = (string as NSString).lineRange(for: selectedRange())
        storage.beginEditing()

        // Walk bottom-up so early edits do not shift later offsets.
        var lineEnd = range.upperBound
        while lineEnd > range.lowerBound {
            let lineRange = (string as NSString).lineRange(for: NSRange(location: lineEnd - 1, length: 0))
            if direction > 0 {
                storage.replaceCharacters(in: NSRange(location: lineRange.lowerBound, length: 0), with: unit)
            } else {
                let head = (string as NSString).substring(with: NSRange(location: lineRange.lowerBound, length: min(unit.count, lineRange.length)))
                if head.hasPrefix(unit) {
                    storage.replaceCharacters(in: NSRange(location: lineRange.lowerBound, length: unit.count), with: "")
                } else if let first = head.first, first == "\t" {
                    storage.replaceCharacters(in: NSRange(location: lineRange.lowerBound, length: 1), with: "")
                }
            }
            lineEnd = lineRange.lowerBound
        }
        storage.endEditing()

        // Restore a selection across the (possibly shifted) lines.
        editor?.refreshCursorStatus()
    }

    // MARK: - Comment toggle (⌘/)

    override func keyDown(with event: NSEvent) {
        if let hud, hud.completions != nil {
            switch event.keyCode {
            case 125: // down
                hud.completionIndex = min((hud.completions?.count ?? 1) - 1, hud.completionIndex + 1)
                return
            case 126: // up
                hud.completionIndex = max(0, hud.completionIndex - 1)
                return
            case 36:  // return / enter
                commitCompletion()
                return
            case 53:  // esc
                hud.completions = nil
                return
            default:
                break
            }
        }
        if event.keyCode == 44, event.modifierFlags.contains(.command) {
            toggleComment()
            return
        }
        if event.modifierFlags.contains(.command), event.modifierFlags.contains(.shift),
           event.keyCode == 15 { // ⇧⌘R — references
            editor?.findReferencesAtCaret()
            return
        }
        if event.modifierFlags.contains(.command), event.keyCode == 15 { // ⌘R — rename
            editor?.renameAtCaret()
            return
        }
        if event.keyCode == 53 {
            hud?.hoverText = nil
            IDEHUD.hideOutline()
            IDEHUD.hidePrompt()
            IDEHUD.hideList()
        }
        super.keyDown(with: event)
    }

    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = super.menu(for: event)
        guard editor?.lspMenuReady == true, let menu else { return menu }
        let items: [(String, Selector)] = [
            ("Rename Symbol…", #selector(menuRename)),
            ("Find References", #selector(menuReferences)),
            ("Code Actions…", #selector(menuCodeActions)),
        ]
        menu.insertItem(NSMenuItem.separator(), at: 0)
        for (index, pair) in items.enumerated() {
            let item = NSMenuItem(title: pair.0, action: pair.1, keyEquivalent: "")
            item.target = self
            menu.insertItem(item, at: index + 1)
        }
        return menu
    }

    @objc private func menuRename() { editor?.renameAtCaret() }
    @objc private func menuReferences() { editor?.findReferencesAtCaret() }
    @objc private func menuCodeActions() { editor?.showCodeActions() }

    private func toggleComment() {
        guard let language = editor?.language,
              let marker = language.lineComment,
              let storage = textStorage else { return }
        let prefix = marker + " "
        let lines = (string as NSString).lineRange(for: selectedRange())

        // Which lines are already comments?
        var allCommented = true
        var offsets: [Int] = []
        var probe = lines.lowerBound
        while probe < lines.upperBound {
            let lineRange = (string as NSString).lineRange(for: NSRange(location: probe, length: 0))
            let content = (string as NSString)
                .substring(with: lineRange)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            offsets.append(lineRange.lowerBound)
            if !content.isEmpty, !content.hasPrefix(marker) {
                allCommented = false
            }
            probe = lineRange.upperBound
        }

        storage.beginEditing()
        for position in offsets.reversed() {
            let lineRange = (string as NSString).lineRange(for: NSRange(location: position, length: 0))
            let contentNS = (string as NSString).substring(with: lineRange) as NSString
            if allCommented {
                // NSString ranges are UTF-16-based and absolute within the
                // line — String.Index .distance math misplaces the strip
                // whenever the line carries non-BMP characters.
                let markerRange = contentNS.range(of: prefix).location != NSNotFound
                    ? contentNS.range(of: prefix)
                    : contentNS.range(of: marker)
                if markerRange.location != NSNotFound, markerRange.location < lineRange.length {
                    storage.replaceCharacters(
                        in: NSRange(location: lineRange.lowerBound + markerRange.location, length: markerRange.length),
                        with: "")
                }
            } else {
                var ws = 0
                while ws < contentNS.length, contentNS.character(at: ws) == 0x20 || contentNS.character(at: ws) == 0x09 {
                    ws += 1
                }
                storage.replaceCharacters(
                    in: NSRange(location: lineRange.lowerBound + ws, length: 0),
                    with: prefix)
            }
        }
        storage.endEditing()
        editor?.refreshCursorStatus()
    }

    // MARK: - Mouse: ⌘-click definition, hover tracking

    override func mouseDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            let point = convert(event.locationInWindow, from: nil)
            let index = insertionIndex(at: point)
            editor?.goToDefinition(at: index)
            return
        }
        hud?.completions = nil
        super.mouseDown(with: event)
    }

    func insertionIndex(at point: CGPoint) -> Int {
        guard let layoutManager, let textContainer else { return 0 }
        let index = layoutManager.characterIndex(
            for: point, in: textContainer,
            fractionOfDistanceBetweenInsertionPoints: nil)
        return min(index, string.utf16.count)
    }

    override func mouseMoved(with event: NSEvent) {
        super.mouseMoved(with: event)
        let point = convert(event.locationInWindow, from: nil)
        lastHoverPoint = point
        hoverTimer?.invalidate()
        hoverTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { [weak self] _ in
            guard let self, let point = self.lastHoverPoint else { return }
            let index = self.insertionIndex(at: point)
            let (line, column) = self.editor?.lineColumn(ofOffset: index) ?? (0, 0)
            let rect = self.screenRect(at: index)
            self.editor?.requestHover(line: line, column: column, screenRect: rect)
        }
    }

    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        hoverTimer?.invalidate()
        hud?.hoverText = nil
    }

    // MARK: - Completion

    func triggerCompletion() {
        guard collapsed() else { return }
        let text = string as NSString
        let location = selectedRange().location
        var prefix = wordPrefixBeforeCaret()
        if prefix.isEmpty {
            // Right after '.' or '(' → member/argument completion: the
            // caret position already tells the server where, with no
            // word prefix left to filter by.
            let afterDot = location > 0 && {
                guard let scalar = Unicode.Scalar(text.character(at: location - 1)) else { return false }
                let ch = Character(scalar)
                return ch == "." || ch == "("
            }()
            guard afterDot else {
                hud?.completions = nil
                return
            }
            prefix = ""
        }
        let (line, column) = editor?.lineColumn(ofOffset: location) ?? (0, 0)
        editor?.requestCompletion(prefix: prefix, line: line, column: column, location: location)
    }

    func wordPrefixBeforeCaret() -> String {
        guard collapsed() else { return "" }
        let text = string as NSString
        var start = selectedRange().location
        while start > 0 {
            let ch = text.character(at: start - 1)
            guard let scalar = Unicode.Scalar(ch) else { break }
            let character = Character(scalar)
            guard character.isLetter || character.isNumber || character == "_" else { break }
            start -= 1
        }
        let location = selectedRange().location
        guard start < location else { return "" }
        return text.substring(with: NSRange(location: start, length: location - start))
    }

    /// Insert a chosen completion, replacing the word prefix at the caret.
    func insertCompletionText(_ insertion: String) {
        let prefix = wordPrefixBeforeCaret()
        let location = selectedRange().location
        let replace = NSRange(location: location - prefix.utf16.count, length: prefix.utf16.count)
        insertText(insertion, replacementRange: replace)
        hud?.completions = nil
        hud?.completionIndex = 0
    }

    func commitCompletion() {
        guard let hud, let items = hud.completions, hud.completionIndex < items.count else { return }
        let item = items[hud.completionIndex]
        let prefix = wordPrefixBeforeCaret()
        let location = selectedRange().location
        let replace = NSRange(location: location - prefix.utf16.count, length: prefix.utf16.count)
        insertText(item.insert, replacementRange: replace)
        hud.completions = nil
        hud.completionIndex = 0
    }

    // MARK: - Geometry for popups

    func screenRect(at index: Int) -> NSRect {
        let clamped = max(0, min(index, string.utf16.count))
        guard let layoutManager, let textContainer else { return .zero }
        layoutManager.ensureLayout(forCharacterRange: NSRange(location: clamped, length: 0))
        let glyphRange = layoutManager.glyphRange(
            forCharacterRange: NSRange(location: clamped, length: 1),
            actualCharacterRange: nil)
        var rect = layoutManager.boundingRect(forGlyphRange: glyphRange, in: textContainer)
        rect.origin.x += textContainerInset.width
        rect.origin.y += textContainerInset.height
        let windowRect = convert(rect, to: nil)
        return window?.convertToScreen(windowRect) ?? .zero
    }

    /// Select + reveal a character range (find results).
    func select(range: NSRange) {
        setSelectedRange(range)
        scrollRangeToVisible(range)
    }
}
