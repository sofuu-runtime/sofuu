import SwiftUI
import AppKit

/// The IDE code viewport — an editable NSTextView with a line-number
/// ruler and debounced syntax highlighting, wrapped for SwiftUI.
///
/// Highlighting is attribute-only (foreground colors over unchanged
/// text storage), so the cursor, selection and undo stack never jump.
/// The line-start index is rebuilt on every edit (single scan) and the
/// ruler does one binary search per visible line.
struct CodeEditor: NSViewRepresentable {
    /// Live buffer text (two-way; the AppStore holds it across tab switches).
    @Binding var text: String
    let path: String
    let dark: Bool
    var editable: Bool = true
    /// Fires (debounced, main) with the final text after edits settle.
    var onEdited: (String) -> Void

    static let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = NSTextView()
        tv.font = Self.font
        tv.isEditable = editable
        tv.isRichText = false            // plain text only — attributes are ours
        tv.allowsUndo = true
        tv.autoresizingMask = [.width]
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = true
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        tv.drawsBackground = false
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

        let ruler = LineNumberRuler()
        ruler.textView = tv
        ruler.coordinator = context.coordinator
        ruler.clientView = tv
        ruler.ruleThickness = 46
        scrollView.verticalRulerView = ruler
        scrollView.hasVerticalRuler = true
        scrollView.rulersVisible = true

        let c = context.coordinator
        c.textView = tv
        c.onEdited = onEdited
        c.lang = Syntax.Lang.guess(path)
        c.isDark = dark
        c.rebuildLineIndex()
        c.applyBaseAttributes()
        c.highlightNow()
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let tv = scrollView.documentView as? NSTextView else { return }
        let c = context.coordinator
        // makeNSView runs once; these go stale the moment the same NSView
        // is reused for another tab (edits would land in the first path).
        c.onEdited = onEdited
        c.lang = Syntax.Lang.guess(path)
        // Sync only on real external changes (tab switch / disk reload) —
        // comparing against the view text keeps typing from being clobbered.
        if tv.string != text {
            tv.string = text
            c.rebuildLineIndex()
            c.applyBaseAttributes()
            c.highlightNow()
        }
        if c.isDark != dark {
            c.isDark = dark
            c.highlightNow()
        }
        tv.isEditable = editable
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        weak var textView: NSTextView?
        var isDark = true
        var onEdited: ((String) -> Void)?
        var lang: Syntax.Lang = .plain

        /// Character offset of every line start (for the ruler).
        var lineStarts: [Int] = [0]

        private var highlightWork: DispatchWorkItem?

        func textDidChange(_ notification: Notification) {
            rebuildLineIndex()
            applyBaseAttributes()
            scheduleHighlight()
            onEdited?(textView?.string ?? "")
        }

        /// One scan: offsets of every line start.
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

        func lineNumber(for charIndex: Int) -> Int {
            var lo = 0, hi = lineStarts.count - 1
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if lineStarts[mid] <= charIndex { lo = mid } else { hi = mid - 1 }
            }
            return lo + 1
        }

        func applyBaseAttributes() {
            guard let storage = textView?.textStorage else { return }
            storage.beginEditing()
            storage.setAttributes([
                .font: CodeEditor.font,
                .foregroundColor: Syntax.nsPlain(dark: isDark),
            ], range: NSRange(location: 0, length: storage.length))
            storage.endEditing()
        }

        /// Attribute-only re-color, debounced 250 ms, token pass on a
        /// background queue, application on main (skipped if the text has
        /// moved on in the meantime).
        func scheduleHighlight() {
            highlightWork?.cancel()
            guard let tv = textView else { return }
            let text = tv.string
            let dark = isDark
            let work = DispatchWorkItem { [weak self] in
                let colored = Syntax.colorRanges(text, lang: self?.lang ?? .plain)
                DispatchQueue.main.async { [weak self] in
                    guard let self, let tv = self.textView, tv.string == text,
                          let storage = tv.textStorage else { return }
                    let full = NSRange(location: 0, length: storage.length)
                    storage.beginEditing()
                    storage.addAttribute(.foregroundColor, value: Syntax.nsPlain(dark: dark), range: full)
                    for (range, cls) in colored where range.location + range.length <= storage.length {
                        storage.addAttribute(.foregroundColor, value: cls.nsColor(dark: dark), range: range)
                    }
                    storage.endEditing()
                }
            }
            highlightWork = work
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.25, execute: work)
        }

        func highlightNow() {
            applyBaseAttributes()
            scheduleHighlight()
        }
    }

    /// Line-number ruler: content-height view (scrolls with the text),
    /// one binary-searched number per visible line fragment.
    final class LineNumberRuler: NSRulerView {
        weak var textView: NSTextView?
        weak var coordinator: Coordinator?

        override var isFlipped: Bool { true }

        init() {
            super.init(scrollView: nil, orientation: .verticalRuler)
        }

        required init(coder: NSCoder) { fatalError("init(coder:) is not used") }

        override func drawHashMarksAndLabels(in rect: NSRect) {
            guard let tv = textView, let lm = tv.layoutManager,
                  let container = tv.textContainer, let c = coordinator else { return }

            // Ruler stays TRANSPARENT — a tinted band here reads as a
            // phantom panel between the tree and the code.

            let visible = tv.visibleRect
            let glyphRange = lm.glyphRange(forBoundingRect: visible, in: container)
            guard glyphRange.length > 0 || tv.string.isEmpty else { return }

            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: NSColor.secondaryLabelColor,
            ]

            var charIndex = lm.characterIndexForGlyph(at: glyphRange.location)
            lm.enumerateLineFragments(forGlyphRange: glyphRange) { fragRect, _, _, fragGlyphRange, _ in
                _ = fragGlyphRange
                let number = c.lineNumber(for: charIndex)
                let size = (String(number) as NSString).size(withAttributes: attrs)
                let y = fragRect.minY + tv.textContainerInset.height
                (String(number) as NSString).draw(
                    at: NSPoint(x: self.ruleThickness - size.width - 8, y: y + (fragRect.height - size.height) / 2),
                    withAttributes: attrs)
                // advance to the next line's first character
                let ns = tv.string as NSString
                let nl = ns.rangeOfCharacter(from: .newlines,
                                             range: NSRange(location: charIndex, length: ns.length - charIndex))
                if nl.location != NSNotFound {
                    charIndex = nl.location + nl.length
                }
            }

            // The trailing empty line after a final newline.
            if lm.extraLineFragmentTextContainer != nil {
                let number = c.lineNumber(for: (tv.string as NSString).length)
                let size = (String(number) as NSString).size(withAttributes: attrs)
                let y = lm.extraLineFragmentRect.minY + tv.textContainerInset.height
                (String(number) as NSString).draw(
                    at: NSPoint(x: ruleThickness - size.width - 8, y: y + (lm.extraLineFragmentRect.height - size.height) / 2),
                    withAttributes: attrs)
            }
        }
    }
}
