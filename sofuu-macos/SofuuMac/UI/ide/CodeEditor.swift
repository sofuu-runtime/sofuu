import SwiftUI
import AppKit

/// Redraw pulse for the line-number gutter. The Canvas reads ALL geometry
/// fresh from the coordinator at draw time — nothing is latched, so
/// transient mid-layout measurements can never stick (the old design
/// published first/count/step once and froze whatever it caught; numbers
/// then only corrected themselves after a scroll).
final class GutterModel: ObservableObject {}

/// The IDE code viewport — an editable NSTextView wrapped for SwiftUI,
/// with the line-number gutter as a SEPARATE SwiftUI Canvas to its left.
///
/// Why not NSRulerView: under TextKit 2 its fragment geometry misbehaves
/// when scrolled (positions come back as document-top geometry, and
/// enumeration stops mid-document) — the ruler drew numbers at the
/// viewport top and then stopped, and erasing within its draw never
/// covered the text reliably. The gutter here is pure arithmetic: the
/// container never soft-wraps (unbounded width, monospaced font), so
/// source line n sits at y = topInset + (n-1) * step. The step is
/// calibrated ONCE from real fragments; scroll changes just re-publish
/// the visible range.
///
/// Highlighting is attribute-only, so the cursor and undo never jump.
struct CodeEditor: NSViewRepresentable {
    /// Live buffer text (two-way; the AppStore holds it across tab switches).
    @Binding var text: String
    let path: String
    let dark: Bool
    let store: AppStore
    var editable: Bool = true
    /// Fires (debounced, main) with the final text after edits settle.
    var onEdited: (String) -> Void

    static let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
    /// Shared gutter geometry — MUST match `tv.textContainerInset.height`.
    static let gutterTopInset: CGFloat = 6
    static let gutterWidth: CGFloat = 46

    func makeCoordinator() -> Coordinator {
        Coordinator(store: store)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = NSTextView()
        tv.font = Self.font
        tv.isEditable = editable
        tv.isRichText = false            // plain text only — attributes are ours
        tv.allowsUndo = true
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = true
        // Both dims are sized manually (sizeToFitContent) from the laid-
        // out content — the standard resizable-text-view contract.
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
        context.coordinator.push()

        let c = context.coordinator
        c.textView = tv
        c.onEdited = onEdited
        c.lang = Syntax.Lang.guess(path)
        c.isDark = dark
        c.rebuildLineIndex()
        c.applyBaseAttributes()
        c.highlightNow()
        c.push()
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
            c.highlightNow()   // highlight + sizing ride the debounced pass
            c.push()
        }
        if c.isDark != dark {
            c.isDark = dark
            c.highlightNow()
        }
        tv.isEditable = editable
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        let store: AppStore
        let gutter: GutterModel
        weak var textView: NSTextView?
        weak var scrollView: NSScrollView?
        private var scrollObserver: NSObjectProtocol?

        var isDark = true
        var onEdited: ((String) -> Void)?
        var lang: Syntax.Lang = .plain

        /// Character offset of every line start (gutter + ruler math).
        var lineStarts: [Int] = [0]
        /// Calibrated uniform line height (measured once at the document
        /// top; the container never soft-wraps, so rows are uniform).
        var lineStep: CGFloat?

        private var highlightWork: DispatchWorkItem?

        init(store: AppStore) {
            self.store = store
            self.gutter = store.ideGutter
            super.init()
            // The gutter Canvas reads geometry from this coordinator at
            // draw time — register so views can reach it, then pulse once.
            MainActor.assumeIsolated {
                store.ideCoordinator = self
            }
            gutter.objectWillChange.send()
        }

        deinit {
            if let scrollObserver {
                NotificationCenter.default.removeObserver(scrollObserver)
            }
        }

        /// Wire the gutter to clip-view bounds changes (fires on every
        /// scroll tick; push() only republished at line crossings).
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
            scheduleHighlight()   // sizing rides the same debounced pass —
            push()                // no full-document layout per keystroke
            onEdited?(textView?.string ?? "")
        }

        /// Grow the text view to its laid-out content size (standard
        /// resizable-text-view contract) — keeps the horizontal scrollbar
        /// honest instead of pinning to an unbounded container width.
        func sizeToFitContent() {
            guard let tv = textView, let lm = tv.layoutManager,
                  let container = tv.textContainer, let sv = scrollView else { return }
            let used = lm.usedRect(for: container)
            let width = max(
                used.width + container.lineFragmentPadding * 2 + tv.textContainerInset.width * 2,
                sv.contentSize.width)
            let height = max(
                used.height + tv.textContainerInset.height * 2,
                sv.contentSize.height)
            tv.setFrameSize(NSSize(width: width, height: height))
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

        /// Publish the visible gutter range. Cheap: only writes (and only
        /// the Canvas observes) when the top line or range really changes.
        func push() {
            if let tv = textView, let lm = tv.layoutManager, !tv.string.isEmpty {
                calibrateStep(ns: tv.string as NSString, lm: lm)
            }
            gutter.objectWillChange.send()
        }

        /// Measure the uniform row pitch ONCE from real fragments at the
        /// document top (scrolled queries are unreliable under TextKit 2).
        /// The container never soft-wraps, so source lines are uniform rows.
        func calibrateStep(ns: NSString, lm: NSLayoutManager) {
            guard lineStep == nil else { return }
            var tops: [CGFloat] = []
            lm.enumerateLineFragments(
                forGlyphRange: NSRange(location: 0, length: min(4, ns.length))
            ) { r, _, _, _, _ in
                if tops.last.map({ abs(r.minY - $0) > 0.5 }) ?? true { tops.append(r.minY) }
            }
            if tops.count >= 2 { lineStep = tops[1] - tops[0] }
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
                    self.sizeToFitContent()
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
}

/// The line-number gutter: a Canvas that reads ALL geometry fresh from
/// the coordinator at draw time (clip-view scroll offset, viewport height,
/// live line count) — transient mid-layout measurements can never stick.
struct LineGutter: View {
    @ObservedObject var pulse: GutterModel   // redraw trigger only
    let coordinator: CodeEditor.Coordinator? // nil until the editor ties in

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .background(c.bgRaised)
        .onAppear { pulse.objectWillChange.send() }
    }

    private func draw(ctx: GraphicsContext, size: CGSize) {
        guard let coordinator, let tv = coordinator.textView, let lm = tv.layoutManager,
              let sv = coordinator.scrollView else { return }
        let ns = tv.string as NSString
        guard ns.length > 0 else { return }
        if coordinator.lineStep == nil {
            coordinator.calibrateStep(ns: ns, lm: lm)
        }
        let step = coordinator.lineStep ?? lm.defaultLineHeight(for: tv.font ?? CodeEditor.font)
        guard step > 0 else { return }

        // Fresh at draw time — never latched from a publish.
        let scrollY = sv.contentView.bounds.origin.y
        let inset = CodeEditor.gutterTopInset
        let total = coordinator.lineStarts.count

        let font = Font.system(size: 11, design: .monospaced).monospacedDigit()
        let lineColor = Color.primary.opacity(0.45)

        // Strictly inside the visible band; numbers stop at the file end.
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
            }
            i += 1
        }
    }
}
