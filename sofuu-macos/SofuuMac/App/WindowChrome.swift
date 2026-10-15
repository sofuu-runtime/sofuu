import AppKit
import SwiftUI

/// Window chrome fixes for the hidden-title design — the SwiftUI
/// `.windowStyle(.hiddenTitleBar)` alone strips the title bar entirely:
/// you can't drag the window from its top strip and double-click doesn't
/// zoom. Restoring the standard chrome invisibly gives both, plus content
/// that flows edge-to-edge:
///
///   * `.titled` + `.fullSizeContentView` → a real (native-behaving)
///     title bar the system draws nothing for: drag + double-click-zoom
///     work on the top strip exactly like any Mac app;
///   * `titlebarAppearsTransparent` + hidden title → nothing is painted
///     there, so our content sits edge to edge under it;
///   * `isMovableByWindowBackground` → empty background areas drag too.
///
/// Interactive views (pills, buttons, text fields) keep their clicks —
/// AppKit routes hits to content controls first.
struct WindowChrome: NSViewRepresentable {

    final class Probe: NSView {
        override func viewDidMoveToWindow() {
            guard let window else { return }
            DispatchQueue.main.async {
                window.styleMask.insert(.titled)
                window.styleMask.insert(.fullSizeContentView)
                window.titlebarAppearsTransparent = true
                window.titleVisibility = .hidden
                window.isMovableByWindowBackground = true
            }
        }
    }

    func makeNSView(context: Context) -> NSView {
        let v = Probe()
        v.frame = CGRect(x: 0, y: 0, width: 1, height: 1)
        return v
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
