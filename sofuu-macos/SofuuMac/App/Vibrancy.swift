import SwiftUI
import AppKit

/// The sidebar's vibrancy layer — the SwiftUI twin of the Tauri app's
/// `window_vibrancy::apply_vibrancy(UnderWindowBackground, ...)`: a native
/// NSVisualEffectView behind the window content, over which the sidebar
/// paints its translucent warm tint (the desktop's CSS `backdrop-filter
/// blur(30px) saturate(190%)` stand-in).
struct Vibrancy: NSViewRepresentable {
    var material: NSVisualEffectView.Material = .underWindowBackground
    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {
        view.material = material
        view.blendingMode = blendingMode
    }
}
