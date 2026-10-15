import AppKit
import SwiftUI

/// The app entry — the mac twin of the iOS `SofuuApp`:
///
///   1. Owns the one `AppStore` (which owns the one engine thread).
///   2. Wraps the tree in `SofuuTheme`, resolving light/dark/system against
///      the system appearance (the native window follows through
///      `preferredColorScheme`).
///   3. Tells the store whether anyone is actually looking — via
///      NSApplication activation, not scenePhase, which means nothing on
///      the desktop.
///   4. The menu bar accelerators (⌘N ⌘, ⌘. ⌘K ⌘B ⌘= ⌘− ⌘0) ride the
///      `SofuuMenus` commands; a local key monitor covers the Esc-closes-overlay rule
///      overlay and the zoom/side toggles when no menu item owns them.
@main
struct SofuuMacApp: App {
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup {
            Root(store: store)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1240, height: 820)
        .commands {
            SofuuMenus(store: store)
        }
    }
}

/// The window content: theme root + document zoom + lifecycle wiring.
private struct Root: View {
    @ObservedObject var store: AppStore

    @Environment(\.colorScheme) private var systemScheme
    @State private var keys: GlobalKeyMonitor?

    var body: some View {
        GeometryReader { geo in
            SofuuTheme(dark: store.resolvedDark(systemScheme == .dark)) {
                // FullBleed paints the app surface across the WHOLE window —
                // including the invisible title-bar band. Without it the
                // band shows the window's own (black) background above the
                // shell: the "dead strip". The GeometryReader's proposal is
                // the full window bounds because .ignoresSafeArea() below
                // lifts the safe-area insets.
                FullBleed {
                    Shell(store: store)
                        // The desktop's document zoom: the shell scales while
                        // the container holds the unzoomed frame.
                        .scaleEffect(store.state.zoom, anchor: .topLeading)
                        .frame(
                            width: geo.size.width / store.state.zoom,
                            height: geo.size.height / store.state.zoom,
                            alignment: .topLeading)
                        .clipped()
                }
            }
        }
        .ignoresSafeArea()
        .onAppear {
            CrashLog.install()
            Notifications.requestAuthorization()
            store.notifier = { title, body in Notifications.post(title: title, body: body) }
            if keys == nil { keys = GlobalKeyMonitor(store: store) }
            // Fill the screen at launch (menu bar + dock stay visible —
            // standard maximize, not full-screen mode).
            DispatchQueue.main.async {
                guard let window = NSApp.windows.first(where: { $0.canBecomeMain && !$0.isSheet }) else { return }
                if let screen = window.screen ?? NSScreen.main {
                    window.setFrame(screen.visibleFrame, display: true)
                }
            }
        }
        .onAppear {
            installFocusWatchers()
        }
    }

    // Focus-driven notifications: a finished turn or a waiting approval
    // only notifies when no Sofuu window is frontmost (the Tauri host.rs
    // rule). Plus the 5-minute model-cache prefetch refresh.
    private func installFocusWatchers() {
        NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { _ in Task { @MainActor in store.foreground = true } }
        NotificationCenter.default.addObserver(
            forName: NSApplication.willResignActiveNotification, object: nil, queue: .main
        ) { _ in Task { @MainActor in store.foreground = false } }

        Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { _ in
            Task { @MainActor in store.refreshModelCache() }
        }
    }
}

/// Paints `bgApp` edge-to-edge (the invisible title-bar band included) and
/// hosts the window-chrome probe.
private struct FullBleed<Content: View>: View {
    @ViewBuilder var content: Content

    @Environment(\.sofuuColors) private var c

    var body: some View {
        ZStack {
            Rectangle().fill(c.bgApp)
                .ignoresSafeArea()
            content
        }
        .background(WindowChrome())
        .ignoresSafeArea()
    }
}

/// Esc closes the top overlay (the desktop's Esc rule) — consumed only when
/// an overlay was actually open. Field editors keep their own Esc behaviour
/// otherwise. Lifetime matches the app process; one instance per Root.
final class GlobalKeyMonitor {
    private var monitor: Any?

    init(store: AppStore) {
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            guard event.keyCode == 53 else { return event } // esc
            let hasOverlay = MainActor.assumeIsolated {
                store.state.settingsOpen || store.state.preview != nil
                    || store.state.sheet != nil || store.state.drawerOpen
            }
            if hasOverlay {
                Task { @MainActor in _ = store.closeTopOverlay() }
                return nil
            }
            return event
        }
    }

    deinit {
        if let monitor { NSEvent.removeMonitor(monitor) }
    }
}

// MARK: - Menus (the NSMenu replacement — desktop lib.rs parity)

/// Sofuu | File(New Chat) | Edit(auto) | Session(Stop/Compact) |
/// View(Zoom/Sidebar) | Window.
struct SofuuMenus: Commands {
    @ObservedObject var store: AppStore

    var body: some Commands {
        // The App menu's Settings (⌘,) opens the in-window settings overlay,
        // like the Tauri app's menu item.
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { store.openSettings(true) }
                .keyboardShortcut(",")
        }
        CommandGroup(replacing: .newItem) {
            Button("New Chat") { store.newChat() }
                .keyboardShortcut("n")
        }
        CommandMenu("Session") {
            Button("Stop") { store.cancel() }
                .keyboardShortcut(".", modifiers: .command)
            Button("Compact Session") { store.compactSession() }
                .keyboardShortcut("k", modifiers: .command)
        }
        CommandMenu("View") {
            Button("Toggle Sidebar") { store.toggleSidebar() }
                .keyboardShortcut("b", modifiers: .command)
            Divider()
            Button("Zoom In") { store.zoomBy(0.1) }
                .keyboardShortcut("=", modifiers: .command)
            Button("Zoom Out") { store.zoomBy(-0.1) }
                .keyboardShortcut("-", modifiers: .command)
            Button("Actual Size") { store.zoomReset() }
                .keyboardShortcut("0", modifiers: .command)
        }
    }
}
