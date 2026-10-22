import AppKit
import Combine
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
    @State private var watchersInstalled = false
    @State private var badgeSub: AnyCancellable?

    var body: some View {
        GeometryReader { geo in
            ZStack {
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
                // Platform overlays ride above the themed shell, outside the
                // document zoom (they are chrome, not content). They get
                // their own SofuuTheme — as siblings of the shell they'd
                // miss its environment and fall back to light colors.
                SofuuTheme(dark: store.resolvedDark(systemScheme == .dark)) {
                    ZStack {
                        if store.state.onboardingVisible {
                            WelcomeCard(store: store)
                        }
                        if store.state.shortcutHelpVisible {
                            ShortcutSheet(store: store)
                        }
                    }
                }
            }
        }
        .ignoresSafeArea()
        // Finder "Open With" / dock drops / sofuu:// deep links all land
        // here (declared in Info.plist: CFBundleDocumentTypes + URL type).
        .onOpenURL { store.openIncoming($0) }
        .onAppear {
            CrashLog.install()
            Notifications.requestAuthorization()
            store.notifier = { title, body in Notifications.post(title: title, body: body) }
            if keys == nil { keys = GlobalKeyMonitor(store: store) }
            store.maybeShowOnboarding()
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
    // rule). Plus the 5-minute model-cache prefetch refresh. Registered
    // ONCE — onAppear can re-fire, and stacked timers multiply engine
    // cache refreshes forever.
    private func installFocusWatchers() {
        guard !watchersInstalled else { return }
        watchersInstalled = true

        // macOS Services: "Send to Sofuu" in the app's Services submenu
        // (NSServices in Info.plist). NSApp retains the provider object.
        // Set once at launch — the Services menu re-reads it on activation
        // (the old NSUpdateServicesMenu global no longer exists in this SDK).
        NSApp.servicesProvider = ServicesBridge(store: store)

        // Dock badge = pending approvals. Cleared again when the queue
        // drains (allowing or denying the last card). objectWillChange
        // fires BEFORE the mutation lands, so hop one tick forward via
        // receive(on:) to see the new count.
        badgeSub = store.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak store] _ in
                let n = store?.state.approvals.count ?? 0
                NSApp.dockTile.badgeLabel = n > 0 ? String(n) : nil
            }

        NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { _ in Task { @MainActor in store.foreground = true } }
        NotificationCenter.default.addObserver(
            forName: NSApplication.willResignActiveNotification, object: nil, queue: .main
        ) { _ in Task { @MainActor in store.foreground = false } }

        Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { _ in
            Task { @MainActor in store.refreshModelCache() }
        }

        // Pump watchdog: a wedged or exited engine thread must surface as
        // a banner, not as a silently dead app.
        Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { _ in
            Task { @MainActor in store.checkPumpHealth() }
        }

        // Language servers hold hundreds of MB each — they must die with
        // the app, not linger as orphans after quit.
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { _ in LSPRegistry.shared.stopAll() }

        // One window by design: the store, engine pump, HUD panels and
        // escape monitor all assume it. A second SwiftUI window (Window
        // menu, state restore, scripts) is closed with an explanation
        // instead of misbehaving silently.
        NotificationCenter.default.addObserver(
            forName: NSWindow.didBecomeKeyNotification, object: nil, queue: .main
        ) { note in
            let mains = NSApp.windows.filter {
                $0.isVisible && $0.canBecomeMain && !($0 is NSPanel)
            }
            guard mains.count > 1, let extra = note.object as? NSWindow,
                  extra !== mains.first else { return }
            extra.orderOut(nil)
            let alert = NSAlert()
            alert.messageText = "Sofuu is a one-window app"
            alert.informativeText = "Sessions, engine state and the IDE live in a single window. The extra window was closed."
            alert.addButton(withTitle: "OK")
            alert.runModal()
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

/// The macOS Services provider behind the "Send to Sofuu" menu item
/// (NSServices in Info.plist declares `NSMessage sendText`). The system
/// calls the selector with a pasteboard owned by the lending app, on an
/// arbitrary thread — marshal onto main before touching the store.
final class ServicesBridge: NSObject {
    private let store: AppStore

    init(store: AppStore) { self.store = store }

    @objc func sendText(_ pboard: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>?) {
        guard let text = pboard.string(forType: .string), !text.isEmpty else { return }
        DispatchQueue.main.async { self.store.receiveSharedText(text) }
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
            let hudOpen = MainActor.assumeIsolated {
                IDEHUD.promptPanel?.isVisible == true || IDEHUD.listPanel?.isVisible == true
                    || IDEHUD.diffPanel?.isVisible == true
            }
            if hudOpen {
                MainActor.assumeIsolated {
                    IDEHUD.hidePrompt()
                    IDEHUD.hideList()
                }
                return nil
            }
            let hasOverlay = MainActor.assumeIsolated {
                store.state.settingsOpen || store.state.preview != nil
                    || store.state.sheet != nil
                    || store.state.shortcutHelpVisible || store.state.onboardingVisible
                    || store.state.chatSearchOpen || store.state.gitPanelOpen
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
            Button(L10n.t("menu.newChat")) { store.newChat() }
                .keyboardShortcut("n")
            Divider()
            Button(L10n.t("menu.saveFile")) { store.saveActiveTab() }
                .keyboardShortcut("s")
            Divider()
            Button(L10n.t("menu.findInFile")) {
                NotificationCenter.default.post(name: .init("IDEFindToggle"), object: nil)
            }
                .keyboardShortcut("f")
            Button(L10n.t("menu.searchProject")) {
                NotificationCenter.default.post(name: .init("IDESearchToggle"), object: nil)
            }
                .keyboardShortcut("f", modifiers: [.command, .shift])
            Divider()
            Button(L10n.t("menu.renameSymbol")) { store.ideCoordinator?.renameAtCaret() }
                .keyboardShortcut("r")
            Button(L10n.t("menu.findReferences")) {
                NotificationCenter.default.post(name: .init("IDERefToggle"), object: nil)
            }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button(L10n.t("menu.findSymbol")) {
                NotificationCenter.default.post(name: .init("IDESymbolToggle"), object: nil)
            }
                .keyboardShortcut("p", modifiers: [.command, .shift])
            Button(L10n.t("menu.goToLine")) {
                NotificationCenter.default.post(name: .init("IDEGoLineToggle"), object: nil)
            }
                .keyboardShortcut("l")
            Button(L10n.t("menu.formatFile")) {
                if let path = store.state.activeTab { store.formatFile(path) }
            }
                .keyboardShortcut("f", modifiers: [.command, .option, .shift])
            Button(L10n.t("menu.outline")) {
                NotificationCenter.default.post(name: .init("IDEOutlineToggle"), object: nil)
            }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            Divider()
            Button(L10n.t("menu.searchChat")) { store.setChatSearchOpen(true) }
                .keyboardShortcut("s", modifiers: [.command, .shift])
            Button(L10n.t("menu.exportMarkdown")) { store.exportSessionMarkdown() }
                .keyboardShortcut("e", modifiers: [.command, .shift])
        }
        CommandMenu("Session") {
            Button(L10n.t("menu.stop")) { store.cancel() }
                .keyboardShortcut(".", modifiers: .command)
            Button(L10n.t("menu.compact")) { store.compactSession() }
                .keyboardShortcut("k", modifiers: .command)
            Divider()
            // Lives here, not in CommandMenu("View"): SwiftUI drops a
            // custom View menu on the system View menu's altar on this
            // macOS (its items never render). Session renders reliably.
            Button(L10n.t(store.state.ideMode ? "menu.exitIdeMode" : "menu.ideMode")) { store.toggleIdeMode() }
                .keyboardShortcut("i", modifiers: [.command, .shift])
            Divider()
            Button(L10n.t("menu.shortcuts")) { store.showShortcutHelp() }
                .keyboardShortcut("/", modifiers: [.command, .shift])
            Button(L10n.t("menu.restartEngine")) { store.restartEngineForEnv() }
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
