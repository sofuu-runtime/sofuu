import SwiftUI

/// The app entry point — the iOS twin of the Android app's `MainActivity`
/// (edge-to-edge host, permission asks, theme root).
///
/// Three responsibilities, and nothing else:
///
///   1. Own the one `AppStore` (which owns the one engine thread).
///   2. Wrap the tree in `SofuuTheme`, resolving `light | dark | system`
///      against the device's current scheme.
///   3. Tell the store whether anyone is actually looking, so a turn that
///      finishes while the app is backgrounded posts a notification instead
///      of silently updating a view nobody can see.
@main
struct SofuuApp: App {
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup {
            Root(store: store)
        }
    }
}

private struct Root: View {
    @ObservedObject var store: AppStore

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.colorScheme) private var systemScheme

    var body: some View {
        SofuuTheme(dark: store.resolvedDark(systemScheme == .dark)) {
            Shell(store: store)
        }
        .onAppear {
            // The Android app asks for POST_NOTIFICATIONS on first launch;
            // this is the same ask at the same moment. Idempotent, so the
            // repeat on every appear is harmless.
            Notifier.requestAuthorization()
            store.notifier = { title, body in
                Notifier.post(title: title, body: body)
            }
        }
        .onChange(of: scenePhase) { phase in
            // Background turns keep running — the engine owns its own
            // thread and the process is only suspended once the system
            // decides to. All this flag changes is whether a finished turn
            // (or a waiting approval) also raises a notification.
            store.foreground = (phase == .active)
        }
    }
}
