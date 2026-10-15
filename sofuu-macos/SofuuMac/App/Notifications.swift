import AppKit
import Foundation
import UserNotifications

/// System notifications for approvals and finished turns — the mac twin of
/// the iOS `Notifier`, plus the two desktop-only side effects the Tauri
/// `host.rs` performed off the event stream:
///   * `approval_request` → dock bounce (`NSApp.requestUserAttention`),
///   * `done` → notification ONLY when no Sofuu window has focus.
enum Notifications {

    static func requestAuthorization() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    static func post(title: String, body: String) {
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            guard settings.authorizationStatus == .authorized
                    || settings.authorizationStatus == .provisional else { return }
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
        }
    }

    /// The Tauri app's dock bounce on a waiting approval (host.rs:149-163).
    /// Main thread only — call from the store's main-actor event reducer.
    static func requestUserAttention() {
        guard !NSApp.isActive else { return }
        NSApp.requestUserAttention(.criticalRequest)
    }
}
