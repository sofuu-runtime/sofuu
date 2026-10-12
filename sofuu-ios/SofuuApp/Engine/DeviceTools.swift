import Foundation
import UIKit
import SwiftUI

/// Native capabilities the agent can call, implemented by the host.
///
/// The runtime's QuickJS sandbox cannot reach UIKit, so anything
/// device-specific has to be answered from here through
/// `SofuuBridge.registerHostTool`. That is the whole point of the host-tool
/// ABI, and on iOS it is what makes the app able to do things a terminal
/// cannot rather than only fewer things.
///
/// Every tool here is registered as a normal agent tool, so it shows up in
/// `/tools`, honours the permission profile (`plan` mode refuses writes,
/// `prompt` mode asks first), and lands in the normal tool timeline.
///
/// Scope is deliberately permission-light: a tool that needed a permission
/// the user might refuse is not registered rather than registered and
/// silently failing.
///
/// The iOS set is the same four the Android app ships, mapped to the
/// platform: no battery service, no vibrator API to call directly, so
/// `vibrate` goes through UIKit's feedback generators, and `device_info`
/// reports the things an iPhone can actually answer.
enum DeviceTools {

    /// One tool: its schema plus the handler that answers a call.
    struct Registration {
        let name: String
        let description: String
        let parameters: String
        let handler: (JSONObject) -> String
    }

    /// The set to hand to `SofuuBridge.registerHostTool`.
    static func all() -> [Registration] {
        [
            Registration(
                name: "device_info",
                description: """
                    Report this phone's model, iOS version, battery level, \
                    free storage and whether the process is running in the \
                    background. Use when you need to know what you are running on.
                    """,
                parameters: #"{"type":"object","properties":{},"required":[]}"#
            ) { _ in deviceInfo() },

            Registration(
                name: "open_url",
                description: """
                    Open a web page in the phone's browser. Only use it when \
                    the user asks you to show them something.
                    """,
                parameters: """
                    {"type":"object","properties":{"url":{"type":"string","description":\
                    "Absolute http(s) URL to open"}},"required":["url"]}
                    """
            ) { args in openURL(args) },

            Registration(
                name: "share_text",
                description: """
                    Open the iOS share sheet so the user can send text (a link, \
                    a note, a summary) to another app.
                    """,
                parameters: """
                    {"type":"object","properties":{"text":{"type":"string","description":\
                    "Text to share"}},"required":["text"]}
                    """
            ) { args in shareText(args) },

            Registration(
                name: "vibrate",
                description: """
                    Buzz the phone briefly to get the user's attention. Use \
                    sparingly — a haptic for every finished turn trains people \
                    to ignore it.
                    """,
                parameters: """
                    {"type":"object","properties":{"style":{"type":"string",\
                    "description":"light | medium | heavy | success | warning | error"}},"required":[]}
                    """
            ) { args in vibrate(args) },
        ]
    }

    /// Attach every tool to a runtime. Engine thread only.
    @discardableResult
    static func registerAll(into bridge: SofuuBridge) -> [String] {
        var registered: [String] = []
        for r in all() {
            let ok = bridge.registerHostTool(
                name: r.name,
                description: r.description,
                parametersJson: r.parameters
            ) { argsJson, _ in
                do {
                    let raw = argsJson.isEmpty ? "{}" : argsJson
                    guard let data = raw.data(using: .utf8),
                          let parsed = try? JSONSerialization.jsonObject(with: data),
                          let obj = parsed as? JSONObject else {
                        return err("arguments were not a JSON object")
                    }
                    return r.handler(obj)
                } catch {
                    // A throwing tool would surface to the model as a hang
                    // (the trampoline returns nil on error), so report it
                    // as a failed call instead.
                    return err("invalid arguments: \(error.localizedDescription)")
                }
            }
            if ok { registered.append(r.name) }
        }
        return registered
    }

    // MARK: - Handlers

    private static func deviceInfo() -> String {
        var d: JSONObject = [:]
        d["model"] = deviceModelName()
        d["system"] = "iOS \(UIDevice.current.systemVersion)"
        d["is_simulator"] = isSimulator

        if #available(iOS 16.0, *) {
            // ProcessInfo reports the OS's own view of the battery; it is
            // only meaningful while the app is foregrounded, which is the
            // only time this tool can be called anyway.
            UIDevice.current.isBatteryMonitoringEnabled = true
            let level = UIDevice.current.batteryLevel
            if level >= 0 { d["battery_percent"] = Int((level * 100).rounded()) }
            let state = UIDevice.current.batteryState
            if state != .unknown { d["battery_state"] = batteryStateName(state) }
        }
        if #available(iOS 16.0, *) {
            UIDevice.current.isBatteryMonitoringEnabled = false
        }

        let url = URL(fileURLWithPath: NSHomeDirectory())
        if let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let free = values.volumeAvailableCapacityForImportantUsage {
            d["storage_free_gb"] = (Double(free) / 1_073_741_824.0).rounded(toPlaces: 2)
        }
        d["foreground"] = UIApplication.shared.applicationState == .active
        return jsonEncode(d)
    }

    private static func openURL(_ args: JSONObject) -> String {
        guard let raw = args["url"] as? String, raw.hasPrefix("http://") || raw.hasPrefix("https://") else {
            return err("open_url needs an http(s) URL")
        }
        guard let url = URL(string: raw) else { return err("could not parse that URL") }
        // The agent runs on the engine thread; UIKit work has to hop to the
        // main thread, and the result is reported synchronously because the
        // tool contract wants one answer per call.
        let opened = DispatchQueue.main.sync { () -> Bool in
            guard UIApplication.shared.canOpenURL(url) else { return false }
            UIApplication.shared.open(url, options: [:], completionHandler: nil)
            return true
        }
        return opened ? ok_("opened \(raw)") : err("iOS would not open that URL")
    }

    private static func shareText(_ args: JSONObject) -> String {
        guard let text = args["text"] as? String, !text.isEmpty else {
            return err("share_text needs text")
        }
        let ok = DispatchQueue.main.sync { () -> Bool in
            guard let root = UIApplication.shared.topViewController() else { return false }
            let sheet = UIActivityViewController(activityItems: [text], applicationActivities: nil)
            sheet.popoverPresentationController?.sourceView = root.view
            sheet.popoverPresentationController?.sourceRect = CGRect(
                x: root.view.bounds.midX, y: root.view.bounds.midY, width: 0, height: 0
            )
            sheet.popoverPresentationController?.permittedArrowDirections = []
            root.present(sheet, animated: true)
            return true
        }
        return ok ? ok_("share sheet opened") : err("could not present the share sheet")
    }

    private static func vibrate(_ args: JSONObject) -> String {
        let style = (args["style"] as? String ?? "medium").lowercased()
        DispatchQueue.main.async {
            switch style {
            case "light": UIImpactFeedbackGenerator(style: .light).impactOccurred()
            case "heavy": UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
            case "success": UINotificationFeedbackGenerator().notificationOccurred(.success)
            case "warning": UINotificationFeedbackGenerator().notificationOccurred(.warning)
            case "error": UINotificationFeedbackGenerator().notificationOccurred(.error)
            default: UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            }
        }
        return ok_("haptic \(style)")
    }

    // MARK: - Helpers

    private static func batteryStateName(_ s: UIDevice.BatteryState) -> String {
        switch s {
        case .charging: return "charging"
        case .full: return "full"
        case .unplugged: return "unplugged"
        default: return "unknown"
        }
    }

    /// The marketing name ("iPhone 15 Pro"), which `uname` does not give.
    private static func deviceModelName() -> String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafePointer(to: &info.machine) { p in
            String(cString: UnsafeRawPointer(p).assumingMemoryBound(to: CChar.self))
        }
        var name: String
        switch machine {
        case "iPhone14,2": name = "iPhone 14 Pro"
        case "iPhone14,3": name = "iPhone 14 Pro Max"
        case "iPhone14,4": name = "iPhone 14"
        case "iPhone14,5": name = "iPhone 14 Plus"
        case "iPhone15,2": name = "iPhone 15 Pro"
        case "iPhone15,3": name = "iPhone 15 Pro Max"
        case "iPhone15,4": name = "iPhone 15"
        case "iPhone15,5": name = "iPhone 15 Plus"
        case "iPhone16,1": name = "iPhone 16 Pro"
        case "iPhone16,2": name = "iPhone 16 Pro Max"
        default: name = machine
        }
        return name
    }

    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    private static func ok_(_ note: String) -> String {
        var d: JSONObject = [:]
        d["ok"] = true
        d["note"] = note
        return jsonEncode(d)
    }

    private static func err(_ message: String) -> String {
        var d: JSONObject = [:]
        d["error"] = message
        return jsonEncode(d)
    }
}

extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}

private extension UIApplication {
    /// The controller a modal should be presented from. `keyWindow` is the
    /// supported answer on the deployment target this app builds for.
    func topViewController() -> UIViewController? {
        let scene = connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }
        var top = scene?.windows.first { $0.isKeyWindow }?.rootViewController
            ?? scene?.windows.first?.rootViewController
        while let presented = top?.presentedViewController { top = presented }
        return top
    }
}