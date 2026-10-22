import AppKit
import Foundation
import IOKit.ps

/// Native capabilities the agent can call, implemented by the host — the
/// mac set (the counterpart of the iOS app's `DeviceTools`): the QuickJS
/// sandbox cannot reach AppKit either, so anything desktop-specific is
/// answered here through `SofuuBridge.registerHostTool`.
///
/// Same four the iOS/Android apps ship, mapped to the platform: `vibrate`
/// has no desktop analogue and is dropped; `device_info` reports Mac
/// facts (battery via IOKit, model via the SMC marketing name when
/// available); `share_text` becomes an NSSharingServicePicker anchored to
/// the window's content view.
///
/// Every tool registers as a normal agent tool: it shows up in /tools,
/// honours the permission profile, and lands in the tool timeline.
enum MacTools {

    struct Registration {
        let name: String
        let description: String
        let parameters: String
        let handler: (JSONObject) -> String
    }

    static func all() -> [Registration] {
        [
            Registration(
                name: "device_info",
                description: """
                    Report this Mac's model, macOS version, battery level \
                    (when a battery is present), free storage and whether \
                    the app is frontmost. Use when you need to know what \
                    you are running on.
                    """,
                parameters: #"{"type":"object","properties":{},"required":[]}"#
            ) { _ in deviceInfo() },

            Registration(
                name: "open_url",
                description: """
                    Open a web page in the user's default browser. Only use \
                    it when the user asks you to show them something.
                    """,
                parameters: """
                    {"type":"object","properties":{"url":{"type":"string","description":\
                    "Absolute http(s) URL to open"}},"required":["url"]}
                    """
            ) { args in openURL(args) },

            Registration(
                name: "share_text",
                description: """
                    Open the macOS sharing picker so the user can send text \
                    (a link, a note, a summary) to another app.
                    """,
                parameters: """
                    {"type":"object","properties":{"text":{"type":"string","description":\
                    "Text to share"}},"required":["text"]}
                    """
            ) { args in shareText(args) },
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
                let raw = argsJson.isEmpty ? "{}" : argsJson
                guard let data = raw.data(using: .utf8),
                      let parsed = try? JSONSerialization.jsonObject(with: data),
                      let obj = parsed as? JSONObject else {
                    return err("arguments were not a JSON object")
                }
                return r.handler(obj)
            }
            if ok { registered.append(r.name) }
        }
        return registered
    }

    // MARK: - Handlers

    /// The host-tool callback runs on the RUNTIME's thread; every AppKit
    /// read below hops to the main queue via sync — the engine thread is
    /// never the main thread, so this cannot self-deadlock (same contract
    /// the iOS DeviceTools documented).
    private static func deviceInfo() -> String {
        let d: JSONObject = DispatchQueue.main.sync {
            var out: JSONObject = [:]
            out["model"] = deviceModelName()
            out["system"] = "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
            out["foreground"] = NSApp.isActive
            // A desktop can be battery-less — the power-sources read
            // legitimately returns nothing there, so the keys are absent
            // rather than a made-up zero.
            if let psInfo = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
               let sources = IOPSCopyPowerSourcesList(psInfo)?.takeRetainedValue() as? [CFTypeRef] {
                for src in sources {
                    if let desc = IOPSGetPowerSourceDescription(psInfo, src)?.takeUnretainedValue() as? [String: Any] {
                        if let level = desc[kIOPSCurrentCapacityKey] as? Int,
                           let max = desc[kIOPSMaxCapacityKey] as? Int, max > 0 {
                            out["battery_percent"] = Int((Double(level) / Double(max) * 100).rounded())
                        }
                        let charging = (desc[kIOPSIsChargingKey] as? Bool) ?? false
                        out["battery_state"] = charging ? "charging" : "unplugged"
                    }
                }
            }
            let url = URL(fileURLWithPath: NSHomeDirectory())
            if let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let free = values.volumeAvailableCapacityForImportantUsage {
                out["storage_free_gb"] = (Double(free) / 1_073_741_824.0).rounded(toPlaces: 2)
            }
            return out
        }
        return jsonEncode(d)
    }

    private static func openURL(_ args: JSONObject) -> String {
        guard let raw = args["url"] as? String, raw.hasPrefix("http://") || raw.hasPrefix("https://") else {
            return err("open_url needs an http(s) URL")
        }
        guard let url = URL(string: raw) else { return err("could not parse that URL") }
        let opened = DispatchQueue.main.sync { NSWorkspace.shared.open(url) }
        return opened ? ok_("opened \(raw)") : err("macOS would not open that URL")
    }

    private static func shareText(_ args: JSONObject) -> String {
        guard let text = args["text"] as? String, !text.isEmpty else {
            return err("share_text needs text")
        }
        let presented = DispatchQueue.main.sync { () -> Bool in
            guard let window = NSApp.keyWindow ?? NSApp.windows.first(where: { $0.canBecomeMain && $0.isVisible }),
                  let content = window.contentView else { return false }
            let picker = NSSharingServicePicker(items: [text])
            picker.show(relativeTo: content.bounds, of: content, preferredEdge: .minY)
            return true
        }
        return presented ? ok_("sharing picker opened") : err("no window to anchor the sharing picker to")
    }

    // MARK: - Helpers

    /// The marketing model identifier ("Mac15,6") — `uname`'s machine
    /// value. The friendly human name needs an SMC read with no Swift
    /// bridge, so the platform code is the honest answer.
    private static func deviceModelName() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) { p in
            String(cString: UnsafeRawPointer(p).assumingMemoryBound(to: CChar.self))
        }
    }

    private static func ok_(_ note: String) -> String {
        jsonEncode(["ok": true, "note": note] as JSONObject)
    }

    private static func err(_ message: String) -> String {
        jsonEncode(["error": message] as JSONObject)
    }
}

extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}
