import Foundation
import Security

/// The app's private container paths. The iOS stand-in for Android's
/// `context.filesDir` — Application Support is the durable, non-user-visible
/// location, and it is what the runtime treats as `$HOME`/`config_root`.
enum AppPaths {
    static let container: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }()

    /// What the engine pins as `$HOME` and `config_root`.
    static var home: String { container.path }

    static var sofuuDir: URL { container.appendingPathComponent(".sofuu", isDirectory: true) }

    /// The app-internal default workspace (a real folder picker can replace
    /// it later, exactly as on Android).
    static var defaultWorkspace: URL { container.appendingPathComponent("workspace", isDirectory: true) }
}

/// Keychain-backed secret store — the iOS counterpart to Android's
/// `EncryptedSharedPreferences` (Keystore-wrapped). API keys never live only
/// in config.json: the desktop keeps them in the OS credential store and
/// injects them at boot, Android uses EncryptedSharedPreferences, and iOS
/// uses the Keychain. `ConfigStore.writeConfig` folds them into the
/// providers' api_key fields, which is what chat.js reads per turn.
final class Keychain {
    private let service = "com.sofuu.app.keys"

    func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var out: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func set(_ account: String, _ value: String) {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let attrs: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]
        let status = SecItemUpdate(query as CFDictionary, attrs as CFDictionary)
        if status == errSecItemNotFound {
            var add = query
            add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            SecItemAdd(add as CFDictionary, nil)
        }
    }

    func delete(_ account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

/// The shared `~/.sofuu/config.json` + the secure key store — the Swift port
/// of the Android app's `ConfigStore.kt` (itself a port of the desktop
/// backend's `config.rs` / `keychain.rs`).
final class ConfigStore {

    private let keys = Keychain()

    var sofuuDir: URL { AppPaths.sofuuDir }
    private var configFile: URL { sofuuDir.appendingPathComponent("config.json") }
    private var stateFile: URL { sofuuDir.appendingPathComponent("mobile.json") }

    // ── config.json ─────────────────────────────────────────────────

    func readConfig() -> JSONObject {
        guard let text = try? String(contentsOf: configFile, encoding: .utf8),
              let obj = parseJSONObject(text) else { return [:] }
        return obj
    }

    func writeConfig(_ cfg: JSONObject) {
        try? FileManager.default.createDirectory(at: sofuuDir, withIntermediateDirectories: true)
        // Inject the secure keys into the provider entries so the engine's
        // per-turn config read sees them (the same role the desktop's
        // boot-time keychain injection plays).
        var out = cfg
        if var providers = out.optArray("providers") as? [JSONObject] {
            for i in providers.indices {
                let name = providers[i].optString("name")
                if !name.isEmpty, let key = keyFor(name), !key.isEmpty {
                    providers[i]["api_key"] = key
                }
            }
            out["providers"] = providers
        }
        if let data = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            try? text.write(to: configFile, atomically: true, encoding: .utf8)
        }
    }

    /// Patch the config (desktop `update_config` semantics: providers merge by
    /// name, unknown keys preserved). Returns the merged config.
    @discardableResult
    func updateConfig(_ patch: JSONObject) -> JSONObject {
        let cur = readConfig()
        var merged = cur
        for (k, v) in patch {
            if k == "providers", let incoming = v as? [Any] {
                merged["providers"] = mergeProviders(cur.optArray("providers"), incoming)
            } else {
                merged[k] = v
            }
        }
        writeConfig(merged)
        return merged
    }

    /// Union by name: incoming wins, existing extras preserved (same rule as
    /// the desktop's `merge_providers`).
    private func mergeProviders(_ existing: [Any]?, _ incoming: [Any]) -> [Any] {
        var byName: [String: JSONObject] = [:]
        var order: [String] = []
        if let existing {
            for case let p as JSONObject in existing {
                let n = p.optString("name")
                if byName[n] == nil { order.append(n) }
                byName[n] = p
            }
        }
        for case let p as JSONObject in incoming {
            let name = p.optString("name")
            if let prev = byName[name] {
                var out = prev
                for (k, v) in p { out[k] = v }
                byName[name] = out
            } else {
                order.append(name)
                byName[name] = p
            }
        }
        return order.compactMap { byName[$0] }
    }

    /// Delete a provider end-to-end: config entry + stored key. The active
    /// provider is refused (switch first) — desktop parity.
    func removeProvider(_ name: String) throws -> JSONObject {
        let cur = readConfig()
        if cur.optString("active") == name || cur.optString("provider") == name {
            throw SofuuError(code: "active_provider", message: "cannot remove the active provider — switch first")
        }
        let kept = (cur.optArray("providers") ?? []).compactMap { $0 as? JSONObject }
            .filter { $0.optString("name") != name }
        var merged = cur
        merged["providers"] = kept
        writeConfig(merged)
        deleteKey(name)
        return merged
    }

    /// Redacted view for the UI: has_api_key flags, no raw keys.
    func redacted(_ cfg: JSONObject? = nil) -> SofuuConfig {
        var out = cfg ?? readConfig()
        if var providers = out.optArray("providers") as? [JSONObject] {
            for i in providers.indices {
                let key = providers[i].optString("api_key")
                if !key.isEmpty { providers[i]["has_api_key"] = true }
                providers[i]["api_key"] = ""
            }
            out["providers"] = providers
        }
        if !out.optString("api_key").isEmpty {
            out["has_api_key"] = true
            out["api_key"] = ""
        }
        return SofuuConfig.fromJson(out)
    }

    // ── mobile.json (app state: project dir, workspaces) ────────────

    func readState() -> JSONObject {
        guard let text = try? String(contentsOf: stateFile, encoding: .utf8),
              let obj = parseJSONObject(text) else { return [:] }
        return obj
    }

    func writeState(_ s: JSONObject) {
        try? FileManager.default.createDirectory(at: sofuuDir, withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(withJSONObject: s, options: [.prettyPrinted, .sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            try? text.write(to: stateFile, atomically: true, encoding: .utf8)
        }
    }

    func projectDir() -> String? {
        readState().optString("project_dir").nilIfEmpty
    }

    func setProjectDir(_ path: String) {
        var s = readState()
        s["project_dir"] = path
        writeState(s)
    }

    func workspaces() -> [String] {
        (readState().optArray("workspaces") ?? []).compactMap { $0 as? String }.filter { !$0.isEmpty }
    }

    func setWorkspaces(_ list: [String]) {
        var s = readState()
        s["workspaces"] = list
        writeState(s)
    }

    // ── The Keychain-backed key store ───────────────────────────────

    func keyFor(_ provider: String) -> String? { keys.get("key:\(provider)") }

    func setKey(_ provider: String, _ key: String) { keys.set("key:\(provider)", key) }

    func deleteKey(_ provider: String) { keys.delete("key:\(provider)") }

    func hasKey(_ provider: String) -> Bool { !(keyFor(provider) ?? "").isEmpty }
}

/// UI preferences that persist across launches (the desktop's localStorage
/// equivalents).
final class UiPrefs {
    private let d = UserDefaults.standard

    var theme: String {
        get { d.string(forKey: "theme") ?? "system" }
        set { d.set(newValue, forKey: "theme") }
    }

    var mode: String {
        get { d.string(forKey: "mode") ?? "agent" }
        set { d.set(newValue, forKey: "mode") }
    }

    var permissions: String {
        get { d.string(forKey: "permissions") ?? "prompt" }
        set { d.set(newValue, forKey: "permissions") }
    }

    var cloudBg: Bool {
        get { d.object(forKey: "cloudBg") as? Bool ?? true }
        set { d.set(newValue, forKey: "cloudBg") }
    }

    var chatFontSize: Int {
        get {
            let v = d.object(forKey: "chatFontSize") as? Int ?? 15
            return min(22, max(13, v))
        }
        set { d.set(min(22, max(13, newValue)), forKey: "chatFontSize") }
    }
}
