import Foundation

/// The shared `~/.sofuu/config.json` + `~/.sofuu/desktop.json` — a faithful
/// port of the Tauri app's `config.rs` (the iOS `ConfigStore` with the
/// desktop contracts restored). Keys live IN the 0600 config file (the
/// CLI's own store): the login-keychain items the Tauri app created carry
/// per-item ACLs that re-prompt the login password on every dev rebuild,
/// so the keychain layer is intentionally absent here:
///
///   * PATCHABLE whitelist — a settings patch must never clobber fields it
///     does not understand ("sync" is deliberately absent: the mesh is
///     built-in, no UI may disable it);
///   * providers merge BY NAME — the UI only ever sees REDACTED entries,
///     so a wholesale replace would wipe CLI-written keys; the patch
///     updates only the fields it names and NEVER touches api_key;
///   * flat-mirror sync — `provider/model/base_url/profile` follow the
///     active provider entry (ChatConfig's contract; the model picker's
///     covenant);
///   * atomic 0600 writes (tmp + rename, no write-then-chmod window);
///   * redaction — every api_key becomes "" plus has_api_key: the UI never
///     sees raw keys, Keychain or file.
///
/// Key storage stays desktop-format (Keychain service `com.sofuu.desktop`,
/// account = provider). `writeConfig` folds the Keychain keys into the
/// provider entries on the way out — chat.js's per-turn config read is the
/// only consumer of the folded value, exactly the role the Tauri boot
/// injection played.
final class ConfigStore {

    var sofuuDir: URL { MacPaths.sofuuDir }
    private var configFile: URL { MacPaths.configFile }
    private var desktopFile: URL { MacPaths.desktopFile }

    init() {
        try? FileManager.default.createDirectory(at: sofuuDir, withIntermediateDirectories: true)
    }

    // ── config.json ─────────────────────────────────────────────────

    func readConfig() -> JSONObject {
        guard let text = try? String(contentsOf: configFile, encoding: .utf8),
              let obj = parseJSONObject(text) else { return [:] }
        return obj
    }

    /// Keys live on disk in the provider entries — kept for call-site
    /// clarity at engine boot.
    func readConfigWithKeys() -> JSONObject { readConfig() }

    /// The PATCHABLE whitelist (config.rs:72-76).
    private static let patchable: Set<String> = [
        "provider", "model", "effort", "brain", "ml", "rlm", "base_url", "profile",
        "embed_provider", "embed_model", "ctx_window", "max_output", "budget_usd", "ghost",
        "recall_min", "recall_budget", "providers", "active",
    ]

    func writeConfig(_ cfg: JSONObject) {
        try? FileManager.default.createDirectory(at: sofuuDir, withIntermediateDirectories: true)
        // Fold the secure keys into the provider entries so the engine's
        // per-turn config read sees them (the same role the desktop's
        // boot-time Keychain injection plays — the file may legally carry
        // CLI-written keys too, so this changes nothing the CLI relies on).
        // api_key fields ride in the entries themselves — nothing to fold.
        atomicWrite0600(json: cfg, to: configFile)
    }

    /// Patch the config (desktop `update_config` semantics). Returns the
    /// REDACTED merged config.
    @discardableResult
    func updateConfig(_ patch: JSONObject) -> JSONObject {
        let cur = readConfig()
        var merged = cur
        for (k, v) in patch {
            if k == "providers", let incoming = v as? [Any] {
                merged["providers"] = mergeProviders(cur.optArray("providers"), incoming)
            } else if Self.patchable.contains(k) {
                merged[k] = v
            }
            // Unknown keys are dropped, deliberately (config.rs:68-76).
        }
        normalizeEndpoints(&merged)
        syncFlatMirror(&merged)
        writeConfig(merged)
        return merged
    }

    /// OpenAI-wire providers need the FULL chat-completions URL — chat.js
    /// never appends the suffix, and a base URL like `…/v1` POSTs to the
    /// wrong path: the router answers 200 with an empty body, a clean EOF
    /// with no [DONE]/finish_reason, which the engine classifies as
    /// truncation ("transfer closed") and burns the retry ladder on.
    /// Append the suffix when it is missing (idempotent; anthropic/local
    /// profiles and URLs that already carry it are untouched).
    private func normalizeEndpoints(_ cfg: inout JSONObject) {
        guard var providers = cfg.optArray("providers") as? [JSONObject] else { return }
        var changed = false
        for i in providers.indices {
            let profile = providers[i].optString("profile").lowercased()
            if profile == "anthropic" || profile == "local" { continue }
            guard var endpoint = providers[i].optString("endpoint").nilIfEmpty else { continue }
            endpoint = endpoint.trimmingCharacters(in: .whitespaces)
            if endpoint.lowercased().contains("/chat/completions") { continue }
            endpoint += endpoint.hasSuffix("/") ? "chat/completions" : "/chat/completions"
            providers[i]["endpoint"] = endpoint
            changed = true
        }
        if changed { cfg["providers"] = providers }
    }

    /// The legacy flat mirror follows the active provider's entry — the
    /// rule ChatConfig applies and the model picker relies on (an empty
    /// model in the entry CLEARS the mirror: "no model selected" until
    /// picked). The existing root api_key survives untouched (Keychain/CLI
    /// own it).
    private func syncFlatMirror(_ cfg: inout JSONObject) {
        let active = cfg.optString("active")
        guard !active.isEmpty else { return }
        let entry = (cfg.optArray("providers") ?? [])
            .compactMap { $0 as? JSONObject }
            .first { $0.optString("name") == active }
        guard let entry else { return }
        let existingKey = cfg["api_key"]
        cfg["provider"] = entry.optString("name")
        cfg["model"] = entry["model"] ?? ""
        cfg["base_url"] = entry["endpoint"] ?? ""
        cfg["profile"] = entry["profile"] ?? ""
        cfg["api_key"] = existingKey ?? ""
    }

    /// Union by name: the patch updates only the fields it names and never
    /// touches api_key/has_api_key; new names append, unknown existing
    /// entries are preserved (config.rs merge_providers, 1:1).
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
            if name.isEmpty { continue }
            if let prev = byName[name] {
                var out = prev
                for (k, v) in p where k != "api_key" && k != "has_api_key" {
                    out[k] = v
                }
                byName[name] = out
            } else {
                var fresh: JSONObject = [:]
                for (k, v) in p where k != "api_key" && k != "has_api_key" {
                    fresh[k] = v
                }
                order.append(name)
                byName[name] = fresh
            }
        }
        return order.compactMap { byName[$0] }
    }

    /// Delete a provider end-to-end: config entry + stored key. The active
    /// provider is refused (engine has no fallback — desktop message
    /// parity), unknown names are refused.
    func removeProvider(_ name: String) throws -> JSONObject {
        let cur = readConfig()
        if name == cur.optString("active") {
            throw SofuuError(code: "active_provider", message: "\(name) is the active provider — switch first")
        }
        var kept: [JSONObject] = []
        for case let p as JSONObject in cur.optArray("providers") ?? [] {
            if p.optString("name") != name { kept.append(p) }
        }
        if kept.count == (cur.optArray("providers") ?? []).count {
            throw SofuuError(code: "no_such_provider", message: "no provider named \(name)")
        }
        var merged = cur
        merged["providers"] = kept
        writeConfig(merged)
        return merged
    }

    /// Redacted view for the UI: has_api_key flags, no raw keys — the
    /// same contract redact() enforces in config.rs.
    func redacted(_ cfg: JSONObject? = nil) -> SofuuConfig {
        let input = cfg ?? readConfig()
        var out = input
        if var providers = out.optArray("providers") as? [JSONObject] {
            for i in providers.indices {
                let fileKey = providers[i].optString("api_key")
                providers[i]["has_api_key"] = !fileKey.isEmpty
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

    // ── desktop.json (project dir + workspace list) ──────────────────

    func readState() -> JSONObject {
        guard let text = try? String(contentsOf: desktopFile, encoding: .utf8),
              let obj = parseJSONObject(text) else { return [:] }
        return obj
    }

    func writeState(_ s: JSONObject) {
        try? FileManager.default.createDirectory(at: sofuuDir, withIntermediateDirectories: true)
        atomicWrite0600(json: s, to: desktopFile)
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

    // ── Key passthrough (the 0600 config file owns the secrets) ──────

    /// Raw provider entry lookup by name (config.rs's provider_entry).
    private func providerEntry(_ provider: String) -> JSONObject? {
        (readConfig().optArray("providers") ?? [])
            .compactMap { $0 as? JSONObject }
            .first { $0.optString("name") == provider }
    }

    func keyFor(_ provider: String) -> String? {
        guard let p = providerEntry(provider) else { return nil }
        let k = p.optString("api_key")
        return k.isEmpty ? nil : k
    }

    /// Fold a key into the provider entry (creating the entry when the
    /// provider is unknown) and persist. The single key store.
    func setKey(_ provider: String, _ key: String) {
        var cfg = readConfig()
        var providers = (cfg.optArray("providers") as? [JSONObject]) ?? []
        if let i = providers.firstIndex(where: { $0.optString("name") == provider }) {
            providers[i]["api_key"] = key
        } else {
            providers.append(["name": provider, "api_key": key])
        }
        cfg["providers"] = providers
        if cfg.optString("provider").isEmpty { cfg["provider"] = provider }
        writeConfig(cfg)
    }

    func hasKey(_ provider: String) -> Bool { !(keyFor(provider) ?? "").isEmpty }

    /// Clear a key: drop the api_key field from the entry.
    func clearKey(_ provider: String) {
        var cfg = readConfig()
        var providers = (cfg.optArray("providers") as? [JSONObject]) ?? []
        if let i = providers.firstIndex(where: { $0.optString("name") == provider }) {
            providers[i].removeValue(forKey: "api_key")
        }
        cfg["providers"] = providers
        writeConfig(cfg)
    }

    // ── The one write primitive ──────────────────────────────────────

    /// Pretty JSON → tmp file created 0600 from the first byte → rename
    /// over the target (config.rs's "no write-then-chmod window").
    private func atomicWrite0600(json: JSONObject, to url: URL) {
        guard let data = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) else {
            return
        }
        let tmp = url.deletingLastPathComponent().appendingPathComponent(
            url.lastPathComponent + ".tmp")
        if !FileManager.default.createFile(atPath: tmp.path, contents: data,
                                           attributes: [.posixPermissions: 0o600]) {
            CrashLog.error("config write failed: \(tmp.path)")
            return
        }
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } catch {
            // replaceItemAt can refuse when the target is a symlink or
            // missing — plain rename covers those.
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.moveItem(at: tmp, to: url)
        }
    }
}

/// UI preferences that persist across launches (the desktop's localStorage
/// equivalents). Zoom + sidebar collapse are mac-only preferences.
final class UiPrefs {
    private let d = UserDefaults.standard

    var theme: String {
        get { d.string(forKey: "theme") ?? "system" }
        set { d.set(newValue, forKey: "theme") }
    }

    var mode: String {
        get {
            let v = d.string(forKey: "mode") ?? "agent"
            return ["agent", "chat"].contains(v) ? v : "agent"
        }
        set { d.set(["agent", "chat"].contains(newValue) ? newValue : "agent", forKey: "mode") }
    }

    var permissions: String {
        get {
            let v = d.string(forKey: "permissions") ?? "prompt"
            return ["full", "edit", "plan", "prompt"].contains(v) ? v : "prompt"
        }
        set { d.set(["full", "edit", "plan", "prompt"].contains(newValue) ? newValue : "prompt",
                     forKey: "permissions") }
    }

    var cloudBg: Bool {
        get { d.object(forKey: "cloudBg") as? Bool ?? true }
        set { d.set(newValue, forKey: "cloudBg") }
    }

    /// Desktop clamp: 14–24 px (localStorage sofuu.* rules).
    var chatFontSize: Int {
        get {
            let v = d.object(forKey: "chatFontSize") as? Int ?? 15
            return min(24, max(14, v))
        }
        set { d.set(min(24, max(14, newValue)), forKey: "chatFontSize") }
    }

    /// 0.5–2.0, rounded to 0.1 (localStorage sofuu.zoom rules).
    var zoom: Double {
        get {
            let v = d.object(forKey: "zoom") as? Double ?? 1.0
            return min(2, max(0.5, (v * 10).rounded() / 10))
        }
        set { d.set(min(2, max(0.5, (newValue * 10).rounded() / 10)), forKey: "zoom") }
    }

    var sidebarCollapsed: Bool {
        get { d.bool(forKey: "sidebarCollapsed") }
        set { d.set(newValue, forKey: "sidebarCollapsed") }
    }

    var ideMode: Bool {
        get { d.bool(forKey: "ideMode") }
        set { d.set(newValue, forKey: "ideMode") }
    }

    /// IDE panel widths in points (the two draggable splitters). 0 means
    /// never dragged — the layout keeps its default. Clamped on both read
    /// and write so a stray write can't squeeze the editor out.
    var ideTreeWidth: Double {
        get {
            let v = d.double(forKey: "ideTreeWidth")
            return v == 0 ? 230 : min(460, max(170, v))
        }
        set { d.set(min(460, max(170, newValue)), forKey: "ideTreeWidth") }
    }

    var ideAgentWidth: Double {
        get {
            let v = d.double(forKey: "ideAgentWidth")
            return v == 0 ? 380 : min(640, max(280, v))
        }
        set { d.set(min(640, max(280, newValue)), forKey: "ideAgentWidth") }
    }
}
