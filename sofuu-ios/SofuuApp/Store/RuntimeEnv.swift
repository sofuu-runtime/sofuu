import Foundation

/// The runtime's environment knobs, surfaced as settings.
///
/// The engine reads these with plain `std::env::var`, so the host applies
/// them with `setenv` before `sofuu_rt_new`. There is no ABI for them and
/// no reason for one: they are process configuration, not runtime state.
///
/// **When they are read matters.** Most are consulted per turn — the turn
/// driver re-evaluates `chatToolDefs()`, the retry ladder resolves lazily on
/// the first transient error — so applying them at boot is both sufficient
/// and correct. `SOFUU_NO_ML` is the exception: it gates *module
/// registration* inside sofuu-core, so it must be in the environment
/// before the runtime exists or the whole `sofuu.ml` namespace is simply
/// absent and the setting silently does nothing.
final class RuntimeEnv {

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // ── The knobs ──────────────────────────────────────────────────────

    /// Every 8 built-in coding tools off; web + MCP only. Read per turn
    /// at `chat.js:816`.
    var noTools: Bool {
        get { defaults.bool(forKey: Key.noTools) }
        set { defaults.set(newValue, forKey: Key.noTools) }
    }

    /// `bash` stays in the tool list but every invocation refuses
    /// (`tools.js:540`). Note this does NOT stop MCP tools that shell
    /// out — it only closes the built-in.
    var noShell: Bool {
        get { defaults.bool(forKey: Key.noShell) }
        set { defaults.set(newValue, forKey: Key.noShell) }
    }

    /// Drops the entire `sofuu.ml` context-economy gate layer.
    /// Compile-gated at registration — needs a restart to take effect.
    var noMl: Bool {
        get { defaults.bool(forKey: Key.noMl) }
        set { defaults.set(newValue, forKey: Key.noMl) }
    }

    /// Memory backend: "" (auto), "hash", "semantic". Read per agent
    /// def/run (`agent.js:693`, `:865`).
    var memoryBackend: String {
        get { defaults.string(forKey: Key.memoryBackend) ?? "" }
        set { defaults.set(newValue, forKey: Key.memoryBackend) }
    }

    /// Default web-search engine: "" (auto/duckduckgo), "brave",
    /// "tavily". Read per search (`web.js:369`).
    var webEngine: String {
        get { defaults.string(forKey: Key.webEngine) ?? "" }
        set { defaults.set(newValue, forKey: Key.webEngine) }
    }

    /// Lets the agent reach private/loopback addresses through web.open.
    /// Off by default; the SSRF guard in `web.js:169` blocks them
    /// otherwise.
    var webAllowInternal: Bool {
        get { defaults.bool(forKey: Key.webAllowInternal) }
        set { defaults.set(newValue, forKey: Key.webAllowInternal) }
    }

    /// Seconds a stalled transport waits before the turn gives up.
    /// 0 keeps the runtime default (300). Read per stream (`ai.rs:1611`).
    var stallTimeoutSecs: Int {
        get { defaults.integer(forKey: Key.stallTimeout) }
        set { defaults.set(max(0, min(newValue, 3600)), forKey: Key.stallTimeout) }
    }

    /// Tool-loop budget for one chat turn. 0 keeps the runtime default
    /// (200). Read per turn (`chat.js:1456`).
    var chatMaxSteps: Int {
        get { defaults.integer(forKey: Key.maxSteps) }
        set { defaults.set(max(0, min(newValue, 2000)), forKey: Key.maxSteps) }
    }

    /// Set before `sofuu_rt_new`. Returns what it actually applied so the
    /// UI can show effective values rather than requested ones.
    @discardableResult
    func apply() -> [String: String] {
        var applied: [String: String] = [:]
        func put(_ key: String, _ value: String?) {
            if let value, !value.isEmpty {
                setenv(key, value, 1)
                applied[key] = value
            } else {
                unsetenv(key)
                applied[key] = ""
            }
        }
        put(Key.noTools, noTools ? "1" : nil)
        put(Key.noShell, noShell ? "1" : nil)
        // Registration-gated: without this the ml namespace is absent and
        // every gate degrades to a silent no-op.
        put(Key.noMl, noMl ? "1" : nil)
        put(Key.memoryBackend, memoryBackend)
        put(Key.webEngine, webEngine)
        put(Key.webAllowInternal, webAllowInternal ? "1" : nil)
        put(Key.stallTimeout, stallTimeoutSecs > 0 ? String(stallTimeoutSecs) : nil)
        put(Key.maxSteps, chatMaxSteps > 0 ? String(chatMaxSteps) : nil)
        return applied
    }

    private enum Key {
        static let noTools = "SOFUU_NO_TOOLS"
        static let noShell = "SOFUU_NO_SHELL"
        static let noMl = "SOFUU_NO_ML"
        static let memoryBackend = "SOFUU_MEMORY_BACKEND"
        static let webEngine = "SOFUU_WEB_ENGINE"
        static let webAllowInternal = "SOFUU_WEB_ALLOW_INTERNAL"
        static let stallTimeout = "SOFUU_STALL_TIMEOUT_SECS"
        static let maxSteps = "SOFUU_CHAT_MAX_STEPS"
    }
}