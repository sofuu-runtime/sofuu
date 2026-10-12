import Foundation
import Sofuu

/// A thin Swift wrapper around the libsofuu C ABI (v2).
///
/// The iOS twin of the Android app's `SofuuBridge.kt` — except iOS needs no
/// JNI. Swift talks to the C ABI directly through the `Sofuu` module that
/// `dist/libsofuu.xcframework` exports, so there is one less layer and one
/// less copy of the contract.
///
/// One `SofuuBridge` per thread; create and use it ONLY on the thread that
/// will drive it (the engine worker). The app funnels every call through
/// `SofuuEngine`, which owns that thread — this class is intentionally thin:
/// raw calls, no state.
///
/// Android's JNI shim had to implement `setenv`/`chdir` in C. On iOS those
/// are plain Darwin calls, so they live here instead of in a bridging file.
final class SofuuBridge {

    /// The opaque `SofuuRuntime *` the C ABI handed back.
    private let rtPtr: OpaquePointer

    /// The ABI version this library implements.
    var abiVersion: Int { Int(sofuu_embed_abi_version()) }

    private init(rtPtr: OpaquePointer) {
        self.rtPtr = rtPtr
    }

    // ── Calls ────────────────────────────────────────────────────────

    /// Evaluate arbitrary JS — the eval escape hatch. Returns the
    /// JSON-stringified completion value, or nil on failure.
    func eval(_ source: String) -> String? {
        var out: UnsafeMutablePointer<CChar>?
        let rc = source.withCString { sofuu_rt_eval(rtPtr, $0, &out) }
        guard rc == 0, let out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Call a funnel method (e.g. "ai.complete", "agent.run"). Returns the
    /// JSON envelope, or nil on failure.
    func call(_ method: String, _ argsJson: String? = nil) -> String? {
        var out: UnsafeMutablePointer<CChar>?
        let rc: Int32
        if let argsJson {
            rc = method.withCString { m in
                argsJson.withCString { a in sofuu_rt_call(rtPtr, m, a, &out) }
            }
        } else {
            rc = method.withCString { m in sofuu_rt_call(rtPtr, m, nil, &out) }
        }
        guard rc == 0, let out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Run a streamable funnel call (agent.run / ai.stream). Events are
    /// delivered to the installed listener on the engine thread. Returns the
    /// cancel id (0 = dispatch failed).
    func callStream(_ method: String, _ argsJson: String?) -> UInt64 {
        guard let listener = SofuuBridge.streamListener else { return 0 }
        _ = listener
        var cancelId: UInt64 = 0
        let rc: Int32
        if let argsJson {
            rc = method.withCString { m in
                argsJson.withCString { a in
                    sofuu_rt_call_stream(rtPtr, m, a, SofuuBridge.streamTrampoline, nil, &cancelId)
                }
            }
        } else {
            rc = method.withCString { m in
                sofuu_rt_call_stream(rtPtr, m, nil, SofuuBridge.streamTrampoline, nil, &cancelId)
            }
        }
        return rc == 0 ? cancelId : 0
    }

    /// Cancel a streaming call by its id.
    func cancel(_ cancelId: UInt64) -> Bool {
        sofuu_rt_cancel(rtPtr, cancelId) == 0
    }

    /// Space manifest JSON (default space, model ids, dims, hashes).
    func embedInfo() -> String? {
        var out: UnsafeMutablePointer<CChar>?
        let rc = sofuu_embed_info(rtPtr, &out)
        guard rc == 0, let out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Release the runtime. Never call native methods after this.
    func close() {
        sofuu_rt_free(rtPtr)
    }

    // ── Streaming plumbing ───────────────────────────────────────────

    /// Host for streaming events (runs on the engine thread).
    typealias StreamListener = (String) -> Void

    /// Process-global listener — the C callback carries no context pointer we
    /// own, so the target is parked here. One runtime per process in this app.
    private static var streamListener: StreamListener?

    private static let streamTrampoline: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { json, _ in
        guard let json, let listener = SofuuBridge.streamListener else { return }
        listener(String(cString: json))
    }

    static func setStreamListener(_ listener: StreamListener?) {
        streamListener = listener
    }

    // ── Process setup (must run before `create`) ─────────────────────

    /// Pin `$HOME` for the process. iOS has no shell, and the runtime derives
    /// config/brain/session paths from `$HOME` when `config_root` is absent —
    /// the app pins it to its private container and also passes `config_root`
    /// explicitly (belt and braces; `config_root` is the documented path).
    static func setHome(_ path: String) {
        setenv("HOME", path, 1)
    }

    /// Set the process working directory (tool-jail default + the cwd spawned
    /// MCP children inherit). Call before `create`.
    static func chdir(_ path: String) {
        _ = path.withCString { Darwin.chdir($0) }
    }

    /// The ABI version this library implements (no runtime needed).
    static func abiVersion() -> Int { Int(sofuu_embed_abi_version()) }

    /// Create the runtime on the calling thread. `configJson` is the
    /// `sofuu_rt_new` config; the app passes its container dir as
    /// `config_root` so `~/.sofuu` lands in private storage.
    static func create(_ configJson: String? = nil) throws -> SofuuBridge {
        let ptr: OpaquePointer? = configJson == nil
            ? sofuu_rt_new(nil)
            : configJson!.withCString { sofuu_rt_new($0) }
        guard let ptr else {
            throw SofuuError(code: "internal", message: "sofuu_rt_new failed")
        }
        return SofuuBridge(rtPtr: ptr)
    }
}

/// A failure surfaced by the runtime (mirrors the C ABI's stable error codes).
struct SofuuError: Error, CustomStringConvertible {
    let code: String
    let message: String
    var description: String { "SofuuError(\(code)): \(message)" }
}
