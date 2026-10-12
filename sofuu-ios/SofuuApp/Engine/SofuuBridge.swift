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

    // ── Vector embeddings ──────────────────────────────────────────────
    //
    // Stateless pure functions in the ABI (sofuu_embed.h): these pass a
    // NULL runtime, so they need no runtime and no thread affinity. The
    // default space is hash-768 — the SAME space the brain stores in, so
    // embed → remember needs no space argument.
    //
    // nil is returned when the space is unknown or the artifact is
    // unavailable, rather than a fake zero vector that would never match
    // anything.

    /// Text → vector. `space` nil/empty selects the default.
    static func embedLocal(_ text: String, space: String? = nil) -> [Float]? {
        var out: UnsafeMutablePointer<Float>?
        var dim = 0
        let rc: Int32
        if let space {
            rc = text.withCString { t in space.withCString { s in
                sofuu_embed_local(nil, t, s, &out, &dim)
            } }
        } else {
            rc = text.withCString { t in sofuu_embed_local(nil, t, nil, &out, &dim) }
        }
        guard rc == 0, let out, dim > 0 else {
            if let out { sofuu_free(out) }
            return nil
        }
        defer { sofuu_free(out) }
        return Array(UnsafeBufferPointer(start: out, count: dim))
    }

    /// Several texts at once → array of vectors.
    static func embedBatch(_ texts: [String], space: String? = nil) -> [[Float]]? {
        guard !texts.isEmpty else { return [] }
        // The ABI takes `const char **`, so this is a C array of borrowed
        // pointers, not a Swift array — the elements stay valid for the
        // whole call because each strdup is freed only afterwards.
        let cstrs: [UnsafePointer<CChar>?] = texts.map { t in
            UnsafePointer(strdup(t))
        }
        defer { for c in cstrs { free(UnsafeMutableRawPointer(mutating: c)) } }

        var out: UnsafeMutablePointer<Float>?
        var got = 0, dim = 0
        let rc: Int32 = cstrs.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress.map { UnsafeMutablePointer(mutating: $0) }
            if let space {
                return space.withCString { s in
                    sofuu_embed_batch(nil, base, buf.count, s, &out, &got, &dim)
                }
            }
            return sofuu_embed_batch(nil, base, buf.count, nil, &out, &got, &dim)
        }
        guard rc == 0, let out, dim > 0 else {
            if let out { sofuu_free(out) }
            return nil
        }
        defer { sofuu_free(out) }
        var result: [[Float]] = []
        result.reserveCapacity(got)
        for i in 0..<got {
            let base = out.advanced(by: i * dim)
            result.append(Array(UnsafeBufferPointer(start: base, count: dim)))
        }
        return result
    }

    /// PNG/JPEG bytes → img1-64 vector (joint with the sem2 geometry).
    static func embedImage(_ bytes: [UInt8]) -> [Float]? {
        var out: UnsafeMutablePointer<Float>?
        var dim = 0
        let rc = bytes.withUnsafeBufferPointer { buf in
            sofuu_embed_image(nil, buf.baseAddress, buf.count, &out, &dim)
        }
        guard rc == 0, let out, dim > 0 else {
            if let out { sofuu_free(out) }
            return nil
        }
        defer { sofuu_free(out) }
        return Array(UnsafeBufferPointer(start: out, count: dim))
    }

    // ── Voice ─────────────────────────────────────────────────────────
    //
    // Thin funnels to ai.transcribe / ai.speak — the same implementation
    // the JS surface uses, so there is no duplicated HTTP here. The host
    // base64-encodes the audio; the library validates the alphabet before
    // it reaches JS, so no string escape is possible.
    //
    // These DO need the runtime (they go through the funnel), so they must
    // run on the engine thread.

    /// Base64 audio → transcript. Returns the JSON envelope.
    func voiceTranscribe(_ audioBase64: String, optsJson: String? = nil) -> String? {
        var out: UnsafeMutablePointer<CChar>?
        // The envelope is returned even on failure — the caller reads
        // ok/error — so the rc is not what decides the return value here.
        _ = audioBase64.withCString { a in
            if let optsJson {
                return optsJson.withCString { o in sofuu_voice_transcribe(rtPtr, a, o, &out) }
            }
            return sofuu_voice_transcribe(rtPtr, a, nil, &out)
        }
        guard let out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Synthesize speech. Resolves {audio, format}.
    func voiceSpeak(_ text: String, optsJson: String? = nil) -> String? {
        var out: UnsafeMutablePointer<CChar>?
        _ = text.withCString { t in
            if let optsJson {
                return optsJson.withCString { o in sofuu_voice_speak(rtPtr, t, o, &out) }
            }
            return sofuu_voice_speak(rtPtr, t, nil, &out)
        }
        guard let out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    // ── Host tools ────────────────────────────────────────────────────
    //
    // sofuu_rt_register_host_tool lets the HOST hand the agent a native
    // capability with no JS in the loop. iOS is where this matters most:
    // the QuickJS sandbox cannot reach UIKit, so the share sheet, the
    // haptics engine and the pasteboard all have to be answered from here.
    //
    // The C callback signature hands over no tool name, so the name, the
    // Swift closure and its context pointer all ride in `opaque` — one
    // box per registration.

    /// A native capability the agent can call.
    typealias HostTool = (String, UInt64) -> String?

    /// Host for streaming events (runs on the engine thread).
    typealias StreamListener = (String) -> Void

    /// Process-global listener — the C callback carries no context pointer we
    /// own, so the target is parked here. One runtime per process in this app.
    private static var streamListener: StreamListener?

    private static let streamTrampoline: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { json, _ in
        guard let json, let listener = SofuuBridge.streamListener else { return }
        listener(String(cString: json))
    }

    /// Boxes a host tool so its identity survives as an `opaque` pointer.
    private final class HostToolBox {
        let name: String
        let tool: HostTool
        init(name: String, tool: @escaping HostTool) { self.name = name; self.tool = tool }
    }

    /// The result has to outlive the callback, and a Swift String has no
    /// address C can hold — so it goes into a buffer that is reallocated
    /// only when a longer result needs it, and whose contents stay valid
    /// until the next host-tool call. The runtime copies the pointer
    /// before returning, so one slot is enough and this cannot grow
    /// unboundedly in practice.
    private static let resultLock = NSLock()
    private static var resultBuf: UnsafeMutablePointer<CChar>? = nil
    private static var resultCap = 0

    private static func storeResult(_ s: String) -> UnsafePointer<CChar>? {
        resultLock.lock()
        defer { resultLock.unlock() }
        let needed = s.utf8.count + 1
        if needed > resultCap {
            resultBuf?.deallocate()
            resultBuf = UnsafeMutablePointer<CChar>.allocate(capacity: needed)
            resultCap = needed
        }
        guard let buf = resultBuf else { return nil }
        s.withCString { src in buf.update(from: src, count: needed) }
        return UnsafePointer(buf)
    }

    private static let hostToolTrampoline: @convention(c) (
        UnsafePointer<CChar>?, UInt64, UnsafeMutableRawPointer?
    ) -> UnsafePointer<CChar>? = { argsJson, callId, opaque in
        guard let opaque else { return nil }
        let box = Unmanaged<HostToolBox>.fromOpaque(opaque).takeUnretainedValue()
        let args = argsJson.map { String(cString: $0) } ?? "{}"
        guard let result = box.tool(args, callId) else { return nil }
        return storeResult(result)
    }

    /// Expose a native capability to the agent as a tool.
    ///
    /// `parametersJson` is the JSON Schema the model sees, so an accurate
    /// one is the difference between the model using the tool correctly
    /// and guessing. It must be a JSON **object** — the runtime rejects
    /// anything else as an argument error rather than surfacing it later
    /// as a confusing model failure.
    ///
    /// Engine thread only, and only while this bridge is alive. Re-using a
    /// name replaces the earlier registration.
    @discardableResult
    func registerHostTool(name: String, description: String, parametersJson: String, tool: @escaping HostTool) -> Bool {
        let box = HostToolBox(name: name, tool: tool)
        let opaque = Unmanaged.passRetained(box).toOpaque()
        let rc = name.withCString { n in
            description.withCString { d in
                parametersJson.withCString { p in
                    sofuu_rt_register_host_tool(rtPtr, n, d, p, SofuuBridge.hostToolTrampoline, opaque)
                }
            }
        }
        if rc != 0 {
            Unmanaged<HostToolBox>.fromOpaque(opaque).release()
            return false
        }
        return true
    }

    /// Settle a call whose host tool returned nil.
    ///
    /// Safe from ANY thread — the completion is marshalled onto the runtime
    /// thread through its event loop. Pass a message in `error` to fail the
    /// call instead; `resultJson` must itself be valid JSON.
    @discardableResult
    func resolveHostTool(_ callId: UInt64, resultJson: String? = nil, error: String? = nil) -> Bool {
        let rc: Int32
        if let resultJson {
            rc = resultJson.withCString { r in
                if let error { return error.withCString { e in sofuu_rt_host_tool_resolve(rtPtr, callId, r, e) } }
                return sofuu_rt_host_tool_resolve(rtPtr, callId, r, nil)
            }
        } else if let error {
            rc = error.withCString { e in sofuu_rt_host_tool_resolve(rtPtr, callId, nil, e) }
        } else {
            return false
        }
        return rc == 0
    }

    /// Release the runtime. Never call native methods after this.
    func close() {
        sofuu_rt_free(rtPtr)
    }

    static func setStreamListener(_ listener: StreamListener?) {
        streamListener = listener
    }

    // ── Host log callback ─────────────────────────────────────────────
    //
    // Without this the engine's console.log/info/warn/error had nowhere to
    // go on iOS (the desktop routes them to stderr via the embed config),
    // so every JS-side warning vanished. The callback fires on the owning
    // engine thread with C strings valid only for the call, so print them
    // and be done — no bridging back into Swift state.

    private static let logTrampoline: @convention(c) (
        UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafeMutableRawPointer?
    ) -> Void = { level, message, _ in
        let lvl = level.map { String(cString: $0) } ?? "info"
        let msg = message.map { String(cString: $0) } ?? ""
        NSLog("[sofuu-js][%@] %@", lvl, msg)
    }

    /// Route the engine's console output to the device log (os_log / Xcode
    /// console, filter `sofuu-js`). Process-global per the ABI — call once,
    /// before any runtime is created, so boot warnings are captured too.
    /// Pass false to send output back to stdio.
    static func setLogEnabled(_ enabled: Bool) {
        sofuu_embed_set_log_cb(enabled ? logTrampoline : nil, nil)
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
