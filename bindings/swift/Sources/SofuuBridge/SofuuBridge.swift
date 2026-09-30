//
//  SofuuBridge.swift
//  A typed Swift surface over the libsofuu C ABI.
//
//  Why this exists: the C ABI is deliberately small and JSON-shaped (it is
//  the contract every language binds to). That is right for stability, but
//  an app developer should not have to hand-build JSON strings to store a
//  memory. Everything below is a thin, honest layer: no caching, no hidden
//  state, no behavior the C ABI does not also give you.
//
//  Quick start:
//
//      import Sofuu
//
//      let sofuu = try Sofuu()
//      let brain = try sofuu.brain(at: url)          // encrypted, local
//      try brain.remember("the sky is blue")
//      let hits = try brain.recall("what colour is the sky?", k: 3)
//      for hit in hits { print(hit.text) }
//
//  Everything runs on-device. No API key, no network, no telemetry.
//

import Foundation
import SofuuBinary

// MARK: - Errors

/// A failure surfaced by the runtime, carrying the *stable* error code the
/// embedding contract defines (never a raw errno, never a numeric-only code).
public struct SofuuError: Error, CustomStringConvertible {
    /// Stable, documented code: `unknown_method`, `js_exception`,
    /// `invalid_arg`, `bad_args`, `internal`, `unknown_space`,
    /// `model_unavailable`, `nomem`, …
    public let code: String
    /// Human-readable explanation, safe to show in a debug UI.
    public let message: String

    public init(code: String, message: String) {
        self.code = code
        self.message = message
    }

    public var description: String { "SofuuError(\(code)): \(message)" }

    /// True when this build simply does not have the requested method —
    /// distinct from the method existing and failing, which matters when
    /// you branch on features at runtime.
    public var isUnsupportedMethod: Bool { code == "unknown_method" }
}

// MARK: - Streaming events

/// One event from `stream(...)`.
public enum StreamEvent {
    /// A chunk of generated text (`ai.stream`).
    case delta(String)
    /// A lifecycle event from `agent.run` (`.start`, `.tool`, `.answer`, …).
    case step([String: Any])
    /// Terminal: the stream finished. `aborted` is true if it was cancelled.
    case done(deltas: Int, aborted: Bool, result: Any?)
    /// Terminal: the stream failed.
    case failure(SofuuError)

    /// The raw event JSON, exactly as the runtime emitted it.
    public var raw: [String: Any]? {
        switch self {
        case .step(let s): return s
        case .delta, .done, .failure: return nil
        }
    }
}

// MARK: - Runtime

/// An embedded Sofuu runtime. One instance owns one JS engine; it is
/// single-threaded — create and use it on one thread (or one Task/queue).
public final class Sofuu {
    private let rt: OpaquePointer

    /// The embedding space ids this build knows. The default (used when you
    /// pass `nil`) is `hash-768` — the same space the brain stores in, so
    /// embed-then-remember needs no space argument.
    public enum Space {
        /// 768-dim hashed embedder. Zero parameters, zero model file, and
        /// the same space the brain uses. This is the default.
        public static let hash = "hash-768"
        /// 64-dim learned space (sem1).
        public static let sem1 = "sem1-64"
        /// 64-dim learned space (sem2). Joint geometry with images.
        public static let sem2 = "sem2-64"
    }

    /// Everything the runtime needs from the outside world, in one struct.
    /// There is no ambient environment: an iOS app has no `$HOME` and no
    /// shell, so the key travels in config, not in the process env.
    public struct Config {
        /// Provider API keys, e.g. ["openai": "sk-…"]. Empty = fully offline.
        public var apiKeys: [String: String] = [:]
        /// Directory for config/state. Defaults to `$HOME/.sofuu` when nil.
        public var configRoot: String? = nil
        /// Explicit brain file, overriding the default location.
        public var brainPath: String? = nil
        /// Whether `process.exit` throws a catchable `ExitError` (true) or
        /// terminates the process (false). Always true when embedded.
        public var embedded: Bool = true

        // Request defaults. Set these once here instead of repeating
        // provider/model on every call — an app has no interactive `/model`
        // picker. A per-call value still wins over these.
        public var provider: String? = nil
        public var model: String? = nil
        /// Endpoint override — set this to point the whole runtime at a
        /// local OpenAI-compatible server (Ollama, vLLM, LM Studio).
        public var baseURL: String? = nil

        public init() {}

        /// Ready-made config for a local OpenAI-compatible server.
        public static func local(baseURL: String, model: String) -> Config {
            var c = Config()
            c.provider = "openai"
            c.model = model
            c.baseURL = baseURL
            return c
        }

        var json: String {
            var obj: [String: Any] = ["embedded": embedded]
            if !apiKeys.isEmpty { obj["api_keys"] = apiKeys }
            if let configRoot { obj["config_root"] = configRoot }
            if let brainPath { obj["brain_path"] = brainPath }
            if let provider { obj["provider"] = provider }
            if let model { obj["model"] = model }
            if let baseURL { obj["base_url"] = baseURL }
            return Sofuu.jsonString(obj)
        }
    }

    /// Create a runtime. Throws if the engine cannot start.
    public init(_ config: Config = Config()) throws {
        let json = config.json
        let created = json.withCString { sofuu_rt_new($0) }
        guard let created else {
            throw SofuuError(code: "internal", message: "sofuu_rt_new returned NULL (bad config?)")
        }
        rt = created
    }

    deinit { sofuu_rt_free(rt) }

    /// The ABI version this build implements (2).
    public static var abiVersion: UInt32 { sofuu_embed_abi_version() }

    // MARK: Embeddings (offline; no key, no network)

    /// Embed one string. Returns a unit-length vector of `space`'s dimension
    /// (768 for the default space). Throws on an unknown space id.
    public func embed(_ text: String, space: String? = nil) throws -> [Float] {
        var out: UnsafeMutablePointer<Float>?
        var dim = 0
        // `space == nil` selects the build default (hash-768), which is the
        // same space the brain stores in.
        let rc = Self.withOptionalCString(space) { s in
            text.withCString { t in sofuu_embed_local(rt, t, s, &out, &dim) }
        }
        guard rc == 0, let out else {
            throw SofuuError(code: "embed_failed",
                             message: "sofuu_embed_local rc=\(rc) for space \(space ?? "default")")
        }
        defer { sofuu_free(out) }
        return Array(UnsafeBufferPointer(start: out, count: dim))
    }

    /// Embed many strings in one call (the batch is one pass, not N).
    public func embedBatch(_ texts: [String], space: String? = nil) throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        cStrings.reserveCapacity(texts.count)
        for t in texts {
            guard let dup = strdup(t) else {
                cStrings.forEach { free($0) }
                throw SofuuError(code: "nomem", message: "out of memory copying text")
            }
            cStrings.append(dup)
        }
        defer { cStrings.forEach { free($0) } }

        var out: UnsafeMutablePointer<Float>?
        var rows = 0, dim = 0
        let rc = Self.withOptionalCString(space) { s in
            cStrings.withUnsafeMutableBufferPointer { buf -> Int32 in
                // `const char **` imports as a mutable outer pointer, so
                // rebind the buffer rather than copying it.
                let p = UnsafeMutableRawPointer(buf.baseAddress!)
                    .assumingMemoryBound(to: UnsafePointer<CChar>?.self)
                return sofuu_embed_batch(rt, p, texts.count, s, &out, &rows, &dim)
            }
        }
        guard rc == 0, let out else {
            throw SofuuError(code: "embed_batch_failed", message: "sofuu_embed_batch rc=\(rc)")
        }
        defer { sofuu_free(out) }
        let buffer = UnsafeBufferPointer(start: out, count: rows * dim)
        return (0..<rows).map { r in Array(buffer[(r * dim)..<((r + 1) * dim)]) }
    }

    /// Embed image bytes (PNG/JPEG) into the image space, which is joint
    /// with `sem2` text geometry — so a *text* query can retrieve images.
    public func embedImage(_ bytes: Data) throws -> [Float] {
        var out: UnsafeMutablePointer<Float>?
        var dim = 0
        let rc = bytes.withUnsafeBytes { raw in
            sofuu_embed_image(rt, raw.bindMemory(to: UInt8.self).baseAddress,
                              raw.count, &out, &dim)
        }
        guard rc == 0, let out else {
            throw SofuuError(code: "embed_image_failed", message: "sofuu_embed_image rc=\(rc)")
        }
        defer { sofuu_free(out) }
        return Array(UnsafeBufferPointer(start: out, count: dim))
    }

    /// Which embedding spaces this build actually has, and which is default.
    public func embeddingSpaces() throws -> [String: Any] {
        let info = try callRaw("ai.embedInfo")
        return info
    }

    // MARK: The JSON funnel (escape hatch for anything not wrapped below)

    /// Call any runtime method by name with a JSON-ish object, and get the
    /// decoded result. This is the same funnel the C API exposes, so the
    /// full runtime is reachable from Swift even as we add wrappers.
    public func call(_ method: String, _ args: [String: Any] = [:]) throws -> Any {
        let argsJSON = Sofuu.jsonString(args)
        var out: UnsafeMutablePointer<CChar>?
        let rc = argsJSON.withCString { a in
            method.withCString { m in sofuu_rt_call(rt, m, a, &out) }
        }
        guard let out else {
            throw SofuuError(code: "internal",
                             message: "sofuu_rt_call(\(method)) produced no output (rc=\(rc))")
        }
        defer { sofuu_free(out) }
        return try Sofuu.decodeEnvelope(String(cString: out))
    }

    // MARK: Streaming

    /// Run a streaming method. The returned handle cancels the in-flight
    /// stream; the stream always ends with a `.done` or `.failure` event.
    public func stream(_ method: String,
                       _ args: [String: Any] = [:],
                       onEvent: @escaping (StreamEvent) -> Void) -> StreamHandle {
        let box = EventBox(onEvent: onEvent, runtime: rt)
        // The box outlives this call via `StreamHandle.box` (strong) and the
        // local `box` during it, so an unretained pointer is correct here —
        // passRetained without a matching release would leak per stream.
        let boxPtr = Unmanaged.passUnretained(box).toOpaque()
        let argsJSON = Sofuu.jsonString(args)

        let cb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void =
            { json, opaque in
                guard let json, let opaque else { return }
                let box = Unmanaged<EventBox>.fromOpaque(opaque).takeUnretainedValue()
                box.handle(String(cString: json))
            }

        var cancelID: UInt64 = 0
        let rc = argsJSON.withCString { a in
            method.withCString { m in
                sofuu_rt_call_stream(rt, m, a, cb, boxPtr, &cancelID)
            }
        }
        box.cancelID = cancelID
        if rc != 0 {
            onEvent(.failure(SofuuError(code: "stream_failed",
                                        message: "sofuu_rt_call_stream rc=\(rc)")))
        }
        return StreamHandle(runtime: rt, cancelID: cancelID, box: box)
    }

    // MARK: Private

    fileprivate func callRaw(_ method: String) throws -> [String: Any] {
        guard let dict = try call(method) as? [String: Any] else {
            throw SofuuError(code: "internal", message: "\(method) did not return an object")
        }
        return dict
    }

    static func jsonString(_ obj: Any) -> String {
        guard JSONSerialization.isValidJSONObject(obj),
              let data = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return s
    }

    /// Call `body` with a C string for `value`, or NULL when `value` is nil
    /// (which is how the C API spells "use the default").
    static func withOptionalCString<T>(_ value: String?,
                                       _ body: (UnsafePointer<CChar>?) -> T) -> T {
        guard let value else { return body(nil) }
        return value.withCString { body($0) }
    }

    /// Decode the `{"ok":…}` envelope, throwing a typed error on `ok:false`.
    static func decodeEnvelope(_ json: String) throws -> Any {
        guard let data = json.data(using: .utf8),
              let envelope = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SofuuError(code: "internal", message: "unparseable envelope: \(json)")
        }
        if let ok = envelope["ok"] as? Bool, ok {
            // `result` may legitimately be NSNull.
            if let result = envelope["result"], !(result is NSNull) { return result }
            return [:]
        }
        let err = envelope["error"] as? [String: Any] ?? [:]
        throw SofuuError(code: err["code"] as? String ?? "unknown",
                         message: err["message"] as? String ?? json)
    }
}

// MARK: - Stream handle

/// A handle to an in-flight stream; call `cancel()` to stop it.
public struct StreamHandle {
    private let runtime: OpaquePointer
    private let cancelID: UInt64
    fileprivate let box: EventBox

    fileprivate init(runtime: OpaquePointer, cancelID: UInt64, box: EventBox) {
        self.runtime = runtime
        self.cancelID = cancelID
        self.box = box
    }

    /// Ask the runtime to stop. The stream then ends with `.done(aborted: true)`.
    public func cancel() {
        sofuu_rt_cancel(runtime, cancelID)
    }
}

/// Bridges runtime events into Swift `StreamEvent`s. Boxed so the C
/// callback can carry a Swift closure across the FFI boundary.
final class EventBox {
    private let onEvent: (StreamEvent) -> Void
    private let runtime: OpaquePointer
    var cancelID: UInt64 = 0

    init(onEvent: @escaping (StreamEvent) -> Void, runtime: OpaquePointer) {
        self.onEvent = onEvent
        self.runtime = runtime
    }

    func handle(_ json: String) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            onEvent(.step(["raw": json]))
            return
        }
        // A terminal done-with-error surfaces as a failure.
        if obj["kind"] as? String == "done" {
            if let err = obj["error"] as? [String: Any] {
                onEvent(.failure(SofuuError(code: err["code"] as? String ?? "unknown",
                                            message: err["message"] as? String ?? "")))
            } else {
                onEvent(.done(deltas: obj["deltas"] as? Int ?? 0,
                              aborted: obj["aborted"] as? Bool ?? false,
                              result: obj["result"]))
            }
            return
        }
        if obj["kind"] as? String == "delta" {
            onEvent(.delta(obj["text"] as? String ?? ""))
            return
        }
        onEvent(.step(obj))
    }
}

// MARK: - The brain (encrypted local memory)

/// An open brain. This is the differentiator: memory the app owns —
/// encrypted on disk, local, exportable, with no provider in the loop.
public final class Brain {
    private let sofuu: Sofuu
    private let handle: Int

    init(sofuu: Sofuu, handle: Int) {
        self.sofuu = sofuu
        self.handle = handle
    }

    /// Open (or create) a brain file. `dimension` should match the embedding
    /// space you store in — use `Sofuu.Space.hash` (the default) unless you
    /// have a reason not to.
    public static func open(at url: URL,
                            dimension: Int = 768,
                            space: String = "hash-v1",
                            sofuu: Sofuu) throws -> Brain {
        let args: [String: Any] = [
            "path": url.path,
            "dim": dimension,
            "embed_id": space,
        ]
        let result = try sofuu.call("memory.open", args)
        guard let dict = result as? [String: Any], let h = dict["handle"] as? Int else {
            throw SofuuError(code: "internal", message: "memory.open returned no handle")
        }
        return Brain(sofuu: sofuu, handle: h)
    }

    /// Open a brain, defaulting to a location in the app's support dir.
    public static func openDefault(name: String = "brain", sofuu: Sofuu) throws -> Brain {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent("Sofuu", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return try open(at: dir.appendingPathComponent("\(name).qtsq"), sofuu: sofuu)
    }

    /// Embed `text` in the brain's own space and store it. Returns the index.
    @discardableResult
    public func remember(_ text: String, role: String = "user", kvPageID: UInt32 = 0) throws -> Int {
        // The brain is hash-768 by default, so embed in the default space
        // to guarantee the dimensions line up.
        let vec = try sofuu.embed(text, space: Sofuu.Space.hash)
        return try remember(vector: vec, text: text, role: role, kvPageID: kvPageID)
    }

    /// Store an already-computed vector. Throws if it does not match the
    /// brain's dimension — the runtime refuses mismatched writes rather
    /// than silently corrupting recall.
    @discardableResult
    public func remember(vector: [Float], text: String,
                         role: String = "user", kvPageID: UInt32 = 0) throws -> Int {
        let args: [String: Any] = [
            "handle": handle,
            "vec": vector,
            "text": text,
            "role": role,
            "kv_page_id": Int(kvPageID),
        ]
        let result = try sofuu.call("memory.remember", args)
        return (result as? Int) ?? -1
    }

    /// Retrieve the `k` most similar memories. Queries are embedded for you.
    public func recall(_ query: String, k: Int = 5) throws -> [MemoryHit] {
        let vec = try sofuu.embed(query, space: Sofuu.Space.hash)
        return try recall(vector: vec, k: k)
    }

    /// Retrieve by pre-computed vector.
    public func recall(vector: [Float], k: Int = 5) throws -> [MemoryHit] {
        let args: [String: Any] = ["handle": handle, "vec": vector, "k": k]
        let result = try sofuu.call("memory.recall", args)
        guard let arr = result as? [[String: Any]] else { return [] }
        return arr.map { MemoryHit(text: $0["text"] as? String ?? "",
                                   score: $0["score"] as? Double ?? 0,
                                   role: $0["role"] as? String ?? "") }
    }

    /// How many memories are stored.
    public func count() throws -> Int {
        (try sofuu.call("memory.count", ["handle": handle]) as? Int) ?? 0
    }

    /// Flush pending writes to disk. The brain persists automatically; call
    /// this before the app is suspended if you want the write to be durable
    /// immediately.
    public func flush() throws {
        _ = try sofuu.call("memory.flush", ["handle": handle])
    }
}

/// One memory returned by `recall`.
public struct MemoryHit {
    public let text: String
    public let score: Double
    public let role: String
}
