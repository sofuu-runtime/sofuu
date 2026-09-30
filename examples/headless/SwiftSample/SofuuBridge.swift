// examples/headless/SwiftSample/SofuuBridge.swift
//
// A minimal Swift wrapper around the libsofuu C ABI (PLAN-HEADLESS H5).
// QuickJS is a pure interpreter (no JIT) → iOS App Store safe by design.
//
// Usage in a SwiftUI view:
//
//   import SwiftUI
//
//   struct ContentView: View {
//       @State private var output = "Ready."
//       private let sofuu = SofuuBridge()
//
//       var body: some View {
//           VStack(spacing: 16) {
//               Text(output)
//               Button("Eval 1+1") { output = sofuu.eval("1 + 1") }
//               Button("Check capabilities") { output = sofuu.checkCaps() }
//           }
//           .padding()
//       }
//   }
//
// To use in Xcode:
//   1. Add the libsofuu.xcframework to your app target (General → Frameworks).
//   2. Add a bridging header with: #import "sofuu_embed.h"
//   3. Or use the module map from the xcframework (import Sofuu).
//

import Foundation

/// A thin Swift wrapper around the Sofuu C ABI.
/// One SofuuRuntime per thread; never share across threads.
public final class SofuuBridge: @unchecked Sendable {

    private let rt: OpaquePointer

    /// Create a runtime. Config is optional (nil = all defaults).
    /// - Parameter config: JSON config string (e.g. `{"embedded":true,"config_root":"..."}`)
    public init(config: String? = nil) {
        if let cfg = config {
            self.rt = cfg.withCString { SofuuBridge.rtNew($0) }
        } else {
            self.rt = SofuuBridge.rtNew(nil)
        }
    }

    deinit {
        sofuu_rt_free(rt)
    }

    /// The ABI version this library implements.
    public static var abiVersion: UInt32 {
        sofuu_embed_abi_version()
    }

    /// Evaluate arbitrary JS — the eval escape hatch.
    /// Returns the JSON-stringified result, or nil on failure.
    public func eval(_ source: String) -> String? {
        var out: UnsafeMutablePointer<CChar>? = nil
        let rc = source.withCString { sofuu_rt_eval(rt, $0, &out) }
        guard rc == 0, let out = out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Call a funnel method (e.g. "ai.complete", "ai.embed", "rlm.query").
    /// - Parameters:
    ///   - method: The dot-path method name.
    ///   - argsJson: JSON arguments (object or array), or nil for no args.
    /// - Returns: The JSON envelope result, or nil on failure.
    public func call(_ method: String, args: String? = nil) -> String? {
        var out: UnsafeMutablePointer<CChar>? = nil
        let rc: Int32
        if let args = args {
            rc = method.withCString { m in
                args.withCString { a in
                    sofuu_rt_call(rt, m, a, &out)
                }
            }
        } else {
            rc = method.withCString { m in
                sofuu_rt_call(rt, m, nil, &out)
            }
        }
        guard rc == 0, let out = out else { return nil }
        let result = String(cString: out)
        sofuu_free(out)
        return result
    }

    /// Check which features are available in this build.
    public func checkCaps() -> String {
        let js = """
        JSON.stringify({
            has_sofuu: typeof sofuu !== 'undefined',
            has_ai: typeof sofuu.ai !== 'undefined',
            has_agent: typeof sofuu.agent !== 'undefined',
            has_rlm: typeof sofuu.rlm !== 'undefined',
            has_web: typeof sofuu.web !== 'undefined',
            has_memory: typeof sofuu.memory !== 'undefined',
            has_mcp: typeof sofuu.mcp !== 'undefined',
            has_http: typeof sofuu.http !== 'undefined',
            has_fs: typeof sofuu.fs !== 'undefined'
        })
        """
        return eval(js) ?? "{}"
    }

    /// Embed text using the bundled offline embedder (no network, no JSON).
    /// - Parameters:
    ///   - text: The text to embed.
    ///   - space: "hash-768" (default — the same space the brain stores in),
    ///     "sem1-64", or "sem2-64".
    /// - Returns: The embedding vector, or nil on failure.
    public func embedLocal(_ text: String, space: String? = nil) -> [Float]? {
        var out: UnsafeMutablePointer<Float>? = nil
        var dim: Int = 0
        let rc: Int32 = text.withCString { t in
            if let space = space {
                return space.withCString { s in sofuu_embed_local(rt, t, s, &out, &dim) }
            }
            return sofuu_embed_local(rt, t, nil, &out, &dim)
        }
        guard rc == 0, let out = out else { return nil }
        let arr = Array(UnsafeBufferPointer(start: out, count: dim))
        sofuu_free(out)
        return arr
    }

    /// Embed many texts in one call (single lock, row-major internally).
    /// - Returns: One vector per input, or nil on failure.
    public func embedBatch(_ texts: [String], space: String? = nil) -> [[Float]]? {
        guard !texts.isEmpty else { return [] }
        var cPtrs: [UnsafeMutablePointer<CChar>?] = texts.map { strdup($0) }
        defer { cPtrs.forEach { free($0) } }
        var out: UnsafeMutablePointer<Float>? = nil
        var outN: Int = 0
        var outDim: Int = 0
        // Mutable buffer pointer for the call (callee takes const char** —
        // it never mutates the array); safe: texts is non-empty above.
        let rc: Int32 = cPtrs.withUnsafeMutableBufferPointer { buf in
            let base = UnsafeMutableRawPointer(buf.baseAddress!).assumingMemoryBound(to: UnsafePointer<CChar>?.self)
            if let space = space {
                return space.withCString { s in
                    sofuu_embed_batch(rt, base, texts.count, s, &out, &outN, &outDim)
                }
            }
            return sofuu_embed_batch(rt, base, texts.count, nil, &out, &outN, &outDim)
        }
        guard rc == 0, let out = out else { return nil }
        var rows: [[Float]] = []
        rows.reserveCapacity(outN)
        for i in 0..<outN {
            rows.append(Array(UnsafeBufferPointer(start: out.advanced(by: i * outDim), count: outDim)))
        }
        sofuu_free(out)
        return rows
    }

    /// Embed image bytes (PNG/JPEG) into the img1-64 space (joint with
    /// sem2-64 text geometry, so text queries retrieve images).
    /// - Returns: The 64-dim unit vector, or nil on failure.
    public func embedImage(_ bytes: [UInt8]) -> [Float]? {
        guard !bytes.isEmpty else { return nil }
        var out: UnsafeMutablePointer<Float>? = nil
        var dim: Int = 0
        let rc: Int32 = bytes.withUnsafeBufferPointer { buf in
            sofuu_embed_image(rt, buf.baseAddress!, bytes.count, &out, &dim)
        }
        guard rc == 0, let out = out else { return nil }
        let arr = Array(UnsafeBufferPointer(start: out, count: dim))
        sofuu_free(out)
        return arr
    }

    /// Provider transcription via the libsofuu funnel (BYO key in config).
    /// For offline use `SofuuVoice` (OS speech) instead.
    /// - Returns: The transcript text, or nil on failure.
    public func transcribeProvider(_ audio: [UInt8], opts: String? = nil) -> String? {
        let b64 = Data(audio).base64EncodedString()
        var out: UnsafeMutablePointer<CChar>? = nil
        let rc: Int32 = b64.withCString { b in
            if let opts = opts {
                return opts.withCString { o in sofuu_voice_transcribe(rt, b, o, &out) }
            }
            return sofuu_voice_transcribe(rt, b, nil, &out)
        }
        guard rc == 0, let out = out else { return nil }
        defer { sofuu_free(out) }
        guard let data = String(cString: out).data(using: .utf8),
              let env = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = env["result"] as? [String: Any],
              let text = result["text"] as? String else { return nil }
        return text
    }

    /// Provider speech via the libsofuu funnel. The audio travels the JSON
    /// envelope as an indexed-byte object; reassembled here.
    /// - Returns: (audio bytes, format), or nil on failure.
    public func speakProvider(_ text: String, opts: String? = nil) -> (bytes: [UInt8], format: String)? {
        var out: UnsafeMutablePointer<CChar>? = nil
        let rc: Int32 = text.withCString { t in
            if let opts = opts {
                return opts.withCString { o in sofuu_voice_speak(rt, t, o, &out) }
            }
            return sofuu_voice_speak(rt, t, nil, &out)
        }
        guard rc == 0, let out = out else { return nil }
        defer { sofuu_free(out) }
        guard let data = String(cString: out).data(using: .utf8),
              let env = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = env["result"] as? [String: Any],
              let indexed = result["audio"] as? [String: Any],
              let format = result["format"] as? String else { return nil }
        let bytes: [UInt8] = indexed.keys.compactMap { Int($0) }.sorted().compactMap { k in
            (indexed[String(k)] as? NSNumber).map { UInt8(truncating: $0) }
        }
        guard !bytes.isEmpty else { return nil }
        return (bytes, format)
    }

    /// Space manifest JSON (default space, model ids, dims, artifact hashes).
    public func embedInfo() -> String? {
        var out: UnsafeMutablePointer<CChar>? = nil
        let rc = sofuu_embed_info(rt, &out)
        guard rc == 0, let out = out else { return nil }
        let s = String(cString: out)
        sofuu_free(out)
        return s
    }

    /// Compute cosine similarity between two vectors.
    public func similarity(_ a: [Float], _ b: [Float]) -> Float {
        let aJS = "[" + a.map { String($0) }.joined(separator: ",") + "]"
        let bJS = "[" + b.map { String($0) }.joined(separator: ",") + "]"
        let js = "sofuu.ai.similarity(new Float32Array(\(aJS)), new Float32Array(\(bJS)))"
        guard let result = eval(js),
              let val = Double(result) else { return 0 }
        return Float(val)
    }

    // MARK: - Private helpers

    private static func rtNew(_ config: UnsafePointer<CChar>?) -> OpaquePointer {
        // sofuu_rt_new returns NULL only on OOM; the handle is non-optional
        // by design, so fail loudly instead of smuggling a null.
        let p: OpaquePointer? = (config == nil) ? sofuu_rt_new(nil) : sofuu_rt_new(config)
        precondition(p != nil, "sofuu_rt_new failed (out of memory)")
        return p!
    }

    private func jsString(_ s: String) -> String {
        // Simple JSON string escaping.
        var escaped = s.replacingOccurrences(of: "\\", with: "\\\\")
        escaped = escaped.replacingOccurrences(of: "\"", with: "\\\"")
        escaped = escaped.replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(escaped)\""
    }
}
