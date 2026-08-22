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

    /// Embed text using the bundled offline embedder (no network).
    public func embedLocal(_ text: String) -> [Float]? {
        let js = "JSON.stringify(Array.from(sofuu.ai.embedLocal(\(jsString(text)))))"
        guard let result = eval(js) else { return nil }
        // Parse the JSON array of numbers.
        guard let data = result.data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [Double] else {
            return nil
        }
        return arr.map { Float($0) }
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
        if let config = config {
            return OpaquePointer(sofuu_rt_new(config))
        } else {
            return OpaquePointer(sofuu_rt_new(nil))
        }
    }

    private func jsString(_ s: String) -> String {
        // Simple JSON string escaping.
        var escaped = s.replacingOccurrences(of: "\\", with: "\\\\")
        escaped = escaped.replacingOccurrences(of: "\"", with: "\\\"")
        escaped = escaped.replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(escaped)\""
    }
}
