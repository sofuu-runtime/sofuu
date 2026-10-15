import Foundation

/// JSON helpers — the Swift stand-in for the Android app's `org.json` usage.
///
/// The engine speaks JSON on every boundary (the C ABI is JSON-shaped by
/// design), so the whole app leans on these. The `opt*` family deliberately
/// mirrors `org.json`'s `optString`/`optInt`/… semantics — missing key or
/// JSON null yields the default, never a crash — which is what makes the
/// port from Kotlin a line-for-line translation rather than a rewrite.

typealias JSONObject = [String: Any]

// ── Decoding ────────────────────────────────────────────────────────

func parseJSONObject(_ text: String) -> JSONObject? {
    guard let data = text.data(using: .utf8) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) as? JSONObject
}

func parseJSONArray(_ text: String) -> [Any]? {
    guard let data = text.data(using: .utf8) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) as? [Any]
}

/// Parse any JSON fragment (object, array, string, number) — used by the
/// reply unwrapper, where the wire value may be a JSON-encoded string that
/// itself contains JSON.
func parseJSONFragment(_ text: String) -> Any? {
    guard let data = text.data(using: .utf8) else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

// ── Encoding ────────────────────────────────────────────────────────

func jsonEncode(_ obj: Any) -> String {
    guard JSONSerialization.isValidJSONObject(obj),
          let data = try? JSONSerialization.data(withJSONObject: obj),
          let s = String(data: data, encoding: .utf8) else { return "{}" }
    return s
}

/// JSON-encode a Swift string as a JS string literal (JSON is a subset).
func jsonQuote(_ s: String) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed]),
          let out = String(data: data, encoding: .utf8) else { return "\"\"" }
    return out
}

// ── org.json-style accessors ────────────────────────────────────────

extension Dictionary where Key == String, Value == Any {

    func has(_ key: String) -> Bool { self[key] != nil }

    private func raw(_ key: String) -> Any? {
        guard let v = self[key], !(v is NSNull) else { return nil }
        return v
    }

    func optString(_ key: String, _ def: String = "") -> String {
        guard let v = raw(key) else { return def }
        if let s = v as? String { return s }
        if let n = v as? NSNumber { return n.stringValue }
        return def
    }

    func optInt(_ key: String, _ def: Int = 0) -> Int {
        guard let v = raw(key) else { return def }
        if let n = v as? NSNumber { return n.intValue }
        if let i = v as? Int { return i }
        if let s = v as? String, let i = Int(s) { return i }
        return def
    }

    /// Long is Int on 64-bit iOS — kept as an alias for parity with the port.
    func optLong(_ key: String, _ def: Int = 0) -> Int { optInt(key, def) }

    func optDouble(_ key: String, _ def: Double = 0) -> Double {
        guard let v = raw(key) else { return def }
        if let n = v as? NSNumber { return n.doubleValue }
        if let d = v as? Double { return d }
        if let s = v as? String, let d = Double(s) { return d }
        return def
    }

    func optBool(_ key: String, _ def: Bool = false) -> Bool {
        guard let v = raw(key) else { return def }
        if let b = v as? Bool { return b }
        if let n = v as? NSNumber { return n.boolValue }
        return def
    }

    func optObject(_ key: String) -> JSONObject? { raw(key) as? JSONObject }

    func optArray(_ key: String) -> [Any]? { raw(key) as? [Any] }

    /// Stringify a nested value the way `org.json`'s `opt(...).toString()`
    /// does — used for the "detail" lines that echo arbitrary payloads.
    func optStringy(_ key: String) -> String? {
        guard let v = raw(key) else { return nil }
        if let s = v as? String { return s }
        if let n = v as? NSNumber { return n.stringValue }
        if let arr = v as? [Any], JSONSerialization.isValidJSONObject(arr) { return jsonEncode(arr) }
        if let o = v as? JSONObject { return jsonEncode(o) }
        return String(describing: v)
    }
}
