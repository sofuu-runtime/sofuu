import Foundation

/// The wire subset we speak. Hand-mapped from the Language Server
/// Protocol spec (JSON-RPC 2.0 over stdio) — implemented from first
/// principles; no generated or third-party layer.
///
/// Servers reply with heterogeneous payloads (hover contents can be a
/// string, markup, or an array of either; symbols can be hierarchical
/// DocumentSymbols or flat SymbolInformation), so parsing runs over
/// JSONSerialization dictionaries with tolerant extraction instead of
/// rigid Codable shapes.
enum LSPWire {

    // MARK: - Geometry

    struct Position {
        let line: Int           // 0-based
        let character: Int      // 0-based UTF-16 offset in the line

        static func parse(_ any: Any?) -> Position? {
            guard let d = any as? [String: Any],
                  let line = d["line"] as? Int,
                  let character = d["character"] as? Int else { return nil }
            return Position(line: line, character: character)
        }
    }

    struct LSPRange {
        let start: Position
        let end: Position

        static func parse(_ any: Any?) -> LSPRange? {
            guard let d = any as? [String: Any] else { return nil }
            return LSPRange(start: Position.parse(d["start"]) ?? .init(line: 0, character: 0),
                            end: Position.parse(d["end"]) ?? .init(line: 0, character: 0))
        }
    }

    // MARK: - Diagnostics

    struct Diagnostic {
        let line: Int           // 0-based start line for the gutter
        let character: Int
        let endLine: Int
        let endCharacter: Int
        let severity: Int       // 1=error, 2=warning, 3=info, 4=hint
        let message: String

        var isError: Bool { severity <= 1 }
        var isWarning: Bool { severity == 2 }
    }

    static func parseDiagnostics(_ any: Any?) -> [Diagnostic] {
        guard let list = any as? [[String: Any]] else { return [] }
        return list.compactMap { d in
            guard let range = LSPRange.parse(d["range"]) else { return nil }
            let message = (d["message"] as? String) ?? ""
            let severity = (d["severity"] as? Int) ?? 1
            return Diagnostic(line: range.start.line,
                              character: range.start.character,
                              endLine: range.end.line,
                              endCharacter: range.end.character,
                              severity: severity,
                              message: message)
        }
    }

    // MARK: - Hover

    /// Flattens MarkupContent | MarkedString | [both] into plain text.
    static func parseHover(_ any: Any?) -> String? {
        guard let d = any as? [String: Any] else { return nil }
        return flattenHover(d["contents"])
    }

    private static func flattenHover(_ contents: Any?) -> String? {
        if let s = contents as? String { return s.isEmpty ? nil : s }
        if let d = contents as? [String: Any], let v = d["value"] as? String { return v.isEmpty ? nil : v }
        if let list = contents as? [Any] {
            let parts = list.compactMap { flattenHover($0) }
            let joined = parts.joined(separator: "\n\n")
            return joined.isEmpty ? nil : joined
        }
        return nil
    }

    // MARK: - Completion

    struct CompletionItem {
        let label: String
        let insert: String
        let detail: String?
        let kind: Int

        init(label: String, insert: String? = nil, detail: String? = nil, kind: Int = 0) {
            self.label = label
            self.insert = insert ?? label
            self.detail = detail
            self.kind = kind
        }
    }

    static func parseCompletionItems(_ any: Any?) -> [CompletionItem] {
        let items: [[String: Any]]
        if let list = any as? [[String: Any]] {
            items = list
        } else if let wrapped = any as? [String: Any],
                  let list = wrapped["items"] as? [[String: Any]] {
            items = list
        } else {
            return []
        }
        return items.compactMap { d in
            guard let label = d["label"] as? String else { return nil }
            let insert = (d["insertText"] as? String)
                ?? (((d["textEdit"] as? [String: Any])?["newText"] as? String) ?? label)
            return CompletionItem(label: label,
                                  insert: insert,
                                  detail: d["detail"] as? String,
                                  kind: (d["kind"] as? Int) ?? 0)
        }
    }

    // MARK: - Symbols (DocumentSymbol[] or SymbolInformation[])

    struct OutlineNode {
        let name: String
        let kind: Int
        let line: Int            // 0-based
        let children: [OutlineNode]
    }

    static func parseSymbols(_ any: Any?) -> [OutlineNode] {
        guard let list = any as? [[String: Any]] else { return [] }
        return list.compactMap { parseSymbol($0) }
    }

    private static func parseSymbol(_ d: [String: Any]) -> OutlineNode? {
        guard let name = d["name"] as? String else { return nil }
        let kind = (d["kind"] as? Int) ?? 0
        var line = 0
        if let range = LSPRange.parse(d["range"]) {
            line = range.start.line
        } else if let location = d["location"] as? [String: Any],
                  let r = LSPRange.parse(location["range"]) {
            line = r.start.line
        }
        let children = (d["children"] as? [[String: Any]])?.compactMap { parseSymbol($0) } ?? []
        return OutlineNode(name: name, kind: kind, line: line, children: children)
    }

    // MARK: - Definitions / references

    struct Location {
        let path: String        // file:// URI → filesystem path
        let line: Int
        let character: Int
    }

    static func parseLocations(_ any: Any?) -> [Location] {
        func one(_ d: [String: Any]) -> Location? {
            guard let uri = d["uri"] as? String, let range = LSPRange.parse(d["range"]) else { return nil }
            return Location(path: pathOf(uri), line: range.start.line, character: range.start.character)
        }
        if let d = any as? [String: Any], d["uri"] != nil {
            return [one(d)].compactMap { $0 }
        }
        if let list = any as? [[String: Any]] {
            return list.compactMap { one($0) }
        }
        return []
    }

    /// file:///Users/x/y.md → /Users/x/y.md
    static func pathOf(_ uri: String) -> String {
        if uri.hasPrefix("file://") {
            return URL(string: uri)?.path ?? String(uri.dropFirst("file://".count))
        }
        return uri
    }

    static func uriOf(_ path: String) -> String {
        URL(fileURLWithPath: path).absoluteString
    }
}
