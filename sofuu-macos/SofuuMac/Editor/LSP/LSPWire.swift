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
        let documentation: String?

        init(label: String, insert: String? = nil, detail: String? = nil, kind: Int = 0,
             documentation: String? = nil) {
            self.label = label
            self.insert = insert ?? label
            self.detail = detail
            self.kind = kind
            self.documentation = documentation
        }
    }

    /// LSP snippet → plain insert: drop tab stops ($n / $0), keep and
    /// unbrace placeholders (${n:text} → text), leave \$ escapes alone.
    static func unwrapSnippet(_ s: String) -> String {
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            let ch = s[i]
            if ch == "\\", s.index(after: i) < s.endIndex {
                out.append(ch)
                i = s.index(after: i)
                out.append(s[i])
                i = s.index(after: i)
                continue
            }
            if ch == "$" {
                let after = s.index(after: i)
                guard after < s.endIndex else { out.append(ch); i = after; continue }
                let next = s[after]
                if next == "{",
                   let close = s[after...].firstIndex(of: "}") {
                    let body = String(s[s.index(after: after)..<close])
                    if let colon = body.firstIndex(of: ":") {
                        out += String(body[body.index(after: colon)...])
                    }
                    i = s.index(after: close)
                    continue
                }
                if next.isNumber {
                    i = after
                    while i < s.endIndex, s[i].isNumber { i = s.index(after: i) }
                    continue
                }
            }
            out.append(ch)
            i = s.index(after: i)
        }
        return out
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
            let raw = (d["insertText"] as? String) ?? nil
            let fromEdit = (((d["textEdit"] as? [String: Any])?["newText"] as? String) ?? nil)
            let insert = unwrapSnippet(raw ?? fromEdit ?? label)
            let documentation: String?
            if let doc = d["documentation"] as? String {
                documentation = doc
            } else if let doc = d["documentation"] as? [String: Any],
                      let value = doc["value"] as? String {
                documentation = value
            } else {
                documentation = nil
            }
            return CompletionItem(
                label: label,
                insert: insert,
                detail: d["detail"] as? String,
                kind: (d["kind"] as? Int) ?? 0,
                documentation: documentation)
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

    // MARK: - Semantic tokens

    /// legend.tokenTypes from capabilities (nil when unsupported). The
    /// provider may be a bare boolean (no legend → unusable here).
    static func parseSemanticLegend(_ result: Any?) -> [String]? {
        guard let root = result as? [String: Any],
              let caps = root["capabilities"] as? [String: Any],
              caps["semanticTokensProvider"] != nil else { return nil }
        guard let provider = caps["semanticTokensProvider"] as? [String: Any],
              let legend = provider["legend"] as? [String: Any],
              let types = legend["tokenTypes"] as? [String], !types.isEmpty else { return nil }
        return types
    }

    /// Decoded semantic spans in LSP line/character space.
    struct SemanticSpan {
        let line: Int
        let startCharacter: Int
        let length: Int
        let tokenType: String
    }

    static func decodeSemanticTokens(_ data: [Int], legend: [String]) -> [SemanticSpan] {
        var spans: [SemanticSpan] = []
        spans.reserveCapacity(data.count / 5)
        var line = 0
        var startCharacter = 0
        var i = 0
        while i + 4 < data.count {
            let deltaLine = data[i]
            let deltaStart = data[i + 1]
            let length = data[i + 2]
            let typeIndex = data[i + 3]
            if deltaLine > 0 {
                line += deltaLine
                startCharacter = deltaStart
            } else {
                startCharacter += deltaStart
            }
            if typeIndex >= 0, typeIndex < legend.count {
                spans.append(SemanticSpan(line: line, startCharacter: startCharacter,
                                          length: length, tokenType: legend[typeIndex]))
            }
            i += 5
        }
        return spans
    }

    // MARK: - Workspace edits (rename / code actions)

    /// One edit in LSP Position space (converted per-file by the caller,
    /// which owns that file's line starts).
    struct RawEdit {
        let line: Int
        let character: Int
        let endLine: Int
        let endCharacter: Int
        let newText: String
    }

    /// file path → raw LSP edits. `changes` shape only (documentChanges
    /// with versioned entries is rarer and unsupported here).
    static func parseWorkspaceEdit(_ any: Any?) -> [String: [RawEdit]] {
        guard let edit = any as? [String: Any],
              let changes = edit["changes"] as? [String: Any] else { return [:] }
        var out: [String: [RawEdit]] = [:]
        for (uri, list) in changes {
            guard let arr = list as? [[String: Any]] else { continue }
            let edits: [RawEdit] = arr.compactMap { te in
                guard let r = LSPRange.parse(te["range"]),
                      let newText = te["newText"] as? String else { return nil }
                return RawEdit(line: r.start.line, character: r.start.character,
                               endLine: r.end.line, endCharacter: r.end.character,
                               newText: newText)
            }
            out[pathOf(uri)] = edits
        }
        return out
    }

    /// title + the parsed edit set + whether it carries only a command
    /// (which we cannot run — surfaced honestly in the list).
    struct CodeAction {
        let title: String
        let isEditOnly: Bool
        let edits: [String: [RawEdit]]
    }

    static func parseCodeActions(_ any: Any?) -> [CodeAction] {
        guard let list = any as? [[String: Any]] else { return [] }
        return list.compactMap { d in
            guard let title = d["title"] as? String else { return nil }
            let editList = parseWorkspaceEdit(d["edit"])
            let hasCommand = d["command"] != nil
            return CodeAction(title: title,
                              isEditOnly: !hasCommand,
                              edits: editList)
        }
    }

    /// textDocument/workspaceSymbol hits (flat SymbolInformation).
    struct WSSymbol {
        let name: String
        let kind: Int
        let path: String
        let line: Int
        let character: Int
    }

    static func parseWorkspaceSymbols(_ any: Any?) -> [WSSymbol] {
        guard let list = any as? [[String: Any]] else { return [] }
        return list.compactMap { d in
            guard let name = d["name"] as? String,
                  let loc = d["location"] as? [String: Any],
                  let uri = loc["uri"] as? String,
                  let r = LSPRange.parse(loc["range"]) else { return nil }
            return WSSymbol(name: name,
                            kind: (d["kind"] as? Int) ?? 0,
                            path: pathOf(uri),
                            line: r.start.line,
                            character: r.start.character)
        }
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
