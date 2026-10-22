import Foundation

/// A from-scratch incremental-friendly tokenizer. No regex, no borrowed
/// code: one pass over the UTF-16 view with an explicit state machine
/// that carries string/comment context across lines.
///
/// Produces:
///  - `tokens`: attributed coloring ranges (guaranteed non-overlapping)
///  - `pairs`: balanced bracket pairs (for match-at-cursor)
///  - `outline`: document symbols from language-aware patterns
enum Lexer {

    enum Kind {
        case plain, comment, string, number, keyword, type, decorator, op
    }

    struct Token { let range: NSRange; let kind: Kind }

    struct OutlineEntry {
        let name: String
        let kind: Int             // heading level for markdown; 0 for code
        let line: Int             // 0-based
    }

    // MARK: - Entry point

    static func lex(_ text: String, language: Language) -> (tokens: [Token], pairs: [(NSRange, NSRange)], outline: [OutlineEntry]) {
        if language.id == .markdown { return lexMarkdown(text) }
        return lexCode(text, language: language)
    }

    // MARK: - Code languages

    private static func lexCode(_ text: String, language: Language) -> (tokens: [Token], pairs: [(NSRange, NSRange)], outline: [OutlineEntry]) {
        let units = Array(text.utf16)
        var tokens: [Token] = []
        var pairs: [(NSRange, NSRange)] = []
        var outline: [OutlineEntry] = []

        enum State { case normal, lineComment, blockComment, dquote, squote, template, backtick }
        var state: State = .normal

        var i = 0
        let n = units.count

        var tokenStart = 0
        var tokenKind: Kind?

        func flush(_ end: Int) {
            guard let kind = tokenKind, end > tokenStart else { return }
            tokens.append(Token(range: NSRange(location: tokenStart, length: end - tokenStart), kind: kind))
            tokenKind = nil
        }
        func begin(_ at: Int, _ kind: Kind) { tokenStart = at; tokenKind = kind }

        func isWordStart(_ u: UTF16.CodeUnit) -> Bool {
            (u >= 65 && u <= 90) || (u >= 97 && u <= 122) || u == 0x5F
        }
        func isWord(_ u: UTF16.CodeUnit) -> Bool {
            isWordStart(u) || (u >= 48 && u <= 57)
        }

        var bracketStack: [(unit: UTF16.CodeUnit, at: Int)] = []
        let opens: [UTF16.CodeUnit] = [0x28, 0x5B, 0x7B]        // ( [ {
        let closes: [UTF16.CodeUnit] = [0x29, 0x5D, 0x7D]       // ) ] }

        // Line starts for outline scanning.
        var lineStarts: [Int] = [0]
        for (idx, u) in units.enumerated() where u == 0x0A {
            lineStarts.append(idx + 1)
        }

        while i < n {
            let u = units[i]

            switch state {
            case .lineComment:
                if u == 0x0A { flush(i); state = .normal } else { i += 1 }

            case .blockComment:
                if u == 0x2A /* * */, i + 1 < n, units[i + 1] == 0x2F /* / */ {
                    i += 2
                    flush(i)
                    state = .normal
                } else { i += 1 }

            case .dquote, .squote, .template, .backtick:
                // Escape pair first.
                if u == 0x5C /* \ */, i + 1 < n { i += 2; continue }
                let closed: Bool
                switch state {
                case .dquote: closed = u == 0x22
                case .squote: closed = u == 0x27
                case .template: closed = u == 0x60
                case .backtick: closed = u == 0x60
                default: closed = false
                }
                if closed || u == 0x0A {
                    i += 1
                    flush(i)
                    state = .normal
                } else {
                    i += 1
                }

            case .normal:
                // Comments
                if let lc = language.lineComment, startsAt(units, i, lc) {
                    flush(i); begin(i, .comment); state = .lineComment
                    i += lc.utf16.count
                    continue
                }
                if let (bo, bc) = language.blockComment, startsAt(units, i, bo) {
                    flush(i); begin(i, .comment); state = .blockComment
                    i += bo.utf16.count
                    _ = bc
                    continue
                }
                // Strings
                if u == 0x22 { flush(i); begin(i, .string); state = .dquote; i += 1; continue }
                if u == 0x27 { flush(i); begin(i, .string); state = .squote; i += 1; continue }
                if u == 0x60 { flush(i); begin(i, .string); state = .template; i += 1; continue }
                // Rust raw strings: r" ..." or r#" ... "#
                if language.id == .rust, u == 0x72 /* r */, i + 1 < n,
                   units[i + 1] == 0x22 || units[i + 1] == 0x23 {
                    var j = i + 1
                    while j < n, units[j] == 0x23 { j += 1 }
                    if j < n, units[j] == 0x22 {
                        // find closing " followed by the same number of #
                        var k = j + 1
                        var hashes = 0
                        while k < n {
                            if units[k] == 0x22 {
                                var m = k + 1
                                while m < n, units[m] == 0x23 { m += 1; hashes += 1 }
                                if hashes == j - i - 1 { k = m; break }
                                hashes = 0
                            }
                            k += 1
                        }
                        flush(i); begin(i, .string)
                        i = min(k + 1, n)
                        flush(i)
                        continue
                    }
                }
                // Decorators / attributes
                if u == 0x40 /* @ */, language.id == .swift || language.id == .python {
                    flush(i); begin(i, .decorator)
                    i += 1
                    while i < n, isWord(units[i]) { i += 1 }
                    flush(i)
                    continue
                }
                if u == 0x23 /* # */, language.id == .rust,
                   i + 1 < n, units[i + 1] == 0x5B /* [ */ {
                    // #[attribute ... ] — one line
                    flush(i); begin(i, .decorator)
                    while i < n, units[i] != 0x0A { i += 1 }
                    flush(i)
                    continue
                }
                // Numbers
                if u >= 48 && u <= 57 {
                    flush(i); begin(i, .number)
                    i += 1
                    while i < n {
                        let d = units[i]
                        if (d >= 48 && d <= 57) || d == 0x2E || d == 0x5F
                            || d == 0x78 || d == 0x62 || d == 0x6F        // x b o radix prefixes
                            || d == 0x65 || d == 0x45                      // e E exponent
                            || ((d == 0x2B || d == 0x2D) && i > 0 && (units[i-1] == 0x65 || units[i-1] == 0x45)) {
                            i += 1
                        } else { break }
                    }
                    flush(i)
                    continue
                }
                // Identifiers / keywords
                if isWordStart(u) {
                    flush(i); begin(i, .plain)
                    let wordStart = i
                    while i < n, isWord(units[i]) { i += 1 }
                    let word = String(decoding: units[wordStart..<i], as: UTF16.self)
                    if language.keywords.contains(word) {
                        tokenKind = .keyword
                    } else if !word.isEmpty, word.first?.isUppercase == true,
                              language.id != .shell {
                        tokenKind = .type
                    }
                    // Outline candidates (line-based check)
                    if let line = lineForOffset(wordStart, lineStarts),
                       isOutlineWord(word, atLine: line, offset: wordStart, units: units, language: language) {
                        // The outline name is the identifier AFTER the
                        // keyword ("func greet" → "greet").
                        var j = i
                        while j < n, !isWord(units[j]) { j += 1 }
                        var nameEnd = j
                        while nameEnd < n, isWord(units[nameEnd]) { nameEnd += 1 }
                        let name = j < nameEnd
                            ? String(decoding: units[j..<nameEnd], as: UTF16.self)
                            : word
                        outline.append(OutlineEntry(name: name, kind: 0, line: line))
                    }
                    continue
                }
                // Operators / punctuation
                if [0x2B, 0x2D, 0x3D, 0x21, 0x3C, 0x3E, 0x26, 0x7C, 0x2A, 0x2F, 0x25, 0x5E, 0x7E, 0x3F, 0x3A].contains(u) {
                    flush(i); begin(i, .op)
                    i += 1
                    while i < n, [0x2B, 0x2D, 0x3D, 0x21, 0x3C, 0x3E, 0x26, 0x7C, 0x2A, 0x2F, 0x25, 0x5E, 0x7E, 0x3F].contains(units[i]) {
                        i += 1
                    }
                    flush(i)
                    continue
                }
                // Brackets
                if opens.contains(u) {
                    flush(i)
                    bracketStack.append((u, i))
                    i += 1
                    continue
                }
                if closes.contains(u) {
                    flush(i)
                    if let last = bracketStack.last, matches(last.unit, u) {
                        pairs.append((NSRange(location: last.at, length: 1), NSRange(location: i, length: 1)))
                        bracketStack.removeLast()
                    }
                    i += 1
                    continue
                }

                i += 1
            }
        }
        flush(n)

        return (tokens, pairs, dedupeOutline(outline))
    }

    // MARK: - Markdown

    private static func lexMarkdown(_ text: String) -> (tokens: [Token], pairs: [(NSRange, NSRange)], outline: [OutlineEntry]) {
        var tokens: [Token] = []
        var outline: [OutlineEntry] = []
        var inFence = false
        var fenceLength = 0

        var lineStart = 0
        let units = Array(text.utf16)
        var idx = 0
        var line = 0

        func addToken(_ range: NSRange, _ kind: Kind) { tokens.append(Token(range: range, kind: kind)) }

        while idx <= units.count {
            let atLineEnd = idx == units.count || units[idx] == 0x0A
            if atLineEnd {
                let lineRange = NSRange(location: lineStart, length: idx - lineStart)
                let lineText = String(decoding: units[lineStart..<min(idx, units.count)], as: UTF16.self)
                    .trimmingCharacters(in: .whitespaces)

                if lineText.hasPrefix("```") {
                    let ticks = prefixRun(lineText, char: "`")
                    if !inFence {
                        inFence = true
                        fenceLength = ticks.count
                        addToken(lineRange, .comment)
                    } else if ticks.count >= fenceLength {
                        inFence = false
                        addToken(lineRange, .comment)
                    }
                } else if inFence {
                    addToken(lineRange, .plain)
                } else if lineText.hasPrefix("#") {
                    let hashes = prefixRun(lineText, char: "#").count
                    addToken(NSRange(location: lineStart, length: min(hashes + 1, lineRange.length)), .keyword)
                    addToken(NSRange(location: lineStart + min(hashes + 1, lineRange.length), length: max(0, lineRange.length - hashes - 1)), .type)
                    if hashes <= 6 {
                        let title = lineText.drop(while: { $0 == "#" || $0 == " " })
                        outline.append(OutlineEntry(name: String(title), kind: hashes, line: line))
                    }
                } else if lineText.hasPrefix("- ") || lineText.hasPrefix("* ") || lineText.hasPrefix("+ ") {
                    addToken(NSRange(location: lineStart, length: 1), .op)
                } else if lineText.hasPrefix("> ") {
                    addToken(lineRange, .comment)
                } else if lineText.contains("**") {
                    tokenizeInline(lineText, rangeLocation: lineStart, add: addToken)
                }
                line += 1
                lineStart = idx + 1
                idx += 1
                if idx > units.count { break }
            } else {
                idx += 1
            }
        }
        return (tokens, [], dedupeOutline(outline))
    }

    private static func tokenizeInline(_ line: String, rangeLocation: Int, add: (NSRange, Kind) -> Void) {
        let units = Array(line.utf16)
        var i = 0
        while i + 1 < units.count {
            if units[i] == 0x2A && units[i + 1] == 0x2A {
                if let close = find(units: units, from: i + 2, pair: 0x2A2A) {
                    add(NSRange(location: rangeLocation + i, length: 2), .keyword)
                    add(NSRange(location: rangeLocation + close, length: 2), .keyword)
                    i = close + 2
                    continue
                }
            }
            i += 1
        }
    }

    // MARK: - Helpers

    private static func startsAt(_ units: [UTF16.CodeUnit], _ at: Int, _ s: String) -> Bool {
        let sc = Array(s.utf16)
        guard at + sc.count <= units.count else { return false }
        for (k, c) in sc.enumerated() where units[at + k] != c { return false }
        return true
    }

    private static func matches(_ open: UTF16.CodeUnit, _ close: UTF16.CodeUnit) -> Bool {
        (open == 0x28 && close == 0x29) || (open == 0x5B && close == 0x5D) || (open == 0x7B && close == 0x7D)
    }

    private static func lineForOffset(_ offset: Int, _ lineStarts: [Int]) -> Int? {
        var lo = 0, hi = lineStarts.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if lineStarts[mid] <= offset { lo = mid } else { hi = mid - 1 }
        }
        return lo
    }

    private static func prefixRun(_ s: String, char: Character) -> String {
        String(s.prefix(while: { $0 == char }))
    }

    private static func find(units: [UTF16.CodeUnit], from: Int, pair: Int) -> Int? {
        var i = from
        while i + 1 < units.count {
            if units[i] == 0x2A && units[i + 1] == 0x2A { return i }
            i += 1
        }
        _ = pair
        return nil
    }

    private static func dedupeOutline(_ entries: [OutlineEntry]) -> [OutlineEntry] {
        var seen = Set<Int>()
        return entries.filter { seen.insert($0.line).inserted }
    }

    // MARK: - Outline patterns (ours)

    private static func isOutlineWord(_ word: String, atLine line: Int, offset: Int, units: [UTF16.CodeUnit], language: Language) -> Bool {
        // Find the line's leading whitespace and the token right after the
        // line start; only a keyword that denotes a definition opens an outline.
        let outlineWords: Set<String>
        switch language.id {
        case .swift:
            outlineWords = ["func", "class", "struct", "enum", "protocol", "extension", "init", "typealias", "actor"]
        case .rust:
            outlineWords = ["fn", "struct", "enum", "trait", "impl", "mod"]
        case .ts:
            outlineWords = ["function", "class", "interface", "enum"]
        case .python:
            outlineWords = ["def", "class"]
        case .go:
            outlineWords = ["func", "type", "struct", "interface"]
        case .java, .csharp:
            outlineWords = ["class", "interface", "enum", "record", "struct"]
        case .kotlin:
            outlineWords = ["fun", "class", "interface", "object"]
        case .ruby:
            outlineWords = ["def", "class", "module"]
        case .php:
            outlineWords = ["function", "class", "interface", "trait"]
        case .lua:
            outlineWords = ["function"]
        case .zig:
            outlineWords = ["fn", "struct", "enum", "union"]
        case .dart:
            outlineWords = ["class", "mixin", "enum", "extension"]
        case .scala:
            outlineWords = ["class", "object", "trait", "def"]
        case .haskell:
            outlineWords = ["data", "newtype", "type"]
        case .elixir:
            outlineWords = ["defmodule", "def", "defp", "defmacro"]
        default:
            outlineWords = []
        }
        guard outlineWords.contains(word) else { return false }

        // The definition keyword must be the first word on its line (after
        // whitespace / visibility-and-declaration modifiers).
        var lineBegin = offset
        while lineBegin > 0, units[lineBegin - 1] != 0x0A { lineBegin -= 1 }
        var j = lineBegin
        let modifiers: Set<String>
        switch language.id {
        case .ts:
            modifiers = ["export", "default", "declare", "abstract", "async"]
        case .rust:
            modifiers = ["pub", "async", "const", "unsafe", "extern"]
        case .python:
            modifiers = ["async"]
        case .java:
            modifiers = ["public", "private", "protected", "static", "final", "abstract", "sealed", "strictfp", "default"]
        case .kotlin:
            modifiers = ["public", "private", "internal", "protected", "open", "override", "final", "abstract", "sealed", "suspend", "inline", "expect", "actual", "external", "operator", "infix"]
        case .csharp:
            modifiers = ["public", "private", "protected", "internal", "static", "sealed", "abstract", "override", "virtual", "async", "partial", "extern", "unsafe", "new", "const"]
        case .dart:
            modifiers = ["abstract", "base", "final", "sealed", "static", "external", "covariant"]
        case .scala:
            modifiers = ["case", "final", "implicit", "lazy", "abstract", "sealed", "private", "protected", "override"]
        case .zig:
            modifiers = ["pub", "export", "extern", "inline", "threadlocal", "const"]
        case .swift:
            modifiers = ["public", "private", "internal", "fileprivate", "open", "static", "final", "async", "required", "convenience", "mutating", "override"]
        default:
            modifiers = []
        }
        let words: [String] = String(decoding: units[lineBegin..<min(offset, units.count)], as: UTF16.self)
            .split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
        for w in words {
            if w == word { break }
            guard modifiers.contains(w) else { return false }
        }
        while j < units.count, units[j] == 0x20 || units[j] == 0x09 { j += 1 }
        // confirm the keyword occupies this exact position
        let kw = Array(word.utf16)
        guard j + kw.count <= units.count else { return false }
        return Array(units[j..<j + kw.count]) == kw
    }

}
