import SwiftUI
import AppKit

/// Lightweight syntax highlighting for the IDE code viewport — a single
/// alternation pass (comments | strings | numbers | attributes | words) so
/// token classes never overlap, then per-word classification against
/// per-language keyword tables. Not a parser: enough color to read code
/// comfortably, fast enough for files up to the 1 MB editor cap on a
/// background queue. Colors are NSColor because they land in NSTextStorage.
enum Syntax {

    // One-Dark hues in the dark theme, slightly deepened for light.
    enum TokenClass {
        case comment, string, number, keyword, type, decorator

        func nsColor(dark: Bool) -> NSColor {
            switch self {
            case .comment: return Syntax.nsColor(dark ? 0x7F848E : 0x9DA0A6)
            case .string: return Syntax.nsColor(dark ? 0x98C379 : 0x50A14F)
            case .number: return Syntax.nsColor(dark ? 0xD19A66 : 0x986801)
            case .keyword: return Syntax.nsColor(dark ? 0xC678DD : 0xA626A4)
            case .type: return Syntax.nsColor(dark ? 0xE5C07B : 0xC18401)
            case .decorator: return Syntax.nsColor(dark ? 0x56B6C2 : 0x0184BC)
            }
        }
    }

    /// Body text color for code (One-Dark foreground).
    static func nsPlain(dark: Bool) -> NSColor {
        nsColor(dark ? 0xABB2BF : 0x383A42)
    }

    private static func nsColor(_ hex: UInt32) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255.0,
                green: CGFloat((hex >> 8) & 0xFF) / 255.0,
                blue: CGFloat(hex & 0xFF) / 255.0,
                alpha: 1)
    }

    enum Lang {
        case cLike      // c, cpp, h, m, mm
        case swift
        case python
        case javaScript // js, ts, jsx, tsx
        case rust
        case json
        case shell
        case plain      // md, txt, unknown

        static func guess(_ path: String) -> Lang {
            switch URL(fileURLWithPath: path).pathExtension.lowercased() {
            case "swift": return .swift
            case "py", "pyi": return .python
            case "js", "ts", "jsx", "tsx", "mjs": return .javaScript
            case "rs": return .rust
            case "json": return .json
            case "sh", "bash", "zsh", "fish", "command": return .shell
            case "c", "h", "cpp", "hpp", "cc", "hh", "m", "mm": return .cLike
            default: return .plain
            }
        }
    }

    private static let keywords: [Lang: Set<String>] = [
        .swift: ["import", "struct", "class", "enum", "func", "let", "var", "if", "else",
                 "guard", "return", "switch", "case", "default", "for", "while", "in",
                 "extension", "protocol", "init", "self", "nil", "true", "false", "try",
                 "catch", "throw", "throws", "async", "await", "static", "private",
                 "public", "internal", "fileprivate", "where", "get", "set", "didSet",
                 "willSet", "defer", "lazy", "some", "any", "as", "is"],
        .python: ["def", "class", "import", "from", "return", "if", "elif", "else", "for",
                  "while", "in", "not", "and", "or", "None", "True", "False", "try",
                  "except", "finally", "raise", "with", "as", "lambda", "yield", "pass",
                  "async", "await", "global", "nonlocal", "self", "assert", "del"],
        .javaScript: ["const", "let", "var", "function", "return", "if", "else", "for",
                      "while", "of", "in", "class", "extends", "new", "this", "null",
                      "undefined", "true", "false", "try", "catch", "finally", "throw",
                      "async", "await", "import", "export", "from", "default", "switch",
                      "case", "break", "continue", "typeof", "instanceof"],
        .rust: ["fn", "let", "mut", "struct", "enum", "impl", "trait", "pub", "use",
                "mod", "match", "if", "else", "for", "while", "loop", "in", "return",
                "self", "Self", "true", "false", "async", "await", "move", "where",
                "dyn", "unsafe", "const", "static", "crate", "ref", "as"],
        .cLike: ["int", "char", "float", "double", "void", "return", "if", "else",
                 "for", "while", "switch", "case", "default", "struct", "class",
                 "public", "private", "protected", "template", "typename", "const",
                 "static", "true", "false", "nullptr", "NULL", "auto", "namespace",
                 "using", "new", "delete", "try", "catch", "throw", "sizeof"],
        .json: ["true", "false", "null"],
        .shell: ["if", "then", "elif", "else", "fi", "for", "while", "do", "done",
                 "case", "esac", "in", "function", "return", "local", "export",
                 "echo", "cd", "set", "source", "exit"],
        .plain: [],
    ]

    /// One compiled alternation per language, with each wrapped alternative
    /// mapped to its token class through its capture-group index. The word
    /// alternative has class `nil` — classified by substring (keyword set
    /// or capitalized → type).
    private static let compiled: [Lang: (regex: NSRegularExpression, classByGroup: [Int: TokenClass])] = {
        var out: [Lang: (NSRegularExpression, [Int: TokenClass])] = [:]
        for lang in [Lang.cLike, .swift, .python, .javaScript, .rust, .json, .shell, .plain] {
            var parts: [(pattern: String, cls: TokenClass?)] = []
            switch lang {
            case .python, .shell:
                parts.append(("#[^\\n]*", .comment))
            case .json, .plain:
                break
            default:
                parts.append(("//[^\\n]*", .comment))
                parts.append(("/\\*[\\s\\S]*?(?:\\*/|$)", .comment))
            }
            if lang == .python {
                parts.append(("\"\"\"[\\s\\S]*?(?:\"\"\"|$)", .string))
                parts.append(("'''[\\s\\S]*?(?:'''|$)", .string))
            }
            parts.append(("\\\"(?:[^\\\"\\\\\\n]|\\\\.)*\\\"?", .string))
            parts.append(("'(?:[^'\\\\\\n]|\\\\.)*'?", .string))
            if lang == .rust {
                parts.append(("r#*\\\"[\\s\\S]*?\\\"#*", .string))
            }
            parts.append(("\\b(?:0[xX][0-9a-fA-F_]+|\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?\\d+)?)\\b", .number))
            if lang == .python || lang == .swift || lang == .rust {
                parts.append(("@[A-Za-z_][A-Za-z0-9_]*", .decorator))
            }
            parts.append(("\\b[A-Za-z_][A-Za-z0-9_]*\\b", nil))

            var classByGroup: [Int: TokenClass] = [:]
            var joined: [String] = []
            var group = 1
            for (pattern, cls) in parts {
                joined.append("(\(pattern))")
                if let cls { classByGroup[group] = cls }
                group += 1
            }
            // group 0 is the whole match; classes live at 1...
            let regex = try? NSRegularExpression(pattern: joined.joined(separator: "|"), options: [])
            if let regex { out[lang] = (regex, classByGroup) }
        }
        return out
    }()

    /// Ranges-per-class for `text`. Callers map each range to its palette
    /// color. Empty result for plain languages (no color pass needed).
    static func colorRanges(_ text: String, lang: Lang) -> [(range: NSRange, cls: Syntax.TokenClass)] {
        guard let (regex, classByGroup) = compiled[lang], text.count < 300_000 else { return [] }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        let keywords = keywords[lang] ?? []
        var out: [(NSRange, Syntax.TokenClass)] = []
        out.reserveCapacity(matches.count)
        for m in matches {
            guard m.range.location != NSNotFound else { continue }
            var cls: Syntax.TokenClass?
            for (idx, c) in classByGroup where m.range(at: idx).location != NSNotFound {
                cls = c
                break
            }
            // The word group: classify by substring.
            if cls == nil {
                for g in 1...m.numberOfRanges - 1 where m.range(at: g).location != NSNotFound {
                    if classByGroup[g] == nil {
                        let word = ns.substring(with: m.range(at: g))
                        if keywords.contains(word) { cls = .keyword }
                        else if word.first?.isUppercase == true { cls = .type }
                        break
                    }
                }
            }
            if let cls { out.append((m.range, cls)) }
        }
        return out
    }
}
