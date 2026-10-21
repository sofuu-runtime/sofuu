import Foundation

/// Language registry for the editor: extensions, comment syntax, indent
/// unit, keyword tables, and which language server can serve the file.
/// All original — this is our own editor, no borrowed code.
enum LangID: String, CaseIterable {
    case swift, rust, ts, python, json, shell, cLike, markdown, plain
}

struct LSPServerSpec {
    /// Search order: first binary found on PATH wins.
    let candidates: [(bin: String, args: [String])]
}

struct Language {
    let id: LangID
    let name: String
    let extensions: [String]
    let lineComment: String?
    let blockComment: (String, String)?
    let indentUnit: String
    let keywords: Set<String>
    let server: LSPServerSpec?

    /// The ⌘/ prefix for a line (with trailing space where the language
    /// convention wants it).
    var commentPrefix: String? {
        guard let lineComment else { return nil }
        return lineComment + " "
    }

    /// The Language Server Protocol `languageId` string didOpen must carry
    /// (our LangID rawValues are internal shorthands, not LSP ids).
    var lspLanguageID: String {
        switch id {
        case .swift: return "swift"
        case .rust: return "rust"
        case .ts: return "typescript"
        case .python: return "python"
        case .json: return "json"
        case .shell: return "shellscript"
        case .cLike: return "c"
        case .markdown: return "markdown"
        case .plain: return "plaintext"
        }
    }
}

enum Languages {
    static func language(forPath path: String) -> Language {
        let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
        let name = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        for lang in all where lang.extensions.contains(ext) {
            return lang
        }
        // Extension-less dotfiles: .env, .gitignore-style → plain.
        if name.hasPrefix(".") { return plain }
        return plain
    }

    static func language(byID id: LangID) -> Language {
        all.first { $0.id == id } ?? plain
    }

    static let plain = Language(
        id: .plain, name: "Plain Text", extensions: ["txt", "log", "env", "example", "conf", "cfg", "ini", "lock"],
        lineComment: nil, blockComment: nil, indentUnit: "    ", keywords: [], server: nil)

    static let all: [Language] = [
        Language(
            id: .swift, name: "Swift", extensions: ["swift"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "associatedtype", "class", "deinit", "enum", "extension", "fileprivate", "func",
                "import", "init", "inout", "internal", "let", "open", "operator", "private",
                "precedencegroup", "protocol", "public", "rethrows", "static", "struct",
                "subscript", "typealias", "var", "break", "case", "catch", "continue",
                "default", "defer", "do", "else", "fallthrough", "for", "guard", "if", "in",
                "repeat", "return", "throw", "switch", "where", "while", "as", "any", "false",
                "is", "nil", "self", "Self", "super", "throws", "true", "try", "await",
                "async", "some", "each", "mutating", "nonmutating", "lazy", "weak",
                "unowned", "infix", "postfix", "prefix", "indirect", "required", "convenience",
                "override", "final", "get", "set", "willSet", "didSet", "if let",
            ],
            server: LSPServerSpec(candidates: [("sourcekit-lsp", [])])),

        Language(
            id: .rust, name: "Rust", extensions: ["rs"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop",
                "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self",
                "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
                "where", "while",
            ],
            server: LSPServerSpec(candidates: [("rust-analyzer", [])])),

        Language(
            id: .ts, name: "TypeScript / JavaScript", extensions: ["ts", "tsx", "js", "jsx", "mjs", "cjs"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "  ",
            keywords: [
                "abstract", "any", "as", "asserts", "async", "await", "boolean", "break",
                "case", "catch", "class", "const", "continue", "debugger", "declare",
                "default", "delete", "do", "else", "enum", "export", "extends", "false",
                "finally", "for", "from", "function", "get", "if", "implements", "import",
                "in", "infer", "instanceof", "interface", "is", "keyof", "let", "namespace",
                "never", "new", "null", "number", "object", "of", "private", "protected",
                "public", "readonly", "return", "satisfies", "set", "static", "string",
                "super", "switch", "symbol", "this", "throw", "true", "try", "type",
                "typeof", "undefined", "unique", "unknown", "var", "void", "while", "with", "yield",
            ],
            server: LSPServerSpec(candidates: [
                ("typescript-language-server", ["--stdio"]),
                ("deno", ["lsp"]),
            ])),

        Language(
            id: .python, name: "Python", extensions: ["py", "pyi"],
            lineComment: "#", blockComment: nil, indentUnit: "    ",
            keywords: [
                "and", "as", "assert", "async", "await", "break", "class", "continue",
                "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
                "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or",
                "pass", "raise", "return", "True", "False", "try", "while", "with", "yield",
                "match", "case", "self",
            ],
            server: LSPServerSpec(candidates: [
                ("pyright-langserver", ["--stdio"]),
                ("pylsp", []),
                ("jedi-language-server", []),
            ])),

        Language(
            id: .json, name: "JSON", extensions: ["json", "jsonc", "json5"],
            lineComment: nil, blockComment: nil, indentUnit: "  ",
            keywords: ["true", "false", "null"],
            server: LSPServerSpec(candidates: [
                ("vscode-json-language-server", ["--stdio"]),
            ])),

        Language(
            id: .shell, name: "Shell", extensions: ["sh", "bash", "zsh", "fish", "command"],
            lineComment: "#", blockComment: nil, indentUnit: "    ",
            keywords: [
                "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done",
                "case", "esac", "in", "function", "return", "local", "export", "echo",
                "cd", "set", "source", "exit", "read", "shift", "trap", "eval",
            ],
            server: LSPServerSpec(candidates: [("bash-language-server", ["start"])])),

        Language(
            id: .cLike, name: "C / C++", extensions: ["c", "h", "cpp", "hpp", "cc", "hh", "cxx", "mm", "m"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "alignas", "alignof", "auto", "bool", "break", "case", "catch", "char",
                "class", "const", "constexpr", "continue", "default", "delete", "do",
                "double", "else", "enum", "explicit", "export", "extern", "false", "float",
                "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
                "namespace", "new", " noexcept", "nullptr", "operator", "private",
                "protected", "public", "return", "short", "signed", "sizeof", "static",
                "struct", "switch", "template", "this", "throw", "true", "try",
                "typedef", "typename", "union", "unsigned", "using", "virtual", "void",
                "while",
            ],
            server: LSPServerSpec(candidates: [("clangd", [])])),

        Language(
            id: .markdown, name: "Markdown", extensions: ["md", "markdown"],
            lineComment: nil, blockComment: nil, indentUnit: "    ",
            keywords: [], server: nil),
    ]

    /// Servers worth probing on PATH — used by the status line and the
    /// registry's debug output.
    static var installedServers: [String: Bool] = [:]
}
