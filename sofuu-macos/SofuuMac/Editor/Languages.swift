import Foundation

/// Language registry for the editor: extensions, comment syntax, indent
/// unit, keyword tables, and which language server can serve the file.
/// All original — this is our own editor, no borrowed code.
///
/// Coverage contract: EVERY entry colors, indents, comments, outlines and
/// finds regardless of what's installed; the LSP features (hover /
/// completion / diagnostics / definition) attach automatically the moment
/// the listed server binary exists on PATH, and the status line says
/// "no language server on PATH" honestly when not.
enum LangID: String, CaseIterable {
    case swift, rust, ts, python, json, shell, cLike, markdown, plain
    case go, java, kotlin, ruby, php, csharp, lua, zig, haskell, ocaml
    case elixir, dart, scala, html, css, yaml, toml, sql, dockerfile
    case perl, xml, r
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
        case .go: return "go"
        case .java: return "java"
        case .kotlin: return "kotlin"
        case .ruby: return "ruby"
        case .php: return "php"
        case .csharp: return "csharp"
        case .lua: return "lua"
        case .zig: return "zig"
        case .haskell: return "haskell"
        case .ocaml: return "ocaml"
        case .elixir: return "elixir"
        case .dart: return "dart"
        case .scala: return "scala"
        case .html: return "html"
        case .css: return "css"
        case .yaml: return "yaml"
        case .toml: return "toml"
        case .sql: return "sql"
        case .dockerfile: return "dockerfile"
        case .perl: return "perl"
        case .xml: return "xml"
        case .r: return "r"
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
        // Named, extension-less recipes: Dockerfile, Makefile, Rakefile.
        if name == "dockerfile" { return language(byID: .dockerfile) }
        // Extension-less dotfiles: .env, .gitignore-style → plain.
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
                "override", "final", "get", "set", "willSet", "didSet",
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
            id: .cLike, name: "C / C++", extensions: ["c", "h", "cpp", "hpp", "cc", "hh", "cxx", "hxx", "mm", "m"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "alignas", "alignof", "auto", "bool", "break", "case", "catch", "char",
                "class", "const", "constexpr", "continue", "default", "delete", "do",
                "double", "else", "enum", "explicit", "export", "extern", "false", "float",
                "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
                "namespace", "new", "noexcept", "nullptr", "operator", "private",
                "protected", "public", "return", "short", "signed", "sizeof", "static",
                "struct", "switch", "template", "this", "throw", "true", "try",
                "typedef", "typename", "union", "unsigned", "using", "virtual", "void",
                "while", "Objective-C", "id", "alloc", "init", "nil", "YES", "NO",
            ],
            server: LSPServerSpec(candidates: [("clangd", [])])),

        Language(
            id: .markdown, name: "Markdown", extensions: ["md", "markdown"],
            lineComment: nil, blockComment: nil, indentUnit: "    ",
            keywords: [], server: LSPServerSpec(candidates: [("marksman", [])])),

        Language(
            id: .go, name: "Go", extensions: ["go"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "\t",
            keywords: [
                "break", "case", "chan", "const", "continue", "default", "defer", "else",
                "fallthrough", "for", "func", "go", "goto", "if", "import", "interface",
                "map", "package", "range", "return", "select", "struct", "switch", "type",
                "var", "nil", "true", "false", "iota", "make", "new", "len", "cap",
                "append", "copy", "delete", "panic", "recover", "string", "int", "int64",
                "float64", "bool", "byte", "rune", "error", "any", "comparable",
            ],
            server: LSPServerSpec(candidates: [("gopls", [])])),

        Language(
            id: .java, name: "Java", extensions: ["java"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
                "class", "const", "continue", "default", "do", "double", "else", "enum",
                "extends", "final", "finally", "float", "for", "goto", "if", "implements",
                "import", "instanceof", "int", "interface", "long", "native", "new",
                "package", "private", "protected", "public", "return", "short", "static",
                "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
                "transient", "try", "void", "volatile", "while", "var", "record", "sealed",
                "permits", "true", "false", "null",
            ],
            server: LSPServerSpec(candidates: [
                ("jdtls", []),
                ("jdt-language-server", []),
            ])),

        Language(
            id: .kotlin, name: "Kotlin", extensions: ["kt", "kts"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "as", "break", "class", "continue", "do", "else", "false", "for", "fun",
                "if", "in", "interface", "is", "null", "object", "package", "return",
                "super", "this", "throw", "true", "try", "typealias", "val", "var",
                "when", "while", "by", "data", "sealed", "open", "override", "private",
                "public", "internal", "suspend", "const", "companion", "vararg",
                "lateinit", "init", "import",
            ],
            server: LSPServerSpec(candidates: [("kotlin-language-server", [])])),

        Language(
            id: .ruby, name: "Ruby", extensions: ["rb", "erb", "gemspec", "rake"],
            lineComment: "#", blockComment: nil, indentUnit: "  ",
            keywords: [
                "def", "end", "class", "module", "if", "elsif", "else", "unless", "while",
                "until", "for", "in", "do", "then", "case", "when", "begin", "rescue",
                "ensure", "raise", "return", "yield", "require", "require_relative",
                "include", "attr_accessor", "attr_reader", "attr_writer", "new", "nil",
                "true", "false", "self", "super", "puts", "lambda", "proc", "and", "or",
                "not", "alias", "private", "protected", "public",
            ],
            server: LSPServerSpec(candidates: [
                ("ruby-lsp", []),
                ("solargraph", ["stdio"]),
            ])),

        Language(
            id: .php, name: "PHP", extensions: ["php", "phtml"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "abstract", "and", "array", "as", "break", "callable", "case", "catch",
                "class", "clone", "const", "continue", "declare", "default", "do", "echo",
                "else", "elseif", "extends", "final", "finally", "fn", "for", "foreach",
                "function", "global", "if", "implements", "include", "instanceof",
                "interface", "isset", "list", "namespace", "new", "or", "print",
                "private", "protected", "public", "readonly", "require", "return",
                "static", "switch", "throw", "trait", "try", "unset", "use", "var",
                "while", "xor", "yield", "true", "false", "null",
            ],
            server: LSPServerSpec(candidates: [("intelephense", ["--stdio"])])),

        Language(
            id: .csharp, name: "C#", extensions: ["cs", "csx"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "abstract", "as", "async", "await", "base", "bool", "break", "byte",
                "case", "catch", "char", "checked", "class", "const", "continue",
                "decimal", "default", "delegate", "do", "double", "else", "enum",
                "event", "explicit", "extern", "false", "finally", "fixed", "float",
                "for", "foreach", "get", "goto", "if", "implicit", "in", "int",
                "interface", "internal", "is", "lock", "long", "namespace", "new",
                "null", "object", "operator", "out", "override", "params", "private",
                "protected", "public", "readonly", "ref", "return", "sbyte", "sealed",
                "set", "short", "sizeof", "stackalloc", "static", "string", "struct",
                "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong",
                "unchecked", "unsafe", "ushort", "using", "var", "virtual", "void",
                "volatile", "while", "record", "init", "required", "global",
            ],
            server: LSPServerSpec(candidates: [
                ("csharp-ls", []),
                ("OmniSharp", ["-lsp"]),
            ])),

        Language(
            id: .lua, name: "Lua", extensions: ["lua"],
            lineComment: "--", blockComment: nil, indentUnit: "  ",
            keywords: [
                "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
                "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
                "then", "true", "until", "while", "self", "require", "pairs", "ipairs",
                "print", "type", "assert", "pcall", "tostring", "tonumber",
                "setmetatable", "getmetatable",
            ],
            server: LSPServerSpec(candidates: [("lua-language-server", [])])),

        Language(
            id: .zig, name: "Zig", extensions: ["zig"],
            lineComment: "//", blockComment: nil, indentUnit: "    ",
            keywords: [
                "align", "allowzero", "anyframe", "anytype", "asm", "break", "catch",
                "comptime", "const", "continue", "defer", "else", "enum", "errdefer",
                "error", "export", "extern", "false", "fn", "for", "if", "inline",
                "noalias", "null", "orelse", "packed", "pub", "resume", "return",
                "linksection", "struct", "suspend", "switch", "test", "threadlocal",
                "true", "undefined", "union", "unreachable", "usingnamespace", "var",
                "volatile", "while", "try", "async", "await",
            ],
            server: LSPServerSpec(candidates: [("zls", [])])),

        Language(
            id: .haskell, name: "Haskell", extensions: ["hs", "lhs"],
            lineComment: "--", blockComment: ("{-", "-}"), indentUnit: "    ",
            keywords: [
                "case", "class", "data", "default", "deriving", "do", "else", "foreign",
                "if", "import", "in", "infix", "infixl", "infixr", "instance", "let",
                "module", "newtype", "of", "then", "type", "where", "True", "False",
                "Nothing", "Just",
            ],
            server: LSPServerSpec(candidates: [
                ("haskell-language-server-wrapper", ["--lsp"]),
            ])),

        Language(
            id: .ocaml, name: "OCaml", extensions: ["ml", "mli"],
            lineComment: nil, blockComment: ("(*", "*)"), indentUnit: "  ",
            keywords: [
                "and", "as", "assert", "begin", "class", "constraint", "do", "done",
                "downto", "else", "end", "exception", "external", "false", "for", "fun",
                "function", "functor", "if", "in", "include", "inherit", "initializer",
                "lazy", "let", "match", "method", "mod", "module", "mutable", "new",
                "object", "of", "open", "private", "rec", "sig", "struct", "then", "to",
                "true", "try", "type", "val", "virtual", "when", "while", "with",
            ],
            server: LSPServerSpec(candidates: [("ocamllsp", [])])),

        Language(
            id: .elixir, name: "Elixir", extensions: ["ex", "exs"],
            lineComment: "#", blockComment: nil, indentUnit: "  ",
            keywords: [
                "def", "defmodule", "defp", "defmacro", "defmacrop", "defstruct",
                "defprotocol", "defimpl", "defexception", "do", "else", "end", "false",
                "fn", "if", "import", "nil", "require", "true", "case", "cond", "for",
                "unless", "use", "alias", "quote", "with", "rescue", "raise", "try",
                "when", "and", "or", "not", "in",
            ],
            server: LSPServerSpec(candidates: [
                ("elixir-ls", []),
                ("language_server.sh", []),
            ])),

        Language(
            id: .dart, name: "Dart", extensions: ["dart"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "  ",
            keywords: [
                "abstract", "as", "assert", "async", "await", "base", "break", "case",
                "catch", "class", "const", "continue", "covariant", "default",
                "deferred", "do", "dynamic", "else", "enum", "export", "extends",
                "extension", "external", "factory", "false", "final", "finally", "for",
                "get", "hide", "if", "implements", "import", "in", "interface", "is",
                "late", "library", "mixin", "new", "null", "on", "operator", "part",
                "required", "rethrow", "return", "sealed", "set", "show", "static",
                "super", "switch", "sync", "this", "throw", "true", "try", "typedef",
                "var", "when", "while", "with", "yield",
            ],
            server: LSPServerSpec(candidates: [("dart", ["language-server"])])),

        Language(
            id: .scala, name: "Scala", extensions: ["scala", "sc"],
            lineComment: "//", blockComment: ("/*", "*/"), indentUnit: "  ",
            keywords: [
                "abstract", "case", "catch", "class", "def", "do", "else", "enum",
                "extends", "false", "final", "finally", "for", "if", "implicit",
                "import", "lazy", "match", "new", "null", "object", "override",
                "package", "private", "protected", "return", "sealed", "super", "this",
                "throw", "trait", "true", "try", "type", "val", "var", "while", "with",
                "yield", "given", "using", "extension", "end",
            ],
            server: LSPServerSpec(candidates: [("metals", [])])),

        Language(
            id: .html, name: "HTML", extensions: ["html", "htm", "xhtml", "vue", "svelte"],
            lineComment: nil, blockComment: ("<!--", "-->"), indentUnit: "  ",
            keywords: [
                "html", "head", "body", "div", "span", "p", "a", "href", "img", "ul",
                "ol", "li", "table", "tr", "td", "th", "script", "style", "link",
                "meta", "title", "form", "input", "button", "section", "header",
                "footer", "nav", "class", "template", "slot",
            ],
            server: LSPServerSpec(candidates: [("vscode-html-language-server", ["--stdio"])])),

        Language(
            id: .css, name: "CSS", extensions: ["css", "scss", "sass", "less"],
            lineComment: nil, blockComment: ("/*", "*/"), indentUnit: "  ",
            keywords: [
                "color", "background", "font", "font-size", "font-weight", "margin",
                "padding", "border", "display", "position", "float", "width", "height",
                "flex", "grid", "important", "inherit", "none", "auto", "absolute",
                "relative", "fixed", "center", "block", "inline",
            ],
            server: LSPServerSpec(candidates: [("vscode-css-language-server", ["--stdio"])])),

        Language(
            id: .yaml, name: "YAML", extensions: ["yaml", "yml"],
            lineComment: "#", blockComment: nil, indentUnit: "  ",
            keywords: ["true", "false", "null", "yes", "no", "on", "off"],
            server: LSPServerSpec(candidates: [("yaml-language-server", ["--stdio"])])),

        Language(
            id: .toml, name: "TOML", extensions: ["toml"],
            lineComment: "#", blockComment: nil, indentUnit: "  ",
            keywords: ["true", "false", "inf", "nan"],
            server: LSPServerSpec(candidates: [("taplo", ["lsp", "stdio"])])),

        Language(
            id: .sql, name: "SQL", extensions: ["sql"],
            lineComment: "--", blockComment: ("/*", "*/"), indentUnit: "    ",
            keywords: [
                "select", "from", "where", "insert", "into", "values", "update", "set",
                "delete", "create", "table", "drop", "alter", "add", "index", "view",
                "join", "inner", "left", "right", "outer", "full", "on", "group", "by",
                "order", "having", "limit", "offset", "as", "and", "or", "not", "null",
                "is", "like", "in", "between", "union", "all", "distinct", "primary",
                "key", "foreign", "references", "default", "unique", "check",
                "cascade", "begin", "commit", "rollback", "transaction", "exists",
                "case", "when", "then", "else", "end",
            ],
            server: LSPServerSpec(candidates: [
                ("sqls", []),
                ("sql-language-server", ["up", "--method", "stdio"]),
            ])),

        Language(
            id: .dockerfile, name: "Dockerfile", extensions: ["dockerfile"],
            lineComment: "#", blockComment: nil, indentUnit: "    ",
            keywords: [
                "from", "run", "cmd", "copy", "add", "entrypoint", "env", "arg",
                "volume", "expose", "label", "user", "workdir", "onbuild",
                "stopsignal", "shell", "healthcheck", "as",
            ],
            server: LSPServerSpec(candidates: [
                ("docker-langserver", ["--stdio"]),
                ("dockerfile-language-server-nodejs", ["--stdio"]),
            ])),

        Language(
            id: .perl, name: "Perl", extensions: ["pl", "pm", "t"],
            lineComment: "#", blockComment: nil, indentUnit: "    ",
            keywords: [
                "my", "our", "local", "sub", "if", "elsif", "else", "unless", "while",
                "until", "for", "foreach", "do", "last", "next", "redo", "return",
                "wantarray", "use", "no", "require", "package", "scalar", "hash",
                "array", "print", "printf", "sprintf", "chomp", "chop", "shift",
                "unshift", "push", "pop", "keys", "values", "each", "delete", "exists",
                "defined", "bless", "ref", "uc", "lc", "length", "index", "substr",
                "join", "split", "grep", "map", "sort", "reverse",
            ],
            server: nil),

        Language(
            id: .xml, name: "XML", extensions: ["xml", "plist", "svg", "xib", "storyboard"],
            lineComment: nil, blockComment: ("<!--", "-->"), indentUnit: "  ",
            keywords: [],
            server: LSPServerSpec(candidates: [("lemminx", [])])),

        Language(
            id: .r, name: "R", extensions: ["r"],
            lineComment: "#", blockComment: nil, indentUnit: "    ",
            keywords: [
                "if", "else", "repeat", "while", "function", "for", "in", "next",
                "break", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "library",
                "require", "source", "return",
            ],
            server: nil),
    ]

    /// Servers worth probing on PATH — used by the status line and the
    /// registry's debug output.
    static var installedServers: [String: Bool] = [:]
}
