import Foundation

enum GitStatusKind: Equatable {
    case changed      // M / R / C — differs from HEAD
    case untracked    // ?? — new to git
    case deleted      // D — in HEAD, gone from the tree
}

extension GitStatusKind {
    var glyph: String {
        switch self {
        case .changed: return "M"
        case .untracked: return "+"
        case .deleted: return "D"
        }
    }
}

/// One hunk of a unified diff: its `@@` header plus the body lines
/// (leading ' ', '+' or '-'). `index` is its position in the file's diff,
/// stable for the lifetime of that diff.
struct GitHunk: Identifiable, Equatable {
    let index: Int
    let header: String
    let lines: [String]
    var id: Int { index }
    var added: Int { lines.filter { $0.hasPrefix("+") }.count }
    var removed: Int { lines.filter { $0.hasPrefix("-") }.count }
}

/// A single-file unified diff split into the file header (everything before
/// the first `@@`) and its hunks.
struct GitDiff: Equatable {
    var fileHeader: [String]
    var hunks: [GitHunk]
}

/// Read-only git working-tree status for the tree's badges. `git status
/// --porcelain` (v1) parsed off the main thread. Rename lines are
/// `R  old -> new` (verified empirically) — the CURRENT path is the one
/// after the arrow.
enum GitMonitor {
    static func status(root: String, completion: @escaping ([String: GitStatusKind]) -> Void) {
        DispatchQueue.global(qos: .utility).async {
            let out = run(["-C", root, "status", "--porcelain"])
            var map: [String: GitStatusKind] = [:]
            for rawLine in out.split(separator: "\n") {
                let line = String(rawLine)
                guard line.count >= 4 else { continue }
                let xy = String(line.prefix(2))
                var path = String(line.dropFirst(3))
                if let arrow = path.range(of: " -> ") {
                    path = String(path[arrow.upperBound...])
                }
                guard !path.isEmpty else { continue }
                let full = root + "/" + path
                if xy == "??" {
                    map[full] = .untracked
                } else if xy.contains("D") {
                    map[full] = .deleted
                } else {
                    map[full] = .changed
                }
            }
            DispatchQueue.main.async { completion(map) }
        }
    }

    /// HEAD blob for a path ("" when the file is untracked or absent).
    static func headText(root: String, relativePath: String) -> String {
        let out = run(["-C", root, "show", "HEAD:\(relativePath)"])
        return out.isEmpty ? "" : String(out.dropLastIfNewline)
    }

    /// Column-split status (staged vs unstaged) + branch name for the
    /// commit panel. Empty branch = not a git repository (or bare).
    static func detailedStatus(root: String, completion: @escaping ([String: GitFileState], String) -> Void) {
        DispatchQueue.global(qos: .utility).async {
            let out = run(["-C", root, "status", "--porcelain"])
            var files: [String: GitFileState] = [:]
            for rawLine in out.split(separator: "\n") {
                let line = String(rawLine)
                guard line.count >= 4 else { continue }
                let xy = Array(line.prefix(2))
                var path = String(line.dropFirst(3))
                if let arrow = path.range(of: " -> ") {
                    path = String(path[arrow.upperBound...])
                }
                guard !path.isEmpty else { continue }
                func kind(_ c: Character) -> GitStatusKind? {
                    if c == "?" { return .untracked }
                    if c == "D" { return .deleted }
                    if c == " " { return nil }
                    return .changed
                }
                let staged = kind(xy[0]) == .untracked ? nil : kind(xy[0])
                let unstaged = kind(xy[1])
                if staged == nil && unstaged == nil { continue }
                files[root + "/" + path] = GitFileState(relative: path, staged: staged, unstaged: unstaged)
            }
            let branch = run(["-C", root, "branch", "--show-current"])
                .split(separator: "\n").first.map(String.init) ?? ""
            DispatchQueue.main.async { completion(files, branch) }
        }
    }

    /// Mutating git for the commit panel. Returns (ok, stderr-ish) so the
    /// store can surface a real failure instead of guessing.
    @discardableResult
    static func mutate(_ root: String, _ args: [String]) -> (ok: Bool, output: String) {
        mutate(root, args, stdin: nil)
    }

    /// `mutate` with a patch on stdin — what hunk staging needs.
    @discardableResult
    static func mutate(_ root: String, _ args: [String], stdin: String?) -> (ok: Bool, output: String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = ["-C", root] + args
        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        do {
            if let stdin {
                let inPipe = Pipe()
                proc.standardInput = inPipe
                try proc.run()
                inPipe.fileHandleForWriting.write(stdin.data(using: .utf8) ?? Data())
                try? inPipe.fileHandleForWriting.close()
            } else {
                try proc.run()
            }
            let data = outPipe.fileHandleForReading.readDataToEndOfFile()
            let err = String(decoding: errPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            proc.waitUntilExit()
            let ok = proc.terminationStatus == 0
            let outText = String(decoding: data, as: UTF8.self)
            return (ok, ok ? outText : err.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch {
            return (false, "\(error)")
        }
    }

    // ── Hunks (stage/unstage a single hunk) ─────────────────────────

    /// Unstaged diff (worktree vs index) for one path. Empty for an
    /// untracked file — it has no index entry to diff against.
    static func worktreeDiff(root: String, relativePath: String) -> String? {
        diff(root: root, relativePath: relativePath, cached: false)
    }

    /// Staged diff (index vs HEAD) for one path.
    static func stagedDiff(root: String, relativePath: String) -> String? {
        diff(root: root, relativePath: relativePath, cached: true)
    }

    private static func diff(root: String, relativePath: String, cached: Bool) -> String? {
        var args = ["-C", root, "diff", "--no-color", "--no-ext-diff", "--unified=3"]
        if cached { args.append("--cached") }
        args += ["--", relativePath]
        let out = run(args)
        return out.isEmpty ? nil : out
    }

    /// Split a unified diff into its file header and hunks. Everything
    /// before the first `@@` is the header the patch needs; each hunk runs
    /// until the next `@@` (or EOF). Body lines keep their prefix so the
    /// patch can be reconstructed byte-for-byte.
    static func parseDiff(_ text: String) -> GitDiff {
        var fileHeader: [String] = []
        var hunks: [GitHunk] = []
        var body: [String] = []
        var header: String?

        func flush() {
            guard let header else { return }
            hunks.append(GitHunk(index: hunks.count, header: header, lines: body))
            body = []
        }

        var lines = text.components(separatedBy: "\n")
        // A trailing newline produces one empty component; drop exactly it.
        if lines.last == "" { lines.removeLast() }
        for line in lines {
            if line.hasPrefix("@@") {
                flush()
                header = line
            } else if header == nil {
                fileHeader.append(line)
            } else if line.hasPrefix("diff --git ") {
                // Belt and braces: one path was requested, but never fold a
                // second file's header into the previous hunk.
                flush()
                header = nil
                fileHeader.append(line)
            } else {
                body.append(line)
            }
        }
        flush()
        return GitDiff(fileHeader: fileHeader, hunks: hunks)
    }

    /// Reconstruct a one-hunk patch: the file header plus the selected
    /// hunk. Nil when there is no header to apply against.
    static func hunkPatch(_ diff: GitDiff, _ hunk: GitHunk) -> String? {
        guard !diff.fileHeader.isEmpty else { return nil }
        var out = diff.fileHeader
        out.append(hunk.header)
        out.append(contentsOf: hunk.lines)
        return out.joined(separator: "\n") + "\n"
    }

    /// Apply a patch to the index only. Reverse=true takes a staged diff
    /// and pulls that hunk back out of the index (hunk unstage); the
    /// worktree is never touched.
    @discardableResult
    static func applyPatch(root: String, patch: String, reverse: Bool) -> (ok: Bool, output: String) {
        var args = ["apply", "--cached", "--recount"]
        if reverse { args.append("--reverse") }
        args.append("-")
        return mutate(root, args, stdin: patch)
    }

    private static func run(_ args: [String]) -> String {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else { return "" }
            return String(decoding: data, as: UTF8.self)
        } catch {
            return ""
        }
    }
}

private extension String {
    var dropLastIfNewline: Substring {
        hasSuffix("\n") ? dropLast() : self[...]
    }
}
