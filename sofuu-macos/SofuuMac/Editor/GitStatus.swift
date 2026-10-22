import Foundation

enum GitStatusKind {
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
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = ["-C", root] + args
        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        do {
            try proc.run()
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
