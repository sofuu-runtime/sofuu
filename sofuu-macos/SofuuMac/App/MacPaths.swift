import Foundation

/// Mac filesystem anchors — the desktop counterpart of the iOS AppPaths.
///
/// The engine gets the REAL `$HOME` (no `setenv("HOME")` restamping like
/// iOS does): `~/.sofuu` is the CLI's shared directory, so config, brain
/// and sessions line up with the terminal on the same machine — the same
/// identity the Tauri desktop app used (`config_root: None`).
///
/// App state that is desktop-only lives in `~/.sofuu/desktop.json`
/// (`config.rs`'s file) so the CLI's config writes can never drop it.
enum MacPaths {

    static var home: String { NSHomeDirectory() }

    static var sofuuDir: URL {
        URL(fileURLWithPath: home, isDirectory: true).appendingPathComponent(".sofuu", isDirectory: true)
    }

    static var configFile: URL { sofuuDir.appendingPathComponent("config.json") }
    static var desktopFile: URL { sofuuDir.appendingPathComponent("desktop.json") }

    /// First-launch workspace. The desktop's own fallback for a missing
    /// project_dir was `$HOME` (commands.rs:20-23) — same default here, so
    /// a fresh Mac app sees what the CLI sees.
    static var defaultWorkspace: String { home }

    /// The session-mesh root of a project — must match what chat.js /
    /// session.rs derive (desktop sessions.rs:60-75): the git toplevel when
    /// the folder lives inside a repo, else the folder itself. Session-file
    /// cleanup at the wrong root would strand QTSQ bytes on disk.
    static func meshRoot(_ project: String) -> String {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        task.arguments = ["rev-parse", "--show-toplevel"]
        task.currentDirectoryURL = URL(fileURLWithPath: project, isDirectory: true)
        let out = Pipe()
        task.standardOutput = out
        task.standardError = Pipe()
        do {
            try task.run()
            let data = out.fileHandleForReading.readDataToEndOfFile()
            task.waitUntilExit()
            if task.terminationStatus == 0,
               let s = String(data: data, encoding: .utf8)?
                   .trimmingCharacters(in: .whitespacesAndNewlines),
               !s.isEmpty {
                return s
            }
        } catch {}
        return project
    }

    /// `<mesh root>/.sofuu/sessions` — where the engine keeps the registry
    /// and the per-session QTSQ files.
    static func sessionsDir(_ project: String) -> URL {
        URL(fileURLWithPath: meshRoot(project), isDirectory: true)
            .appendingPathComponent(".sofuu", isDirectory: true)
            .appendingPathComponent("sessions", isDirectory: true)
    }
}
