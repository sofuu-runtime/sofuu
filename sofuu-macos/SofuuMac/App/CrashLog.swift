import AppKit
import Foundation

/// Crash + error reporting to stderr — the `frontend_log` command's twin
/// (errlog.ts + ErrorBoundary.tsx): a release app has no console, so the
/// JS-side errors and uncaught Swift exceptions land in stderr where
/// `Console.app` / a terminal launch can see them.
enum CrashLog {

    private static func log(_ line: String) {
        // The ring receives the line too — Settings ▸ Console is the
        // in-app surface for engine + app errors.
        EngineConsole.ring.append("error: \(line)")
        FileHandle.standardError.write(Data("[sofuu-mac] error: \(line)\n".utf8))
    }

    static func install() {
        EngineConsole.install()
        NSSetUncaughtExceptionHandler { ex in
            let text = "uncaught exception: \(ex.name.rawValue): \(ex.reason ?? "-")\n"
                + ex.callStackSymbols.joined(separator: "\n")
            EngineConsole.ring.append(text)
            FileHandle.standardError.write(
                Data("[sofuu-mac] \(text)\n".utf8))
        }
    }

    static func error(_ message: String) { log(message) }
}
