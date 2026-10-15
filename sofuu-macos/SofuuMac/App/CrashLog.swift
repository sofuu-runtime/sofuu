import AppKit
import Foundation

/// Crash + error reporting to stderr — the `frontend_log` command's twin
/// (errlog.ts + ErrorBoundary.tsx): a release app has no console, so the
/// JS-side errors and uncaught Swift exceptions land in stderr where
/// `Console.app` / a terminal launch can see them.
enum CrashLog {

    private static func log(_ line: String) {
        FileHandle.standardError.write(Data("[sofuu-mac] \(line)\n".utf8))
    }

    static func install() {
        NSSetUncaughtExceptionHandler { ex in
            FileHandle.standardError.write(
                Data("[sofuu-mac] uncaught exception: \(ex.name.rawValue): \(ex.reason ?? "-")\n\(ex.callStackSymbols.joined(separator: "\n"))\n".utf8))
        }
    }

    static func error(_ message: String) { log("error: \(message)") }
}
