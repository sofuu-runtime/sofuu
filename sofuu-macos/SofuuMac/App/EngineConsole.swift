import AppKit

/// The engine console. The Rust engine logs to stderr; capturing it used to
/// require launching from a terminal. This tees fd 2 through a pipe into
/// (1) an in-memory ring for the Settings ▸ Console pane and (2) an
/// on-disk log (the Android `files/engine-console.log` twin), while the
/// original stderr keeps receiving everything — terminal output unchanged.
///
/// One global side effect, installed once: dup2 moves fd 2 onto a pipe.
/// CrashLog lines ride the same ring so errors ARE the console.
enum EngineConsole {
    static let ring = Ring()
    private static var installed = false
    /// The capture pipe must outlive the process or the reader sees EOF.
    private static var retainedPipe: Pipe?
    private static var logHandle: FileHandle?

    /// Ordered, capped, thread-safe line store.
    final class Ring {
        private var lines: [String] = []
        private let lock = NSLock()
        static let cap = 2000

        func append(_ line: String) {
            lock.lock()
            lines.append(line)
            if lines.count > Ring.cap {
                lines.removeFirst(lines.count - Ring.cap)
            }
            lock.unlock()
        }

        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return lines
        }

        func clear() {
            lock.lock()
            lines = []
            lock.unlock()
        }
    }

    static func install() {
        guard !installed else { return }
        installed = true

        // Preserve the real stderr for pass-through, then move fd 2 onto
        // the capture pipe.
        let savedFD = dup(STDERR_FILENO)
        let capture = Pipe()
        retainedPipe = capture
        guard dup2(capture.fileHandleForWriting.fileDescriptor, STDERR_FILENO) != -1 else {
            return
        }

        // On-disk log: ~/Library/Application Support/SofuuMac/engine-console.log
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SofuuMac", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let logURL = dir.appendingPathComponent("engine-console.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        logHandle = try? FileHandle(forWritingTo: logURL)
        logHandle?.seekToEndOfFile()

        let readHandle = capture.fileHandleForReading
        let savedHandle = FileHandle(fileDescriptor: savedFD, closeOnDealloc: false)
        DispatchQueue.global(qos: .utility).async {
            while true {
                // Blocking read: the pipe's write end (fd 2) never closes.
                let data = readHandle.availableData
                guard !data.isEmpty else { continue }
                savedHandle.write(data)
                logHandle?.write(data)
                // A chunk can carry several lines or a partial line; keep
                // the ring line-oriented on a best-effort basis.
                let text = String(decoding: data, as: UTF8.self)
                for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
                    ring.append(String(line))
                }
            }
        }
    }

    /// Direct app-side console line (CrashLog paths).
    static func log(_ line: String) {
        ring.append(line)
        let data = Data("[app] \(line)\n".utf8)
        logHandle?.write(data)
    }
}
