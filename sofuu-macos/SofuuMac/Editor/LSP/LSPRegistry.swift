import Foundation

/// Resolves and runs language servers: one client per language per app
/// run, spawned against the workspace root. Detection is honest — if a
/// server binary is not on PATH, that language simply has no LSP features
/// and the editor still works (highlighting, find, file ops).
///
/// Launches are ASYNC by construction: `which` and the server spawn run
/// on a private queue, never the main thread. Blocking a task wait inside
/// NSViewRepresentable.makeNSView pumps the run loop, SwiftUI re-enters
/// and deadlocks on the registry lock (observed hard hang).
final class LSPRegistry {
    static let shared = LSPRegistry()

    private var clients: [String: LSPClient] = [:]          // lang id → client
    private var whichCache: [String: String?] = [:]         // spawnQueue only
    private var launching: Set<String> = []                 // lock-guarded
    private var waiters: [String: [(LSPClient?) -> Void]] = [:]  // lock-guarded
    private let lock = NSLock()
    private let spawnQueue = DispatchQueue(label: "sofuu.lsp.spawn", qos: .userInitiated)

    private init() {}

    /// The running client for a language id, if any. Non-blocking.
    func runningClient(langID raw: String) -> LSPClient? {
        lock.lock()
        defer { lock.unlock() }
        if let existing = clients[raw] {
            if existing.isRunning { return existing }
            clients[raw] = nil
        }
        return nil
    }

    /// Resolve + launch the server for a language off the main thread.
    /// The completion runs on the MAIN queue with the client, or nil when
    /// no server binary is on PATH. Concurrent requests for the same
    /// language coalesce onto one launch and all get the outcome.
    func clientAsync(for language: Language, rootPath: String,
                     _ completion: @escaping (LSPClient?) -> Void) {
        guard language.server != nil else { completion(nil); return }
        let key = language.id.rawValue
        if let existing = runningClient(langID: key) {
            DispatchQueue.main.async { completion(existing) }
            return
        }
        lock.lock()
        waiters[key, default: []].append(completion)
        let shouldLaunch = !launching.contains(key)
        launching.insert(key)
        lock.unlock()
        guard shouldLaunch else { return }

        spawnQueue.async { [weak self] in
            let client = self?.launch(language: language, rootPath: rootPath, cacheKey: key)
            self?.deliver(key: key, client: client)
        }
    }

    /// spawnQueue only.
    private func launch(language: Language, rootPath: String, cacheKey: String) -> LSPClient? {
        guard let spec = language.server else { return nil }
        for candidate in spec.candidates {
            guard let binary = resolve(candidate.bin), !binary.isEmpty else { continue }
            guard let transport = LSPTransport(
                executablePath: binary,
                args: candidate.args,
                workingDirectory: rootPath) else { continue }
            return LSPClient(
                transport: transport,
                rootPath: rootPath,
                languageName: language.name)
        }
        return nil
    }

    private func deliver(key: String, client: LSPClient?) {
        lock.lock()
        launching.remove(key)
        let toDeliver = waiters[key] ?? []
        waiters[key] = nil
        if let client { clients[key] = client }
        lock.unlock()
        DispatchQueue.main.async {
            for deliver in toDeliver { deliver(client) }
        }
    }

    /// Where a binary lives on PATH (cached). spawnQueue only.
    private func resolve(_ bin: String) -> String? {
        if let cached = whichCache[bin] { return cached }
        let found: String? = runWhich(bin)
        whichCache[bin] = found
        return found
    }

    private func runWhich(_ bin: String) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        proc.arguments = [bin]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            let path = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (path?.isEmpty == false) ? path : nil
        } catch {
            return nil
        }
    }

    /// Ask every running server to exit (app quit path).
    func stopAll() {
        lock.lock()
        let all = Array(clients.values)
        lock.unlock()
        for client in all { client.shutdown() }
        lock.lock()
        clients = [:]
        lock.unlock()
    }
}
