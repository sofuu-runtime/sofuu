import Foundation

/// Resolves and runs language servers: one client per (language, root).
/// Detection is honest — if a server binary is not on PATH, that language
/// simply has no LSP features and the editor still works (highlighting,
/// find, file ops).
///
/// Launches are ASYNC by construction: `which` and the server spawn run
/// on a private queue, never the main thread. Blocking a task wait inside
/// NSViewRepresentable.makeNSView pumps the run loop, SwiftUI re-enters
/// and deadlocks on the registry lock (observed hard hang).
final class LSPRegistry {
    static let shared = LSPRegistry()

    /// "<langID>#<rootPath>" → client. The root is part of the identity:
    /// a workspace switch must never feed a foreign root to a live server.
    private var clients: [String: LSPClient] = [:]
    private var whichCache: [String: String?] = [:]         // spawnQueue only
    private var launching: Set<String> = []                 // lock-guarded
    private var waiters: [String: [(LSPClient?) -> Void]] = [:]  // lock-guarded
    private let lock = NSLock()
    private let spawnQueue = DispatchQueue(label: "sofuu.lsp.spawn", qos: .userInitiated)
    private var reaperTimer: Timer?

    private init() {}

    private func key(_ langID: String, _ rootPath: String) -> String {
        langID + "#" + rootPath
    }

    /// The running client for a language at a root, if any. Non-blocking.
    func runningClient(langID raw: String, rootPath: String) -> LSPClient? {
        lock.lock()
        defer { lock.unlock() }
        let k = key(raw, rootPath)
        if let existing = clients[k] {
            if existing.isRunning {
                existing.touch()
                return existing
            }
            clients.removeValue(forKey: k)
        }
        return nil
    }

    /// Resolve + launch the server for a language at a root off the main
    /// thread. The completion runs on the MAIN queue with the client, or
    /// nil when no server binary is on PATH. Concurrent requests for the
    /// same (language, root) coalesce onto one launch.
    func clientAsync(for language: Language, rootPath: String,
                     _ completion: @escaping (LSPClient?) -> Void) {
        guard language.server != nil else { completion(nil); return }
        let k = key(language.id.rawValue, rootPath)
        if let existing = runningClient(langID: language.id.rawValue, rootPath: rootPath) {
            DispatchQueue.main.async { completion(existing) }
            return
        }
        lock.lock()
        waiters[k, default: []].append(completion)
        let shouldLaunch = !launching.contains(k)
        launching.insert(k)
        lock.unlock()
        guard shouldLaunch else { return }

        spawnQueue.async { [weak self] in
            let client = self?.launch(language: language, rootPath: rootPath)
            self?.deliver(key: k, client: client)
        }
        ensureReaper()
    }

    /// spawnQueue only.
    private func launch(language: Language, rootPath: String) -> LSPClient? {
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

    private func deliver(key k: String, client: LSPClient?) {
        lock.lock()
        launching.remove(k)
        let toDeliver = waiters[k] ?? []
        waiters[k] = nil
        if let client { clients[k] = client }
        lock.unlock()
        DispatchQueue.main.async {
            for deliver in toDeliver { deliver(client) }
        }
    }

    /// Kill every server whose root is NOT the given one — the workspace
    /// changed, and the old servers (keyed by the old root) have no
    /// business holding hundreds of MB against documents that no longer
    /// exist there. New opens re-spawn at the new root on demand.
    func stopAll(exceptRoot root: String) {
        lock.lock()
        let stale = clients.filter { $0.value.rootPath != root }
        for (k, _) in stale { clients.removeValue(forKey: k) }
        lock.unlock()
        for (_, client) in stale { client.shutdown() }
    }

    /// Shut everything down (app quit path).
    func stopAll() {
        lock.lock()
        let all = Array(clients.values)
        clients = [:]
        let stranded = waiters
        waiters = [:]
        launching = []
        lock.unlock()
        if !stranded.isEmpty {
            DispatchQueue.main.async {
                for (_, fs) in stranded { for f in fs { f(nil) } }
            }
        }
        for client in all { client.shutdown() }
    }

    // MARK: - Idle reaper

    /// A running server with no open documents and no activity for 10
    /// minutes costs hundreds of MB forever — shut it down; the next open
    /// in that language re-spawns it transparently.
    private func ensureReaper() {
        guard reaperTimer == nil else { return }
        reaperTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.reapIdle()
        }
    }

    private func reapIdle() {
        lock.lock()
        let snapshot = clients
        lock.unlock()
        for (k, client) in snapshot {
            if client.isRunning, !client.hasOpenDocs,
               Date().timeIntervalSince(client.lastActivity) > 600 {
                lock.lock()
                clients.removeValue(forKey: k)
                lock.unlock()
                client.shutdown()
            }
        }
        lock.lock()
        let idleRegistry = clients.isEmpty && launching.isEmpty
        lock.unlock()
        if idleRegistry {
            reaperTimer?.invalidate()
            reaperTimer = nil
        }
    }

    // MARK: - PATH resolution

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
}
