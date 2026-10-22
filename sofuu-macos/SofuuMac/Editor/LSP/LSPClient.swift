import Foundation

/// The language-server protocol client: lifecycle (initialize → ready →
/// shutdown), text sync (open/change/save/close), and the request set the
/// editor uses (hover, completion, definition, symbols). Written from
/// scratch on top of LSPTransport; server replies are dispatched on main.
final class LSPClient {
    let rootPath: String
    let languageName: String
    private let transport: LSPTransport

    private var nextID = 1
    private var pending: [Int: (Any?) -> Void] = [:]   // id → handler (main)
    private var ready = false
    private var queuedOps: [() -> Void] = []           // ops captured pre-initialize
    private var openVersions: [String: Int] = [:]      // uri → doc version
    private(set) var isRunning = true

    /// (filesystemPath, diagnostics) per publishDiagnostics — main thread.
    var onDiagnostics: ((String, [LSPWire.Diagnostic]) -> Void)?
    /// The server process exited on its own.
    var onServerExit: (() -> Void)?

    init(transport: LSPTransport, rootPath: String, languageName: String) {
        self.transport = transport
        self.rootPath = rootPath
        self.languageName = languageName

        transport.onExit = { [weak self] in
            guard let self else { return }
            self.isRunning = false
            // Ambient the stranded requests — their handlers capture caller
            // state and would otherwise sit in `pending` forever.
            let stranded = self.pending
            self.pending = [:]
            for (_, handler) in stranded { handler(nil) }
            self.onServerExit?()
        }
        transport.onMessage = { [weak self] message in
            DispatchQueue.main.async { self?.handle(message) }
        }

        // Handshake: initialize → initialized → flush queued sync ops.
        request("initialize", params: [
            "processId": Int(ProcessInfo.processInfo.processIdentifier),
            "rootUri": LSPWire.uriOf(rootPath),
            "rootPath": rootPath,
            "capabilities": [
                "textDocument": [
                    "hover": ["contentFormat": ["plaintext", "markdown"]],
                    "completion": ["completionItem": ["snippetSupport": false]],
                ],
            ],
            "workspaceFolders": [[
                "uri": LSPWire.uriOf(rootPath),
                "name": (rootPath as NSString).lastPathComponent,
            ]],
        ]) { [weak self] result in
            guard let self else { return }
            _ = result
            self.notify("initialized", params: [:])
            self.ready = true
            let ops = self.queuedOps
            self.queuedOps = []
            ops.forEach { $0() }
        }
        isRunning = true
    }

    // MARK: - Wire plumbing

    private func request(_ method: String, params: [String: Any]?, then: ((Any?) -> Void)? = nil) {
        let id = nextID
        nextID += 1
        var message: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
        if let params { message["params"] = params }
        if let then { pending[id] = then }
        transport.send(message)
    }

    private func notify(_ method: String, params: [String: Any]) {
        transport.send(["jsonrpc": "2.0", "method": method, "params": params])
    }

    /// Route one decoded message: response (has id) → pending handler;
    /// notification (has method) → diagnostics/etc; server requests with
    /// unknown methods get a null result so servers never hang.
    private func handle(_ message: [String: Any]) {
        if let id = message["id"] as? Int, message["method"] == nil {
            if let handler = pending.removeValue(forKey: id) {
                let error = message["error"] as? [String: Any]
                let result = (message["result"] as Any?)
                if error != nil {
                    handler(nil)
                } else {
                    handler(result)
                }
            }
            return
        }

        guard let method = message["method"] as? String else { return }

        if method == "textDocument/publishDiagnostics" {
            let params = message["params"] as? [String: Any]
            let uri = params?["uri"] as? String ?? ""
            let path = LSPWire.pathOf(uri)
            let diags = LSPWire.parseDiagnostics(params?["diagnostics"])
            onDiagnostics?(path, diags)
            return
        }

        // Server → client request we do not implement: reply null.
        if message["id"] != nil {
            let id = message["id"] as? Int ?? -1
            transport.send(["jsonrpc": "2.0", "id": id, "result": NSNull()])
        }
    }

    private func whenReady(_ op: @escaping () -> Void) {
        if ready { op() } else { queuedOps.append(op) }
    }

    var isReady: Bool { ready }

    // MARK: - Document sync

    /// Open the document on the server (idempotent per path).
    func didOpen(path: String, text: String) {
        whenReady { [weak self] in
            guard let self else { return }
            let uri = LSPWire.uriOf(path)
            guard self.openVersions[uri] == nil else { return }
            let version = 1
            self.openVersions[uri] = version
            let languageID = Languages.language(forPath: path).lspLanguageID
            self.notify("textDocument/didOpen", params: [
                "textDocument": [
                    "uri": uri,
                    "languageId": languageID,
                    "version": version,
                    "text": text,
                ],
            ])
        }
    }

    /// Full-text sync (universally supported); debounced by the caller.
    func didChange(path: String, text: String) {
        whenReady { [weak self] in
            guard let self else { return }
            let uri = LSPWire.uriOf(path)
            guard self.openVersions[uri] != nil else { return }
            let version = (self.openVersions[uri] ?? 1) + 1
            self.openVersions[uri] = version
            self.notify("textDocument/didChange", params: [
                "textDocument": ["uri": uri, "version": version],
                "contentChanges": [["text": text]],
            ])
        }
    }

    func didSave(path: String) {
        whenReady { [weak self] in
            guard let self, self.openVersions[LSPWire.uriOf(path)] != nil else { return }
            self.notify("textDocument/didSave", params: ["textDocument": ["uri": LSPWire.uriOf(path)]])
        }
    }

    func didClose(path: String) {
        whenReady { [weak self] in
            guard let self else { return }
            let uri = LSPWire.uriOf(path)
            guard self.openVersions[uri] != nil else { return }
            self.openVersions.removeValue(forKey: uri)
            self.notify("textDocument/didClose", params: ["textDocument": ["uri": uri]])
        }
    }

    /// If the document was never opened (viewer-only path), open it now —
    /// hover/completion need the server to know the file.
    private func ensureOpen(path: String, text: String?) {
        if openVersions[LSPWire.uriOf(path)] == nil, let text {
            didOpen(path: path, text: text)
        }
    }

    // MARK: - Requests

    func hover(path: String, line: Int, character: Int, text: String?, then: @escaping (String?) -> Void) {
        guard ready else { return then(nil) }
        ensureOpen(path: path, text: text)
        request("textDocument/hover", params: [
            "textDocument": ["uri": LSPWire.uriOf(path)],
            "position": ["line": line, "character": character],
        ]) { result in
            then(LSPWire.parseHover(result))
        }
    }

    func completion(path: String, line: Int, character: Int, text: String?, then: @escaping ([LSPWire.CompletionItem]) -> Void) {
        guard ready else { return then([]) }
        ensureOpen(path: path, text: text)
        request("textDocument/completion", params: [
            "textDocument": ["uri": LSPWire.uriOf(path)],
            "position": ["line": line, "character": character],
        ]) { result in
            then(LSPWire.parseCompletionItems(result))
        }
    }

    func definition(path: String, line: Int, character: Int, text: String?, then: @escaping ([LSPWire.Location]) -> Void) {
        guard ready else { return then([]) }
        ensureOpen(path: path, text: text)
        request("textDocument/definition", params: [
            "textDocument": ["uri": LSPWire.uriOf(path)],
            "position": ["line": line, "character": character],
        ]) { result in
            then(LSPWire.parseLocations(result))
        }
    }

    func symbols(path: String, text: String?, then: @escaping ([LSPWire.OutlineNode]) -> Void) {
        guard ready else { return then([]) }
        ensureOpen(path: path, text: text)
        request("textDocument/documentSymbol", params: [
            "textDocument": ["uri": LSPWire.uriOf(path)],
        ]) { result in
            then(LSPWire.parseSymbols(result))
        }
    }

    // MARK: - Shutdown

    func shutdown() {
        guard isRunning else { return }
        let paths = Array(openVersions.keys)
        for uri in paths {
            notify("textDocument/didClose", params: ["textDocument": ["uri": uri]])
        }
        notify("shutdown", params: [:])
        notify("exit", params: [:])
        transport.stop()
        isRunning = false
    }
}

// MARK: - Payload helpers
