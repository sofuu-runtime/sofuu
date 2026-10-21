import Foundation

/// JSON-RPC 2.0 over stdio for a spawned language-server process.
/// Owns framing (Content-Length headers), the reader loop, and a
/// serialized writer. No third-party code.
final class LSPTransport {
    enum State { case idle, running, exited }

    let process = Process()
    private var writeHandle: FileHandle?
    /// Request/reply stream as plain bytes: Array indices are ALWAYS
    /// zero-based. (Data's indices shift after removeFirst on some
    /// representations, and offset-slicing it trapped at runtime.)
    private var buffer: [UInt8] = []
    private let stateQueue = DispatchQueue(label: "ide.lsp.state")
    private(set) var state: State = .idle

    /// Delivered on the MAIN thread with one decoded JSON body.
    var onMessage: (([String: Any]) -> Void)?
    var onExit: (() -> Void)?

    init?(executablePath: String, args: [String], workingDirectory: String?) {
        guard FileManager.default.isExecutableFile(atPath: executablePath) else { return nil }
        let url = URL(fileURLWithPath: executablePath)

        process.executableURL = url
        process.arguments = args
        if let workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        }
        process.standardInput = Pipe()
        process.standardOutput = Pipe()
        process.standardError = FileHandle.nullDevice

        guard let inPipe = process.standardInput as? Pipe,
              let outPipe = process.standardOutput as? Pipe else { return nil }
        writeHandle = inPipe.fileHandleForWriting

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard let self, !chunk.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            self.stateQueue.async { [weak self] in
                self?.ingest(chunk)
            }
        }

        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async { [weak self] in
                self?.state = .exited
                self?.onExit?()
            }
        }

        do {
            try process.run()
            state = .running
        } catch {
            return nil
        }
    }

    /// Append a raw chunk and peel off every complete framed message.
    /// Runs on the state queue; messages jump to main.
    private func ingest(_ chunk: Data) {
        buffer.append(contentsOf: chunk)
        while true {
            switch nextFrame() {
            case .message(let body, let consumed):
                buffer.removeFirst(consumed)
                if let object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] {
                    DispatchQueue.main.async { [weak self] in
                        self?.onMessage?(object)
                    }
                }
            case .malformed(let skip):
                // Corrupt header — discard it so the stream never wedges.
                buffer.removeFirst(skip)
            case .incomplete:
                return
            }
        }
    }

    private enum Frame {
        case message(Data, Int)   // body, bytes consumed
        case malformed(Int)       // bytes to discard
        case incomplete
    }

    /// Read one `<header>\r\n\r\n<json>` frame from the buffer.
    private func nextFrame() -> Frame {
        let sep = Array("\r\n\r\n".utf8)
        guard let sepRange = range(of: sep, in: buffer) else { return .incomplete }
        let headerText = String(decoding: buffer[..<sepRange.lowerBound], as: UTF8.self)
        guard let lengthLine = headerText
            .components(separatedBy: "\r\n")
            .first(where: { $0.lowercased().hasPrefix("content-length:") }),
            let length = Int(lengthLine.split(separator: ":").last.map(String.init)?
                .trimmingCharacters(in: .whitespaces) ?? "") else {
            return .malformed(sepRange.upperBound)
        }
        let total = sepRange.upperBound + length
        guard buffer.count >= total else { return .incomplete }
        return .message(Data(buffer[sepRange.upperBound..<total]), total)
    }

    /// First occurrence of `needle` at or after `from`, as buffer offsets.
    private func range(of needle: [UInt8], in haystack: [UInt8], from: Int = 0) -> Range<Int>? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var i = max(0, from)
        let last = haystack.count - needle.count
        while i <= last {
            if haystack[i] == needle[0] {
                var j = 1
                while j < needle.count, haystack[i + j] == needle[j] { j += 1 }
                if j == needle.count { return i..<(i + needle.count) }
            }
            i += 1
        }
        return nil
    }

    /// Send one JSON-RPC message, framed with Content-Length headers.
    func send(_ object: [String: Any]) {
        stateQueue.async { [weak self] in
            guard let self, self.state == .running, self.process.isRunning,
                  let payload = try? JSONSerialization.data(withJSONObject: object) else { return }
            var frame = Data("Content-Length: \(payload.count)\r\n\r\n".utf8)
            frame.append(payload)
            self.writeHandle?.write(frame)
        }
    }

    func stop() {
        guard state == .running else { return }
        state = .exited
        if process.isRunning {
            process.terminate()
        }
    }
}
