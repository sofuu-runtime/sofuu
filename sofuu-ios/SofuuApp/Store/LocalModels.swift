import Foundation

/// GGUF models stored on the device.
///
/// The infer engine opens a model by absolute path through `mmap`
/// (`crates/sofuu-core/src/infer/mmap.rs:21`), so a file the user picked
/// from Files is only usable once it lives somewhere the app owns. Every
/// import is copied into the container's `models/` directory and referred to
/// by that path.
///
/// The app already pins `HOME` at the container (SofuuBridge.setHome), so
/// `~/model.gguf` would also resolve — but an absolute path is what the
/// engine logs and what `clean_model_path` round-trips cleanly, so the
/// absolute form is what gets stored and passed.
///
/// Nothing here talks to the engine; that is the engine's job.
enum LocalModels {

    /// Where imports land. App-private, so no permission is involved.
    static var dir: URL {
        let u = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        return u
    }

    /// Installed models, newest first.
    static func list() -> [URL] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return [] }
        return names
            .filter { $0.lowercased().hasSuffix(".gguf") }
            .map { dir.appendingPathComponent($0) }
            .filter { fm.fileExists(atPath: $0.path) }
            .sorted {
                (try? fm.attributesOfItem(atPath: $0.path)[.modificationDate] as? Date) ?? Date.distantPast
                    > (try? fm.attributesOfItem(atPath: $1.path)[.modificationDate] as? Date) ?? Date.distantPast
            }
    }

    /// Copy a picked model into `dir`. Returns the destination, or nil on
    /// any failure (unreadable source, disk full, a rejected name).
    @discardableResult
    static func importModel(from source: URL) -> URL? {
        // The picked name is untrusted input: keep the last path component
        // only, and insist it really is a .gguf — a file provider can hand
        // back anything.
        let name = source.lastPathComponent
        guard !name.isEmpty, name != ".", name != "..", !name.hasPrefix("."),
              name.lowercased().hasSuffix(".gguf") else { return nil }

        let dest = dir.appendingPathComponent(name)
        let fm = FileManager.default
        do {
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            // A picked document may be outside the sandbox and security-
            // scoped; startAccessing makes it readable for the copy.
            let scoped = source.startAccessingSecurityScopedResource()
            defer { if scoped { source.stopAccessingSecurityScopedResource() } }
            try fm.copyItem(at: source, to: dest)
        } catch {
            try? fm.removeItem(at: dest)
            return nil
        }
        return dest
    }

    /// Remove an installed model. Refuses anything outside `dir`.
    @discardableResult
    static func delete(_ url: URL) -> Bool {
        guard url.deletingLastPathComponent().standardizedFileURL == dir.standardizedFileURL else {
            return false
        }
        return (try? FileManager.default.removeItem(at: url)) != nil
    }

    /// Human label for the model picker: the file name minus the suffix.
    static func label(_ url: URL) -> String {
        var n = url.lastPathComponent
        if n.lowercased().hasSuffix(".gguf") { n = String(n.dropLast(5)) }
        return n
    }

    /// Size in bytes, or nil when it cannot be read.
    static func size(_ url: URL) -> Int64? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? NSNumber)?.int64Value
    }
}