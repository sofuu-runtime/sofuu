import Foundation

/// GGUF models available to the inference engine.
///
/// The engine opens a model by absolute path through `mmap`, so the import
/// flow copies the picked file into `~/.sofuu/models/` — the shared home's
/// Sofuu directory (real `$HOME` on the Mac, no container restamping), where
/// the CLI's own model imports also land.
///
/// Nothing here talks to the engine; that is the engine's job.
enum LocalModels {

    /// Where imports land.
    static var dir: URL {
        let u = MacPaths.sofuuDir.appendingPathComponent("models", isDirectory: true)
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
        let name = source.lastPathComponent
        guard !name.isEmpty, name != ".", name != "..", !name.hasPrefix("."),
              name.lowercased().hasSuffix(".gguf") else { return nil }

        let dest = dir.appendingPathComponent(name)
        let fm = FileManager.default
        do {
            // Copy to a temp name and move into place. Removing the old
            // file FIRST meant a failed copy left the user with neither
            // the new model nor the old one.
            let tmp = dir.appendingPathComponent(".\(name).incoming")
            try? fm.removeItem(at: tmp)
            let scoped = source.startAccessingSecurityScopedResource()
            defer { if scoped { source.stopAccessingSecurityScopedResource() } }
            try fm.copyItem(at: source, to: tmp)
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: tmp, to: dest)
        } catch {
            try? fm.removeItem(at: dir.appendingPathComponent(".\(name).incoming"))
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
