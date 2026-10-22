import Foundation

/// Project-wide text search: a plain background walk of the workspace
/// honoring FileListing's heavy-directory skips (node_modules/.git/…),
/// ≤1 MB UTF-8 files only — identical rules to what the editor can open,
/// so every hit is clickable. Capped at 400 hits.
struct SearchHit {
    let path: String
    let line: Int        // 1-based for display
    let column: Int      // 0-based UTF-16 offset in the line
    let snippet: String
}

enum ProjectSearch {
    static let maxHits = 400

    static func search(root: String, query: String, caseSensitive: Bool,
                       completion: @escaping ([SearchHit]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let hits = run(root: root, query: query, caseSensitive: caseSensitive)
            DispatchQueue.main.async { completion(hits) }
        }
    }

    private static func run(root: String, query: String, caseSensitive: Bool) -> [SearchHit] {
        guard !query.isEmpty else { return [] }
        var hits: [SearchHit] = []
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: URL(fileURLWithPath: root),
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [],
            errorHandler: { _, _ in true }) else { return [] }

        let options: NSString.CompareOptions = caseSensitive ? [] : [.caseInsensitive]
        for case let url as URL in enumerator {
            if hits.count >= maxHits { break }
            let name = url.lastPathComponent
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                if FileListing.skippedDirs.contains(name) { enumerator.skipDescendants() }
                continue
            }
            if FileListing.skippedFiles.contains(name) { continue }
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize,
                  size <= 1_000_000,
                  let data = try? Data(contentsOf: url),
                  let text = String(data: data, encoding: .utf8) else { continue }

            let ns = text as NSString
            var lineIndex = 0
            var probe = 0
            while probe < ns.length, hits.count < maxHits {
                let lineRange = ns.lineRange(for: NSRange(location: probe, length: 0))
                guard lineRange.length > 0 else { break }
                let lineText = ns.substring(with: lineRange)
                let found = (lineText as NSString).range(of: query, options: options)
                if found.location != NSNotFound {
                    let trimmed = lineText.trimmingCharacters(in: .whitespacesAndNewlines)
                    hits.append(SearchHit(
                        path: url.path,
                        line: lineIndex + 1,
                        column: found.location,
                        snippet: String(trimmed.prefix(160))))
                }
                probe = NSMaxRange(lineRange)
                lineIndex += 1
            }
        }
        return hits
    }
}
