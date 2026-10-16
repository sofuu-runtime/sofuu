import SwiftUI
import AppKit

/// IDE left panel — the workspace file tree.
///
/// Directories load lazily: a directory is listed only while expanded,
/// and heavy directories (.git, node_modules, build outputs, caches…)
/// plus symlinked dirs (link cycles) never render. Dirs sort first,
/// names case-insensitively. Refresh re-lists from disk; a workspace
/// change resets expansion and caches by view identity.
struct FileTree: View {
    let store: AppStore

    @State private var expanded: Set<String> = []
    @State private var refreshTick = 0

    var body: some View {
        VStack(spacing: 0) {
            // Traffic-light gutter: in IDE mode the tree is the window's
            // leftmost content, under the hidden titlebar's buttons.
            Color.clear.frame(height: 28)
            header
            Divider().overlay(Color.primary.opacity(0.06))
            if let root = store.state.projectDir {
                ScrollView {
                    DirectoryRows(store: store, dir: root, depth: 0, expanded: $expanded)
                        .padding(.vertical, 4)
                }
                .id(root)                                   // workspace change → fresh tree
                .onChange(of: store.state.projectDir) { _ in expanded = [] }
            } else {
                Spacer()
                Text("No workspace open")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
    }

    private var header: some View {
        HStack(spacing: 6) {
            Text(store.state.projectDir.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "FILES")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            // Refresh re-lists from disk (the agent may have created files).
            Button {
                refreshTick += 1
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .help("Refresh tree")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }
}

// MARK: - Model

struct FileEntry: Identifiable {
    let name: String
    let path: String
    let isDir: Bool
    var id: String { path }
}

enum FileListing {
    static let skippedDirs: Set<String> = [
        ".git", "node_modules", ".sofuu", ".build", "build", "dist", "target",
        "DerivedData", ".venv", "venv", "__pycache__", ".gradle", ".idea",
        "Pods", "vendor", ".next", ".cache", ".mypy_cache",
    ]
    static let skippedFiles: Set<String> = [".DS_Store"]

    static func entries(of dir: String) -> [FileEntry] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir) else { return [] }
        var out: [FileEntry] = []
        for name in names {
            if skippedDirs.contains(name) || skippedFiles.contains(name) { continue }
            let path = dir + "/" + name
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let url = URL(fileURLWithPath: path)
                if (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true { continue }
                out.append(FileEntry(name: name, path: path, isDir: true))
            } else {
                out.append(FileEntry(name: name, path: path, isDir: false))
            }
        }
        return out.sorted {
            if $0.isDir != $1.isDir { return $0.isDir }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }
}

// MARK: - Rows

private struct DirectoryRows: View {
    let store: AppStore
    let dir: String
    let depth: Int
    @Binding var expanded: Set<String>

    var body: some View {
        let entries = FileListing.entries(of: dir)
        if entries.isEmpty {
            Text(depth == 0 ? "Empty workspace" : "empty")
                .font(.system(size: 11))
                .foregroundColor(Color.secondary.opacity(0.7))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, CGFloat(depth) * 12 + 30)
                .padding(.vertical, 3)
        }
        ForEach(entries) { entry in
            EntryRow(store: store, entry: entry, depth: depth, expanded: $expanded)
        }
    }
}

private struct EntryRow: View {
    let store: AppStore
    let entry: FileEntry
    let depth: Int
    @Binding var expanded: Set<String>

    private var isExpanded: Bool { expanded.contains(entry.path) }
    private var isSelected: Bool { store.state.activeTab == entry.path }

    var body: some View {
        HStack(spacing: 5) {
            Group {
                if entry.isDir {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .semibold))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .frame(width: 10)
                } else {
                    Color.clear.frame(width: 10)
                }
            }
            .foregroundColor(.secondary)

            Image(systemName: entry.isDir ? (isExpanded ? "folder.fill" : "folder") : "doc.text")
                .font(.system(size: 11))
                .foregroundColor(entry.isDir ? Color.secondary : (isSelected ? Color.accentColor : Color.secondary))

            Text(entry.name)
                .font(.system(size: 12))
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundColor(isSelected ? .primary : Color.primary.opacity(0.85))

            Spacer(minLength: 0)
        }
        .padding(.leading, CGFloat(depth) * 12 + 8)
        .padding(.trailing, 6)
        .padding(.vertical, 2.5)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(isSelected ? Color.primary.opacity(0.08) : Color.clear)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            if entry.isDir {
                if isExpanded { expanded.remove(entry.path) } else { expanded.insert(entry.path) }
            } else {
                store.openFile(entry.path)
            }
        }
        .help(entry.path)

        if entry.isDir && isExpanded {
            DirectoryRows(store: store, dir: entry.path, depth: depth + 1, expanded: $expanded)
        }
    }
}
