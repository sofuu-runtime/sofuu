import SwiftUI
import AppKit

/// IDE left panel — the workspace file tree, with real-IDE file ops:
///
/// - Working refresh: re-lists rows from disk AND reloads clean open tabs
///   (the agent may have created or edited files behind our back).
/// - New file / new folder from the header (workspace root) and from
///   hover quick-actions on any folder row; the name goes through an
///   inline composer — Enter commits, Esc cancels, duplicates refused.
/// - Rename and Move-to-Trash via context menus (delete is confirmed).
///
/// Directories load lazily (listed only while expanded), heavy directories
/// and symlinked dirs never render, and dirs sort first.
struct FileTree: View {
    let store: AppStore

    @State private var expanded: Set<String> = []
    @State private var refreshTick = 0
    @State private var composer: EntryComposer?
    @FocusState private var composerFocused: Bool

    struct EntryComposer: Equatable {
        enum Mode { case newFile, newFolder, rename }
        let mode: Mode
        let dir: String        // parent directory of the entry
        var initial: String    // the field's live text
        var original: String = ""  // rename: the pre-edit name (the source path)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Traffic-light gutter: the tree is the window's leftmost
            // content, under the hidden titlebar's buttons.
            Color.clear.frame(height: 28)
            header
            Divider().overlay(Color.primary.opacity(0.06))
            if let root = store.state.projectDir {
                ScrollView {
                    DirectoryRows(
                        store: store, dir: root, depth: 0,
                        expanded: $expanded, composer: $composer,
                        composerFocused: $composerFocused)
                        .padding(.vertical, 4)
                }
                // Fresh row identity on workspace change AND refresh:
                // everything re-lists from disk; expansion survives.
                .id("\(root)#\(refreshTick)")
                .onChange(of: store.state.projectDir) { _ in
                    expanded = []
                    composer = nil
                }
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
        HStack(spacing: 4) {
            Text(store.state.projectDir.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "FILES")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button {
                guard let root = store.state.projectDir else { return }
                composer = EntryComposer(mode: .newFile, dir: root, initial: "untitled.md")
                composerFocused = true
            } label: {
                Image(systemName: "doc.badge.plus").iconFrame()
            }
            .buttonStyle(.plain)
            .disabled(store.state.projectDir == nil)
            .help("New file")

            Button {
                guard let root = store.state.projectDir else { return }
                composer = EntryComposer(mode: .newFolder, dir: root, initial: "new-folder")
                composerFocused = true
            } label: {
                Image(systemName: "folder.badge.plus").iconFrame()
            }
            .buttonStyle(.plain)
            .disabled(store.state.projectDir == nil)
            .help("New folder")

            // Refresh re-lists from disk and reloads clean open tabs —
            // the agent may have created or edited files behind our back.
            Button {
                refreshTick += 1
                store.refreshTabsAfterTurn()
            } label: {
                Image(systemName: "arrow.clockwise").iconFrame()
            }
            .buttonStyle(.plain)
            .help("Refresh (re-scan from disk)")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }
}

private extension Image {
    func iconFrame() -> some View {
        font(.system(size: 12))
            .foregroundColor(.secondary)
            .frame(width: 26, height: 26)
            .contentShape(Rectangle())
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
    @Binding var composer: FileTree.EntryComposer?
    var composerFocused: FocusState<Bool>.Binding

    var body: some View {
        let entries = FileListing.entries(of: dir)
        VStack(spacing: 0) {
            // The inline composer lives inside the directory it targets.
            if let c = composer, c.dir == dir {
                RowComposer(store: store, composer: $composer,
                            focused: composerFocused, isFolder: c.mode == .newFolder)
            }
            if entries.isEmpty && composer?.dir != dir {
                Text(depth == 0 ? "Empty workspace" : "empty folder")
                    .font(.system(size: 11))
                    .foregroundColor(Color.secondary.opacity(0.7))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, CGFloat(depth) * 12 + 30)
                    .padding(.vertical, 3)
            }
            ForEach(entries) { entry in
                EntryRow(store: store, entry: entry, depth: depth,
                         expanded: $expanded,
                         composer: $composer, composerFocused: composerFocused)
            }
        }
    }
}

private struct EntryRow: View {
    let store: AppStore
    let entry: FileEntry
    let depth: Int
    @Binding var expanded: Set<String>
    @Binding var composer: FileTree.EntryComposer?
    var composerFocused: FocusState<Bool>.Binding

    @Environment(\.sofuuColors) private var c
    @State private var hovering = false

    private var isExpanded: Bool { expanded.contains(entry.path) }
    private var isSelected: Bool { store.state.activeTab == entry.path }
    private var parentDir: String { (entry.path as NSString).deletingLastPathComponent }

    var body: some View {
        VStack(spacing: 0) {
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

                // Hover quick-create on folders: new file / new folder
                // land inside THIS folder (and auto-expand it).
                if entry.isDir && hovering && composer == nil {
                    HStack(spacing: 2) {
                        Button {
                            composer = FileTree.EntryComposer(
                                mode: .newFile, dir: entry.path, initial: "untitled.md")
                            expanded.insert(entry.path)
                            composerFocused.wrappedValue = true
                        } label: {
                            Image(systemName: "doc.badge.plus").miniIcon()
                        }
                        .buttonStyle(.plain)
                        .help("New file in \(entry.name)")

                        Button {
                            composer = FileTree.EntryComposer(
                                mode: .newFolder, dir: entry.path, initial: "new-folder")
                            expanded.insert(entry.path)
                            composerFocused.wrappedValue = true
                        } label: {
                            Image(systemName: "folder.badge.plus").miniIcon()
                        }
                        .buttonStyle(.plain)
                        .help("New folder in \(entry.name)")
                    }
                }
            }
            .padding(.leading, CGFloat(depth) * 12 + 8)
            .padding(.trailing, 6)
            .padding(.vertical, 2.5)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(isSelected ? Color.primary.opacity(0.08) : Color.clear)
            )
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .onTapGesture {
                if entry.isDir {
                    if isExpanded { expanded.remove(entry.path) } else { expanded.insert(entry.path) }
                } else {
                    store.openFile(entry.path)
                }
            }
            .contextMenu {
                Button("Rename…") {
                    composer = FileTree.EntryComposer(
                        mode: .rename, dir: parentDir, initial: entry.name,
                        original: entry.name)
                    composerFocused.wrappedValue = true
                }
                Button("Move to Trash…", role: .destructive) {
                    deleteConfirmed()
                }
            }
            .help(entry.path)

            if entry.isDir && isExpanded {
                if let cm = composer, cm.dir == entry.path {
                    RowComposer(store: store, composer: $composer,
                                focused: composerFocused, isFolder: cm.mode == .newFolder)
                        .padding(.leading, CGFloat(depth) * 12 + 22)
                }
                DirectoryRows(store: store, dir: entry.path, depth: depth + 1,
                              expanded: $expanded,
                              composer: $composer, composerFocused: composerFocused)
            }
        }
    }

    private func deleteConfirmed() {
        let alert = NSAlert()
        alert.messageText = "Move “\(entry.name)” to the Trash?"
        alert.informativeText = entry.isDir
            ? "The folder and everything inside it will be moved to the Trash."
            : "You can restore it from the Trash."
        alert.addButton(withTitle: "Move to Trash")
        alert.addButton(withTitle: "Cancel")
        alert.alertStyle = .warning
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        if let err = store.deleteFile(entry.path) {
            store.systemRow(err)
        }
    }
}

private extension Image {
    func miniIcon() -> some View {
        font(.system(size: 10))
            .foregroundColor(.secondary)
            .frame(width: 20, height: 20)
            .contentShape(Rectangle())
    }
}

/// The inline naming field for create/rename. Enter commits, Esc cancels.
private struct RowComposer: View {
    let store: AppStore
    @Binding var composer: FileTree.EntryComposer?
    var focused: FocusState<Bool>.Binding
    let isFolder: Bool

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: isFolder ? "folder.badge.plus" : "doc.badge.plus")
                .font(.system(size: 11))
                .foregroundColor(.accentColor)
            TextField(isFolder ? "Folder name" : "File name", text: Binding(
                get: { composer?.initial ?? "" },
                set: { composer?.initial = $0 }))
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .focused(focused)
                .onSubmit { commit() }
                .onExitCommand { composer = nil }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 3)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                focused.wrappedValue = true
            }
        }
    }

    private func commit() {
        guard let c = composer else { return }
        let name = composer?.initial ?? c.initial
        var err: String?
        switch c.mode {
        case .newFile:
            err = store.createFile(at: c.dir, name: name)
        case .newFolder:
            err = store.createFolder(at: c.dir, name: name)
        case .rename:
            // The source path uses the ORIGINAL name: `initial` is the
            // live field text, which the user is changing right now.
            err = store.renameFile(c.dir + "/" + c.original, newName: name)
        }
        if let err { store.systemRow(err) }
        composer = nil
    }
}
