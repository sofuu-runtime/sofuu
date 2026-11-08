import SwiftUI

/// The IDE git panel (#3 follow-up: staging + commit, read-only status
/// existed since Phase 1). Rides the IDE layout's top edge like the chat
/// search bar. Whole-file staging is the default; a file row expands into
/// hunk rows so one change can be staged without the rest of the file.
/// Deletion rows show D; untracked show +; restores are via
/// `git restore --staged` (unstaged content is never touched).
struct GitPanel: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var message = ""
    @State private var errorText: String?
    /// Expanded hunk lists, keyed "u:<relative>" (worktree) or
    /// "s:<relative>" (index).
    @State private var expanded: Set<String> = []
    @State private var hunks: [String: [GitHunk]] = [:]
    @State private var loading: Set<String> = []
    @FocusState private var messageFocused: Bool

    private var staged: [GitFileState] {
        store.state.gitFiles.values.filter { $0.staged != nil }.sorted { $0.relative < $1.relative }
    }

    private var unstaged: [GitFileState] {
        store.state.gitFiles.values
            .filter { $0.unstaged != nil || $0.staged == nil }
            .sorted { $0.relative < $1.relative }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(c.separator)

            if store.state.gitBranch.isEmpty && store.state.gitFiles.isEmpty {
                Text(L10n.t("git.notRepo"))
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .padding(14)
            } else if staged.isEmpty && unstaged.isEmpty {
                Text(L10n.t("git.clean"))
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .padding(14)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        if !staged.isEmpty { section(L10n.t("git.staged"), staged, isStaged: true) }
                        if !unstaged.isEmpty { section(L10n.t("git.unstaged"), unstaged, isStaged: false) }
                    }
                }
                .frame(maxHeight: 300)

                Divider().overlay(c.separator)
                commitRow
            }
            if let errorText {
                Text(errorText)
                    .font(.system(size: 11))
                    .foregroundStyle(c.err)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
            }
        }
        .frame(width: 640)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(c.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.24), radius: 16, y: 6)
        .transition(.opacity)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.triangle.branch")
                .font(.system(size: 11))
                .foregroundStyle(c.textMuted)
            Text(store.state.gitBranch.isEmpty ? L10n.t("git.title") : "\(L10n.t("git.title")) · \(store.state.gitBranch)")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(c.text)
            Spacer(minLength: 8)
            Button {
                store.refreshGitStatus(force: true)
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(c.textMuted)
                    .frame(width: 22, height: 22)
                    .background(c.bgCardHover, in: RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .help(L10n.t("git.refresh"))
            Button {
                store.setGitPanelOpen(false)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(c.textMuted)
                    .frame(width: 22, height: 22)
                    .background(c.bgCardHover, in: RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .help(L10n.t("search.close"))
        }
        .padding(.horizontal, 12)
        .frame(height: 38)
    }

    private func section(_ label: String, _ files: [GitFileState], isStaged: Bool) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(c.textFaint)
                .padding(.horizontal, 12)
                .padding(.top, 10)
                .padding(.bottom, 4)
            ForEach(files) { f in
                VStack(alignment: .leading, spacing: 0) {
                    row(f, isStaged: isStaged)
                    if expandedHunks(f, isStaged: isStaged) {
                        hunkRows(f.relative, isStaged: isStaged)
                    }
                }
            }
        }
    }

    private func hunkKey(_ relative: String, isStaged: Bool) -> String {
        (isStaged ? "s:" : "u:") + relative
    }

    /// A file can show hunks only when the corresponding diff exists:
    /// untracked files have no worktree diff (whole-file stage is the only
    /// option) and a fully-unstaged file has no index diff.
    private func canExpand(_ f: GitFileState, isStaged: Bool) -> Bool {
        if isStaged { return f.staged != nil }
        guard let unstaged = f.unstaged else { return false }
        return unstaged != .untracked
    }

    private func expandedHunks(_ f: GitFileState, isStaged: Bool) -> Bool {
        expanded.contains(hunkKey(f.relative, isStaged: isStaged))
    }

    private func toggleHunks(_ relative: String, isStaged: Bool) {
        let key = hunkKey(relative, isStaged: isStaged)
        if expanded.contains(key) {
            expanded.remove(key)
            return
        }
        expanded.insert(key)
        guard hunks[key] == nil else { return }
        loading.insert(key)
        store.gitLoadHunks(relative, staged: isStaged) { list in
            loading.remove(key)
            hunks[key] = list
        }
    }

    private func reloadHunks(_ relative: String, isStaged: Bool) {
        let key = hunkKey(relative, isStaged: isStaged)
        hunks[key] = nil
        guard expanded.contains(key) else { return }
        loading.insert(key)
        store.gitLoadHunks(relative, staged: isStaged) { list in
            loading.remove(key)
            hunks[key] = list
        }
    }

    @ViewBuilder
    private func hunkRows(_ relative: String, isStaged: Bool) -> some View {
        let key = hunkKey(relative, isStaged: isStaged)
        let list = hunks[key] ?? []
        if loading.contains(key) {
            Text(L10n.t("git.loadingHunks"))
                .font(.system(size: 10))
                .foregroundStyle(c.textFaint)
                .padding(.leading, 42)
                .padding(.vertical, 3)
        } else if list.isEmpty {
            Text(L10n.t("git.noHunks"))
                .font(.system(size: 10))
                .foregroundStyle(c.textFaint)
                .padding(.leading, 42)
                .padding(.vertical, 3)
        } else {
            ForEach(list) { h in
                HStack(spacing: 8) {
                    Text(h.header)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(c.textMuted)
                        .lineLimit(1)
                    Text("+\(h.added) −\(h.removed)")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(c.textFaint)
                    Spacer(minLength: 8)
                    Button(isStaged ? L10n.t("git.unstageHunk") : L10n.t("git.stageHunk")) {
                        applyHunk(relative, hunk: h, isStaged: isStaged)
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(c.accent)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(c.bgCardHover, in: Capsule())
                    .help(h.header)
                }
                .padding(.leading, 42)
                .padding(.trailing, 12)
                .padding(.vertical, 2)
            }
        }
    }

    private func applyHunk(_ relative: String, hunk: GitHunk, isStaged: Bool) {
        let apply: (@escaping (Bool, String) -> Void) -> Void = isStaged
            ? { done in store.gitUnstageHunk(relative, hunkIndex: hunk.index, hunkHeader: hunk.header, onDone: done) }
            : { done in store.gitStageHunk(relative, hunkIndex: hunk.index, hunkHeader: hunk.header, onDone: done) }
        apply { ok, err in
            guard ok else {
                errorText = err
                return
            }
            errorText = nil
            reloadHunks(relative, isStaged: isStaged)
        }
    }

    private func row(_ f: GitFileState, isStaged: Bool) -> some View {
        let tail = (f.relative as NSString).lastPathComponent
        return HStack(spacing: 8) {
            if canExpand(f, isStaged: isStaged) {
                Button {
                    toggleHunks(f.relative, isStaged: isStaged)
                } label: {
                    Image(systemName: expandedHunks(f, isStaged: isStaged)
                            ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(c.textFaint)
                        .frame(width: 12, height: 12)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(L10n.t("git.hunks"))
            } else {
                Color.clear.frame(width: 12, height: 12)
            }
            Text(f.staged?.glyph ?? (f.unstaged?.glyph ?? "M"))
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(f.unstaged != nil ? c.warn : c.textMuted)
                .frame(width: 14)
            Button(tail) {
                store.openFile(store.state.projectDir.map { $0 + "/" + f.relative } ?? f.relative)
            }
            .buttonStyle(.plain)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(c.text)
            .lineLimit(1)
            Text(f.relative)
                .font(.system(size: 10))
                .foregroundStyle(c.textFaint)
                .lineLimit(1)
                .truncationMode(.head)
            Spacer(minLength: 8)
            if f.staged != nil {
                icon("minus.circle", L10n.t("git.unstage")) { store.gitUnstage([f.relative]) }
            }
            if f.unstaged != nil || f.staged == nil {
                icon("plus.circle", L10n.t("git.stage")) { store.gitStage([f.relative]) }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .help(f.relative)
    }

    private func icon(_ system: String, _ help: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: 12))
                .foregroundStyle(c.textMuted)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    private var commitRow: some View {
        HStack(spacing: 8) {
            TextField(L10n.t("git.messagePlaceholder"), text: $message, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(c.text)
                .lineLimit(1...3)
                .focused($messageFocused)
            Button(action: commit) {
                Text(L10n.t("git.commit"))
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(c.accentContrast)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(c.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(message.trimmingCharacters(in: .whitespaces).isEmpty
                      || store.state.gitFiles.values.allSatisfy { $0.staged == nil })
        }
        .padding(10)
        .onAppear { DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { messageFocused = true } }
        .onSubmit { commit() }
    }

    private func commit() {
        store.gitCommit(message: message) { ok, err in
            errorText = ok ? nil : err
            if ok {
                message = ""
                messageFocused = false
            }
        }
    }
}
