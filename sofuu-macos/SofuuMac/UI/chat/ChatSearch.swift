import SwiftUI

/// The ⇧⌘S chat search bar: rides over the transcript's top edge (like
/// the IDE HUD panels), lists matches across user + assistant text, and
/// jumps the transcript scroll to the picked row via the store's
/// target/tick handshake (ChatScreen's ScrollViewReader consumes it).
/// Esc closes through closeTopOverlay — the bar is NOT the first
/// responder gate; the global monitor handles it.
struct ChatSearchBar: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var query = ""
    @State private var selectedIndex = -1
    @FocusState private var focused: Bool

    private struct Hit: Identifiable, Equatable {
        let id: String
        let role: String
        let snippet: String
    }

    private var hits: [Hit] {
        guard !query.isEmpty else { return [] }
        return store.state.messages
            .filter { ($0.role == .user || $0.role == .assistant) && $0.status != "streaming" }
            .filter { $0.content.localizedCaseInsensitiveContains(query) }
            .prefix(200)
            .map { m in
                Hit(id: m.id,
                    role: m.role == .user ? "You" : "Sofuu",
                    snippet: Self.snippet(content: m.content, query: query))
            }
    }

    /// ~160 chars around the first hit, hit-clamped at both ends.
    static func snippet(content: String, query: String) -> String {
        let text = content.replacingOccurrences(of: "\n", with: " ")
        guard let range = text.range(of: query, options: .caseInsensitive) else {
            return String(text.prefix(160))
        }
        let mid = text.distance(from: text.startIndex, to: range.lowerBound)
        let half = 75
        let start = text.index(text.startIndex, offsetBy: max(0, mid - half), limitedBy: text.endIndex) ?? text.startIndex
        let end = text.index(start, offsetBy: 160, limitedBy: text.endIndex) ?? text.endIndex
        var slice = String(text[start..<end]).trimmingCharacters(in: .whitespaces)
        if start > text.startIndex { slice = "…" + slice }
        if end < text.endIndex { slice += "…" }
        return slice
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                TextField("Search chat…", text: $query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundStyle(c.text)
                    .focused($focused)
                    .onSubmit { step(1) }
                if !hits.isEmpty {
                    Text("\(min(selectedIndex + 1, hits.count))/\(hits.count)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(c.textFaint)
                }
                hudButton("chevron.up", "Previous match") { step(-1) }
                    .disabled(hits.isEmpty)
                hudButton("chevron.down", "Next match") { step(1) }
                    .disabled(hits.isEmpty)
                hudButton("xmark", "Close (Esc)") { store.closeChatSearch() }
            }
            .padding(.horizontal, 10)
            .frame(height: 38)
            .background(c.bgCard)

            Divider().overlay(c.separator)

            if query.isEmpty {
                Text("Search your messages and Sofuu's replies")
                    .font(.system(size: 11))
                    .foregroundStyle(c.textFaint)
                    .padding(10)
            } else if hits.isEmpty {
                Text("No matches")
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .padding(10)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(Array(hits.enumerated()), id: \.element.id) { i, hit in
                            Button { jump(i) } label: {
                                HStack(alignment: .top, spacing: 8) {
                                    Text(hit.role)
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(hit.role == "You" ? c.textMuted : c.accent)
                                        .frame(width: 36, alignment: .leading)
                                        .padding(.top, 1)
                                    Text(hit.snippet)
                                        .font(.system(size: 12))
                                        .foregroundStyle(c.text)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.leading)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(i == selectedIndex ? c.bgCardHover : Color.clear)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(maxHeight: 260)
            }
        }
        .frame(width: 560)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(c.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.24), radius: 16, y: 6)
        .transition(.opacity)
        .onAppear { focused = true }
    }

    private func hudButton(_ icon: String, _ help: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(c.textMuted)
                .frame(width: 22, height: 22)
                .background(c.bgCardHover, in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    private func step(_ delta: Int) {
        guard !hits.isEmpty else { return }
        let from = selectedIndex < 0 ? (delta > 0 ? 0 : hits.count - 1)
            : ((selectedIndex + delta + hits.count) % hits.count)
        jump(from)
    }

    private func jump(_ index: Int) {
        guard hits.indices.contains(index) else { return }
        selectedIndex = index
        store.jumpToMessage(hits[index].id)
    }
}
