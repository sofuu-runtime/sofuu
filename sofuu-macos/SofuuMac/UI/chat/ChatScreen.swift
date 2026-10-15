import SwiftUI

/// The message list: user bubbles, assistant markdown + activity timeline +
/// usage, and the live phase line.
struct ChatScreen: View {
    let messages: [ChatMessage]
    let phase: String?
    let streaming: Bool
    let chatFontSize: Int
    let preview: SessionPreview?
    /// TUI streaming clock: spinner glyph + phase-sweep cursor (120ms).
    var streamFrame: Int = 0
    /// Joined think deltas — the live "▸ thinking …" line.
    var liveThink: String = ""

    @Environment(\.sofuuColors) private var c

    var body: some View {
        if let preview {
            PreviewPane(preview: preview, fontSize: chatFontSize)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(messages) { m in
                            MessageRow(m: m, fontSize: chatFontSize,
                                       phase: phase, streamFrame: streamFrame,
                                       liveThink: liveThink)
                                .id(m.id)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.top, 12)
                    .padding(.bottom, 16)
                }
                // Stick to the bottom while new content streams.
                .onChange(of: messages.count) { _ in
                    if let last = messages.last { withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(last.id, anchor: .bottom) } }
                }
                .onChange(of: messages.last?.content.count ?? 0) { _ in
                    if let last = messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }
}

private struct MessageRow: View {
    let m: ChatMessage
    let fontSize: Int
    var phase: String? = nil
    var streamFrame: Int = 0
    var liveThink: String = ""

    @Environment(\.sofuuColors) private var c

    var body: some View {
        switch m.role {
        case .user:
            HStack {
                Spacer(minLength: 40)
                Text(m.content)
                    .font(.system(size: CGFloat(fontSize)))
                    .lineSpacing(4)
                    .foregroundStyle(c.text)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(c.bgCard, in: RoundedRectangle(cornerRadius: 14))
                    .padding(.vertical, 6).padding(.horizontal, 4)
            }

        case .system:
            Text(m.content)
                .font(.system(size: CGFloat(fontSize - 3), design: .monospaced))
                .foregroundStyle(c.textMuted)
                .padding(.horizontal, 8).padding(.vertical, 6)

        case .assistant:
            VStack(alignment: .leading, spacing: 0) {
                if m.status == "streaming" {
                    TUILiveLine(phase: phase, frame: streamFrame,
                               think: m.content.isEmpty ? liveThink : "",
                               fontSize: fontSize - 1)
                }
                // Live think rows stay out of the timeline while the turn
                // runs — the "▸ thinking" line carries them, exactly like
                // the TUI (which erases them when the answer starts).
                ToolTimelineView(entries: m.status == "streaming"
                                 ? m.timeline.filter { $0.kind != "think" }
                                 : m.timeline)

                if !m.content.isEmpty {
                    HStack(alignment: .bottom, spacing: 2) {
                        MarkdownBody(text: m.content, fontSize: fontSize)
                        if m.status == "streaming" { SpinnerCursor(frame: streamFrame) }
                    }
                }
                if m.status == "error", let error = m.error {
                    Text(error)
                        .font(.system(size: CGFloat(fontSize - 2)))
                        .foregroundStyle(c.err)
                        .padding(10)
                        .background(c.errBg, in: RoundedRectangle(cornerRadius: 8))
                        .padding(.top, 6)
                }
                if m.status == "done", let usage = m.usage, let line = usage.line() {
                    Text(line)
                        .font(.system(size: CGFloat(fontSize - 4)))
                        .foregroundStyle(c.textFaint)
                        .padding(.top, 4)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 8).padding(.bottom, 10)
        }
    }
}

/// The TUI's live streaming lines (chat.rs driver, ported 1:1):
///
///   * the phase label with ONE dim letter sweeping left→right every
///     120ms tick (`tickPhase` — "[2m" on `PHASE_FRAME`);
///   * the live think line — "  ▸ thinking " + all think chunks joined,
///     italic (`outLast('  ▸ thinking ' + thinks.join(''))`), cleared the
///     moment answer text starts.
struct TUILiveLine: View {
    let phase: String?
    let frame: Int
    let think: String
    let fontSize: Int

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let phase, !phase.isEmpty {
                Text(phaseSweep(phase, frame: frame))
            }
            if !think.isEmpty {
                thinkLine(think)
            }
        }
        .padding(.vertical, 3)
    }

    /// The "▸ thinking …" row: ▸ marker + all think chunks joined,
    /// italic monospace (the TUI's `outLast('  ▸ thinking ' + …)` line).
    private func thinkLine(_ think: String) -> Text {
        var attrs = AttributeContainer()
        attrs.font = .system(size: CGFloat(fontSize), weight: .regular, design: .monospaced)
        attrs.foregroundColor = c.textFaint
        return Text(AttributedString("▸ thinking \(think)", attributes: attrs))
    }

    /// `label` with one dim letter at `frame % label.count` — the sweep,
    /// advancing one position per 120ms tick upstream.
    private func phaseSweep(_ label: String, frame: Int) -> AttributedString {
        var out = AttributedString("  ")  // the TUI's two-space indent
        let chars = Array(label)
        for (i, ch) in chars.enumerated() {
            var a = AttributeContainer()
                .font(.system(size: 13, design: .monospaced))
            a.foregroundColor = i == frame % chars.count ? c.textFaint : c.textMuted
            out.append(AttributedString(String(ch), attributes: a))
        }
        return out
    }
}

/// The answer's live spinner — the TUI's braille frames verbatim
/// (SPINNER = ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏, one frame per 120ms tick), drawn at the
/// end of the growing answer text.
struct SpinnerCursor: View {
    let frame: Int

    @Environment(\.sofuuColors) private var c

    private static let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    var body: some View {
        Text(Self.frames[abs(frame) % Self.frames.count])
            .font(.system(size: 14, design: .monospaced))
            .foregroundStyle(c.textMuted)
    }
}

// MARK: - Tool timeline

private struct TimelineDisplayRow: Identifiable {
    let id: String
    let icon: String
    let title: String
    let badge: String?
    let ok: Bool?
    let detail: String?
    let expandable: Bool
}

/// The tool timeline under an assistant message.
struct ToolTimelineView: View {
    let entries: [TimelineEntry]

    @Environment(\.sofuuColors) private var c
    @State private var expandedId: String?

    var body: some View {
        let rows = displayRows()
        if !rows.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(rows) { r in
                    TimelineRowView(
                        row: r,
                        expanded: expandedId == r.id,
                        onToggle: r.expandable ? {
                            expandedId = (expandedId == r.id) ? nil : r.id
                        } : nil)
                }
            }
            .padding(.top, 6).padding(.bottom, 4)
        }
    }

    /// Consecutive same-tool rows (3+) collapse into one summary row.
    private func displayRows() -> [TimelineDisplayRow] {
        var rows: [TimelineDisplayRow] = []
        var i = 0
        while i < entries.count {
            let e = entries[i]
            var j = i + 1
            if e.kind == "tool" {
                while j < entries.count, entries[j].kind == "tool",
                      (entries[j].tool ?? entries[j].title) == (e.tool ?? e.title) {
                    j += 1
                }
            }
            if j - i >= 3 {
                rows.append(TimelineDisplayRow(
                    id: "run-\(i)", icon: Timeline.iconId(of: e),
                    title: Timeline.composeSummary(Array(entries[i..<j])),
                    badge: nil, ok: nil, detail: nil, expandable: false))
                i = j
                continue
            }
            rows.append(TimelineDisplayRow(
                id: e.id, icon: Timeline.iconId(of: e), title: e.title,
                badge: e.badge, ok: e.ok, detail: e.result ?? e.detail, expandable: true))
            i += 1
        }
        return rows
    }
}

private struct TimelineRowView: View {
    let row: TimelineDisplayRow
    let expanded: Bool
    let onToggle: (() -> Void)?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: iconFor(row.icon))
                    .font(.system(size: 11))
                    .foregroundStyle(row.icon == Timeline.iconWarn ? c.warn : c.textFaint)
                    .frame(width: 14)
                Text(row.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(c.textMuted)
                if let badge = row.badge {
                    Text(badge).font(.system(size: 12)).foregroundStyle(c.textFaint)
                }
                if let ok = row.ok {
                    Text(ok ? "✓" : "✗")
                        .font(.system(size: 12))
                        .foregroundStyle(ok ? c.ok : c.err)
                }
                if onToggle != nil {
                    Image(systemName: Icons2.chevronDown)
                        .font(.system(size: 10))
                        .foregroundStyle(c.textFaint)
                        .opacity(expanded ? 1 : 0.5)
                }
            }
            if expanded, let detail = row.detail {
                Text(String(detail.prefix(1500)))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(c.textFaint)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(c.codeBlock, in: RoundedRectangle(cornerRadius: 6))
                    .padding(.leading, 22).padding(.top, 2).padding(.bottom, 4)
            }
        }
        .padding(.vertical, 3)
        .contentShape(Rectangle())
        .onTapGesture { onToggle?() }
    }
}

// MARK: - Read-only transcript

private struct PreviewPane: View {
    let preview: SessionPreview
    let fontSize: Int

    @Environment(\.sofuuColors) private var c

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                Text("Read-only transcript · \(preview.label) · \(preview.turns.count) turns")
                    .font(.system(size: 12))
                    .foregroundStyle(c.textFaint)
                    .padding(.bottom, 8)

                ForEach(preview.turns) { t in
                    HStack {
                        Spacer(minLength: 40)
                        Text(t.prompt)
                            .font(.system(size: CGFloat(fontSize)))
                            .foregroundStyle(c.text)
                            .padding(.horizontal, 14).padding(.vertical, 10)
                            .background(c.bgCard, in: RoundedRectangle(cornerRadius: 14))
                            .padding(.vertical, 6)
                    }
                    if !t.answer.isEmpty {
                        MarkdownBody(text: t.answer, fontSize: fontSize)
                            .padding(.vertical, 6)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 12)
            .padding(.bottom, 16)
        }
    }
}
