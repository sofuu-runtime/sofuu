import SwiftUI

/// The composer card: text input on top, bottom bar with model/effort pills +
/// icons left and the send pill right; the mode + permissions pills sit below.
/// Same controls as the desktop composer, laid out for touch.
struct Composer: View {
    let streaming: Bool
    let mode: String
    let permissions: String
    let config: SofuuConfig?
    let ctx: CtxMeter?
    let images: [String]
    let pendingMentions: String

    let onSubmit: (String, [String]) -> Void
    let onCancel: () -> Void
    let onModeChange: (String) -> Void
    let onPermissionsChange: (String) -> Void
    let onConfigPatch: (JSONObject) -> Void
    let onOpenSettings: (String) -> Void
    let onAttach: () -> Void
    let onRemoveImage: (Int) -> Void
    let onMentionsConsumed: () -> Void
    let onSheet: (ComposerSheet) -> Void

    @Environment(\.sofuuColors) private var c
    @State private var text = ""
    @FocusState private var focused: Bool

    private var slashOpen: Bool { text.hasPrefix("/") }
    private var slashQuery: String {
        slashOpen ? String(text.dropFirst()).trimmingCharacters(in: .whitespaces).lowercased() : ""
    }
    private var slashMatches: [SlashCommand] {
        guard slashOpen else { return [] }
        return SLASH_COMMANDS.filter {
            slashQuery.isEmpty
                || $0.name.contains(slashQuery)
                || $0.desc.lowercased().contains(slashQuery)
        }
    }

    private var modelLabel: String {
        if let m = config?.model, !m.isEmpty { return m }
        return (config?.providers.isEmpty == false) ? "Choose model" : "No model"
    }
    private var effortLabel: String {
        let e = (config?.effort).flatMap { $0.isEmpty ? nil : $0 } ?? "high"
        return e == "off" ? "Off" : e.prefix(1).uppercased() + e.dropFirst()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !images.isEmpty { imageStrip }

            // Mode + permissions pills — their own row, just ABOVE the input
            // card (never inside the text field).
            HStack(spacing: 6) {
                Pill(label: mode == "chat" ? "Chat only" : "Agent", icon: Icons2.chat) { onSheet(.mode) }
                Pill(label: permissionsLabel(permissions), icon: Icons2.shield) { onSheet(.permissions) }
            }
            .padding(.bottom, 6)

            VStack(alignment: .leading, spacing: 0) {
                TextField("Ask, @mention files, or type / for commands…", text: $text, axis: .vertical)
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundStyle(c.text)
                    .tint(c.accent)
                    .lineLimit(1...8)
                    .focused($focused)
                    .submitLabel(.return)

                if slashOpen && !slashMatches.isEmpty {
                    SlashPalette(commands: slashMatches) { cmd in
                        onSubmit("/" + cmd.name, [])
                        text = ""
                    }
                    .padding(.top, 8)
                }

                HStack(spacing: 6) {
                    Pill(label: modelLabel, icon: Icons2.spark) { onSheet(.model) }
                    Pill(label: effortLabel, icon: nil) { onSheet(.effort) }
                    Spacer(minLength: 8)
                    IconAction(icon: Icons2.attach, description: "Attach") { onAttach() }
                    IconAction(icon: Icons2.layers, description: "Context layers") { onSheet(.layers) }
                    CtxRing(ctx: ctx)
                    if streaming {
                        SendButton(icon: Icons2.stop, tint: c.errStrong) { onCancel() }
                    } else {
                        SendButton(icon: Icons2.send, tint: c.accent) { submit() }
                    }
                }
                .padding(.top, 8)
            }
            .padding(12)
            .background(c.bgCard, in: RoundedRectangle(cornerRadius: 16))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        // @mentions spliced by the file picker land in the input.
        .onChange(of: pendingMentions) { m in
            guard !m.isEmpty else { return }
            text += m
            onMentionsConsumed()
        }
    }

    private var imageStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.offset) { i, url in
                    ZStack(alignment: .topTrailing) {
                        if let img = decodeDataUrl(url) {
                            Image(uiImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 44, height: 44)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        } else {
                            RoundedRectangle(cornerRadius: 8).fill(c.bgCardHover)
                                .frame(width: 44, height: 44)
                        }
                        Button { onRemoveImage(i) } label: {
                            Text("×")
                                .font(.system(size: 11))
                                .foregroundStyle(.white)
                                .frame(width: 18, height: 18)
                                .background(c.errStrong, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .offset(x: 4, y: -4)
                    }
                }
            }
            .padding(.bottom, 8)
        }
    }

    private func submit() {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !streaming, !t.isEmpty || !images.isEmpty else { return }
        onSubmit(text, images)
        text = ""
    }

    private func permissionsLabel(_ p: String) -> String {
        switch p {
        case "full": return "Full access"
        case "edit": return "Edit only"
        case "plan": return "Plan mode"
        default: return "Ask me"
        }
    }
}

enum ComposerSheet: String {
    case model, effort, layers, mode, permissions
}

// MARK: - Small controls

/// A composer bar icon: ONE touch target and ONE glyph size, so every action
/// in the bar is pixel-identical.
struct IconAction: View {
    let icon: String
    let description: String
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            Image(systemName: icon)
                .font(.system(size: 17))
                .foregroundStyle(c.textMuted)
                .frame(width: 40, height: 40)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(description)
    }
}

struct Pill: View {
    let label: String
    let icon: String?
    let onClick: () -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        Button(action: onClick) {
            HStack(spacing: 5) {
                if let icon {
                    Image(systemName: icon).font(.system(size: 11)).foregroundStyle(c.accent)
                }
                Text(label).font(.system(size: 13)).foregroundStyle(c.text).lineLimit(1)
                Image(systemName: Icons2.caret).font(.system(size: 9)).foregroundStyle(c.textFaint)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 40)
            .background(c.bgCardHover, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

private struct SendButton: View {
    let icon: String
    let tint: Color
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            Image(systemName: icon)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(tint, in: Circle())
        }
        .buttonStyle(.plain)
    }
}

/// Live context-window fill ring (the desktop's ContextRing).
struct CtxRing: View {
    let ctx: CtxMeter?

    @Environment(\.sofuuColors) private var c

    var body: some View {
        if let ctx, ctx.window > 0 {
            ZStack {
                Circle()
                    .stroke(c.border, lineWidth: 2.2)
                    .frame(width: 22, height: 22)
                Circle()
                    .trim(from: 0, to: CGFloat(ctx.fraction))
                    .stroke(ringColor(ctx.fraction), style: StrokeStyle(lineWidth: 2.2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 22, height: 22)
                Text("\(ctx.pct)").font(.system(size: 8)).foregroundStyle(c.textFaint)
            }
            .frame(width: 24, height: 24)
        }
    }

    private func ringColor(_ f: Double) -> Color {
        if f > 0.85 { return c.err }
        if f > 0.7 { return c.warn }
        return c.accent
    }
}

/// The "/" autocomplete card.
private struct SlashPalette: View {
    let commands: [SlashCommand]
    let onPick: (SlashCommand) -> Void

    @Environment(\.sofuuColors) private var c

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(commands) { cmd in
                    Button { onPick(cmd) } label: {
                        HStack(spacing: 4) {
                            Text("/\(cmd.name)").foregroundStyle(c.accent)
                            if let arg = cmd.arg { Text(arg).foregroundStyle(c.textFaint) }
                            Spacer(minLength: 8)
                            Text(cmd.desc).foregroundStyle(c.textMuted).lineLimit(1)
                        }
                        .font(.system(size: 13, design: .monospaced))
                        .padding(.horizontal, 10).padding(.vertical, 7)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxHeight: 190)
        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - image helper

/// Decode a `data:image/...;base64,` URL into a UIImage.
func decodeDataUrl(_ dataUrl: String) -> UIImage? {
    guard let comma = dataUrl.firstIndex(of: ",") else { return nil }
    let b64 = String(dataUrl[dataUrl.index(after: comma)...])
    guard let data = Data(base64Encoded: b64) else { return nil }
    return UIImage(data: data)
}
