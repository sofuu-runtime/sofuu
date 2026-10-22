import SwiftUI

/// First-run welcome card + the ⇧⌘/ keyboard-shortcut cheat sheet.
/// Both are plain root-layer overlays (chrome, not content) — they ride
/// above the themed shell, outside the document zoom. Esc closes them
/// via the store's closeTopOverlay chain; the welcome card's first-run
/// gate is `sofuu.onboardingShown` in UserDefaults.
struct WelcomeCard: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        scrim {
            VStack(alignment: .leading, spacing: 0) {
                Text(L10n.t("onboarding.title"))
                    .font(.system(size: 30, weight: .bold, design: .serif))
                    .foregroundStyle(c.text)
                Text("A local-first coding agent for macOS — chat with it, or hand it a folder and let it work in the built-in IDE.")
                    .font(.system(size: 13))
                    .foregroundStyle(c.textMuted)
                    .padding(.top, 8)
                    .fixedSize(horizontal: false, vertical: true)

                VStack(alignment: .leading, spacing: 12) {
                    welcomeRow(icon: "folder",
                               title: "Open a folder",
                               detail: "The IDE lives inside the chat — ⇧⌘I switches between them.")
                    welcomeRow(icon: "bubble.left.and.bubble.right",
                               title: "Pick a model",
                               detail: "Settings (⌘,) manages providers and API keys — keys stay on this Mac.")
                    welcomeRow(icon: "keyboard",
                               title: "Learn the keys",
                               detail: "⇧⌘/ shows the shortcut sheet from anywhere.")
                }
                .padding(.top, 22)

                HStack {
                    Spacer(minLength: 0)
                    Button {
                        store.showShortcutHelp()
                    } label: {
                        Text("Keyboard Shortcuts")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(c.textMuted)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(c.bgCardHover, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                    Button {
                        store.dismissOnboarding()
                    } label: {
                        Text(L10n.t("onboarding.getStarted"))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(c.accentContrast)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(c.accent, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 24)
            }
            .padding(26)
            .frame(width: 460)
            .background(c.bgCard, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(c.border, lineWidth: 1))
            .shadow(color: .black.opacity(0.30), radius: 24, y: 10)
        }
    }

    private func welcomeRow(icon: String, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 15))
                .foregroundStyle(c.accent)
                .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(c.text)
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(c.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func scrim<V: View>(@ViewBuilder _ card: () -> V) -> some View {
        ZStack {
            Rectangle().fill(c.backdropVeil).ignoresSafeArea()
            card()
        }
        .transition(.opacity)
    }
}

struct ShortcutSheet: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    var body: some View {
        ZStack {
            Rectangle().fill(c.backdropVeil).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("Keyboard Shortcuts")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(c.text)
                    Spacer(minLength: 0)
                    Button {
                        store.dismissShortcutHelp()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(c.textFaint)
                            .frame(width: 26, height: 26)
                            .background(c.bgCardHover, in: Circle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.bottom, 14)

                HStack(alignment: .top, spacing: 28) {
                    sectionColumn(Self.left)
                    sectionColumn(Self.right)
                }
            }
            .padding(24)
            .frame(width: 620)
            .background(c.bgCard, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(c.border, lineWidth: 1))
            .shadow(color: .black.opacity(0.30), radius: 24, y: 10)
        }
        .transition(.opacity)
    }

    private func sectionColumn(_ sections: [(String, [ShortcutRow])]) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            ForEach(sections, id: \.0) { section in
                VStack(alignment: .leading, spacing: 6) {
                    Text(section.0)
                        .font(.system(size: 10, weight: .semibold))
                        .tracking(0.8)
                        .foregroundStyle(c.textFaint)
                    ForEach(section.1, id: \.label) { row in
                        HStack {
                            Text(row.label)
                                .font(.system(size: 12.5))
                                .foregroundStyle(c.text)
                            Spacer(minLength: 12)
                            keyCaps(row.combo)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// "⇧⌘F" → individual key caps. Plain glyphs (/, =, −) render as a
    /// single cap.
    private func keyCaps(_ combo: String) -> some View {
        HStack(spacing: 3) {
            ForEach(Array(combo), id: \.self) { ch in
                Text(String(ch))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(c.textMuted)
                    .frame(minWidth: 20, minHeight: 20)
                    .background(c.bgCardHover, in: RoundedRectangle(cornerRadius: 5))
                    .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(c.border, lineWidth: 1))
            }
        }
    }

    // Mirrors the real accelerators registered in SofuuMenus — the cheat
    // sheet lies if they drift, so keep them in step.
    static let left: [(String, [ShortcutRow])] = [
        ("Chat", [
            ShortcutRow(label: "New chat", combo: "⌘N"),
            ShortcutRow(label: "Stop", combo: "⌘."),
            ShortcutRow(label: "Compact session", combo: "⌘K"),
            ShortcutRow(label: "Settings", combo: "⌘,"),
            ShortcutRow(label: "IDE mode", combo: "⇧⌘I"),
            ShortcutRow(label: "This sheet", combo: "⇧⌘/"),
        ]),
        ("Window", [
            ShortcutRow(label: "Toggle sidebar", combo: "⌘B"),
            ShortcutRow(label: "Zoom in / out / actual size", combo: "⌘= ⌘− ⌘0"),
            ShortcutRow(label: "Escape closes overlays", combo: "Esc"),
        ]),
    ]

    static let right: [(String, [ShortcutRow])] = [
        ("Editor", [
            ShortcutRow(label: "Save file", combo: "⌘S"),
            ShortcutRow(label: "Find in file", combo: "⌘F"),
            ShortcutRow(label: "Search project", combo: "⇧⌘F"),
            ShortcutRow(label: "Go to line", combo: "⌘L"),
            ShortcutRow(label: "Document outline", combo: "⇧⌘O"),
            ShortcutRow(label: "Format file", combo: "⌘⌥⇧F"),
            ShortcutRow(label: "Rename symbol", combo: "⌘R"),
            ShortcutRow(label: "Find references", combo: "⇧⌘R"),
            ShortcutRow(label: "Find symbol in workspace", combo: "⇧⌘P"),
        ]),
    ]
}

struct ShortcutRow {
    let label: String
    let combo: String
}
