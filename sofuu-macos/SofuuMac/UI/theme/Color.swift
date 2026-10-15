import SwiftUI

/// The desktop palette (styles.css `:root` tokens, cream light + warm dark),
/// ported token-for-token through the Android app's `Color.kt`.
///
/// One deliberate divergence carried over from Android: the accent is BLUE,
/// not the desktop's purple, so the UI matches the light-blue cloud
/// background. `#2B6CB0` is 5.4:1 against white text on cream, and `#84B3E8`
/// is the exact cloud sky blue at 7.6:1 on the warm-dark ground.

extension Color {
    /// 0xRRGGBB or 0xAARRGGBB.
    init(hex: UInt32) {
        let a, r, g, b: Double
        if hex > 0xFFFFFF {
            a = Double((hex >> 24) & 0xFF) / 255
            r = Double((hex >> 16) & 0xFF) / 255
            g = Double((hex >> 8) & 0xFF) / 255
            b = Double(hex & 0xFF) / 255
        } else {
            a = 1
            r = Double((hex >> 16) & 0xFF) / 255
            g = Double((hex >> 8) & 0xFF) / 255
            b = Double(hex & 0xFF) / 255
        }
        self.init(.sRGB, red: r, green: g, blue: b, opacity: a)
    }
}

/// Every color token the desktop styles.css defines, resolved for the active
/// theme. Views read `@Environment(\.sofuuColors)` instead of hardcoding.
struct SofuuColors {
    let dark: Bool
    let bgApp: Color
    let bgAppAlt: Color
    let bgSidebar: Color
    let bgSidebarActive: Color
    let bgCard: Color
    let bgCardHover: Color
    let border: Color
    let borderStrong: Color
    let separator: Color
    let text: Color
    let textMuted: Color
    let textFaint: Color
    let textFaint2: Color
    let accent: Color
    let accentHover: Color
    let accentContrast: Color
    let toggleOff: Color
    let bgRaised: Color
    let monoFieldBg: Color
    let monoFieldText: Color
    let ok: Color
    let warn: Color
    let err: Color
    let errStrong: Color
    let errBg: Color
    let errBorder: Color
    let diffAddBg: Color
    let diffAddFg: Color
    let diffDelBg: Color
    let diffDelFg: Color
    let hoverVeil: Color
    let codeChip: Color
    let codeBlock: Color
    let scrollThumb: Color
    let backdropVeil: Color

    // Light (cream)
    static let light = SofuuColors(
        dark: false,
        bgApp: Color(hex: 0xFFFBF0),
        bgAppAlt: Color(hex: 0xFEF9EC),
        bgSidebar: Color(hex: 0xF2EEDD),
        bgSidebarActive: Color(hex: 0xEEE6D1),
        bgCard: Color(hex: 0xF7F0DE),
        bgCardHover: Color(hex: 0xEFE8D6),
        border: Color(hex: 0xE8E0C9),
        borderStrong: Color(hex: 0xE6DDC3),
        separator: Color(hex: 0xF0E8D0),
        text: Color(hex: 0x1A1A1A),
        textMuted: Color(hex: 0x6B6760),
        textFaint: Color(hex: 0xA8A29A),
        textFaint2: Color(hex: 0x8A857E),
        accent: Color(hex: 0x2B6CB0),
        accentHover: Color(hex: 0x245A94),
        accentContrast: Color(hex: 0xFFFFFF),
        toggleOff: Color(hex: 0xDDD8C8),
        bgRaised: Color(hex: 0xFFFFFF),
        monoFieldBg: Color(hex: 0xFFFFFBFA),
        monoFieldText: Color(hex: 0x5A5752),
        ok: Color(hex: 0x15803D),
        warn: Color(hex: 0xB45309),
        err: Color(hex: 0xB91C1C),
        errStrong: Color(hex: 0xB3362B),
        errBg: Color(hex: 0xFDF3F3),
        errBorder: Color(hex: 0xEECACA),
        // NOTE: the Android light token for this one is a 6-digit value
        // (0x24EA80) which Compose renders fully opaque — almost certainly a
        // typo, since its dark twin (0x242EA043) is a translucent green and
        // the intent is a soft diff wash. iOS uses the intended translucent
        // green here.
        diffAddBg: Color(hex: 0x242EA043),
        diffAddFg: Color(hex: 0x1A7F37),
        diffDelBg: Color(hex: 0x24F85149),
        diffDelFg: Color(hex: 0xCF222E),
        hoverVeil: Color(hex: 0x0D000000),
        codeChip: Color(hex: 0x12000000),
        codeBlock: Color(hex: 0x0F000000),
        scrollThumb: Color(hex: 0x26000000),
        backdropVeil: Color(hex: 0x2E000000))

    // Dark (warm)
    static let dark = SofuuColors(
        dark: true,
        bgApp: Color(hex: 0x211D19),
        bgAppAlt: Color(hex: 0x262219),
        bgSidebar: Color(hex: 0x2A251F),
        bgSidebarActive: Color(hex: 0x3A332A),
        bgCard: Color(hex: 0x2B2620),
        bgCardHover: Color(hex: 0x37312A),
        border: Color(hex: 0x3B352D),
        borderStrong: Color(hex: 0x4A4336),
        separator: Color(hex: 0x332D26),
        text: Color(hex: 0xECE7DC),
        textMuted: Color(hex: 0xA89F8F),
        textFaint: Color(hex: 0x7D766A),
        textFaint2: Color(hex: 0x948C7E),
        accent: Color(hex: 0x84B3E8),
        accentHover: Color(hex: 0xA9CCF1),
        accentContrast: Color(hex: 0x10203A),
        toggleOff: Color(hex: 0x4A4336),
        bgRaised: Color(hex: 0x2E2922),
        monoFieldBg: Color(hex: 0x262119),
        monoFieldText: Color(hex: 0xB5AC9C),
        ok: Color(hex: 0x4CC97A),
        warn: Color(hex: 0xE0A03C),
        err: Color(hex: 0xE57373),
        errStrong: Color(hex: 0xE57373),
        errBg: Color(hex: 0x2AB91C1C),
        errBorder: Color(hex: 0x40B91C1C),
        diffAddBg: Color(hex: 0x242EA043),
        diffAddFg: Color(hex: 0x4CC97A),
        diffDelBg: Color(hex: 0x24F85149),
        diffDelFg: Color(hex: 0xFF7B72),
        hoverVeil: Color(hex: 0x0FFFFFFF),
        codeChip: Color(hex: 0x1AFFFFFF),
        codeBlock: Color(hex: 0x12FFFFFF),
        scrollThumb: Color(hex: 0x38FFFFFF),
        backdropVeil: Color(hex: 0x80000000))
}

private struct SofuuColorsKey: EnvironmentKey {
    static let defaultValue = SofuuColors.light
}

extension EnvironmentValues {
    var sofuuColors: SofuuColors {
        get { self[SofuuColorsKey.self] }
        set { self[SofuuColorsKey.self] = newValue }
    }
}

/// The app's theme root: picks the palette and provides it to the tree.
/// JetBrains Mono is not bundled on iOS (the Android app ships the TTFs);
/// the monospace system face carries the same "terminal, but warm" feel
/// without a font license to ship.
struct SofuuTheme<Content: View>: View {
    let dark: Bool
    @ViewBuilder var content: Content

    var body: some View {
        content
            .environment(\.sofuuColors, dark ? .dark : .light)
            .preferredColorScheme(dark ? .dark : .light)
            .tint((dark ? SofuuColors.dark : SofuuColors.light).accent)
    }
}

/// The monospace face used for code, fields and the composer.
let sofMono = Font.system(.body, design: .monospaced)
