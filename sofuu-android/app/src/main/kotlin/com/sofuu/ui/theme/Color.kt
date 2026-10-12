package com.sofuu.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.sofuu.R

// ── The desktop palette (styles.css :root tokens, cream light +
// [data-theme="dark"] warm dark), ported token-for-token. One
// deliberate divergence: the accent is BLUE here, not the desktop's
// purple, so the UI matches the light-blue cloud background.

// Light (cream)
val BgApp = Color(0xFFFFFBF0)
val BgAppAlt = Color(0xFFFEF9EC)
val BgSidebar = Color(0xFFF2EEDD)
val BgSidebarActive = Color(0xFFEEE6D1)
val BgCard = Color(0xFFF7F0DE)
val BgCardHover = Color(0xFFEFE8D6)
val Border = Color(0xFFE8E0C9)
val BorderStrong = Color(0xFFE6DDC3)
val Separator = Color(0xFFF0E8D0)
val TextPrimary = Color(0xFF1A1A1A)
val TextMuted = Color(0xFF6B6760)
val TextFaint = Color(0xFFA8A29A)
val TextFaint2 = Color(0xFF8A857E)
// Accent — BLUE, deliberately diverging from the desktop's purple
// (#7c2d8a). The cloud background is light blue (see CloudBackground.kt:
// #FEFEFF glow → #DCECFD wash → #A9CCF1 periwinkle → #84B3E8 sky), and a
// purple accent on a blue field read as two competing brands. These hues
// are the cloud palette's own blue family, darkened to stay legible as
// a control: #2B6CB0 is 5.4:1 against white text on cream, and #84B3E8
// is the exact cloud sky blue at 7.6:1 on the warm-dark ground.
val Accent = Color(0xFF2B6CB0)
val AccentHover = Color(0xFF245A94)
val AccentContrast = Color(0xFFFFFFFF)
val ToggleOff = Color(0xFFDDD8C8)
val BgRaised = Color(0xFFFFFFFF)
val MonoFieldBg = Color(0xFFFFFBFA)
val MonoFieldText = Color(0xFF5A5752)
val Ok = Color(0xFF15803D)
val Warn = Color(0xFFB45309)
val Err = Color(0xFFB91C1C)
val ErrStrong = Color(0xFFB3362B)
val ErrBg = Color(0xFFFDF3F3)
val ErrBorder = Color(0xFFEECACA)
val DiffAddBg = Color(0x24EA80)
val DiffAddFg = Color(0xFF1A7F37)
val DiffDelBg = Color(0x24F85149)
val DiffDelFg = Color(0xFFCF222E)
val HoverVeil = Color(0x0D000000)
val CodeChip = Color(0x12000000)
val CodeBlock = Color(0x0F000000)
val ScrollThumb = Color(0x26000000)
val BackdropVeil = Color(0x2E000000)

// Dark (warm)
val DkBgApp = Color(0xFF211D19)
val DkBgAppAlt = Color(0xFF262219)
val DkBgSidebar = Color(0xFF2A251F)
val DkBgSidebarActive = Color(0xFF3A332A)
val DkBgCard = Color(0xFF2B2620)
val DkBgCardHover = Color(0xFF37312A)
val DkBorder = Color(0xFF3B352D)
val DkBorderStrong = Color(0xFF4A4336)
val DkSeparator = Color(0xFF332D26)
val DkTextPrimary = Color(0xFFECE7DC)
val DkTextMuted = Color(0xFFA89F8F)
val DkTextFaint = Color(0xFF7D766A)
val DkTextFaint2 = Color(0xFF948C7E)
val DkAccent = Color(0xFF84B3E8)
val DkAccentHover = Color(0xFFA9CCF1)
val DkAccentContrast = Color(0xFF10203A)
val DkToggleOff = Color(0xFF4A4336)
val DkBgRaised = Color(0xFF2E2922)
val DkMonoFieldBg = Color(0xFF262119)
val DkMonoFieldText = Color(0xFFB5AC9C)
val DkOk = Color(0xFF4CC97A)
val DkWarn = Color(0xFFE0A03C)
val DkErr = Color(0xFFE57373)
val DkErrStrong = Color(0xFFE57373)
val DkErrBg = Color(0x2AB91C1C)
val DkErrBorder = Color(0x40B91C1C)
val DkDiffAddBg = Color(0x242EA043)
val DkDiffAddFg = Color(0xFF4CC97A)
val DkDiffDelBg = Color(0x24F85149)
val DkDiffDelFg = Color(0xFFFF7B72)
val DkHoverVeil = Color(0x0FFFFFFF)
val DkCodeChip = Color(0x1AFFFFFF)
val DkCodeBlock = Color(0x12FFFFFF)
val DkScrollThumb = Color(0x38FFFFFF)
val DkBackdropVeil = Color(0x80000000)

/**
 * Every color token the desktop styles.css defines, resolved for the
 * active theme. Components read `Sofuu.colors.x` instead of hardcoding.
 */
data class SofuuColors(
    val dark: Boolean,
    val bgApp: Color,
    val bgAppAlt: Color,
    val bgSidebar: Color,
    val bgSidebarActive: Color,
    val bgCard: Color,
    val bgCardHover: Color,
    val border: Color,
    val borderStrong: Color,
    val separator: Color,
    val text: Color,
    val textMuted: Color,
    val textFaint: Color,
    val textFaint2: Color,
    val accent: Color,
    val accentHover: Color,
    val accentContrast: Color,
    val toggleOff: Color,
    val bgRaised: Color,
    val monoFieldBg: Color,
    val monoFieldText: Color,
    val ok: Color,
    val warn: Color,
    val err: Color,
    val errStrong: Color,
    val errBg: Color,
    val errBorder: Color,
    val diffAddBg: Color,
    val diffAddFg: Color,
    val diffDelBg: Color,
    val diffDelFg: Color,
    val hoverVeil: Color,
    val codeChip: Color,
    val codeBlock: Color,
    val scrollThumb: Color,
    val backdropVeil: Color,
)

val LightColors = SofuuColors(
    dark = false,
    bgApp = BgApp, bgAppAlt = BgAppAlt, bgSidebar = BgSidebar,
    bgSidebarActive = BgSidebarActive, bgCard = BgCard, bgCardHover = BgCardHover,
    border = Border, borderStrong = BorderStrong, separator = Separator,
    text = TextPrimary, textMuted = TextMuted, textFaint = TextFaint, textFaint2 = TextFaint2,
    accent = Accent, accentHover = AccentHover, accentContrast = AccentContrast,
    toggleOff = ToggleOff, bgRaised = BgRaised, monoFieldBg = MonoFieldBg, monoFieldText = MonoFieldText,
    ok = Ok, warn = Warn, err = Err, errStrong = ErrStrong, errBg = ErrBg, errBorder = ErrBorder,
    diffAddBg = DiffAddBg, diffAddFg = DiffAddFg, diffDelBg = DiffDelBg, diffDelFg = DiffDelFg,
    hoverVeil = HoverVeil, codeChip = CodeChip, codeBlock = CodeBlock,
    scrollThumb = ScrollThumb, backdropVeil = BackdropVeil,
)

val DarkColors = SofuuColors(
    dark = true,
    bgApp = DkBgApp, bgAppAlt = DkBgAppAlt, bgSidebar = DkBgSidebar,
    bgSidebarActive = DkBgSidebarActive, bgCard = DkBgCard, bgCardHover = DkBgCardHover,
    border = DkBorder, borderStrong = DkBorderStrong, separator = DkSeparator,
    text = DkTextPrimary, textMuted = DkTextMuted, textFaint = DkTextFaint, textFaint2 = DkTextFaint2,
    accent = DkAccent, accentHover = DkAccentHover, accentContrast = DkAccentContrast,
    toggleOff = DkToggleOff, bgRaised = DkBgRaised, monoFieldBg = DkMonoFieldBg, monoFieldText = DkMonoFieldText,
    ok = DkOk, warn = DkWarn, err = DkErr, errStrong = DkErrStrong, errBg = DkErrBg, errBorder = DkErrBorder,
    diffAddBg = DkDiffAddBg, diffAddFg = DkDiffAddFg, diffDelBg = DkDiffDelBg, diffDelFg = DkDiffDelFg,
    hoverVeil = DkHoverVeil, codeChip = DkCodeChip, codeBlock = DkCodeBlock,
    scrollThumb = DkScrollThumb, backdropVeil = DkBackdropVeil,
)

// JetBrains Mono everywhere, per the desktop recipe. The variable TTF is
// bundled in res/font (converted from the desktop woff2); static weights
// keep minSdk 24 happy.
val JetBrainsMono = FontFamily(
    Font(R.font.jetbrains_mono_regular, FontWeight.Normal),
    Font(R.font.jetbrains_mono_medium, FontWeight.Medium),
    Font(R.font.jetbrains_mono_bold, FontWeight.Bold),
)

val SofuuTypography = Typography(
    bodyLarge = TextStyle(fontFamily = JetBrainsMono, fontSize = 14.sp, lineHeight = 21.sp),
    bodyMedium = TextStyle(fontFamily = JetBrainsMono, fontSize = 13.sp, lineHeight = 19.sp),
    bodySmall = TextStyle(fontFamily = JetBrainsMono, fontSize = 12.sp, lineHeight = 17.sp),
    titleLarge = TextStyle(fontFamily = JetBrainsMono, fontSize = 17.sp, fontWeight = FontWeight.Medium, lineHeight = 24.sp),
    titleMedium = TextStyle(fontFamily = JetBrainsMono, fontSize = 15.sp, fontWeight = FontWeight.Medium, lineHeight = 21.sp),
    titleSmall = TextStyle(fontFamily = JetBrainsMono, fontSize = 13.sp, fontWeight = FontWeight.Medium, lineHeight = 18.sp),
    labelLarge = TextStyle(fontFamily = JetBrainsMono, fontSize = 13.sp, fontWeight = FontWeight.Medium),
    labelMedium = TextStyle(fontFamily = JetBrainsMono, fontSize = 12.sp, fontWeight = FontWeight.Medium),
    labelSmall = TextStyle(fontFamily = JetBrainsMono, fontSize = 11.sp, fontWeight = FontWeight.Normal),
)

/** Local composition so any composable can read the resolved tokens. */
val LocalSofuuColors = staticCompositionLocalOf { LightColors }

@Composable
fun SofuuTheme(
    dark: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (dark) DarkColors else LightColors
    val scheme = if (dark) {
        darkColorScheme(
            primary = colors.accent,
            onPrimary = colors.accentContrast,
            background = colors.bgApp,
            onBackground = colors.text,
            surface = colors.bgCard,
            onSurface = colors.text,
            surfaceVariant = colors.bgCardHover,
            onSurfaceVariant = colors.textMuted,
            outline = colors.border,
            error = colors.err,
        )
    } else {
        lightColorScheme(
            primary = colors.accent,
            onPrimary = colors.accentContrast,
            background = colors.bgApp,
            onBackground = colors.text,
            surface = colors.bgCard,
            onSurface = colors.text,
            surfaceVariant = colors.bgCardHover,
            onSurfaceVariant = colors.textMuted,
            outline = colors.border,
            error = colors.err,
        )
    }
    androidx.compose.runtime.CompositionLocalProvider(LocalSofuuColors provides colors) {
        MaterialTheme(colorScheme = scheme, typography = SofuuTypography, content = content)
    }
}
