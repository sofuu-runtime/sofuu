/*
 * v2_ui/theme.js — the exact sofuu-android v1 palette, token for token.
 *
 * Source of truth: sofuu-android/.../ui/theme/Color.kt (itself a port of the
 * desktop styles.css cream/warm-dark tokens), so the GPU shell is
 * pixel-comparable with v1 rather than "close enough".
 *
 * Packing matters: sofuu-ui stores colors as little-endian RGBA8
 * (`r | g<<8 | b<<16 | a<<24`), not the 0xRRGGBBAA order Kotlin's Color uses.
 * `rgba()` converts once at load; every token below stays written in the same
 * 0xRRGGBB + alpha form as Color.kt so the two files can be diffed by eye.
 */
(function () {
  'use strict';
  var g = globalThis;

  function rgba(rgb, a) {
    var r = (rgb >> 16) & 0xFF;
    var gr = (rgb >> 8) & 0xFF;
    var b = rgb & 0xFF;
    var al = (a === undefined) ? 255 : a;
    return ((r | (gr << 8) | (b << 16) | (al << 24)) >>> 0);
  }

  var light = {
    dark: false,
    bg: rgba(0xFFFBF0),
    bgAlt: rgba(0xFEF9EC),
    sidebar: rgba(0xF2EEDD),
    sidebarActive: rgba(0xEEE6D1),
    card: rgba(0xF7F0DE),
    cardHover: rgba(0xEFE8D6),
    border: rgba(0xE8E0C9),
    borderStrong: rgba(0xE6DDC3),
    separator: rgba(0xF0E8D0),
    text: rgba(0x1A1A1A),
    muted: rgba(0x6B6760),
    faint: rgba(0xA8A29A),
    faint2: rgba(0x8A857E),
    accent: rgba(0x2B6CB0),
    accentHover: rgba(0x245A94),
    onAccent: rgba(0xFFFFFF),
    toggleOff: rgba(0xDDD8C8),
    raised: rgba(0xFFFFFF),
    monoFieldBg: rgba(0xFFFBFA),
    monoFieldText: rgba(0x5A5752),
    ok: rgba(0x15803D),
    warn: rgba(0xB45309),
    err: rgba(0xB91C1C),
    errStrong: rgba(0xB3362B),
    errBg: rgba(0xFDF3F3),
    errBorder: rgba(0xEECACA),
    hoverVeil: rgba(0x000000, 0x0D),
    codeChip: rgba(0x000000, 0x12),
    codeBlock: rgba(0x000000, 0x0F),
    scrollThumb: rgba(0x000000, 0x26),
    backdropVeil: rgba(0x000000, 0x2E)
  };

  var dark = {
    dark: true,
    bg: rgba(0x211D19),
    bgAlt: rgba(0x262219),
    sidebar: rgba(0x2A251F),
    sidebarActive: rgba(0x3A332A),
    card: rgba(0x2B2620),
    cardHover: rgba(0x37312A),
    border: rgba(0x3B352D),
    borderStrong: rgba(0x4A4336),
    separator: rgba(0x332D26),
    text: rgba(0xECE7DC),
    muted: rgba(0xA89F8F),
    faint: rgba(0x7D766A),
    faint2: rgba(0x948C7E),
    accent: rgba(0x84B3E8),
    accentHover: rgba(0xA9CCF1),
    onAccent: rgba(0x10203A),
    toggleOff: rgba(0x4A4336),
    raised: rgba(0x2E2922),
    monoFieldBg: rgba(0x262119),
    monoFieldText: rgba(0xB5AC9C),
    ok: rgba(0x4CC97A),
    warn: rgba(0xE0A03C),
    err: rgba(0xE57373),
    errStrong: rgba(0xE57373),
    errBg: rgba(0xB91C1C, 0x2A),
    errBorder: rgba(0xB91C1C, 0x40),
    hoverVeil: rgba(0xFFFFFF, 0x0F),
    codeChip: rgba(0xFFFFFF, 0x1A),
    codeBlock: rgba(0xFFFFFF, 0x12),
    scrollThumb: rgba(0xFFFFFF, 0x38),
    backdropVeil: rgba(0x000000, 0x80)
  };

  // v1 runs every sp value through UI_FONT_SCALE = 1.15 (Color.kt). Hold the
  // scaled sizes in one place so screens read `T.type.body` instead of
  // re-deriving 13 * 1.15 by hand (dp layout stays unscaled, exactly like v1).
  var scale = 1.15;
  function sp(v) { return v * scale; }
  var type = {
    displayLarge: sp(32),
    displayMedium: sp(28),
    displaySmall: sp(24),
    headlineLarge: sp(22),
    headlineMedium: sp(20),
    headlineSmall: sp(18),
    bodyLarge: sp(14),
    bodyMedium: sp(13),
    bodySmall: sp(12),
    titleLarge: sp(17),
    titleMedium: sp(15),
    titleSmall: sp(13),
    labelLarge: sp(13),
    labelMedium: sp(12),
    labelSmall: sp(11),
    // Component-local sizes that v1 sets inline (Composer/HomePrompt).
    starterTitle: sp(13.5),
    starterDesc: sp(11.5),
    composerInput: sp(15),
    pill: sp(13)
  };

  g.SofuuTokens = { light: light, dark: dark, type: type, sp: sp, rgba: rgba };
})();
