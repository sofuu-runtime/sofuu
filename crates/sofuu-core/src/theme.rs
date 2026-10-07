//! TUI themes: user-selectable color schemes, independent of the OS or
//! terminal theme.
//!
//! Every color is a fixed 256-palette SGR parameter string — never the
//! terminal-themed 30–37 range — so a theme renders identically under any
//! terminal theme, light or dark. All hues are muted (no neons); each
//! theme is tagged dark/light for the background it was designed against.
//!
//! Roles (precomposed with their conventional attributes, e.g. bold for
//! headings/errors and the diff background tints baked into add/del):
//!   accent   model names, titles, the current-step marker
//!   tool     tool names (⏺ lines)
//!   delegate sub-agent names
//!   heading  markdown headings
//!   code     inline code spans
//!   warn     warnings (answer cap, cautions)
//!   error    errors (tool failures)
//!   add      added diff lines (fg + bg tint)
//!   del      removed diff lines (fg + bg tint)
//!   hunk     @@ hunk headers
//!   panel    welcome-panel borders (dim + accent hue)
//!
//! The default theme ("sofuu", dark) is a faithful port of the
//! pre-theme look, so upgrading changes nothing until the user runs
//! /theme.

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub dark: bool,
    pub accent: &'static str,
    pub tool: &'static str,
    pub delegate: &'static str,
    pub heading: &'static str,
    pub code: &'static str,
    pub warn: &'static str,
    pub error: &'static str,
    pub add: &'static str,
    pub del: &'static str,
    pub hunk: &'static str,
    pub panel: &'static str,
}

pub const DEFAULT_THEME: &str = "sofuu";

/// All 25 themes: 13 designed for dark terminals, 12 for light ones.
pub const THEMES: &[Theme] = &[
    // ── dark ──────────────────────────────────────────────────────
    Theme { name: "sofuu", dark: true, // faithful port of the classic look
        accent: "1;38;5;139", tool: "38;5;80", delegate: "38;5;140",
        heading: "1;38;5;80", code: "38;5;180", warn: "38;5;173",
        error: "1;38;5;174", add: "1;38;5;157;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;80", panel: "2;38;5;139" },
    Theme { name: "slate", dark: true, // blue-gray monochrome calm
        accent: "1;38;5;110", tool: "38;5;109", delegate: "38;5;103",
        heading: "1;38;5;110", code: "38;5;186", warn: "38;5;179",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;109", panel: "2;38;5;110" },
    Theme { name: "moss", dark: true, // green-dominant, warm gray accents
        accent: "1;38;5;150", tool: "38;5;108", delegate: "38;5;144",
        heading: "1;38;5;150", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;22", del: "1;38;5;181;48;5;52",
        hunk: "38;5;108", panel: "2;38;5;150" },
    Theme { name: "clay", dark: true, // terracotta warmth
        accent: "1;38;5;173", tool: "38;5;180", delegate: "38;5;174",
        heading: "1;38;5;173", code: "38;5;186", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;180", panel: "2;38;5;173" },
    Theme { name: "plum", dark: true, // muted purple, low saturation
        accent: "1;38;5;140", tool: "38;5;139", delegate: "38;5;183",
        heading: "1;38;5;140", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;52",
        hunk: "38;5;139", panel: "2;38;5;140" },
    Theme { name: "tide", dark: true, // deep teal water
        accent: "1;38;5;115", tool: "38;5;80", delegate: "38;5;109",
        heading: "1;38;5;115", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;23", del: "1;38;5;174;48;5;52",
        hunk: "38;5;80", panel: "2;38;5;115" },
    Theme { name: "ember", dark: true, // ember orange on charcoal
        accent: "1;38;5;180", tool: "38;5;173", delegate: "38;5;174",
        heading: "1;38;5;180", code: "38;5;186", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;52",
        hunk: "38;5;173", panel: "2;38;5;180" },
    Theme { name: "dusk", dark: true, // dusky blue-violet evening
        accent: "1;38;5;140", tool: "38;5;110", delegate: "38;5;139",
        heading: "1;38;5;140", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;110", panel: "2;38;5;140" },
    Theme { name: "forest", dark: true, // deep pine, bark browns
        accent: "1;38;5;108", tool: "38;5;150", delegate: "38;5;144",
        heading: "1;38;5;108", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;22", del: "1;38;5;181;48;5;88",
        hunk: "38;5;150", panel: "2;38;5;108" },
    Theme { name: "wine", dark: true, // muted oxblood red family
        accent: "1;38;5;174", tool: "38;5;181", delegate: "38;5;139",
        heading: "1;38;5;174", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;88",
        hunk: "38;5;181", panel: "2;38;5;174" },
    Theme { name: "storm", dark: true, // cold steel gray-blue
        accent: "1;38;5;109", tool: "38;5;110", delegate: "38;5;103",
        heading: "1;38;5;109", code: "38;5;186", warn: "38;5;179",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;110", panel: "2;38;5;109" },
    Theme { name: "honey", dark: true, // warm amber, softened gold
        accent: "1;38;5;180", tool: "38;5;186", delegate: "38;5;173",
        heading: "1;38;5;180", code: "38;5;223", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;186", panel: "2;38;5;180" },
    Theme { name: "lagoon", dark: true, // pale aqua on deep slate
        accent: "1;38;5;116", tool: "38;5;115", delegate: "38;5;110",
        heading: "1;38;5;116", code: "38;5;186", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;23", del: "1;38;5;181;48;5;52",
        hunk: "38;5;115", panel: "2;38;5;116" },
    // ── light ─────────────────────────────────────────────────────
    // Light themes use dark foregrounds (readable on pale backgrounds)
    // and light tints behind diff rows (dark text stays legible).
    Theme { name: "paper", dark: false, // ink on paper, restrained blue
        accent: "1;38;5;25", tool: "38;5;29", delegate: "38;5;95",
        heading: "1;38;5;25", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;29", panel: "2;38;5;25" },
    Theme { name: "linen", dark: false, // warm neutral, umber ink
        accent: "1;38;5;95", tool: "38;5;100", delegate: "38;5;130",
        heading: "1;38;5;95", code: "38;5;136", warn: "38;5;130",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;100", panel: "2;38;5;95" },
    Theme { name: "porcelain", dark: false, // cool celadon on white
        accent: "1;38;5;29", tool: "38;5;24", delegate: "38;5;66",
        heading: "1;38;5;29", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;24", panel: "2;38;5;29" },
    Theme { name: "wheat", dark: false, // harvest golds, toasted brown
        accent: "1;38;5;130", tool: "38;5;136", delegate: "38;5;95",
        heading: "1;38;5;130", code: "38;5;100", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;136", panel: "2;38;5;130" },
    Theme { name: "mist", dark: false, // fog gray-blue, quiet
        accent: "1;38;5;60", tool: "38;5;66", delegate: "38;5;95",
        heading: "1;38;5;60", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;66", panel: "2;38;5;60" },
    Theme { name: "bone", dark: false, // pale neutral, graphite ink
        accent: "1;38;5;58", tool: "38;5;59", delegate: "38;5;95",
        heading: "1;38;5;58", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;59", panel: "2;38;5;58" },
    Theme { name: "sage-light", dark: false, // pale garden green-gray
        accent: "1;38;5;64", tool: "38;5;29", delegate: "38;5;100",
        heading: "1;38;5;64", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;29", panel: "2;38;5;64" },
    Theme { name: "clay-light", dark: false, // sunbaked terracotta tint
        accent: "1;38;5;131", tool: "38;5;130", delegate: "38;5;95",
        heading: "1;38;5;131", code: "38;5;100", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;130", panel: "2;38;5;131" },
    Theme { name: "ink", dark: false, // maximal contrast, navy ink
        accent: "1;38;5;18", tool: "38;5;24", delegate: "38;5;90",
        heading: "1;38;5;18", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;24", panel: "2;38;5;18" },
    Theme { name: "rose-light", dark: false, // blush, muted cranberry
        accent: "1;38;5;131", tool: "38;5;95", delegate: "38;5;60",
        heading: "1;38;5;131", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;95", panel: "2;38;5;131" },
    Theme { name: "moss-light", dark: false, // pale lichen, deep pine ink
        accent: "1;38;5;29", tool: "38;5;64", delegate: "38;5;66",
        heading: "1;38;5;29", code: "38;5;130", warn: "38;5;136",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;64", panel: "2;38;5;29" },
    Theme { name: "dune", dark: false, // desert sand, dark bronze
        accent: "1;38;5;136", tool: "38;5;100", delegate: "38;5;95",
        heading: "1;38;5;136", code: "38;5;64", warn: "38;5;130",
        error: "1;38;5;124", add: "38;5;28;48;5;194", del: "38;5;124;48;5;224",
        hunk: "38;5;100", panel: "2;38;5;136" },
];

/// Look up a theme by name (case-insensitive); unknown names fall back
/// to the default so a typo in config can never unstyle the TUI.
pub fn lookup(name: &str) -> &'static Theme {
    let want = name.trim().to_lowercase();
    for t in THEMES {
        if t.name == want {
            return t;
        }
    }
    &THEMES[0]
}

/// A theme name is only accepted if it resolves to a real theme (not the
/// fallback) — used when applying /theme so typos are reported, not
/// silently mapped to the default.
pub fn resolve(name: &str) -> Option<&'static Theme> {
    let want = name.trim().to_lowercase();
    THEMES.iter().find(|t| t.name == want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The contract: exactly 25 themes, unique names, a dark/light split,
    /// every slot a well-formed 256-palette SGR param string.
    #[test]
    fn theme_registry_shape() {
        assert_eq!(THEMES.len(), 25, "must ship exactly 25 themes");
        let names: HashSet<&str> = THEMES.iter().map(|t| t.name).collect();
        assert_eq!(names.len(), 25, "theme names must be unique");
        assert_eq!(THEMES[0].name, DEFAULT_THEME);
        let dark = THEMES.iter().filter(|t| t.dark).count();
        let light = THEMES.iter().filter(|t| !t.dark).count();
        assert!(dark > 0 && light > 0, "need both dark and light themes");
        assert_eq!(dark + light, 25);
    }

    fn valid_params(s: &str) -> bool {
        // semicolon-separated ints, each 0-255; color indexes must be
        // 256-palette (16-255) or base SGR attributes — never the
        // terminal-themed 30-37/90-97 foreground range, which would make
        // the theme follow the terminal instead of overriding it.
        if s.is_empty() {
            return false;
        }
        let parts: Vec<&str> = s.split(';').collect();
        // attr + fg-extended + bg-extended is 7 parts max
        // ("1;38;5;157;48;5;22"); anything longer is malformed.
        if parts.is_empty() || parts.len() > 7 {
            return false;
        }
        let mut i = 0;
        while i < parts.len() {
            let n: i32 = match parts[i].parse() {
                Ok(v) => v,
                Err(_) => return false,
            };
            if n == 38 || n == 48 {
                // extended color: 38;5;N or 48;5;N — N must be 256-palette
                if i + 2 >= parts.len() {
                    return false;
                }
                if parts[i + 1] != "5" {
                    return false;
                }
                let c: i32 = match parts[i + 2].parse() {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if c < 16 || c > 255 {
                    return false;
                }
                i += 3;
                continue;
            }
            // bare attribute (0/1/2/3/4/7) or base color 30-47 — the 30-37
            // foregrounds are terminal-themed, but they only ever appear
            // here as part of... (none should; flag them)
            if (30..=37).contains(&n) || (90..=97).contains(&n) {
                return false;
            }
            if !(0..=8).contains(&n) && !(40..=47).contains(&n) {
                return false;
            }
            i += 1;
        }
        true
    }

    #[test]
    fn theme_colors_are_terminal_independent() {
        for t in THEMES {
            for (role, v) in [
                ("accent", t.accent),
                ("tool", t.tool),
                ("delegate", t.delegate),
                ("heading", t.heading),
                ("code", t.code),
                ("warn", t.warn),
                ("error", t.error),
                ("add", t.add),
                ("del", t.del),
                ("hunk", t.hunk),
                ("panel", t.panel),
            ] {
                assert!(
                    valid_params(v),
                    "theme {} role {role} has non-256 params: {v:?}",
                    t.name
                );
            }
        }
    }

    #[test]
    fn theme_lookup_falls_back_and_resolves() {
        assert_eq!(lookup("slate").name, "slate");
        assert_eq!(lookup("SLATE").name, "slate");
        assert_eq!(lookup("no-such-theme").name, DEFAULT_THEME);
        assert!(resolve("moss-light").is_some());
        assert!(resolve("nope").is_none());
    }
}
