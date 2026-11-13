//! TUI themes: user-selectable color schemes, independent of the OS or
//! terminal theme.
//!
//! Every color is a fixed 256-palette SGR parameter string — never the
//! terminal-themed 30–37 range — so a theme renders identically under any
//! terminal theme, light or dark. All hues are muted (no neons).
//!
//! Includes both dark themes (233..=238, deep hue-tinted darks) and light
//! themes (soft off-white/paper/cream/mist/stone, paired with explicit dark
//! ink foreground `fg` so unstyled text stays sharp even inside dark-mode
//! terminal emulators).
//!
//! Roles (precomposed with their conventional attributes, e.g. bold for
//! headings/errors and the diff background tints baked into add/del):
//!   accent   model names, titles, the current-step marker
//!   tool     tool names (⏺ lines)
//!   delegate sub-agent names
//!   heading  markdown headings
//!   code     inline code spans
//!   ok       confirmations, checkmarks, active indicators
//!   warn     warnings (answer cap, cautions)
//!   error    errors (tool failures)
//!   add      added diff lines (fg + bg tint — always distinct from bg)
//!   del      removed diff lines (fg + bg tint — always distinct from bg)
//!   hunk     @@ hunk headers
//!   panel    welcome-panel and input box borders
//!   subtle   hints, descriptions, secondary metrics
//!
//! The default theme ("sofuu", bg 235) is a faithful port of the
//! pre-theme look, so upgrading changes nothing until the user runs
//! /theme.

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub dark: bool,
    /// Window background, 256-palette index (dark themes 235, light 255).
    pub bg: u8,
    /// Default text ink, 256-palette index (-1 keeps terminal fg for dark themes;
    /// light themes carry explicit charcoal ink like 234 so dark terminals stay legible).
    pub fg: i16,
    pub accent: &'static str,
    pub tool: &'static str,
    pub delegate: &'static str,
    pub heading: &'static str,
    pub code: &'static str,
    pub ok: &'static str,
    pub warn: &'static str,
    pub error: &'static str,
    pub add: &'static str,
    pub del: &'static str,
    pub hunk: &'static str,
    pub panel: &'static str,
    pub subtle: &'static str,
}

pub const DEFAULT_THEME: &str = "sofuu";

/// All 18 shipped themes: 13 dark themes + 5 light themes.
pub const THEMES: &[Theme] = &[
    // ── dark ──────────────────────────────────────────────────────
    Theme { name: "sofuu", dark: true, bg: 235, fg: -1, // faithful port of the classic look
        accent: "1;38;5;139", tool: "38;5;80", delegate: "38;5;140",
        heading: "1;38;5;80", code: "38;5;180", ok: "38;5;114", warn: "38;5;173",
        error: "1;38;5;174", add: "1;38;5;157;48;5;28", del: "1;38;5;174;48;5;52",
        hunk: "38;5;80", panel: "2;38;5;139", subtle: "38;5;246" },
    Theme { name: "slate", dark: true, bg: 236, fg: -1, // blue-gray monochrome calm
        accent: "1;38;5;110", tool: "38;5;109", delegate: "38;5;103",
        heading: "1;38;5;110", code: "38;5;186", ok: "38;5;114", warn: "38;5;179",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;109", panel: "2;38;5;110", subtle: "38;5;152" },
    Theme { name: "moss", dark: true, bg: 22, fg: -1, // green-dominant, warm gray accents
        accent: "1;38;5;150", tool: "38;5;108", delegate: "38;5;144",
        heading: "1;38;5;150", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;28", del: "1;38;5;181;48;5;52",
        hunk: "38;5;108", panel: "2;38;5;150", subtle: "38;5;151" },
    Theme { name: "clay", dark: true, bg: 94, fg: -1, // terracotta warmth
        accent: "1;38;5;173", tool: "38;5;180", delegate: "38;5;174",
        heading: "1;38;5;173", code: "38;5;186", ok: "38;5;114", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;180", panel: "2;38;5;173", subtle: "38;5;187" },
    Theme { name: "plum", dark: true, bg: 53, fg: -1, // muted purple, low saturation
        accent: "1;38;5;140", tool: "38;5;139", delegate: "38;5;183",
        heading: "1;38;5;140", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;52",
        hunk: "38;5;139", panel: "2;38;5;140", subtle: "38;5;182" },
    Theme { name: "tide", dark: true, bg: 23, fg: -1, // deep teal water
        accent: "1;38;5;115", tool: "38;5;80", delegate: "38;5;109",
        heading: "1;38;5;115", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;29", del: "1;38;5;174;48;5;52",
        hunk: "38;5;80", panel: "2;38;5;115", subtle: "38;5;116" },
    Theme { name: "ember", dark: true, bg: 233, fg: -1, // ember orange on charcoal
        accent: "1;38;5;180", tool: "38;5;173", delegate: "38;5;174",
        heading: "1;38;5;180", code: "38;5;186", ok: "38;5;114", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;52",
        hunk: "38;5;173", panel: "2;38;5;180", subtle: "38;5;245" },
    Theme { name: "dusk", dark: true, bg: 54, fg: -1, // dusky blue-violet evening
        accent: "1;38;5;140", tool: "38;5;110", delegate: "38;5;139",
        heading: "1;38;5;140", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;110", panel: "2;38;5;140", subtle: "38;5;189" },
    Theme { name: "forest", dark: true, bg: 22, fg: -1, // deep pine, bark browns
        accent: "1;38;5;108", tool: "38;5;150", delegate: "38;5;144",
        heading: "1;38;5;108", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;28", del: "1;38;5;181;48;5;88",
        hunk: "38;5;150", panel: "2;38;5;108", subtle: "38;5;107" },
    Theme { name: "wine", dark: true, bg: 52, fg: -1, // muted oxblood red family
        accent: "1;38;5;174", tool: "38;5;181", delegate: "38;5;139",
        heading: "1;38;5;174", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;181;48;5;88",
        hunk: "38;5;181", panel: "2;38;5;174", subtle: "38;5;181" },
    Theme { name: "storm", dark: true, bg: 238, fg: -1, // cold steel gray-blue
        accent: "1;38;5;109", tool: "38;5;110", delegate: "38;5;103",
        heading: "1;38;5;109", code: "38;5;186", ok: "38;5;114", warn: "38;5;179",
        error: "1;38;5;174", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;110", panel: "2;38;5;109", subtle: "38;5;249" },
    Theme { name: "honey", dark: true, bg: 94, fg: -1, // warm amber, softened gold
        accent: "1;38;5;180", tool: "38;5;186", delegate: "38;5;173",
        heading: "1;38;5;180", code: "38;5;223", ok: "38;5;114", warn: "38;5;179",
        error: "1;38;5;167", add: "1;38;5;151;48;5;22", del: "1;38;5;174;48;5;52",
        hunk: "38;5;186", panel: "2;38;5;180", subtle: "38;5;223" },
    Theme { name: "lagoon", dark: true, bg: 24, fg: -1, // pale aqua on deep slate
        accent: "1;38;5;116", tool: "38;5;115", delegate: "38;5;110",
        heading: "1;38;5;116", code: "38;5;186", ok: "38;5;114", warn: "38;5;180",
        error: "1;38;5;174", add: "1;38;5;157;48;5;29", del: "1;38;5;181;48;5;52",
        hunk: "38;5;115", panel: "2;38;5;116", subtle: "38;5;153" },
    // ── light ─────────────────────────────────────────────────────
    Theme { name: "paper", dark: false, bg: 255, fg: 234, // neutral off-white daylight
        accent: "1;38;5;54", tool: "38;5;30", delegate: "38;5;55",
        heading: "1;38;5;24", code: "38;5;130", ok: "38;5;28", warn: "38;5;166",
        error: "1;38;5;124", add: "1;38;5;22;48;5;194", del: "1;38;5;88;48;5;224",
        hunk: "38;5;31", panel: "38;5;96", subtle: "38;5;241" },
    Theme { name: "snow", dark: false, bg: 231, fg: 233, // crisp editorial pure white
        accent: "1;38;5;25", tool: "38;5;31", delegate: "38;5;61",
        heading: "1;38;5;25", code: "38;5;130", ok: "38;5;28", warn: "38;5;166",
        error: "1;38;5;160", add: "1;38;5;22;48;5;194", del: "1;38;5;88;48;5;224",
        hunk: "38;5;25", panel: "38;5;67", subtle: "38;5;242" },
    Theme { name: "ivory", dark: false, bg: 230, fg: 235, // warm cream book paper
        accent: "1;38;5;94", tool: "38;5;29", delegate: "38;5;95",
        heading: "1;38;5;94", code: "38;5;131", ok: "38;5;22", warn: "38;5;130",
        error: "1;38;5;124", add: "1;38;5;22;48;5;194", del: "1;38;5;88;48;5;224",
        hunk: "38;5;29", panel: "38;5;137", subtle: "38;5;101" },
    Theme { name: "frost", dark: false, bg: 195, fg: 234, // cool ice-blue daylight mist
        accent: "1;38;5;24", tool: "38;5;23", delegate: "38;5;60",
        heading: "1;38;5;24", code: "38;5;94", ok: "38;5;28", warn: "38;5;130",
        error: "1;38;5;124", add: "1;38;5;22;48;5;194", del: "1;38;5;88;48;5;224",
        hunk: "38;5;24", panel: "38;5;66", subtle: "38;5;60" },
    Theme { name: "dawn", dark: false, bg: 254, fg: 233, // calm neutral stone-gray
        accent: "1;38;5;89", tool: "38;5;30", delegate: "38;5;53",
        heading: "1;38;5;24", code: "38;5;130", ok: "38;5;28", warn: "38;5;166",
        error: "1;38;5;124", add: "1;38;5;22;48;5;194", del: "1;38;5;88;48;5;224",
        hunk: "38;5;31", panel: "38;5;95", subtle: "38;5;240" },
];

/// Palette index of a role's `38;5;N` foreground — feeds the selection
/// bar, which wears the theme accent (opencode-style), so a mouse drag
/// blends on every present and future theme without a table entry.
/// None when the params carry no 256-palette foreground.
pub fn accent_index(params: &str) -> Option<u8> {
    let mut it = params.split(';').peekable();
    while let Some(p) = it.next() {
        if p.trim() == "38" && it.peek() == Some(&"5") {
            it.next();
            if let Some(n) = it.next() {
                if let Ok(v) = n.trim().parse::<i32>() {
                    if (16..=255).contains(&v) {
                        return Some(v as u8);
                    }
                }
            }
            return None;
        }
    }
    None
}

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

/// Foreground remap from one theme to another: (old_params, new_params)
/// per role, longest old-params first.
///
/// Applied to already-painted buffer rows on a theme switch so the old
/// transcript flips foregrounds along with the window background instead
/// of going unreadable across a dark/light switch. Roles pair by position
/// (accent→accent … panel→panel); pairs with identical params are dropped.
/// Matching is on the full escape (`\x1b[<params>m`), never a bare
/// substring, and the two themes' param sets are disjoint by construction
/// (dark roles never use light values), so one pass cannot double-remap.
/// Caveat, documented not hidden: rows from tools/models may carry raw
/// ANSI that happens to equal an old role sequence — that span remaps too
/// (display-only; the session transcript underneath is untouched).
pub fn fg_remap_pairs(from: &Theme, to: &Theme) -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = vec![
        (from.accent, to.accent),
        (from.tool, to.tool),
        (from.delegate, to.delegate),
        (from.heading, to.heading),
        (from.code, to.code),
        (from.ok, to.ok),
        (from.warn, to.warn),
        (from.error, to.error),
        (from.add, to.add),
        (from.del, to.del),
        (from.hunk, to.hunk),
        (from.panel, to.panel),
        (from.subtle, to.subtle),
    ];
    v.retain(|(a, b)| a != b);
    v.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The contract: 18 themes (13 dark + 5 light), unique names, every
    /// slot a well-formed 256-palette SGR param string.
    #[test]
    fn theme_registry_shape() {
        assert_eq!(THEMES.len(), 18, "must ship 18 themes (13 dark + 5 light)");
        let names: HashSet<&str> = THEMES.iter().map(|t| t.name).collect();
        assert_eq!(names.len(), 18, "theme names must be unique");
        assert_eq!(THEMES[0].name, DEFAULT_THEME);
        assert_eq!(THEMES.iter().filter(|t| t.dark).count(), 13);
        assert_eq!(THEMES.iter().filter(|t| !t.dark).count(), 5);
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
                ("ok", t.ok),
                ("warn", t.warn),
                ("error", t.error),
                ("add", t.add),
                ("del", t.del),
                ("hunk", t.hunk),
                ("panel", t.panel),
                ("subtle", t.subtle),
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
        assert_eq!(lookup("paper").name, "paper");
        assert_eq!(lookup("PAPER").name, "paper");
        assert_eq!(lookup("no-such-theme").name, DEFAULT_THEME);
        assert!(resolve("lagoon").is_some());
        assert!(resolve("snow").is_some());
        assert!(resolve("nope").is_none());
    }

    /// Selection-bar derivation: the bar wears the accent's palette
    /// index (bg) with near-black text (dark themes) or white text (light themes).
    #[test]
    fn accent_index_finds_the_256_foreground() {
        assert_eq!(accent_index("1;38;5;139"), Some(139));
        assert_eq!(accent_index("38;5;29"), Some(29));
        assert_eq!(accent_index("2;38;5;110"), Some(110));
        assert_eq!(accent_index("1;35"), None);
        assert_eq!(accent_index(""), None);
        assert_eq!(accent_index("38;5;7"), None);
        assert_eq!(accent_index("38;5;300"), None);
        // Every shipped accent must derive (else that theme gets no bar).
        for t in THEMES {
            assert!(
                accent_index(t.accent).is_some(),
                "theme {} accent {:?} must carry 38;5;N",
                t.name, t.accent
            );
        }
    }

    /// Window-background contract: dark themes carry curated dark bg + fg=-1;
    /// light themes carry curated light bg + explicit charcoal ink fg (232..=236).
    /// Plus no theme's add/del bg tint may equal its own window bg.
    #[test]
    fn theme_backgrounds_are_dark_and_tints_stay_visible() {
        const DARK_BGS: [u8; 13] = [22, 23, 24, 52, 53, 54, 94, 233, 234, 235, 236, 237, 238];
        const LIGHT_BGS: [u8; 5] = [195, 230, 231, 254, 255];
        for t in THEMES {
            if t.dark {
                assert!(
                    DARK_BGS.contains(&t.bg),
                    "dark theme {} must carry a curated dark bg, got {}",
                    t.name, t.bg
                );
                assert_eq!(t.fg, -1, "dark theme {} keeps terminal default fg", t.name);
            } else {
                assert!(
                    LIGHT_BGS.contains(&t.bg),
                    "light theme {} must carry a curated light bg, got {}",
                    t.name, t.bg
                );
                assert!(
                    (232..=236).contains(&t.fg),
                    "light theme {} must set dark charcoal ink fg (232..=236), got {}",
                    t.name, t.fg
                );
                assert!(
                    !t.panel.starts_with("2;") && !t.subtle.starts_with("2;"),
                    "light theme {} must avoid terminal dim SGR 2 on panel/subtle",
                    t.name
                );
            }
            for (role, v) in [("add", t.add), ("del", t.del)] {
                // The bg tint is the trailing `48;5;N` of the role params.
                let tint: u8 = v
                    .rsplit(';')
                    .next()
                    .and_then(|n| n.parse().ok())
                    .expect("role params end in a tint index");
                assert!(
                    tint != t.bg,
                    "theme {} role {role}: tint {tint} would vanish on bg {}",
                    t.name, t.bg
                );
            }
        }
        // The default stays the classic neutral so upgrading changes nothing.
        assert_eq!(lookup("sofuu").bg, 235);
        assert!(!lookup("paper").dark);
    }

    /// The remap used on a switch: role pairs, longest-first, disjoint
    /// param sets (one pass cannot double-remap).
    #[test]
    fn remap_pairs_are_longest_first_and_disjoint() {
        let from = lookup("slate");
        let to = lookup("moss");
        let pairs = fg_remap_pairs(from, to);
        assert!(!pairs.is_empty(), "a theme switch must remap roles");
        for w in pairs.windows(2) {
            assert!(
                w[0].0.len() >= w[1].0.len(),
                "pairs must be longest-old-first"
            );
        }
        // Check dark -> light yields pairs for every distinct role too.
        let dl = fg_remap_pairs(lookup("sofuu"), lookup("paper"));
        assert!(!dl.is_empty(), "dark->light must remap roles");
        // Same-theme remap is empty (nothing to do).
        assert!(fg_remap_pairs(from, from).is_empty());
    }
}
