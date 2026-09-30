// ml/freshness/features.rs — the freshness feature extractor
// (PLAN-ML-GATES §5). ONE implementation shared by the trainer and the
// runtime (the trainer crate depends on this module), so train/serve skew
// is structurally impossible (§4).
//
// 28 scalar f32 features. Design rule learned from the first training
// attempt: whole-document anchor cosines are too DILUTED to carry the
// implicit-staleness signal (a "deprecated" phrase inside a paragraph
// barely moves the document's trigram vector). The load-bearing features
// are the lexical scans the plan always described — keyword-hit densities
// for the stale/hedge/fresh/legacy vocabularies — plus the year machinery
// and sentence-max anchor cosines as a secondary semantic channel. Each
// computed feature costs ZERO parameters, so the 8.7k budget is spent on
// depth, not on a 768-dim input layer.

use std::sync::LazyLock;

use crate::rt::ai::sofuu_tfidf_embed;

pub const FRESHNESS_FEATURES: usize = 28;
const EMBED_DIM: usize = 768;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceKind {
    Web,
    Memory,
    Tool,
    File,
}

impl SourceKind {
    fn one_hots(self) -> [f32; 4] {
        match self {
            SourceKind::Web => [1.0, 0.0, 0.0, 0.0],
            SourceKind::Memory => [0.0, 1.0, 0.0, 0.0],
            SourceKind::Tool => [0.0, 0.0, 1.0, 0.0],
            SourceKind::File => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

pub struct FreshnessInput<'a> {
    pub text: &'a str,
    pub task: &'a str,
    pub kind: SourceKind,
    /// Brain record strength 0..1 (0 when not a brain record).
    pub strength: f32,
    /// Record age in days (0 when not a brain record).
    pub age_days: f32,
    /// Calendar year "now" — injected so eval fixtures are deterministic.
    pub now_year: u32,
}

/* ── Keyword vocabularies — the primary implicit-staleness channel ── */

const STALE_KEYWORDS: &[&str] = &[
    "deprecated", "obsolete", "no longer maintained", "no longer supported",
    "unmaintained", "archived", "end of life", "discontinued", "retired",
    "abandoned", "superseded", "yanked", "sunset", "dead code", "support ended",
    "development halted", "development stopped", "kept for reference",
    "phased out", "removed from the current", "switched off",
    "recommend a drop-in replacement", "receives no", "no further patches",
];

const HEDGE_KEYWORDS: &[&str] = &[
    "may be outdated", "might have changed", "possibly stale", "verify before",
    "could be newer", "accurate when published", "treat as stale", "could lag",
    "earlier review", "re-verify", "as of last check", "at the time of writing",
    "whether they still apply", "drifted since", "not be current",
    "might not be current", "may not be current",
];

const FRESH_KEYWORDS: &[&str] = &[
    "just shipped", "announced today", "latest release", "now available",
    "actively developed", "new stable", "rolled out", "fresh out",
    "most recent", "just published", "current generation", "this week",
    "this month", "this morning", "brand new", "weekly releases",
    "rapid patch", "new build", "newest update", "new version",
];

const LEGACY_KEYWORDS: &[&str] = &[
    "legacy", "previous generation", "retired module", "deprecated call",
    "obsolete configuration", "older release line", "older series",
    "earlier major version", "superseded by the current", "phased out upstream",
    "compatibility with the older",
];

fn keyword_density(text: &str, keywords: &[&str]) -> f32 {
    let lower = text.to_lowercase();
    let mut hits = 0.0f32;
    for w in keywords {
        hits += lower.matches(w).count() as f32;
    }
    // Normalize per ~200 chars and saturate: one clear hit in a short
    // block is a full signal; long documents need proportionally more.
    (hits / (text.len() as f32 / 200.0 + 1.0)).min(1.0)
}

/* ── Anchor embedder (secondary semantic channel, sentence-max) ──── */

const STALE_ANCHOR: &str = "deprecated obsolete outdated stale legacy no longer supported \
    removed discontinued sunset end of life unmaintained archived abandoned retired \
    superseded replaced by old version previous version earlier version";
const HEDGING_ANCHOR: &str = "as of might have changed may be outdated possibly old \
    at the time of writing could be newer verify check whether still valid last checked \
    information may not be current";
const TIME_SENSITIVE_TASK: &str = "latest newest current version today now recent release \
    update upgrade what is new best currently which version how much does it cost now \
    what does it charge this year pricing this month status \
    price stock weather news who is the president right now";

fn embed_anchor(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(text.as_bytes(), &mut v, EMBED_DIM);
    v
}

static STALE_VEC: LazyLock<Vec<f32>> = LazyLock::new(|| embed_anchor(STALE_ANCHOR));
static HEDGING_VEC: LazyLock<Vec<f32>> = LazyLock::new(|| embed_anchor(HEDGING_ANCHOR));
static TIME_TASK_VEC: LazyLock<Vec<f32>> = LazyLock::new(|| embed_anchor(TIME_SENSITIVE_TASK));

fn embed_text(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(text.as_bytes(), &mut v, EMBED_DIM);
    v
}

/// Scalar cosine, fixed accumulation order — bit-identical everywhere
/// (deliberately NOT the SIMD kernel: the gates must be platform-exact).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Max cosine of any sentence to an anchor — concentrates a short
/// "deprecated" phrase that a whole-document embedding would dilute.
fn sentence_max_cosine(text: &str, anchor: &[f32]) -> f32 {
    let mut best = 0.0f32;
    for sent in text.split(['.', '!', '?', '\n', ';']) {
        let s = sent.trim();
        if s.len() < 15 {
            continue;
        }
        let v = embed_text(s);
        let c = cosine(&v, anchor);
        if c > best {
            best = c;
        }
    }
    best
}

/* ── Lexical scanners (pure, deterministic, no regex dependency) ─── */

/// Maximal 4-digit runs in 1900..=2100 → (byte offset, year). Maximal-run
/// rule keeps "12345" from matching and is trivially deterministic. Byte
/// offsets let callers reason about proximity without slicing UTF-8.
pub fn year_positions(text: &str) -> Vec<(usize, u32)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i - start == 4 {
                let v = b[start..i]
                    .iter()
                    .fold(0u32, |acc, &c| acc * 10 + (c - b'0') as u32);
                if (1900..=2100).contains(&v) {
                    out.push((start, v));
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

pub fn scan_years(text: &str) -> Vec<u32> {
    year_positions(text).into_iter().map(|(_, y)| y).collect()
}

const MONTHS: &[&str] = &[
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
];

/// Explicit calendar dates: YYYY-MM-DD, "Month DD, YYYY", "DD Month YYYY",
/// "Month YYYY". Cheap scans, no regex, UTF-8-safe (ASCII byte scans only —
/// never slices the string at computed offsets).
pub fn has_explicit_date(text: &str) -> bool {
    let b = text.as_bytes();
    // YYYY-MM-DD
    for i in 0..b.len().saturating_sub(9) {
        if b[i].is_ascii_digit()
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
            && b[i + 3].is_ascii_digit()
            && b[i + 4] == b'-'
            && b[i + 5].is_ascii_digit()
            && b[i + 7] == b'-'
            && b[i + 8].is_ascii_digit()
        {
            return true;
        }
    }
    // Month-name proximity: a month word with a 4-digit year within 12
    // bytes. ASCII-fold the text (byte positions preserved exactly —
    // to_lowercase() on a String can change byte lengths for non-ASCII).
    let lower: Vec<u8> = b.iter().map(|c| c.to_ascii_lowercase()).collect();
    let years = year_positions(text);
    for m in MONTHS {
        let mb = m.as_bytes();
        let mut from = 0usize;
        while from + mb.len() <= lower.len() {
            let Some(rel) = lower[from..].windows(mb.len()).position(|w| w == mb) else {
                break;
            };
            let p = from + rel;
            let lo = p.saturating_sub(12);
            let hi = p + mb.len() + 12;
            for &(pos, _) in &years {
                if pos >= lo && pos <= hi {
                    return true;
                }
            }
            from = p + mb.len();
        }
    }
    false
}

/// Version strings: digit '.' digit (e.g. 3.11, 1.4.2, v2.0).
pub fn has_version_string(text: &str) -> bool {
    let b = text.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if b[i] == b'.' && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            return true;
        }
    }
    false
}

const RELATIVE_TIME_WORDS: &[&str] = &[
    "recently", "currently", "latest", "nowadays", "this year", "last year", "this month",
    "last month", "as of", "so far", "at the moment", "right now", "these days",
];

fn relative_time_density(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut hits = 0.0f32;
    for w in RELATIVE_TIME_WORDS {
        hits += lower.matches(w).count() as f32;
    }
    (hits / (text.len() as f32 / 200.0 + 1.0)).min(1.0)
}

const TASK_TIME_MARKERS: &[&str] = &[
    "latest", "newest", "current", "currently", "recent", "today", "now", "update",
    "updated", "version", "release", "released", "what's new", "whats new", "price",
    "cost", "news", "weather", "stock", "still maintained", "this year", "this month",
    "right now",
];

fn task_time_markers(task: &str) -> f32 {
    let lower = task.to_lowercase();
    let mut hits = 0.0f32;
    for w in TASK_TIME_MARKERS {
        if lower.contains(w) {
            hits += 1.0;
        }
    }
    (hits / 3.0).min(1.0)
}

/// URL shape that hints at time-sensitive or dated material:
/// docs/news/blog/changelog/release segments or a year in the path.
fn url_shape(text: &str) -> f32 {
    let lower = text.to_lowercase();
    if !lower.contains("://") && !lower.contains("http") {
        return 0.0;
    }
    let mut score = 0.0f32;
    for seg in ["/news", "/blog", "/changelog", "/release", "/announce", "/20"] {
        if lower.contains(seg) {
            score += 0.5;
        }
    }
    if lower.contains("://docs.") || lower.contains("/docs/") {
        score += 0.25;
    }
    score.min(1.0)
}

fn digit_density(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let digits = text.bytes().filter(|b| b.is_ascii_digit()).count();
    (digits as f32 / text.len() as f32 * 10.0).min(1.0)
}

fn len_norm(text: &str) -> f32 {
    ((text.len() as f32 + 1.0).ln() / 12.0).min(1.0)
}

/* ── The extractor ───────────────────────────────────────────────── */

/// Feature layout (28):
///   0 stale keyword density      10 distinct-years norm
///   1 hedge keyword density      11 version-string flag
///   2 fresh keyword density      12 relative-time density
///   3 legacy keyword density     13 url shape
///   4 years-behind norm          14 sim(content, task)
///   5 newest-year norm           15 task time markers
///   6 has-old-year flag          16 task time-anchor cosine
///   7 future-year flag           17 task length norm
///   8 explicit-date flag         18 content length norm
///   9 date × years-behind        19 digit density
///   20-23 kind one-hots (web/memory/tool/file)
///   24 brain strength            25 brain age norm
///   26 stale sentence-max cosine 27 hedge sentence-max cosine
///
/// Pure and deterministic for a fixed `now_year` — the trainer and the
/// runtime both call this.
pub fn extract(inp: &FreshnessInput) -> [f32; FRESHNESS_FEATURES] {
    let text = inp.text;
    let years = scan_years(text);
    let newest = years.iter().copied().max().unwrap_or(0);
    let distinct = {
        let mut ys = years.clone();
        ys.sort_unstable();
        ys.dedup();
        ys.len() as u32
    };
    let now = inp.now_year;
    let years_behind = if newest > 0 && now >= newest {
        (now - newest) as f32
    } else {
        0.0
    };
    let years_behind_norm = (years_behind / 10.0).min(1.0);
    let has_old_year = years.iter().any(|&y| now >= y && now - y >= 3);
    let has_future = years.iter().any(|&y| y > now);
    let explicit = has_explicit_date(text);

    let taskv = embed_text(inp.task);

    let mut f = [0.0f32; FRESHNESS_FEATURES];
    f[0] = keyword_density(text, STALE_KEYWORDS);
    f[1] = keyword_density(text, HEDGE_KEYWORDS);
    f[2] = keyword_density(text, FRESH_KEYWORDS);
    f[3] = keyword_density(text, LEGACY_KEYWORDS);
    f[4] = years_behind_norm;
    f[5] = if newest > 0 { (newest as f32 - 1900.0) / 200.0 } else { 0.0 };
    f[6] = if has_old_year { 1.0 } else { 0.0 };
    f[7] = if has_future { 1.0 } else { 0.0 };
    f[8] = if explicit { 1.0 } else { 0.0 };
    f[9] = if explicit { years_behind_norm } else { 0.0 };
    f[10] = (distinct as f32 / 5.0).min(1.0);
    f[11] = if has_version_string(text) { 1.0 } else { 0.0 };
    f[12] = relative_time_density(text);
    f[13] = url_shape(text);
    f[14] = cosine(&embed_text(text), &taskv);
    f[15] = task_time_markers(inp.task);
    f[16] = cosine(&taskv, &TIME_TASK_VEC);
    f[17] = len_norm(inp.task);
    f[18] = len_norm(text);
    f[19] = digit_density(text);
    let oh = inp.kind.one_hots();
    f[20] = oh[0];
    f[21] = oh[1];
    f[22] = oh[2];
    f[23] = oh[3];
    // clamp() passes NaN through — a non-finite caller scalar must land
    // at the neutral value, not propagate (Phase 1.3; §5.2 "reject or
    // sanitize non-finite numeric values").
    f[24] = if inp.strength.is_finite() { inp.strength.clamp(0.0, 1.0) } else { 0.0 };
    f[25] = if inp.age_days.is_finite() { (inp.age_days.max(0.0) / 730.0).min(1.0) } else { 0.0 };
    f[26] = sentence_max_cosine(text, &STALE_VEC);
    f[27] = sentence_max_cosine(text, &HEDGING_VEC);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp<'a>(text: &'a str, task: &'a str) -> FreshnessInput<'a> {
        FreshnessInput {
            text,
            task,
            kind: SourceKind::Web,
            strength: 0.0,
            age_days: 0.0,
            now_year: 2026,
        }
    }

    #[test]
    fn years_scanned_by_maximal_run() {
        assert_eq!(scan_years("released in 2019 and 2021"), vec![2019, 2021]);
        assert_eq!(scan_years("id 12345 is not a year"), Vec::<u32>::new());
        assert_eq!(scan_years("version 3.11 from 2024-05-01"), vec![2024]);
    }

    #[test]
    fn explicit_dates_found() {
        assert!(has_explicit_date("published 2021-03-04"));
        assert!(has_explicit_date("Posted on January 5, 2020 by admin"));
        assert!(!has_explicit_date("no dates here, just version 1.4"));
    }

    #[test]
    fn feature_vector_shape_and_determinism() {
        let a = extract(&inp(
            "This library was deprecated in March 2019 and is no longer maintained.",
            "what is the latest version of this library",
        ));
        let b = extract(&inp(
            "This library was deprecated in March 2019 and is no longer maintained.",
            "what is the latest version of this library",
        ));
        assert_eq!(a.len(), FRESHNESS_FEATURES);
        assert_eq!(a.to_vec(), b.to_vec(), "bit-exact for fixed inputs");
        assert!(a[4] > 0.5, "years-behind fires for 2019 material in 2026");
        assert!(a[0] > 0.3, "stale keyword density fires on 'deprecated'");
        assert!(a[8] > 0.0, "explicit date found");
        assert!(a[15] > 0.3, "task markers fire on 'latest version'");
    }

    /// Phase 1.3 (§5.3): empty/whitespace-only, control chars, embedded
    /// NULs, mixed Unicode, and path-like text must all yield finite,
    /// in-band vectors — never panic, never NaN.
    #[test]
    fn hostile_text_stays_finite_and_bounded() {
        let hostiles = [
            "",
            "   \t\n  ",
            "\u{0}\u{0}\u{0}",
            "配列のテスト😀🎉\u{2028}\u{2029}",
            "a/b/c/d/e/f.rs::mod::path",
            "!!!!!!!!!!!!!!!!!!!!",
            &"deprecated ".repeat(500),
            &"2020 ".repeat(200),
        ];
        for text in hostiles {
            for task in ["", "latest version of the library", "\u{0}task\u{0}"] {
                let f = extract(&FreshnessInput {
                    text,
                    task,
                    kind: SourceKind::Tool,
                    strength: f32::NAN, // sanitized upstream; extractor clamps
                    age_days: f32::NEG_INFINITY,
                    now_year: 2026,
                });
                for (i, x) in f.iter().enumerate() {
                    assert!(x.is_finite(), "feature {i} NaN/Inf for {text:?} x {task:?}: {x}");
                    assert!((-1.0..=2.0).contains(x), "feature {i} out of band: {x}");
                }
            }
        }
        // year scanning on NUL-heavy and digit-heavy text stays sane.
        assert!(scan_years("\u{0}1234\u{0}56789").is_empty(), "maximal-run rule holds");
    }

    #[test]
    fn timeless_content_stays_quiet() {
        let f = extract(&inp(
            "A B-tree is a self-balancing tree data structure that maintains sorted data.",
            "what is a b-tree",
        ));
        assert_eq!(f[4], 0.0, "no years → no staleness signal");
        assert_eq!(f[0], 0.0, "no stale vocabulary");
        assert_eq!(f[1], 0.0, "no hedging");
        assert_eq!(f[15], 0.0, "timeless task has no time markers");
    }
}
