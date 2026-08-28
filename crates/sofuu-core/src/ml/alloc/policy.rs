// ml/alloc/policy.rs — the MECHANICAL layer of the context allocator.
//
// Layer 0 + Layer 1 of the alloc design, the part that needs no net:
//
//   * resolve() — fetch the SELECTED MODEL's details FIRST (model_caps
//     registry → error-learned limits → conservative defaults), clamp
//     explicit user config (/ctx, /maxout) to the model's hard limits,
//     and hand back a Resolved window + max output that every budget
//     decision below is derived from. No fixed per-provider amounts.
//
//   * note_limit_error() — when a provider STILL rejects a request with a
//     limit error (the estimator was wrong, or the registry stale), parse
//     the real limit out of the error text, cache it for the session, and
//     let the caller retry once with corrected config. Next time the
//     allocation is right from the start — the error "should not happen
//     in the first place", and after one lesson it doesn't.
//
// The learned layer (model.rs) advises WITHIN the bounds resolved here;
// nothing it outputs ever escapes the clamps below.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::rt::model_caps;

/// Conservative window for a model nobody knows (registry miss, no
/// learned limit). Small on purpose: over-estimating the window is what
/// produces provider 400s; under-estimating only compacts a little early.
pub const UNKNOWN_WINDOW: i64 = 32_768;
/// Conservative max output for an unknown model.
pub const UNKNOWN_MAX_OUTPUT: i64 = 4_096;
/// Hard floors — an allocation never goes below these even on the
/// smallest known window.
pub const MIN_WINDOW: i64 = 2_048;
pub const MIN_MAX_OUTPUT: i64 = 512;

/// Where the resolved numbers came from (diagnostics / visible notes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// Explicit user config (/ctx, /maxout), clamped to the model's caps.
    Config,
    /// The baked model_caps registry.
    Registry,
    /// Parsed from a provider limit error this session.
    Learned,
    /// Conservative default — nothing is known about this model.
    Default,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Config => "config",
            Source::Registry => "registry",
            Source::Learned => "learned",
            Source::Default => "default",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Resolved {
    /// Working context window (tokens). Always ≥ MIN_WINDOW.
    pub window: i64,
    /// Maximum output tokens the selected model accepts. Always ≥
    /// MIN_MAX_OUTPUT.
    pub max_output: i64,
    /// True when at least one real source (config/registry/learned)
    /// reported the limits — false means conservative defaults.
    pub known: bool,
    pub source: Source,
    /// True when explicit config was clamped DOWN to the model's caps —
    /// the caller should surface a visible note.
    pub clamped_config: bool,
}

/// Limits learned from provider errors this session, keyed by model name.
/// (window, max_output) — either may be None if only one was parsed.
static LEARNED: LazyLock<Mutex<HashMap<String, (Option<i64>, Option<i64>)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolve the selected model's working limits. Order per side:
/// explicit config (clamped to the model's caps when known) → registry →
/// error-learned → conservative default. `cfg_window` / `cfg_max_output`
/// are 0 when the user set nothing.
pub fn resolve(model: Option<&str>, cfg_window: i64, cfg_max_output: i64) -> Resolved {
    let caps = model_caps::lookup(model);
    let learned = model
        .and_then(|m| LEARNED.lock().ok().and_then(|g| g.get(m).copied()))
        .unwrap_or((None, None));

    let mut clamped_config = false;
    let mut source = Source::Default;

    /* The strictest KNOWN bound wins over config: the registry cap and the
     * learned limit (a real provider 400 — ground truth) both constrain an
     * explicit /ctx or /maxout. An override above either is clamped down. */
    fn strictest(registry: i64, learned_v: Option<i64>) -> Option<i64> {
        match (registry, learned_v) {
            (r, Some(l)) if r > 0 => Some(r.min(l)),
            (r, None) if r > 0 => Some(r),
            (_, Some(l)) => Some(l),
            _ => None,
        }
    }

    // Window: config wins but never exceeds the model's real window.
    let window = if cfg_window > 0 {
        source = Source::Config;
        match strictest(caps.ctx_window as i64, learned.0) {
            Some(cap) if cfg_window > cap => {
                clamped_config = true;
                cap
            }
            _ => cfg_window,
        }
    } else if caps.ctx_window > 0 {
        source = Source::Registry;
        match learned.0 {
            Some(l) if l < caps.ctx_window as i64 => {
                source = Source::Learned;
                l
            }
            _ => caps.ctx_window as i64,
        }
    } else if let Some(w) = learned.0 {
        source = Source::Learned;
        w
    } else {
        UNKNOWN_WINDOW
    };

    // Max output: same ladder.
    let max_output = if cfg_max_output > 0 {
        if source == Source::Default {
            source = Source::Config;
        }
        match strictest(caps.max_output as i64, learned.1) {
            Some(cap) if cfg_max_output > cap => {
                clamped_config = true;
                cap
            }
            _ => cfg_max_output,
        }
    } else if caps.max_output > 0 {
        if source == Source::Default {
            source = Source::Registry;
        }
        match learned.1 {
            Some(l) if l < caps.max_output as i64 => {
                if source != Source::Config {
                    source = Source::Learned;
                }
                l
            }
            _ => caps.max_output as i64,
        }
    } else if let Some(o) = learned.1 {
        if source == Source::Default {
            source = Source::Learned;
        }
        o
    } else {
        UNKNOWN_MAX_OUTPUT
    };

    let known = caps.known() || learned.0.is_some() || learned.1.is_some() || cfg_window > 0 || cfg_max_output > 0;

    Resolved {
        window: window.max(MIN_WINDOW),
        max_output: max_output.max(MIN_MAX_OUTPUT),
        known,
        source,
        clamped_config,
    }
}

/// What a provider limit error told us.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LimitKind {
    /// Context window (prompt too long).
    Context,
    /// Output cap (max_tokens too large).
    Output,
}

/// Parse a real limit out of a provider error message. Conservative on
/// purpose: only unambiguous "… N tokens/maximum …" shapes are trusted;
/// anything odd returns None and the caller keeps the old config.
pub fn parse_limit(err: &str) -> Option<(LimitKind, i64)> {
    let lower = err.to_lowercase();

    // Output-side shapes first (they name max_tokens / output tokens).
    if lower.contains("max_tokens") || lower.contains("max_completion_tokens")
        || lower.contains("maximum output")
    {
        if let Some(n) = number_near(&lower, &["maximum", "limit", "supports", "is"]) {
            if (1..=10_000_000).contains(&n) {
                return Some((LimitKind::Output, n));
            }
        }
    }

    // Context-side shapes: "maximum context length is 128000 tokens",
    // "prompt is too long: 9000 tokens > 8192 maximum", "reduce the
    // length … to at most N tokens", "context length exceeded … N".
    if lower.contains("context length") || lower.contains("context_length")
        || lower.contains("too long") || lower.contains("too many tokens")
        || lower.contains("reduce the length") || lower.contains("token limit")
    {
        // "a > b maximum" — the LIMIT is the second number.
        if let Some(n) = gt_pattern_limit(&lower) {
            if (1..=10_000_000).contains(&n) {
                return Some((LimitKind::Context, n));
            }
        }
        if let Some(n) = number_near(&lower, &["maximum", "limit", "is", "most", "up to", "at most"]) {
            if (1..=10_000_000).contains(&n) {
                return Some((LimitKind::Context, n));
            }
        }
    }
    None
}

/// Find the first integer within a few words AFTER any of the keywords.
fn number_near(lower: &str, keywords: &[&str]) -> Option<i64> {
    for kw in keywords {
        let mut search_from = 0usize;
        while let Some(rel) = lower[search_from..].find(kw) {
            let at = search_from + rel + kw.len();
            let window_end = (at + 60).min(lower.len());
            if let Some(n) = first_int(&lower[at..window_end]) {
                return Some(n);
            }
            search_from = at;
        }
    }
    None
}

/// "… X tokens > Y maximum" style (Anthropic): the limit is Y, the
/// number that sits right before "maximum"/"max".
fn gt_pattern_limit(lower: &str) -> Option<i64> {
    let at = lower.find("maximum")?;
    // Scan backwards from "maximum" for the nearest integer.
    let head = &lower[..at];
    let bytes = head.as_bytes();
    let mut end = None;
    for (i, b) in bytes.iter().enumerate().rev() {
        if b.is_ascii_digit() {
            end = Some(i + 1);
            break;
        } else if end.is_some() || !b.is_ascii_whitespace() {
            break;
        }
    }
    let end = end?;
    let mut start = end;
    for (i, b) in bytes[..end].iter().enumerate().rev() {
        if b.is_ascii_digit() {
            start = i;
        } else {
            break;
        }
    }
    lower[start..end].parse::<i64>().ok()
}

fn first_int(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                i += 1;
            }
            let clean: String = s[start..i].chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = clean.parse::<i64>() {
                return Some(n);
            }
        }
        i += 1;
    }
    None
}

/// Record a provider limit error for the model. Returns true when a
/// limit was actually parsed and cached (the caller may then retry once
/// with corrected config).
pub fn note_limit_error(model: &str, err: &str) -> bool {
    note_limit_error_kind(model, err).is_some()
}

/// Like `note_limit_error`, but reports WHICH kind of limit was learned —
/// callers retry once PER KIND (a session can breach the output cap and
/// the context window in the same turn; one total retry is not enough).
pub fn note_limit_error_kind(model: &str, err: &str) -> Option<LimitKind> {
    if model.is_empty() {
        return None;
    }
    let (kind, n) = parse_limit(err)?;
    if let Ok(mut g) = LEARNED.lock() {
        let e = g.entry(model.to_string()).or_insert((None, None));
        match kind {
            LimitKind::Context => e.0 = Some(n),
            LimitKind::Output => e.1 = Some(n),
        }
    }
    Some(kind)
}

/// Test/diagnostic seam — drop everything learned this session.
pub fn clear_learned() {
    if let Ok(mut g) = LEARNED.lock() {
        g.clear();
    }
}

/// Test/diagnostic seam — what was learned for a model.
pub fn learned_for(model: &str) -> (Option<i64>, Option<i64>) {
    LEARNED.lock().map(|g| g.get(model).copied().unwrap_or((None, None))).unwrap_or((None, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_model_gets_conservative_defaults() {
        let r = resolve(Some("totally-unknown-model-xyz"), 0, 0);
        assert_eq!(r.window, UNKNOWN_WINDOW);
        assert_eq!(r.max_output, UNKNOWN_MAX_OUTPUT);
        assert!(!r.known);
        assert_eq!(r.source, Source::Default);
    }

    #[test]
    fn known_model_uses_the_registry() {
        let r = resolve(Some("claude-sonnet-4-20250514"), 0, 0);
        assert!(r.known);
        assert!(r.window >= 100_000, "sonnet window from registry, got {}", r.window);
        assert!(r.max_output >= 8_000, "sonnet max output from registry, got {}", r.max_output);
        assert_eq!(r.source, Source::Registry);
    }

    #[test]
    fn config_is_clamped_to_the_models_caps() {
        // Ask for 1M on a 200k model — the model's hard limit wins.
        let r = resolve(Some("claude-sonnet-4-20250514"), 1_000_000, 500_000);
        assert!(r.clamped_config);
        assert!(r.window <= 250_000, "window clamped to caps, got {}", r.window);
        assert!(r.max_output <= 130_000, "max output clamped to caps, got {}", r.max_output);
    }

    #[test]
    fn config_below_caps_is_honoured() {
        let r = resolve(Some("claude-sonnet-4-20250514"), 16_384, 1_024);
        assert!(!r.clamped_config);
        assert_eq!(r.window, 16_384);
        assert_eq!(r.max_output, 1_024);
        assert_eq!(r.source, Source::Config);
    }

    #[test]
    fn parses_openai_style_context_error() {
        let err = "This model's maximum context length is 128000 tokens. \
                   However, your messages resulted in 131072 tokens.";
        assert_eq!(parse_limit(err), Some((LimitKind::Context, 128000)));
    }

    #[test]
    fn parses_anthropic_style_too_long_error() {
        let err = "prompt is too long: 90000 tokens > 81920 maximum";
        assert_eq!(parse_limit(err), Some((LimitKind::Context, 81920)));
    }

    #[test]
    fn parses_max_tokens_exceeded_error() {
        let err = "requested max_tokens of 200000 exceeds the model's maximum output limit of 65536";
        assert_eq!(parse_limit(err), Some((LimitKind::Output, 65536)));
    }

    #[test]
    fn unrelated_errors_parse_to_nothing() {
        assert_eq!(parse_limit("rate limit exceeded, retry after 30 seconds"), None);
        assert_eq!(parse_limit("internal server error"), None);
    }

    #[test]
    fn learned_limit_feeds_resolve() {
        clear_learned();
        let model = "mystery-local-model-abc";
        assert!(note_limit_error(model, "maximum context length is 24576 tokens"));
        let (w, o) = learned_for(model);
        assert_eq!(w, Some(24576));
        assert_eq!(o, None);
        let r = resolve(Some(model), 0, 0);
        assert_eq!(r.window, 24576);
        assert!(r.known, "a learned limit makes the model known");
        assert_eq!(r.source, Source::Learned);
        clear_learned();
    }
}
