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
    /// Caps the endpoint itself published for this exact model (harvested
    /// from its model listing — provider-agnostic). Second-best evidence
    /// after a real 400.
    Discovered,
    /// Parsed from a provider limit error this session.
    Learned,
    /// The same model id's numbers published on a DIFFERENT endpoint
    /// (cross-root fallback when the serving endpoint and the registry
    /// both know nothing).
    CrossRoot,
    /// Conservative default — nothing is known about this model.
    Default,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Config => "config",
            Source::Registry => "registry",
            Source::Discovered => "discovered",
            Source::Learned => "learned",
            Source::CrossRoot => "cross-root",
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
    /// True when explicit config EXCEEDED the strongest real evidence
    /// (a 1M /ctx on a model whose only known bound is 32k). The config
    /// is obeyed — silently shrinking it is what made /ctx look broken —
    /// but the caller must surface a visible note carrying the bound, so
    /// the user can see what the model/endpoint is believed to allow.
    pub clamped_config: bool,
    /// The strongest real bound an explicit config value exceeded, if any
    /// (window or output side, whichever fired). `None` when the config
    /// was within evidence or no evidence exists.
    pub config_exceeds_evidence: Option<i64>,
    /// Per-side sources: `window` and `max_output` resolve INDEPENDENTLY
    /// (a registry entry may know the ctx window but not the max output).
    /// Consumers that must not send a made-up number (e.g. max_tokens for
    /// a model nobody knows) check `max_source != Default` first.
    pub win_source: Source,
    pub max_source: Source,
}

/// True when ANY real evidence exists for this (model, endpoint): the
/// registry, the endpoint's own listing, or a learned-400 this session.
/// The wire layer uses it to refuse gambling an explicit config cap on a
/// model nothing knows (the empty-gateway class).
pub fn has_any_evidence(model: Option<&str>, base_url: Option<&str>) -> bool {
    let caps = model_caps::lookup_for(model, base_url);
    if caps.known() {
        return true;
    }
    if let Some(m) = model {
        if learned_entry(base_url, m).is_some() {
            return true;
        }
    }
    if let (Some(u), Some(m)) = (base_url, model) {
        crate::rt::model_caps_discovered::ensure_loaded();
        if crate::rt::model_caps_discovered::lookup(u, m).is_some() {
            return true;
        }
        if crate::rt::model_caps_discovered::lookup_any_root(m).is_some() {
            return true;
        }
    }
    false
}

/// Limits learned from provider errors this session, keyed by
/// (endpoint root, model) when the caller knows the endpoint, else by
/// model name alone. (window, max_output) — either may be None if only
/// one was parsed. Keying by endpoint is what stops a real 400 from one
/// gateway (a free tier serving the same id at 32k) from silently
/// shrinking the same model on a different endpoint that serves it at
/// 1M; the name-only key remains for callers that do not have a URL.
static LEARNED: LazyLock<Mutex<HashMap<String, (Option<i64>, Option<i64>)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The learned-store key for (endpoint, model). Endpoints are normalized
/// (scheme/host case, trailing slash, query/fragment) so `/v1` and `/v1/`
/// address the same learned entry.
pub fn learned_key(base_url: Option<&str>, model: &str) -> String {
    match base_url.map(crate::rt::model_caps_discovered::normalize_api_root) {
        Some(root) if !root.is_empty() => format!("{}\u{1}{}", root, model),
        _ => model.to_string(),
    }
}

/// Learned entry for (endpoint, model): the endpoint-specific lesson when
/// there is one, else any model-name-only lesson (learned before the root
/// was known), else None.
fn learned_entry(base_url: Option<&str>, model: &str) -> Option<(Option<i64>, Option<i64>)> {
    let g = LEARNED.lock().ok()?;
    if let Some(root) = base_url.map(crate::rt::model_caps_discovered::normalize_api_root) {
        if !root.is_empty() {
            if let Some(v) = g.get(&format!("{}\u{1}{}", root, model)) {
                return Some(*v);
            }
        }
    }
    g.get(model).copied()
}

/// Public read seam for diagnostics (e.g. `sofuu doctor`).
pub fn learned_for_endpoint(base_url: Option<&str>, model: &str) -> (Option<i64>, Option<i64>) {
    learned_entry(base_url, model).unwrap_or((None, None))
}

/// Resolve the selected model's working limits. Order per side:
/// explicit config (clamped to the model's caps when known) → registry →
/// caps discovered from the endpoint's own listing → error-learned →
/// conservative default. `cfg_window` / `cfg_max_output` are 0 when the
/// user set nothing. `base_url` is the endpoint the request will hit
/// (Any spelling; normalized internally) — None resolves registry-only,
/// which is the historical behavior.
///
/// Evidence strength, weakest → strongest: registry (a name-keyed guess)
/// < discovered (the endpoint's own numbers for THIS model) < learned (a
/// real 400 from THIS endpoint — ground truth). A stronger source
/// overrides a weaker one; user config is clamped to whatever real
/// evidence exists. All sides resolve INDEPENDENTLY — a source may know
/// only one number.
/// All the evidence resolve() consults, gathered in ONE pass by the
/// caller so the ladder itself is pure (Phase 1.1: the tests under this
/// module run in parallel with the rt/ai wire tests and the
/// discovered-caps tests, which mutate these same process-global stores).
#[derive(Clone, Copy, Default)]
struct Evidence {
    /// registry entry for the model name (PURE — the discovered store not
    /// folded in).
    reg_win: i64,
    reg_out: i64,
    /// caps the endpoint itself published for this exact (root, model),
    /// per side (None = not published / 0).
    disc_win: Option<i64>,
    disc_out: Option<i64>,
    /// limits learned from provider errors this session, per side.
    learned_win: Option<i64>,
    learned_out: Option<i64>,
    /// the same model id's caps published on a DIFFERENT endpoint
    /// (cross-root fallback), per side.
    cross_win: Option<i64>,
    cross_out: Option<i64>,
}

pub fn resolve(
    model: Option<&str>,
    cfg_window: i64,
    cfg_max_output: i64,
    base_url: Option<&str>,
) -> Resolved {
    resolve_explicit(model, cfg_window, cfg_max_output, base_url, false)
}

/// `resolve` with the explicitness switch. `explicit = true` when the
/// numbers came from a live `/ctx` / `/maxout` command in this session —
/// those are obeyed as typed (see `resolve_with_mode`); an inherited
/// `ctx_window` from config.json is not.
pub fn resolve_explicit(
    model: Option<&str>,
    cfg_window: i64,
    cfg_max_output: i64,
    base_url: Option<&str>,
    explicit: bool,
) -> Resolved {
    /* PURE registry numbers for per-side resolution — `lookup_for` folds
     * the discovered store into caps, which would mislabel discovered
     * truth as "registry" when both sides report below. */
    let pure_reg = model_caps::lookup(model);
    let learned = model
        .and_then(|m| learned_entry(base_url, m))
        .unwrap_or((None, None));
    /* Discovered numbers for this exact (endpoint, model) — per side. */
    let disc = base_url.and_then(|u| {
        model.and_then(|m| {
            crate::rt::model_caps_discovered::ensure_loaded();
            crate::rt::model_caps_discovered::lookup(u, m)
        })
    });
    /* Cross-root fallback: the same model id's numbers from another
     * endpoint — only when this endpoint and the registry know nothing. */
    let cross = model.and_then(|m| {
        crate::rt::model_caps_discovered::ensure_loaded();
        crate::rt::model_caps_discovered::lookup_any_root(m)
    });
    let ev = Evidence {
        reg_win: pure_reg.ctx_window as i64,
        reg_out: pure_reg.max_output as i64,
        disc_win: disc.filter(|d| d.ctx_window > 0).map(|d| d.ctx_window as i64),
        disc_out: disc.filter(|d| d.max_output > 0).map(|d| d.max_output as i64),
        learned_win: learned.0,
        learned_out: learned.1,
        cross_win: cross.filter(|c| c.ctx_window > 0).map(|c| c.ctx_window as i64),
        cross_out: cross.filter(|c| c.max_output > 0).map(|c| c.max_output as i64),
    };
    resolve_with_mode(&ev, cfg_window, cfg_max_output, explicit)
}

/// The pure evidence ladder — every input explicit, no global store reads.
/// Evidence strength, weakest → strongest: registry (a name-keyed guess)
/// < discovered (the endpoint's own numbers for THIS model) < learned (a
/// real 400 from THIS endpoint — ground truth). A stronger source
/// overrides a weaker one; user config is clamped to whatever real
/// evidence exists. All sides resolve INDEPENDENTLY — a source may know
/// only one number.
/// The pure evidence ladder with the INHERITED-global semantics (a
/// config.json number shrinks to the model's strongest real bound).
#[cfg(test)]
fn resolve_with(ev: &Evidence, cfg_window: i64, cfg_max_output: i64) -> Resolved {
    resolve_with_mode(ev, cfg_window, cfg_max_output, false)
}

/// The ladder with an explicit mode switch. `explicit = true` means the
/// value came from a live user command for THIS model (`/ctx 1000000`)
/// and is obeyed as given even above known evidence; `false` means it is
/// an inherited global that must shrink to the model's real bound.
fn resolve_with_mode(
    ev: &Evidence,
    cfg_window: i64,
    cfg_max_output: i64,
    explicit: bool,
) -> Resolved {
    let disc_win = ev.disc_win;
    let disc_out = ev.disc_out;
    let learned = (ev.learned_win, ev.learned_out);
    let cross_win = ev.cross_win.map(|v| (v, Source::CrossRoot));
    let cross_out = ev.cross_out.map(|v| (v, Source::CrossRoot));
    let caps_known = ev.reg_win > 0 || ev.reg_out > 0
        || disc_win.is_some()
        || disc_out.is_some()
        || learned.0.is_some()
        || learned.1.is_some()
        || cross_win.is_some()
        || cross_out.is_some();

    let mut clamped_config = false;
    let mut config_exceeds_evidence: Option<i64> = None;
    let mut source = Source::Default;
    let mut win_source = Source::Default;
    let mut max_source = Source::Default;

    // Window: an INHERITED global (config.json, carried across models)
    // must SHRINK to the model's strongest real bound — that is the P0
    // ring bug (a flat ctx_window=1M shadowing a 262k model). An
    // EXPLICIT value the user just typed (`/ctx 1000000`) is obeyed as
    // given: silently crushing it to 32k with no output is what made the
    // command look broken. Either way the bound is recorded so the caller
    // can surface a visible advisory, and a wrong guess is corrected by
    // the provider's own 400 via note_limit_error on the next turn.
    let window = if cfg_window > 0 {
        source = Source::Config;
        win_source = Source::Config;
        match strongest_side(
            ev.reg_win,
            disc_win,
            learned.0,
            Source::Registry,
            Source::Discovered,
            Source::Learned,
            cross_win,
        ) {
            Some((bound, src)) if cfg_window > bound => {
                clamped_config = true;
                config_exceeds_evidence = Some(bound);
                if explicit {
                    cfg_window
                } else {
                    win_source = src;
                    bound
                }
            }
            _ => cfg_window,
        }
    } else if let Some(strongest) = strongest_side(
        ev.reg_win,
        disc_win,
        learned.0,
        Source::Registry,
        Source::Discovered,
        Source::Learned,
        cross_win,
    ) {
        source = strongest.1;
        win_source = strongest.1;
        strongest.0
    } else {
        UNKNOWN_WINDOW
    };

    // Max output: same two modes as the window.
    let max_output = if cfg_max_output > 0 {
        if source == Source::Default {
            source = Source::Config;
        }
        max_source = Source::Config;
        match strongest_side(
            ev.reg_out,
            disc_out,
            learned.1,
            Source::Registry,
            Source::Discovered,
            Source::Learned,
            cross_out,
        ) {
            Some((bound, src)) if cfg_max_output > bound => {
                clamped_config = true;
                config_exceeds_evidence =
                    Some(config_exceeds_evidence.map_or(bound, |prev| prev.min(bound)));
                if explicit {
                    cfg_max_output
                } else {
                    max_source = src;
                    bound
                }
            }
            _ => cfg_max_output,
        }
    } else if let Some(strongest) = strongest_side(
        ev.reg_out,
        disc_out,
        learned.1,
        Source::Registry,
        Source::Discovered,
        Source::Learned,
        cross_out,
    ) {
        if source == Source::Default {
            source = strongest.1;
        }
        max_source = strongest.1;
        strongest.0
    } else {
        UNKNOWN_MAX_OUTPUT
    };

    let known = caps_known
        || disc_win.is_some()
        || disc_out.is_some()
        || learned.0.is_some()
        || learned.1.is_some()
        || cross_win.is_some()
        || cross_out.is_some()
        || cfg_window > 0
        || cfg_max_output > 0;

    Resolved {
        window: window.max(MIN_WINDOW),
        max_output: max_output.max(MIN_MAX_OUTPUT),
        known,
        source,
        clamped_config,
        config_exceeds_evidence,
        win_source,
        max_source,
    }
}

/// Strongest available side among registry / discovered / learned.
/// Returns None when no side knows. Strength order (weakest→strongest):
/// registry < discovered < learned — a learned 400 is ground truth from
/// this exact endpoint; discovered is what it publishes; the registry is
/// a name-keyed guess. A stronger number OVERRIDES a weaker one even
/// when smaller: a gateway hosting gpt-4o at 32k is right, the table
/// wrong. The source tag follows the side that supplied the number.
/// `cross_root` (optional) is the same model id's numbers from a
/// DIFFERENT endpoint — consulted only above the registry rung's absence
/// (weaker than same-root discovery: a mirror may re-serve the model
/// with its own per-plan caps).
fn strongest_side(
    registry: i64,
    discovered: Option<i64>,
    learned: Option<i64>,
    src_registry: Source,
    src_discovered: Source,
    src_learned: Source,
    cross_root: Option<(i64, Source)>,
) -> Option<(i64, Source)> {
    if let Some(l) = learned.filter(|v| *v > 0) {
        return Some((l, src_learned));
    }
    if let Some(d) = discovered.filter(|v| *v > 0) {
        return Some((d, src_discovered));
    }
    if registry > 0 {
        return Some((registry, src_registry));
    }
    cross_root.filter(|(v, _)| *v > 0)
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

    /* P1-20 (AUDIT-2026-09-01): CONTEXT side is checked FIRST when a
     * context keyword is present. Real dual-keyword errors read like
     * "The requested max_tokens (100000) exceeds the model's maximum
     * context length of 32768" — the old Output-first order classified
     * that as Output with the FIRST number near a keyword (often the
     * request's own size), and one such error poisoned max_output
     * persistently (the learned ladder is the strongest rung). Only a
     * message naming max_tokens with NO context wording falls to the
     * Output branch. Trigger list extended with the wordings
     * llama.cpp / LM Studio / gateways actually emit ("context window",
     * "context size", "context_window"). */
    let has_context_kw = lower.contains("context length") || lower.contains("context_length")
        || lower.contains("context window") || lower.contains("context_window")
        || lower.contains("context size") || lower.contains("too long")
        || lower.contains("too many tokens") || lower.contains("reduce the length")
        || lower.contains("token limit") || lower.contains("maximum context");

    if has_context_kw {
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
        // Context wording present but no number parsed: DO NOT fall
        // through to the Output branch — a wrong Output guess is worse
        // than no learning (see P1-20 note above).
        return None;
    }

    // Output-side shapes (they name max_tokens / output tokens with no
    // context wording anywhere).
    if lower.contains("max_tokens") || lower.contains("max_completion_tokens")
        || lower.contains("maximum output")
    {
        if let Some(n) = number_near(&lower, &["maximum", "limit", "supports", "is", "at most", "up to", "most"]) {
            if (1..=10_000_000).contains(&n) {
                return Some((LimitKind::Output, n));
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
    // Scan backwards from "maximum" for the nearest digit run, then parse
    // it with separators ("65,536 maximum").
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
        if b.is_ascii_digit() || *b == b',' || *b == b'_' {
            start = i;
        } else {
            break;
        }
    }
    let clean: String = lower[start..end].chars().filter(|c| c.is_ascii_digit()).collect();
    clean.parse::<i64>().ok()
}

/// Scan digits (and digit separators — providers format large numbers as
/// "65,536" or "65_536") starting at `start`. Returns the parsed value
/// and the index one past the run.
fn digits_at(s: &str, start: usize) -> Option<(i64, usize)> {
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b',' || bytes[i] == b'_') {
        i += 1;
    }
    // Trailing separators belong to the surrounding prose, not the number
    // ("131072, got 384000" — the comma after the limit is punctuation).
    // Trim them; then a run must still start and end with a digit
    // (no ",5" or "1,,").
    while i > start + 1 && (bytes[i - 1] == b',' || bytes[i - 1] == b'_') {
        i -= 1;
    }
    if i == start || !bytes[i - 1].is_ascii_digit() {
        return None;
    }
    let clean: String = s[start..i].chars().filter(|c| c.is_ascii_digit()).collect();
    clean.parse::<i64>().ok().map(|n| (n, i))
}

fn first_int(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            return digits_at(s, i).map(|(n, _)| n);
        }
        i += 1;
    }
    None
}

/// Record a provider limit error for the model. Returns true when a
/// limit was actually parsed and cached (the caller may then retry once
/// with corrected config). Pass the endpoint the request hit so the lesson
/// is scoped to it; `None` keys by model name alone.
pub fn note_limit_error(model: &str, err: &str) -> bool {
    note_limit_error_kind(model, err).is_some()
}

/// Like `note_limit_error`, but scoped to the endpoint that produced the
/// error, and reports WHICH kind of limit was learned — callers retry
/// once PER KIND (a session can breach the output cap and the context
/// window in the same turn; one total retry is not enough).
pub fn note_limit_error_at(base_url: Option<&str>, model: &str, err: &str) -> Option<LimitKind> {
    if model.is_empty() {
        return None;
    }
    let (kind, n) = parse_limit(err)?;
    if let Ok(mut g) = LEARNED.lock() {
        let e = g.entry(learned_key(base_url, model)).or_insert((None, None));
        match kind {
            LimitKind::Context => e.0 = Some(n),
            LimitKind::Output => e.1 = Some(n),
        }
    }
    Some(kind)
}

/// Name-only variant for callers without an endpoint in hand.
pub fn note_limit_error_kind(model: &str, err: &str) -> Option<LimitKind> {
    note_limit_error_at(None, model, err)
}

/// Record an output cap learned WITHOUT a parseable 400 — the length-empty
/// retry converged at `value` (a cap that produced text: ground truth from
/// THIS endpoint), or exhausted its allowance just above it. Same store and
/// strength as a parsed 400-learned limit: the resolve ladder's strongest
/// rung, so every later request this session clamps to it instead of
/// re-burning full turns against the gateway's ceiling. Values below
/// MIN_MAX_OUTPUT are ignored (a smaller cap means an unusable endpoint,
/// not a lesson). Returns the recorded value, or 0 when ignored.
pub fn note_max_output_at(base_url: Option<&str>, model: &str, value: i64) -> i64 {
    if model.is_empty() || value < MIN_MAX_OUTPUT {
        return 0;
    }
    if let Ok(mut g) = LEARNED.lock() {
        let e = g.entry(learned_key(base_url, model)).or_insert((None, None));
        e.1 = Some(value);
    }
    value
}

/// Test/diagnostic seam — drop everything learned this session.
pub fn clear_learned() {
    if let Ok(mut g) = LEARNED.lock() {
        g.clear();
    }
}

/// Test/diagnostic seam — what was learned for a model (any endpoint).
pub fn learned_for(model: &str) -> (Option<i64>, Option<i64>) {
    learned_entry(None, model).unwrap_or((None, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    /* Store-touching tests (LEARNED + discovered-caps) hold the shared ML
     * test lock for their whole body: the rt/ai wire tests and the
     * discovered-caps tests mutate the same process-global stores, and
     * cargo runs tests on parallel threads. */
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::ml::TEST_LOCK.lock().unwrap()
    }

    #[test]
    fn explicit_config_is_obeyed_above_evidence_with_advisory() {
        // The reported bug: `/ctx 1000000` on a 1M model whose caps nothing
        // knows used to land on the 32k default with no output, so the
        // command looked broken. Explicit values are obeyed as typed; the
        // evidence is reported as an advisory instead.
        let r = resolve_with_mode(&ev((0, 0), None, None, None), 1_000_000, 0, true);
        assert_eq!(r.window, 1_000_000, "an explicit /ctx must not be silently shrunk");
        assert_eq!(r.win_source, Source::Config);
        assert!(!r.clamped_config, "nothing is known, so nothing is exceeded");
        assert_eq!(r.config_exceeds_evidence, None);

        // With weak evidence present, obey the number AND report the bound.
        let r2 = resolve_with_mode(&ev((32_768, 4_096), None, None, None), 1_000_000, 0, true);
        assert_eq!(r2.window, 1_000_000);
        assert!(r2.clamped_config);
        assert_eq!(r2.config_exceeds_evidence, Some(32_768));

        // The INHERITED global keeps shrinking to the model's real bound —
        // this is the ring-bug guard, unchanged.
        let r3 = resolve_with_mode(&ev((32_768, 4_096), None, None, None), 1_000_000, 0, false);
        assert_eq!(r3.window, 32_768, "an inherited global must still shrink");
        assert_eq!(r3.win_source, Source::Registry);
        assert_eq!(r3.config_exceeds_evidence, Some(32_768));
    }

    #[test]
    fn explicit_output_cap_is_obeyed_above_evidence() {
        let r = resolve_with_mode(&ev((0, 0), Some((65_536, 8_192)), None, None), 0, 384_000, true);
        assert_eq!(r.max_output, 384_000, "explicit /maxout must not be silently shrunk");
        assert_eq!(r.max_source, Source::Config);
        assert!(r.clamped_config);
        let r2 = resolve_with_mode(&ev((0, 0), Some((65_536, 8_192)), None, None), 0, 384_000, false);
        assert_eq!(r2.max_output, 8_192, "inherited global still clamps");
    }

    #[test]
    fn explicit_config_below_evidence_is_honored_quietly() {
        // A deliberate budget smaller than the ceiling is not "exceeding"
        // anything — no advisory.
        let r = resolve_with_mode(&ev((128_000, 16_384), None, None, None), 16_384, 1_024, true);
        assert_eq!(r.window, 16_384);
        assert_eq!(r.max_output, 1_024);
        assert!(!r.clamped_config);
        assert_eq!(r.config_exceeds_evidence, None);
    }

    #[test]
    fn learned_limit_is_scoped_to_its_endpoint() {
        // F3: a real 400 on one gateway must not shrink the same model id
        // served by a different endpoint.
        let _store = lock();
        clear_learned();
        crate::rt::model_caps_discovered::clear();
        let model = "shared-id-zz7";
        let a = "https://free-tier.example.com/v1";
        let b = "https://premium.example.com/v1";
        assert_eq!(
            note_limit_error_at(Some(a), model, "maximum context length is 32768 tokens"),
            Some(LimitKind::Context)
        );
        // The endpoint that produced the 400 is corrected...
        let ra = resolve(Some(model), 0, 0, Some(a));
        assert_eq!(ra.window, 32_768);
        assert_eq!(ra.win_source, Source::Learned);
        // ...the other endpoint knows nothing and keeps the honest default.
        let rb = resolve(Some(model), 0, 0, Some(b));
        assert_eq!(rb.window, UNKNOWN_WINDOW, "a 400 from one root must not leak to another");
        assert_eq!(rb.win_source, Source::Default);
        assert!(!rb.known);
        // Trailing-slash spellings address the same learned entry.
        let ra2 = resolve(Some(model), 0, 0, Some("https://free-tier.example.com/v1/"));
        assert_eq!(ra2.window, 32_768);
        clear_learned();
    }

    #[test]
    fn unknown_model_gets_conservative_defaults() {
        let r = resolve(Some("totally-unknown-model-xyz"), 0, 0, None);
        assert_eq!(r.window, UNKNOWN_WINDOW);
        assert_eq!(r.max_output, UNKNOWN_MAX_OUTPUT);
        assert!(!r.known);
        assert_eq!(r.source, Source::Default);
    }

    #[test]
    fn known_model_uses_the_registry() {
        let r = resolve(Some("claude-sonnet-4-20250514"), 0, 0, None);
        assert!(r.known);
        assert!(r.window >= 100_000, "sonnet window from registry, got {}", r.window);
        assert!(r.max_output >= 8_000, "sonnet max output from registry, got {}", r.max_output);
        assert_eq!(r.source, Source::Registry);
    }

    #[test]
    fn config_is_clamped_to_the_models_caps() {
        // Ask for 1M on a 200k model — the model's hard limit wins.
        let r = resolve(Some("claude-sonnet-4-20250514"), 1_000_000, 500_000, None);
        assert!(r.clamped_config);
        assert!(r.window <= 250_000, "window clamped to caps, got {}", r.window);
        assert!(r.max_output <= 130_000, "max output clamped to caps, got {}", r.max_output);
    }

    #[test]
    fn config_below_caps_is_honoured() {
        let r = resolve(Some("claude-sonnet-4-20250514"), 16_384, 1_024, None);
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

    /// P1-20: the dual-keyword trap — max_tokens AND context length in one
    /// message is a CONTEXT error; the old Output-first order grabbed the
    /// request's own size (100000) and poisoned max_output with it.
    #[test]
    fn dual_keyword_error_is_context_not_output() {
        let err = "The requested max_tokens (100000) exceeds the model's maximum context length of 32768";
        assert_eq!(parse_limit(err), Some((LimitKind::Context, 32768)));
    }

    /// P1-20: the "context window/size" wordings llama.cpp / LM Studio /
    /// gateways actually emit used to learn NOTHING.
    #[test]
    fn context_window_wordings_parse() {
        let a = "requested context window of 131072 exceeds the model's maximum context window of 32768 tokens";
        assert_eq!(parse_limit(a), Some((LimitKind::Context, 32768)));
        let b = "context size exceeded: 9000 > 8192 maximum";
        assert_eq!(parse_limit(b), Some((LimitKind::Context, 8192)));
        let c = "prompt exceeds the model's context_window limit of 8192 tokens";
        assert_eq!(parse_limit(c), Some((LimitKind::Context, 8192)));
    }

    #[test]
    fn unrelated_errors_parse_to_nothing() {
        assert_eq!(parse_limit("rate limit exceeded, retry after 30 seconds"), None);
        assert_eq!(parse_limit("internal server error"), None);
    }

    #[test]
    fn learned_limit_feeds_resolve() {
        let _store = lock();
        clear_learned();
        let model = "mystery-local-model-abc";
        assert!(note_limit_error(model, "maximum context length is 24576 tokens"));
        let (w, o) = learned_for(model);
        assert_eq!(w, Some(24576));
        assert_eq!(o, None);
        let r = resolve(Some(model), 0, 0, None);
        assert_eq!(r.window, 24576);
        assert!(r.known, "a learned limit makes the model known");
        assert_eq!(r.source, Source::Learned);
        clear_learned();
    }

    /// 2026-10-09: the length-empty retry converged at a working cap with
    /// no parseable 400 in hand — note_max_output_at records it into the
    /// same session-scoped store, and resolve() must honor it as the
    /// strongest rung (later turns clamp instead of re-burning).
    #[test]
    fn noted_max_output_feeds_resolve() {
        let _store = lock();
        clear_learned();
        let model = "deepseek-v4.1-flash-test-only";
        // Below the usable floor: ignored, not a lesson.
        assert_eq!(note_max_output_at(None, model, 100), 0);
        assert_eq!(learned_for(model), (None, None));
        // A converged working cap sticks and beats the registry guess.
        assert_eq!(note_max_output_at(Some("https://gw.example/v1"), model, 2048), 2048);
        let r = resolve(Some(model), 0, 0, Some("https://gw.example/v1/chat/completions"));
        assert_eq!(r.max_output, 2048);
        assert_eq!(r.max_source, Source::Learned);
        // Empty model: ignored.
        assert_eq!(note_max_output_at(None, "", 2048), 0);
        clear_learned();
    }

    /* ── Discovered-caps rung (provider-agnostic: keyed by api root) ── */

    /* The ladder tests below run against the PURE resolve_with seam
     * (every evidence input passed in explicitly) — no global store, no
     * lock needed, no possible race with the rt/ai wire tests or the
     * discovered-caps tests, which run in parallel and mutate those
     * stores. The ladder ORDER is what these pin; the stores that feed
     * it are pinned by the discovered-caps tests themselves.
     *
     * Evidence builders mirror the real rungs: reg() is the registry
     * entry (name-keyed guess), disc() is what the endpoint published
     * for this exact model, learned() is a real 400, cross() is the same
     * model id's caps from a different endpoint. */

    fn ev(
        reg: (i64, i64),
        disc: Option<(i64, i64)>,
        learned: Option<(i64, i64)>,
        cross: Option<(i64, i64)>,
    ) -> Evidence {
        Evidence {
            reg_win: reg.0,
            reg_out: reg.1,
            disc_win: disc.map(|d| d.0),
            disc_out: disc.map(|d| d.1),
            learned_win: learned.map(|l| l.0),
            learned_out: learned.map(|l| l.1),
            cross_win: cross.map(|c| c.0),
            cross_out: cross.map(|c| c.1),
        }
    }

    #[test]
    fn discovered_caps_override_the_registry_and_default() {
        // An UNKNOWN model the endpoint told us about (registry knows
        // nothing, the listing is the only evidence).
        let r = resolve_with(&ev((0, 0), Some((65_536, 8_192)), None, None), 0, 0);
        assert!(r.known);
        assert_eq!(r.window, 65_536, "window from the endpoint's listing");
        assert_eq!(r.max_output, 8_192, "max output from the endpoint's listing");
        assert_eq!(r.source, Source::Discovered);

        // A KNOWN registry family the SAME endpoint hosts smaller: the
        // endpoint's own numbers beat the name-keyed table.
        let r2 = resolve_with(&ev((128_000, 16_384), Some((32_000, 4_096)), None, None), 0, 0);
        assert_eq!(r2.window, 32_000, "gateway's real window overrides the registry guess");
        assert_eq!(r2.max_output, 4_096);
        assert_eq!(r2.source, Source::Discovered);

        // The SAME model id at a DIFFERENT root (nothing discovered
        // there) keeps the registry truth.
        let r3 = resolve_with(&ev((128_000, 16_384), None, None, None), 0, 0);
        assert_eq!(r3.window, 128_000, "no discovery at this root → registry");
        assert_eq!(r3.source, Source::Registry);
    }

    #[test]
    fn learned_400_beats_discovered_listing() {
        // The listing claims 64k; a real 400 says the window is 32k.
        // Ground truth wins. The other side still comes from discovery.
        let r = resolve_with(
            &ev((0, 0), Some((65_536, 8_192)), Some((32_768, 0)), None),
            0,
            0,
        );
        assert_eq!(r.window, 32_768, "the real 400 is ground truth");
        assert_eq!(r.max_output, 8_192, "the other side still comes from discovery");
        assert_eq!(r.source, Source::Learned);
    }

    #[test]
    fn oversized_config_is_clamped_by_discovered_evidence() {
        // The exact failure class: global config maxout 384000 / ctx 1M
        // against an unknown model whose endpoint publishes 65,536 /
        // 8,192. No 400 may ever go out.
        let r = resolve_with(
            &ev((0, 0), Some((65_536, 8_192)), None, None),
            1_000_000,
            384_000,
        );
        assert!(r.clamped_config, "explicit config must be clamped by real evidence");
        assert_eq!(r.window, 65_536);
        assert_eq!(r.max_output, 8_192);
        assert_eq!(r.source, Source::Config);
    }

    #[test]
    fn per_side_sources_split_window_and_output() {
        // The ring bug: a 262k model's window must not be shadowed by a
        // flat global 1M, and per-side sources must report where EACH
        // number came from (the endpoint can know the window but not the
        // cap — disc max = 0 stays out of the ladder for that side).
        let r = resolve_with(
            &ev((0, 0), Some((262_144, 0)), None, None),
            1_000_000, // the stale flat global
            0,
        );
        assert_eq!(r.window, 262_144, "262k model must not inherit the 1M global");
        assert!(r.clamped_config);
        assert_eq!(r.win_source, Source::Discovered);
        // Output side: nobody knows it → Default; the driver must omit
        // max_tokens rather than send the conservative 4096.
        assert_eq!(r.max_source, Source::Default);
        assert_eq!(r.max_output, UNKNOWN_MAX_OUTPUT);
    }

    #[test]
    fn unknown_model_reports_default_side_sources() {
        // Pass-31 rule: a model nobody knows must report max_source
        // Default so drivers omit max_tokens instead of sending the
        // conservative 4096.
        let r = resolve_with(&ev((0, 0), None, None, None), 0, 0);
        assert_eq!(r.max_source, Source::Default);
        assert_eq!(r.win_source, Source::Default);
        assert_eq!(r.max_output, UNKNOWN_MAX_OUTPUT);
        assert_eq!(r.window, UNKNOWN_WINDOW);
        assert!(!r.known);
    }

    #[test]
    fn cross_root_evidence_beats_config_and_default() {
        // The live tokenrouter case: the serving endpoint publishes NO
        // numbers and the registry has no entry, but openrouter lists
        // the same model id with real caps. Those must clamp a flat 1M
        // global — the ring never shows the stale global again.
        let cross = Some((1_310_720, 131_072));
        // The real bug shape: a STALE global ABOVE the model's true
        // window must clamp down (2M > glm-5.3's 1.31M).
        let r = resolve_with(&ev((0, 0), None, None, cross), 2_000_000, 0);
        assert_eq!(r.window, 1_310_720);
        assert_eq!(r.win_source, Source::CrossRoot);
        assert!(r.clamped_config, "the oversized global must be clamped by the model's real numbers");
        // And with no config at all, cross-root truth is the window.
        let r_nocfg = resolve_with(&ev((0, 0), None, None, cross), 0, 0);
        assert_eq!(r_nocfg.window, 1_310_720);
        assert_eq!(r_nocfg.win_source, Source::CrossRoot);
        assert_eq!(r_nocfg.max_source, Source::CrossRoot);
        assert_eq!(r_nocfg.max_output, 131_072);
        // Same-root discovery still outranks cross-root truth.
        let r2 = resolve_with(
            &ev((0, 0), Some((65_536, 0)), None, cross),
            1_000_000,
            0,
        );
        assert_eq!(r2.window, 65_536);
        assert_eq!(r2.win_source, Source::Discovered);
    }

    #[test]
    fn free_tier_variant_does_not_inherit_paid_stem_window() {
        // 2026-09-15 over-report regression: a `-free` id with two
        // cross-root STEM entries of differing windows must resolve to
        // the 32,768 default, not the max (1,310,720) nor the min.
        // Synthetic ids avoid any registry entry for this family.
        let _store = lock();
        clear_learned();
        crate::rt::model_caps_discovered::clear();
        let stem = "regression-vendor/regression-paid-zz9";
        let free = "regression-vendor/regression-paid-zz9-free";
        let a = serde_json::json!({
            "id": stem, "context_length": 1_310_720, "max_tokens": 131_072
        });
        let b = serde_json::json!({
            "id": stem, "context_length": 1_000_000, "max_tokens": 128_000
        });
        assert!(crate::rt::model_caps_discovered::ingest_one(
            "https://openrouter.ai/api/v1", stem, &a
        ));
        assert!(crate::rt::model_caps_discovered::ingest_one(
            "https://api.orcarouter.ai/v1", stem, &b
        ));
        // No stem inheritance at the store layer.
        assert!(
            crate::rt::model_caps_discovered::lookup_any_root(free).is_none(),
            "free variant must not inherit any stem window"
        );
        // End-to-end through the ladder: serving root publishes nothing,
        // registry knows nothing → honest-conservative default.
        let r = resolve(
            Some(free),
            0,
            0,
            Some("https://api.tokenrouter.com/v1"),
        );
        assert_eq!(r.window, UNKNOWN_WINDOW, "must not inherit 1.31M stem window");
        assert_eq!(r.max_output, UNKNOWN_MAX_OUTPUT);
        assert_eq!(r.win_source, Source::Default);
        assert_eq!(r.max_source, Source::Default);
        assert_eq!(r.source, Source::Default);
        assert!(!r.known);
        crate::rt::model_caps_discovered::clear();
        clear_learned();
    }
}
