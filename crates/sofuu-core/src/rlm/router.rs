// rlm/router.rs — route a turn between a plain model call and the RLM
// loop (PLAN-RLM R3): heuristics first, every decision logged; the learned
// ≤300KB router is only built if these logs prove the heuristics misroute.

use std::io::Write as _;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Route {
    Plain,
    Rlm,
}

impl Route {
    pub fn as_str(&self) -> &'static str {
        match self {
            Route::Plain => "plain",
            Route::Rlm => "rlm",
        }
    }
}

/// Single-word holistic markers — matched on word tokens (not substrings,
/// so "small" never fires "all").
const HOLISTIC_WORDS: &[&str] = &["summarize", "all", "every", "overall", "total"];
/// Multi-word holistic markers — matched as substrings after lowercasing.
const HOLISTIC_PHRASES: &[&str] = &["across the", "whole document", "how many"];

/// R3 heuristic v0:
/// - context over 80% of the effective window → Rlm (can't window it whole)
/// - else over 40% AND a holistic question (summarize-all / how-many / …)
///   → Rlm (answer needs the whole corpus)
/// - else → Plain
pub fn route(ctx_tokens: usize, effective_window_tokens: usize, question: &str) -> Route {
    let ctx = ctx_tokens as f64;
    let window = effective_window_tokens as f64;
    if ctx > 0.8 * window {
        return Route::Rlm;
    }
    if ctx > 0.4 * window && is_holistic(question) {
        return Route::Rlm;
    }
    Route::Plain
}

fn is_holistic(question: &str) -> bool {
    let lower = question.to_lowercase();
    if HOLISTIC_PHRASES.iter().any(|p| contains_phrase(&lower, p)) {
        return true;
    }
    // Word-token match so "small"/"totally" don't false-trigger.
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .any(|w| HOLISTIC_WORDS.contains(&w))
}

/// Substring match with word boundaries on both sides, so the phrase
/// "whole document" does not fire inside "whole documents".
fn contains_phrase(haystack: &str, phrase: &str) -> bool {
    let mut start = 0usize;
    while let Some(rel) = haystack.get(start..).and_then(|h| h.find(phrase)) {
        let abs = start + rel;
        let before_ok = haystack[..abs]
            .chars()
            .next_back()
            .map_or(true, |c| !c.is_alphanumeric());
        let after_ok = haystack[abs + phrase.len()..]
            .chars()
            .next()
            .map_or(true, |c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

/// Rotate at 5MB, single generation: routing_log.jsonl → routing_log.jsonl.1.
const ROTATE_BYTES: u64 = 5 * 1024 * 1024;

/// Append one decision to the JSONL log (`~/.sofuu/routing_log.jsonl` in
/// production). One line per call: `{ts_ms, ctx_tokens, window_tokens,
/// question, route}`. Question is truncated to 200 chars. Errors surface to
/// the caller as io::Error (logging must never break a turn).
pub fn log_decision(
    path: &Path,
    ctx_tokens: usize,
    window_tokens: usize,
    question: &str,
    decision: Route,
) -> std::io::Result<()> {
    log_decision_full(path, ctx_tokens, window_tokens, question, decision, None, None)
}

/// Same append as `log_decision`, with the R3 schema's optional outcome
/// fields (`latency_ms`, `rlm_calls`) attached when the caller knows them —
/// the chat hook logs after the query finishes, the JS API carries them in.
#[allow(clippy::too_many_arguments)]
pub fn log_decision_full(
    path: &Path,
    ctx_tokens: usize,
    window_tokens: usize,
    question: &str,
    decision: Route,
    latency_ms: Option<u64>,
    rlm_calls: Option<u32>,
) -> std::io::Result<()> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() >= ROTATE_BYTES {
            let mut rotated = path.as_os_str().to_owned();
            rotated.push(".1");
            std::fs::rename(path, rotated)?;
        }
    }
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut line = serde_json::json!({
        "ts_ms": ts_ms,
        "ctx_tokens": ctx_tokens,
        "window_tokens": window_tokens,
        "question": truncate_chars(question, 200),
        "route": decision.as_str(),
    });
    if let Some(l) = latency_ms {
        line["latency_ms"] = serde_json::json!(l);
    }
    if let Some(c) = rlm_calls {
        line["rlm_calls"] = serde_json::json!(c);
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    writeln!(f, "{line}")
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        static CTR: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "sofuu-rlm-router-test-{}-{}-{tag}.jsonl",
            std::process::id(),
            CTR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn route_matrix() {
        // > 80% of the window → Rlm regardless of the question.
        assert_eq!(route(900, 1000, "where is X defined"), Route::Rlm);
        assert_eq!(route(801, 1000, "anything"), Route::Rlm);
        // 40–80%: only holistic questions route to Rlm.
        assert_eq!(route(500, 1000, "summarize all the errors"), Route::Rlm);
        assert_eq!(route(500, 1000, "how many times does it crash"), Route::Rlm);
        assert_eq!(route(500, 1000, "where is X defined"), Route::Plain);
        // ≤ 40%: always Plain.
        assert_eq!(route(399, 1000, "summarize all the errors"), Route::Plain);
        assert_eq!(route(0, 1000, "summarize all"), Route::Plain);
        // Degenerate window.
        assert_eq!(route(1, 0, "hi"), Route::Rlm);
        assert_eq!(route(0, 0, "hi"), Route::Plain);
    }

    #[test]
    fn holistic_detection_no_substring_traps() {
        assert!(route(500, 1000, "give me the overall picture") == Route::Rlm);
        assert!(route(500, 1000, "every occurrence across the repo") == Route::Rlm);
        assert!(route(500, 1000, "TOTAL count please") == Route::Rlm);
        // Substring traps must NOT fire.
        assert_eq!(route(500, 1000, "a small talk about totals"), Route::Plain);
        assert_eq!(route(500, 1000, "totally redesign the parser"), Route::Plain);
        assert_eq!(route(500, 1000, "the whole documents API"), Route::Plain); // not "whole document"
    }

    #[test]
    fn jsonl_log_appends_valid_lines() {
        let path = tmp_path("append");
        log_decision(&path, 5000, 8192, "summarize all the things", Route::Rlm).expect("log");
        log_decision(&path, 100, 8192, "where is X", Route::Plain).expect("log");
        let text = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("valid json");
        assert_eq!(first["ctx_tokens"], 5000);
        assert_eq!(first["window_tokens"], 8192);
        assert_eq!(first["route"], "rlm");
        assert_eq!(first["question"], "summarize all the things");
        assert!(first["ts_ms"].as_u64().unwrap() > 0);
        let second: serde_json::Value = serde_json::from_str(lines[1]).expect("valid json");
        assert_eq!(second["route"], "plain");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_rotates_at_five_mb_single_generation() {
        let path = tmp_path("rotate");
        std::fs::write(&path, vec![b'x'; ROTATE_BYTES as usize + 1]).expect("seed");
        log_decision(&path, 1, 2, "q", Route::Plain).expect("log");
        let mut rotated = path.as_os_str().to_owned();
        rotated.push(".1");
        let rotated = std::path::PathBuf::from(rotated);
        // Old content moved aside; fresh log holds only the new line.
        assert_eq!(std::fs::metadata(&rotated).unwrap().len(), ROTATE_BYTES + 1);
        let text = std::fs::read_to_string(&path).expect("read");
        assert_eq!(text.lines().count(), 1);
        serde_json::from_str::<serde_json::Value>(text.trim()).expect("valid json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&rotated);
    }
}
