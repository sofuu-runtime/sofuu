// ml/compaction/features.rs — feature extraction for the compaction gate
// (PLAN-ML-GATES §12).
//
// One feature vector per history SEGMENT (a message or tool result in the
// chat history). The load-bearing signal is "is it referenced by recent
// turns" — everything else (age, size, kind, retrievability, boilerplate,
// duplication, decision language) supports the disposable-vs-load-bearing
// judgment. Deterministic: scalar f32, fixed order, injected context.

use std::sync::LazyLock;

use crate::ml::freshness::features::cosine;
use crate::rt::ai::sofuu_tfidf_embed;

pub const COMPACTION_FEATURES: usize = 33;
const EMBED_DIM: usize = 768;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SegKind {
    User,
    Assistant,
    ToolCall,
    ToolResult,
}

impl SegKind {
    fn one_hots(self) -> [f32; 4] {
        match self {
            SegKind::User => [1.0, 0.0, 0.0, 0.0],
            SegKind::Assistant => [0.0, 1.0, 0.0, 0.0],
            SegKind::ToolCall => [0.0, 0.0, 1.0, 0.0],
            SegKind::ToolResult => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SegmentInput<'a> {
    pub text: &'a str,
    pub tokens: u32,
    /// Turns elapsed since this segment (0 = part of the newest turn).
    pub age_steps: u32,
    pub kind: SegKind,
    /// Cheaply re-obtainable (file read, grep, web result…) — a compacted
    /// copy can be fetched again; unique derivations can't.
    pub retrievable: bool,
    /// Already summarized by an earlier pass.
    pub already_compacted: bool,
}

/// Everything a segment is scored against. `recent` is the text of the
/// protected keep-window (the newest turns, concatenated); `summary` is
/// the existing compacted summary ("" when none).
pub struct CompactionContext<'a> {
    pub task: &'a str,
    pub summary: &'a str,
    pub recent: &'a str,
    pub segments: &'a [SegmentInput<'a>],
}

/* ── Anchors ─────────────────────────────────────────────────────── */

const BOILERPLATE_ANCHOR: &str = "cookies help us deliver our services by using \
    this site you agree privacy policy terms of service apply navigation home about \
    contact sitemap all rights reserved subscribe newsletter copyright notice \
    disclaimer warranty limitation liability";

const DECISION_WORDS: &[&str] = &[
    "decided", "decision", "agreed", "concluded", "conclusion", "we will use",
    "chose", "choice", "settled", "the plan is", "must", "requirement",
    "accepted", "rejected", "final answer",
];

const ERROR_WORDS: &[&str] = &[
    "error", "failed", "failure", "traceback", "exception", "panic",
    "segfault", "assertion", "fatal", "not found",
];

/// Imperative/instruction language — user constraints that bind future
/// work ("keep the X unchanged", "do not merge until Y"). Distinct from
/// DECISION_WORDS: decisions record what was chosen, instructions bind
/// what may still happen. Both are load-bearing.
const INSTRUCTION_WORDS: &[&str] = &[
    "make sure", "keep the", "keep every", "keep all", "do not", "always",
    "never", "ensure", "avoid", "remember to", "must", "before you",
    "after every", "until the", "update the", "document",
];

static BOILERPLATE_VEC: LazyLock<Vec<f32>> = LazyLock::new(|| embed_anchor(BOILERPLATE_ANCHOR));

fn embed_anchor(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(text.as_bytes(), &mut v, EMBED_DIM);
    v
}

fn embed_text(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(text.as_bytes(), &mut v, EMBED_DIM);
    v
}

/* ── Lexical helpers ─────────────────────────────────────────────── */

fn word_hits(text: &str, words: &[&str]) -> f32 {
    let lower = text.to_lowercase();
    let mut hits = 0.0f32;
    for w in words {
        hits += lower.matches(w).count() as f32;
    }
    (hits / (text.len() as f32 / 200.0 + 1.0)).min(1.0)
}

/// Distinct/total token ratio over ascii word chars — low means
/// repetitive/verbose material.
fn distinct_token_ratio(text: &str) -> f32 {
    let mut seen = std::collections::HashSet::new();
    let mut total = 0usize;
    for tok in text.split(|c: char| !c.is_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        total += 1;
        seen.insert(tok.to_ascii_lowercase());
    }
    if total == 0 {
        return 1.0;
    }
    seen.len() as f32 / total as f32
}

fn density(text: &str, pred: impl Fn(char) -> bool) -> f32 {
    let mut n = 0usize;
    let mut total = 0usize;
    for c in text.chars() {
        total += 1;
        if pred(c) {
            n += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    (n as f32 / total as f32).min(1.0)
}

/// Path-like token density (`src/foo.rs`, `./x`, `crate::y`).
fn path_density(text: &str) -> f32 {
    let mut hits = 0.0f32;
    for tok in text.split_whitespace() {
        let t = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != ':' && c != '_');
        if (t.contains('/') && t.len() >= 5 && !t.starts_with("http"))
            || t.contains("::")
            || t.ends_with(".rs")
            || t.ends_with(".js")
            || t.ends_with(".ts")
            || t.ends_with(".py")
            || t.ends_with(".md")
            || t.ends_with(".toml")
            || t.ends_with(".json")
        {
            hits += 1.0;
        }
    }
    (hits / (text.len() as f32 / 200.0 + 1.0)).min(1.0)
}

/// Lexical overlap: fraction of the segment's distinctive whitespace
/// tokens (len ≥ 4) that appear in `other` — cheap reference detection
/// that complements the embedding cosine. Whitespace tokenization keeps
/// paths and symbols INTACT ("scripts/migrate_db.rs" is one token), and
/// long verbatim matches (paths, symbols) weigh 1.5×.
fn lexical_overlap(text: &str, other: &str) -> f32 {
    let other_lower = other.to_lowercase();
    let mut total = 0usize;
    let mut hit = 0.0f32;
    for tok in text.split_whitespace() {
        let t = tok.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != ':'
        });
        if t.len() < 4 {
            continue;
        }
        total += 1;
        let tl = t.to_ascii_lowercase();
        if other_lower.contains(&tl) {
            hit += if t.len() >= 8 { 1.5 } else { 1.0 };
        }
    }
    if total == 0 {
        return 0.0;
    }
    (hit / total as f32).min(1.0)
}

/* ── Extraction ──────────────────────────────────────────────────── */

/// Extract the 33-feature vector for segment `idx`. Embeddings are
/// computed once per call site that needs them (the caller batches via
/// `extract_all` in practice).
pub fn extract(ctx: &CompactionContext, idx: usize, emb: &[Vec<f32>]) -> [f32; COMPACTION_FEATURES] {
    let seg = &ctx.segments[idx];
    let mut f = [0.0f32; COMPACTION_FEATURES];

    let max_age = ctx
        .segments
        .iter()
        .map(|s| s.age_steps)
        .max()
        .unwrap_or(0)
        .max(1);
    let max_tokens = ctx.segments.iter().map(|s| s.tokens).max().unwrap_or(1).max(1);
    let total_tokens: u32 = ctx.segments.iter().map(|s| s.tokens).sum::<u32>().max(1);

    // 0-1: age
    f[0] = seg.age_steps as f32 / max_age as f32;
    f[1] = ((1.0 + seg.age_steps as f32).ln() / 10.0).min(1.0);
    // 2-5: kind one-hots
    let kh = seg.kind.one_hots();
    f[2] = kh[0];
    f[3] = kh[1];
    f[4] = kh[2];
    f[5] = kh[3];
    // 6-8: size
    f[6] = seg.tokens as f32 / max_tokens as f32;
    f[7] = (seg.tokens as f32 / 4000.0).min(1.0);
    f[8] = seg.tokens as f32 / total_tokens as f32;
    // 9-10: referenced by the recent keep-window (load-bearing signal)
    if !ctx.recent.is_empty() {
        f[9] = cosine(&emb[idx], &embed_text(ctx.recent));
        f[10] = lexical_overlap(seg.text, ctx.recent);
    }
    // 11: on-task similarity
    if !ctx.task.is_empty() {
        f[11] = cosine(&emb[idx], &embed_text(ctx.task));
    }
    // 12: redundancy with the existing compacted summary
    if !ctx.summary.is_empty() {
        f[12] = cosine(&emb[idx], &embed_text(ctx.summary));
    }
    // 13: retrievable
    f[13] = if seg.retrievable { 1.0 } else { 0.0 };
    // 14-15: boilerplate / repetitiveness
    f[14] = cosine(&emb[idx], &BOILERPLATE_VEC);
    f[15] = 1.0 - distinct_token_ratio(seg.text);
    // 16: near-duplicate of an EARLIER segment (max cosine, same text = 1)
    let mut dup = 0.0f32;
    for j in 0..idx {
        let c = cosine(&emb[idx], &emb[j]);
        if c > dup {
            dup = c;
        }
    }
    f[16] = dup;
    // 17-18: protection flags (age 0/1 = keep window; 0 = newest turn)
    f[17] = if seg.age_steps <= 1 { 1.0 } else { 0.0 };
    f[18] = if seg.age_steps == 0 { 1.0 } else { 0.0 };
    // 19-22: shape densities
    f[19] = density(seg.text, |c| matches!(c, '{' | '}' | '(' | ')' | ';' | '=')) * 4.0;
    if f[19] > 1.0 {
        f[19] = 1.0;
    }
    f[20] = density(seg.text, |c| c.is_ascii_digit()) * 5.0;
    if f[20] > 1.0 {
        f[20] = 1.0;
    }
    f[21] = density(seg.text, |c| c.is_ascii_uppercase()) * 5.0;
    if f[21] > 1.0 {
        f[21] = 1.0;
    }
    f[22] = path_density(seg.text);
    // 23-24: decision + error language
    f[23] = word_hits(seg.text, DECISION_WORDS);
    f[24] = word_hits(seg.text, ERROR_WORDS);
    // 25: user question (instructions/questions stay)
    let trimmed = seg.text.trim_end();
    f[25] = if seg.kind == SegKind::User && trimmed.ends_with('?') { 1.0 } else { 0.0 };
    // 26: tool result carrying an error (the model must react — keep)
    f[26] = if seg.kind == SegKind::ToolResult && f[24] >= 0.15 { 1.0 } else { 0.0 };
    // 27: already compacted once
    f[27] = if seg.already_compacted { 1.0 } else { 0.0 };
    // 28: raw length
    f[28] = (seg.text.len() as f32 / 8000.0).min(1.0);
    // 29-30: interactions — old-but-referenced stays, old-and-retrievable goes
    f[29] = f[0] * f[9].max(f[10]);
    f[30] = f[0] * f[13];
    // 31: position (fraction through the history)
    f[31] = idx as f32 / ctx.segments.len().max(1) as f32;
    // 32: imperative/instruction language (user constraints bind future
    // work — load-bearing regardless of age or size)
    f[32] = word_hits(seg.text, INSTRUCTION_WORDS);

    f
}

/// Embed every segment once, then extract all vectors (the JS shim's
/// batch path — embedding is the expensive part).
pub fn extract_all(ctx: &CompactionContext) -> Vec<[f32; COMPACTION_FEATURES]> {
    let emb: Vec<Vec<f32>> = ctx.segments.iter().map(|s| embed_text(s.text)).collect();
    (0..ctx.segments.len()).map(|i| extract(ctx, i, &emb)).collect()
}

/* ── Tests ───────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &'static str, tokens: u32, age: u32, kind: SegKind, retr: bool) -> SegmentInput<'static> {
        SegmentInput { text, tokens, age_steps: age, kind, retrievable: retr, already_compacted: false }
    }

    #[test]
    fn vector_shape_and_determinism() {
        let segs = [
            seg("old verbose tool output with lots of detail about the database migration run", 400, 9, SegKind::ToolResult, true),
            seg("we decided to use the ripgrep binary for all searches going forward", 40, 5, SegKind::Assistant, false),
            seg("what is the current status of the database migration?", 20, 0, SegKind::User, false),
        ];
        let ctx = CompactionContext {
            task: "finish the database migration",
            summary: "",
            recent: "what is the current status of the database migration?",
            segments: &segs,
        };
        let a = extract_all(&ctx);
        let b = extract_all(&ctx);
        assert_eq!(a.len(), 3);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert_eq!(va.len(), COMPACTION_FEATURES);
            for (x, y) in va.iter().zip(vb.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "extraction must be deterministic");
            }
        }
    }

    #[test]
    fn reference_and_protection_signals_fire() {
        let segs = [
            seg("the migration script lives at scripts/migrate_db.rs and rewrites the schema", 60, 8, SegKind::Assistant, false),
            seg("unrelated cookie banner boilerplate about services and privacy policy", 50, 7, SegKind::ToolResult, false),
            seg("please check scripts/migrate_db.rs again", 15, 0, SegKind::User, false),
        ];
        let ctx = CompactionContext {
            task: "fix the migration script",
            summary: "",
            recent: "please check scripts/migrate_db.rs again",
            segments: &segs,
        };
        let v = extract_all(&ctx);
        // The referenced segment outranks the boilerplate on both channels.
        assert!(v[0][9].max(v[0][10]) > v[1][9].max(v[1][10]) + 0.1);
        // Newest turn sits in the keep window.
        assert_eq!(v[2][17], 1.0);
        assert_eq!(v[2][18], 1.0);
        assert_eq!(v[0][17], 0.0);
    }

    #[test]
    fn duplicate_detection_fires_on_repeat() {
        let segs = [
            seg("directory listing: src, tests, crates, README.md, Makefile — five entries", 30, 6, SegKind::ToolResult, true),
            seg("a different result entirely about the weather in oslo today", 25, 3, SegKind::ToolResult, true),
            seg("directory listing: src, tests, crates, README.md, Makefile — five entries", 30, 1, SegKind::ToolResult, true),
        ];
        let ctx = CompactionContext { task: "list files", summary: "", recent: "", segments: &segs };
        let v = extract_all(&ctx);
        assert!(v[2][16] > 0.9, "identical earlier segment must read as duplicate");
        assert!(v[1][16] < 0.6, "unrelated segment must not");
    }
}
