// rlm/sandbox.rs — hermetic nested QuickJS sandbox (PLAN-RLM R0).
//
// Safe-Rust facade over `sofuu_ffi::qjs_rt::QjsSandbox`: owns the context
// string + chunk map host-side (never copied wholesale into JS), injects
// the llm() answer cache before each eval, and maps the \0-sentinels the
// natives throw (`\0RLM_SUSPEND:` / `\0RLM_FINAL:`) onto `Outcome`.
//
// Suspension model: a snippet that calls llm() with an uncached batch
// suspends the WHOLE eval; the caller fulfills the sub-prompts and re-runs
// `eval_snippet` with the same source and an extended cache. Re-execution
// is kept side-effect-clean by a watermark: events a suspended attempt
// added are truncated before the re-run, so emits/traces never duplicate.
// To keep re-runs bit-for-bit reproducible the sandbox has no wall clock
// or true randomness: the bootstrap replaces Date/Math.random with counter
// stubs reset before every attempt, and the API surface (helpers + natives)
// is frozen so model code cannot redefine it mid-episode (PLAN-RLM G1).
//
// NOTE: after the deadline interrupt or an out-of-memory abort, the
// QuickJS runtime may be inconsistent (poisoned) — do not reuse the
// Sandbox; Drop is the cleanup path. The abort itself surfaces as
// `Outcome::Error`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use sofuu_ffi::qjs_rt::{QjsSandbox, RawOutcome, SandboxEvent, SandboxState};

use super::ToolCall;

const BOOTSTRAP: &str = include_str!("../../js/rlm_bootstrap.js");

/// Characters of overlap between consecutive chunks (PLAN-RLM R1.4).
const CHUNK_OVERLAP_CHARS: usize = 200;

/// What a snippet eval produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Snippet completed; `result_repr` is the last expression's value,
    /// stringified (truncated to 4000 chars).
    Done { result_repr: String },
    /// Snippet called `llm`/`llmBatch` with answers not in the cache. The
    /// caller must fulfill each prompt and re-run `eval_snippet` with the
    /// same source plus the new cache entries.
    Suspended { prompts: Vec<String> },
    /// Snippet called `tool`/`toolBatch` with results not in the cache
    /// (PLAN-AGENTS A4). `key` is the RAW batch JSON the bootstrap
    /// stringified (byte-exact cache key — do NOT re-serialize the calls,
    /// serde's Value map ordering need not match QuickJS insertion order).
    /// The caller executes each call through the agent's tool path and
    /// re-runs the snippet with the results cached under `key`.
    ToolSuspended { key: String, calls: Vec<ToolCall> },
    /// Snippet called `final(answer)`.
    Final { answer: String },
    /// JS exception (syntax/runtime error, memory limit, deadline abort).
    Error { message: String },
}

pub struct Sandbox {
    // Field order matters: `qjs` drops FIRST — its Drop unregisters the
    // state pointer and frees the JS side before the state Box is freed.
    qjs: QjsSandbox,
    state: Box<SandboxState>,
    /// events.len() before the current snippet's first attempt.
    watermark: usize,
    last_snippet: Option<String>,
}

impl Sandbox {
    /// New sandbox with a memory cap, stack cap, and wall-clock deadline
    /// (checked by the QuickJS interrupt handler). The bootstrap API
    /// (len/count/chunk/peek/grep/lines/llm/llmBatch/final/emit) is
    /// installed before this returns.
    pub fn new(mem_limit_bytes: usize, stack_bytes: usize, deadline: Duration) -> Result<Self, String> {
        // The sandbox id rides to the natives in a JS int32 — keep 31 bits.
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed) & 0x7fff_ffff;
        let mut state = Box::new(SandboxState::new());
        let state_ptr: *mut SandboxState = &mut *state;
        let qjs = QjsSandbox::new(id, mem_limit_bytes, stack_bytes, deadline, state_ptr)
            .ok_or_else(|| "failed to create QuickJS sandbox".to_string())?;
        let sb = Sandbox { qjs, state, watermark: 0, last_snippet: None };
        match sb.qjs.eval(BOOTSTRAP) {
            RawOutcome::Value(_) => Ok(sb),
            RawOutcome::Exception(e) => Err(format!(
                "sandbox bootstrap failed: {}",
                String::from_utf8_lossy(&e)
            )),
        }
    }

    /// Host defaults from PLAN-RLM R0.
    pub fn with_defaults(deadline: Duration) -> Result<Self, String> {
        Self::new(32 * 1024 * 1024, 1024 * 1024, deadline)
    }

    /// Store the full context host-side and (re)build the chunk map:
    /// UTF-8-safe, paragraph-preferred splitting with a 200-char overlap
    /// between consecutive chunks. JS sees chunk()/peek() in CHARACTER
    /// offsets; the stored ranges are byte offsets (both ends char-aligned).
    pub fn set_context(&mut self, text: &str, chunk_chars: usize) {
        self.state.char_len = text.chars().count();
        self.state.chunks = build_chunks(text, chunk_chars);
        self.state.context = text.to_string();
        self.state.events.clear();
        self.watermark = 0;
        self.last_snippet = None;
    }

    /// Number of chunks the last `set_context` produced.
    pub fn chunk_count(&self) -> usize {
        self.state.chunks.len()
    }

    /// Total context length in characters (= the sandbox's `len()`).
    pub fn context_chars(&self) -> usize {
        self.state.char_len
    }

    /// Take the sandbox-side trace events recorded so far.
    pub fn take_events(&mut self) -> Vec<SandboxEvent> {
        std::mem::take(&mut self.state.events)
    }

    /// Sandbox-side events refused past the trace cap (folded into
    /// RlmResult.dropped_events by the episode).
    pub fn dropped_events(&self) -> u32 {
        self.state.dropped_events
    }

    /// Re-arm the interrupt deadline for the next eval(s): the sandbox's
    /// construction-time deadline is the episode wall cap; `slice` is the
    /// per-eval time slice (Episode passes min(remaining wall, eval_slice_ms)
    /// before every snippet). Without a re-arm, a single aborted eval would
    /// leave the interrupt handler permanently expired.
    pub fn reset_eval_deadline(&mut self, slice: Duration) {
        self.qjs.reset_deadline(slice);
    }

    /// Evaluate one model-written snippet. `llm_cache` maps prompt batches
    /// (exactly as the model passed them to `llmBatch`) to their answers;
    /// `tool_cache` maps tool batches the same way (keyed by the RAW batch
    /// JSON, see `Outcome::ToolSuspended`). Both are injected before every
    /// run.
    pub fn eval_snippet(
        &mut self,
        src: &str,
        llm_cache: &[(Vec<String>, Vec<String>)],
        tool_cache: &[(String, Vec<String>)],
    ) -> Outcome {
        // 1. Inject the answer/result caches (re-runs after a suspension
        //    carry more entries than the first attempt). Same eval also
        //    resets the deterministic clock/rng, so a suspension re-run
        //    replays Date.now()/Math.random() identically.
        if let RawOutcome::Exception(e) = self.qjs.eval(&inject_script(llm_cache, tool_cache)) {
            return Outcome::Error {
                message: format!("failed to install caches: {}", String::from_utf8_lossy(&e)),
            };
        }
        // 2. Suspension watermark: re-evaluating the SAME snippet wipes the
        //    trace events its previous (suspended) attempt added.
        if self.last_snippet.as_deref() == Some(src) {
            self.state.events.truncate(self.watermark);
        } else {
            self.watermark = self.state.events.len();
            self.last_snippet = Some(src.to_string());
        }
        // 3. Evaluate (deadline + memory limit enforced inside QuickJS).
        self.state.eval_started = Some(Instant::now());
        let raw = self.qjs.eval(src);
        self.state.eval_started = None;
        match raw {
            RawOutcome::Value(repr) => Outcome::Done { result_repr: truncate_chars(&repr, 4000) },
            RawOutcome::Exception(bytes) => sentinel_outcome(&bytes),
        }
    }
}

/// Build the per-eval injection script: install `__LLM_CACHE` and
/// `__TOOL_CACHE` (`{ JSON.stringify(batch): [answers] }`) and reset the
/// deterministic clock/rng. Each JSON text is embedded as a JS string
/// literal via a second serde_json::to_string pass (valid JS string
/// escaping for free), so quote/backslash/NUL/sentinel-looking bytes in
/// answers can neither break the literal nor reach the sentinel parse path
/// (that path only ever sees native-thrown exceptions).
///
/// Size cap (G2): the COMBINED serialized caches are bounded at 256KB per
/// eval. An over-budget entry keeps its key — a suspended batch still
/// resolves — but its answers are replaced with a loud truncation marker;
/// later entries that no longer fit get an omission marker. (In normal flow
/// the episode already bounds answers to 4000 chars each and the number of
/// batches to max_llm_calls/max_tool_calls; this cap is the adversarial-
/// case backstop.)
const CACHE_BUDGET_BYTES: usize = 256 * 1024;
const MARK_TRUNC: &str = "[rlm:truncated]";
const MARK_OMIT: &str = "[rlm:omitted]";

/// One cache rendered as a JSON object literal's inner text, under a
/// shared byte budget. `key_of` produces the raw key string (already valid
/// JSON); `answers` are the result strings.
fn cache_entries_json<K: AsRef<str>>(
    entries: &[(K, Vec<String>)],
    used: &mut usize,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (key, answers) in entries {
        let key = key.as_ref();
        // Key embedded as a JSON string (escaping for free), the value is
        // the answers array.
        let key_lit = serde_json::to_string(key).unwrap_or_default();
        let entry_overhead = key_lit.len() + 1; // key + ':'
        let mut value = serde_json::Value::from(answers.clone());
        if used
            .saturating_add(entry_overhead)
            .saturating_add(value.to_string().len())
            > CACHE_BUDGET_BYTES
        {
            // Over budget: shrink the first answer into the remaining room
            // (or omit outright), marker included, so the JS-side answer is
            // visibly incomplete rather than silently wrong.
            let room = CACHE_BUDGET_BYTES
                .saturating_sub(*used)
                .saturating_sub(entry_overhead)
                .saturating_sub(64);
            let shrunk: Vec<String> = answers
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if i == 0 && room > 128 {
                        format!("{}{}", truncate_bytes(a, room), MARK_TRUNC)
                    } else {
                        MARK_OMIT.to_string()
                    }
                })
                .collect();
            value = serde_json::Value::from(shrunk);
        }
        *used = used
            .saturating_add(entry_overhead)
            .saturating_add(value.to_string().len());
        parts.push(format!("{}:{}", key_lit, value));
    }
    parts.join(",")
}

fn inject_script(
    llm_cache: &[(Vec<String>, Vec<String>)],
    tool_cache: &[(String, Vec<String>)],
) -> String {
    // Vec<String> keys are re-serialized (no nested objects → stable); tool
    // keys are RAW strings (byte-exact, see Outcome::ToolSuspended).
    let llm_owned: Vec<(String, Vec<String>)> = llm_cache
        .iter()
        .map(|(prompts, answers)| {
            (
                serde_json::to_string(prompts).unwrap_or_default(),
                answers.clone(),
            )
        })
        .collect();
    let mut used_llm: usize = 2; // "{}"
    let llm_json = format!("{{{}}}", cache_entries_json(&llm_owned, &mut used_llm));
    let mut used_tool: usize = 2;
    let tool_json = format!("{{{}}}", cache_entries_json(tool_cache, &mut used_tool));
    let llm_lit = serde_json::to_string(&llm_json).unwrap_or_else(|_| "\"{}\"".into());
    let tool_lit = serde_json::to_string(&tool_json).unwrap_or_else(|_| "\"{}\"".into());
    format!(
        "globalThis.__LLM_CACHE = JSON.parse({llm_lit}); \
         globalThis.__TOOL_CACHE = JSON.parse({tool_lit}); \
         globalThis.__rlmResetDeterminism();"
    )
}

/// Map a raw exception (possibly \0-prefixed sentinel) onto Outcome.
fn sentinel_outcome(bytes: &[u8]) -> Outcome {
    const SUSPEND: &[u8] = b"\0RLM_SUSPEND:";
    const FINAL: &[u8] = b"\0RLM_FINAL:";
    const TOOL_SUSPEND: &[u8] = b"\0RLM_TOOL_SUSPEND:";
    if let Some(payload) = bytes.strip_prefix(TOOL_SUSPEND) {
        let raw = String::from_utf8_lossy(payload).into_owned();
        match serde_json::from_str::<Vec<ToolCall>>(&raw) {
            Ok(calls) => Outcome::ToolSuspended { key: raw, calls },
            Err(e) => Outcome::Error { message: format!("malformed RLM_TOOL_SUSPEND payload: {e}") },
        }
    } else if let Some(payload) = bytes.strip_prefix(SUSPEND) {
        match serde_json::from_slice::<Vec<String>>(payload) {
            Ok(prompts) => Outcome::Suspended { prompts },
            Err(e) => Outcome::Error { message: format!("malformed RLM_SUSPEND payload: {e}") },
        }
    } else if let Some(payload) = bytes.strip_prefix(FINAL) {
        Outcome::Final { answer: String::from_utf8_lossy(payload).into_owned() }
    } else {
        Outcome::Error { message: String::from_utf8_lossy(bytes).into_owned() }
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Byte-budget truncation (cache cap — size accounting is in bytes):
/// cut at the last char boundary ≤ `max_bytes`.
fn truncate_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Byte offset of character index `ci` (text.len() when past the end).
fn byte_of_char(text: &str, ci: usize) -> usize {
    text.char_indices().nth(ci).map(|(b, _)| b).unwrap_or(text.len())
}

/// Split `text` into char-aligned byte ranges, preferring paragraph breaks
/// ("\n\n", falling back to "\n") at or before each `chunk_chars`-wide
/// window; hard-cut at the character boundary when no break exists.
/// Consecutive chunks overlap by `CHUNK_OVERLAP_CHARS` characters.
fn build_chunks(text: &str, chunk_chars: usize) -> Vec<(usize, usize)> {
    let chunk_chars = chunk_chars.max(1);
    let total = text.chars().count();
    if total == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0usize; // character offsets
    while start < total {
        let mut end = (start + chunk_chars).min(total);
        if end < total {
            let sb = byte_of_char(text, start);
            let eb = byte_of_char(text, end);
            let window = &text[sb..eb];
            let break_at = window.rfind("\n\n").map(|p| p + 2).or_else(|| window.rfind('\n').map(|p| p + 1));
            if let Some(pos) = break_at {
                let chars = window[..pos].chars().count();
                if start + chars > start {
                    end = start + chars;
                }
            }
        }
        chunks.push((byte_of_char(text, start), byte_of_char(text, end)));
        if end >= total {
            break;
        }
        // Overlap, but never backwards — always make progress.
        let ns = end.saturating_sub(CHUNK_OVERLAP_CHARS);
        start = if ns > start { ns } else { end };
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sandbox with the real defaults and a generous deadline.
    fn sb() -> Sandbox {
        Sandbox::with_defaults(Duration::from_secs(30)).expect("sandbox")
    }

    fn done_repr(o: &Outcome) -> &str {
        match o {
            Outcome::Done { result_repr } => result_repr,
            other => panic!("expected Done, got {other:?}"),
        }
    }

    // ── chunker ─────────────────────────────────────────────────

    #[test]
    fn chunker_empty_context() {
        assert!(build_chunks("", 16).is_empty());
    }

    #[test]
    fn chunker_one_giant_paragraph() {
        let text = "a".repeat(1000);
        let chunks = build_chunks(&text, 400);
        // No paragraph breaks → hard cuts at 400 chars with 200 overlap:
        // [0,400) [200,600) [400,800) [600,1000)
        assert_eq!(chunks, vec![(0, 400), (200, 600), (400, 800), (600, 1000)]);
    }

    #[test]
    fn chunker_prefers_paragraph_breaks() {
        let text = "aaaa\n\nbbbb\n\ncccc";
        let chunks = build_chunks(text, 7);
        // Window 7 chars ends mid-paragraph; the last "\n\n" inside wins.
        assert_eq!(chunks[0], (0, 6)); // "aaaa\n\n"
        assert_eq!(chunks[1], (6, 12)); // "bbbb\n\n" (200-char overlap clamps to progress)
        assert_eq!(chunks[2], (12, 16)); // "cccc"
        for (a, b) in &chunks {
            assert!(text.get(*a..*b).is_some());
        }
    }

    #[test]
    fn chunker_many_tiny_paragraphs() {
        let text: String = (0..200).map(|i| format!("p{i}")).collect::<Vec<_>>().join("\n\n");
        let chunks = build_chunks(&text, 64);
        assert!(chunks.len() > 1, "tiny paragraphs must still be grouped");
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks.last().unwrap().1, text.len());
        for w in chunks.windows(2) {
            assert!(w[0].0 < w[1].0, "chunk starts must advance: {w:?}");
        }
        for (a, b) in &chunks {
            assert!(text.get(*a..*b).is_some(), "char-aligned range {a}..{b}");
        }
    }

    #[test]
    fn chunker_chunk_smaller_than_a_word() {
        let text = "abcdefghij"; // one 10-char word
        let chunks = build_chunks(text, 3);
        // Mid-word cuts are legal; must always progress (no infinite loop).
        assert_eq!(chunks, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
    }

    // ── sandbox round-trips ─────────────────────────────────────

    #[test]
    fn roundtrip_len_count_chunk() {
        let mut s = sb();
        s.set_context("hello world", 4);
        let o = s.eval_snippet(r#"count() + "|" + len() + "|" + chunk(1)"#, &[], &[]);
        assert_eq!(done_repr(&o), "3|11|o wo");
    }

    #[test]
    fn multibyte_boundaries_are_char_safe() {
        let mut s = sb();
        // Emoji (4-byte) + CJK (3-byte) + ASCII mixed.
        let text = "😀ab😂 cde中文段落\n\n🙂 fgh中文第二段 ijk😈 lmnop";
        s.set_context(text, 8);
        // Every stored range must be a valid str slice (no mid-char cuts).
        for (a, b) in &s.state.chunks {
            assert!(text.get(*a..*b).is_some(), "range {a}..{b} splits a char");
        }
        // chunk(i) round-trips the exact sliced text.
        for i in 0..s.chunk_count() {
            let (a, b) = s.state.chunks[i];
            let o = s.eval_snippet(&format!("chunk({i})"), &[], &[]);
            assert_eq!(done_repr(&o), &text[a..b], "chunk({i})");
        }
        // peek() takes char offsets: chars are [😀,a,b,😂,' ',c,d,e,中,...].
        let o = s.eval_snippet("peek(1, 3)", &[], &[]);
        assert_eq!(done_repr(&o), "ab");
        let o = s.eval_snippet("peek(8, 10)", &[], &[]);
        assert_eq!(done_repr(&o), "中文");
        // Out-of-range ends clamp instead of throwing.
        let o = s.eval_snippet("peek(0, 999999)", &[], &[]);
        assert_eq!(done_repr(&o), text);
        // len() counts characters, not bytes.
        let o = s.eval_snippet("len()", &[], &[]);
        assert_eq!(done_repr(&o), text.chars().count().to_string());
    }

    #[test]
    fn eval_returns_last_expression() {
        let mut s = sb();
        s.set_context("x", 16);
        assert_eq!(done_repr(&s.eval_snippet("len(); 40 + 2", &[], &[])), "42");
        assert_eq!(done_repr(&s.eval_snippet("\"hi there\"", &[], &[])), "hi there");
        assert_eq!(done_repr(&s.eval_snippet("var a = 1;", &[], &[])), "undefined");
    }

    #[test]
    fn allocation_bomb_is_an_error_not_a_crash() {
        // 16MB cap; the push loop grows without bound until QuickJS's
        // allocator refuses ("out of memory"). (A bare `new Array(99999999)`
        // only sets .length in QuickJS — the fill loop is the real bomb.)
        let mut s = Sandbox::new(16 * 1024 * 1024, 1024 * 1024, Duration::from_secs(30))
            .expect("sandbox");
        let o = s.eval_snippet(r#"var a = []; for (var i = 0;; i++) a.push("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");"#, &[], &[]);
        match o {
            Outcome::Error { message } => assert!(!message.is_empty()),
            other => panic!("memory bomb must error, got {other:?}"),
        }
    }

    #[test]
    fn infinite_loop_dies_at_the_deadline() {
        let mut s = Sandbox::new(32 * 1024 * 1024, 1024 * 1024, Duration::from_millis(200))
            .expect("sandbox");
        let t0 = Instant::now();
        let o = s.eval_snippet("while (1) {}", &[], &[]);
        assert!(t0.elapsed() < Duration::from_secs(10), "deadline must bite fast");
        match o {
            Outcome::Error { message } => assert!(message.contains("interrupted"), "got: {message}"),
            other => panic!("while(1) must error, got {other:?}"),
        }
        // After an abort the sandbox may be poisoned; Drop is the cleanup.
    }

    #[test]
    fn grep_helper_finds_hits() {
        let mut s = sb();
        let text = "alpha line\nneedle one here\nplain\n\nanother block\n\nlast needle two\n";
        s.set_context(text, 24);
        let o = s.eval_snippet(r#"JSON.stringify(grep("needle", 10))"#, &[], &[]);
        let repr = done_repr(&o);
        assert!(repr.contains("needle one here"), "repr: {repr}");
        assert!(repr.contains("needle two"), "repr: {repr}");
        assert!(repr.contains("\"line\":1"), "line numbers: {repr}");
        assert!(repr.contains("\"chunk\":"), "hits carry chunk indexes: {repr}");
    }

    #[test]
    fn suspend_carries_prompts_verbatim() {
        let mut s = sb();
        s.set_context("whatever", 16);
        let src = r#"llmBatch(["p-α first 😀", "second \"quoted\"\nnewline"])"#;
        match s.eval_snippet(src, &[], &[]) {
            Outcome::Suspended { prompts } => assert_eq!(
                prompts,
                vec!["p-α first 😀".to_string(), "second \"quoted\"\nnewline".to_string()]
            ),
            other => panic!("expected Suspended, got {other:?}"),
        }
    }

    #[test]
    fn cache_hit_returns_answers_and_rerun_dedups_emits() {
        let mut s = sb();
        s.set_context("whatever", 16);
        let src = r#"emit("note", "pre"); var r = llmBatch(["A", "B"]); r.join("/")"#;
        match s.eval_snippet(src, &[], &[]) {
            Outcome::Suspended { prompts } => assert_eq!(prompts, vec!["A".to_string(), "B".to_string()]),
            other => panic!("expected Suspended, got {other:?}"),
        }
        // Attempt 1 emitted "note" once before suspending. The re-run must
        // not duplicate it (watermark truncation).
        let cache = vec![
            (vec!["A".to_string(), "B".to_string()], vec!["x".to_string(), "y".to_string()]),
        ];
        let o = s.eval_snippet(src, &cache, &[]);
        assert_eq!(done_repr(&o), "x/y");
        let notes = s.state.events.iter().filter(|e| e.kind == "note").count();
        assert_eq!(notes, 1, "emit must not duplicate across the re-run");
    }

    #[test]
    fn final_comes_back_through_the_sentinel() {
        let mut s = sb();
        s.set_context("whatever", 16);
        match s.eval_snippet(r#"final("the answer 😀")"#, &[], &[]) {
            Outcome::Final { answer } => assert_eq!(answer, "the answer 😀"),
            other => panic!("expected Final, got {other:?}"),
        }
    }

    // ── A4: host tools inside the sandbox ───────────────────────

    #[test]
    fn tool_call_suspends_with_calls_verbatim() {
        let mut s = sb();
        s.set_context("whatever", 16);
        let src = r#"toolBatch([{ name: "get_weather", args: { city: "Paris 😀" } }, { name: "noop", args: {} }])"#;
        match s.eval_snippet(src, &[], &[]) {
            Outcome::ToolSuspended { key, calls } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].args["city"], "Paris 😀");
                assert_eq!(calls[1].name, "noop");
                // The key is the RAW bootstrap JSON — byte-exact for the
                // cache lookup after the re-run.
                assert!(key.starts_with(r#"[{"name":"get_weather""#), "key: {key}");
            }
            other => panic!("expected ToolSuspended, got {other:?}"),
        }
        // tool(name, args) is toolBatch([{name, args}])[0].
        match s.eval_snippet(r#"tool("solo", { x: 1 })"#, &[], &[]) {
            Outcome::ToolSuspended { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "solo");
                assert_eq!(calls[0].args["x"], 1);
            }
            other => panic!("expected ToolSuspended, got {other:?}"),
        }
    }

    #[test]
    fn tool_cache_hit_returns_results_and_rerun_dedups_emits() {
        let mut s = sb();
        s.set_context("whatever", 16);
        let src = r#"emit("note", "pre"); var r = tool("echo", { m: "hi" }); "got:" + r"#;
        let key = match s.eval_snippet(src, &[], &[]) {
            Outcome::ToolSuspended { key, .. } => key,
            other => panic!("expected ToolSuspended, got {other:?}"),
        };
        // Attempt 1 emitted "note" once before suspending. The re-run must
        // not duplicate it (watermark truncation) and returns the result.
        let tcache = vec![(key, vec!["ECHO-HI".to_string()])];
        assert_eq!(done_repr(&s.eval_snippet(src, &[], &tcache)), "got:ECHO-HI");
        let notes = s.state.events.iter().filter(|e| e.kind == "note").count();
        assert_eq!(notes, 1, "emit must not duplicate across the re-run");
    }

    #[test]
    fn tool_and_llm_caches_coexist_in_one_snippet() {
        let mut s = sb();
        s.set_context("whatever", 16);
        // llm suspends FIRST (leftmost call); feeding only the llm cache
        // re-runs to the tool suspension; feeding both completes.
        let src = r#"var a = llm("q1"); var b = tool("echo", {}); a + "/" + b"#;
        match s.eval_snippet(src, &[], &[]) {
            Outcome::Suspended { prompts } => assert_eq!(prompts, vec!["q1".to_string()]),
            other => panic!("expected Suspended, got {other:?}"),
        }
        let lcache = vec![(vec!["q1".to_string()], vec!["A".to_string()])];
        let key = match s.eval_snippet(src, &lcache, &[]) {
            Outcome::ToolSuspended { key, .. } => key,
            other => panic!("expected ToolSuspended after llm resume, got {other:?}"),
        };
        let tcache = vec![(key, vec!["B".to_string()])];
        assert_eq!(done_repr(&s.eval_snippet(src, &lcache, &tcache)), "A/B");
    }

    #[test]
    fn bad_snippet_is_an_error_with_js_text() {
        let mut s = sb();
        s.set_context("x", 16);
        match s.eval_snippet("var x = ;", &[], &[]) {
            Outcome::Error { message } => assert!(message.contains("SyntaxError"), "got: {message}"),
            other => panic!("expected Error, got {other:?}"),
        }
        match s.eval_snippet("noSuchFn()", &[], &[]) {
            Outcome::Error { message } => assert!(message.contains("ReferenceError"), "got: {message}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn sandbox_sees_no_host_surface() {
        let mut s = sb();
        let o = s.eval_snippet(
            r#"typeof fetch + "|" + typeof process + "|" + typeof require + "|" + typeof sofuu + "|" + typeof std + "|" + typeof os"#,
            &[], &[],
        );
        assert_eq!(done_repr(&o), "undefined|undefined|undefined|undefined|undefined|undefined");
        // …but the whitelist and the bootstrap API are visible.
        let o = s.eval_snippet(r#"typeof __hx_len + "|" + typeof len + "|" + typeof grep + "|" + typeof tool + "|" + typeof __hx_tool"#, &[], &[]);
        assert_eq!(done_repr(&o), "function|function|function|function|function");
    }

    #[test]
    fn unknown_native_name_raises_reference_error() {
        let mut s = sb();
        match s.eval_snippet("__hx_total()", &[], &[]) {
            Outcome::Error { message } => assert!(message.contains("ReferenceError"), "got: {message}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // ── G1: hardening ───────────────────────────────────────────

    #[test]
    fn whitelist_is_frozen_against_clobbering() {
        let mut s = sb();
        s.set_context("whatever", 16);
        // Sloppy-mode global assignment to a frozen name silently no-ops.
        let o = s.eval_snippet(
            "try { llm = 42; } catch (e) {}\ntry { globalThis.chunk = function() { return \"x\"; }; } catch (e) {}\ntry { delete globalThis.final; } catch (e) {}\ntypeof llm + \"|\" + typeof chunk + \"|\" + typeof final + \"|\" + typeof __hx_len",
            &[], &[],
        );
        assert_eq!(done_repr(&o), "function|function|function|function");
        // …and the REAL helpers still work in a FOLLOWING snippet —
        // including final(), which must still fire its sentinel.
        let o = s.eval_snippet("len()", &[], &[]);
        assert_eq!(done_repr(&o), "8");
        match s.eval_snippet(r#"final("helpers intact")"#, &[], &[]) {
            Outcome::Final { answer } => assert_eq!(answer, "helpers intact"),
            other => panic!("expected Final, got {other:?}"),
        }
    }

    #[test]
    fn date_and_random_are_deterministic_across_suspension() {
        let mut s = sb();
        s.set_context("whatever", 16);
        // Within one attempt: values differ per call…
        let o = s.eval_snippet("String(Math.random() === Math.random()) + \"|\" + (Date.now() - Date.now())", &[], &[]);
        assert_eq!(done_repr(&o), "false|-1");
        // …and every eval starts from the same seed: identical output when
        // re-run, including across a suspension + cache-extended re-run.
        let src = r#"var a = Math.random(); var d = Date.now(); var ans = llm("q"); a + "," + d + "," + Math.random() + "," + ans"#;
        match s.eval_snippet(src, &[], &[]) {
            Outcome::Suspended { prompts } => assert_eq!(prompts, vec!["q".to_string()]),
            other => panic!("expected Suspended, got {other:?}"),
        }
        let cache = vec![(vec!["q".to_string()], vec!["A".to_string()])];
        let r2 = s.eval_snippet(src, &cache, &[]);
        let r3 = s.eval_snippet(src, &cache, &[]);
        let a2 = done_repr(&r2).to_string();
        assert_eq!(done_repr(&r3), a2, "suspension re-run must be reproducible");
        // First Date.now() of an attempt is always the fixed epoch stub.
        assert!(a2.contains(",1700000000000,"), "repr: {a2}");
        assert!(a2.ends_with(",A"), "llm answer present: {a2}");
    }

    #[test]
    fn eval_deadline_rearms_per_eval() {
        let mut s = sb();
        s.eval_snippet("1", &[], &[]); // warm
        // A 200ms slice kills a runaway snippet…
        s.reset_eval_deadline(Duration::from_millis(200));
        let t0 = Instant::now();
        let o = s.eval_snippet("while (1) {}", &[], &[]);
        assert!(t0.elapsed() < Duration::from_secs(10));
        assert!(matches!(o, Outcome::Error { .. }), "must abort: {o:?}");
        // …and after re-arming, the SAME sandbox keeps working (the
        // construction-time deadline is not one-shot).
        s.reset_eval_deadline(Duration::from_secs(10));
        assert_eq!(done_repr(&s.eval_snippet("1 + 1", &[], &[])), "2");
    }

    // ── G2/G3: caps and injection ───────────────────────────────

    #[test]
    fn cache_injection_caps_at_256kb_with_markers() {
        let mut s = sb();
        s.set_context("x", 16);
        let big = "B".repeat(400_000);
        let cache = vec![
            (vec!["big".to_string()], vec![big]),
            (vec!["small".to_string()], vec!["S".repeat(64)]),
        ];
        let o = s.eval_snippet(
            r#"var a = llm("big"); var b = llm("small");
               String(a.length < 400000) + "|" + a.slice(-15) + "|" + b"#,
            &cache, &[],
        );
        // First answer truncated into the remaining budget, second omitted.
        assert_eq!(done_repr(&o), "true|[rlm:truncated]|[rlm:omitted]");
    }

    #[test]
    fn cache_answers_with_hostile_bytes_round_trip_verbatim() {
        let mut s = sb();
        s.set_context("x", 16);
        // Quotes, backslashes, newlines, NUL and the SENTINEL strings as
        // data. None of it may break the injected literal or reach the
        // sentinel parse path (that path only sees native-thrown strings).
        let hostile =
            "quote\" backslash\\ newline\n tab\t nul\u{0} fake1\u{0}RLM_FINAL:notreal fake2\u{0}RLM_SUSPEND:[\"x\"]";
        let cache = vec![(vec!["k1".to_string()], vec![hostile.to_string()])];
        let js_expected = r#""quote\" backslash\\ newline\n tab\t nul\u0000 fake1\u0000RLM_FINAL:notreal fake2\u0000RLM_SUSPEND:[\"x\"]""#;
        let o = s.eval_snippet(
            &format!("var a = llm(\"k1\"); String(a === {js_expected}) + \"|\" + a.length"),
            &cache, &[],
        );
        let o = done_repr(&o).to_string();
        let (eq, len) = o.split_once('|').expect("shape");
        assert_eq!(eq, "true", "answer must round-trip verbatim");
        assert_eq!(len.parse::<usize>().unwrap(), hostile.chars().count());
    }

    #[test]
    fn context_sentinel_strings_do_not_spoof() {
        let mut s = sb();
        // A hostile document carrying both sentinel markers as plain text.
        let text = "intro \u{0}RLM_FINAL:spoofed middle \u{0}RLM_SUSPEND:[\"x\"] end";
        s.set_context(text, 16_000);
        let o = s.eval_snippet(r#"chunk(0).indexOf("\u0000RLM_FINAL:spoofed") >= 0"#, &[], &[]);
        // Done (a value), NOT Final("spoofed…") — data can never become a
        // sentinel; only the native throw path parses them.
        assert_eq!(done_repr(&o), "true");
    }
}
