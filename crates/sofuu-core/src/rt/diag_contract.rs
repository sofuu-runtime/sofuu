// rt/diag_contract.rs — PLAN-UNIFIED-DIAGNOSTICS: one diagnostic contract.
//
// A diagnostic is a bounded, versioned description of a failure: a stable
// code, a factual message, an optional verified location, a known explanation,
// safe hints and only the frames that were actually observed. It never
// changes authorization, retries, exit status or side effects, and it never
// invents a location, a cause or a fix.
//
// The module is independent of its consumers: the CLI renders human text
// (render_human), machine consumers read the versioned JSON (to_json), and
// the JavaScript surface reaches it as `sofuu.diagnostics.*`. The event
// writer remains the ONE event boundary; a diagnostic does not auto-emit an
// event or add a JSONL format.
//
// Codes are never renamed. classify() adds a known explanation and a safe
// next step for codes that already exist; an unknown failure stays
// UNCLASSIFIED rather than being guessed into a known class.

use std::ffi::{CStr, CString};
use std::os::raw::c_int;

use serde_json::{json, Map, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

/// Shape version of a diagnostic. Independent of the event envelope's
/// `schema_version`; a consumer that understands 1 may ignore unknown fields.
pub const DIAGNOSTIC_VERSION: u64 = 1;

pub const MAX_CODE: usize = 64;
pub const MAX_MESSAGE: usize = 512;
pub const MAX_HINT: usize = 256;
pub const MAX_SNIPPET_LINE: usize = 200;
pub const MAX_HINTS: usize = 4;
pub const MAX_FRAMES: usize = 16;

/// The one fallback code. Deliberately not an `E_` taxonomy code: an
/// unclassifiable failure must not masquerade as a known one.
pub const UNCLASSIFIED: &str = "UNCLASSIFIED";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Rejection,
    Cancellation,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Rejection => "rejection",
            Severity::Cancellation => "cancellation",
        }
    }

    fn parse(s: &str) -> Option<Severity> {
        match s.trim().to_ascii_lowercase().as_str() {
            "error" => Some(Severity::Error),
            "warning" => Some(Severity::Warning),
            "rejection" => Some(Severity::Rejection),
            "cancellation" => Some(Severity::Cancellation),
            _ => None,
        }
    }
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Error
    }
}

/// A verified source location. `line`/`column` are 1-based. Every field is
/// optional because "not available" is a real answer and must not be guessed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceLocation {
    pub source: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
    /// True when this names generated/transformed code, not the user source.
    pub generated: bool,
    pub label: Option<String>,
}

impl SourceLocation {
    pub fn at(source: impl Into<String>, line: u32, column: Option<u32>) -> Self {
        SourceLocation {
            source: Some(source.into()),
            line: if line == 0 { None } else { Some(line) },
            column,
            ..SourceLocation::default()
        }
    }

    pub fn generated_at(source: impl Into<String>, line: u32) -> Self {
        SourceLocation {
            generated: true,
            ..SourceLocation::at(source, line, None)
        }
    }

    fn normalize(&mut self) {
        if self.line == Some(0) { self.line = None; }
        if self.column == Some(0) { self.column = None; }
        if self.end_line == Some(0) { self.end_line = None; }
        if self.end_column == Some(0) { self.end_column = None; }
        self.source = self.source.take().and_then(|s| {
            let s = sanitize_line(&s, MAX_MESSAGE);
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });
        self.label = self.label.take().and_then(|s| {
            let s = sanitize_line(&s, 64);
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });
    }

    fn present(&self) -> bool {
        self.source.is_some() || self.line.is_some()
    }

    fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(v) = self.source.as_ref() {
            m.insert("source".into(), json!(v));
        }
        if let Some(v) = self.line {
            m.insert("line".into(), json!(v));
        }
        if let Some(v) = self.column {
            m.insert("column".into(), json!(v));
        }
        if let Some(v) = self.end_line {
            m.insert("end_line".into(), json!(v));
        }
        if let Some(v) = self.end_column {
            m.insert("end_column".into(), json!(v));
        }
        if self.generated {
            m.insert("generated".into(), json!(true));
        }
        if let Some(v) = self.label.as_ref() {
            m.insert("label".into(), json!(v));
        }
        Value::Object(m)
    }
}

/// One failure, in the shape every surface can carry.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub diagnostic_version: u64,
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub operation: Option<String>,
    pub subsystem: Option<String>,
    pub location: Option<SourceLocation>,
    /// A known explanation (not observed causality).
    pub reason: Option<String>,
    pub hints: Vec<String>,
    /// Bounded frames observed in the error's reported stack.
    pub frames: Vec<String>,
}

impl Default for Diagnostic {
    fn default() -> Self {
        Diagnostic {
            diagnostic_version: DIAGNOSTIC_VERSION,
            code: UNCLASSIFIED.to_string(),
            severity: Severity::Error,
            message: "The operation failed. No further detail was reported.".to_string(),
            operation: None,
            subsystem: None,
            location: None,
            reason: None,
            hints: Vec::new(),
            frames: Vec::new(),
        }
    }
}

impl Diagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: Severity,
        message: impl Into<String>,
    ) -> Diagnostic {
        let message = sanitize_line(&message.into(), MAX_MESSAGE);
        Diagnostic {
            diagnostic_version: DIAGNOSTIC_VERSION,
            code: normalize_code(&code.into()),
            severity,
            message: if message.is_empty() {
                "The operation failed. No further detail was reported.".to_string()
            } else {
                message
            },
            operation: None,
            subsystem: None,
            location: None,
            reason: None,
            hints: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Construct from an existing code and apply its known explanation/hint.
    pub fn from_code(code: impl Into<String>, message: impl Into<String>) -> Diagnostic {
        let mut d = Diagnostic::new(code, Severity::Error, message);
        d.apply_classification();
        d
    }

    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = normalize_ident(Some(operation.into()), 96);
        self
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        let mut location = location;
        location.normalize();
        self.location = Some(location);
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        if self.hints.len() >= MAX_HINTS {
            return self;
        }
        let h = sanitize_line(&hint.into(), MAX_HINT);
        if !h.is_empty() && !self.hints.iter().any(|x| x == &h) {
            self.hints.push(h);
        }
        self
    }

    /// Add the known explanation and the safest next step for this code, if
    /// the tree has one. Never changes the code or invents a cause.
    pub fn apply_classification(&mut self) {
        let Some(info) = classify(&self.code) else {
            return;
        };
        if self.subsystem.is_none() {
            self.subsystem = Some(info.subsystem.to_string());
        }
        if self.reason.is_none() {
            self.reason = Some(info.reason.to_string());
        }
        if self.hints.is_empty() && !info.hint.is_empty() {
            self.hints.push(info.hint.to_string());
        }
        if info.severity != Severity::Error {
            self.severity = info.severity;
        }
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("diagnostic_version".into(), json!(DIAGNOSTIC_VERSION));
        m.insert("code".into(), json!(self.code));
        m.insert("severity".into(), json!(self.severity.as_str()));
        m.insert("message".into(), json!(self.message));
        if let Some(v) = self.operation.as_ref() {
            m.insert("operation".into(), json!(v));
        }
        if let Some(v) = self.subsystem.as_ref() {
            m.insert("subsystem".into(), json!(v));
        }
        if let Some(loc) = self.location.as_ref().filter(|l| l.present()) {
            m.insert("location".into(), loc.to_json());
        }
        if let Some(v) = self.reason.as_ref() {
            m.insert("reason".into(), json!(v));
        }
        if !self.hints.is_empty() {
            m.insert("hints".into(), json!(self.hints));
        }
        if !self.frames.is_empty() {
            m.insert("frames".into(), json!(self.frames));
        }
        Value::Object(m)
    }

    /// Tolerant reader for the machine surface: missing fields become
    /// unavailable, unknown fields are ignored, and nothing is guessed.
    pub fn from_json(v: &Value) -> Diagnostic {
        let code = match v.get("code") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => UNCLASSIFIED.to_string(),
        };
        let severity = v
            .get("severity")
            .and_then(Value::as_str)
            .and_then(Severity::parse)
            .unwrap_or(Severity::Error);
        let message = v
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("The operation failed. No further detail was reported.");
        let mut d = Diagnostic::new(code, severity, message);
        d.operation = v
            .get("operation")
            .and_then(Value::as_str)
            .and_then(|s| normalize_ident(Some(s.to_string()), 96));
        d.subsystem = v
            .get("subsystem")
            .and_then(Value::as_str)
            .and_then(|s| normalize_ident(Some(s.to_string()), 48));
        d.reason = v
            .get("reason")
            .and_then(Value::as_str)
            .map(|s| sanitize_line(s, MAX_MESSAGE))
            .filter(|s| !s.is_empty());
        d.location = v.get("location").and_then(location_from_json);
        if let Some(items) = v.get("hints").and_then(Value::as_array) {
            for item in items {
                if let Some(h) = item.as_str() {
                    d = d.with_hint(h.to_string());
                }
            }
        }
        if let Some(items) = v.get("frames").and_then(Value::as_array) {
            d.frames = items.iter().take(MAX_FRAMES).filter_map(Value::as_str)
                .map(|s| sanitize_line(s, MAX_MESSAGE)).filter(|s| !s.is_empty()).collect();
        }
        d
    }

    /// Render the human form. Only observed/known facts are printed; absent
    /// fields are omitted rather than guessed. The compact block is the
    /// default; `full` shows every hint.
    pub fn render_human(&self, full: bool, color: bool) -> String {
        let mut out = String::new();
        let label = self.severity.as_str();
        let head = if self.code == UNCLASSIFIED {
            format!("{}: {}", label, self.message)
        } else {
            format!("{}[{}]: {}", label, self.code, self.message)
        };
        if color {
            let colour = match self.severity {
                Severity::Error => "1;31",
                Severity::Warning => "1;33",
                Severity::Rejection => "1;35",
                Severity::Cancellation => "1;36",
            };
            out.push_str(&format!("\x1b[{colour}m{head}\x1b[0m\n"));
        } else {
            out.push_str(&head);
            out.push('\n');
        }
        if let Some(loc) = self.location.as_ref().filter(|l| l.present()) {
            if let Some(src) = loc.source.as_ref() {
                match (loc.line, loc.column) {
                    (Some(l), Some(c)) => out.push_str(&format!("  --> {src}:{l}:{c}\n")),
                    (Some(l), None) => out.push_str(&format!("  --> {src}:{l}\n")),
                    _ => out.push_str(&format!("  --> {src}\n")),
                }
            } else if let Some(l) = loc.line {
                out.push_str(&format!("  --> line {l}\n"));
            }
            if let Some(label) = &loc.label {
                out.push_str(&format!("   = location: {label}\n"));
            }
        }
        if let Some(reason) = self.reason.as_ref() {
            out.push_str(&format!("   = reason: {reason}\n"));
        }
        let hints = if full {
            self.hints.len()
        } else {
            self.hints.len().min(1)
        };
        for h in &self.hints[..hints] {
            out.push_str(&format!("   = help: {h}\n"));
        }
        if full && !self.frames.is_empty() {
            out.push_str("Trace (reported stack):\n");
            for frame in &self.frames {
                out.push_str("  ");
                out.push_str(frame);
                out.push('\n');
            }
        }
        out
    }
}

/// Whether the caller asked for full detail. Read per call so a wrapper can
/// set it for one process; this is not a persisted format decision.
pub fn full_detail_requested() -> bool {
    std::env::var("SOFUU_DIAGNOSTICS")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "full" | "1" | "verbose"
            )
        })
        .unwrap_or(false)
}

// ── classification ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct CodeInfo {
    pub subsystem: &'static str,
    pub reason: &'static str,
    pub hint: &'static str,
    pub severity: Severity,
}

/// The highest-value codes the tree already publishes. A code absent here is
/// not guessed: it stays unclassified and its message is shown as written.
pub fn classify(code: &str) -> Option<CodeInfo> {
    use Severity::*;
    let (subsystem, reason, hint, severity) = match code {
        "E_TIMEOUT" => (
            "deadline",
            "the operation exceeded its deadline",
            "retry only if the operation is safe to repeat",
            Error,
        ),
        "E_DENIED" | "E_CAP_DENIED" | "E_POLICY_DENIED" => (
            "policy",
            "the active policy does not allow this operation",
            "review the policy or ask its owner to grant the capability",
            Rejection,
        ),
        "E_CAPABILITY" => (
            "runtime",
            "the capability is unavailable in this build",
            "check the capability manifest with `sofuu doctor`",
            Error,
        ),
        "E_SCHEMA" => (
            "validation",
            "a value did not match the expected schema",
            "fix the value or the schema before retrying",
            Error,
        ),
        "E_INTERNAL" => (
            "runtime",
            "the runtime hit an internal invariant",
            "capture the trace with `sofuu logs --last-error`",
            Error,
        ),
        "E_KAZE_PARSE" => (
            "kaze",
            "the Kaze source did not parse",
            "fix the reported line and column",
            Error,
        ),
        "E_KAZE_DIV0" => (
            "kaze",
            "a Kaze division or modulo used zero",
            "guard the divisor before dividing",
            Error,
        ),
        "E_KAZE_RESULT" => (
            "kaze",
            "an asynchronous capability returned a value outside its contract",
            "review the provider result and validate it before using it",
            Error,
        ),
        "E_KAZE_OUTCOME_UNKNOWN" => (
            "kaze",
            "a dispatched tool may have completed without a received result",
            "inspect the operation's actual state before deciding whether to retry",
            Error,
        ),
        "E_CONTEXT_CAP" => (
            "agent",
            "the agent's context budget was exceeded",
            "reduce the history, tools or memories before retrying",
            Error,
        ),
        "E_SAFETY_VIOLATION" => (
            "robotics",
            "a safety gate vetoed the action",
            "review the gate's rule; the veto must not be bypassed",
            Rejection,
        ),
        _ => return None,
    };
    Some(CodeInfo {
        subsystem,
        reason,
        hint,
        severity,
    })
}

// ── sanitization ────────────────────────────────────────────────────────────

/// Strip ANSI CSI/OSC escapes from untrusted text before it reaches a
/// terminal or a log line.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.peek().copied() {
            Some('[') => {
                chars.next();
                for c2 in chars.by_ref() {
                    if ('@'..='~').contains(&c2) {
                        break;
                    }
                }
            }
            Some(']') => {
                chars.next();
                while let Some(c2) = chars.next() {
                    if c2 == '\u{7}' {
                        break;
                    }
                    if c2 == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            Some(_) => {
                chars.next();
            }
            None => {}
        }
    }
    out
}

fn is_bidi_control(c: char) -> bool {
    matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200B}' | '\u{200E}' | '\u{200F}' | '\u{FEFF}')
}

fn escape_untrusted(s: &str, preserve_layout: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' | '\t' | '\r' if preserve_layout => out.push(c),
            '\n' | '\t' | '\r' => out.push(' '),
            c if c.is_control() || is_bidi_control(c) => out.push('\u{FFFD}'),
            c => out.push(c),
        }
    }
    out
}

fn find_ascii_ci(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    for i in 0..=h.len() - n.len() {
        if h[i..i + n.len()].eq_ignore_ascii_case(n) {
            return Some(i);
        }
    }
    None
}

fn token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'+' | b'/' | b'=')
}

fn scrub_scheme_token(s: &str, scheme: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let Some(k) = find_ascii_ci(&s[i..], scheme) else {
            out.push_str(&s[i..]);
            break;
        };
        let start = i + k;
        out.push_str(&s[i..start]);
        out.push_str(&s[start..start + scheme.len()]);
        let mut j = start + scheme.len();
        let mut saw_space = false;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            saw_space = true;
            j += 1;
        }
        if !saw_space {
            i = start + scheme.len();
            continue;
        }
        let token_start = j;
        while j < bytes.len() && token_byte(bytes[j]) {
            j += 1;
        }
        if j > token_start {
            out.push(' ');
            out.push_str("[REDACTED]");
            i = j;
        } else {
            out.push_str(&s[start + scheme.len()..j]);
            i = j;
        }
    }
    out
}

const SECRET_KEYS: &[&str] = &[
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "secret_key",
    "client_secret",
    "private_key",
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
];

fn scrub_key_values(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut copy_from = 0;
    let mut i = 0;
    while i < bytes.len() {
        // Scan left to right, not in key-list order: otherwise a later
        // api_key can cause an earlier password to be copied unredacted.
        let boundary = |b: u8| !b.is_ascii_alphanumeric() && b != b'_';
        let before = i == 0 || boundary(bytes[i - 1]);
        let key = if before {
            SECRET_KEYS.iter().find(|key| {
                let end = i + key.len();
                end <= bytes.len()
                    && bytes[i..end].eq_ignore_ascii_case(key.as_bytes())
                    && (end == bytes.len() || boundary(bytes[end]))
            })
        } else { None };
        let Some(key) = key else {
            i += 1;
            continue;
        };
        let mut j = i + key.len();
        if j < bytes.len() && matches!(bytes[j], b'"' | b'\'') { j += 1; }
        while j < bytes.len() && matches!(bytes[j], b' ' | b'\t') { j += 1; }
        if j >= bytes.len() || !matches!(bytes[j], b':' | b'=') {
            i += key.len();
            continue;
        }
        j += 1;
        while j < bytes.len() && matches!(bytes[j], b' ' | b'\t') { j += 1; }
        let quote = if j < bytes.len() && matches!(bytes[j], b'"' | b'\'') {
            let q = bytes[j];
            j += 1;
            Some(q)
        } else { None };
        let value_start = j;
        if s[value_start..].starts_with("[REDACTED]") {
            let end = value_start + "[REDACTED]".len();
            if end == bytes.len() || matches!(bytes[end], b' ' | b'\t' | b'\r' | b'\n' | b',' | b';' | b'&' | b'#' | b'"' | b'\'' | b'}' | b']') {
                i = end;
                continue;
            }
        }
        if let Some(q) = quote {
            while j < bytes.len() {
                if bytes[j] == b'\\' && j + 1 < bytes.len() { j += 2; continue; }
                if bytes[j] == q { break; }
                j += 1;
            }
        } else {
            while j < bytes.len() && !matches!(bytes[j], b' ' | b'\t' | b'\r' | b'\n' | b',' | b';' | b'&' | b'#' | b'"' | b'\'' | b'}' | b']') { j += 1; }
        }
        if j > value_start {
            out.push_str(&s[copy_from..value_start]);
            out.push_str("[REDACTED]");
            copy_from = j;
        }
        i = j.max(i + key.len());
    }
    out.push_str(&s[copy_from..]);
    out
}

fn scrub_home(s: &str) -> String {
    let mut s = s.to_string();
    for prefix in ["/Users/", "/home/"] {
        loop {
            let Some(i) = s.find(prefix) else { break };
            let after = &s[i + prefix.len()..];
            match after.find('/') {
                Some(j) => {
                    let mut out = String::with_capacity(s.len());
                    out.push_str(&s[..i]);
                    out.push('~');
                    out.push_str(&after[j..]);
                    s = out;
                }
                None => break,
            }
        }
    }
    s
}

/// Redact common credential prefixes without treating every ordinary word
/// containing a short prefix as a secret. This is not a complete detector.
fn scrub_token_prefixes(s: &str) -> String {
    const PREFIXES: &[&str] = &["github_pat_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "xoxb-", "xoxp-", "glpat-", "sk-", "sk_", "AKIA", "ASIA", "AIza"];
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut copy = 0;
    let mut i = 0;
    while i < b.len() {
        let before = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if before {
            if let Some(prefix) = PREFIXES.iter().find(|p| s[i..].starts_with(**p)) {
                let mut end = i + prefix.len();
                while end < b.len() && (b[end].is_ascii_alphanumeric() || matches!(b[end], b'_' | b'-')) { end += 1; }
                if end >= i + prefix.len() + 8 {
                    out.push_str(&s[copy..i]);
                    out.push_str("[REDACTED]");
                    copy = end;
                    i = end;
                    continue;
                }
            }
        }
        i += s[i..].chars().next().map_or(1, char::len_utf8);
    }
    out.push_str(&s[copy..]);
    out
}

fn scrub_url_userinfo(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    while let Some(rel) = s[pos..].find("://") {
        let start = pos + rel + 3;
        let authority_end = s[start..].find(|c: char| matches!(c, '/' | '?' | '#' | ' ' | '\n' | '\r' | '\t' | '"' | '\''))
            .map_or(s.len(), |n| start + n);
        if let Some(at) = s[start..authority_end].rfind('@') {
            out.push_str(&s[pos..start]);
            out.push_str("[REDACTED]@");
            pos = start + at + 1;
        } else {
            out.push_str(&s[pos..start]);
            pos = start;
        }
    }
    out.push_str(&s[pos..]);
    out
}

fn scrub_private_keys(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    while let Some(rel) = s[pos..].find("-----BEGIN ") {
        let start = pos + rel;
        let Some(header_end) = s[start + 11..].find("-----").map(|n| start + 11 + n + 5) else {
            if s[start..].contains("PRIVATE KEY") {
                out.push_str(&s[pos..start]);
                out.push_str("[REDACTED PRIVATE KEY]");
                return out;
            }
            break;
        };
        if !s[start..header_end].contains("PRIVATE KEY") {
            out.push_str(&s[pos..header_end]);
            pos = header_end;
            continue;
        }
        out.push_str(&s[pos..start]);
        out.push_str("[REDACTED PRIVATE KEY]");
        let rest = &s[header_end..];
        pos = rest.find("-----END ")
            .and_then(|n| rest[n + 9..].find("-----").map(|end| header_end + n + 9 + end + 5))
            .unwrap_or(s.len());
    }
    out.push_str(&s[pos..]);
    out
}

fn scrub_sensitive(s: &str) -> String {
    let s = strip_ansi(s);
    let s = scrub_private_keys(&s);
    let s = scrub_url_userinfo(&s);
    let s = scrub_scheme_token(&s, "bearer");
    let s = scrub_scheme_token(&s, "basic");
    let s = scrub_key_values(&s);
    let s = scrub_token_prefixes(&s);
    scrub_home(&s)
}

/// The shared text sanitizer: strip terminal escapes, redact secret-shaped
/// substrings, normalize private home paths, and replace dangerous controls.
/// Applied by both diagnostic rendering and the event redaction seam.
pub fn redact_text(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    let bounded = truncate_chars(s, 64 * 1024);
    let s = scrub_sensitive(&bounded);
    escape_untrusted(&s, false)
}

/// Stored events may carry recoverable content: keep whitespace and do not
/// silently truncate it. Terminal controls and secret-shaped text remain
/// sanitized. Callers must not treat a changed checkpoint as recoverable.
pub fn redact_event_text(s: &str) -> String {
    let s = scrub_sensitive(s);
    escape_untrusted(&s, true)
}

/// Mask the contents of string literals so a source snippet can be shown
/// without echoing a secret. The quote characters and line length stay, so a
/// caret keeps pointing at the same column.
#[cfg(test)]
pub fn mask_source_literals(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in line.chars() {
        match quote {
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                escaped = false;
                out.push(c);
            }
            Some(q) => {
                if escaped {
                    escaped = false;
                    out.push('*');
                } else if c == '\\' {
                    escaped = true;
                    out.push('*');
                } else if c == q {
                    quote = None;
                    out.push(c);
                } else {
                    out.push('*');
                }
            }
            None => out.push(c),
        }
    }
    out
}

pub fn truncate_chars(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(3);
    let mut out: String = s.chars().take(keep).collect();
    out.push_str("...");
    out
}

/// Sanitize a single-line human field: no control characters, no secrets, and
/// bounded length.
pub fn sanitize_line(s: &str, max: usize) -> String {
    let cleaned = redact_text(s);
    truncate_chars(&cleaned, max)
}

fn normalize_code(code: &str) -> String {
    let s = sanitize_line(code, MAX_CODE);
    let trimmed = s.trim();
    if trimmed.is_empty() {
        UNCLASSIFIED.to_string()
    } else {
        trimmed.to_string()
    }
}

fn location_from_json(v: &Value) -> Option<SourceLocation> {
    if v.is_null() {
        return None;
    }
    let num = |key: &str| v.get(key).and_then(Value::as_u64).map(|n| n.min(u32::MAX as u64) as u32);
    let mut loc = SourceLocation {
        source: v.get("source").and_then(Value::as_str).map(str::to_string),
        line: num("line"),
        column: num("column"),
        end_line: num("end_line"),
        end_column: num("end_column"),
        generated: v.get("generated").and_then(Value::as_bool).unwrap_or(false),
        label: v.get("label").and_then(Value::as_str).map(str::to_string),
    };
    loc.normalize();
    if loc.present() {
        Some(loc)
    } else {
        None
    }
}

fn normalize_ident(v: Option<String>, max: usize) -> Option<String> {
    let raw = sanitize_line(&v?, MAX_MESSAGE);
    let mut out = String::with_capacity(raw.len().min(max));
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':' | '/') {
            out.push(c);
        } else if !c.is_whitespace() {
            out.push('-');
        }
        if out.len() >= max {
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// ── JavaScript exception adapter ────────────────────────────────────────────

/// The first parseable location in a reported QuickJS stack, when present.
/// Only `file:line:col` (or `file:line`) shapes are accepted; anything else is
/// reported as unavailable rather than guessed.
pub fn first_stack_location(stack: &str) -> Option<SourceLocation> {
    for raw in stack.lines() {
        let line = raw.trim().strip_prefix("at ").unwrap_or(raw.trim());
        if let Some(open) = line.rfind(" (") {
            if line.ends_with(')') {
                if let Some(loc) = parse_location_text(&line[open + 2..line.len() - 1]) {
                    if loc.line.is_some() { return Some(loc); }
                }
            }
        }
        if let Some(loc) = parse_location_text(line) {
            if loc.line.is_some() {
                return Some(loc);
            }
        }
    }
    None
}

fn parse_location_text(text: &str) -> Option<SourceLocation> {
    let text = text.trim();
    if text.is_empty() || (text.starts_with('<') && !text.contains(':')) {
        return None;
    }
    let (prefix, last) = text.rsplit_once(':')?;
    let last = last.parse::<u32>().ok().filter(|n| *n > 0)?;
    let (source, line, column) = match prefix.rsplit_once(':') {
        Some((src, l)) if l.parse::<u32>().ok().filter(|n| *n > 0).is_some() =>
            (src, l.parse::<u32>().ok()?, Some(last)),
        _ => (prefix, last, None),
    };
    if source.is_empty() { return None; }
    let mut loc = SourceLocation::at(source, line, column);
    loc.label = Some("reported stack; source mapping not verified".into());
    loc.normalize();
    Some(loc)
}

/// Read only own string-valued data properties from a genuine QuickJS Error.
/// Proxies, accessors, inherited properties and object coercions are skipped.
pub unsafe fn js_error_own_string(ctx: *mut JSContext, obj: JSValueConst, key: &CStr) -> Option<String> {
    if qjs::JS_IsError(ctx, obj) == 0 { return None; }
    let atom = qjs::JS_NewAtom(ctx, key.as_ptr());
    if atom == 0 { return None; }
    let mut desc = std::mem::MaybeUninit::<qjs::JSPropertyDescriptor>::uninit();
    let status = qjs::JS_GetOwnProperty(ctx, desc.as_mut_ptr(), obj, atom);
    qjs::JS_FreeAtom(ctx, atom);
    if status < 0 {
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        return None;
    }
    if status == 0 { return None; }
    let desc = desc.assume_init();
    let c = if desc.flags & (1 << 4) == 0 && qjs::is_string(desc.value) {
        qjs::sofuu_js_to_cstring(ctx, desc.value)
    } else { std::ptr::null() };
    let out = if c.is_null() {
        None
    } else {
        let s = CStr::from_ptr(c).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, c);
        Some(s)
    };
    qjs::sofuu_js_free_value(ctx, desc.value);
    qjs::sofuu_js_free_value(ctx, desc.getter);
    qjs::sofuu_js_free_value(ctx, desc.setter);
    out
}

/// Read a primitive exception without calling user-defined conversion hooks.
pub unsafe fn safe_exception_fallback(ctx: *mut JSContext, reason: JSValueConst) -> String {
    if !qjs::is_string(reason) && !qjs::is_number(reason)
        && reason.tag as i32 != qjs::JS_TAG_BOOL && !qjs::is_null(reason)
        && !qjs::is_undefined(reason) {
        return "The operation failed. No further detail was reported.".into();
    }
    let c = qjs::sofuu_js_to_cstring(ctx, reason);
    if c.is_null() { return "The operation failed. No further detail was reported.".into(); }
    let out = CStr::from_ptr(c).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, c);
    out
}

/// Build a diagnostic from a live JavaScript exception. `fallback` is the
/// message the runtime already computed, used when the value has no message.
///
/// # Safety
/// `ctx` and `reason` must belong to a live context.
pub unsafe fn from_js_exception_message(
    ctx: *mut JSContext,
    reason: JSValueConst,
    fallback: &str,
) -> Diagnostic {
    let code = js_error_own_string(ctx, reason, c"code");
    let name = js_error_own_string(ctx, reason, c"name");
    let message = js_error_own_string(ctx, reason, c"message").unwrap_or_else(|| fallback.to_string());
    let stack = js_error_own_string(ctx, reason, c"stack");
    let op = js_error_own_string(ctx, reason, c"op");

    let mut d = match code.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
        Some(code) => Diagnostic::from_code(code, message),
        None => Diagnostic::new(UNCLASSIFIED, Severity::Error, message),
    };
    if let Some(name) = name {
        if name != "Error" {
            d.reason = Some(sanitize_line(
                &format!("{name} raised by the JavaScript execution"),
                MAX_MESSAGE,
            ));
        }
    }
    if let Some(op) = op {
        d.operation = normalize_ident(Some(op), 96);
    }
    if let Some(stack) = stack {
        d.location = first_stack_location(&stack);
        d.frames = stack.lines().map(str::trim).filter(|s| s.starts_with("at "))
            .take(MAX_FRAMES).map(|s| sanitize_line(s, MAX_MESSAGE)).collect();
    }
    if d
        .location
        .as_ref()
        .and_then(|l| l.source.as_deref())
        .is_some_and(|s| s.ends_with(".ts") || s.ends_with(".tsx"))
    {
        if let Some(loc) = d.location.as_mut() {
            loc.generated = true;
            loc.label = Some("type-stripped source; mapping not verified".to_string());
        }
    }
    d
}

/// # Safety
/// `ctx` and `reason` must belong to a live context.
pub unsafe fn from_js_exception(ctx: *mut JSContext, reason: JSValueConst) -> Diagnostic {
    let fallback = safe_exception_fallback(ctx, reason);
    from_js_exception_message(ctx, reason, &fallback)
}

// ── JavaScript surface: sofuu.diagnostics.* ─────────────────────────────────

fn json_string(ctx: *mut JSContext, v: &Value) -> JSValue {
    let text = serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string());
    let cs = CString::new(text).unwrap_or_default();
    unsafe { qjs::sofuu_js_new_string(ctx, cs.as_ptr()) }
}

unsafe fn read_json_arg(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Option<Value> {
    if argc < 1 || argv.is_null() {
        return None;
    }
    let v = qjs::sofuu_js_to_cstring(ctx, *argv);
    if v.is_null() {
        return None;
    }
    let raw = CStr::from_ptr(v).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, v);
    serde_json::from_str::<Value>(&raw).ok()
}

unsafe fn read_text_arg(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Option<String> {
    if argc < 1 || argv.is_null() {
        return None;
    }
    let v = qjs::sofuu_js_to_cstring(ctx, *argv);
    if v.is_null() {
        return None;
    }
    let s = CStr::from_ptr(v).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, v);
    Some(s)
}

unsafe extern "C" fn js_diag_version(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    qjs::sofuu_js_new_int32(ctx, DIAGNOSTIC_VERSION as i32)
}

unsafe extern "C" fn js_diag_render(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let v = read_json_arg(ctx, argc, argv).unwrap_or(Value::Null);
    let text = Diagnostic::from_json(&v).render_human(full_detail_requested(), false);
    let cs = CString::new(text).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_diag_from_error(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || argv.is_null() {
        return json_string(ctx, &Diagnostic::default().to_json());
    }
    let d = unsafe { from_js_exception(ctx, *argv) };
    json_string(ctx, &d.to_json())
}

unsafe extern "C" fn js_diag_redact(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let text = unsafe { read_text_arg(ctx, argc, argv) }.unwrap_or_default();
    let cleaned = sanitize_line(&text, 64 * 1024);
    let cs = CString::new(cleaned).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.diagnostics.*`. Additive: every existing surface and code
/// is unchanged; this is a new, versioned view of the same failures.
pub unsafe fn mod_diag_contract_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }
    let obj = qjs::sofuu_js_new_object(ctx);
    let funcs: [(
        &CStr,
        unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue,
    ); 4] = [
        (c"version", js_diag_version),
        (c"render", js_diag_render),
        (c"fromError", js_diag_from_error),
        (c"redact", js_diag_redact),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"diagnostics".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Diagnostic {
        Diagnostic::from_code("E_KAZE_DIV0", "division by zero")
            .with_location(SourceLocation::at("agent.kaze", 8, Some(18)))
    }

    #[test]
    fn unknown_failure_stays_unclassified() {
        let d = Diagnostic::from_code("E_SOMETHING_NEW", "boom");
        assert_eq!(d.code, "E_SOMETHING_NEW");
        assert!(d.reason.is_none());
        assert!(d.hints.is_empty());
        assert_eq!(d.severity, Severity::Error);
    }

    #[test]
    fn async_kaze_failures_gain_safe_result_and_retry_guidance() {
        let invalid_result = Diagnostic::from_code("E_KAZE_RESULT", "invalid category");
        assert_eq!(invalid_result.subsystem.as_deref(), Some("kaze"));
        assert!(invalid_result.reason.unwrap().contains("outside its contract"));

        let unknown = Diagnostic::from_code("E_KAZE_OUTCOME_UNKNOWN", "result lost");
        assert!(unknown.hints[0].contains("actual state"));
        assert!(unknown.hints[0].contains("retry"));
    }

    #[test]
    fn known_codes_gain_one_explanation_and_one_safe_hint() {
        let d = Diagnostic::from_code("E_CAP_DENIED", "tool call refused");
        assert_eq!(d.severity, Severity::Rejection);
        assert!(d.reason.as_deref().unwrap().contains("policy"));
        assert!(d.hints[0].contains("policy"));
        assert_eq!(d.subsystem.as_deref(), Some("policy"));
        assert!(!d.hints[0].to_ascii_lowercase().contains("bypass"));
    }

    #[test]
    fn sanitizer_redacts_bearer_keys_and_home_paths() {
        let raw = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig \
                   api_key=super-secret-value \
                   /Users/alice/project/file.rs \u{1b}[31mred\u{1b}[0m";
        let clean = redact_text(raw);
        for leak in [
            "eyJhbGciOiJIUzI1NiJ9.payload.sig",
            "super-secret-value",
            "alice",
            "\u{1b}",
        ] {
            assert!(!clean.contains(leak), "leaked {leak:?} in {clean:?}");
        }
        assert!(clean.contains("[REDACTED]"), "{clean}");
        assert!(clean.contains("~/project/file.rs"), "{clean}");
    }

    #[test]
    fn sanitizer_redacts_ordered_and_quoted_values_without_losing_other_text() {
        let raw = r#"password=firstsecret api_key=secondsecret {"password":"a\\\"b-secret","token":"x"} café"#;
        let clean = redact_text(raw);
        for leak in ["firstsecret", "secondsecret", "b-secret", "\"x\""] {
            assert!(!clean.contains(leak), "leaked {leak:?}: {clean:?}");
        }
        assert!(clean.contains("café"), "{clean}");
        assert!(clean.matches("[REDACTED]").count() >= 4, "{clean}");
        assert_eq!(redact_text(&clean), clean);
        assert!(!redact_text("Bearer abc").contains("abc"));
    }

    #[test]
    fn sanitizer_redacts_common_tokens_url_userinfo_and_private_keys() {
        let raw = "sk-abcdefgh123456 https://alice:privatepass@example.test/path\n-----BEGIN PRIVATE KEY-----\nFAKE-PRIVATE-MATERIAL\n-----END PRIVATE KEY-----\nfinished";
        let clean = redact_text(raw);
        for secret in ["abcdefgh123456", "alice", "privatepass", "FAKE-PRIVATE-MATERIAL"] {
            assert!(!clean.contains(secret), "leaked {secret:?}: {clean:?}");
        }
        assert!(clean.contains("example.test/path") && clean.contains("finished"), "{clean}");
        assert_eq!(redact_text(&clean), clean);
    }

    #[test]
    fn event_redaction_keeps_newlines_and_long_nonsecret_values() {
        let raw = format!("first\nsecond\t{}", "x".repeat(70_000));
        assert_eq!(redact_event_text(&raw), raw);
    }

    #[test]
    fn json_locations_and_frames_are_sanitized_and_bounded() {
        let value = json!({"message": "failure", "location": {
            "source": format!("\u{1b}[2J{}", "s".repeat(900)), "line": 7,
            "label": "password=syntheticsecret"
        }, "frames": ["at \u{1b}[31mrun password=syntheticsecret"]});
        let d = Diagnostic::from_json(&value);
        let out = d.render_human(true, false);
        assert!(!out.contains('\u{1b}'), "{out:?}");
        assert!(!out.contains("syntheticsecret"), "{out:?}");
        assert!(d.location.unwrap().source.unwrap().chars().count() <= MAX_MESSAGE);
    }

    #[test]
    fn reported_native_frame_does_not_hide_next_source_frame() {
        let loc = first_stack_location(" at x (native)\n at y (https://host/path.js:42:3)").unwrap();
        assert_eq!(loc.source.as_deref(), Some("https://host/path.js"));
        assert_eq!(loc.line, Some(42));
        assert_eq!(loc.column, Some(3));
    }

    #[test]
    fn token_count_is_not_a_secret() {
        let clean = redact_text("token count: 12, max tokens: 4096");
        assert!(clean.contains("token count: 12"), "{clean}");
    }

    #[test]
    fn source_literals_are_masked_without_moving_the_caret() {
        let src = r#"let result = tool.call("hunter2", question)"#;
        let masked = mask_source_literals(src);
        assert!(!masked.contains("hunter2"), "{masked}");
        assert_eq!(masked.chars().count(), src.chars().count());
        assert!(masked.contains(r#""*******""#), "{masked}");
    }

    #[test]
    fn render_shows_location_reason_and_help_without_inventing_a_column() {
        let d = sample();
        let text = d.render_human(false, false);
        assert!(text.contains("error[E_KAZE_DIV0]"), "{text}");
        assert!(text.contains("--> agent.kaze:8:18"), "{text}");
        assert!(text.contains("= reason:"), "{text}");
        assert!(text.contains("= help:"), "{text}");
        assert!(!text.contains("\u{1b}"), "{text}");

        let no_col = Diagnostic::from_code("E_KAZE_CHECK", "bad")
            .with_location(SourceLocation::at("agent.kaze", 2, None));
        let text = no_col.render_human(false, false);
        assert!(text.contains("agent.kaze:2"), "{text}");
        assert!(!text.contains("^"), "{text}");
    }

    #[test]
    fn missing_source_renders_location_only() {
        let d = Diagnostic::from_code("E_KAZE_PARSE", "bad")
            .with_location(SourceLocation::at("agent.kaze", 3, Some(4)));
        let text = d.render_human(false, false);
        assert!(text.contains("agent.kaze:3:4"), "{text}");
        assert!(!text.contains("|"), "{text}");
    }

    #[test]
    fn truncation_bounds_arrays_and_strings() {
        let mut d = Diagnostic::from_code("E_INTERNAL", "x".repeat(4000));
        for i in 0..50 {
            d = d.with_hint(format!("hint {i}"));
        }
        assert!(d.message.chars().count() <= MAX_MESSAGE);
        assert_eq!(d.hints.len(), MAX_HINTS);
    }

    #[test]
    fn json_keeps_the_observed_fields_and_no_internal_source() {
        let v = sample().to_json();
        assert_eq!(v["diagnostic_version"], 1);
        assert_eq!(v["code"], "E_KAZE_DIV0");
        assert_eq!(v["location"]["line"], 8);
        assert!(v.get("source_text").is_none(), "{v}");
    }

    #[test]
    fn only_a_verified_stack_location_is_taken() {
        let loc = first_stack_location("    at run (agent.kaze:8:18)\n    at x").unwrap();
        assert_eq!(loc.source.as_deref(), Some("agent.kaze"));
        assert_eq!(loc.line, Some(8));
        assert_eq!(loc.column, Some(18));
        assert!(first_stack_location("    at mystery").is_none());
    }

    #[test]
    fn control_characters_and_bidi_are_escaped() {
        let clean = redact_text("a\u{0}b\u{202E}c\u{7}");
        assert!(!clean.contains('\u{0}'));
        assert!(!clean.contains('\u{202E}'));
        assert!(clean.contains('\u{FFFD}'));
    }
}
