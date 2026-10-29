// rt/events.rs — the R6 unified event envelope (SPEC-RUNTIME-CONTRACT §7).
//
// One envelope, one validator, one redaction choke point, and one append-only
// event log per session (`<project>/.sofuu/sessions/<sid>/events.ndjson`).
// The substrate is deliberately conservative: it validates and stores events,
// it does not restructure the drivers. chat.js/agent.js stamp the envelope
// fields additively at their emit choke points, and every host can validate
// what it receives through the same code the conformance suite uses.
//
// JS surface (JSON in, JSON out — funnel-friendly):
//   sofuu.events.schema()                       the §7.2 definition
//   sofuu.events.validate(event)                {ok, errors[]}
//   sofuu.events.redact(event)                  secret-bearing fields stripped
//   sofuu.events.append({project, sid, events, path, durable})
//                                               validate + redact + ndjson
//   sofuu.events.read({project, sid, path})     parse + validate the log
//   sofuu.events.reconstruct({project, sid})    fold events into turn records
//
// The 24/7 agent plan writes its run logs (`<project>/.sofuu/logs/`) through
// this same append and reads them back in `rt/observability.rs`: one format,
// one validator, only the directory differs (OBSERVABILITY-SPEC §1/§2).
//
// Phase P2 (R6.1/R6.2/R6.6). This module only reads and appends; it changes
// no engine behavior.

use serde_json::{json, Map, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use std::ffi::CString;
use std::io::{BufRead, Write};
use std::os::raw::c_int;
use std::path::PathBuf;

pub const SCHEMA_VERSION: u64 = 1;

/// The ten kinds the spec REQUIRES (each with payload rules, §7.2).
pub const KINDS: [&str; 10] = [
    "delta", "tool", "tool_result", "usage", "ctx", "approval", "approval_result", "error",
    "served", "stop",
];

/// Kinds the shipped drivers additionally emit, recognized with envelope-only
/// validation (identity fields required, no payload rules). The required ten
/// are a subset; a driver may not invent kinds outside this registry.
pub const KNOWN_KINDS: [&str; 36] = [
    "start", "plan", "think", "answer", "answer_delta", "llm", "delegate", "recall",
    "consolidated", "stored", "learned", "evidence", "schema", "verify", "warn", "mlgate",
    "allocgate", "compact", "done", "approval_request", "approval_resolved",
    "prompt", "tools",
    // OBSERVABILITY plan §Component 1: the agent run vocabulary. Envelope-only
    // validation — the payloads belong to the driver, the envelope is ours.
    "run_start", "turn_start", "tool_call", "model_complete", "memory_recall",
    "memory_store", "checkpoint", "warning", "fatal_error", "retry", "halt",
    "run_complete", "stream_chunk",
];

pub const SURFACES: [&str; 6] = ["js", "cli", "http", "cabi", "swift", "kotlin"];

pub const STOP_REASONS: [&str; 4] = ["done", "cancelled", "budget", "error"];

/// Fields every event carries regardless of kind (SPEC §7.2).
const ENVELOPE_FIELDS: [&str; 7] =
    ["kind", "schema_version", "sid", "turn_id", "seq", "ts", "surface"];

/// Required payload keys per kind. An event may carry them inside `payload`
/// (the canonical shape) or inline at the top level (the additive shape the
/// JS drivers stamp today); the validator accepts both and reports which.
fn required_payload(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "delta" => &[("text", "string")],
        "tool" => &[("id", "string"), ("name", "string")],
        "tool_result" => &[("id", "string"), ("name", "string"), ("ok", "bool")],
        "usage" => &[
            ("in", "number"),
            ("out", "number"),
            ("model", "string"),
            ("provider", "string"),
        ],
        "ctx" => &[("used", "number"), ("limit", "number")],
        "approval" => &[("id", "string"), ("tool", "string")],
        "approval_result" => &[("id", "string"), ("allow", "bool")],
        "error" => &[("code", "string"), ("op", "string"), ("retryable", "bool")],
        "served" => &[("by", "string")],
        "stop" => &[("reason", "string")],
        _ => &[],
    }
}

fn type_ok(v: &Value, want: &str) -> bool {
    match want {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "bool" => v.is_boolean(),
        _ => false,
    }
}

/// Keys whose string values must never reach a log, trace, or export (R6.6).
fn is_secret_key(key: &str) -> bool {
    let k = key.to_lowercase();
    k.contains("apikey") || k.contains("api_key") || k.contains("secret")
        || k.contains("password") || k == "authorization" || k.contains("bearer")
        || k == "token" || k.ends_with("_token")
}

fn redact_value(v: &mut Value) {
    match v {
        Value::Object(map) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                let secret = is_secret_key(&k);
                let entry = map.get_mut(&k).unwrap();
                if secret && entry.is_string() {
                    *entry = Value::String("[REDACTED]".into());
                } else {
                    redact_value(entry);
                }
            }
        }
        Value::Array(items) => {
            for it in items.iter_mut() {
                redact_value(it);
            }
        }
        _ => {}
    }
}

/// Validate one event against §7.2. Returns the list of violations (empty ok).
pub fn validate(ev: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let Some(obj) = ev.as_object() else {
        return vec!["event must be a JSON object".into()];
    };

    for f in ENVELOPE_FIELDS {
        if !obj.contains_key(f) {
            errs.push(format!("missing envelope field `{f}`"));
        }
    }
    if obj.get("schema_version").and_then(Value::as_u64) != Some(SCHEMA_VERSION) {
        errs.push(format!("schema_version must be {SCHEMA_VERSION}"));
    }
    let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
    if !KINDS.contains(&kind) && !KNOWN_KINDS.contains(&kind) {
        errs.push(format!(
            "unknown kind {kind:?} (required: {} | known: {})",
            KINDS.join("|"),
            KNOWN_KINDS.join("|")
        ));
        return errs; // further checks need a known kind
    }
    if obj.get("seq").and_then(Value::as_u64).is_none() {
        errs.push("`seq` must be a non-negative integer".into());
    }
    if let Some(ts) = obj.get("ts").and_then(Value::as_str) {
        // Loose RFC3339 check: date + time separator + zone designator.
        let ok = ts.len() >= 20
            && ts.as_bytes()[4] == b'-'
            && (ts.contains('T') || ts.contains('t'))
            && (ts.ends_with('Z') || ts.ends_with('z') || ts[19..].contains('+'));
        if !ok {
            errs.push(format!("`ts` is not RFC3339: {ts:?}"));
        }
    }
    if let Some(surface) = obj.get("surface").and_then(Value::as_str) {
        if !SURFACES.contains(&surface) {
            errs.push(format!("unknown surface {surface:?}"));
        }
    }
    if obj.get("sid").and_then(Value::as_str).is_none() {
        errs.push("`sid` must be a string".into());
    }
    if obj.get("turn_id").and_then(Value::as_str).is_none() {
        errs.push("`turn_id` must be a string".into());
    }

    // Payload: canonical (inside `payload`) or additive (top level).
    let payload = obj.get("payload").filter(|p| p.is_object());
    for (key, want) in required_payload(kind) {
        let present = payload
            .and_then(|p| p.get(*key))
            .or_else(|| obj.get(*key))
            .map(|v| type_ok(v, want))
            .unwrap_or(false);
        if !present {
            errs.push(format!("{kind} requires payload `{key}` ({want})"));
        }
    }
    if kind == "stop" {
        let reason = obj
            .get("payload")
            .and_then(|p| p.get("reason"))
            .or_else(|| obj.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !STOP_REASONS.contains(&reason) {
            errs.push(format!("stop reason {reason:?} not in {:?}", STOP_REASONS));
        }
    }
    if kind == "served" {
        let by = obj
            .get("payload")
            .and_then(|p| p.get("by"))
            .or_else(|| obj.get("by"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if by != "local" && by != "cloud" {
            errs.push(format!("served.by {by:?} must be local|cloud"));
        }
    }
    errs
}

/// R6.6: strip secret-bearing fields before any log, trace, or export.
pub fn redact(ev: &Value) -> Value {
    let mut out = ev.clone();
    redact_value(&mut out);
    out
}

pub(crate) fn sessions_dir(project: &str, sid: &str) -> PathBuf {
    PathBuf::from(project)
        .join(".sofuu")
        .join("sessions")
        .join(sid)
}

pub(crate) fn log_path(project: &str, sid: &str) -> PathBuf {
    sessions_dir(project, sid).join("events.ndjson")
}

/// Advisory lock held for one append batch. A run log can have two writers at
/// once (a parent agent and the subagent it spawned, or two headless runs
/// sharing a run id); without this the read-tail-then-append sequence races and
/// both writers mint the same seq. Mirrors the output-archive lock so the tree
/// keeps one locking idiom (see output_archive::with_archive_lock).
struct LogLock(#[allow(dead_code)] std::fs::File);

impl LogLock {
    fn acquire(file: &std::fs::File) -> Result<LogLock, String> {
        let handle = file.try_clone().map_err(|e| format!("lock clone: {e}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: the duplicated descriptor stays open for as long as the guard lives.
            if unsafe { libc::flock(handle.as_raw_fd(), libc::LOCK_EX) } != 0 {
                return Err("cannot lock the event log".into());
            }
        }
        #[cfg(windows)]
        {
            if handle.lock().is_err() {
                return Err("cannot lock the event log".into());
            }
        }
        Ok(LogLock(handle))
    }
}

impl Drop for LogLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: same open file description that took the lock.
            let _ = unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
        }
        #[cfg(windows)]
        {
            let _ = self.0.unlock();
        }
    }
}

/// Last seq already on disk, read under the batch lock. Scans the tail first:
/// a 24/7 run log grows to megabytes, and re-reading all of it on every append
/// turns the loop quadratic. Falls back to a full scan only when the tail holds
/// no parseable seq (huge lines, or a log shorter than the window).
fn last_seq_locked(file: &mut std::fs::File) -> u64 {
    use std::io::{Read, Seek, SeekFrom};
    const TAIL: u64 = 64 * 1024;
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return 0;
    }
    let start = len.saturating_sub(TAIL);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return 0;
    }
    let mut buf = String::new();
    if file.read_to_string(&mut buf).is_err() {
        return 0;
    }
    // A tail window that starts mid-file begins mid-line; drop that fragment.
    let text = if start > 0 {
        match buf.find('\n') {
            Some(i) => &buf[i + 1..],
            None => "",
        }
    } else {
        buf.as_str()
    };
    let mut last = 0u64;
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if let Some(s) = v.get("seq").and_then(Value::as_u64) {
                last = s;
            }
        }
    }
    if last == 0 && start > 0 {
        return file
            .seek(SeekFrom::Start(0))
            .ok()
            .and_then(|_| {
                let mut all = String::new();
                file.read_to_string(&mut all).ok().map(|_| all)
            })
            .map(|all| {
                all.lines()
                    .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                    .filter_map(|v| v.get("seq").and_then(Value::as_u64))
                    .last()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
    }
    last
}

/// Kinds whose batch forces `sync_all` even without `durable`: these are the
/// events that move the resume point (checkpoint) or end the run, which is
/// exactly the window the plan's crash-resilience guarantee covers
/// (OBSERVABILITY-SPEC §2).
fn forces_sync(kind: &str) -> bool {
    matches!(
        kind,
        "checkpoint" | "halt" | "fatal_error" | "run_complete" | "stop"
    )
}

/// `run_id` is the plan spelling of the identity `sid` already carries (§0),
/// so it is folded in before validation rather than refused: a plan-shaped
/// event then validates exactly as written.
fn normalise_run_id(ev: &mut Value) {
    let Some(obj) = ev.as_object_mut() else {
        return;
    };
    if obj.get("sid").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
        return;
    }
    let Some(run_id) = obj.get("run_id").and_then(Value::as_str) else {
        return;
    };
    if !run_id.is_empty() {
        obj.insert("sid".into(), json!(run_id));
    }
}

/// Where a batch (or a read) lands: an explicit `path` wins — relative paths
/// resolve against `project`, so a driver can pass
/// `.sofuu/logs/agent-<run_id>.jsonl` unchanged — and the session directory
/// owns the file otherwise.
fn log_target(project: &str, sid: &str, args: &Value) -> Result<PathBuf, Value> {
    let raw = match args.get("path") {
        None => return Ok(log_path(project, sid)),
        Some(v) if v.is_null() => return Ok(log_path(project, sid)),
        Some(v) => v.as_str().unwrap_or(""),
    };
    // `..` would walk out of the project tree: these logs are project-local
    // artifacts, never a general-purpose file writer.
    if raw.is_empty() || raw.contains("..") {
        return Err(json!({"ok": false, "code": "E_SCHEMA", "message": "invalid path"}));
    }
    let p = PathBuf::from(raw);
    Ok(if p.is_absolute() { p } else { PathBuf::from(project).join(p) })
}

/// Validate + redact + append events to a log file: the session's
/// events.ndjson, or the explicit `path` the agent runs write to.
/// Assigns `seq` when absent; rejects regressions (R6.3: total order).
///
/// `durable` forces `sync_all` after the batch; the checkpoint-class kinds
/// force it regardless, because the resume point is the one line that must
/// survive a power loss (OBSERVABILITY plan §Component 1).
pub fn append(args: &Value) -> Value {
    let project = args.get("project").and_then(Value::as_str).unwrap_or(".");
    let sid = args
        .get("sid")
        .and_then(Value::as_str)
        .or_else(|| args.get("run_id").and_then(Value::as_str));
    let Some(sid) = sid else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "sid is required"});
    };
    if sid.contains('/') || sid.contains("..") || sid.is_empty() {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "invalid sid"});
    }
    let Some(events) = args.get("events").and_then(Value::as_array) else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "events array is required"});
    };
    let max_events = args
        .get("max_queue_events")
        .and_then(Value::as_u64)
        .unwrap_or(1024) as usize;
    if events.len() > max_events {
        // R2.4: bounded queues — overflow is a typed error, never silent.
        return json!({"ok": false, "code": "E_BACKPRESSURE",
            "message": format!("event batch exceeds the {} event bound", max_events)});
    }

    let path = match log_target(project, sid, args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && std::fs::create_dir_all(parent).is_err() {
            return json!({"ok": false, "code": "E_INTERNAL", "message": "cannot create the log directory"});
        }
    }
    // read+append: the tail scan below needs to read, and O_APPEND still
    // guarantees every write lands at EOF regardless of the read cursor.
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            return json!({"ok": false, "code": "E_INTERNAL", "message": format!("open: {e}")})
        }
    };
    // Held until this function returns: it makes "read the last seq, then
    // append this batch" atomic against another process doing the same.
    let _lock = match LogLock::acquire(&file) {
        Ok(l) => l,
        Err(e) => return json!({"ok": false, "code": "E_INTERNAL", "message": e}),
    };
    let mut seq = last_seq_locked(&mut file);
    let mut written = 0usize;
    let mut errors: Vec<Value> = Vec::new();
    let mut sync_wanted = args.get("durable").and_then(Value::as_bool).unwrap_or(false);
    for ev in events {
        let mut ev = redact(ev);
        normalise_run_id(&mut ev);
        // Auto-assign seq when the caller did not; reject regressions.
        if ev.get("seq").and_then(Value::as_u64).is_none() {
            seq += 1;
            if let Some(o) = ev.as_object_mut() {
                o.insert("seq".into(), json!(seq));
            }
        } else {
            let s = ev.get("seq").and_then(Value::as_u64).unwrap_or(0);
            if s <= seq {
                errors.push(json!({"seq": s, "error": "seq regression (R6.3 total order)"}));
                continue;
            }
            seq = s;
        }
        let errs = validate(&ev);
        if !errs.is_empty() {
            errors.push(json!({"seq": ev.get("seq"), "errors": errs}));
            continue;
        }
        if let Ok(mut line) = serde_json::to_string(&ev) {
            // ONE write syscall for the whole line. Body-then-newline is two
            // syscalls, and O_APPEND only makes a single write atomic: a
            // concurrent appender slipping between them glues two JSON objects
            // onto one line and both records are lost (stress case A11).
            line.push('\n');
            if file.write_all(line.as_bytes()).is_ok() {
                written += 1;
                // Only a line that actually landed moves the resume point.
                if forces_sync(ev.get("kind").and_then(Value::as_str).unwrap_or("")) {
                    sync_wanted = true;
                }
            }
        }
    }
    // Flush unconditionally: `File` is unbuffered today, so this costs nothing
    // now and keeps "the batch left the process" a property of the writer if
    // the handle ever becomes buffered.
    let _ = file.flush();
    let mut synced = false;
    if sync_wanted {
        if file.sync_all().is_ok() {
            synced = true;
        } else {
            // A failed fsync means the batch is NOT durable; say so instead of
            // reporting `ok`.
            errors.push(json!({"error": "sync_all failed"}));
        }
    }
    json!({
        "ok": errors.is_empty(),
        "path": path.display().to_string(),
        "written": written,
        "seq_end": seq,
        "errors": errors,
        "synced": synced,
    })
}

/// Read + validate a session's event log, or the explicit `path` a run log
/// lives at (same resolution rules as `append`).
pub fn read(args: &Value) -> Value {
    let project = args.get("project").and_then(Value::as_str).unwrap_or(".");
    let sid = args
        .get("sid")
        .and_then(Value::as_str)
        .or_else(|| args.get("run_id").and_then(Value::as_str));
    let Some(sid) = sid else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "sid is required"});
    };
    let path = match log_target(project, sid, args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let Ok(f) = std::fs::File::open(&path) else {
        return json!({"ok": true, "events": [], "count": 0, "path": path.display().to_string()});
    };
    let reader = std::io::BufReader::new(f);
    let mut events: Vec<Value> = Vec::new();
    let mut bad = 0usize;
    for line in reader.lines().map_while(Result::ok) {
        match serde_json::from_str::<Value>(&line) {
            Ok(v) => {
                if !validate(&v).is_empty() {
                    bad += 1;
                }
                events.push(v);
            }
            Err(_) => bad += 1,
        }
    }
    json!({"ok": bad == 0, "events": events, "count": events.len(),
           "invalid": bad, "path": path.display().to_string()})
}

/// R6.2: fold the event log into turn records — proof the log alone
/// reconstructs what happened (answers, tools, usage, terminal state).
pub fn reconstruct(args: &Value) -> Value {
    let res = read(args);
    let Some(events) = res.get("events").and_then(Value::as_array) else {
        return res;
    };
    let mut order: Vec<String> = Vec::new();
    let mut turns: Map<String, Value> = Map::new();
    for ev in events {
        let turn = ev.get("turn_id").and_then(Value::as_str).unwrap_or("?").to_string();
        if !turns.contains_key(&turn) {
            order.push(turn.clone());
            turns.insert(
                turn.clone(),
                json!({"turn_id": turn, "answer": "", "tools": [], "usage": {"in": 0, "out": 0}}),
            );
        }
        let t = turns.get_mut(&turn).unwrap();
        let Some(obj) = t.as_object_mut() else { continue };
        let kind = ev.get("kind").and_then(Value::as_str).unwrap_or("");
        let p = ev.get("payload").cloned().unwrap_or_else(|| ev.clone());
        match kind {
            "delta" => {
                let text = p.get("text").and_then(Value::as_str).unwrap_or("");
                let cur = obj.get("answer").and_then(Value::as_str).unwrap_or("").to_string();
                obj.insert("answer".into(), json!(format!("{cur}{text}")));
            }
            "answer" => {
                /* The chat driver emits one terminal `answer` event carrying
                 * the full turn text alongside the deltas (P2). Idempotent:
                 * a turn that already folded its deltas would double the
                 * text, so only use it when the turn has no folded text. */
                let text = p.get("answer").and_then(Value::as_str)
                    .or_else(|| p.get("text").and_then(Value::as_str)).unwrap_or("");
                let cur = obj.get("answer").and_then(Value::as_str).unwrap_or("");
                if cur.is_empty() && !text.is_empty() {
                    obj.insert("answer".into(), json!(text));
                }
            }
            "prompt" => {
                if !obj.get("prompt").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
                    obj.insert("prompt".into(), p.get("text").cloned().unwrap_or(json!("")));
                }
            }
            "served" => {
                obj.insert("served_by".into(), p.get("by").cloned().unwrap_or(json!("")));
            }
            "tool" | "tool_result" => {
                let name = p.get("name").and_then(Value::as_str).unwrap_or("?");
                let id = p.get("id").and_then(Value::as_str).unwrap_or("?");
                let tools = obj.get_mut("tools").and_then(Value::as_array_mut).unwrap();
                if let Some(existing) = tools.iter_mut().find(|t| t.get("id").and_then(Value::as_str) == Some(id)) {
                    if kind == "tool_result" {
                        existing["ok"] = p.get("ok").cloned().unwrap_or(json!(false));
                        existing["ms"] = p.get("ms").cloned().unwrap_or(json!(0));
                    }
                } else {
                    tools.push(json!({"id": id, "name": name, "ok": null, "ms": 0}));
                }
            }
            "usage" => {
                let i = p.get("in").and_then(Value::as_u64).unwrap_or(0);
                let o = p.get("out").and_then(Value::as_u64).unwrap_or(0);
                let u = obj.get_mut("usage").unwrap();
                u["in"] = json!(u["in"].as_u64().unwrap_or(0) + i);
                u["out"] = json!(u["out"].as_u64().unwrap_or(0) + o);
            }
            "stop" => {
                obj.insert("stop_reason".into(), p.get("reason").cloned().unwrap_or(json!("?")));
                obj.insert("terminal".into(), json!(true));
            }
            _ => {}
        }
    }
    let turns: Vec<Value> = order.iter().filter_map(|k| turns.remove(k)).collect();
    json!({"ok": res["ok"].clone(), "turns": turns, "count": turns.len()})
}

/// T6.7: normalize the three producer usage shapes into the §7.2 payload.
/// Accepts `{in,out}` (canonical), `{pt,ct}` (chat persisted log), and
/// `{promptTokens,completionTokens}` (agent/ai) — plus model/provider/cost
/// under any of their aliases. Cost defaults to 0 (local semantics).
pub fn canonical_usage(args: &Value) -> Value {
    let pick = |names: &[&str]| -> Value {
        for n in names {
            if let Some(v) = args.get(*n) {
                if v.is_number() {
                    return v.clone();
                }
            }
        }
        json!(0)
    };
    let str_pick = |names: &[&str]| -> String {
        for n in names {
            if let Some(v) = args.get(*n) {
                if let Some(s) = v.as_str() {
                    if !s.is_empty() {
                        return s.to_string();
                    }
                }
            }
        }
        "unknown".into()
    };
    json!({
        "in": pick(&["in", "in_tokens", "pt", "promptTokens", "prompt_tokens"]),
        "out": pick(&["out", "out_tokens", "ct", "completionTokens", "completion_tokens"]),
        "model": str_pick(&["model"]),
        "provider": str_pick(&["provider", "served_by", "servedBy"]),
        "cost_usd": pick(&["cost_usd", "costUsd", "cost"]),
    })
}

/// T6.3: the deterministic text trace renderer — one line per event, seq
/// order, stable field order. This is the render contract every client
/// shares; a golden fixture (contract/fixtures/golden-trace.*) pins it.
pub fn render(events: &[Value]) -> String {
    let mut sorted: Vec<&Value> = events.iter().collect();
    sorted.sort_by_key(|e| e.get("seq").and_then(Value::as_u64).unwrap_or(0));
    let mut out = String::new();
    for e in sorted {
        let seq = e.get("seq").and_then(Value::as_u64).unwrap_or(0);
        let kind = e.get("kind").and_then(Value::as_str).unwrap_or("?");
        let sid = e.get("sid").and_then(Value::as_str).unwrap_or("?");
        let turn = e.get("turn_id").and_then(Value::as_str).unwrap_or("?");
        let payload = e.get("payload").cloned().unwrap_or_else(|| e.clone());
        let detail = match kind {
            "delta" => format!("text={:?}", payload.get("text").and_then(Value::as_str).unwrap_or("")),
            "tool" => format!("id={} name={}", fs(&payload, "id"), fs(&payload, "name")),
            "tool_result" => format!(
                "id={} ok={} ms={}",
                fs(&payload, "id"),
                payload.get("ok").and_then(Value::as_bool).unwrap_or(false),
                payload.get("ms").and_then(Value::as_u64).unwrap_or(0)
            ),
            "usage" => format!(
                "in={} out={} model={} provider={}",
                payload.get("in").and_then(Value::as_u64).unwrap_or(0),
                payload.get("out").and_then(Value::as_u64).unwrap_or(0),
                fs(&payload, "model"),
                fs(&payload, "provider")
            ),
            "stop" => format!("reason={}", fs(&payload, "reason")),
            "served" => format!("by={}", fs(&payload, "by")),
            "ctx" => format!("used={} limit={}", payload.get("used").and_then(Value::as_u64).unwrap_or(0), payload.get("limit").and_then(Value::as_u64).unwrap_or(0)),
            _ => {
                // Deterministic JSON (serde_json maps sort keys by default).
                serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
            }
        };
        out.push_str(&format!("{:04} {:<14} {}/{} {}
", seq, kind, sid, turn, detail));
    }
    out
}

fn fs(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("?").to_string()
}

/// The §7.2 definition itself, for hosts and the conformance suite.
pub fn schema() -> Value {
    let mut payload = Map::new();
    for k in KINDS {
        let fields: Vec<Value> = required_payload(k)
            .iter()
            .map(|(f, t)| json!({"name": f, "type": t}))
            .collect();
        payload.insert(k.to_string(), Value::Array(fields));
    }
    json!({
        "schema_version": SCHEMA_VERSION,
        "envelope_fields": ENVELOPE_FIELDS,
        "required_kinds": KINDS,
        "known_kinds": &KNOWN_KINDS[..],
        "surfaces": SURFACES,
        "stop_reasons": STOP_REASONS,
        "required_payload": payload,
        "log_format": "ndjson",
        "log_path": "<project>/.sofuu/sessions/<sid>/events.ndjson",
    })
}

// ── JS surface ───────────────────────────────────────────────────────────────

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

/// One-JSON-argument convention (mirrors policy.rs): the host passes a JSON
/// object or string; a non-object argument is reported as a typed error.
unsafe fn read_json_arg(ctx: *mut JSContext, arg: JSValueConst) -> Option<Value> {
    let mut len = 0usize;
    let c = qjs::JS_ToCStringLen2(ctx, &mut len, arg, 0);
    if c.is_null() {
        return None;
    }
    let raw = std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, c);
    serde_json::from_str::<Value>(&raw).ok().or_else(|| {
        // A bare object literal cannot arrive through JS_ToCString; treat a
        // parse failure as absent so the caller reports E_SCHEMA.
        None
    })
}

unsafe extern "C" fn js_events_schema(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, _argv: *const JSValueConst,
) -> JSValue {
    let cs = json_str(&serde_json::to_string(&schema()).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_validate(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = match read_json_arg(ctx, *argv) {
        Some(ev) => {
            let errs = validate(&ev);
            json!({"ok": errs.is_empty(), "errors": errs})
        }
        None => json!({"ok": false, "errors": ["argument must be a JSON event object"]}),
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_redact(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv).map(|ev| redact(&ev)).unwrap_or_else(|| {
        json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON event object"})
    });
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_append(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| append(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_read(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| read(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_reconstruct(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| reconstruct(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_render(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let events: Vec<Value> = read_json_arg(ctx, *argv)
        .filter(|a| !a.is_null())
        .and_then(|a| {
            if a.is_array() {
                Some(a.as_array().cloned().unwrap_or_default())
            } else {
                a.get("events").and_then(Value::as_array).cloned()
            }
        })
        .unwrap_or_default();
    let cs = json_str(&render(&events));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_events_canonical_usage(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = if argc < 1 {
        canonical_usage(&json!({}))
    } else {
        read_json_arg(ctx, *argv)
            .map(|a| canonical_usage(&a))
            .unwrap_or_else(|| json!({"in": 0, "out": 0, "model": "unknown", "provider": "unknown", "cost_usd": 0}))
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.events.*`. Called from `engine_register_builtins`.
pub unsafe fn mod_events_register(ctx: *mut JSContext) {
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
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 8] = [
        (c"schema", js_events_schema),
        (c"validate", js_events_validate),
        (c"redact", js_events_redact),
        (c"append", js_events_append),
        (c"read", js_events_read),
        (c"reconstruct", js_events_reconstruct),
        (c"render", js_events_render),
        (c"canonicalUsage", js_events_canonical_usage),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"events".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(kind: &str, turn: &str, seq: u64, payload: Value) -> Value {
        json!({
            "kind": kind, "schema_version": 1, "sid": "s_test",
            "turn_id": turn, "seq": seq, "surface": "js",
            "ts": "2026-10-25T12:00:00Z", "payload": payload,
        })
    }

    #[test]
    fn accepts_a_valid_event_of_every_kind() {
        let events = vec![
            ("delta", json!({"text": "hi"})),
            ("tool", json!({"id": "c1", "name": "read_file"})),
            ("tool_result", json!({"id": "c1", "name": "read_file", "ok": true, "ms": 3})),
            ("usage", json!({"in": 2, "out": 4, "model": "m", "provider": "local"})),
            ("ctx", json!({"used": 10, "limit": 100})),
            ("approval", json!({"id": "a1", "tool": "bash"})),
            ("approval_result", json!({"id": "a1", "allow": false})),
            ("error", json!({"code": "E_TIMEOUT", "op": "ai.stream", "retryable": true})),
            ("served", json!({"by": "local"})),
            ("stop", json!({"reason": "done"})),
        ];
        for (i, (kind, payload)) in events.iter().enumerate() {
            let e = ev(kind, "t1", i as u64 + 1, payload.clone());
            let errs = validate(&e);
            assert!(errs.is_empty(), "{kind}: {errs:?}");
        }
    }

    #[test]
    fn rejects_unknown_kind_and_missing_payload() {
        let mut e = ev("wat", "t1", 1, json!({}));
        assert!(!validate(&e).is_empty());
        e = ev("delta", "t1", 1, json!({})); // delta requires text
        let errs = validate(&e);
        assert!(errs.iter().any(|x| x.contains("text")));
        e["seq"] = json!("not-a-number");
        assert!(!validate(&e).is_empty());
    }

    #[test]
    fn redaction_strips_secret_fields_recursively() {
        let mut e = ev("tool", "t1", 1, json!({
            "id": "c1", "name": "net",
            "api_key": "sk-live-123", "Authorization": "Bearer abc",
            "nested": {"session_token": "t0k", "note": "keep me"}
        }));
        redact_value(&mut e);
        let s = serde_json::to_string(&e).unwrap();
        assert!(!s.contains("sk-live-123") && !s.contains("Bearer abc") && !s.contains("t0k"));
        assert!(s.contains("[REDACTED]"));
        assert!(s.contains("keep me"));
    }

    #[test]
    fn append_read_reconstruct_roundtrip() {
        let dir = std::env::temp_dir().join(format!("sofuu-ev-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let project = dir.to_str().unwrap().to_string();
        let args = json!({"project": project, "sid": "s1", "events": [
            ev("delta", "t1", 1, json!({"text": "Hel"})),
            ev("delta", "t1", 2, json!({"text": "lo"})),
            ev("tool", "t1", 3, json!({"id": "c1", "name": "grep"})),
            ev("tool_result", "t1", 4, json!({"id": "c1", "name": "grep", "ok": true, "ms": 5})),
            ev("usage", "t1", 5, json!({"in": 3, "out": 2, "model": "m", "provider": "local"})),
            ev("stop", "t1", 6, json!({"reason": "done"})),
        ]});
        let res = append(&args);
        assert_eq!(res["ok"], true, "{}", res);
        assert_eq!(res["written"], 6);

        let read = read(&json!({"project": project, "sid": "s1"}));
        assert_eq!(read["count"], 6);
        assert_eq!(read["ok"], true);

        let rec = reconstruct(&json!({"project": project, "sid": "s1"}));
        assert_eq!(rec["turns"][0]["answer"], "Hello");
        assert_eq!(rec["turns"][0]["tools"][0]["name"], "grep");
        assert_eq!(rec["turns"][0]["tools"][0]["ok"], true);
        assert_eq!(rec["turns"][0]["usage"]["out"], 2);
        assert_eq!(rec["turns"][0]["stop_reason"], "done");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_is_deterministic_and_seq_ordered() {
        let events = vec![
            ev("usage", "t1", 5, json!({"in": 3, "out": 2, "model": "m", "provider": "local"})),
            ev("delta", "t1", 1, json!({"text": "Hel"})),
            ev("stop", "t1", 6, json!({"reason": "done"})),
        ];
        let r1 = render(&events);
        let r2 = render(&events);
        assert_eq!(r1, r2, "render must be deterministic");
        let lines: Vec<&str> = r1.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("0001 delta"), "seq order: {}", lines[0]);
        assert!(lines[1].starts_with("0005 usage"));
        assert!(lines[2].starts_with("0006 stop"));
        assert!(lines[2].contains("reason=done"));
    }

    #[test]
    fn canonical_usage_unifies_the_three_producer_shapes() {
        let a = canonical_usage(&json!({"in": 3, "out": 4, "model": "m", "provider": "local"}));
        let b = canonical_usage(&json!({"pt": 3, "ct": 4, "model": "m"}));
        let c = canonical_usage(&json!({"promptTokens": 3, "completionTokens": 4}));
        assert_eq!(a["in"], b["in"]);
        assert_eq!(a["out"], c["out"]);
        assert_eq!(a["cost_usd"], 0, "local cost semantics");
        let cloud = canonical_usage(&json!({"promptTokens": 5, "completionTokens": 7, "costUsd": 0.0135}));
        assert_eq!(cloud["cost_usd"], 0.0135);
    }

    #[test]
    fn seq_regression_is_rejected_not_written() {
        let dir = std::env::temp_dir().join(format!("sofuu-ev2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let project = dir.to_str().unwrap().to_string();
        let res = append(&json!({"project": project, "sid": "s1", "events": [
            ev("delta", "t1", 5, json!({"text": "a"})),
            ev("delta", "t1", 3, json!({"text": "b"})), // regression
        ]}));
        assert_eq!(res["ok"], false);
        assert_eq!(res["written"], 1);
        assert_eq!(res["errors"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
