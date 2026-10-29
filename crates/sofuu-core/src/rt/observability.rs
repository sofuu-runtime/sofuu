// rt/observability.rs — reading a run back: `sofuu logs`, `sofuu resume`, and
// the crash post-mortem card (OBSERVABILITY plan §Component 3/4/5, contract
// OBSERVABILITY-SPEC §2).
//
// `rt/events.rs` stays the only writer of the envelope: this module reads the
// same lines back and answers the four questions a crashed daemon leaves
// behind — what ran, where it stopped, what state is resumable, and why it
// died. No second format appears here, so the CLI renders exactly what the
// conformance suite validates.
//
// JS surface (JSON in, JSON out — funnel-friendly):
//   sofuu.observability.runs({project})           one row per run log
//   sofuu.observability.lastCheckpoint({...})     newest resumable checkpoint
//   sofuu.observability.postmortem({...})         write a crash card
//   sofuu.observability.crashCard({...})          newest card for a run
//
// OBSERVABILITY lane A. Reads only; no engine behavior change.

use serde_json::{json, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use std::ffi::CString;
use std::io::{BufRead, Write};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn sofuu_dir(project: &str) -> PathBuf {
    PathBuf::from(project).join(".sofuu")
}

/// `<project>/.sofuu/logs` — the plan's agent run logs (contract §1).
fn logs_dir(project: &str) -> PathBuf {
    sofuu_dir(project).join("logs")
}

/// `<project>/.sofuu/crashes` — one JSON card per crash (contract §1).
fn crashes_dir(project: &str) -> PathBuf {
    sofuu_dir(project).join("crashes")
}

fn sessions_root(project: &str) -> PathBuf {
    sofuu_dir(project).join("sessions")
}

fn project_arg(args: &Value) -> String {
    args.get("project").and_then(Value::as_str).unwrap_or(".").to_string()
}

/// An id ends up inside a file name, so it must be exactly one path segment —
/// the rule `events::append` already applies to `sid`. A `..` or `/` here
/// would let a derived run id address files outside the project tree.
fn valid_id(id: &str) -> bool {
    !id.is_empty() && !id.contains('/') && !id.contains("..")
}

/// Whole-second epoch -> civil UTC. The same conversion `session_doc` uses:
/// the envelope's `ts` strings are RFC3339, and a timestamp we synthesize
/// must not invent a second shape.
fn utc_parts(secs: u64) -> (i64, u64, u64, u64, u64, u64) {
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo as u64, d as u64, h, mi, s)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn iso_utc(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `YYYYMMDD_HHMMSS` — the compact stamp the plan's crash ids carry
/// (`crash_20261029_145000_bot01`).
fn compact_utc(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(secs);
    format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}")
}

fn file_mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Every line of a log that parses as JSON. A missing file is an empty log; a
/// truncated tail line (kill -9 mid-write) does not parse and is dropped
/// rather than aborting the scan — the surviving lines are the whole point.
fn read_events(path: &Path) -> Vec<Value> {
    let Ok(f) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
        if let Ok(v) = serde_json::from_str::<Value>(&line) {
            out.push(v);
        }
    }
    out
}

/// The file a run reads from: the explicit `path` when the caller has one
/// (relative paths resolve against `project`, as in `events::append`), else
/// the plan's `agent-<run_id>.jsonl`.
fn run_log_path(project: &str, run_id: &str, args: &Value) -> Option<PathBuf> {
    match args.get("path") {
        Some(v) if !v.is_null() => {
            let raw = v.as_str().unwrap_or("");
            if raw.is_empty() || raw.contains("..") {
                return None;
            }
            let p = PathBuf::from(raw);
            Some(if p.is_absolute() { p } else { PathBuf::from(project).join(p) })
        }
        _ if valid_id(run_id) => Some(logs_dir(project).join(format!("agent-{run_id}.jsonl"))),
        _ => None,
    }
}

/// The furthest step the log reached, or the number of checkpoints when the
/// driver never stamped `payload.step`: a run that crashed before its first
/// step still reports progress instead of zero.
fn steps_of(events: &[Value]) -> u64 {
    let mut max = 0u64;
    let mut checkpoints = 0u64;
    for ev in events {
        let step = ev
            .get("payload")
            .and_then(|p| p.get("step"))
            .and_then(Value::as_u64)
            .or_else(|| ev.get("step").and_then(Value::as_u64));
        if let Some(s) = step {
            max = max.max(s);
        }
        if ev.get("kind").and_then(Value::as_str) == Some("checkpoint") {
            checkpoints += 1;
        }
    }
    if max > 0 { max } else { checkpoints }
}

/// Contract §5: one status vocabulary, derived from the last terminal event.
/// `retry` only decides the answer while no terminal event has landed — a run
/// that retried twice and then completed is OK, not RETRYING.
fn status_of(events: &[Value]) -> &'static str {
    let mut terminal: Option<&'static str> = None;
    let mut retried = false;
    for ev in events {
        match ev.get("kind").and_then(Value::as_str).unwrap_or("") {
            "run_complete" | "stop" => terminal = Some("OK"),
            "halt" | "fatal_error" => terminal = Some("HALTED"),
            "retry" => retried = true,
            _ => {}
        }
    }
    match terminal {
        Some(s) => s,
        None if retried => "RETRYING",
        None => "RUNNING",
    }
}

fn files_with_suffix(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.to_string_lossy().ends_with(suffix))
        .collect();
    out.sort();
    out
}

/// One `runs` row. `mtime` is epoch SECONDS, not an ISO string: the shipped
/// renderer (src/js/observability.js) formats it as a number, and the run
/// file is created when the run starts, which is the STARTED column.
fn run_row(path: &Path, run_id: &str, kind: &str) -> (u64, String, Value) {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mtime = file_mtime_secs(path);
    let events = read_events(path);
    let row = json!({
        "run_id": run_id,
        "path": path.display().to_string(),
        "kind": kind,
        "bytes": bytes,
        "mtime": mtime,
        "steps": steps_of(&events),
        "status": status_of(&events),
    });
    (mtime, run_id.to_string(), row)
}

/// `sofuu logs`: every run log in the project, newest first. Both kinds are
/// discovered because contract §1 keeps one envelope and only splits the
/// directory: `<.sofuu/logs/*.jsonl>` and `<.sofuu/sessions/*/events.ndjson>`.
pub fn runs(args: &Value) -> Value {
    let project = project_arg(args);
    let mut rows: Vec<(u64, String, Value)> = Vec::new();
    for path in files_with_suffix(&logs_dir(&project), ".jsonl") {
        let run_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.strip_prefix("agent-").unwrap_or(s).to_string())
            .unwrap_or_default();
        rows.push(run_row(&path, &run_id, "agent"));
    }
    if let Ok(rd) = std::fs::read_dir(sessions_root(&project)) {
        for entry in rd.filter_map(Result::ok) {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let Some(sid) = dir.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let log = dir.join("events.ndjson");
            if !log.is_file() {
                continue;
            }
            rows.push(run_row(&log, sid, "session"));
        }
    }
    // Newest first; the run id breaks mtime ties so the order is stable.
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let runs: Vec<Value> = rows.into_iter().map(|(_, _, row)| row).collect();
    let count = runs.len();
    json!({"ok": true, "runs": runs, "count": count})
}

/// The newest line that still validates as a `checkpoint` event. The plan's
/// resume contract is "scan the log in reverse until the latest valid
/// checkpoint" (§Component 4), so a half-written tail line is skipped, never
/// fatal.
pub fn last_checkpoint(args: &Value) -> Value {
    let project = project_arg(args);
    let Some(run_id) = args.get("run_id").and_then(Value::as_str) else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "run_id is required"});
    };
    if !valid_id(run_id) {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "invalid run_id"});
    }
    let Some(path) = run_log_path(&project, run_id, args) else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "invalid path"});
    };
    let events = read_events(&path);
    for ev in events.iter().rev() {
        if ev.get("kind").and_then(Value::as_str) != Some("checkpoint") {
            continue;
        }
        // The writer's own validator decides what "valid" means, so resume
        // can never rehydrate a line the conformance suite would reject.
        if !crate::rt::events::validate(ev).is_empty() {
            continue;
        }
        let payload = ev.get("payload").cloned().unwrap_or_else(|| json!({}));
        let step = payload
            .get("step")
            .and_then(Value::as_u64)
            .or_else(|| ev.get("step").and_then(Value::as_u64));
        return json!({
            "ok": true,
            "found": true,
            "step": step,
            "seq": ev.get("seq").cloned().unwrap_or(Value::Null),
            "ts": ev.get("ts").cloned().unwrap_or(Value::Null),
            "checkpoint": payload,
        });
    }
    json!({
        "ok": true, "found": false, "step": null, "seq": null, "ts": null,
        "checkpoint": null,
    })
}

/// The normative error taxonomy (contract §0) plus the transport names
/// `async_contract::CODES` declares. Copied rather than imported so the set of
/// codes a card may carry is fixed by the contract, not by whichever module
/// happens to grow next.
const ERROR_CODES: [&str; 13] = [
    "E_SCHEMA", "E_TIMEOUT", "E_RATE_LIMIT", "E_DENIED", "E_CIRCUIT_OPEN",
    "E_TOOL", "E_CAPABILITY", "E_BACKPRESSURE", "E_INTERNAL", "E_CANCELLED",
    "E_AUTH", "E_NETWORK", "E_PROTOCOL",
];

/// Contract §0: a card's `code` comes from the taxonomy. The plan doc predates
/// it and still shows the old names, so fold those in; anything unrecognised
/// maps to E_INTERNAL exactly like `async_contract::make_error` does, because
/// inventing a code is the one thing that is not allowed.
fn canonical_error_code(code: &str) -> &'static str {
    let folded = match code {
        "E_NETWORK_TIMEOUT" => "E_TIMEOUT",
        "E_POLICY_DENIED" => "E_DENIED",
        "E_SERVER_ERROR" | "E_UNCAUGHT_THROW" => "E_INTERNAL",
        c => c,
    };
    ERROR_CODES
        .iter()
        .find(|c| **c == folded)
        .copied()
        .unwrap_or("E_INTERNAL")
}

/// Write the plan's §Component 3 diagnosis card and return it. This is the
/// fail-stop / halt artifact that `sofuu logs --last-error` renders, so its
/// shape is the plan's, key for key.
pub fn postmortem(args: &Value) -> Value {
    let project = project_arg(args);
    let Some(run_id) = args.get("run_id").and_then(Value::as_str) else {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "run_id is required"});
    };
    if !valid_id(run_id) {
        return json!({"ok": false, "code": "E_SCHEMA", "message": "invalid run_id"});
    }
    let now = now_secs();
    let crash_id = format!("crash_{}_{}", compact_utc(now), run_id);
    let err = args.get("error").cloned().unwrap_or_else(|| json!({}));
    let ctx = args.get("context").cloned().unwrap_or_else(|| json!({}));
    // No hint from the driver is no excuse for a card that tells the operator
    // nothing: name the command that rehydrates the run.
    let recovery_hint = args
        .get("recovery_hint")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            format!("Run `sofuu resume {run_id}` to resume from the last checkpoint.")
        });
    let mut card = json!({
        "crash_id": crash_id,
        "run_id": run_id,
        "timestamp": iso_utc(now),
        "failed_step": args.get("step").cloned().unwrap_or(json!(0)),
        "error": {
            "code": canonical_error_code(err.get("code").and_then(Value::as_str).unwrap_or("")),
            "message": err.get("message").and_then(Value::as_str).unwrap_or(""),
            "details": err.get("details").cloned().unwrap_or_else(|| json!({})),
            "stack": err.get("stack").and_then(Value::as_str).unwrap_or(""),
        },
        "context_snapshot": {
            "active_model": ctx.get("model").and_then(Value::as_str).unwrap_or(""),
            "context_tokens_used": ctx.get("context_tokens").cloned().unwrap_or(json!(0)),
            "last_user_message": ctx.get("last_user_message").and_then(Value::as_str).unwrap_or(""),
            "last_tool_in_flight": ctx.get("last_tool").and_then(Value::as_str).unwrap_or(""),
        },
        "recovery_hint": recovery_hint,
    });
    // The plan card has no provider or trace slot, but the args carry both:
    // keep them only when the caller supplied them, so the default card stays
    // exactly the §Component 3 shape.
    if let Some(provider) = ctx.get("provider").and_then(Value::as_str).filter(|p| !p.is_empty()) {
        card["context_snapshot"]["active_provider"] = json!(provider);
    }
    if let Some(trace) = args.get("trace_path").and_then(Value::as_str).filter(|t| !t.is_empty()) {
        card["trace_path"] = json!(trace);
    }

    let dir = crashes_dir(&project);
    if std::fs::create_dir_all(&dir).is_err() {
        return json!({"ok": false, "code": "E_INTERNAL", "message": "cannot create the crash directory"});
    }
    let path = dir.join(format!("{crash_id}.json"));
    let text = match serde_json::to_string_pretty(&card) {
        Ok(t) => t,
        Err(e) => {
            return json!({"ok": false, "code": "E_INTERNAL", "message": format!("serialize: {e}")})
        }
    };
    // A card is written on the way to a halt, so it is fsynced before we
    // report success: a power loss right after must not be what loses it.
    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            return json!({"ok": false, "code": "E_INTERNAL", "message": format!("create: {e}")})
        }
    };
    if file.write_all(format!("{text}\n").as_bytes()).is_err() || file.sync_all().is_err() {
        return json!({"ok": false, "code": "E_INTERNAL", "message": "cannot write the crash card"});
    }
    json!({
        "ok": true,
        "crash_id": crash_id,
        "path": path.display().to_string(),
        "card": card,
    })
}

/// The newest crash card, optionally for one run. Newest is the card's own
/// `timestamp` (it names when the crash happened); mtime breaks a tie between
/// two cards written in the same second.
pub fn crash_card(args: &Value) -> Value {
    let project = project_arg(args);
    let filter = args.get("run_id").and_then(Value::as_str);
    if let Some(id) = filter {
        if !valid_id(id) {
            return json!({"ok": false, "code": "E_SCHEMA", "message": "invalid run_id"});
        }
    }
    let mut best: Option<(String, u64, Value)> = None;
    for path in files_with_suffix(&crashes_dir(&project), ".json") {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(card) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(id) = filter {
            if card.get("run_id").and_then(Value::as_str) != Some(id) {
                continue;
            }
        }
        let ts = card.get("timestamp").and_then(Value::as_str).unwrap_or("").to_string();
        let mtime = file_mtime_secs(&path);
        let newer = match &best {
            None => true,
            Some((best_ts, best_mtime, _)) => (&ts, mtime) > (best_ts, *best_mtime),
        };
        if newer {
            best = Some((ts, mtime, card));
        }
    }
    match best {
        Some((_, _, card)) => json!({"ok": true, "card": card}),
        None => json!({"ok": true, "card": null}),
    }
}

// ── JS surface ───────────────────────────────────────────────────────────────

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

/// One-JSON-argument convention (mirrors events.rs): the host passes a JSON
/// string. A bare JS object is stringified first, so a script can hand over
/// `{...}` without pre-serialising it.
unsafe fn read_json_arg(ctx: *mut JSContext, arg: JSValueConst) -> Option<Value> {
    let mut len = 0usize;
    let c = qjs::JS_ToCStringLen2(ctx, &mut len, arg, 0);
    if !c.is_null() {
        let raw = std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, c);
        if let Ok(v) = serde_json::from_str::<Value>(&raw) {
            return Some(v);
        }
    }
    if qjs::is_object(arg) {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let json_obj = qjs::sofuu_js_get_property_str(ctx, global, c"JSON".as_ptr());
        let stringify_fn = qjs::sofuu_js_get_property_str(ctx, json_obj, c"stringify".as_ptr());
        let mut argv = [arg];
        let res = qjs::JS_Call(ctx, stringify_fn, json_obj, 1, argv.as_mut_ptr());
        let mut slen = 0usize;
        let sc = qjs::JS_ToCStringLen2(ctx, &mut slen, res, 0);
        let mut val = None;
        if !sc.is_null() {
            let sraw = std::ffi::CStr::from_ptr(sc).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, sc);
            val = serde_json::from_str::<Value>(&sraw).ok();
        }
        qjs::sofuu_js_free_value(ctx, res);
        qjs::sofuu_js_free_value(ctx, stringify_fn);
        qjs::sofuu_js_free_value(ctx, json_obj);
        qjs::sofuu_js_free_value(ctx, global);
        return val;
    }
    None
}

unsafe extern "C" fn js_observability_runs(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| runs(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_observability_last_checkpoint(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| last_checkpoint(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_observability_postmortem(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| postmortem(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_observability_crash_card(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|args| crash_card(&args))
        .unwrap_or_else(|| json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object"}));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.observability.*`. Called from `engine_register_builtins`
/// directly after `sofuu.events`, whose files it reads.
pub unsafe fn mod_observability_register(ctx: *mut JSContext) {
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
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 4] = [
        (c"postmortem", js_observability_postmortem),
        (c"lastCheckpoint", js_observability_last_checkpoint),
        (c"runs", js_observability_runs),
        (c"crashCard", js_observability_crash_card),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"observability".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A private project dir per test: cargo runs tests in parallel threads of
    /// one process, so the pid alone does not make the name unique.
    fn project_dir(tag: &str) -> (PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("sofuu-obs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let project = dir.to_str().unwrap().to_string();
        (dir, project)
    }

    fn ev(kind: &str, sid: &str, seq: u64, payload: Value) -> Value {
        json!({
            "kind": kind, "schema_version": 1, "sid": sid, "turn_id": "t1",
            "seq": seq, "surface": "js", "ts": "2026-10-29T12:00:00Z",
            "payload": payload,
        })
    }

    /// The plan-shaped envelope: `run_id`, no `sid` — append has to normalise
    /// it before the validator sees the event.
    fn plan_ev(kind: &str, run_id: &str, seq: u64, payload: Value) -> Value {
        json!({
            "kind": kind, "schema_version": 1, "run_id": run_id, "turn_id": "t1",
            "seq": seq, "surface": "js", "ts": "2026-10-29T12:00:00Z",
            "payload": payload,
        })
    }

    /// Write through the real writer: the tests must hit the same path, seq and
    /// validation rules the drivers hit.
    fn append_run(project: &str, run_id: &str, events: Vec<Value>) -> Value {
        crate::rt::events::append(&json!({
            "project": project, "sid": run_id,
            "path": format!(".sofuu/logs/agent-{run_id}.jsonl"),
            "events": events,
        }))
    }

    #[test]
    fn postmortem_roundtrip_writes_the_plan_card() {
        let (dir, project) = project_dir("pm");
        let res = postmortem(&json!({
            "project": project,
            "run_id": "bot-01",
            "step": 14,
            "error": {
                "code": "E_SCHEMA",
                "message": "tool transfer_funds failed argument validation",
                "details": {"missing_fields": ["recipient_account"], "received_payload": {"amount": 100}},
                "stack": "Error: E_SCHEMA\n  at runTool (plugins.js:142)\n  at Agent.execute (agent.js:89)",
            },
            "context": {
                "model": "llama3.2",
                "provider": "local",
                "context_tokens": 4210,
                "last_user_message": "Send $100 to Alice",
                "last_tool": "transfer_funds",
            },
            "recovery_hint": "Run `sofuu resume bot-01` after updating the tool schema.",
            "trace_path": ".sofuu/logs/agent-bot-01.jsonl",
        }));
        assert_eq!(res["ok"], true, "{res}");
        let crash_id = res["crash_id"].as_str().unwrap();
        assert!(crash_id.starts_with("crash_"), "{crash_id}");
        assert!(crash_id.ends_with("_bot-01"), "{crash_id}");
        let card = &res["card"];
        assert_eq!(card["run_id"], "bot-01");
        assert_eq!(card["failed_step"], 14);
        assert_eq!(card["error"]["code"], "E_SCHEMA");
        assert_eq!(card["error"]["details"]["missing_fields"][0], "recipient_account");
        assert_eq!(card["context_snapshot"]["active_model"], "llama3.2");
        assert_eq!(card["context_snapshot"]["active_provider"], "local");
        assert_eq!(card["context_snapshot"]["context_tokens_used"], 4210);
        assert_eq!(card["context_snapshot"]["last_user_message"], "Send $100 to Alice");
        assert_eq!(card["context_snapshot"]["last_tool_in_flight"], "transfer_funds");
        assert_eq!(card["trace_path"], ".sofuu/logs/agent-bot-01.jsonl");
        // The advertised path must be the file that actually holds the card.
        let path = PathBuf::from(res["path"].as_str().unwrap());
        assert!(path.starts_with(dir.join(".sofuu").join("crashes")), "{}", path.display());
        let on_disk: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(&on_disk, card, "the card on disk is the card returned");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn last_checkpoint_picks_the_newest_valid_checkpoint() {
        let (dir, project) = project_dir("cp");
        let wrote = append_run(&project, "a1", vec![
            plan_ev("run_start", "a1", 1, json!({"model": "llama3.2"})),
            plan_ev("checkpoint", "a1", 2, json!({"step": 1, "task": "one"})),
            plan_ev("turn_start", "a1", 3, json!({"text": "go"})),
            plan_ev("checkpoint", "a1", 4, json!({"step": 2, "task": "two"})),
            plan_ev("checkpoint", "a1", 5, json!({"step": 3, "task": "three"})),
        ]);
        assert_eq!(wrote["ok"], true, "{wrote}");
        assert_eq!(wrote["written"], 5);
        // What a kill -9 leaves behind: a half-written tail line. It must be
        // skipped, not returned as the newest checkpoint.
        let log = PathBuf::from(&project).join(".sofuu").join("logs").join("agent-a1.jsonl");
        let mut text = std::fs::read_to_string(&log).unwrap();
        text.push_str("{\"kind\":\"checkpoint\",\"seq\":6,\"pay");
        std::fs::write(&log, text).unwrap();

        let res = last_checkpoint(&json!({"project": project, "run_id": "a1"}));
        assert_eq!(res["ok"], true, "{res}");
        assert_eq!(res["found"], true);
        assert_eq!(res["step"], 3);
        assert_eq!(res["seq"], 5);
        assert_eq!(res["checkpoint"]["task"], "three");
        // A run with no checkpoint is "not found", never an error.
        let none = last_checkpoint(&json!({"project": project, "run_id": "nope"}));
        assert_eq!(none["ok"], true);
        assert_eq!(none["found"], false);
        assert!(none["checkpoint"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runs_derives_status_and_steps_from_each_log() {
        let (dir, project) = project_dir("runs");
        append_run(&project, "a1", vec![
            ev("run_start", "a1", 1, json!({})),
            ev("checkpoint", "a1", 2, json!({"step": 1})),
            ev("checkpoint", "a1", 3, json!({"step": 2})),
            ev("run_complete", "a1", 4, json!({"steps": 2})),
        ]);
        append_run(&project, "a2", vec![
            ev("run_start", "a2", 1, json!({})),
            ev("fatal_error", "a2", 2, json!({"step": 4, "code": "E_TOOL"})),
            ev("halt", "a2", 3, json!({"step": 4})),
        ]);
        append_run(&project, "a3", vec![
            ev("retry", "a3", 1, json!({"attempt": 1, "delay_ms": 500})),
        ]);
        // A chat session log is discovered beside the agent runs with the same
        // derivation: contract §1 keeps one envelope and splits only the dir.
        let session = crate::rt::events::append(&json!({"project": project, "sid": "chat1",
            "events": [ev("stop", "chat1", 1, json!({"reason": "done"}))]}));
        assert_eq!(session["ok"], true, "{session}");

        let res = runs(&json!({"project": project}));
        assert_eq!(res["ok"], true, "{res}");
        let rows = res["runs"].as_array().unwrap();
        assert_eq!(rows.len(), 4, "{res}");
        let find = |id: &str| rows.iter().find(|r| r["run_id"] == id).unwrap().clone();
        assert_eq!(find("a1")["kind"], "agent");
        assert_eq!(find("a1")["status"], "OK");
        assert_eq!(find("a1")["steps"], 2);
        assert!(find("a1")["bytes"].as_u64().unwrap() > 0);
        // The renderer does arithmetic on mtime, so it must stay a number.
        assert!(find("a1")["mtime"].is_number());
        assert!(find("a1")["path"].as_str().unwrap().ends_with("agent-a1.jsonl"));
        assert_eq!(find("a2")["status"], "HALTED");
        assert_eq!(find("a2")["steps"], 4);
        assert_eq!(find("a3")["status"], "RETRYING");
        assert_eq!(find("chat1")["kind"], "session");
        assert_eq!(find("chat1")["status"], "OK");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn crash_card_returns_the_newest_and_filters_by_run() {
        let (dir, project) = project_dir("card");
        let crashes = dir.join(".sofuu").join("crashes");
        std::fs::create_dir_all(&crashes).unwrap();
        // Explicit timestamps: "newest" must be a property of the card, not of
        // the test wall clock.
        let write = |name: &str, card: Value| {
            let text = serde_json::to_string(&card).unwrap();
            std::fs::write(crashes.join(name), text).unwrap();
        };
        write("crash_old.json", json!({"crash_id": "crash_old", "run_id": "r1",
            "timestamp": "2026-10-29T14:00:00Z", "failed_step": 1}));
        write("crash_new.json", json!({"crash_id": "crash_new", "run_id": "r2",
            "timestamp": "2026-10-29T15:00:00Z", "failed_step": 2}));

        assert_eq!(crash_card(&json!({"project": project}))["card"]["crash_id"], "crash_new");
        assert_eq!(
            crash_card(&json!({"project": project, "run_id": "r1"}))["card"]["crash_id"],
            "crash_old"
        );
        assert!(crash_card(&json!({"project": project, "run_id": "nobody"}))["card"].is_null());
        // A card written by postmortem is found by the same reader.
        let pm = postmortem(&json!({"project": project, "run_id": "r3", "step": 7,
            "error": {"code": "E_TIMEOUT", "message": "deadline"}}));
        assert_eq!(pm["ok"], true, "{pm}");
        let found = crash_card(&json!({"project": project, "run_id": "r3"}));
        assert_eq!(found["card"]["crash_id"], pm["crash_id"]);
        assert_eq!(found["card"]["failed_step"], 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_batches_sync_and_delta_batches_do_not() {
        let (dir, project) = project_dir("sync");
        let path = ".sofuu/logs/agent-a1.jsonl";
        let delta = crate::rt::events::append(&json!({"project": project, "sid": "a1",
            "path": path, "events": [ev("delta", "a1", 1, json!({"text": "hi"}))]}));
        assert_eq!(delta["ok"], true, "{delta}");
        assert_eq!(delta["synced"], false, "a delta batch is not fsynced: {delta}");
        // The resume point is the one line a power loss may not eat (§2).
        let cp = crate::rt::events::append(&json!({"project": project, "sid": "a1",
            "path": path, "events": [ev("checkpoint", "a1", 2, json!({"step": 1}))]}));
        assert_eq!(cp["ok"], true, "{cp}");
        assert_eq!(cp["synced"], true, "a checkpoint batch must fsync: {cp}");
        // And `durable` buys the same guarantee for any other kind.
        let forced = crate::rt::events::append(&json!({"project": project, "sid": "a1",
            "path": path, "durable": true,
            "events": [ev("delta", "a1", 3, json!({"text": "more"}))]}));
        assert_eq!(forced["synced"], true, "durable must fsync: {forced}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_error_codes_fold_into_the_taxonomy() {
        assert_eq!(canonical_error_code("E_NETWORK_TIMEOUT"), "E_TIMEOUT");
        assert_eq!(canonical_error_code("E_POLICY_DENIED"), "E_DENIED");
        assert_eq!(canonical_error_code("E_SERVER_ERROR"), "E_INTERNAL");
        assert_eq!(canonical_error_code("E_UNCAUGHT_THROW"), "E_INTERNAL");
        assert_eq!(canonical_error_code("E_TOOL"), "E_TOOL");
        assert_eq!(canonical_error_code("E_CIRCUIT_OPEN"), "E_CIRCUIT_OPEN");
        // A made-up code never reaches a card.
        assert_eq!(canonical_error_code("E_WAT"), "E_INTERNAL");
    }
}
