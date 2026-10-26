// rt/session_doc.rs — the R4 session document v1 (SPEC-RUNTIME-CONTRACT §5.2).
//
// ONE portable format for a conversation: {schema_version, session, turns,
// events}. Building, validating, redacting, and importing live here so the
// CLI (`sofuu session export|import`) and the JS surface (`sofuu.session`)
// cannot drift — both call the same functions.
//
// JS surface (JSON in, JSON out — same convention as sofuu.events):
//   sofuu.session.schemaVersion                 1
//   sofuu.session.list('{"project":"."}')       the session docs on disk
//   sofuu.session.export('{"project":".","sid":"…"}')   the doc as a string
//   sofuu.session.import('<doc json>')          validate + persist; {ok, id}
//   sofuu.session.sync('{"sid":"…","meta":{…}}') rebuild + write session.json
//
// The document is the portability unit; `events.ndjson` stays the local
// append-only log and continues fresh after an import (the doc embeds the
// reconstructed turns, which is what every surface consumes).

use serde_json::{json, Map, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use std::ffi::CString;
use std::io::Write as _;
use std::os::raw::c_int;
use std::path::PathBuf;

use super::events;

pub const DOC_SCHEMA_VERSION: u64 = 1;

/// Top-level keys the format allows (R4.5: no app-private side channel — an
/// unknown section is a validation error, not tolerated data).
const DOC_KEYS: [&str; 4] = ["schema_version", "session", "turns", "events"];

// ── Build ────────────────────────────────────────────────────────────────────

fn sessions_root(project: &str) -> PathBuf {
    PathBuf::from(project).join(".sofuu").join("sessions")
}

fn doc_path(project: &str, sid: &str) -> PathBuf {
    events::sessions_dir(project, sid).join("session.json")
}

/// Whole-second epoch → RFC3339 (the envelope's ts strings are RFC3339; a
/// synthesized timestamp must not invent a different shape).
fn rfc3339(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Build the §5.2 document for a session: prefer the committed session.json
/// (it is already canonical); otherwise fold the event log. `meta` overrides
/// (from the caller — chat knows model/provider/title at persist time).
pub fn build(project: &str, sid: &str, meta: Option<&Value>) -> Result<Value, String> {
    if sid.is_empty() || sid.contains('/') || sid.contains("..") {
        return Err("invalid sid".into());
    }
    // A caller that passes a turns array (chat's persist) is declaring the
    // fresh truth — the docfile shortcut and the event fold must not win.
    let meta_turns = meta
        .and_then(|m| m.get("turns"))
        .and_then(Value::as_array);
    if meta_turns.is_none() {
        if let Ok(text) = std::fs::read_to_string(doc_path(project, sid)) {
            if let Ok(doc) = serde_json::from_str::<Value>(&text) {
                let errs = validate(&doc);
                if errs.is_empty() {
                    let mut doc = doc;
                    if let Some(m) = meta {
                        apply_meta(&mut doc, m);
                        redact_in_place(&mut doc);
                    }
                    return Ok(doc);
                }
                // A broken doc is rebuilt from the log below — the self-healing
                // path, but it must not be silent (R4: report what was lost).
                eprintln!(
                    "\x1b[33mWarning:\x1b[0m session.json for {sid} failed validation ({}) — rebuilding from the event log",
                    errs.join("; ")
                );
            }
        }
    }

    let internal_turns: Vec<Value> = if let Some(mt) = meta_turns {
        mt.clone()
    } else {
        let rec = events::reconstruct(&json!({"project": project, "sid": sid}));
        match rec.get("turns").and_then(Value::as_array) {
            Some(a) => a.clone(),
            None => return Err(format!("no event log for session {sid}")),
        }
    };

    // Header timestamps + the session-level served_by from the raw events.
    let mut created = String::new();
    let mut updated = String::new();
    let mut served_by = String::new();
    if let Ok(text) = std::fs::read_to_string(events::log_path(project, sid)) {
        for line in text.lines() {
            let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
            if created.is_empty() {
                if let Some(ts) = ev.get("ts").and_then(Value::as_str) {
                    created = ts.to_string();
                }
            }
            if let Some(ts) = ev.get("ts").and_then(Value::as_str) {
                updated = ts.to_string();
            }
            if served_by.is_empty() {
                if let Some(by) = ev.pointer("/payload/by").and_then(Value::as_str) {
                    served_by = by.to_string();
                }
            }
        }
    }
    // The persisting driver (chat) may also carry the header values.
    if let Some(m) = meta {
        for (k, slot) in [("created", &mut created), ("updated", &mut updated)] {
            if let Some(ts) = m.get(k).and_then(Value::as_str) {
                *slot = ts.to_string();
            }
        }
    }

    // Session-level model/provider (used as the usage fallback below).
    let mut model = String::new();
    let mut provider = String::new();
    let mut title = String::new();
    if let Some(m) = meta {
        if let Some(v) = m.get("model").and_then(Value::as_str) {
            model = v.to_string();
        }
        if let Some(v) = m.get("provider").and_then(Value::as_str) {
            provider = v.to_string();
        }
        if let Some(v) = m.get("title").and_then(Value::as_str) {
            title = clip(v, 200);
        }
    }

    let mut turns: Vec<Value> = Vec::new();
    if let Some(doc_turns) = meta_turns {
        // The persisting driver is the turn engine (chat serializes its own
        // turn records); its array is the source of truth. Each turn passes
        // through the same canonicalization the fold path uses so both
        // writers emit one shape (R4.2: "same turns, same usage").
        for (i, t) in doc_turns.iter().enumerate() {
            let stop = t.get("status").and_then(Value::as_str).unwrap_or("done");
            let status = if stop == "cancelled" || stop == "error" { stop } else { "done" };
            let raw_usage = t.get("usage").cloned().unwrap_or_else(|| json!({}));
            let usage = events::canonical_usage(&merge_meta_into_usage(
                &raw_usage, &model, &provider,
            ));
            turns.push(json!({
                "id": t.get("id").cloned().unwrap_or(json!(format!("t_{i}"))),
                "index": i,
                "prompt": t.get("prompt").cloned().unwrap_or(json!("")),
                "answer": t.get("answer").cloned().unwrap_or(json!("")),
                "status": status,
                "usage": usage,
                "tools": t.get("tools").cloned().unwrap_or_else(|| json!([])),
                "served_by": t.get("served_by").cloned().unwrap_or(json!(served_by)),
                "error": t.get("error").cloned().unwrap_or(Value::Null),
            }));
        }
    } else {
        // Event-log fold: reconstruct's turns → the §5.2 shape. Usage
        // normalizes through the T6.7 canonical shape (one usage shape
        // across the format — R4.2 "same usage", not three).
        for (i, t) in internal_turns.iter().enumerate() {
            let stop = t.get("stop_reason").and_then(Value::as_str).unwrap_or("");
            let status = if stop == "cancelled" {
                "cancelled"
            } else if stop == "error" {
                "error"
            } else {
                "done"
            };
            let raw_usage = t.get("usage").cloned().unwrap_or_else(|| json!({}));
            let usage =
                events::canonical_usage(&merge_meta_into_usage(&raw_usage, &model, &provider));
            turns.push(json!({
                "id": t.get("turn_id").cloned().unwrap_or(json!(format!("t_{i}"))),
                "index": i,
                "prompt": t.get("prompt").cloned().unwrap_or(json!("")),
                "answer": t.get("answer").cloned().unwrap_or(json!("")),
                "status": status,
                "usage": usage,
                "tools": t.get("tools").cloned().unwrap_or_else(|| json!([])),
                "served_by": t.get("served_by").cloned().unwrap_or(json!(served_by)),
                "error": Value::Null,
            }));
        }
    }

    let mut sobj = Map::new();
    sobj.insert("id".into(), json!(sid));
    sobj.insert(
        "created".into(),
        json!(if created.is_empty() { rfc3339(now_secs()) } else { created }),
    );
    sobj.insert(
        "updated".into(),
        json!(if updated.is_empty() { rfc3339(now_secs()) } else { updated }),
    );
    sobj.insert("project".into(), json!(clip(project, 200)));
    sobj.insert("title".into(), json!(title));
    sobj.insert("model".into(), json!(model));
    sobj.insert("provider".into(), json!(provider));

    let mut doc = json!({
        "schema_version": DOC_SCHEMA_VERSION,
        "session": Value::Object(sobj),
        "turns": turns,
        "events": {"format": "ndjson", "path": "events.ndjson"},
    });
    redact_in_place(&mut doc);
    Ok(doc)
}

fn merge_meta_into_usage(usage: &Value, model: &str, provider: &str) -> Value {
    let mut u = usage.clone();
    if let Some(o) = u.as_object_mut() {
        let m = o.get("model").and_then(Value::as_str).unwrap_or("");
        if m.is_empty() && !model.is_empty() {
            o.insert("model".into(), json!(model));
        }
        let p = o.get("provider").and_then(Value::as_str).unwrap_or("");
        if p.is_empty() && !provider.is_empty() {
            o.insert("provider".into(), json!(provider));
        }
    }
    u
}

fn apply_meta(doc: &mut Value, meta: &Value) {
    let Some(s) = doc.get_mut("session").and_then(Value::as_object_mut) else { return };
    for k in ["title", "model", "provider", "project", "updated"] {
        if let Some(v) = meta.get(k) {
            s.insert(k.into(), v.clone());
        }
    }
}

// ── Validate ─────────────────────────────────────────────────────────────────

/// R4.5: unknown top-level sections are rejected; every turn carries an id;
/// usage numbers must be numbers. Returns the violations (empty = ok).
pub fn validate(doc: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    if doc.get("schema_version").and_then(Value::as_u64) != Some(DOC_SCHEMA_VERSION) {
        errs.push(format!(
            "schema_version must be {DOC_SCHEMA_VERSION} (got {})",
            doc.get("schema_version")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "absent".into())
        ));
    }
    match doc.as_object() {
        Some(obj) => {
            for k in obj.keys() {
                if !DOC_KEYS.contains(&k.as_str()) {
                    errs.push(format!(
                        "unknown top-level section '{k}' (R4.5: add it to the format, not an app-private channel)"
                    ));
                }
            }
        }
        None => errs.push("document must be a JSON object".into()),
    }
    if doc.pointer("/session/id").and_then(Value::as_str).is_none() {
        errs.push("session.id is required".into());
    }
    match doc.get("turns") {
        None | Some(Value::Null) => {}
        Some(Value::Array(turns)) => {
            for (i, t) in turns.iter().enumerate() {
                if t.get("id").and_then(Value::as_str).is_none() {
                    errs.push(format!("turns[{i}].id is required"));
                }
                if let Some(u) = t.get("usage") {
                    for k in ["in", "out"] {
                        if let Some(v) = u.get(k) {
                            if !v.is_number() {
                                errs.push(format!("turns[{i}].usage.{k} must be a number"));
                            }
                        }
                    }
                }
            }
        }
        Some(_) => errs.push("turns must be an array".into()),
    }
    errs
}

// ── Redaction (R4.6 — the same secret-key walker as R6.6) ────────────────────

pub fn redact_in_place(doc: &mut Value) {
    // events::redact strips secret-bearing STRING fields recursively; applied
    // to the whole document it covers provider keys parked in turns, configs
    // echoed into notes, or any app field that would leak through an export.
    *doc = events::redact(&doc.clone());
}

// ── Persist / import ─────────────────────────────────────────────────────────

/// Write the doc as session.json (0600: turns may carry private text).
pub fn persist(project: &str, sid: &str, doc: &Value) -> Result<PathBuf, String> {
    let dir = events::sessions_dir(project, sid);
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let path = doc_path(project, sid);
    let body = serde_json::to_string_pretty(doc).map_err(|e| format!("encode: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        f.write_all(body.as_bytes())
            .map_err(|e| format!("write: {e}"))?;
    }
    #[cfg(not(unix))]
    std::fs::write(&path, body).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Import a document: validate (typed refusal — silent data loss is a defect,
/// R4.4), redact secrets, persist under its OWN id, return a report.
/// Idempotent: importing the same doc twice yields the same doc + files.
pub fn import(project: &str, doc_text: &str) -> Result<Value, String> {
    let doc: Value = serde_json::from_str(doc_text)
        .map_err(|e| format!("E_SCHEMA: not a JSON document: {e}"))?;
    let errs = validate(&doc);
    if !errs.is_empty() {
        return Err("E_SCHEMA: ".to_string() + &errs.join("; "));
    }
    let sid = doc
        .pointer("/session/id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if sid.is_empty() {
        return Err("E_SCHEMA: session.id is required".into());
    }
    if sid.contains('/') || sid.contains("..") {
        return Err("E_SCHEMA: invalid session.id".into());
    }
    let mut doc = doc;
    redact_in_place(&mut doc);
    // Overwriting a DIFFERENT existing doc with the same id is reported, not
    // silent (R4.4); the write still happens (import is the user's intent).
    let existing = std::fs::read_to_string(doc_path(project, &sid))
        .ok()
        .and_then(|old| serde_json::from_str::<Value>(&old).ok());
    let overwritten = existing.map(|old| old != doc).unwrap_or(false);
    let path = persist(project, &sid, &doc)?;
    Ok(json!({
        "ok": true,
        "id": sid,
        "path": path.display().to_string(),
        "turns": doc.get("turns").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0),
        "overwritten_existing": overwritten,
    }))
}

/// List the session docs on disk ({id, title, updated, turns}; log dirs only —
/// the sibling `<id>.qtsq` mesh payloads are not documents).
pub fn list(project: &str) -> Value {
    let root = sessions_root(project);
    let mut out: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            if !e.path().is_dir() {
                continue;
            }
            let sid = e.file_name().to_string_lossy().into_owned();
            let dp = e.path().join("session.json");
            let Ok(text) = std::fs::read_to_string(&dp) else { continue };
            let Ok(doc) = serde_json::from_str::<Value>(&text) else {
                out.push(json!({"id": sid, "error": "session.json unparseable"}));
                continue;
            };
            out.push(json!({
                "id": doc.pointer("/session/id").cloned().unwrap_or(json!(sid)),
                "title": doc.pointer("/session/title").cloned().unwrap_or(json!("")),
                "updated": doc.pointer("/session/updated").cloned().unwrap_or(json!("")),
                "turns": doc.get("turns").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0),
            }));
        }
    }
    json!({"ok": true, "sessions": out})
}

// ── CLI helpers (called from the bin's `sofuu session` arm) ──────────────────

/// `sofuu session export <sid> [--out <file>]` — the doc to stdout or a file.
/// Secrets are stripped before anything leaves the process (R4.6).
pub fn cli_export(project: &str, sid: &str, out: Option<&str>) -> i32 {
    match build(project, sid, None) {
        Ok(mut doc) => {
            redact_in_place(&mut doc);
            let body = serde_json::to_string_pretty(&doc).unwrap_or_default();
            match out {
                Some(p) => {
                    if let Err(e) = std::fs::write(p, body.as_bytes()) {
                        eprintln!("\x1b[31mError:\x1b[0m cannot write {p}: {e}");
                        1
                    } else {
                        println!("exported {sid} → {p}");
                        0
                    }
                }
                None => {
                    println!("{body}");
                    0
                }
            }
        }
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m export failed: {e}");
            1
        }
    }
}

/// `sofuu session import <file|->` — `-` reads the doc from stdin.
pub fn cli_import(project: &str, src: &str) -> i32 {
    let text = if src == "-" {
        use std::io::Read;
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("\x1b[31mError:\x1b[0m cannot read stdin");
            return 1;
        }
        s
    } else {
        match std::fs::read_to_string(src) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m cannot read {src}: {e}");
                return 1;
            }
        }
    };
    match import(project, &text) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
            0
        }
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m import failed: {e}");
            1
        }
    }
}

// ── JS surface ───────────────────────────────────────────────────────────────

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

unsafe fn read_json_arg(ctx: *mut JSContext, arg: JSValueConst) -> Option<Value> {
    let mut len = 0usize;
    let c = qjs::JS_ToCStringLen2(ctx, &mut len, arg, 0);
    if c.is_null() {
        return None;
    }
    let raw = std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, c);
    serde_json::from_str::<Value>(&raw).ok()
}

fn project_arg(args: &Value) -> String {
    args.get("project").and_then(Value::as_str).unwrap_or(".").to_string()
}

unsafe extern "C" fn js_session_list(
    ctx: *mut JSContext, _t: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = read_json_arg(ctx, *argv)
        .map(|a| list(&project_arg(&a)))
        .unwrap_or_else(|| list("."));
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_session_export(
    ctx: *mut JSContext, _t: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out: String = match read_json_arg(ctx, *argv) {
        Some(a) => {
            let sid = a.get("sid").and_then(Value::as_str).unwrap_or("").to_string();
            match build(&project_arg(&a), &sid, None) {
                Ok(mut d) => {
                    redact_in_place(&mut d);
                    serde_json::to_string(&d).unwrap_or_default()
                }
                Err(e) => json!({"ok": false, "code": "E_SCHEMA", "message": e}).to_string(),
            }
        }
        None => json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object string"}).to_string(),
    };
    let cs = json_str(&out);
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_session_import(
    ctx: *mut JSContext, _t: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out: Value = match read_json_arg(ctx, *argv) {
        Some(d) => match import(".", &d.to_string()) {
            Ok(r) => r,
            Err(e) => json!({"ok": false, "code": "E_SCHEMA", "message": e}),
        },
        None => json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON document string"}),
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_session_sync(
    ctx: *mut JSContext, _t: JSValueConst, _argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out: Value = match read_json_arg(ctx, *argv) {
        Some(a) => {
            let sid = a.get("sid").and_then(Value::as_str).unwrap_or("").to_string();
            let project = project_arg(&a);
            match build(&project, &sid, a.get("meta"))
                .and_then(|doc| {
                    let id = doc.pointer("/session/id").and_then(Value::as_str).unwrap_or("").to_string();
                    let turns = doc.get("turns").and_then(Value::as_array).map(|a| a.len()).unwrap_or(0);
                    persist(&project, &id, &doc).map(|p| {
                        json!({"ok": true, "path": p.display().to_string(), "turns": turns})
                    })
                }) {
                Ok(r) => r,
                Err(e) => json!({"ok": false, "code": "E_INTERNAL", "message": e}),
            }
        }
        None => json!({"ok": false, "code": "E_SCHEMA", "message": "argument must be a JSON object string"}),
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.session.*`. Called from `engine_register_builtins`.
pub unsafe fn mod_session_doc_register(ctx: *mut JSContext) {
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
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"schemaVersion".as_ptr(),
        qjs::sofuu_js_new_int64(ctx, DOC_SCHEMA_VERSION as i64),
    );
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 4] = [
        (c"list", js_session_list),
        (c"export", js_session_export),
        (c"import", js_session_import),
        (c"sync", js_session_sync),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"session".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}
