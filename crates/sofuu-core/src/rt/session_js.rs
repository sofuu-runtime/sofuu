// rt/session_js.rs — synchronous session-mesh primitives for shipped JS
// (PLAN-DESKTOP workstream C).
//
// src/js/chat.js owns the session mesh for embedded hosts (desktop): it
// writes the plaintext `registry.json` and the per-session `.qtsq` data
// files in exactly the format the CLI's session.rs produces, so TUI and
// desktop sessions see each other in the same project mesh. Two things the
// JS side cannot do itself:
//
//   - the `.qtsq` codec (compression, plus legacy encrypt/decrypt) lives in
//     the QTSQ C library —
//     exposed here through sofuu-ffi's safe wrappers (which degrade to
//     "no QTSQ linked" failures instead of crashing);
//   - `sofuu.fs.readFile/writeFile` are ASYNC (libuv thread pool), but the
//     desktop serves `sofuu.chat.sessionTurns(id)` synchronously through a
//     blocking engine query — the mesh reads/writes below must be sync too.
//
// Registered globals (all synchronous):
//   __qtsq_session_save(path, json, [legacy-password-ignored]) -> 0 | -1
//   __qtsq_session_load(path, password)       -> string | null
//   __session_write_file(path, text)          -> bool   (mkdirs parents)
//   __session_read_file(path)                 -> string | null
//   __session_project_root()                  -> string | null
//
// The third save argument is accepted for compatibility with older shipped
// JS. New records are plaintext QTSQ cache entries; load still accepts the
// derived password for legacy encrypted files.

use std::os::raw::c_int;

use sofuu_ffi::bridge::{js_new_string, register_global_fn, JSCFunction};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

/// `__qtsq_session_save(path, json, [legacy-password-ignored])` → 0 on success,
/// -1 on any failure (missing QTSQ linkage, bad path, codec error).
unsafe extern "C" fn js_qtsq_session_save(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::sofuu_js_new_int32(ctx, -1);
    }
    // SAFETY: argv holds argc live values on this ctx.
    let args = unsafe { std::slice::from_raw_parts(argv, 2) };
    let path = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[0]) };
    let data = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[1]) };
    let (Some(path), Some(data)) = (path, data) else {
        return qjs::sofuu_js_new_int32(ctx, -1);
    };
    let rc = sofuu_ffi::qtsq_session_save(&path, data.as_bytes());
    qjs::sofuu_js_new_int32(ctx, if rc == 0 { 0 } else { -1 })
}

/// `__qtsq_session_load(path, password)` → JSON string, or null when the file
/// is missing, a legacy password is wrong, or QTSQ is not linked.
unsafe extern "C" fn js_qtsq_session_load(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::sofuu_js_null();
    }
    // SAFETY: argv holds argc live values on this ctx.
    let args = unsafe { std::slice::from_raw_parts(argv, 2) };
    let path = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[0]) };
    let pw = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[1]) };
    let (Some(path), Some(pw)) = (path, pw) else {
        return qjs::sofuu_js_null();
    };
    match sofuu_ffi::qtsq_session_load(&path, &pw) {
        Some(bytes) => match String::from_utf8(bytes) {
            Ok(s) => unsafe { js_new_string(ctx, &s) },
            Err(_) => qjs::sofuu_js_null(),
        },
        None => qjs::sofuu_js_null(),
    }
}

/// `__session_write_file(path, text)` → bool; creates parent directories
/// (the mesh dir may not exist yet on a project's first session).
unsafe extern "C" fn js_session_write_file(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::sofuu_js_new_int32(ctx, 0);
    }
    // SAFETY: argv holds argc live values on this ctx.
    let args = unsafe { std::slice::from_raw_parts(argv, 2) };
    let path = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[0]) };
    let text = unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[1]) };
    let (Some(path), Some(text)) = (path, text) else {
        return qjs::sofuu_js_new_int32(ctx, 0);
    };
    let p = std::path::Path::new(&path);
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Atomic-ish write: tmp file + rename in the same directory, matching
    // session.rs's registry write discipline (readers never see a torn file).
    // P3: the tmp name carries pid + subsec-nanos — a fixed `{name}.tmp`
    // sibling let two processes writing the same session rename each
    // other's half-written tmp (torn reads / lost writes).
    let tmp = match p.file_name() {
        Some(name) => p.with_file_name(format!(
            "{}.{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        )),
        None => p.to_path_buf(),
    };
    let ok = std::fs::write(&tmp, text.as_bytes()).is_ok() && std::fs::rename(&tmp, p).is_ok();
    qjs::sofuu_js_new_int32(ctx, if ok { 1 } else { 0 })
}

/// `__session_read_file(path)` → file contents as a string, or null.
unsafe extern "C" fn js_session_read_file(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_null();
    }
    // SAFETY: argv holds argc live values on this ctx.
    let args = unsafe { std::slice::from_raw_parts(argv, 1) };
    let Some(path) = (unsafe { sofuu_ffi::bridge::js_to_string(ctx, args[0]) }) else {
        return qjs::sofuu_js_null();
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => unsafe { js_new_string(ctx, &s) },
        Err(_) => qjs::sofuu_js_null(),
    }
}

/// `__session_project_root()` → the session-mesh root for the current
/// directory, derived EXACTLY like session.rs::project_root: $SOFUU_PROJECT
/// → git toplevel → cwd. chat.js must hash the same root string the Rust
/// mesh code does, or the per-project .qtsq password diverges and the two
/// hosts can't read each other's session files (symlink canonicalization
/// like /tmp → /private/tmp makes naive string equality fail).
unsafe extern "C" fn js_session_project_root(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let root = match std::env::var("SOFUU_PROJECT") {
        Ok(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
        _ => match std::process::Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()
        {
            Ok(out) if out.status.success() => {
                let s = String::from_utf8_lossy(&out.stdout);
                let p = std::path::PathBuf::from(s.trim());
                if p.as_os_str().is_empty() {
                    std::env::current_dir().unwrap_or_default()
                } else {
                    p
                }
            }
            _ => std::env::current_dir().unwrap_or_default(),
        },
    };
    let s = root.to_string_lossy();
    if s.is_empty() {
        return qjs::sofuu_js_null();
    }
    unsafe { js_new_string(ctx, &s) }
}

/// `__chat_note_no_think(model)` — persist "this model cannot think" into
/// the chat config (P2-4): the shipped chat.js driver learns the same fact
/// the TUI does, so the learning survives restarts and the model picker
/// reports it. Read-modify-write of the config's `no_think_models` array —
/// the same file chat.rs's ChatConfig persists, resolved through the same
/// config-root logic (embedded hosts redirect). chat.rs is bin-only, so
/// the field is patched by value here.
fn note_no_think_persist(model: &str) -> bool {
    use serde_json::Value;
    if model.is_empty() {
        return false;
    }
    let home = crate::embed_config::home_dir().unwrap_or_else(|| ".".to_string());
    let dir = match crate::embed_config::get_config_root() {
        Some(root) => std::path::PathBuf::from(root),
        None => std::path::PathBuf::from(home).join(".sofuu"),
    };
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("config.json");
    let mut root: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let Some(obj) = root.as_object_mut() else {
        return false;
    };
    let mut list: Vec<String> = obj
        .get("no_think_models")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    if list.iter().any(|m| m == model) {
        return true; // already known
    }
    list.push(model.to_string());
    obj.insert(
        "no_think_models".into(),
        serde_json::to_value(&list).unwrap_or(Value::Array(Vec::new())),
    );
    // tmp+rename like ChatConfig::save, 0600 (config carries the API key).
    let tmp = dir.join("config.json.tmp");
    let wrote = (|| -> std::io::Result<()> {
        use std::io::Write;
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)?;
            f.write_all(serde_json::to_string(&root)?.as_bytes())?;
        }
        #[cfg(not(unix))]
        {
            std::fs::write(&tmp, serde_json::to_string(&root)?)?;
        }
        std::fs::rename(&tmp, &path)
    })();
    wrote.is_ok()
}

unsafe extern "C" fn js_chat_note_no_think(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 0);
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let model = unsafe { std::ffi::CStr::from_ptr(path_ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, path_ptr) };
    let ok = note_no_think_persist(&model);
    unsafe { qjs::sofuu_js_new_bool(ctx, if ok { 1 } else { 0 }) }
}

/// Register the session primitives. Called from engine_register_builtins;
/// unconditional (no has_qtsq gate) because the ffi wrappers already degrade
/// gracefully when the QTSQ codec is not linked.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn mod_session_js_register(ctx: *mut JSContext) {
    // SAFETY: ctx is live; every native is 'static.
    unsafe {
        register_global_fn(ctx, "__qtsq_session_save", js_qtsq_session_save as JSCFunction);
        register_global_fn(ctx, "__qtsq_session_load", js_qtsq_session_load as JSCFunction);
        register_global_fn(ctx, "__session_write_file", js_session_write_file as JSCFunction);
        register_global_fn(ctx, "__session_read_file", js_session_read_file as JSCFunction);
        register_global_fn(ctx, "__session_project_root", js_session_project_root as JSCFunction);
        register_global_fn(ctx, "__chat_note_no_think", js_chat_note_no_think as JSCFunction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::TEST_LOOP_LOCK;

    #[test]
    fn session_file_roundtrip_and_qtsq_degrade() {
        // The loop-global lock serializes with every other engine-booting test.
        let _guard = TEST_LOOP_LOCK.lock().unwrap();
        let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");

        let dir = std::env::temp_dir().join(format!("sofuu-session-js-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested/registry.json");
        let path_js = serde_json::to_string(&path.to_string_lossy()).unwrap();

        // Sync write (mkdirs parents) + read round-trip.
        let script = format!(
            "var ok = __session_write_file({path_js}, '{{\"sessions\":[]}}'); \
             if (ok !== 1 && ok !== true) throw new Error('write failed: ' + ok); \
             var back = __session_read_file({path_js}); \
             if (back !== '{{\"sessions\":[]}}') throw new Error('read mismatch: ' + back); \
             if (__session_read_file('/nonexistent/nope.json') !== null) throw new Error('expected null');"
        );
        assert_eq!(rt.eval_string(&script, "<session-js-test>"), 0);

        // QTSQ save/load: with the codec linked this round-trips; without
        // it the natives degrade to -1/null (never a crash).
        let qtsq_path = dir.join("s-test.qtsq");
        let qtsq_js = serde_json::to_string(&qtsq_path.to_string_lossy()).unwrap();
        let script = format!(
            "var rc = __qtsq_session_save({qtsq_js}, '{{\"schema\":1}}', 'legacy-pw'); \
             if (rc === 0) {{ \
               var loaded = __qtsq_session_load({qtsq_js}, 'pw-test'); \
               if (loaded !== '{{\"schema\":1}}') throw new Error('qtsq roundtrip failed: ' + loaded); \
               if (__qtsq_session_load({qtsq_js}, 'wrong-pw') !== '{{\"schema\":1}}') throw new Error('plaintext qtsq must not require a password'); \
             }} else {{ \
               if (rc !== -1) throw new Error('expected -1 without qtsq, got ' + rc); \
               if (__qtsq_session_load({qtsq_js}, 'pw-test') !== null) throw new Error('expected null without qtsq'); \
             }}"
        );
        assert_eq!(rt.eval_string(&script, "<session-qtsq-test>"), 0);
        if let Ok(meta) = std::fs::metadata(&qtsq_path) {
            let raw = std::fs::read(&qtsq_path).expect("read qtsq test record");
            assert!(meta.len() < 1000, "plaintext session record unexpectedly large: {} bytes", meta.len());
            assert_eq!(raw.get(4), Some(&1), "small record should use compact QTSQ v1 header");
            assert_eq!(raw.get(5), Some(&0), "small record should use raw strategy");
            let flags = u16::from_le_bytes([raw[6], raw[7]]);
            assert_eq!(flags & 0x0010, 0, "small record must not be encrypted");
            assert_eq!(raw.len(), 80 + b"{\"schema\":1}".len());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
