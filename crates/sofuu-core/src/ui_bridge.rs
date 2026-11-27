//! Zero-static-bloat dynamic loader & JS host bridge for `libsofuu_gpu.{dylib,so}` (`--ui`).
//!
//! `sofuu-core` has ZERO compile-time dependency on `sofuu-ui` or `sofuu-gpu`.
//! When `--ui` is passed on `sofuu run` or compiled standalone agent binaries, this module:
//!   1. Dynamically loads `libsofuu_gpu.{dylib,so}` via `dlopen()`.
//!   2. Hands Thread 0 (main OS thread) to `sofuu_ui_run_json(...)` for AppKit/Metal/GLES.
//!   3. Executes the Sofuu JS/agent runtime on Thread 1 while forwarding JSON `ScenePatch`
//!      updates and polling semantic `UIEvent` payloads cleanly across threads.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};
use std::sync::Arc;

use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSContext, JSValue,
    JSValueConst,
};
use sofuu_ffi::qjs;

pub const EXPECTED_UI_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type WorkerCb = extern "C" fn(session: *mut c_void, user_ctx: *mut c_void);
type RunJsonFn = unsafe extern "C" fn(
    title_utf8: *const c_char,
    width: f32,
    height: f32,
    initial_scene_json: *const u8,
    initial_scene_len: usize,
    worker_cb: Option<WorkerCb>,
    user_ctx: *mut c_void,
) -> i32;
type PushPatchJsonFn =
    unsafe extern "C" fn(session: *mut c_void, patch_json: *const u8, patch_len: usize) -> i32;
type PollEventJsonFn =
    unsafe extern "C" fn(session: *mut c_void, out_buf: *mut u8, out_cap: usize) -> i64;

static ACTIVE_SESSION: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ACTIVE_PUSH_FN: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static ACTIVE_POLL_FN: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());
static UI_MOUNTED: AtomicBool = AtomicBool::new(false);
static WINDOW_CLOSED_SEEN: AtomicBool = AtomicBool::new(false);

/// Strip `--ui` (before an optional `--` separator) and return `(enabled, remaining_args)`.
pub fn take_ui_arg(args: &[String]) -> (bool, Vec<String>) {
    let mut enabled = false;
    let mut out = Vec::with_capacity(args.len());
    let mut passthrough = false;
    for a in args {
        if a == "--" {
            passthrough = true;
            out.push(a.clone());
            continue;
        }
        if !passthrough && a == "--ui" {
            enabled = true;
            continue;
        }
        out.push(a.clone());
    }
    (enabled, out)
}

#[cfg(target_os = "macos")]
const UI_LIB_FILENAME: &str = "libsofuu_gpu.dylib";
#[cfg(target_os = "windows")]
const UI_LIB_FILENAME: &str = "sofuu_gpu.dll";
#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const UI_LIB_FILENAME: &str = "libsofuu_gpu.so";

/// Search ordered candidate paths where `libsofuu_gpu` may live.
pub fn candidate_ui_lib_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(env_p) = std::env::var("SOFUU_UI_LIB") {
        let trimmed = env_p.trim();
        if !trimmed.is_empty() {
            paths.push(PathBuf::from(trimmed));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join(UI_LIB_FILENAME));
            paths.push(dir.join("lib").join(UI_LIB_FILENAME));
            if let Some(parent) = dir.parent() {
                paths.push(parent.join("release").join(UI_LIB_FILENAME));
                paths.push(parent.join("debug").join(UI_LIB_FILENAME));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join(UI_LIB_FILENAME));
        paths.push(cwd.join("lib").join(UI_LIB_FILENAME));
        paths.push(cwd.join("target/release").join(UI_LIB_FILENAME));
        paths.push(cwd.join("target/debug").join(UI_LIB_FILENAME));
    }
    if let Ok(home) = std::env::var("HOME") {
        paths.push(PathBuf::from(home).join(".sofuu/lib").join(UI_LIB_FILENAME));
    }
    paths
}

struct LoadedUiLib {
    #[cfg(unix)]
    handle: *mut c_void,
    run_json: RunJsonFn,
    push_patch_json: PushPatchJsonFn,
    poll_event_json: PollEventJsonFn,
}

#[cfg(unix)]
impl Drop for LoadedUiLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

#[cfg(unix)]
fn load_ui_lib() -> Result<LoadedUiLib, String> {
    let explicit_env = std::env::var("SOFUU_UI_LIB")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let candidates = if let Some(ref p) = explicit_env {
        vec![PathBuf::from(p)]
    } else {
        candidate_ui_lib_paths()
    };
    let mut handle: *mut c_void = ptr::null_mut();
    let mut last_err = String::new();

    for p in &candidates {
        if !p.exists() {
            continue;
        }
        let Ok(cpath) = CString::new(p.to_string_lossy().as_bytes()) else {
            continue;
        };
        let h = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !h.is_null() {
            handle = h;
            break;
        }
        let err_ptr = unsafe { libc::dlerror() };
        if !err_ptr.is_null() {
            last_err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy().into_owned();
        }
    }

    if handle.is_null() && explicit_env.is_none() {
        // Final fallback: let dynamic linker search OS library paths by filename.
        if let Ok(cpath) = CString::new(UI_LIB_FILENAME) {
            let h = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !h.is_null() {
                handle = h;
            }
        }
    }

    if handle.is_null() {
        let tried: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
        let detail = if last_err.is_empty() {
            format!("searched: {}", tried.join(", "))
        } else {
            last_err
        };
        return Err(format!(
            "E_UI_LIB_MISSING: could not load {UI_LIB_FILENAME} ({detail}). Build with `cargo build -p sofuu-gpu` or set SOFUU_UI_LIB."
        ));
    }

    unsafe fn sym<T>(handle: *mut c_void, name: &CStr) -> Result<T, String> {
        let ptr = libc::dlsym(handle, name.as_ptr());
        if ptr.is_null() {
            return Err(format!(
                "E_UI_ABI_MISMATCH: missing symbol `{}` in {UI_LIB_FILENAME}",
                name.to_string_lossy()
            ));
        }
        Ok(std::mem::transmute_copy(&ptr))
    }

    let abi_fn: AbiVersionFn = unsafe { sym(handle, c"sofuu_ui_abi_version")? };
    let abi_ver = unsafe { abi_fn() };
    if abi_ver != EXPECTED_UI_ABI_VERSION {
        unsafe { libc::dlclose(handle) };
        return Err(format!(
            "E_UI_ABI_MISMATCH: expected ABI v{EXPECTED_UI_ABI_VERSION}, got v{abi_ver}"
        ));
    }

    Ok(LoadedUiLib {
        handle,
        run_json: unsafe { sym(handle, c"sofuu_ui_run_json")? },
        push_patch_json: unsafe { sym(handle, c"sofuu_ui_push_patch_json")? },
        poll_event_json: unsafe { sym(handle, c"sofuu_ui_poll_event_json")? },
    })
}

#[cfg(not(unix))]
fn load_ui_lib() -> Result<LoadedUiLib, String> {
    Err(format!(
        "E_UI_LIB_MISSING: dynamic UI loading on this OS requires Unix dlopen ({UI_LIB_FILENAME})"
    ))
}

/// Minimal domain-agnostic Light-theme boot scene shown instantaneously before JS mounts.
pub fn default_boot_scene_json(title: &str) -> String {
    let scene = serde_json::json!({
        "title": title,
        "theme": "light",
        "root": {
            "id": "boot_root",
            "padding": { "top": 18.0, "right": 18.0, "bottom": 18.0, "left": 18.0 },
            "kind": {
                "type": "container",
                "axis": "column",
                "align": "stretch",
                "gap": 12.0,
                "fold_width": null,
                "scrollable": false,
                "style": null,
                "children": [
                    {
                        "id": "boot_hdr",
                        "kind": {
                            "type": "container",
                            "axis": "row",
                            "align": "center",
                            "gap": 10.0,
                            "fold_width": null,
                            "scrollable": false,
                            "style": null,
                            "children": [
                                {
                                    "id": "boot_title",
                                    "width": { "fill": 1.0 },
                                    "kind": {
                                        "type": "text",
                                        "content": title,
                                        "size": 17.0,
                                        "color": null,
                                        "mono": false,
                                        "wrap": false
                                    }
                                },
                                {
                                    "id": "theme_toggle",
                                    "kind": { "type": "theme_toggle", "size": 34.0 }
                                }
                            ]
                        }
                    },
                    {
                        "id": "boot_stat",
                        "width": { "fill": 1.0 },
                        "height": { "fill": 1.0 },
                        "kind": {
                            "type": "stat",
                            "label": "Sofuu GPU Runtime",
                            "value": 1.0,
                            "value_text": "Ready",
                            "badge": "LIVE",
                            "caption": "Waiting for sofuu.ui.mount(...) or agent events",
                            "tone": "info",
                            "variant": "compact"
                        }
                    }
                ]
            }
        }
    });
    scene.to_string()
}

struct WorkerCtx {
    run: Option<Box<dyn FnOnce() -> i32 + Send>>,
    exit_code: Arc<AtomicI32>,
}

extern "C" fn ui_worker_trampoline(session: *mut c_void, user_ctx: *mut c_void) {
    ACTIVE_SESSION.store(session, Ordering::Release);
    UI_MOUNTED.store(false, Ordering::Release);
    WINDOW_CLOSED_SEEN.store(false, Ordering::Release);

    let ctx = unsafe { Box::from_raw(user_ctx as *mut WorkerCtx) };
    let rc = if let Some(f) = ctx.run { f() } else { 1 };
    ctx.exit_code.store(rc, Ordering::Release);

    // If the script succeeded and mounted a UI without running its own event wait loop,
    // keep the window interactive cleanly until the user closes it.
    if rc == 0 && UI_MOUNTED.load(Ordering::Acquire) && !WINDOW_CLOSED_SEEN.load(Ordering::Acquire) {
        while !WINDOW_CLOSED_SEEN.load(Ordering::Acquire) {
            match poll_event_raw() {
                Ok(Some(json)) => {
                    if json.contains("\"window_closed\"") {
                        WINDOW_CLOSED_SEEN.store(true, Ordering::Release);
                        break;
                    }
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(40)),
                Err(_) => break,
            }
        }
    }

    ACTIVE_SESSION.store(ptr::null_mut(), Ordering::Release);
}

/// Launch the native GPU UI window on Thread 0 and execute `worker` on Thread 1.
pub fn run_with_ui<F>(title: &str, worker: F) -> ExitCode
where
    F: FnOnce() -> i32 + Send + 'static,
{
    let lib = match load_ui_lib() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return ExitCode::FAILURE;
        }
    };

    ACTIVE_PUSH_FN.store(lib.push_patch_json as *mut c_void, Ordering::Release);
    ACTIVE_POLL_FN.store(lib.poll_event_json as *mut c_void, Ordering::Release);

    let boot_scene = default_boot_scene_json(title);
    let title_c = CString::new(title).unwrap_or_else(|_| CString::new("Sofuu").unwrap());
    let exit_code = Arc::new(AtomicI32::new(0));
    let worker_box = Box::new(WorkerCtx {
        run: Some(Box::new(worker)),
        exit_code: Arc::clone(&exit_code),
    });
    let worker_ptr = Box::into_raw(worker_box) as *mut c_void;

    let ui_rc = unsafe {
        (lib.run_json)(
            title_c.as_ptr(),
            1080.0,
            720.0,
            boot_scene.as_ptr(),
            boot_scene.len(),
            Some(ui_worker_trampoline),
            worker_ptr,
        )
    };

    ACTIVE_SESSION.store(ptr::null_mut(), Ordering::Release);
    ACTIVE_PUSH_FN.store(ptr::null_mut(), Ordering::Release);
    ACTIVE_POLL_FN.store(ptr::null_mut(), Ordering::Release);

    if ui_rc != 0 {
        eprintln!("\x1b[31mError:\x1b[0m E_UI_NO_DISPLAY: UI window exited with status {ui_rc}");
        return ExitCode::FAILURE;
    }

    let code = exit_code.load(Ordering::Acquire);
    if code == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from((code.clamp(1, 255)) as u8)
    }
}

pub fn is_ui_attached() -> bool {
    !ACTIVE_SESSION.load(Ordering::Acquire).is_null()
}

pub fn push_patch_raw(json: &str) -> i32 {
    let sess = ACTIVE_SESSION.load(Ordering::Acquire);
    let fn_ptr = ACTIVE_PUSH_FN.load(Ordering::Acquire);
    if sess.is_null() || fn_ptr.is_null() {
        return -2;
    }
    let push_fn: PushPatchJsonFn = unsafe { std::mem::transmute(fn_ptr) };
    let rc = unsafe { push_fn(sess, json.as_ptr(), json.len()) };
    if rc == 0 {
        UI_MOUNTED.store(true, Ordering::Release);
    }
    rc
}

pub fn poll_event_raw() -> Result<Option<String>, i32> {
    let sess = ACTIVE_SESSION.load(Ordering::Acquire);
    let fn_ptr = ACTIVE_POLL_FN.load(Ordering::Acquire);
    if sess.is_null() || fn_ptr.is_null() {
        return Err(-2);
    }
    let poll_fn: PollEventJsonFn = unsafe { std::mem::transmute(fn_ptr) };
    let mut buf = vec![0u8; 4096];
    let n = unsafe { poll_fn(sess, buf.as_mut_ptr(), buf.len()) };
    if n == 0 {
        return Ok(None);
    }
    if n < 0 {
        let needed = (-n) as usize;
        if needed == 0 || needed > 4 * 1024 * 1024 {
            return Err(-1);
        }
        buf.resize(needed, 0);
        let n2 = unsafe { poll_fn(sess, buf.as_mut_ptr(), buf.len()) };
        if n2 <= 0 {
            return Ok(None);
        }
        let s = String::from_utf8_lossy(&buf[..n2 as usize]).into_owned();
        if s.contains("\"window_closed\"") {
            WINDOW_CLOSED_SEEN.store(true, Ordering::Release);
        }
        return Ok(Some(s));
    }
    let s = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
    if s.contains("\"window_closed\"") {
        WINDOW_CLOSED_SEEN.store(true, Ordering::Release);
    }
    Ok(Some(s))
}

unsafe extern "C" fn js_ui_is_attached(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    js_new_bool(ctx, is_ui_attached())
}

unsafe extern "C" fn js_ui_push_patch(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_int32(ctx, -1);
    }
    let Some(json) = js_to_string(ctx, *argv) else {
        return qjs::sofuu_js_new_int32(ctx, -1);
    };
    let rc = push_patch_raw(&json);
    qjs::sofuu_js_new_int32(ctx, rc)
}

unsafe extern "C" fn js_ui_poll_event(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    match poll_event_raw() {
        Ok(Some(json)) => js_new_string(ctx, &json),
        _ => qjs::sofuu_js_null(),
    }
}

pub unsafe fn mod_ui_register(ctx: *mut JSContext) {
    register_global_fn(ctx, "__sofuu_ui_is_attached", js_ui_is_attached);
    register_global_fn(ctx, "__sofuu_ui_push_patch", js_ui_push_patch);
    register_global_fn(ctx, "__sofuu_ui_poll_event", js_ui_poll_event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_ui_arg_strips_only_ui_flag_before_separator() {
        let args: Vec<String> = ["sofuu", "run", "--ui", "agent.js", "--", "--ui"]
            .into_iter()
            .map(ToString::to_string)
            .collect();
        let (has_ui, clean) = take_ui_arg(&args);
        assert!(has_ui);
        assert_eq!(clean, vec!["sofuu", "run", "agent.js", "--", "--ui"]);

        let plain: Vec<String> = ["sofuu", "run", "agent.js"]
            .into_iter()
            .map(ToString::to_string)
            .collect();
        let (has_ui2, clean2) = take_ui_arg(&plain);
        assert!(!has_ui2);
        assert_eq!(clean2, plain);
    }

    #[test]
    fn default_boot_scene_json_is_valid_light_scene() {
        let raw = default_boot_scene_json("Test Title");
        let val: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(val["title"], "Test Title");
        assert_eq!(val["theme"], "light");
        assert_eq!(val["root"]["kind"]["type"], "container");
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn candidate_paths_respect_env_override_first() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var_os("SOFUU_UI_LIB");
        std::env::set_var("SOFUU_UI_LIB", "/tmp/custom_libsofuu_gpu.dylib");
        let paths = candidate_ui_lib_paths();
        assert_eq!(paths[0], Path::new("/tmp/custom_libsofuu_gpu.dylib"));
        if let Some(p) = prev {
            std::env::set_var("SOFUU_UI_LIB", p);
        } else {
            std::env::remove_var("SOFUU_UI_LIB");
        }
    }

    #[test]
    fn explicit_missing_env_override_returns_diagnostic_code() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var_os("SOFUU_UI_LIB");
        std::env::set_var("SOFUU_UI_LIB", "/tmp/definitely_missing_sofuu_gpu_999999.dylib");
        let err = match load_ui_lib() {
            Ok(_) => panic!("expected E_UI_LIB_MISSING error for missing SOFUU_UI_LIB"),
            Err(e) => e,
        };
        if let Some(p) = prev {
            std::env::set_var("SOFUU_UI_LIB", p);
        } else {
            std::env::remove_var("SOFUU_UI_LIB");
        }
        assert!(err.starts_with("E_UI_LIB_MISSING:"), "unexpected error: {err}");
    }
}
