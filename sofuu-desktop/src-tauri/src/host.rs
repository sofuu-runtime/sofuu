// host.rs — the JS→desktop event sink (PLAN-DESKTOP D).
//
// The engine worker registers two bridge natives into the QuickJS global:
//
//   __desktop_event(json) — chat.js's onEvent stream (one JSON envelope per
//                           agent step / approval / lifecycle event). Each
//                           call is re-emitted to the WebView as the Tauri
//                           event "sofuu://event".
//   __desktop_reply(json) — one-shot reply channel for query evals
//                           (list_sessions, session_turns, …): the engine
//                           parks a sender in REPLY before the eval and the
//                           native hands the answer to it.
//
// JSCFunctions are 'static and capture nothing, so the AppHandle lives in a
// process-wide OnceLock set once during tauri::setup.

use std::ffi::c_int;
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};

use sofuu_ffi::bridge::{js_to_string, register_global_fn, JSCFunction, JSContext, JSValue, JSValueConst};
use tauri::{AppHandle, Emitter, Manager};

/// The one event channel the frontend listens on.
pub const EVENT_CHANNEL: &str = "sofuu://event";

static APP: OnceLock<AppHandle> = OnceLock::new();
/// Reply slot for the currently-running query eval (engine thread sets it,
/// the native drains it; queries are strictly sequential).
static REPLY: Mutex<Option<mpsc::Sender<String>>> = Mutex::new(None);

/// Park the AppHandle for the 'static bridge natives. Called once from
/// tauri::setup before the engine thread boots.
pub fn set_app_handle(handle: AppHandle) {
    let _ = APP.set(handle);
}

pub fn app_handle() -> Option<&'static AppHandle> {
    APP.get()
}

/// Arm the reply slot; returns the receiver the caller blocks on.
pub fn arm_reply() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    if let Ok(mut slot) = REPLY.lock() {
        *slot = Some(tx);
    }
    rx
}

/// `__desktop_event(json)` — one streaming event from chat.js.
unsafe extern "C" fn js_desktop_event(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return sofuu_ffi::qjs::sofuu_js_undefined();
    }
    // SAFETY: ctx is the live engine context; argv[0] belongs to it.
    let json = unsafe { js_to_string(ctx, *argv) };
    let Some(json) = json else {
        return sofuu_ffi::qjs::sofuu_js_undefined();
    };
    if let Some(app) = app_handle() {
        // Parse so the WebView receives a real object, not a string blob.
        let payload: serde_json::Value =
            serde_json::from_str(&json).unwrap_or(serde_json::Value::String(json.clone()));
        let _ = app.emit(EVENT_CHANNEL, payload.clone());
        macos_side_effects(app, &payload);
    }
    sofuu_ffi::qjs::sofuu_js_undefined()
}

/// `__desktop_reply(json)` — answer for the parked query eval.
unsafe extern "C" fn js_desktop_reply(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return sofuu_ffi::qjs::sofuu_js_undefined();
    }
    // SAFETY: same contract as js_desktop_event.
    let json = unsafe { js_to_string(ctx, *argv) };
    let Some(json) = json else {
        return sofuu_ffi::qjs::sofuu_js_undefined();
    };
    if let Ok(mut slot) = REPLY.lock() {
        if let Some(tx) = slot.take() {
            let _ = tx.send(json);
        }
    }
    sofuu_ffi::qjs::sofuu_js_undefined()
}

/// Register both natives into the engine context.
///
/// # Safety
/// `ctx` must be the live engine context on the engine thread.
pub unsafe fn register_host_natives(ctx: *mut JSContext) {
    // SAFETY: ctx is live; both fns are 'static.
    unsafe {
        register_global_fn(ctx, "__desktop_event", js_desktop_event as JSCFunction);
        register_global_fn(ctx, "__desktop_reply", js_desktop_reply as JSCFunction);
    }
}

/// macOS-native side effects of selected events (PLAN-DESKTOP §4):
/// notifications + dock attention for approvals and finished turns.
fn macos_side_effects(app: &AppHandle, payload: &serde_json::Value) {
    let kind = payload.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    match kind {
        "approval_request" => {
            let tool = payload
                .get("payload")
                .and_then(|p| p.get("tool"))
                .and_then(|t| t.as_str())
                .unwrap_or("a tool");
            notify(app, "Sofuu needs your approval", &format!("{tool} is waiting for approval"));
            bounce_dock(app);
        }
        "done" => {
            // Only pull the user back when they left.
            let focused = app
                .get_webview_window("main")
                .and_then(|w| w.is_focused().ok())
                .unwrap_or(true);
            if !focused {
                notify(app, "Sofuu finished", "The turn completed.");
            }
        }
        _ => {}
    }
}

fn notify(app: &AppHandle, title: &str, body: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app
        .notification()
        .builder()
        .title(title)
        .body(body)
        .show();
}

fn bounce_dock(app: &AppHandle) {
    let _ = app;
    // Tauri core has no dock API — message NSApplication directly. Both
    // requestUserAttention: and dockTile setBadgeLabel: are documented
    // thread-safe, which matters: this runs on the engine thread.
    #[cfg(target_os = "macos")]
    unsafe {
        let cls = objc2::ffi::objc_getClass(c"NSApplication".as_ptr());
        if cls.is_null() {
            return;
        }
        let nsapp: *mut objc2::runtime::AnyObject = objc2::msg_send![cls, sharedApplication];
        if nsapp.is_null() {
            return;
        }
        // NSRequestUserAttentionType: NSCriticalRequest = 0 (bounces until
        // the app is activated).
        let _: i64 = objc2::msg_send![nsapp, requestUserAttention: 0i64];
    }
}

/// Set (or clear, with None) the dock badge — e.g. pending approval count.
#[allow(dead_code)]
pub fn set_dock_badge(text: Option<&str>) {
    #[cfg(target_os = "macos")]
    unsafe {
        let cls = objc2::ffi::objc_getClass(c"NSApplication".as_ptr());
        if cls.is_null() {
            return;
        }
        let nsapp: *mut objc2::runtime::AnyObject = objc2::msg_send![cls, sharedApplication];
        if nsapp.is_null() {
            return;
        }
        let tile: *mut objc2::runtime::AnyObject = objc2::msg_send![nsapp, dockTile];
        if tile.is_null() {
            return;
        }
        let label: *mut objc2::runtime::AnyObject = match text {
            Some(t) => {
                let Ok(ct) = std::ffi::CString::new(t) else { return };
                let str_cls = objc2::ffi::objc_getClass(c"NSString".as_ptr());
                if str_cls.is_null() {
                    return;
                }
                objc2::msg_send![str_cls, stringWithUTF8String: ct.as_ptr()]
            }
            None => std::ptr::null_mut(),
        };
        let _: () = objc2::msg_send![tile, setBadgeLabel: label];
    }
    let _ = text;
}
