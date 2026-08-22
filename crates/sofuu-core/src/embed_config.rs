// embed_config.rs — process-global config for embedded mode (PLAN-HEADLESS H2).
//
// When Sofuu is used as a library (libsofuu), the host sets these values
// via sofuu_rt_new(config_json) in the capi crate. The core modules
// (process, console, ai) read them to behave as polite guests:
//   - process.exit → catchable ExitError (not libc::exit)
//   - signal handlers gated (not installed by default)
//   - console output → log callback (not raw stdout/stderr)
//   - config_root replaces $HOME/.sofuu derivations

use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Mutex;

/// True when the runtime is in embedded/hosted-library mode.
static EMBEDDED: AtomicBool = AtomicBool::new(false);

/// Replaces $HOME/.sofuu derivations. None = use env var fallback.
static CONFIG_ROOT: Mutex<Option<String>> = Mutex::new(None);

/// Whether signal handlers should be installed (default off when embedded).
static ENABLE_SIGNALS: AtomicBool = AtomicBool::new(false);

/// Host log callback (console output routing). None = write to stdio.
/// Called with NUL-terminated `level` ("log"/"info"/"warn"/"error"/...)
/// and `message` strings; both are only valid for the call's duration.
pub type LogCallback = unsafe extern "C" fn(level: *const c_char, msg: *const c_char, opaque: *mut c_void);

static LOG_CB: Mutex<Option<LogCallback>> = Mutex::new(None);
static LOG_OPAQUE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// Host-provided API keys by provider name ("openai"/"anthropic").
/// Consulted after an explicit per-call api_key, before env vars.
static API_KEYS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Host-provided default brain file path (agent memory). None = the
/// $HOME/.sofuu_brain.qtsq fallback in agent.js.
static BRAIN_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Set the embedded config. Called once from the capi at runtime creation.
pub fn configure(embedded: bool, config_root: Option<String>, enable_signals: bool) {
    EMBEDDED.store(embedded, Ordering::Relaxed);
    ENABLE_SIGNALS.store(enable_signals, Ordering::Relaxed);
    if let Ok(mut cr) = CONFIG_ROOT.lock() {
        *cr = config_root;
    }
}

/// True when the runtime is in embedded mode.
pub fn is_embedded() -> bool {
    EMBEDDED.load(Ordering::Relaxed)
}

/// Whether signal handlers should be installed.
pub fn signals_enabled() -> bool {
    ENABLE_SIGNALS.load(Ordering::Relaxed)
}

/// Get the config root path (replaces $HOME/.sofuu). Returns None if not
/// configured (caller should fall back to $HOME).
pub fn get_config_root() -> Option<String> {
    CONFIG_ROOT.lock().ok()?.clone()
}

/// Derive a path under the config root. Falls back to $HOME/.sofuu if
/// config_root is not set.
pub fn config_dir(subdir: &str) -> String {
    let base = get_config_root().unwrap_or_else(|| {
        std::env::var("HOME")
            .map(|h| format!("{}/.sofuu", h))
            .unwrap_or_else(|_| ".sofuu".to_string())
    });
    if subdir.is_empty() {
        base
    } else {
        format!("{}/{}", base, subdir)
    }
}

/// Install (or clear, with None) the host log callback. Process-global:
/// applies to every runtime in this process.
pub fn set_log_callback(cb: Option<LogCallback>, opaque: *mut c_void) {
    LOG_OPAQUE.store(opaque, Ordering::Relaxed);
    if let Ok(mut slot) = LOG_CB.lock() {
        *slot = cb;
    }
}

/// Route a console line to the host log callback. Returns true when a
/// callback consumed the message (caller must NOT also write to stdio).
pub fn log(level: &str, msg: &str) -> bool {
    let cb = match LOG_CB.lock() {
        Ok(slot) => *slot,
        Err(_) => None,
    };
    let Some(cb) = cb else { return false };
    let level_c = CString::new(level).unwrap_or_default();
    let msg_c = CString::new(msg).unwrap_or_default();
    unsafe { cb(level_c.as_ptr(), msg_c.as_ptr(), LOG_OPAQUE.load(Ordering::Relaxed)) };
    true
}

/// Register a host-provided API key for a provider name ("openai"/
/// "anthropic"). Replaces any earlier key for the same name.
pub fn set_api_key(name: &str, key: String) {
    if let Ok(mut keys) = API_KEYS.lock() {
        keys.retain(|(n, _)| n != name);
        keys.push((name.to_string(), key));
    }
}

/// Look up a host-provided API key by provider name.
pub fn api_key(name: &str) -> Option<String> {
    let keys = API_KEYS.lock().ok()?;
    keys.iter().find(|(n, _)| n == name).map(|(_, k)| k.clone())
}

/// Set (or clear, with None) the host-provided default brain file path.
pub fn set_brain_path(path: Option<String>) {
    if let Ok(mut slot) = BRAIN_PATH.lock() {
        *slot = path;
    }
}

/// Host-provided default brain file path, if configured.
pub fn brain_path() -> Option<String> {
    BRAIN_PATH.lock().ok()?.clone()
}
