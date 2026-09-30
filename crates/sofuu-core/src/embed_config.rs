// embed_config.rs — process-global config for embedded mode (PLAN-HEADLESS H2).
//
// When Sofuu is used as a library (libsofuu), the host sets these values
// via sofuu_rt_new(config_json) in the capi crate. The core modules
// (process, console, ai) read them to behave as polite guests:
//   - process.exit → catchable ExitError (not libc::exit)
//   - signal handlers gated (not installed by default)
//   - console output → log callback (not raw stdout/stderr)
//   - config_root replaces $HOME/.sofuu derivations

use std::cell::RefCell;
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
/// project-local default chain in agent.js brainFor (<cwd>/.sofuu/brain/
/// brain.qtsq, then ~/.sofuu/brain/brain.qtsq).
static BRAIN_PATH: Mutex<Option<String>> = Mutex::new(None);

// ── Per-runtime settings (F-3, AUDIT-2026-09-01) ──────────────────
//
// The statics above are process-globals. When a host creates MULTIPLE
// runtimes in one process (the ABI promises this is supported), each
// sofuu_rt_new used to clobber them, silently retargeting every earlier
// runtime's config_root, provider keys and brain path.
//
// Fix: the capi installs a runtime's COMPLETE config into a thread-local
// on the runtime's owning thread (ABI contract: one runtime per thread).
// All readers below resolve the thread-local FIRST and exclusively — a
// per-runtime config never falls through to the globals, so runtime B's
// settings can no longer leak into runtime A. Threads with no installed
// runtime (CLI/desktop/TUI never install one) keep reading the globals,
// which preserves their behavior exactly.

/// A runtime's complete config, as parsed by the capi from config_json.
#[derive(Clone, Default)]
struct RtSettings {
    embedded: bool,
    config_root: Option<String>,
    enable_signals: bool,
    api_keys: Vec<(String, String)>,
    brain_path: Option<String>,
    /// E2: headless request defaults. A CLI user picks provider+model
    /// interactively (`/model`), so the runtime deliberately has no global
    /// default there. An *embedder* has no session to pick in, so repeating
    /// provider/model on every single call is pure friction — these let a
    /// host set them once in `sofuu_rt_new` config. They are only consulted
    /// when a call omits them, so they never override an explicit request.
    default_provider: Option<String>,
    default_model: Option<String>,
    default_base_url: Option<String>,
}

thread_local! {
    static CURRENT: RefCell<Option<RtSettings>> = const { RefCell::new(None) };
}

/// Install the calling thread's runtime settings. Called by the capi's
/// sofuu_rt_new, which runs on the runtime's owning thread. The most
/// recent install on a thread wins for that thread (a fresh rt_new on the
/// same thread — e.g. sequential create/destroy — replaces the old one).
#[allow(clippy::too_many_arguments)]
pub fn install_runtime_settings(
    embedded: bool,
    config_root: Option<String>,
    enable_signals: bool,
    api_keys: Vec<(String, String)>,
    brain_path: Option<String>,
    default_provider: Option<String>,
    default_model: Option<String>,
    default_base_url: Option<String>,
) {
    CURRENT.with(|c| {
        *c.borrow_mut() = Some(RtSettings {
            embedded,
            config_root,
            enable_signals,
            api_keys,
            brain_path,
            default_provider,
            default_model,
            default_base_url,
        });
    });
}

/// This thread's installed runtime settings, if any.
fn current_settings() -> Option<RtSettings> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Set the embedded config. Called once from the capi at runtime creation.
pub fn configure(embedded: bool, config_root: Option<String>, enable_signals: bool) {
    EMBEDDED.store(embedded, Ordering::Relaxed);
    ENABLE_SIGNALS.store(enable_signals, Ordering::Relaxed);
    if let Ok(mut cr) = CONFIG_ROOT.lock() {
        *cr = config_root;
    }
}

/// True when the runtime is in embedded mode. Per-runtime when the thread
/// has installed settings, else the process-global.
pub fn is_embedded() -> bool {
    if let Some(s) = current_settings() {
        return s.embedded;
    }
    EMBEDDED.load(Ordering::Relaxed)
}

/// Whether signal handlers should be installed. Per-runtime when the thread
/// has installed settings, else the process-global.
pub fn signals_enabled() -> bool {
    if let Some(s) = current_settings() {
        return s.enable_signals;
    }
    ENABLE_SIGNALS.load(Ordering::Relaxed)
}

/// Get the config root path (replaces $HOME/.sofuu). Per-runtime when the
/// thread has installed settings, else the process-global. Returns None if
/// not configured (caller should fall back to $HOME).
pub fn get_config_root() -> Option<String> {
    if let Some(s) = current_settings() {
        return s.config_root;
    }
    CONFIG_ROOT.lock().ok()?.clone()
}

/// User home directory across hosts: $HOME, then %USERPROFILE% (Windows),
/// else None. All $HOME/.sofuu-style derivations should go through this.
pub fn home_dir() -> Option<String> {
    std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
}

/// Derive a path under the config root. Falls back to $HOME/.sofuu if
/// config_root is not set.
pub fn config_dir(subdir: &str) -> String {
    let base = get_config_root().unwrap_or_else(|| {
        home_dir()
            .map(|h| format!("{h}/.sofuu"))
            .unwrap_or_else(|| ".sofuu".to_string())
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
    // P3 (AUDIT-2026-09-07): an interior NUL used to make CString::new fail
    // and unwrap_or_default hand the host an EMPTY string, silently dropping
    // the log line. Escape NULs instead so the message always arrives.
    let level_c = CString::new(level.replace('\0', "\\0")).unwrap_or_default();
    let msg_c = CString::new(msg.replace('\0', "\\0")).unwrap_or_default();
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

/// Drop a host-provided API key (the desktop's keychain_delete). Without
/// this the in-process engine keeps serving the stale key after a delete.
pub fn clear_api_key(name: &str) {
    if let Ok(mut keys) = API_KEYS.lock() {
        keys.retain(|(n, _)| n != name);
    }
}

/// Look up a host-provided API key by provider name. Per-runtime when the
/// thread has installed settings (the installed key set is the runtime's
/// COMPLETE set — an absent name means that runtime has no key for the
/// provider, never a fall-through to another runtime's key), else the
/// process-global.
pub fn api_key(name: &str) -> Option<String> {
    if let Some(s) = current_settings() {
        return s.api_keys.iter().find(|(n, _)| n == name).map(|(_, k)| k.clone());
    }
    let keys = API_KEYS.lock().ok()?;
    keys.iter().find(|(n, _)| n == name).map(|(_, k)| k.clone())
}

/// Set (or clear, with None) the host-provided default brain file path.
pub fn set_brain_path(path: Option<String>) {
    if let Ok(mut slot) = BRAIN_PATH.lock() {
        *slot = path;
    }
}

/// Host-provided default brain file path, if configured. Per-runtime when
/// the thread has installed settings, else the process-global.
pub fn brain_path() -> Option<String> {
    if let Some(s) = current_settings() {
        return s.brain_path;
    }
    BRAIN_PATH.lock().ok()?.clone()
}

// ── E2: headless request defaults (provider / model / base_url) ──
//
// These exist ONLY to save an embedder from repeating them per call. They
// are read exclusively from the runtime's installed settings (never the
// process globals), so runtime B's defaults can never retarget runtime A —
// the same isolation the api_keys reader has.

/// The runtime's default provider name, if the host configured one.
pub fn default_provider() -> Option<String> {
    current_settings()?.default_provider
}

/// The runtime's default model, if the host configured one.
pub fn default_model() -> Option<String> {
    current_settings()?.default_model
}

/// The runtime's default endpoint override, if the host configured one.
/// This is what lets an embedder point the whole runtime at a local
/// OpenAI-compatible server (Ollama, vLLM, LM Studio) in one place.
pub fn default_base_url() -> Option<String> {
    current_settings()?.default_base_url
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop this thread's installed settings so a test never leaks them
    /// into another test sharing the thread.
    fn clear_current() {
        CURRENT.with(|c| *c.borrow_mut() = None);
    }

    // F-3 (AUDIT-2026-09-01): runtime B's config must not retarget runtime
    // A. The capi installs each runtime's settings on its owning thread;
    // readers must resolve them exclusively — no fall-through to the
    // globals another runtime clobbered, and no leak across threads.
    #[test]
    fn runtime_settings_are_isolated_per_thread() {
        install_runtime_settings(
            true,
            Some("/f3-iso-a".into()),
            false,
            vec![("openai".into(), "key-a".into())],
            Some("/f3-brain-a.qtsq".into()),
            None,
            None,
            None,
        );

        // A second thread installs runtime B's settings while A is alive.
        let t = std::thread::spawn(move || {
            install_runtime_settings(
                false,
                Some("/f3-iso-b".into()),
                true,
                vec![("openai".into(), "key-b".into())],
                None,
            None,
            None,
            None,
            );
            (
                get_config_root(),
                api_key("openai"),
                is_embedded(),
                signals_enabled(),
                brain_path(),
            )
        });
        let seen_b = t.join().unwrap();

        // This thread (runtime A) must still see A — B's install changed
        // nothing here, and A's missing pieces must NOT fall through to
        // B's values.
        assert_eq!(get_config_root().as_deref(), Some("/f3-iso-a"), "B's config_root leaked into A's thread");
        assert_eq!(api_key("openai").as_deref(), Some("key-a"), "B's api key leaked into A's thread");
        assert!(is_embedded(), "B's embedded flag leaked into A's thread");
        assert!(!signals_enabled(), "B's signals flag leaked into A's thread");
        assert_eq!(brain_path().as_deref(), Some("/f3-brain-a.qtsq"), "B's brain path leaked into A's thread");

        // B's thread saw B exclusively.
        assert_eq!(seen_b.0.as_deref(), Some("/f3-iso-b"));
        assert_eq!(seen_b.1.as_deref(), Some("key-b"));
        assert!(!seen_b.2);
        assert!(seen_b.3);
        assert_eq!(seen_b.4, None, "B's unset brain path must not fall through to the globals");

        clear_current();
    }

    // The per-runtime key set is the runtime's COMPLETE set: an unlisted
    // provider resolves to None even when the globals hold a key.
    #[test]
    fn installed_settings_never_fall_through_to_globals() {
        set_api_key("f3-probe", "global-key".into());
        set_brain_path(Some("/f3-global-brain.qtsq".into()));

        install_runtime_settings(true, None, false, Vec::new(), None, None, None, None);
        assert_eq!(api_key("f3-probe"), None, "installed runtime must not inherit the global key");
        assert_eq!(brain_path(), None, "installed runtime must not inherit the global brain path");
        assert_eq!(get_config_root(), None);

        // Same thread, no install → globals visible again.
        clear_current();
        assert_eq!(api_key("f3-probe").as_deref(), Some("global-key"));
        assert_eq!(brain_path().as_deref(), Some("/f3-global-brain.qtsq"));

        clear_api_key("f3-probe");
        set_brain_path(None);
    }
}
