// engine.rs — the dedicated engine worker thread (PLAN-DESKTOP D).
//
// The sofuu engine evaluates a turn as ONE blocking eval on ONE thread
// (eval_string drains the libuv loop + microtask queue before returning),
// and its registries are thread-local. So the desktop runs exactly one
// worker thread that:
//
//   1. puts the runtime in polite-guest mode (embed_config),
//   2. chdirs to the user-picked project dir (Finder-launched apps boot in
//      / — the project dir is the tool jail root + session-mesh root),
//   3. creates the ONE SofuuRuntime for the app's lifetime (P0-7),
//   4. registers the host natives (__desktop_event/__desktop_reply),
//   5. injects Keychain API keys + evals sofuu.chat.init(…),
//   6. loops over a command channel until Shutdown.
//
// Cancel + approval resolution do NOT go through this channel — they use
// sofuu_core::rt::host_poke (uv_async), which is safe to call from any
// thread while a turn's eval is blocked.

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use sofuu_ffi::SofuuRuntime;

use crate::host;
use crate::keychain;

/// Commands the engine worker accepts. All are Send; the runtime itself
/// never crosses threads.
pub enum EngineCmd {
    /// Submit a turn: one blocking eval of sofuu.chat.submit(…). Events
    /// stream out through __desktop_event while the eval runs; the final
    /// outcome arrives as the terminal "done"/"error" event.
    Turn { text: String, opts: serde_json::Value },
    /// Eval a chat.js expression that answers via __desktop_reply(json).
    /// Used for reads (sessions, state) that must not race a running turn.
    Query {
        /// JS expression whose value is passed to __desktop_reply, e.g.
        /// `__desktop_reply(JSON.stringify(sofuu.chat.sessions()))`.
        script: String,
        reply: mpsc::Sender<Result<String, String>>,
    },
    /// Drop the runtime on its own thread and exit the loop (app quit).
    Shutdown,
}

/// Handle shared with the Tauri commands (cheap clone).
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCmd>,
    ready: Arc<(Mutex<ReadyState>, Condvar)>,
}

#[derive(Default)]
struct ReadyState {
    done: bool,
    error: Option<String>,
}

impl EngineHandle {
    /// Enqueue a turn. Returns Err if the engine thread is gone.
    pub fn send_turn(&self, text: String, opts: serde_json::Value) -> Result<(), String> {
        self.tx
            .send(EngineCmd::Turn { text, opts })
            .map_err(|_| "engine thread is not running".to_string())
    }

    /// Run a query eval and wait for the reply (bounded).
    pub fn query(&self, script: String, timeout: Duration) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(EngineCmd::Query { script, reply: tx })
            .map_err(|_| "engine thread is not running".to_string())?;
        match rx.recv_timeout(timeout) {
            Ok(res) => res,
            Err(_) => Err("engine query timed out".to_string()),
        }
    }

    /// Ask the engine to shut down (idempotent-ish; join afterwards).
    pub fn shutdown(&self) {
        let _ = self.tx.send(EngineCmd::Shutdown);
    }

    /// Block until the engine finished booting; Err carries the boot error.
    pub fn wait_ready(&self, timeout: Duration) -> Result<(), String> {
        let (lock, cvar) = &*self.ready;
        let mut state = lock.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        while !state.done {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err("engine boot timed out".to_string());
            }
            let (s, _wait) = cvar
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            state = s;
        }
        match state.error.clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Spawn the engine worker. Returns the handle immediately; call
/// `wait_ready` before relying on it.
pub fn spawn(project_dir: Option<String>) -> (EngineHandle, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<EngineCmd>();
    let ready = Arc::new((Mutex::new(ReadyState::default()), Condvar::new()));
    let ready2 = Arc::clone(&ready);
    let thread = std::thread::Builder::new()
        .name("sofuu-engine".into())
        .spawn(move || engine_main(rx, ready2, project_dir))
        .expect("spawn engine thread");
    (EngineHandle { tx, ready }, thread)
}

fn signal_ready(ready: &Arc<(Mutex<ReadyState>, Condvar)>, error: Option<String>) {
    let (lock, cvar) = &**ready;
    let mut state = lock.lock().unwrap();
    state.done = true;
    state.error = error;
    cvar.notify_all();
}

/// The worker's main loop. Runs on the engine thread, owns the runtime.
fn engine_main(
    rx: mpsc::Receiver<EngineCmd>,
    ready: Arc<(Mutex<ReadyState>, Condvar)>,
    project_dir: Option<String>,
) {
    // 1. Polite-guest mode BEFORE anything touches process globals:
    //    catchable process.exit, no signal handlers, shared ~/.sofuu config
    //    (config_root = None keeps the CLI's identity).
    sofuu_core::embed_config::configure(true, None, false);
    sofuu_core::embed_config::set_log_callback(Some(host_log), std::ptr::null_mut());

    // 2. Project dir before runtime init so the engine boots inside it.
    if let Some(dir) = &project_dir {
        if let Err(e) = std::env::set_current_dir(dir) {
            eprintln!("[desktop] chdir {dir} failed: {e}");
        }
    }

    // 3. The ONE runtime (P0-7: never a second before this one is dropped).
    let Some(rt) = SofuuRuntime::init() else {
        signal_ready(&ready, Some("sofuu runtime init failed".into()));
        return;
    };

    // 4. Host natives (event sink + reply channel).
    let ctx = rt.engine_ctx() as *mut sofuu_ffi::bridge::JSContext;
    if ctx.is_null() {
        signal_ready(&ready, Some("engine context unavailable".into()));
        return;
    }
    // SAFETY: ctx is the live engine context on this thread.
    unsafe { host::register_host_natives(ctx) };

    // 5a. Keychain keys → embedded provider keys (preferred over env).
    // Inject EVERY configured provider's key, not just the two built-ins:
    // a custom provider's Keychain key was previously live after save but
    // lost on restart (config-file key was the only fallback). Config-file
    // keys stay as fallback for CLI-written entries.
    {
        let cfg = crate::config::get_config_redacted();
        let mut names: Vec<String> = cfg
            .get("providers")
            .and_then(|p| p.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.get("name")?.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        // Always cover the built-ins even with no config yet.
        for builtin in ["openai", "anthropic"] {
            if !names.iter().any(|n| n == builtin) {
                names.push(builtin.to_string());
            }
        }
        for provider in names {
            if let Some(key) = keychain::get_key(&provider) {
                if !key.is_empty() {
                    sofuu_core::embed_config::set_api_key(&provider, key);
                }
            }
        }
    }

    // 5b. Boot the shared turn engine (chat.js ships with the runtime and is
    //     eval'd at engine boot by shipped.rs). Desktop opts in to the
    //     interactive permission gate + a human-scale tool timeout.
    let init = serde_json::json!({
        "host": "desktop",
        "project": project_dir.unwrap_or_default(),
        "permissions": "prompt",
        "toolTimeoutMs": 3_600_000u64, // approvals wait for a human, not 30s
        "pastTurns": 40,               // desktop resume wants deeper history
    });
    let script = format!(
        "if (globalThis.sofuu && sofuu.chat && sofuu.chat.init) {{ \
           sofuu.chat.init({init}); \
         }} else {{ throw new Error('sofuu.chat.init missing (chat.js not shipped yet)'); }}",
    );
    let rc = rt.eval_string(&script, "<desktop-init>");
    if rc != 0 {
        // chat.js lands with workstream C — until then the app boots with a
        // dead chat engine and the UI shows the boot error.
        signal_ready(
            &ready,
            Some("sofuu.chat.init failed (rc != 0) — chat.js turn engine not available yet".into()),
        );
        // Keep the thread alive so queries/turns report clean errors instead
        // of a dead channel; Shutdown still works.
    } else {
        signal_ready(&ready, None);
    }

    // 6. Command loop.
    while let Ok(cmd) = rx.recv() {
        match cmd {
            EngineCmd::Turn { text, opts } => {
                // JSON string literals are valid JS — serde does the escaping.
                let text_js = serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".into());
                let opts_js = serde_json::to_string(&opts)
                    .unwrap_or_else(|_| "{}".into());
                let script = format!(
                    "(function () {{ \
                       var opts = {opts_js}; \
                       opts.onEvent = function (e) {{ __desktop_event(JSON.stringify(e)); }}; \
                       Promise.resolve(sofuu.chat.submit({text_js}, opts)).then( \
                         function () {{ __desktop_event(JSON.stringify({{kind:'turn_finished'}})); }}, \
                         function (err) {{ __desktop_event(JSON.stringify({{kind:'turn_error', payload:{{message: String(err && err.message || err)}}}})); }} \
                       ); \
                     }})();"
                );
                let rc = rt.eval_string(&script, "<turn>");
                if rc != 0 {
                    // Synchronous driver failure (the promise path reports its
                    // own errors through the event above).
                    let _ = send_event_json(
                        r#"{"kind":"turn_error","payload":{"message":"engine eval failed"}}"#,
                    );
                }
            }
            EngineCmd::Query { script, reply } => {
                let rx = host_arm_reply();
                let rc = rt.eval_string(&script, "<query>");
                let outcome = if rc != 0 {
                    Err("query eval failed".to_string())
                } else {
                    match rx.recv_timeout(Duration::from_secs(30)) {
                        Ok(json) => Ok(json),
                        Err(_) => Err("query produced no reply".to_string()),
                    }
                };
                let _ = reply.send(outcome);
            }
            EngineCmd::Shutdown => break,
        }
    }

    // Drop the runtime on its own thread (P0-7 teardown discipline).
    drop(rt);
}

/// Route a JSON event into the WebView without going through JS (used for
/// engine-side synthetic events like eval failures).
fn send_event_json(json: &str) -> Result<(), ()> {
    use tauri::Emitter;
    let payload: serde_json::Value =
        serde_json::from_str(json).unwrap_or(serde_json::Value::Null);
    match crate::host::app_handle() {
        Some(app) => {
            let _ = app.emit(host::EVENT_CHANNEL, payload);
            Ok(())
        }
        None => Err(()),
    }
}

/// Arm the host reply slot from the engine thread (host.rs owns the slot).
fn host_arm_reply() -> mpsc::Receiver<String> {
    crate::host::arm_reply()
}

/// embed_config log callback: console output from the embedded runtime.
/// Routed to stderr for now (a dev console pane can subscribe later).
///
/// # Safety
/// `level`/`msg` are NUL-terminated and valid for the call's duration only.
unsafe extern "C" fn host_log(
    level: *const std::ffi::c_char,
    msg: *const std::ffi::c_char,
    _opaque: *mut std::ffi::c_void,
) {
    let level = if level.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(level).to_string_lossy().into_owned()
    };
    let msg = if msg.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
    };
    eprintln!("[sofuu:{level}] {msg}");
}
