//! Stable C ABI exported by the standalone shared library (`libsofuu_gpu.{dylib,so,dll}`).
//!
//! Keeps `sofuu` / compiled agent binaries 100% separate from the UI stack: when `--ui` is
//! passed, the CLI/agent dynamically loads this shared library via `dlopen()` / `LoadLibrary`
//! and communicates across these JSON/C-ABI entry points, while external native apps can link
//! or load the exact same library directly.

use std::ffi::{c_void, CStr};
use std::ptr;
use std::slice;
use std::sync::mpsc;
use std::sync::Mutex;

use sofuu_ui::{AutoAgentEvent, AutoComposer, AutoTriggerStat, Scene, ScenePatch, TableRow, ThemeMode, UIEvent};

use crate::{run_window, UiPatchSender, WindowConfig};

/// Incremented only on breaking C ABI changes.
pub const SOFUU_UI_ABI_VERSION: u32 = 2;

/// Auto-Composer state owned by the native session (ABI v2).
///
/// The JS driver forwards generic telemetry here instead of rebuilding the
/// dashboard in JavaScript; `AutoComposer` builds the scene once and then
/// emits minimal `ScenePatch` deltas.
#[derive(Default)]
struct AutoUiState {
    composer: Option<AutoComposer>,
    mounted: bool,
}

/// Opaque handle handed to Thread 1 (`worker_cb`) while Thread 0 runs the native GPU window.
pub struct SofuuUiSession {
    sender: UiPatchSender,
    events: mpsc::Receiver<UIEvent>,
    auto: Mutex<AutoUiState>,
}

pub type SofuuUiWorkerFn = extern "C" fn(session: *mut SofuuUiSession, user_ctx: *mut c_void);

#[no_mangle]
pub extern "C" fn sofuu_ui_abi_version() -> u32 {
    SOFUU_UI_ABI_VERSION
}

/// Open the native GPU window on Thread 0 (caller's main thread), run `worker_cb` on a
/// spawned worker thread (`Thread 1`), and block Thread 0 until the window closes.
///
/// Returns `0` on clean exit, negative error codes on invalid arguments or GPU/display failure.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_run_json(
    title_utf8: *const libc::c_char,
    width: f32,
    height: f32,
    initial_scene_json: *const u8,
    initial_scene_len: usize,
    worker_cb: Option<SofuuUiWorkerFn>,
    user_ctx: *mut c_void,
) -> i32 {
    if initial_scene_json.is_null() || initial_scene_len == 0 {
        return -1;
    }
    let scene_bytes = slice::from_raw_parts(initial_scene_json, initial_scene_len);
    let scene: Scene = match serde_json::from_slice(scene_bytes) {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let title = if title_utf8.is_null() {
        scene.title.clone()
    } else {
        CStr::from_ptr(title_utf8).to_string_lossy().into_owned()
    };

    let cfg = WindowConfig {
        title,
        width: if width > 100.0 { width } else { 1024.0 },
        height: if height > 100.0 { height } else { 680.0 },
        ..WindowConfig::default()
    };

    let user_ctx_addr = user_ctx as usize;
    let mut worker_join: Option<std::thread::JoinHandle<()>> = None;
    let res = run_window(cfg, scene, |handle| {
        if let Some(cb) = worker_cb {
            let (sender, events) = handle.split();
            worker_join = Some(std::thread::spawn(move || {
                let mut session = SofuuUiSession {
                    sender,
                    events,
                    auto: Mutex::new(AutoUiState::default()),
                };
                cb(&mut session, user_ctx_addr as *mut c_void);
            }));
        }
    });

    if let Some(jh) = worker_join {
        let _ = jh.join();
    }

    match res {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Enqueue a JSON [`ScenePatch`] from Thread 1 and wake Thread 0 immediately.
/// Returns `0` on success, `-1` on null/invalid JSON, `-2` if window already closed.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_push_patch_json(
    session: *mut SofuuUiSession,
    patch_json: *const u8,
    patch_len: usize,
) -> i32 {
    if session.is_null() || patch_json.is_null() || patch_len == 0 {
        return -1;
    }
    let sess = &*session;
    let bytes = slice::from_raw_parts(patch_json, patch_len);
    let patch: ScenePatch = match serde_json::from_slice(bytes) {
        Ok(p) => p,
        Err(_) => return -1,
    };
    if sess.sender.push_patch(patch) {
        0
    } else {
        -2
    }
}

/// Non-blocking poll for the next semantic [`UIEvent`] serialized as UTF-8 JSON.
///
/// Returns bytes written (`> 0`) if an event was copied into `out_buf`, `0` if queue empty,
/// or negative needed length `-needed` if `out_cap` was too small.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_poll_event_json(
    session: *mut SofuuUiSession,
    out_buf: *mut u8,
    out_cap: usize,
) -> i64 {
    if session.is_null() {
        return -1;
    }
    let sess = &*session;
    let Ok(ev) = sess.events.try_recv() else {
        return 0;
    };
    let Ok(json) = serde_json::to_vec(&ev) else {
        return -1;
    };
    if out_buf.is_null() || json.len() > out_cap {
        return -(json.len() as i64);
    }
    ptr::copy_nonoverlapping(json.as_ptr(), out_buf, json.len());
    json.len() as i64
}

// ── ABI v2: session-backed Auto-Composer ────────────────────────────────
//
// Additive symbols. A v1 caller is unaffected; a v2 driver can forward the
// generic runtime telemetry it already observes and let Rust own the
// dashboard scene + incremental patch stream.

fn lock_auto(session: &SofuuUiSession) -> std::sync::MutexGuard<'_, AutoUiState> {
    session.auto.lock().unwrap_or_else(|e| e.into_inner())
}

fn mount_auto(auto: &mut AutoUiState, title: &str, theme: ThemeMode) -> Vec<ScenePatch> {
    let composer = auto
        .composer
        .get_or_insert_with(|| AutoComposer::new(title, theme));
    if auto.mounted {
        return Vec::new();
    }
    auto.mounted = true;
    vec![ScenePatch::ReplaceScene(composer.build_scene())]
}

fn emit_patches(session: &SofuuUiSession, patches: Vec<ScenePatch>) -> i32 {
    for patch in patches {
        if !session.sender.push_patch(patch) {
            return -2;
        }
    }
    0
}

/// Ingest one generic agent/runtime telemetry event. Returns `0` on success,
/// `-1` on bad JSON, `-2` if the window already closed.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_ingest_agent_event_json(
    session: *mut SofuuUiSession,
    event_json: *const u8,
    event_len: usize,
) -> i32 {
    if session.is_null() || event_json.is_null() || event_len == 0 {
        return -1;
    }
    let bytes = slice::from_raw_parts(event_json, event_len);
    let Ok(raw) = std::str::from_utf8(bytes) else {
        return -1;
    };
    let Some(ev) = AutoAgentEvent::from_runtime_json(raw) else {
        return -1;
    };
    let sess = &*session;
    let mut auto = lock_auto(sess);
    let title = if ev.agent_name.is_empty() {
        "Sofuu Agent Runtime".to_string()
    } else {
        format!("Sofuu // {}", ev.agent_name)
    };
    let mut patches = mount_auto(&mut auto, &title, ThemeMode::Light);
    if let Some(composer) = auto.composer.as_mut() {
        patches.extend(composer.ingest_event(ev));
    }
    emit_patches(sess, patches)
}

/// Update the reactive-trigger telemetry table + gate KPI. Accepts
/// `sofuu.triggers.stats()` JSON verbatim (camelCase aliases included).
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_update_triggers_json(
    session: *mut SofuuUiSession,
    stats_json: *const u8,
    stats_len: usize,
) -> i32 {
    if session.is_null() || stats_json.is_null() || stats_len == 0 {
        return -1;
    }
    let bytes = slice::from_raw_parts(stats_json, stats_len);
    let Ok(stats) = serde_json::from_slice::<Vec<AutoTriggerStat>>(bytes) else {
        return -1;
    };
    let sess = &*session;
    let mut auto = lock_auto(sess);
    let mut patches = mount_auto(&mut auto, "Sofuu // Reactive Runtime", ThemeMode::Light);
    if let Some(composer) = auto.composer.as_mut() {
        patches.extend(composer.update_triggers(&stats));
    }
    emit_patches(sess, patches)
}

/// Replace generic operational state rows in the auto table.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_auto_state_rows_json(
    session: *mut SofuuUiSession,
    rows_json: *const u8,
    rows_len: usize,
) -> i32 {
    if session.is_null() || rows_json.is_null() || rows_len == 0 {
        return -1;
    }
    let bytes = slice::from_raw_parts(rows_json, rows_len);
    let Ok(rows) = serde_json::from_slice::<Vec<TableRow>>(bytes) else {
        return -1;
    };
    let sess = &*session;
    let mut auto = lock_auto(sess);
    let mut patches = mount_auto(&mut auto, "Sofuu // Reactive Runtime", ThemeMode::Light);
    if let Some(composer) = auto.composer.as_mut() {
        patches.extend(composer.update_state_rows(rows));
    }
    emit_patches(sess, patches)
}

/// Mount (or re-mount) the Auto-Composer scene with an optional
/// `{"title": "...", "theme": "dark"}` config.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_auto_mount_json(
    session: *mut SofuuUiSession,
    config_json: *const u8,
    config_len: usize,
) -> i32 {
    if session.is_null() {
        return -1;
    }
    let cfg: serde_json::Value = if config_json.is_null() || config_len == 0 {
        serde_json::Value::Null
    } else {
        let bytes = slice::from_raw_parts(config_json, config_len);
        serde_json::from_slice(bytes).unwrap_or(serde_json::Value::Null)
    };
    let title = cfg
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Sofuu Agent Runtime")
        .to_string();
    let theme = if cfg.get("theme").and_then(|v| v.as_str()) == Some("dark") {
        ThemeMode::Dark
    } else {
        ThemeMode::Light
    };
    let sess = &*session;
    let mut auto = lock_auto(sess);
    auto.mounted = true;
    let scene = if let Some(composer) = auto.composer.as_mut() {
        composer.title = title;
        composer.theme = theme;
        composer.build_scene()
    } else {
        let composer = AutoComposer::new(title, theme);
        let scene = composer.build_scene();
        auto.composer = Some(composer);
        scene
    };
    emit_patches(sess, vec![ScenePatch::ReplaceScene(scene)])
}

/// Serialize the current Auto-Composer scene (buffer contract matches
/// [`sofuu_ui_poll_event_json`]).
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_auto_scene_json(
    session: *mut SofuuUiSession,
    out_buf: *mut u8,
    out_cap: usize,
) -> i64 {
    if session.is_null() {
        return -1;
    }
    let sess = &*session;
    let mut auto = lock_auto(sess);
    let composer = auto
        .composer
        .get_or_insert_with(|| AutoComposer::new("Sofuu Agent Runtime", ThemeMode::Light));
    let Ok(json) = serde_json::to_vec(&composer.build_scene()) else {
        return -1;
    };
    if out_buf.is_null() || json.len() > out_cap {
        return -(json.len() as i64);
    }
    ptr::copy_nonoverlapping(json.as_ptr(), out_buf, json.len());
    json.len() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::{Axis, Align, Node, NodeKind, ThemeMode};

    #[test]
    fn abi_version_and_json_patch_schema_roundtrip() {
        assert_eq!(sofuu_ui_abi_version(), 2);
        let scene = Scene::new(
            "Test",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![],
                },
            ),
        );
        let json = serde_json::to_vec(&scene).unwrap();
        let parsed: Scene = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed, scene);

        let patch = ScenePatch::SetText {
            id: "a".into(),
            content: "ok".into(),
        };
        let pjson = serde_json::to_vec(&patch).unwrap();
        let pparsed: ScenePatch = serde_json::from_slice(&pjson).unwrap();
        assert_eq!(pparsed, patch);
    }

    #[test]
    fn runtime_event_and_trigger_normalization_match_js_streams() {
        let ev = AutoAgentEvent::from_runtime_json(
            r#"{"runId":"r1","agentName":"researcher","depth":1,"kind":"tool",
                "payload":{"name":"web.search","args":"{\"q\":\"x\"}","duration":42}}"#,
        )
        .expect("camelCase stream event");
        assert_eq!(ev.run_id, "r1");
        assert_eq!(ev.agent_name, "researcher");
        assert_eq!(ev.tool.as_deref(), Some("web.search"));
        assert_eq!(ev.elapsed_ms, Some(42));

        let approval = AutoAgentEvent::from_runtime_json(
            r#"{"kind":"approval_request",
                "payload":{"id":"a1","tool":"bash","args":"rm -rf","mode":"async"}}"#,
        )
        .expect("approval event");
        assert_eq!(approval.approval.unwrap().id, "a1");

        let stats: Vec<AutoTriggerStat> = serde_json::from_str(
            r#"[{"id":"t1","running":true,"ticks":4,"gateHits":2,"gateSkips":2,
                "aiCalls":1,"lastDurationMs":9}]"#,
        )
        .expect("triggers.stats() camelCase");
        assert_eq!(stats[0].gate_hits, 2);
        assert_eq!(stats[0].last_duration_ms, 9);
    }
}
