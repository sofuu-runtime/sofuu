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

use sofuu_ui::{Scene, ScenePatch, UIEvent};

use crate::{run_window, UiPatchSender, WindowConfig};

/// Incremented only on breaking C ABI changes.
pub const SOFUU_UI_ABI_VERSION: u32 = 1;

/// Opaque handle handed to Thread 1 (`worker_cb`) while Thread 0 runs the native GPU window.
pub struct SofuuUiSession {
    sender: UiPatchSender,
    events: mpsc::Receiver<UIEvent>,
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
    let res = run_window(cfg, scene, move |handle| {
        if let Some(cb) = worker_cb {
            let (sender, events) = handle.split();
            std::thread::spawn(move || {
                let mut session = SofuuUiSession { sender, events };
                cb(&mut session, user_ctx_addr as *mut c_void);
            });
        }
    });

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

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::{Axis, Align, Node, NodeKind, ThemeMode};

    #[test]
    fn abi_version_and_json_patch_schema_roundtrip() {
        assert_eq!(sofuu_ui_abi_version(), 1);
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
}
