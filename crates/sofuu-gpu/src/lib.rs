//! Direct OS windowing + native GPU backends (`sofuu-gpu`).
//!
//! 2-Thread Runtime Model:
//! - Thread 0 (main thread): Owns the OS window (`NSWindow`/`CAMetalLayer` on macOS) and runs the
//!   3 direct GPU shaders (`sdf_quad`, `aa_stroke`, `sdf_text`) at up to 120Hz when animating and
//!   0% CPU/GPU when idle.
//! - Thread 1+ (Sofuu agent / JS runtime): Sends `ScenePatch` updates via [`UiHandle::push_patch`]
//!   (which wakes Thread 0 immediately) and receives semantic [`UIEvent`] interactions.

use std::sync::mpsc;

pub use sofuu_ui::{Scene, ScenePatch, UIEvent};

pub mod ffi;

#[cfg(target_os = "macos")]
mod macos;

#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: f32,
    pub height: f32,
    pub min_width: f32,
    pub min_height: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Sofuu".to_string(),
            width: 1024.0,
            height: 680.0,
            min_width: 360.0,
            min_height: 280.0,
        }
    }
}

/// Thread-safe handle passed to Thread 1 (agent / runtime loop) to drive the native GPU UI.
pub struct UiHandle {
    patch_tx: mpsc::Sender<ScenePatch>,
    event_rx: mpsc::Receiver<UIEvent>,
    wake_fn: fn(),
}

impl UiHandle {
    /// Enqueue a `ScenePatch` and immediately wake Thread 0 from idle sleep.
    pub fn push_patch(&self, patch: ScenePatch) -> bool {
        let ok = self.patch_tx.send(patch).is_ok();
        if ok {
            (self.wake_fn)();
        }
        ok
    }

    /// Split into a cloneable patch sender + event receiver so multiple agent tasks can push UI updates.
    pub fn split(self) -> (UiPatchSender, mpsc::Receiver<UIEvent>) {
        (
            UiPatchSender {
                patch_tx: self.patch_tx,
                wake_fn: self.wake_fn,
            },
            self.event_rx,
        )
    }

    pub fn try_recv_event(&self) -> Option<UIEvent> {
        self.event_rx.try_recv().ok()
    }

    pub fn recv_event(&self) -> Option<UIEvent> {
        self.event_rx.recv().ok()
    }
}

#[derive(Clone)]
pub struct UiPatchSender {
    patch_tx: mpsc::Sender<ScenePatch>,
    wake_fn: fn(),
}

impl UiPatchSender {
    pub fn push_patch(&self, patch: ScenePatch) -> bool {
        let ok = self.patch_tx.send(patch).is_ok();
        if ok {
            (self.wake_fn)();
        }
        ok
    }
}

/// Open the native GPU window on Thread 0 and hand [`UiHandle`] to `on_ready` (e.g. spawn Thread 1).
pub fn run_window<F>(config: WindowConfig, initial_scene: Scene, on_ready: F) -> Result<(), String>
where
    F: FnOnce(UiHandle),
{
    #[cfg(target_os = "macos")]
    {
        return macos::run_macos_window(config, initial_scene, on_ready);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (config, initial_scene, on_ready);
        Err("sofuu-gpu native windowing currently implements macOS Metal; Linux Vulkan/Wayland and Windows OpenGL/Win32 backends are queued next".to_string())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn msl_shader_source_is_nonempty_and_declares_all_three_kernels() {
        let src = include_str!("shaders/msl/ui_shaders.metal");
        assert!(src.contains("quad_vert") && src.contains("quad_frag"));
        assert!(src.contains("stroke_vert") && src.contains("stroke_frag"));
        assert!(src.contains("glyph_vert") && src.contains("glyph_frag"));
    }
}
