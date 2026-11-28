//! Direct OS windowing + native GPU backends (`sofuu-gpu`).
//!
//! 2-Thread Runtime Model:
//! - Thread 0 (main thread): Owns the OS window (AppKit/Metal on macOS, Wayland/X11+EGL on Linux,
//!   Win32/WGL on Windows, `ANativeWindow`+EGL on Android; opt-in Vulkan on Linux/Windows/Android)
//!   and runs the four direct GPU pipelines (`sdf_quad`, `ribbon`, `aa_stroke`, `sdf_text`) at up
//!   to 120 Hz when animating and 0% CPU/GPU when idle.
//! - Thread 1+ (Sofuu agent / JS runtime): Sends `ScenePatch` updates via [`UiHandle::push_patch`]
//!   (which wakes Thread 0 immediately) and receives semantic [`UIEvent`] interactions.

use std::sync::mpsc;

pub use sofuu_ui::{InputEvent, Scene, ScenePatch, UIEvent};

/// The `sofuu-ui` draw ABI this crate's shaders and vertex/instance layouts
/// were built for. Bump this in the same change that bumps
/// [`sofuu_ui::draw::SOFUU_DRAW_ABI_VERSION`]; the assertion below turns a
/// mismatch into a `cargo check` failure instead of corrupted GPU geometry.
pub const EXPECTED_GPU_DRAW_ABI_VERSION: u32 = 1;
const _: () = assert!(sofuu_ui::draw::SOFUU_DRAW_ABI_VERSION == EXPECTED_GPU_DRAW_ABI_VERSION);

pub mod backend;
pub mod ffi;
pub mod offscreen;
mod loop_core;

#[derive(Clone, Debug)]
pub struct EmulatorWindow {
    /// Active platform id (`ios`, `android`, `macos`, `windows`, `linux`).
    pub platform: String,
    /// Active device id.
    pub device: String,
    /// `sofuu-emu profiles --json` manifest for the native strip popups.
    pub profiles_json: String,
}

#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: f32,
    pub height: f32,
    pub min_width: f32,
    pub min_height: f32,
    /// Native title bar / resize chrome. Emulator screens set this to `false`
    /// so the client area is exactly the simulated screen.
    pub decorations: bool,
    /// Whether the user may free-resize the window. Device presets are fixed.
    pub resizable: bool,
    /// Cap the backing scale (e.g. `2.625` for a 411 pt phone). `None` uses the
    /// display's native backing scale.
    pub scale: Option<f32>,
    /// Screen corner radius in points (`0.0` = square, for v1 parity).
    pub corner_radius: f32,
    /// Attach the native emulator control strip (E6).
    pub strip: bool,
    /// Draw a live touch reticle while the pointer is down (E7). Captures
    /// never enable it.
    pub reticle: bool,
    /// Emulator profile/device metadata for the strip. `None` for plain apps.
    pub emulator: Option<EmulatorWindow>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Sofuu".to_string(),
            width: 1024.0,
            height: 680.0,
            min_width: 360.0,
            min_height: 280.0,
            decorations: true,
            resizable: true,
            scale: None,
            corner_radius: 18.0,
            strip: false,
            reticle: false,
            emulator: None,
        }
    }
}

/// Thread-safe handle passed to Thread 1 (agent / runtime loop) to drive the native GPU UI.
pub struct UiHandle {
    patch_tx: mpsc::Sender<ScenePatch>,
    input_tx: mpsc::Sender<InputEvent>,
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

    /// Split into a cloneable patch/input sender + event receiver so multiple
    /// agent tasks can push UI updates and synthetic input.
    pub fn split(self) -> (UiPatchSender, mpsc::Receiver<UIEvent>) {
        (
            UiPatchSender {
                patch_tx: self.patch_tx,
                input_tx: self.input_tx,
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
    input_tx: mpsc::Sender<InputEvent>,
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

    /// Inject a synthetic input event (on-screen keyboard, agent gesture).
    pub fn send_input(&self, event: InputEvent) -> bool {
        let ok = self.input_tx.send(event).is_ok();
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
    loop_core::run(config, initial_scene, on_ready)
}

/// Freeze/unfreeze the live engine clock (E7 step-frame).
pub fn set_paused(paused: bool) {
    loop_core::set_paused(paused);
}

/// Advance a paused engine by exactly one 60 Hz frame.
pub fn step_frame() {
    loop_core::step_frame();
}

pub fn is_paused() -> bool {
    loop_core::is_paused()
}

/// Toggle the live layout-bounds overlay (emulator strip "Bounds").
pub fn set_bounds(on: bool) {
    loop_core::set_bounds(on);
}

pub fn is_bounds() -> bool {
    loop_core::is_bounds()
}

/// Compile-time backend family for diagnostics (`metal`, `wayland/x11 + egl`, ...).
pub const fn compiled_backend_family() -> &'static str {
    backend::compiled_backend_family()
}

#[cfg(test)]
mod tests {
    #[test]
    fn pause_and_step_are_idempotent() {
        assert!(!crate::is_paused());
        crate::set_paused(true);
        assert!(crate::is_paused());
        crate::step_frame();
        crate::step_frame();
        crate::set_paused(false);
        assert!(!crate::is_paused());
    }

    #[test]
    fn bounds_toggle_roundtrips() {
        assert!(!crate::is_bounds());
        crate::set_bounds(true);
        assert!(crate::is_bounds());
        crate::set_bounds(false);
        assert!(!crate::is_bounds());
    }

    use sofuu_ui::{
        EFFECT_AURORA, EFFECT_BORDER_BEAM, EFFECT_CURSOR_GLOW, EFFECT_PULSE_RING, EFFECT_SHIMMER,
        EFFECT_TOUCH_RIPPLE, EFFECT_VOICE_ORB, PARTICLE_CIRCLE, PARTICLE_RIBBON, PARTICLE_SPARK,
    };

    #[test]
    fn msl_shader_source_is_nonempty_and_declares_all_four_kernels() {
        let src = sofuu_ui::shaders::METAL_SRC;
        assert!(src.contains("quad_vert") && src.contains("quad_frag"));
        assert!(src.contains("ribbon_vert") && src.contains("ribbon_frag"));
        assert!(src.contains("stroke_vert") && src.contains("stroke_frag"));
        assert!(src.contains("glyph_vert") && src.contains("glyph_frag"));
    }

    /// The effect/particle switch indices are a wire contract shared by `fx.rs`, every GLSL
    /// flavor and the MSL kernels; a renumbering in one place must fail here instead of
    /// rendering the wrong effect on one backend.
    #[test]
    fn fx_kind_ids_match_every_shader_family() {
        assert_eq!(
            [
                EFFECT_SHIMMER,
                EFFECT_TOUCH_RIPPLE,
                EFFECT_BORDER_BEAM,
                EFFECT_CURSOR_GLOW,
                EFFECT_AURORA,
                EFFECT_PULSE_RING,
                EFFECT_VOICE_ORB,
            ],
            [0, 1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            [PARTICLE_CIRCLE, PARTICLE_RIBBON, PARTICLE_SPARK],
            [0, 1, 2]
        );

        let metal = sofuu_ui::shaders::METAL_SRC;
        let msl_effects = [
            "shimmer",
            "ripple",
            "beam",
            "cursor_glow",
            "aurora",
            "pulse_ring",
            "voice_orb",
        ];
        for (id, name) in msl_effects.iter().enumerate() {
            let arm = format!("case {id}u: c = effect_{name}(in); break;");
            assert!(metal.contains(&arm), "MSL effect arm missing: {arm}");
        }
        assert!(metal.contains("if (in.shape == 1u)"));
        assert!(metal.contains("if (in.shape == 2u)"));
        for kernel in [
            "effect_vert",
            "effect_frag",
            "particle_vert",
            "particle_frag",
            "backdrop_vert",
            "backdrop_frag",
            "blur_downsample",
            "blur_kawase",
        ] {
            assert!(metal.contains(kernel), "MSL kernel missing: {kernel}");
        }

        for src in [
            sofuu_ui::shaders::GLSL_330_SRC,
            sofuu_ui::shaders::GLSL_ES3_SRC,
            sofuu_ui::shaders::GLSL_450_SRC,
        ] {
            for id in 0..=6u32 {
                assert!(
                    src.contains(&format!("v_kind == {id}u")),
                    "GLSL effect branch {id} missing"
                );
            }
            for shape in 1..=2u32 {
                assert!(
                    src.contains(&format!("v_shape == {shape}u")),
                    "GLSL particle shape branch {shape} missing"
                );
            }
        }
    }
}
