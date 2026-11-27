//! Shared 2-thread UI driver: scene compilation, spring/scroll pacing and 0%-idle waits.
//!
//! Thread 0 owns the native window + GPU backend through [`NativeWindowBackend`]. Thread 1+
//! pushes [`ScenePatch`] values and receives semantic [`UIEvent`]s through [`crate::UiHandle`].
//! Once every spring has converged and no patch is pending, Thread 0 blocks in the OS event
//! loop (Wayland/X11 `poll`, Win32 `MsgWaitForMultipleObjectsEx`, AppKit `nextEventMatchingMask`)
//! so a settled UI consumes 0% CPU/GPU. A patch from any thread wakes it immediately.

use std::sync::mpsc;
use std::time::Instant;

use sofuu_ui::{DrawList, FrameUniforms, InputEvent, Scene, ScenePatch, ThemeMode, UIEvent, UiEngine};

use crate::backend;
use crate::{UiHandle, WindowConfig};

/// Compile and present frames until the native window closes.
pub fn run<F>(config: WindowConfig, initial_scene: Scene, on_ready: F) -> Result<(), String>
where
    F: FnOnce(UiHandle),
{
    let mut engine = UiEngine::new(initial_scene, [config.width.max(1.0), config.height.max(1.0)]);
    let mut backend = backend::create_window(&config, &engine.atlas)?;

    let (w, h, mut scale) = backend.metrics();
    scale = scale.max(1.0);
    engine.set_top_inset(backend.top_inset());
    let _ = engine.handle_input(InputEvent::Resize {
        width: w.max(1.0),
        height: h.max(1.0),
        scale_factor: scale,
    });

    let (patch_tx, patch_rx) = mpsc::channel::<ScenePatch>();
    let (ui_ev_tx, ui_ev_rx) = mpsc::channel::<UIEvent>();

    on_ready(UiHandle {
        patch_tx,
        event_rx: ui_ev_rx,
        wake_fn: backend::wake_main_loop,
    });

    let start = Instant::now();
    let mut last_frame = start;
    let mut raw_events: Vec<InputEvent> = Vec::with_capacity(32);
    let mut draw_list = DrawList::new();
    let mut last_dark: Option<bool> = None;

    loop {
        while let Ok(patch) = patch_rx.try_recv() {
            engine.apply_patch(patch);
        }

        let mut scale_changed = false;
        for ev in raw_events.drain(..) {
            // DPI changes (Windows WM_DPICHANGED, Wayland fractional-scale events) arrive as
            // Resize; track the latest scale so FrameUniforms and the render viewport stay sharp.
            if let InputEvent::Resize { scale_factor, .. } = &ev {
                let next = (*scale_factor).max(1.0);
                scale_changed |= (next - scale).abs() > 1e-3;
                scale = next;
            }
            for ui_ev in engine.handle_input(ev) {
                let _ = ui_ev_tx.send(ui_ev);
            }
        }
        // A scale-only change keeps the logical viewport identical, so `handle_input` does not
        // dirty the scene; force a recompile/repaint at the new physical pixel size anyway.
        if scale_changed {
            engine.mark_dirty();
        }

        // Rebuild a lost surface/context even if the scene is settled; otherwise an idle
        // window would stay frozen until the next user interaction. When recovery keeps
        // failing (Android app backgrounded, GPU reset) wait for a host event with a slow
        // backoff instead of recompiling every frame.
        if backend.context_lost() {
            backend.recover_context();
            if backend.context_lost() {
                if !backend.poll_or_wait(false, 1.0, &mut raw_events) {
                    break;
                }
                continue;
            }
            engine.mark_dirty();
        }

        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f32().clamp(1.0 / 240.0, 0.05);
        last_frame = now;

        let status = engine.tick_and_compile(dt, &mut draw_list);

        if status.compiled {
            let dark = matches!(engine.theme.mode, ThemeMode::Dark);
            if last_dark != Some(dark) {
                backend.set_theme_dark(dark);
                last_dark = Some(dark);
            }
            let vp = engine.viewport();
            let uniforms = FrameUniforms {
                viewport_size: vp,
                atlas_size: [engine.atlas.width as f32, engine.atlas.height as f32],
                scale_factor: scale,
                time_sec: (now - start).as_secs_f32(),
                _pad: [0.0; 2],
            };
            backend.render(&uniforms, &draw_list);

            if backend.context_lost() {
                backend.recover_context();
                // A surface/context rebuild invalidates GPU-side text/layout uploads.
                engine.mark_dirty();
            }
        }

        // 120 Hz while springs/kinetic scroll move; blocking OS wait (0% CPU/GPU) once settled.
        let wait_forever = !(status.animating || engine.is_dirty_or_animating());
        let alive = backend.poll_or_wait(wait_forever, 1.0 / 120.0, &mut raw_events);
        if !alive {
            break;
        }
    }

    Ok(())
}
