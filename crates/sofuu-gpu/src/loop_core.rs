//! Shared 2-thread UI driver: scene compilation, spring/scroll pacing and 0%-idle waits.
//!
//! Thread 0 owns the native window + GPU backend through [`NativeWindowBackend`]. Thread 1+
//! pushes [`ScenePatch`] values and receives semantic [`UIEvent`]s through [`crate::UiHandle`].
//! Once every spring has converged and no patch is pending, Thread 0 blocks in the OS event
//! loop (Wayland/X11 `poll`, Win32 `MsgWaitForMultipleObjectsEx`, AppKit `nextEventMatchingMask`)
//! so a settled UI consumes 0% CPU/GPU. A patch from any thread wakes it immediately.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use sofuu_ui::{
    DrawList, FrameUniforms, InputEvent, Scene, ScenePatch, ThemeMode, UIEvent, UiEngine,
};

use crate::backend;
use crate::{UiHandle, WindowConfig};

/// Step-frame control (E7). `PAUSED` freezes the engine clock; `PENDING_STEPS`
/// advances exactly `1/60 s` per request so springs can be inspected mid-flight.
static PAUSED: AtomicBool = AtomicBool::new(false);
static PENDING_STEPS: AtomicU32 = AtomicU32::new(0);
/// Live layout-bounds overlay (E6 strip "Bounds"). Off for captures.
static BOUNDS: AtomicBool = AtomicBool::new(false);

/// Freeze/unfreeze the engine clock. Pausing wakes Thread 0 so the UI repaints
/// once and then blocks instead of spinning at 120 Hz.
pub fn set_paused(paused: bool) {
    PAUSED.store(paused, Ordering::Release);
    backend::wake_main_loop();
}

pub fn is_paused() -> bool {
    PAUSED.load(Ordering::Acquire)
}

/// Advance a paused engine by exactly one 60 Hz frame.
pub fn step_frame() {
    PENDING_STEPS.fetch_add(1, Ordering::AcqRel);
    backend::wake_main_loop();
}

/// Toggle the live layout-bounds overlay.
pub fn set_bounds(on: bool) {
    BOUNDS.store(on, Ordering::Release);
    backend::wake_main_loop();
}

pub fn is_bounds() -> bool {
    BOUNDS.load(Ordering::Acquire)
}

/// Compile and present frames until the native window closes.
pub fn run<F>(config: WindowConfig, initial_scene: Scene, on_ready: F) -> Result<(), String>
where
    F: FnOnce(UiHandle),
{
    let mut engine = UiEngine::new(
        initial_scene,
        [config.width.max(1.0), config.height.max(1.0)],
    );
    let mut backend = backend::create_window(&config, &engine.atlas)?;

    let (w, h, mut scale) = backend.metrics();
    scale = scale.max(1.0);
    engine.set_top_inset(backend.top_inset());
    engine.set_touch_reticle(config.reticle);
    if let Some(emu) = &config.emulator {
        engine.set_scroll_physics(sofuu_ui::hit_test::ScrollPhysics::for_platform(&emu.platform));
    }
    let _ = engine.handle_input(InputEvent::Resize {
        width: w.max(1.0),
        height: h.max(1.0),
        scale_factor: scale,
    });

    let (patch_tx, patch_rx) = mpsc::channel::<ScenePatch>();
    let (input_tx, input_rx) = mpsc::channel::<InputEvent>();
    let (ui_ev_tx, ui_ev_rx) = mpsc::channel::<UIEvent>();

    on_ready(UiHandle {
        patch_tx,
        input_tx,
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
        // Synthetic input from Thread 1 (on-screen keyboard, agent gestures)
        // joins the native event stream before it is dispatched.
        while let Ok(ev) = input_rx.try_recv() {
            raw_events.push(ev);
        }
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

        // Accessibility / window lifecycle hints drive the ambient-effect contract: freeze
        // continuous shaders when the OS asks for reduced motion or the app is backgrounded.
        engine.set_reduced_motion(backend.reduce_motion());
        engine.set_ambient_paused(!backend.focused());

        let now = Instant::now();
        // The animation step is clamped for spring/scroll stability; the engine
        // clock gets the real elapsed time so a wake scheduled at a caret
        // deadline advances the blink by the full sleep, not by 50 ms.
        let mut elapsed = (now - last_frame).as_secs_f64();
        let mut dt = (elapsed as f32).clamp(1.0 / 240.0, 0.05);
        last_frame = now;
        if PAUSED.load(Ordering::Acquire) {
            let pending = PENDING_STEPS.load(Ordering::Acquire);
            if pending > 0 {
                PENDING_STEPS.fetch_sub(1, Ordering::AcqRel);
                dt = 1.0 / 60.0;
                engine.mark_dirty();
            } else {
                // Frozen clock: no spring/scroll advance, no repaint.
                dt = 0.0;
                elapsed = 0.0;
            }
        }

        engine.set_show_bounds(BOUNDS.load(Ordering::Acquire));
        let status = engine.tick_with_elapsed(dt, elapsed, &mut draw_list);

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
        // A paused engine blocks even mid-spring: only `step_frame` wakes it. A timed deadline
        // (caret blink) is a wake too, but it does not need a 120 Hz present loop: sleep until
        // the deadline and render exactly one toggle frame.
        // The draw list is the whole schedule: `needs_next_frame` covers
        // springs, kinetic scroll, effects and timed wakes; `next_wake_s` is
        // the absolute engine-clock deadline for the caret blink. The host
        // never reaches into `UiEngine` state to decide whether to sleep.
        let wake_s = draw_list
            .next_wake_s
            .map(|due| (due - engine.now_s).max(0.0));
        let wait_forever =
            PAUSED.load(Ordering::Acquire) || !draw_list.needs_next_frame;
        let timeout = wake_s
            .map(|s| s.clamp(1.0 / 1000.0, 1.0))
            .unwrap_or(1.0 / 120.0);
        let alive = backend.poll_or_wait(wait_forever, timeout, &mut raw_events);
        if !alive {
            break;
        }
    }

    Ok(())
}
