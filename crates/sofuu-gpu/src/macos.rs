//! Direct macOS AppKit (`NSWindow` + `CAMetalLayer`) + Metal 2D backend.

use std::ffi::{c_void, CString};
use std::sync::mpsc;
use std::time::Instant;

use sofuu_ui::{DrawList, FrameUniforms, InputEvent, Scene, ScenePatch, UIEvent, UiEngine};

use crate::{UiHandle, WindowConfig};

const MSL_SOURCE: &str = concat!(include_str!("shaders/msl/ui_shaders.metal"), "\0");

const EV_POINTER_MOVE: u32 = 1;
const EV_POINTER_DOWN: u32 = 2;
const EV_POINTER_UP: u32 = 3;
const EV_SCROLL: u32 = 4;
const EV_RESIZE: u32 = 5;
const EV_CLOSE: u32 = 6;
const EV_KEY_DOWN: u32 = 7;
const EV_TEXT_CHAR: u32 = 8;

#[repr(C)]
struct SofuuMacWindow {
    _unused: [u8; 0],
}

type EventCb = extern "C" fn(*mut c_void, u32, f32, f32, f32, f32, u32);

extern "C" {
    fn sofuu_macos_create_window(
        title_utf8: *const libc::c_char,
        width: f32,
        height: f32,
        min_w: f32,
        min_h: f32,
        msl_source_utf8: *const libc::c_char,
        atlas_pixels: *const u8,
        atlas_w: u32,
        atlas_h: u32,
        cb: EventCb,
        user_ctx: *mut c_void,
    ) -> *mut SofuuMacWindow;
    fn sofuu_macos_destroy_window(win: *mut SofuuMacWindow);
    fn sofuu_macos_wake();
    fn sofuu_macos_get_metrics(
        win: *mut SofuuMacWindow,
        out_w: *mut f32,
        out_h: *mut f32,
        out_scale: *mut f32,
    );
    fn sofuu_macos_render(
        win: *mut SofuuMacWindow,
        uniforms: *const FrameUniforms,
        clips: *const c_void,
        clips_count: u32,
        quads: *const c_void,
        quads_count: u32,
        ribbons: *const c_void,
        ribbons_count: u32,
        strokes: *const c_void,
        strokes_count: u32,
        glyphs: *const c_void,
        glyphs_count: u32,
        cmds: *const c_void,
        cmds_count: u32,
    );
    fn sofuu_macos_poll_or_wait(
        win: *mut SofuuMacWindow,
        wait_forever: libc::c_int,
        timeout_sec: f64,
    ) -> libc::c_int;
}

struct EventCollector {
    raw_events: Vec<InputEvent>,
    scale_factor: f32,
    closed: bool,
}

extern "C" fn on_native_event(
    ctx: *mut c_void,
    kind: u32,
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    u: u32,
) {
    if ctx.is_null() {
        return;
    }
    let state = unsafe { &mut *(ctx as *mut EventCollector) };
    match kind {
        EV_POINTER_MOVE => state.raw_events.push(InputEvent::PointerMove { x, y }),
        EV_POINTER_DOWN => state.raw_events.push(InputEvent::PointerDown {
            x,
            y,
            button: u as u8,
        }),
        EV_POINTER_UP => state.raw_events.push(InputEvent::PointerUp {
            x,
            y,
            button: u as u8,
        }),
        EV_SCROLL => state.raw_events.push(InputEvent::Scroll {
            x,
            y,
            delta_x: a,
            delta_y: b,
        }),
        EV_RESIZE => {
            state.scale_factor = a.max(1.0);
            state.raw_events.push(InputEvent::Resize {
                width: x,
                height: y,
                scale_factor: state.scale_factor,
            });
        }
        EV_CLOSE => {
            state.closed = true;
            state.raw_events.push(InputEvent::CloseRequested);
        }
        EV_KEY_DOWN => {
            state.raw_events.push(InputEvent::KeyDown {
                key: format!("keycode:{u}"),
                modifiers: 0,
            });
        }
        EV_TEXT_CHAR => {
            if let Some(ch) = char::from_u32(u) {
                state.raw_events.push(InputEvent::TextInput {
                    text: ch.to_string(),
                });
            }
        }
        _ => {}
    }
}

pub fn wake_main_loop() {
    unsafe {
        sofuu_macos_wake();
    }
}

pub fn run_macos_window<F>(config: WindowConfig, initial_scene: Scene, spawn_agent: F) -> Result<(), String>
where
    F: FnOnce(UiHandle),
{
    let mut engine = UiEngine::new(initial_scene, [config.width, config.height]);
    engine.set_top_inset(28.0);
    let mut draw_list = DrawList::new();

    let title_c = CString::new(config.title.as_str()).unwrap_or_else(|_| CString::new("Sofuu").unwrap());
    let mut collector = Box::new(EventCollector {
        raw_events: Vec::with_capacity(32),
        scale_factor: 1.0,
        closed: false,
    });

    let win = unsafe {
        sofuu_macos_create_window(
            title_c.as_ptr(),
            config.width,
            config.height,
            config.min_width,
            config.min_height,
            MSL_SOURCE.as_ptr() as *const libc::c_char,
            engine.atlas.pixels.as_ptr(),
            engine.atlas.width,
            engine.atlas.height,
            on_native_event,
            (&mut *collector as *mut EventCollector).cast(),
        )
    };
    if win.is_null() {
        return Err("failed to initialize macOS NSWindow / Metal device".to_string());
    }

    let mut w = config.width;
    let mut h = config.height;
    let mut scale = 1.0f32;
    unsafe {
        sofuu_macos_get_metrics(win, &mut w, &mut h, &mut scale);
    }
    collector.scale_factor = scale.max(1.0);
    let _ = engine.handle_input(InputEvent::Resize {
        width: w,
        height: h,
        scale_factor: collector.scale_factor,
    });

    let (patch_tx, patch_rx) = mpsc::channel::<ScenePatch>();
    let (ui_ev_tx, ui_ev_rx) = mpsc::channel::<UIEvent>();

    spawn_agent(UiHandle {
        patch_tx,
        event_rx: ui_ev_rx,
        wake_fn: wake_main_loop,
    });

    let start = Instant::now();
    let mut last_frame = start;

    loop {
        while let Ok(patch) = patch_rx.try_recv() {
            engine.apply_patch(patch);
        }

        for ev in collector.raw_events.drain(..) {
            for ui_ev in engine.handle_input(ev) {
                let _ = ui_ev_tx.send(ui_ev);
            }
        }

        if collector.closed {
            break;
        }

        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f32().clamp(1.0 / 240.0, 0.05);
        last_frame = now;

        let status = engine.tick_and_compile(dt, &mut draw_list);
        if status.compiled {
            let vp = engine.viewport();
            let uniforms = FrameUniforms {
                viewport_size: vp,
                atlas_size: [engine.atlas.width as f32, engine.atlas.height as f32],
                scale_factor: collector.scale_factor,
                time_sec: (now - start).as_secs_f32(),
                _pad: [0.0; 2],
            };
            unsafe {
                sofuu_macos_render(
                    win,
                    &uniforms,
                    draw_list.clips.as_ptr().cast(),
                    draw_list.clips.len() as u32,
                    draw_list.quads.as_ptr().cast(),
                    draw_list.quads.len() as u32,
                    draw_list.ribbons.as_ptr().cast(),
                    draw_list.ribbons.len() as u32,
                    draw_list.strokes.as_ptr().cast(),
                    draw_list.strokes.len() as u32,
                    draw_list.glyphs.as_ptr().cast(),
                    draw_list.glyphs.len() as u32,
                    draw_list.cmds.as_ptr().cast(),
                    draw_list.cmds.len() as u32,
                );
            }
        }

        // Sleep completely (0% CPU/GPU) when idle; run at 120Hz cadence while springs/scroll animate.
        let wait_forever = if status.animating || engine.is_dirty_or_animating() {
            0
        } else {
            1
        };
        let alive = unsafe { sofuu_macos_poll_or_wait(win, wait_forever, 1.0 / 120.0) };
        if alive == 0 {
            break;
        }
    }

    unsafe {
        sofuu_macos_destroy_window(win);
    }
    Ok(())
}
