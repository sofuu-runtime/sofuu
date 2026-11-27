//! Direct macOS AppKit (`NSWindow` + `CAMetalLayer`) + Metal 2D backend.
//!
//! Extracted from the original `macos.rs` into the shared [`NativeWindowBackend`] contract.
//! Behaviour is intentionally byte-for-byte equivalent: frameless transparent title bar,
//! 28 px top inset, native window drag on the title strip, 120 Hz animation cadence and a
//! blocking AppKit idle wait that consumes 0% CPU/GPU once the scene settles.

use std::ffi::{c_void, CString};

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas, InputEvent};

use crate::backend::NativeWindowBackend;
use crate::WindowConfig;

const MSL_SOURCE: &str = concat!(include_str!("../shaders/msl/ui_shaders.metal"), "\0");

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

pub struct MacosMetalBackend {
    win: *mut SofuuMacWindow,
    collector: Box<EventCollector>,
}

// The backend is created and used exclusively on Thread 0 (the AppKit main thread).
unsafe impl Send for MacosMetalBackend {}

impl MacosMetalBackend {
    pub fn new(config: &WindowConfig, atlas: &GlyphAtlas) -> Result<Self, String> {
        let title_c =
            CString::new(config.title.as_str()).unwrap_or_else(|_| CString::new("Sofuu").unwrap());
        let collector = Box::new(EventCollector {
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
                atlas.pixels.as_ptr(),
                atlas.width,
                atlas.height,
                on_native_event,
                (&*collector as *const EventCollector as *mut EventCollector).cast(),
            )
        };
        if win.is_null() {
            return Err(
                "E_UI_NO_DISPLAY: failed to initialize macOS NSWindow / Metal device".to_string(),
            );
        }

        let mut backend = Self { win, collector };
        let (_, _, scale) = backend.metrics();
        backend.collector.scale_factor = scale.max(1.0);
        Ok(backend)
    }
}

impl Drop for MacosMetalBackend {
    fn drop(&mut self) {
        unsafe { sofuu_macos_destroy_window(self.win) };
        self.win = std::ptr::null_mut();
    }
}

impl NativeWindowBackend for MacosMetalBackend {
    fn metrics(&self) -> (f32, f32, f32) {
        let mut w = 1.0f32;
        let mut h = 1.0f32;
        let mut scale = 1.0f32;
        unsafe { sofuu_macos_get_metrics(self.win, &mut w, &mut h, &mut scale) };
        (w.max(1.0), h.max(1.0), scale.max(1.0))
    }

    fn poll_or_wait(
        &mut self,
        wait_forever: bool,
        timeout_sec: f64,
        out_events: &mut Vec<InputEvent>,
    ) -> bool {
        let alive = unsafe {
            sofuu_macos_poll_or_wait(
                self.win,
                if wait_forever { 1 } else { 0 },
                timeout_sec,
            )
        };
        for ev in self.collector.raw_events.drain(..) {
            if let InputEvent::Resize { scale_factor, .. } = &ev {
                self.collector.scale_factor = (*scale_factor).max(1.0);
            }
            out_events.push(ev);
        }
        alive != 0 && !self.collector.closed
    }

    fn render(&mut self, uniforms: &FrameUniforms, dl: &DrawList) {
        unsafe {
            sofuu_macos_render(
                self.win,
                uniforms,
                dl.clips.as_ptr().cast(),
                dl.clips.len() as u32,
                dl.quads.as_ptr().cast(),
                dl.quads.len() as u32,
                dl.ribbons.as_ptr().cast(),
                dl.ribbons.len() as u32,
                dl.strokes.as_ptr().cast(),
                dl.strokes.len() as u32,
                dl.glyphs.as_ptr().cast(),
                dl.glyphs.len() as u32,
                dl.cmds.as_ptr().cast(),
                dl.cmds.len() as u32,
            );
        }
    }

    fn top_inset(&self) -> f32 {
        28.0
    }

    fn backend_name(&self) -> &'static str {
        "metal"
    }
}

pub fn wake_main_loop() {
    unsafe { sofuu_macos_wake() };
}
