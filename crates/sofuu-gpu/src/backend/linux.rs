//! Linux / FreeBSD / OpenBSD window backend: Wayland (`xdg_wm_base`) with an X11 fallback,
//! both presenting through EGL (GLES 3.0 first, desktop OpenGL 3.3 second).
//!
//! Window creation, EGL setup, input translation and the `eventfd`/`poll` idle wake all live in
//! `native/linux_wayland_egl.c` so this crate never hard-links `libwayland-client`, `libX11`,
//! `libEGL` or `libGLESv2`. The shared [`GlesRenderer`](super::gles::GlesRenderer) owns every GL
//! object and draws the exact same instanced pipelines as Metal.
//!
//! Set `SOFUU_UI_BACKEND=vulkan` to try the direct Vulkan renderer first; any Vulkan
//! initialization failure is reported on stderr and falls back to GL/GLES without aborting.
//! `SOFUU_UI_SCALE=1.5` overrides the compositor-reported DPI scale.

use std::ffi::{c_void, CStr, CString};

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas, InputEvent};

use super::gles::{GlesRenderer, GlslFlavor};
use super::vulkan::{NativeSurface, SurfaceKind};
use super::{key_name, NativeWindowBackend};
use crate::WindowConfig;

const EV_POINTER_MOVE: u32 = 1;
const EV_POINTER_DOWN: u32 = 2;
const EV_POINTER_UP: u32 = 3;
const EV_SCROLL: u32 = 4;
const EV_RESIZE: u32 = 5;
const EV_CLOSE: u32 = 6;
const EV_KEY_DOWN: u32 = 7;
const EV_TEXT_CHAR: u32 = 8;

#[repr(C)]
struct SofuuLinuxWindow {
    _unused: [u8; 0],
}

type EventCb = extern "C" fn(*mut c_void, u32, f32, f32, f32, f32, u32);

extern "C" {
    fn sofuu_linux_create_window(
        title_utf8: *const libc::c_char,
        width: f32,
        height: f32,
        min_w: f32,
        min_h: f32,
        cb: EventCb,
        user_ctx: *mut c_void,
    ) -> *mut SofuuLinuxWindow;
    fn sofuu_linux_destroy_window(win: *mut SofuuLinuxWindow);
    fn sofuu_linux_get_metrics(
        win: *mut SofuuLinuxWindow,
        out_w: *mut f32,
        out_h: *mut f32,
        out_scale: *mut f32,
    );
    fn sofuu_linux_poll_or_wait(
        win: *mut SofuuLinuxWindow,
        wait_forever: libc::c_int,
        timeout_sec: f64,
    ) -> libc::c_int;
    fn sofuu_linux_make_current(win: *mut SofuuLinuxWindow) -> libc::c_int;
    fn sofuu_linux_swap_buffers(win: *mut SofuuLinuxWindow);
    fn sofuu_linux_get_proc_address(name: *const libc::c_char) -> *mut c_void;
    fn sofuu_linux_wake();
    fn sofuu_linux_set_theme_dark(win: *mut SofuuLinuxWindow, dark: libc::c_int);
    fn sofuu_linux_context_lost(win: *mut SofuuLinuxWindow) -> libc::c_int;
    fn sofuu_linux_recover(win: *mut SofuuLinuxWindow) -> libc::c_int;
    fn sofuu_linux_api_kind(win: *mut SofuuLinuxWindow) -> libc::c_int;
    fn sofuu_linux_surface_kind(win: *mut SofuuLinuxWindow) -> libc::c_int;
    fn sofuu_linux_display_handle(win: *mut SofuuLinuxWindow) -> *mut c_void;
    fn sofuu_linux_window_handle(win: *mut SofuuLinuxWindow) -> *mut c_void;
    fn sofuu_linux_last_error() -> *const libc::c_char;
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
        EV_KEY_DOWN => state.raw_events.push(InputEvent::KeyDown {
            key: key_name(u),
            // The shims pass the SOFUU_MOD_* bitmask in `a` (Shift=1, Ctrl=2, Alt=4, Meta=8).
            modifiers: a.max(0.0) as u32,
        }),
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

enum Presenter {
    Gles(GlesRenderer),
    Vulkan(super::vulkan::VulkanRenderer),
}

pub struct LinuxBackend {
    win: *mut SofuuLinuxWindow,
    collector: Box<EventCollector>,
    presenter: Option<Presenter>,
    surface_kind: SurfaceKind,
}

// Thread 0 owns the Wayland/X11 display connection and EGL context for the whole session.
unsafe impl Send for LinuxBackend {}

impl LinuxBackend {
    pub fn new(config: &WindowConfig, atlas: &GlyphAtlas) -> Result<Self, String> {
        let title_c =
            CString::new(config.title.as_str()).unwrap_or_else(|_| CString::new("Sofuu").unwrap());
        let collector = Box::new(EventCollector {
            raw_events: Vec::with_capacity(32),
            scale_factor: 1.0,
            closed: false,
        });
        let win = unsafe {
            sofuu_linux_create_window(
                title_c.as_ptr(),
                config.width,
                config.height,
                config.min_width,
                config.min_height,
                on_native_event,
                (&*collector as *const EventCollector as *mut EventCollector).cast(),
            )
        };
        if win.is_null() {
            let msg = unsafe {
                let p = sofuu_linux_last_error();
                if p.is_null() {
                    "E_UI_NO_DISPLAY: no Wayland or X11 display server available; omit --ui to run headless".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(msg);
        }

        let surface_kind = match unsafe { sofuu_linux_surface_kind(win) } {
            1 => SurfaceKind::Xlib,
            _ => SurfaceKind::Wayland,
        };

        let want_vulkan = std::env::var("SOFUU_UI_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("vulkan"))
            .unwrap_or(false);

        let mut presenter = None;
        if want_vulkan {
            let surface = NativeSurface {
                kind: surface_kind,
                display: unsafe { sofuu_linux_display_handle(win) },
                window: unsafe { sofuu_linux_window_handle(win) },
            };
            match unsafe { super::vulkan::VulkanRenderer::new(surface, atlas) } {
                Ok(vk) => presenter = Some(Presenter::Vulkan(vk)),
                Err(err) => {
                    eprintln!("sofuu-gpu: Vulkan unavailable ({err}); falling back to GL/GLES");
                }
            }
        }

        let presenter = match presenter {
            Some(p) => p,
            None => {
                let flavor = if unsafe { sofuu_linux_api_kind(win) } == 1 {
                    GlslFlavor::Desktop330
                } else {
                    GlslFlavor::Es300
                };
                let renderer = match GlesRenderer::new(
                    |name| {
                        let c = CString::new(name).unwrap_or_default();
                        unsafe { sofuu_linux_get_proc_address(c.as_ptr()) }
                    },
                    atlas,
                    flavor,
                ) {
                    Ok(renderer) => renderer,
                    Err(err) => {
                        // No backend owns the native window yet; release it before returning.
                        unsafe { sofuu_linux_destroy_window(win) };
                        return Err(err);
                    }
                };
                Presenter::Gles(renderer)
            }
        };

        let mut backend = Self {
            win,
            collector,
            presenter: Some(presenter),
            surface_kind,
        };
        let (_, _, scale) = backend.metrics();
        backend.collector.scale_factor = scale.max(1.0);
        Ok(backend)
    }

}

impl Drop for LinuxBackend {
    fn drop(&mut self) {
        // Release GPU objects while the EGL context is still current, then destroy the window.
        unsafe { sofuu_linux_make_current(self.win) };
        drop(self.presenter.take());
        unsafe { sofuu_linux_destroy_window(self.win) };
        self.win = std::ptr::null_mut();
    }
}

impl NativeWindowBackend for LinuxBackend {
    fn metrics(&self) -> (f32, f32, f32) {
        let mut w = 1.0f32;
        let mut h = 1.0f32;
        let mut scale = 1.0f32;
        unsafe { sofuu_linux_get_metrics(self.win, &mut w, &mut h, &mut scale) };
        (w.max(1.0), h.max(1.0), scale.max(1.0))
    }

    fn poll_or_wait(
        &mut self,
        wait_forever: bool,
        timeout_sec: f64,
        out_events: &mut Vec<InputEvent>,
    ) -> bool {
        let alive = unsafe {
            sofuu_linux_poll_or_wait(
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
        match self.presenter.as_mut().unwrap() {
            Presenter::Gles(renderer) => {
                // EGL/WGL may have lost the surface/context; recover before issuing GL calls
                // against a dead context.
                if unsafe { sofuu_linux_context_lost(self.win) } != 0 {
                    return;
                }
                let _ = unsafe { sofuu_linux_make_current(self.win) };
                if let Err(err) = renderer.render(uniforms, dl) {
                    eprintln!("sofuu-gpu: GL render error: {err}");
                }
                unsafe { sofuu_linux_swap_buffers(self.win) };
            }
            Presenter::Vulkan(vk) => {
                if let Err(err) = vk.render(uniforms, dl) {
                    eprintln!("sofuu-gpu: Vulkan render error: {err}");
                }
            }
        }
    }

    fn set_theme_dark(&mut self, dark: bool) {
        unsafe { sofuu_linux_set_theme_dark(self.win, if dark { 1 } else { 0 }) };
    }

    fn backend_name(&self) -> &'static str {
        match self.presenter.as_ref().unwrap() {
            Presenter::Gles(renderer) => match renderer.flavor() {
                GlslFlavor::Desktop330 => "gl",
                GlslFlavor::Es300 => "gles",
            },
            Presenter::Vulkan(_) => "vulkan",
        }
    }

    fn context_lost(&self) -> bool {
        match self.presenter.as_ref().unwrap() {
            Presenter::Gles(_) => unsafe { sofuu_linux_context_lost(self.win) != 0 },
            Presenter::Vulkan(vk) => vk.context_lost(),
        }
    }

    fn recover_context(&mut self) {
        match self.presenter.as_mut().unwrap() {
            Presenter::Gles(renderer) => {
                let recreated = unsafe { sofuu_linux_recover(self.win) != 0 };
                let _ = unsafe { sofuu_linux_make_current(self.win) };
                if recreated {
                    let _ = renderer.rebuild_after_loss();
                }
            }
            Presenter::Vulkan(vk) => vk.recover_context(),
        }
    }
}

pub fn wake_main_loop() {
    unsafe { sofuu_linux_wake() };
}

/// Native handles used by the Vulkan surface factory (kept for diagnostics/tests).
pub fn native_surface(backend: &LinuxBackend) -> NativeSurface {
    NativeSurface {
        kind: backend.surface_kind,
        display: unsafe { sofuu_linux_display_handle(backend.win) },
        window: unsafe { sofuu_linux_window_handle(backend.win) },
    }
}
