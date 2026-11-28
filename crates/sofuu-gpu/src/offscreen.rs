//! Headless GPU offscreen rendering (E2b).
//!
//! Renders a [`DrawList`] into an offscreen texture and reads it back as
//! RGBA8. No window, layer or drawable is created, so captures are identical
//! whether or not a live emulator window is open.
//!
//! macOS uses Metal (`MTLTexture` + shared-storage readback). Other platforms
//! return [`OFFSCREEN_UNSUPPORTED`] and the emulator keeps its software
//! rasterizer, reporting `renderer: "software"` honestly.

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas};

/// Stable error prefix for platforms without an offscreen GPU path.
pub const OFFSCREEN_UNSUPPORTED: &str = "E_UI_OFFSCREEN_UNSUPPORTED";

/// A reusable offscreen GPU renderer. Creation is expensive (shader compile);
/// keep one per capture session and render many frames.
pub struct OffscreenRenderer {
    #[cfg(target_os = "macos")]
    inner: macos::Renderer,
    #[cfg(not(target_os = "macos"))]
    _private: (),
}

impl OffscreenRenderer {
    /// Create a renderer that uploads `atlas` once. The atlas must match the
    /// one used to compile the `DrawList` (the built-in UI atlas by default).
    pub fn new(atlas: &GlyphAtlas) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                inner: macos::Renderer::new(atlas)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = atlas;
            Err(format!(
                "{OFFSCREEN_UNSUPPORTED}: no offscreen GPU backend on this platform"
            ))
        }
    }

    /// Backend name reported in capture metadata (`metal`, `egl`, `vulkan`).
    pub fn backend_name(&self) -> &'static str {
        #[cfg(target_os = "macos")]
        {
            "metal"
        }
        #[cfg(not(target_os = "macos"))]
        {
            "software"
        }
    }

    /// Render `dl` at `width × height` pixels and return tightly packed RGBA8.
    pub fn render(
        &mut self,
        uniforms: &FrameUniforms,
        dl: &DrawList,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        #[cfg(target_os = "macos")]
        {
            self.inner.render(uniforms, dl, width, height)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (uniforms, dl, width, height);
            Err(format!(
                "{OFFSCREEN_UNSUPPORTED}: no offscreen GPU backend on this platform"
            ))
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use std::ffi::c_void;
    use std::os::raw::c_char;

    type Handle = *mut c_void;

    extern "C" {
        fn sofuu_macos_create_offscreen(
            msl_source_utf8: *const c_char,
            atlas_pixels: *const u8,
            atlas_w: u32,
            atlas_h: u32,
        ) -> Handle;
        fn sofuu_macos_render_offscreen(
            handle: Handle,
            width: u32,
            height: u32,
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
            effects: *const c_void,
            effects_count: u32,
            particles: *const c_void,
            particles_count: u32,
            backdrops: *const c_void,
            backdrops_count: u32,
            cmds: *const c_void,
            cmds_count: u32,
            out_rgba: *mut u8,
            out_len: usize,
        ) -> i32;
    }

    pub struct Renderer {
        handle: Handle,
    }

    // The renderer is created and used on one thread; the raw handle is only
    // touched through `&mut self`.
    unsafe impl Send for Renderer {}

    impl Renderer {
        pub fn new(atlas: &GlyphAtlas) -> Result<Self, String> {
            let handle = unsafe {
                sofuu_macos_create_offscreen(
                    crate::backend::macos_metal::MSL_SOURCE.as_ptr() as *const c_char,
                    atlas.pixels.as_ptr(),
                    atlas.width,
                    atlas.height,
                )
            };
            if handle.is_null() {
                return Err(
                    "E_UI_NO_GPU: failed to create the Metal offscreen renderer".to_string(),
                );
            }
            Ok(Self { handle })
        }

        pub fn render(
            &mut self,
            uniforms: &FrameUniforms,
            dl: &DrawList,
            width: u32,
            height: u32,
        ) -> Result<Vec<u8>, String> {
            let mut out = vec![0u8; width as usize * height as usize * 4];
            let rc = unsafe {
                sofuu_macos_render_offscreen(
                    self.handle,
                    width,
                    height,
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
                    dl.effects.as_ptr().cast(),
                    dl.effects.len() as u32,
                    dl.particles.as_ptr().cast(),
                    dl.particles.len() as u32,
                    dl.backdrops.as_ptr().cast(),
                    dl.backdrops.len() as u32,
                    dl.cmds.as_ptr().cast(),
                    dl.cmds.len() as u32,
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            if rc != 0 {
                return Err(format!("E_UI_OFFSCREEN_RENDER: Metal returned {rc}"));
            }
            Ok(out)
        }
    }

    impl Drop for Renderer {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { crate::backend::macos_metal::destroy_raw(self.handle) };
                self.handle = std::ptr::null_mut();
            }
        }
    }
}

// ── C ABI for the headless emulator ─────────────────────────────────────
//
// `sofuu-emu` loads `libsofuu_gpu` dynamically (it never links this crate), so
// these symbols are the whole contract. The atlas is the built-in UI atlas,
// matching `UiEngine::new`.

/// Create a headless Metal renderer. Returns null when Metal is unavailable.
#[no_mangle]
pub extern "C" fn sofuu_gpu_offscreen_create() -> *mut std::ffi::c_void {
    let atlas = GlyphAtlas::builtin();
    match OffscreenRenderer::new(&atlas) {
        Ok(r) => Box::into_raw(Box::new(r)).cast(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Render one frame into `out_rgba` (`out_len` bytes). Returns `0` on success.
///
/// # Safety
/// `handle` must come from [`sofuu_gpu_offscreen_create`]; every array pointer
/// must be valid for its count; `out_rgba` must be writable for `out_len`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_gpu_offscreen_render(
    handle: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    uniforms: *const FrameUniforms,
    clips: *const std::ffi::c_void,
    clips_count: u32,
    quads: *const std::ffi::c_void,
    quads_count: u32,
    ribbons: *const std::ffi::c_void,
    ribbons_count: u32,
    strokes: *const std::ffi::c_void,
    strokes_count: u32,
    glyphs: *const std::ffi::c_void,
    glyphs_count: u32,
    effects: *const std::ffi::c_void,
    effects_count: u32,
    particles: *const std::ffi::c_void,
    particles_count: u32,
    backdrops: *const std::ffi::c_void,
    backdrops_count: u32,
    cmds: *const std::ffi::c_void,
    cmds_count: u32,
    out_rgba: *mut u8,
    out_len: usize,
) -> i64 {
    if handle.is_null() || uniforms.is_null() || out_rgba.is_null() {
        return -1;
    }
    let renderer = &mut *(handle as *mut OffscreenRenderer);
    let uniforms = &*uniforms;
    // Rebuild a DrawList view without copying: the arrays are `repr(C)` and
    // owned by the caller for the duration of this call.
    unsafe fn slice_or_empty<'a, T>(ptr: *const std::ffi::c_void, count: u32) -> &'a [T] {
        if count == 0 || ptr.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(ptr as *const T, count as usize)
        }
    }
    let mut dl = DrawList::new();
    dl.clips = slice_or_empty(clips, clips_count).to_vec();
    dl.quads = slice_or_empty(quads, quads_count).to_vec();
    dl.ribbons = slice_or_empty(ribbons, ribbons_count).to_vec();
    dl.strokes = slice_or_empty(strokes, strokes_count).to_vec();
    dl.glyphs = slice_or_empty(glyphs, glyphs_count).to_vec();
    dl.effects = slice_or_empty(effects, effects_count).to_vec();
    dl.particles = slice_or_empty(particles, particles_count).to_vec();
    dl.backdrops = slice_or_empty(backdrops, backdrops_count).to_vec();
    dl.cmds = slice_or_empty(cmds, cmds_count).to_vec();

    let out = std::slice::from_raw_parts_mut(out_rgba, out_len);
    match renderer.render(uniforms, &dl, width, height) {
        Ok(pixels) if pixels.len() == out.len() => {
            out.copy_from_slice(&pixels);
            0
        }
        Ok(_) => -2,
        Err(_) => -3,
    }
}

/// Destroy a renderer created by [`sofuu_gpu_offscreen_create`].
#[no_mangle]
pub unsafe extern "C" fn sofuu_gpu_offscreen_destroy(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut OffscreenRenderer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::draw::QuadInstance;

    #[test]
    fn offscreen_renders_rgba_without_a_window() {
        let atlas = GlyphAtlas::builtin();
        let mut renderer = match OffscreenRenderer::new(&atlas) {
            Ok(r) => r,
            Err(_) => return, // no Metal device (CI container) — software fallback covers it
        };
        let mut dl = DrawList::new();
        dl.reset([64.0, 64.0]);
        dl.push_quad(QuadInstance {
            rect: [0.0, 0.0, 64.0, 64.0],
            color_top: 0xFF0000FF, // packed RGBA8: opaque red
            color_bottom: 0xFF0000FF,
            ..Default::default()
        });
        let uniforms = FrameUniforms {
            viewport_size: [64.0, 64.0],
            atlas_size: [atlas.width as f32, atlas.height as f32],
            scale_factor: 1.0,
            time_sec: 0.0,
            _pad: [0.0; 2],
        };
        let px = renderer.render(&uniforms, &dl, 64, 64).expect("offscreen render");
        assert_eq!(px.len(), 64 * 64 * 4);
        let i = (32 * 64 + 32) * 4;
        assert!(px[i] > 200, "red channel = {}", px[i]);
        assert!(px[i + 1] < 60, "green channel = {}", px[i + 1]);
        assert_eq!(px[i + 3], 255, "alpha");
    }
}
