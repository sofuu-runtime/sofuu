//! Optional GPU offscreen capture (E2b).
//!
//! `sofuu-emu` never links `sofuu-gpu`; it loads `libsofuu_gpu` at runtime and
//! calls the stable `sofuu_gpu_offscreen_*` C ABI. When the library or a GPU
//! backend is unavailable the caller keeps the software rasterizer and reports
//! `renderer: "software"` honestly.

use std::ffi::{c_void, CString};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas};

type CreateFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type RenderFn = unsafe extern "C" fn(
    handle: *mut c_void,
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
) -> i64;

struct GpuLib {
    #[cfg(unix)]
    handle: *mut c_void,
    create: CreateFn,
    render: RenderFn,
    destroy: DestroyFn,
}

// The library handle and function pointers are process-wide and immutable.
unsafe impl Send for GpuLib {}

impl Drop for GpuLib {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

struct GpuRenderer {
    lib: GpuLib,
    handle: *mut c_void,
}

unsafe impl Send for GpuRenderer {}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.lib.destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

static GPU: OnceLock<Mutex<Option<GpuRenderer>>> = OnceLock::new();

fn lib_filename() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libsofuu_gpu.dylib"
    }
    #[cfg(target_os = "windows")]
    {
        "sofuu_gpu.dll"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "libsofuu_gpu.so"
    }
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var("SOFUU_GPU_LIB") {
        let p = p.trim();
        if !p.is_empty() {
            paths.push(PathBuf::from(p));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join(lib_filename()));
            paths.push(dir.join("lib").join(lib_filename()));
            if let Some(parent) = dir.parent() {
                paths.push(parent.join("release").join(lib_filename()));
                paths.push(parent.join("debug").join(lib_filename()));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join(lib_filename()));
        paths.push(cwd.join("target/release").join(lib_filename()));
        paths.push(cwd.join("target/debug").join(lib_filename()));
    }
    if let Ok(home) = std::env::var("HOME") {
        paths.push(PathBuf::from(home).join(".sofuu/lib").join(lib_filename()));
    }
    paths
}

#[cfg(unix)]
fn load_lib() -> Option<GpuLib> {
    for path in candidate_paths() {
        if !path.exists() {
            continue;
        }
        let Ok(cpath) = CString::new(path.to_string_lossy().as_bytes()) else {
            continue;
        };
        let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }
        unsafe fn sym<T: Copy>(handle: *mut c_void, name: &CString) -> Option<T> {
            let p = libc::dlsym(handle, name.as_ptr());
            if p.is_null() {
                None
            } else {
                Some(std::mem::transmute_copy(&p))
            }
        }
        let create = unsafe { sym::<CreateFn>(handle, &CString::new("sofuu_gpu_offscreen_create").ok()?) };
        let render = unsafe { sym::<RenderFn>(handle, &CString::new("sofuu_gpu_offscreen_render").ok()?) };
        let destroy = unsafe { sym::<DestroyFn>(handle, &CString::new("sofuu_gpu_offscreen_destroy").ok()?) };
        match (create, render, destroy) {
            (Some(create), Some(render), Some(destroy)) => {
                return Some(GpuLib {
                    handle,
                    create,
                    render,
                    destroy,
                });
            }
            _ => unsafe {
                libc::dlclose(handle);
            },
        }
    }
    None
}

#[cfg(not(unix))]
fn load_lib() -> Option<GpuLib> {
    None
}

fn renderer() -> Option<std::sync::MutexGuard<'static, Option<GpuRenderer>>> {
    let cell = GPU.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        let lib = load_lib()?;
        let handle = unsafe { (lib.create)() };
        if handle.is_null() {
            return None;
        }
        *guard = Some(GpuRenderer { lib, handle });
    }
    Some(guard)
}

/// Render `dl` with the GPU offscreen backend. Returns `(rgba, render_ms)` or
/// `None` when no GPU library/backend is available.
pub fn render(
    dl: &DrawList,
    atlas: &GlyphAtlas,
    logical: [f32; 2],
    scale: f32,
) -> Option<(Vec<u8>, f64)> {
    // `SOFUU_EMU_RENDERER=software` forces the deterministic CPU rasterizer
    // (golden tests, machines without a GPU). `metal`/`gpu` force the GPU path.
    if let Ok(mode) = std::env::var("SOFUU_EMU_RENDERER") {
        if mode.eq_ignore_ascii_case("software") || mode.eq_ignore_ascii_case("cpu") {
            return None;
        }
    }
    let width = (logical[0] * scale).round().max(1.0) as u32;
    let height = (logical[1] * scale).round().max(1.0) as u32;
    let uniforms = FrameUniforms {
        viewport_size: logical,
        atlas_size: [atlas.width as f32, atlas.height as f32],
        scale_factor: scale,
        time_sec: 0.0,
        _pad: [0.0; 2],
    };
    let mut guard = renderer()?;
    let r = guard.as_mut()?;
    let mut out = vec![0u8; width as usize * height as usize * 4];
    let start = std::time::Instant::now();
    let rc = unsafe {
        (r.lib.render)(
            r.handle,
            width,
            height,
            &uniforms,
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
        return None;
    }
    Some((out, start.elapsed().as_secs_f64() * 1000.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::draw::QuadInstance;

    #[test]
    fn gpu_matches_software_within_aa_tolerance() {
        let atlas = GlyphAtlas::builtin();
        let mut dl = DrawList::new();
        dl.reset([120.0, 80.0]);
        dl.push_quad(QuadInstance {
            rect: [0.0, 0.0, 120.0, 80.0],
            color_top: 0xFFFFFFFF,
            color_bottom: 0xFFFFFFFF,
            ..Default::default()
        });
        dl.push_quad(QuadInstance {
            rect: [8.0, 8.0, 104.0, 64.0],
            radii: [12.0; 4],
            color_top: 0x3366FFFF,
            color_bottom: 0x3366FFFF,
            ..Default::default()
        });
        let Some((gpu, _)) = render(&dl, &atlas, [120.0, 80.0], 2.0) else {
            return; // no GPU library on this machine — software fallback is the contract
        };
        let (sw, _) = crate::raster::rasterize(
            &dl,
            &atlas,
            [120.0, 80.0],
            2.0,
            [255, 255, 255, 255],
        );
        assert_eq!(gpu.len(), sw.pixels.len());
        let total: u64 = gpu
            .iter()
            .zip(sw.pixels.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();
        let mae = total as f64 / gpu.len() as f64;
        assert!(mae < 8.0, "GPU/software MAE {mae:.2} exceeds AA tolerance");
    }
}
