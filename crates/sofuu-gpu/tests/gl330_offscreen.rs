//! Runtime checks for the shared GL 3.3 renderer using a macOS CGL offscreen context.
//!
//! `backend/gles.rs` and the committed GLSL 330 shaders are shared by Linux and Windows, but
//! those platforms cannot be executed on the development host. macOS ships an OpenGL 4.1 Core
//! driver, so this test creates a headless CGL context, compiles all four pipelines on a real
//! GPU driver and verifies geometry, clip windows and premultiplied blending by reading pixels
//! back from an `RGBA8` FBO.
//!
//! This is **not** the Metal-vs-GL pixel comparison required by the G2 gate; it validates the
//! GL path itself. The test skips (with a diagnostic) when CGL cannot provide a context, e.g.
//! in a headless CI sandbox.

#![cfg(target_os = "macos")]

use std::ffi::{c_void, CString};

use sofuu_gpu::backend::gles::{GlesRenderer, GlslFlavor};
use sofuu_ui::{
    rgba8, BackdropRegion, ClipRect, DrawCmd, DrawList, EffectQuadInstance, FrameUniforms,
    GlyphAtlas, ParticleInstance, QuadInstance,
};

type CGLPixelFormatObj = *mut c_void;
type CGLContextObj = *mut c_void;

/// Minimal `CGL` bindings resolved from the OpenGL framework at run time.
struct Cgl {
    handle: *mut c_void,
    choose_pixel_format: unsafe extern "C" fn(*const u32, *mut CGLPixelFormatObj, *mut i32) -> i32,
    create_context:
        unsafe extern "C" fn(CGLPixelFormatObj, CGLContextObj, *mut CGLContextObj) -> i32,
    set_current: unsafe extern "C" fn(CGLContextObj) -> i32,
    destroy_context: unsafe extern "C" fn(CGLContextObj) -> i32,
    destroy_pixel_format: unsafe extern "C" fn(CGLPixelFormatObj) -> i32,
}

impl Cgl {
    fn open() -> Option<Self> {
        let path = CString::new("/System/Library/Frameworks/OpenGL.framework/OpenGL").ok()?;
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return None;
        }
        // Every function pointer below is resolved from the same framework image, so the
        // transmute is between ABI-compatible representations of the same address.
        unsafe fn sym<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
            let name = CString::new(name).ok()?;
            let ptr = libc::dlsym(handle, name.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute_copy(&ptr))
            }
        }
        Some(Self {
            handle,
            choose_pixel_format: unsafe { sym(handle, "CGLChoosePixelFormat")? },
            create_context: unsafe { sym(handle, "CGLCreateContext")? },
            set_current: unsafe { sym(handle, "CGLSetCurrentContext")? },
            destroy_context: unsafe { sym(handle, "CGLDestroyContext")? },
            destroy_pixel_format: unsafe { sym(handle, "CGLDestroyPixelFormat")? },
        })
    }

    /// Create a Core Profile context and make it current on this thread.
    fn make_current(&self) -> Option<CGLContextObj> {
        // kCGLPFAOpenGLProfile = 99, kCGLOGLPVersion_3_2_Core = 0x3200,
        // kCGLPFAAllowOfflineRenderers = 96 (required on headless/VM sessions),
        // kCGLPFAColorSize = 8, kCGLPFAAlphaSize = 11, terminator = 0.
        let attrs: [u32; 8] = [99, 0x3200, 96, 8, 24, 11, 8, 0];
        let mut format: CGLPixelFormatObj = std::ptr::null_mut();
        let mut count = 0i32;
        let err = unsafe { (self.choose_pixel_format)(attrs.as_ptr(), &mut format, &mut count) };
        if err != 0 || format.is_null() {
            return None;
        }
        let mut ctx: CGLContextObj = std::ptr::null_mut();
        let err = unsafe { (self.create_context)(format, std::ptr::null_mut(), &mut ctx) };
        unsafe { (self.destroy_pixel_format)(format) };
        if err != 0 || ctx.is_null() {
            return None;
        }
        if unsafe { (self.set_current)(ctx) } != 0 {
            unsafe { (self.destroy_context)(ctx) };
            return None;
        }
        Some(ctx)
    }
}

impl Drop for Cgl {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

fn uniforms(atlas: &GlyphAtlas) -> FrameUniforms {
    FrameUniforms {
        viewport_size: [128.0, 128.0],
        atlas_size: [atlas.width as f32, atlas.height as f32],
        scale_factor: 2.0,
        time_sec: 0.0,
        _pad: [0.0; 2],
    }
}

fn quad(rect: [f32; 4], color: u32, clip_id: u32) -> QuadInstance {
    QuadInstance {
        rect,
        color_top: color,
        color_bottom: color,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id,
        ..Default::default()
    }
}

fn cmd(start: u32, count: u32) -> DrawCmd {
    DrawCmd { kind: 0, start, count, _pad: 0 }
}

/// `glReadPixels` returns rows bottom-up; expose top-left coordinates.
fn pixel(buf: &[u8], w: u32, h: u32, x: u32, y: u32) -> [u8; 4] {
    let i = (((h - 1 - y) * w + x) * 4) as usize;
    [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
}

fn assert_near(got: [u8; 4], want: [u8; 4], tol: i32, label: &str) {
    for i in 0..4 {
        let delta = got[i] as i32 - want[i] as i32;
        assert!(
            delta.abs() <= tol,
            "{label}: channel {i} got {} want {} (full pixel {got:?} vs {want:?})",
            got[i],
            want[i]
        );
    }
}

#[test]
fn gl330_offscreen_geometry_clips_and_blending() {
    // macOS implements OpenGL on top of Metal and compiles GLSL through the `MTLCompiler` XPC
    // service. Sandboxed/background sessions cannot reach it and the driver aborts the process
    // (not a catchable error), so this opt-in must be enabled from a normal GUI login session.
    if std::env::var_os("SOFUU_GPU_GL_OFFSCREEN").is_none() {
        eprintln!(
            "skip: set SOFUU_GPU_GL_OFFSCREEN=1 in a GUI login session to run the GL offscreen checks"
        );
        return;
    }
    let Some(cgl) = Cgl::open() else {
        eprintln!("skip: macOS OpenGL.framework is unavailable");
        return;
    };
    let Some(ctx) = cgl.make_current() else {
        eprintln!("skip: CGL cannot create an offscreen Core Profile context");
        return;
    };

    let atlas = GlyphAtlas::builtin();
    let loader = |name: &str| -> *const c_void {
        let Ok(name) = CString::new(name) else {
            return std::ptr::null();
        };
        unsafe { libc::dlsym(cgl.handle, name.as_ptr()) as *const c_void }
    };

    // Compiling all four programs here proves the committed GLSL 330 sources link on a real
    // driver (the shader-source unit tests only inspect text/bindings).
    let mut renderer = GlesRenderer::new(loader, &atlas, GlslFlavor::Desktop330)
        .expect("GL 3.3 renderer init (shader compile/link, VAOs, clip UBO, R8 atlas)");
    assert_eq!(renderer.flavor(), GlslFlavor::Desktop330);

    let (w, h) = (256u32, 256u32);
    let (fbo, tex) = renderer.create_offscreen_target(w, h).expect("offscreen FBO");
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    let u = uniforms(&atlas);

    // 1. Solid full-viewport quad at scale 2.0 -> every sampled interior pixel is the flat fill.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    dl.quads.push(quad([0.0, 0.0, 128.0, 128.0], rgba8(10, 20, 30, 255), 0));
    dl.cmds.push(cmd(0, 1));
    renderer.bind_offscreen_target(fbo);
    renderer.render(&u, &dl).expect("solid render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read solid");
    for (x, y) in [(128, 128), (10, 10), (245, 245)] {
        assert_near(pixel(&pixels, w, h, x, y), [10, 20, 30, 255], 1, "solid quad");
    }

    // 2. Root clip covers the left half: the right half must stay fully transparent.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    let left = dl.push_clip(ClipRect::new(0.0, 0.0, 64.0, 128.0));
    dl.quads.push(quad(
        [0.0, 0.0, 128.0, 128.0],
        rgba8(200, 40, 40, 255),
        left,
    ));
    dl.cmds.push(cmd(0, 1));
    renderer.render(&u, &dl).expect("clip render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read clip");
    assert_near(
        pixel(&pixels, w, h, 32 * 2, 64 * 2),
        [200, 40, 40, 255],
        1,
        "clipped left half",
    );
    assert_near(
        pixel(&pixels, w, h, 96 * 2, 64 * 2),
        [0, 0, 0, 0],
        1,
        "clipped right half",
    );

    // 3. 300 clips force the ordered clip-window split (> 256). Instance 255 belongs to the
    // left window, instance 299 to the second window and must be rebased to slot 43.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    for _ in 1..256 {
        dl.clips.push(ClipRect::new(0.0, 0.0, 64.0, 128.0));
    }
    for _ in 256..300 {
        dl.clips.push(ClipRect::new(64.0, 0.0, 64.0, 128.0));
    }
    assert_eq!(dl.clips.len(), 300);
    dl.quads.push(quad([0.0, 0.0, 128.0, 128.0], rgba8(220, 30, 30, 255), 255));
    dl.quads.push(quad([0.0, 0.0, 128.0, 128.0], rgba8(30, 30, 220, 255), 299));
    dl.cmds.push(cmd(0, 2));
    renderer.render(&u, &dl).expect("clip-window render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read clip windows");
    assert_near(
        pixel(&pixels, w, h, 32 * 2, 64 * 2),
        [220, 30, 30, 255],
        1,
        "300-clip first window",
    );
    assert_near(
        pixel(&pixels, w, h, 96 * 2, 64 * 2),
        [30, 30, 220, 255],
        1,
        "300-clip second window",
    );

    // 4. Premultiplied alpha: 50% red then 50% blue over a transparent clear yields
    // src + dst*(1-src.a) = (64, 0, 128, 192).
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    dl.quads.push(quad([0.0, 0.0, 128.0, 128.0], rgba8(255, 0, 0, 128), 0));
    dl.quads.push(quad([0.0, 0.0, 128.0, 128.0], rgba8(0, 0, 255, 128), 0));
    dl.cmds.push(cmd(0, 2));
    renderer.render(&u, &dl).expect("blend render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read blend");
    assert_near(
        pixel(&pixels, w, h, 128, 128),
        [64, 0, 128, 192],
        2,
        "premultiplied blend",
    );

    // 5. Effect pipeline: cursor glow centred in the rect renders its primary colour at full
    // alpha. This exercises the 80-byte effect instance layout and kind 5 dispatch.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    dl.effects.push(EffectQuadInstance {
        rect: [0.0, 0.0, 128.0, 128.0],
        radii: [0.0; 4],
        params0: [64.0, 64.0, 48.0, 1.0],
        params1: [0.0; 4],
        color0: rgba8(255, 128, 64, 255),
        color1: rgba8(0, 0, 0, 0),
        color2: 0,
        kind_and_clip: 3,
    });
    dl.cmds.push(DrawCmd { kind: 5, start: 0, count: 1, _pad: 0 });
    renderer.render(&u, &dl).expect("effect render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read effect");
    assert_near(
        pixel(&pixels, w, h, 128, 128),
        [255, 128, 64, 255],
        2,
        "cursor glow centre",
    );

    // 6. Particle pipeline: a circle at the same centre must keep its exact colour.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    dl.particles.push(ParticleInstance {
        pos: [64.0, 64.0],
        size: [30.0, 30.0],
        angle_rad: 0.0,
        color: rgba8(0, 200, 80, 255),
        shape_clip: 0,
        _pad: 0,
    });
    dl.cmds.push(DrawCmd { kind: 6, start: 0, count: 1, _pad: 0 });
    renderer.render(&u, &dl).expect("particle render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read particle");
    assert_near(
        pixel(&pixels, w, h, 128, 128),
        [0, 200, 80, 255],
        2,
        "particle circle centre",
    );

    // 7. Backdrop marker: after painting a red field, a frosted pane must sample the blurred
    // framebuffer (uniform red stays red) and leave the surrounding field untouched.
    let mut dl = DrawList::new();
    dl.reset([128.0, 128.0]);
    dl.quads.push(quad(
        [0.0, 0.0, 128.0, 128.0],
        rgba8(200, 40, 40, 255),
        0,
    ));
    dl.backdrops.push(BackdropRegion {
        rect: [32.0, 32.0, 64.0, 64.0],
        radii: [8.0; 4],
        tint: rgba8(0, 0, 0, 0),
        rim: rgba8(255, 255, 255, 0),
        params: [18.0, 1.0, 0.0, 1.0],
        clip_id: 0,
        _pad: 0,
    });
    dl.cmds.push(cmd(0, 1));
    dl.cmds.push(DrawCmd { kind: 7, start: 0, count: 1, _pad: 0 });
    renderer.render(&u, &dl).expect("backdrop render");
    renderer.read_offscreen(w, h, &mut pixels).expect("read backdrop");
    assert_near(
        pixel(&pixels, w, h, 128, 128),
        [200, 40, 40, 255],
        4,
        "backdrop blurred pane",
    );
    assert_near(
        pixel(&pixels, w, h, 16, 16),
        [200, 40, 40, 255],
        2,
        "backdrop outside pane",
    );

    renderer.destroy_offscreen_target(fbo, tex);
    drop(renderer);
    unsafe { (cgl.set_current)(std::ptr::null_mut()) };
    unsafe { (cgl.destroy_context)(ctx) };
}
