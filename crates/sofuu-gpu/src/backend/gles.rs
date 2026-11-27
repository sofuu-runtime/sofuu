#![allow(non_snake_case)]

//! Shared OpenGL 3.3 Core / OpenGL ES 3.0 renderer for Linux, Windows, Android and RPi 5.
//!
//! The renderer is deliberately API-thin: every GL entry point is resolved dynamically from
//! the platform window shim (`eglGetProcAddress` / `wglGetProcAddress` / `dlsym`), so the
//! shared library never hard-links a GPU driver. Four instanced pipelines consume the exact
//! `#[repr(C)]` `sofuu-ui` arrays; all geometry, SDF and alpha math mirrors
//! `shaders/msl/ui_shaders.metal` line-for-line.
//!
//! ## Layout contract
//! - QuadInstance 80 B: `vec4 rect`, `vec4 radii`, `uvec4 colors`, `vec4 params`, `uint clip`, 3 pad
//! - RibbonInstance 48 B: `vec4 (x0,y0t,y0b,x1)`, `vec2 (y1t,y1b)`, `uvec4 (ct,cb,clip,pad)`
//! - StrokeInstance 48 B: `vec4 (p0,p1)`, `vec4 (width,feather,dash,gap)`, `uvec4 (color,clip,pad,pad)`
//! - GlyphInstance 48 B: `vec4 (pos,size)`, `vec4 uv`, `uint color`, `float px_range`, `uint clip`
//!
//! Clip rectangles (16 B each) are uploaded into a std140 `ClipsBlock` UBO of 256 `vec4`s (4 KiB,
//! well inside the guaranteed 16 KiB GLES 3.0 minimum). Scenes with more than 256 clips are split
//! into ordered clip windows automatically without changing draw/blend order.

use std::ffi::c_void;

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas};
#[cfg(test)]
use sofuu_ui::{QuadInstance, RibbonInstance, StrokeInstance};

/// Maximum clip rectangles per std140 UBO upload. Oversized scenes are split into windows.
pub const MAX_CLIPS: usize = 256;

const QUAD_STRIDE: usize = 80;
const RIBBON_STRIDE: usize = 48;
const STROKE_STRIDE: usize = 48;
const GLYPH_STRIDE: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslFlavor {
    Desktop330,
    Es300,
}

pub type GLenum = u32;
pub type GLuint = u32;
pub type GLint = i32;
pub type GLsizei = i32;
pub type GLboolean = u8;
pub type GLbitfield = u32;
pub type GLchar = libc::c_char;
pub type GLsizeiptr = isize;
pub type GLintptr = isize;
pub type GLfloat = f32;

const GL_FRAGMENT_SHADER: GLenum = 0x8B30;
const GL_VERTEX_SHADER: GLenum = 0x8B31;
const GL_COMPILE_STATUS: GLenum = 0x8B81;
const GL_LINK_STATUS: GLenum = 0x8B82;
const GL_INFO_LOG_LENGTH: GLenum = 0x8B84;
const GL_ARRAY_BUFFER: GLenum = 0x8892;
const GL_UNIFORM_BUFFER: GLenum = 0x8A11;
const GL_STREAM_DRAW: GLenum = 0x88E0;
const GL_STATIC_DRAW: GLenum = 0x88E4;
const GL_TRIANGLE_STRIP: GLenum = 0x0005;
const GL_FLOAT: GLenum = 0x1406;
const GL_UNSIGNED_INT: GLenum = 0x1405;
const GL_UNSIGNED_BYTE: GLenum = 0x1401;
const GL_UNPACK_ALIGNMENT: GLenum = 0x0CF5;
const GL_TEXTURE_2D: GLenum = 0x0DE1;
const GL_TEXTURE0: GLenum = 0x84C0;
const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
const GL_TEXTURE_WRAP_S: GLenum = 0x2802;
const GL_TEXTURE_WRAP_T: GLenum = 0x2803;
const GL_LINEAR: GLenum = 0x2601;
const GL_CLAMP_TO_EDGE: GLenum = 0x812F;
const GL_R8: GLenum = 0x8229;
const GL_RED: GLenum = 0x1903;
const GL_RGBA8: GLenum = 0x8058;
const GL_RGBA: GLenum = 0x1908;
const GL_COLOR_BUFFER_BIT: GLbitfield = 0x0000_4000;
const GL_BLEND: GLenum = 0x0BE2;
const GL_ONE: GLenum = 1;
const GL_ONE_MINUS_SRC_ALPHA: GLenum = 0x0303;
const GL_VERSION: GLenum = 0x1F02;
const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;
const GL_NO_ERROR: GLenum = 0;

macro_rules! gl_api {
    ($( fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty; )*) => {
        /// Dynamically resolved GL entry points.
        #[allow(dead_code)]
        #[derive(Clone, Copy)]
        pub struct GlApi {
            $(pub $name: Option<unsafe extern "C" fn($($ty),*) -> $ret>,)*
        }

        impl GlApi {
            /// Resolve every entry point through the platform loader. Missing symbols are left
            /// as `None`; [`GlesRenderer::new`] fails with a precise name when a required one is
            /// absent.
            ///
            /// # Safety
            ///
            /// `loader` must return entry points for the currently current GL/GLES context; the
            /// resulting function pointers are only valid while that context exists.
            pub unsafe fn load<F>(loader: F) -> Self
            where
                F: Fn(&str) -> *const c_void,
            {
                $(
                    let $name = {
                        let p = loader(stringify!($name));
                        if p.is_null() {
                            None
                        } else {
                            Some(std::mem::transmute::<
                                *const c_void,
                                unsafe extern "C" fn($($ty),*) -> $ret,
                            >(p))
                        }
                    };
                )*
                Self { $($name,)* }
            }
        }
    };
}

gl_api! {
    fn glGetString(name: GLenum) -> *const u8;
    fn glGetError() -> GLenum;
    fn glViewport(x: GLint, y: GLint, w: GLsizei, h: GLsizei) -> ();
    fn glClearColor(r: GLfloat, g: GLfloat, b: GLfloat, a: GLfloat) -> ();
    fn glClear(mask: GLbitfield) -> ();
    fn glEnable(cap: GLenum) -> ();
    fn glDisable(cap: GLenum) -> ();
    fn glBlendFuncSeparate(srgb: GLenum, drgb: GLenum, sa: GLenum, da: GLenum) -> ();
    fn glGenBuffers(n: GLsizei, out: *mut GLuint) -> ();
    fn glBindBuffer(target: GLenum, buf: GLuint) -> ();
    fn glBufferData(target: GLenum, size: GLsizeiptr, data: *const c_void, usage: GLenum) -> ();
    fn glBufferSubData(target: GLenum, offset: GLintptr, size: GLsizeiptr, data: *const c_void) -> ();
    fn glDeleteBuffers(n: GLsizei, bufs: *const GLuint) -> ();
    fn glGenVertexArrays(n: GLsizei, out: *mut GLuint) -> ();
    fn glBindVertexArray(vao: GLuint) -> ();
    fn glDeleteVertexArrays(n: GLsizei, vaos: *const GLuint) -> ();
    fn glEnableVertexAttribArray(index: GLuint) -> ();
    fn glVertexAttribPointer(index: GLuint, size: GLint, ty: GLenum, normalized: GLboolean, stride: GLsizei, ptr: *const c_void) -> ();
    fn glVertexAttribIPointer(index: GLuint, size: GLint, ty: GLenum, stride: GLsizei, ptr: *const c_void) -> ();
    fn glVertexAttribDivisor(index: GLuint, divisor: GLuint) -> ();
    fn glCreateShader(stage: GLenum) -> GLuint;
    fn glShaderSource(shader: GLuint, count: GLsizei, src: *const *const GLchar, len: *const GLint) -> ();
    fn glCompileShader(shader: GLuint) -> ();
    fn glGetShaderiv(shader: GLuint, pname: GLenum, out: *mut GLint) -> ();
    fn glGetShaderInfoLog(shader: GLuint, cap: GLsizei, len: *mut GLsizei, out: *mut GLchar) -> ();
    fn glDeleteShader(shader: GLuint) -> ();
    fn glCreateProgram() -> GLuint;
    fn glAttachShader(program: GLuint, shader: GLuint) -> ();
    fn glLinkProgram(program: GLuint) -> ();
    fn glGetProgramiv(program: GLuint, pname: GLenum, out: *mut GLint) -> ();
    fn glGetProgramInfoLog(program: GLuint, cap: GLsizei, len: *mut GLsizei, out: *mut GLchar) -> ();
    fn glUseProgram(program: GLuint) -> ();
    fn glDeleteProgram(program: GLuint) -> ();
    fn glGetUniformLocation(program: GLuint, name: *const GLchar) -> GLint;
    fn glUniform1i(loc: GLint, v: GLint) -> ();
    fn glUniform1f(loc: GLint, v: GLfloat) -> ();
    fn glUniform2f(loc: GLint, x: GLfloat, y: GLfloat) -> ();
    fn glGetUniformBlockIndex(program: GLuint, name: *const GLchar) -> GLuint;
    fn glUniformBlockBinding(program: GLuint, block_index: GLuint, binding: GLuint) -> ();
    fn glBindBufferBase(target: GLenum, index: GLuint, buf: GLuint) -> ();
    fn glGenTextures(n: GLsizei, out: *mut GLuint) -> ();
    fn glBindTexture(target: GLenum, tex: GLuint) -> ();
    fn glTexImage2D(target: GLenum, level: GLint, internal: GLint, w: GLsizei, h: GLsizei, border: GLint, format: GLenum, ty: GLenum, data: *const c_void) -> ();
    fn glTexParameteri(target: GLenum, pname: GLenum, value: GLint) -> ();
    fn glActiveTexture(unit: GLenum) -> ();
    fn glPixelStorei(pname: GLenum, value: GLint) -> ();
    fn glDeleteTextures(n: GLsizei, tex: *const GLuint) -> ();
    fn glDrawArraysInstanced(mode: GLenum, first: GLint, count: GLsizei, instances: GLsizei) -> ();
    fn glGenFramebuffers(n: GLsizei, out: *mut GLuint) -> ();
    fn glBindFramebuffer(target: GLenum, fb: GLuint) -> ();
    fn glFramebufferTexture2D(target: GLenum, attachment: GLenum, textarget: GLenum, texture: GLuint, level: GLint) -> ();
    fn glCheckFramebufferStatus(target: GLenum) -> GLenum;
    fn glDeleteFramebuffers(n: GLsizei, fbs: *const GLuint) -> ();
    fn glReadPixels(x: GLint, y: GLint, w: GLsizei, h: GLsizei, format: GLenum, ty: GLenum, out: *mut c_void) -> ();
}

impl GlApi {
    fn req<T>(v: Option<T>, name: &str) -> Result<T, String> {
        v.ok_or_else(|| format!("E_UI_GL_MISSING: required GL entry point `{name}` is unavailable"))
    }
}

#[derive(Clone, Copy)]
struct Program {
    id: GLuint,
    loc_viewport: GLint,
    loc_atlas_size: GLint,
    loc_scale: GLint,
    loc_time: GLint,
}

struct StreamingBuffer {
    id: GLuint,
    capacity: usize,
}

impl StreamingBuffer {
    fn new() -> Self {
        Self {
            id: 0,
            capacity: 0,
        }
    }
}

/// Shared GL/GLES renderer. Construction and every draw happen on the thread that owns the
/// current EGL/WGL context.
pub struct GlesRenderer {
    api: GlApi,
    flavor: GlslFlavor,
    programs: [Program; 4],
    vaos: [GLuint; 4],
    corner_vbo: GLuint,
    instance_vbos: [StreamingBuffer; 4],
    clip_ubo: GLuint,
    atlas_tex: GLuint,
    atlas_w: u32,
    atlas_h: u32,
    atlas_px_range: f32,
    atlas_pixels: Vec<u8>,
    initialized: bool,
}

impl GlesRenderer {
    /// Build all GL resources. `loader` is the platform shim's `get_proc_address` function.
    pub fn new<F>(loader: F, atlas: &GlyphAtlas, flavor: GlslFlavor) -> Result<Self, String>
    where
        F: Fn(&str) -> *const c_void,
    {
        let api = unsafe { GlApi::load(loader) };
        let mut renderer = Self {
            api,
            flavor,
            programs: [
                Program {
                    id: 0,
                    loc_viewport: -1,
                    loc_atlas_size: -1,
                    loc_scale: -1,
                    loc_time: -1,
                };
                4
            ],
            vaos: [0; 4],
            corner_vbo: 0,
            instance_vbos: [
                StreamingBuffer::new(),
                StreamingBuffer::new(),
                StreamingBuffer::new(),
                StreamingBuffer::new(),
            ],
            clip_ubo: 0,
            atlas_tex: 0,
            atlas_w: atlas.width,
            atlas_h: atlas.height,
            atlas_px_range: atlas.px_range,
            atlas_pixels: atlas.pixels.clone(),
            initialized: false,
        };
        renderer.init_resources()?;
        Ok(renderer)
    }

    pub fn flavor(&self) -> GlslFlavor {
        self.flavor
    }

    pub fn initialize_after_context_loss(&mut self) -> Result<(), String> {
        self.init_resources()
    }

    fn init_resources(&mut self) -> Result<(), String> {
        let api = self.api;
        unsafe {
            let version = GlApi::req(api.glGetString, "glGetString")?(GL_VERSION);
            if version.is_null() {
                return Err("E_UI_GL_CONTEXT: glGetString(GL_VERSION) returned null".to_string());
            }
            let version = std::ffi::CStr::from_ptr(version as *const libc::c_char)
                .to_string_lossy()
                .into_owned();
            if self.flavor == GlslFlavor::Desktop330 && version.contains("OpenGL ES") {
                return Err(format!(
                    "E_UI_GL_CONTEXT: requested desktop GL 3.3 but got `{version}`"
                ));
            }
            if self.flavor == GlslFlavor::Es300 && !version.contains("OpenGL ES") {
                return Err(format!(
                    "E_UI_GL_CONTEXT: requested GLES 3.0 but got `{version}`"
                ));
            }

            for (i, name) in ["QUAD", "RIBBON", "STROKE", "GLYPH"].iter().enumerate() {
                self.programs[i] = self.make_program(name)?;
            }

            let mut corner = [0.0f32; 8];
            corner[2] = 1.0;
            corner[5] = 1.0;
            corner[6] = 1.0;
            corner[7] = 1.0;
            let mut vbo = 0u32;
            GlApi::req(api.glGenBuffers, "glGenBuffers")?(1, &mut vbo);
            self.corner_vbo = vbo;
            GlApi::req(api.glBindBuffer, "glBindBuffer")?(GL_ARRAY_BUFFER, vbo);
            GlApi::req(api.glBufferData, "glBufferData")?(
                GL_ARRAY_BUFFER,
                std::mem::size_of_val(&corner) as isize,
                corner.as_ptr().cast(),
                GL_STATIC_DRAW,
            );

            for i in 0..4 {
                let mut ids = [0u32; 1];
                GlApi::req(api.glGenBuffers, "glGenBuffers")?(1, ids.as_mut_ptr());
                self.instance_vbos[i].id = ids[0];
                self.instance_vbos[i].capacity = 0;
            }

            for (i, program) in self.programs.iter().enumerate() {
                let mut vao = 0u32;
                GlApi::req(api.glGenVertexArrays, "glGenVertexArrays")?(1, &mut vao);
                self.vaos[i] = vao;
                GlApi::req(api.glBindVertexArray, "glBindVertexArray")?(vao);
                GlApi::req(api.glBindBuffer, "glBindBuffer")?(GL_ARRAY_BUFFER, self.corner_vbo);
                GlApi::req(api.glEnableVertexAttribArray, "glEnableVertexAttribArray")?(0);
                GlApi::req(api.glVertexAttribPointer, "glVertexAttribPointer")?(
                    0,
                    2,
                    GL_FLOAT,
                    0,
                    8,
                    std::ptr::null(),
                );
                GlApi::req(api.glBindBuffer, "glBindBuffer")?(
                    GL_ARRAY_BUFFER,
                    self.instance_vbos[i].id,
                );
                self.configure_attributes(i, 0)?;
                let _ = program;
            }

            let mut clips = 0u32;
            GlApi::req(api.glGenBuffers, "glGenBuffers")?(1, &mut clips);
            self.clip_ubo = clips;
            GlApi::req(api.glBindBuffer, "glBindBuffer")?(GL_UNIFORM_BUFFER, clips);
            GlApi::req(api.glBufferData, "glBufferData")?(
                GL_UNIFORM_BUFFER,
                (MAX_CLIPS * 16) as isize,
                std::ptr::null(),
                GL_STREAM_DRAW,
            );
            GlApi::req(api.glBindBufferBase, "glBindBufferBase")?(GL_UNIFORM_BUFFER, 0, clips);

            let mut tex = 0u32;
            GlApi::req(api.glGenTextures, "glGenTextures")?(1, &mut tex);
            self.atlas_tex = tex;
            GlApi::req(api.glActiveTexture, "glActiveTexture")?(GL_TEXTURE0);
            GlApi::req(api.glBindTexture, "glBindTexture")?(GL_TEXTURE_2D, tex);
            GlApi::req(api.glPixelStorei, "glPixelStorei")?(GL_UNPACK_ALIGNMENT, 1);
            GlApi::req(api.glTexImage2D, "glTexImage2D")?(
                GL_TEXTURE_2D,
                0,
                GL_R8 as GLint,
                self.atlas_w as GLsizei,
                self.atlas_h as GLsizei,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                self.atlas_pixels.as_ptr().cast(),
            );
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as GLint);
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as GLint);

            GlApi::req(api.glDisable, "glDisable")?(0x0B71); // GL_DEPTH_TEST
            GlApi::req(api.glEnable, "glEnable")?(GL_BLEND);
            // Shaders output premultiplied alpha (rgba = rgb*alpha, alpha), exactly like Metal.
            GlApi::req(api.glBlendFuncSeparate, "glBlendFuncSeparate")?(
                GL_ONE,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_ONE,
                GL_ONE_MINUS_SRC_ALPHA,
            );
            GlApi::req(api.glBindVertexArray, "glBindVertexArray")?(0);
            let err = api.glGetError.map(|f| f()).unwrap_or(0);
            if err != GL_NO_ERROR {
                return Err(format!("E_UI_GL_INIT: OpenGL error 0x{err:04x} during resource setup"));
            }
        }
        self.initialized = true;
        Ok(())
    }

    fn make_program(&self, pipeline: &str) -> Result<Program, String> {
        let api = &self.api;
        let (vert_src, frag_src) = shader_sources(self.flavor, pipeline);
        unsafe {
            let vs = compile_shader(api, GL_VERTEX_SHADER, &vert_src, pipeline, "vertex")?;
            let fs = match compile_shader(api, GL_FRAGMENT_SHADER, &frag_src, pipeline, "fragment") {
                Ok(s) => s,
                Err(e) => {
                    let _ = api.glDeleteShader.map(|f| f(vs));
                    return Err(e);
                }
            };
            let program = GlApi::req(api.glCreateProgram, "glCreateProgram")?();
            GlApi::req(api.glAttachShader, "glAttachShader")?(program, vs);
            GlApi::req(api.glAttachShader, "glAttachShader")?(program, fs);
            GlApi::req(api.glLinkProgram, "glLinkProgram")?(program);
            GlApi::req(api.glDeleteShader, "glDeleteShader")?(vs);
            GlApi::req(api.glDeleteShader, "glDeleteShader")?(fs);

            let mut ok = 0;
            GlApi::req(api.glGetProgramiv, "glGetProgramiv")?(program, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                let log = program_log(api, program);
                let _ = api.glDeleteProgram.map(|f| f(program));
                return Err(format!(
                    "E_UI_GL_LINK: {pipeline} program link failed: {log}"
                ));
            }

            let name = std::ffi::CString::new("ClipsBlock").unwrap();
            let block = api
                .glGetUniformBlockIndex
                .map(|f| f(program, name.as_ptr()))
                .unwrap_or(u32::MAX);
            if block != u32::MAX {
                if let Some(f) = api.glUniformBlockBinding {
                    f(program, block, 0);
                }
            }

            let loc = |n: &str| -> GLint {
                let c = std::ffi::CString::new(n).unwrap();
                api.glGetUniformLocation
                    .map(|f| f(program, c.as_ptr()))
                    .unwrap_or(-1)
            };
            let p = Program {
                id: program,
                loc_viewport: loc("u_viewport_size"),
                loc_atlas_size: loc("u_atlas_size"),
                loc_scale: loc("u_scale_factor"),
                loc_time: loc("u_time_sec"),
            };
            if let Some(f) = api.glUseProgram {
                f(program);
            }
            if let Some(f) = api.glUniform1i {
                f(loc("u_atlas"), 0);
            }
            Ok(p)
        }
    }

    unsafe fn configure_attributes(&self, kind: usize, offset: usize) -> Result<(), String> {
        let api = self.api;
        let b = offset as *const c_void;
        let stride = match kind {
            0 => QUAD_STRIDE,
            1 => RIBBON_STRIDE,
            2 => STROKE_STRIDE,
            _ => GLYPH_STRIDE,
        } as GLsizei;
        let ptr = |o: usize| (offset + o) as *const c_void;
        let en = GlApi::req(api.glEnableVertexAttribArray, "glEnableVertexAttribArray")?;
        let f = GlApi::req(api.glVertexAttribPointer, "glVertexAttribPointer")?;
        let i = GlApi::req(api.glVertexAttribIPointer, "glVertexAttribIPointer")?;
        let d = GlApi::req(api.glVertexAttribDivisor, "glVertexAttribDivisor")?;
        let _ = b;
        match kind {
            0 => {
                en(1); f(1, 4, GL_FLOAT, 0, stride, ptr(0));
                en(2); f(2, 4, GL_FLOAT, 0, stride, ptr(16));
                en(3); i(3, 4, GL_UNSIGNED_INT, stride, ptr(32));
                en(4); f(4, 4, GL_FLOAT, 0, stride, ptr(48));
                en(5); i(5, 1, GL_UNSIGNED_INT, stride, ptr(64));
                d(1, 1); d(2, 1); d(3, 1); d(4, 1); d(5, 1);
            }
            1 => {
                en(1); f(1, 4, GL_FLOAT, 0, stride, ptr(0));
                en(2); f(2, 2, GL_FLOAT, 0, stride, ptr(16));
                en(3); i(3, 4, GL_UNSIGNED_INT, stride, ptr(24));
                d(1, 1); d(2, 1); d(3, 1);
            }
            2 => {
                en(1); f(1, 4, GL_FLOAT, 0, stride, ptr(0));
                en(2); f(2, 4, GL_FLOAT, 0, stride, ptr(16));
                en(3); i(3, 4, GL_UNSIGNED_INT, stride, ptr(32));
                d(1, 1); d(2, 1); d(3, 1);
            }
            _ => {
                en(1); f(1, 4, GL_FLOAT, 0, stride, ptr(0));
                en(2); f(2, 4, GL_FLOAT, 0, stride, ptr(16));
                en(3); i(3, 1, GL_UNSIGNED_INT, stride, ptr(32));
                en(4); f(4, 1, GL_FLOAT, 0, stride, ptr(36));
                en(5); i(5, 1, GL_UNSIGNED_INT, stride, ptr(40));
                d(1, 1); d(2, 1); d(3, 1); d(4, 1); d(5, 1);
            }
        }
        Ok(())
    }

    unsafe fn upload_streaming(
        &mut self,
        kind: usize,
        bytes: &[u8],
    ) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        let api = self.api;
        let buf = &mut self.instance_vbos[kind];
        if buf.id == 0 {
            return Err(format!("E_UI_GL_BUFFER: instance buffer {kind} is not allocated"));
        }
        let target = GL_ARRAY_BUFFER;
        GlApi::req(api.glBindBuffer, "glBindBuffer")?(target, buf.id);
        let needed = bytes.len().max(4096);
        if needed > buf.capacity {
            let mut cap = 4096usize;
            while cap < needed {
                cap = cap.saturating_mul(2);
            }
            GlApi::req(api.glBufferData, "glBufferData")?(
                target,
                cap as isize,
                std::ptr::null(),
                GL_STREAM_DRAW,
            );
            buf.capacity = cap;
        } else {
            // Orphan the previous storage so queued draws keep their snapshot.
            GlApi::req(api.glBufferData, "glBufferData")?(
                target,
                buf.capacity as isize,
                std::ptr::null(),
                GL_STREAM_DRAW,
            );
        }
        GlApi::req(api.glBufferSubData, "glBufferSubData")?(
            target,
            0,
            bytes.len() as isize,
            bytes.as_ptr().cast(),
        );
        Ok(())
    }

    unsafe fn upload_clips(&mut self, clips: &[sofuu_ui::ClipRect], base: usize, count: usize) -> Result<(), String> {
        let api = self.api;
        let mut raw = vec![[0.0f32; 4]; count];
        for (i, c) in clips.iter().skip(base).take(count).enumerate() {
            raw[i] = [c.x, c.y, c.w, c.h];
        }
        GlApi::req(api.glBindBuffer, "glBindBuffer")?(GL_UNIFORM_BUFFER, self.clip_ubo);
        GlApi::req(api.glBufferData, "glBufferData")?(
            GL_UNIFORM_BUFFER,
            (MAX_CLIPS * 16) as isize,
            std::ptr::null(),
            GL_STREAM_DRAW,
        );
        if !raw.is_empty() {
            GlApi::req(api.glBufferSubData, "glBufferSubData")?(
                GL_UNIFORM_BUFFER,
                0,
                (raw.len() * 16) as isize,
                raw.as_ptr().cast(),
            );
        }
        Ok(())
    }

    /// Render one frame into the currently bound framebuffer and present it.
    ///
    /// The caller is responsible for binding the native default framebuffer (or an offscreen FBO
    /// in parity tests) and for executing the platform swap after this returns.
    pub fn render(&mut self, uniforms: &FrameUniforms, dl: &DrawList) -> Result<(), String> {
        if !self.initialized {
            return Err("E_UI_GL_INIT: renderer is not initialized".to_string());
        }
        let api = self.api;
        let scale = uniforms.scale_factor.max(1.0);
        let fb_w = (uniforms.viewport_size[0] * scale).round().max(1.0) as GLsizei;
        let fb_h = (uniforms.viewport_size[1] * scale).round().max(1.0) as GLsizei;
        unsafe {
            GlApi::req(api.glViewport, "glViewport")?(0, 0, fb_w, fb_h);
            GlApi::req(api.glClearColor, "glClearColor")?(0.0, 0.0, 0.0, 0.0);
            GlApi::req(api.glClear, "glClear")?(GL_COLOR_BUFFER_BIT);
            if dl.cmds.is_empty() {
                return Ok(());
            }

            unsafe fn bytes_of<T>(v: &[T]) -> &[u8] {
                std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v))
            }
            let qbytes = bytes_of(&dl.quads);
            let rbytes = bytes_of(&dl.ribbons);
            let sbytes = bytes_of(&dl.strokes);
            let gbytes = bytes_of(&dl.glyphs);
            self.upload_streaming(0, qbytes)?;
            self.upload_streaming(1, rbytes)?;
            self.upload_streaming(2, sbytes)?;
            self.upload_streaming(3, gbytes)?;

            GlApi::req(api.glActiveTexture, "glActiveTexture")?(GL_TEXTURE0);
            GlApi::req(api.glBindTexture, "glBindTexture")?(GL_TEXTURE_2D, self.atlas_tex);
            GlApi::req(api.glEnable, "glEnable")?(GL_BLEND);
            GlApi::req(api.glBlendFuncSeparate, "glBlendFuncSeparate")?(
                GL_ONE,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_ONE,
                GL_ONE_MINUS_SRC_ALPHA,
            );

            for (base, count) in clip_windows(dl.clips.len()) {
                self.upload_clips(&dl.clips, base, count)?;
                GlApi::req(api.glBindBufferBase, "glBindBufferBase")?(GL_UNIFORM_BUFFER, 0, self.clip_ubo);

                for cmd in &dl.cmds {
                    if cmd.count == 0 {
                        continue;
                    }
                    let kind = cmd.kind as usize;
                    if kind > 3 {
                        continue;
                    }
                    let stride = [QUAD_STRIDE, RIBBON_STRIDE, STROKE_STRIDE, GLYPH_STRIDE][kind];
                    let (array_len, start, count_i) =
                        (cmd.start as usize, cmd.start as usize, cmd.count as usize);
                    let end = start.saturating_add(count_i);
                    let max_len = match kind {
                        0 => dl.quads.len(),
                        1 => dl.ribbons.len(),
                        2 => dl.strokes.len(),
                        _ => dl.glyphs.len(),
                    };
                    if start >= max_len || end > max_len {
                        continue;
                    }

                    let full_pass = dl.clips.len() <= MAX_CLIPS;
                    let filtered: Option<Vec<u8>> = if full_pass {
                        None
                    } else {
                        // Rebase clip ids into the UBO window: instance `clip_id` becomes
                        // `clip_id - base`, matching the `count` clips uploaded at slot 0.
                        let remap = |clip: u32| clip_in_window(clip, base, count);
                        let mut out: Vec<u8> = Vec::new();
                        match kind {
                            0 => {
                                for inst in &dl.quads[start..end] {
                                    if let Some(id) = remap(inst.clip_id) {
                                        let mut copy = *inst;
                                        copy.clip_id = id;
                                        out.extend_from_slice(bytes_of(std::slice::from_ref(&copy)));
                                    }
                                }
                            }
                            1 => {
                                for inst in &dl.ribbons[start..end] {
                                    if let Some(id) = remap(inst.clip_id) {
                                        let mut copy = *inst;
                                        copy.clip_id = id;
                                        out.extend_from_slice(bytes_of(std::slice::from_ref(&copy)));
                                    }
                                }
                            }
                            2 => {
                                for inst in &dl.strokes[start..end] {
                                    if let Some(id) = remap(inst.clip_id) {
                                        let mut copy = *inst;
                                        copy.clip_id = id;
                                        out.extend_from_slice(bytes_of(std::slice::from_ref(&copy)));
                                    }
                                }
                            }
                            _ => {
                                for inst in &dl.glyphs[start..end] {
                                    if let Some(id) = remap(inst.clip_id) {
                                        let mut copy = *inst;
                                        copy.clip_id = id;
                                        out.extend_from_slice(bytes_of(std::slice::from_ref(&copy)));
                                    }
                                }
                            }
                        }
                        if out.is_empty() {
                            continue;
                        }
                        Some(out)
                    };

                    if let Some(ref filtered) = filtered {
                        self.upload_streaming(kind, filtered)?;
                    }

                    let program = &self.programs[kind];
                    GlApi::req(api.glUseProgram, "glUseProgram")?(program.id);
                    if program.loc_viewport >= 0 {
                        if let Some(f) = api.glUniform2f {
                            f(program.loc_viewport, uniforms.viewport_size[0], uniforms.viewport_size[1]);
                        }
                    }
                    if program.loc_atlas_size >= 0 {
                        if let Some(f) = api.glUniform2f {
                            f(program.loc_atlas_size, uniforms.atlas_size[0], uniforms.atlas_size[1]);
                        }
                    }
                    if program.loc_scale >= 0 {
                        if let Some(f) = api.glUniform1f {
                            f(program.loc_scale, uniforms.scale_factor);
                        }
                    }
                    if program.loc_time >= 0 {
                        if let Some(f) = api.glUniform1f {
                            f(program.loc_time, uniforms.time_sec);
                        }
                    }

                    GlApi::req(api.glBindVertexArray, "glBindVertexArray")?(self.vaos[kind]);
                    GlApi::req(api.glBindBuffer, "glBindBuffer")?(
                        GL_ARRAY_BUFFER,
                        self.instance_vbos[kind].id,
                    );
                    let (offset, draw_count) = match &filtered {
                        Some(filtered) => (0, (filtered.len() / stride) as GLsizei),
                        None => (array_len * stride, count_i as GLsizei),
                    };
                    self.configure_attributes(kind, offset)?;
                    if draw_count > 0 {
                        GlApi::req(api.glDrawArraysInstanced, "glDrawArraysInstanced")?(
                            GL_TRIANGLE_STRIP,
                            0,
                            4,
                            draw_count,
                        );
                    }
                }
            }

            GlApi::req(api.glBindVertexArray, "glBindVertexArray")?(0);
            GlApi::req(api.glUseProgram, "glUseProgram")?(0);
        }
        Ok(())
    }

    /// Upload a new glyph atlas without rebuilding pipelines (runtime font swap / surface loss).
    pub fn update_atlas(&mut self, atlas: &GlyphAtlas) -> Result<(), String> {
        self.atlas_w = atlas.width;
        self.atlas_h = atlas.height;
        self.atlas_px_range = atlas.px_range;
        self.atlas_pixels = atlas.pixels.clone();
        let api = self.api;
        unsafe {
            GlApi::req(api.glActiveTexture, "glActiveTexture")?(GL_TEXTURE0);
            GlApi::req(api.glBindTexture, "glBindTexture")?(GL_TEXTURE_2D, self.atlas_tex);
            GlApi::req(api.glPixelStorei, "glPixelStorei")?(GL_UNPACK_ALIGNMENT, 1);
            GlApi::req(api.glTexImage2D, "glTexImage2D")?(
                GL_TEXTURE_2D,
                0,
                GL_R8 as GLint,
                self.atlas_w as GLsizei,
                self.atlas_h as GLsizei,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                self.atlas_pixels.as_ptr().cast(),
            );
        }
        Ok(())
    }

    /// Re-create every GPU object after a lost EGL surface/context. Scene state lives in
    /// `sofuu-ui` and is intentionally untouched.
    pub fn rebuild_after_loss(&mut self) -> Result<(), String> {
        self.delete_resources();
        self.init_resources()
    }

    fn delete_resources(&mut self) {
        let api = self.api;
        unsafe {
            if let Some(f) = api.glDeleteTextures {
                if self.atlas_tex != 0 {
                    f(1, &self.atlas_tex);
                }
            }
            if let Some(f) = api.glDeleteBuffers {
                if self.clip_ubo != 0 {
                    f(1, &self.clip_ubo);
                }
                if self.corner_vbo != 0 {
                    f(1, &self.corner_vbo);
                }
                for b in &self.instance_vbos {
                    if b.id != 0 {
                        f(1, &b.id);
                    }
                }
            }
            if let Some(f) = api.glDeleteVertexArrays {
                for v in &self.vaos {
                    if *v != 0 {
                        f(1, v);
                    }
                }
            }
            if let Some(f) = api.glDeleteProgram {
                for p in &self.programs {
                    if p.id != 0 {
                        f(p.id);
                    }
                }
            }
        }
        self.atlas_tex = 0;
        self.clip_ubo = 0;
        self.corner_vbo = 0;
        self.vaos = [0; 4];
        for b in &mut self.instance_vbos {
            b.id = 0;
            b.capacity = 0;
        }
        self.programs = [
            Program {
                id: 0,
                loc_viewport: -1,
                loc_atlas_size: -1,
                loc_scale: -1,
                loc_time: -1,
            };
            4
        ];
        self.initialized = false;
    }

    /// Offscreen FBO helper used by the macOS/EGL golden parity harness.
    pub fn create_offscreen_target(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(GLuint, GLuint), String> {
        let api = self.api;
        unsafe {
            let mut tex = 0u32;
            GlApi::req(api.glGenTextures, "glGenTextures")?(1, &mut tex);
            GlApi::req(api.glBindTexture, "glBindTexture")?(GL_TEXTURE_2D, tex);
            GlApi::req(api.glTexImage2D, "glTexImage2D")?(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8 as GLint,
                width as GLsizei,
                height as GLsizei,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                std::ptr::null(),
            );
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
            GlApi::req(api.glTexParameteri, "glTexParameteri")?(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
            let mut fbo = 0u32;
            GlApi::req(api.glGenFramebuffers, "glGenFramebuffers")?(1, &mut fbo);
            GlApi::req(api.glBindFramebuffer, "glBindFramebuffer")?(GL_FRAMEBUFFER, fbo);
            GlApi::req(api.glFramebufferTexture2D, "glFramebufferTexture2D")?(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                tex,
                0,
            );
            let status = GlApi::req(api.glCheckFramebufferStatus, "glCheckFramebufferStatus")?(GL_FRAMEBUFFER);
            if status != GL_FRAMEBUFFER_COMPLETE {
                return Err(format!(
                    "E_UI_GL_FBO: offscreen framebuffer incomplete (0x{status:04x})"
                ));
            }
            Ok((fbo, tex))
        }
    }

    /// Bind an offscreen target created by [`Self::create_offscreen_target`].
    pub fn bind_offscreen_target(&self, fbo: GLuint) {
        if let Some(f) = self.api.glBindFramebuffer {
            unsafe { f(GL_FRAMEBUFFER, fbo) };
        }
    }

    /// Read an offscreen RGBA8 target bottom-up (OpenGL order).
    pub fn read_offscreen(&self, width: u32, height: u32, out: &mut [u8]) -> Result<(), String> {
        let api = self.api;
        if out.len() < (width * height * 4) as usize {
            return Err("E_UI_GL_READ: output buffer too small".to_string());
        }
        unsafe {
            GlApi::req(api.glReadPixels, "glReadPixels")?(
                0,
                0,
                width as GLsizei,
                height as GLsizei,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                out.as_mut_ptr().cast(),
            );
        }
        Ok(())
    }

    /// Delete an offscreen target created by [`Self::create_offscreen_target`].
    pub fn destroy_offscreen_target(&self, fbo: GLuint, tex: GLuint) {
        unsafe {
            if let Some(f) = self.api.glDeleteFramebuffers {
                f(1, &fbo);
            }
            if let Some(f) = self.api.glDeleteTextures {
                f(1, &tex);
            }
        }
    }
}

impl Drop for GlesRenderer {
    fn drop(&mut self) {
        if self.initialized {
            self.delete_resources();
        }
    }
}

fn program_log(api: &GlApi, program: GLuint) -> String {
    unsafe {
        let mut len = 0;
        if let Some(f) = api.glGetProgramiv {
            f(program, GL_INFO_LOG_LENGTH, &mut len);
        }
        if len <= 1 {
            return String::new();
        }
        let mut buf = vec![0u8; len as usize];
        if let Some(f) = api.glGetProgramInfoLog {
            f(program, len, std::ptr::null_mut(), buf.as_mut_ptr().cast());
        }
        String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
    }
}

fn shader_log(api: &GlApi, shader: GLuint) -> String {
    unsafe {
        let mut len = 0;
        if let Some(f) = api.glGetShaderiv {
            f(shader, GL_INFO_LOG_LENGTH, &mut len);
        }
        if len <= 1 {
            return String::new();
        }
        let mut buf = vec![0u8; len as usize];
        if let Some(f) = api.glGetShaderInfoLog {
            f(shader, len, std::ptr::null_mut(), buf.as_mut_ptr().cast());
        }
        String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
    }
}

unsafe fn compile_shader(
    api: &GlApi,
    stage: GLenum,
    source: &str,
    pipeline: &str,
    stage_name: &str,
) -> Result<GLuint, String> {
    let shader = GlApi::req(api.glCreateShader, "glCreateShader")?(stage);
    let c = std::ffi::CString::new(source).map_err(|_| {
        format!("E_UI_GL_SHADER: {pipeline} {stage_name} source contains a NUL byte")
    })?;
    let ptr = c.as_ptr();
    let len = source.len() as GLint;
    GlApi::req(api.glShaderSource, "glShaderSource")?(shader, 1, &ptr, &len);
    GlApi::req(api.glCompileShader, "glCompileShader")?(shader);
    let mut ok = 0;
    GlApi::req(api.glGetShaderiv, "glGetShaderiv")?(shader, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let log = shader_log(api, shader);
        let _ = api.glDeleteShader.map(|f| f(shader));
        return Err(format!(
            "E_UI_GL_SHADER: {pipeline} {stage_name} compile failed: {log}"
        ));
    }
    Ok(shader)
}

/// Partition the scene's clip rectangles into contiguous windows small enough for the std140
/// `ClipsBlock` UBO. Every clip id in `0..clip_count` appears in exactly one window, in order.
fn clip_windows(clip_count: usize) -> Vec<(usize, usize)> {
    if clip_count <= MAX_CLIPS {
        return vec![(0, clip_count)];
    }
    (0..clip_count)
        .step_by(MAX_CLIPS)
        .map(|base| (base, (clip_count - base).min(MAX_CLIPS)))
        .collect()
}

/// Translate a scene clip id into the slot index used by the current clip window. Returns
/// `None` when the instance belongs to a different window and must be skipped here.
fn clip_in_window(clip: u32, base: usize, count: usize) -> Option<u32> {
    let c = clip as usize;
    if c >= base && c < base + count {
        Some((c - base) as u32)
    } else {
        None
    }
}

fn shader_sources(flavor: GlslFlavor, pipeline: &str) -> (String, String) {
    let body = match flavor {
        GlslFlavor::Desktop330 => include_str!("../shaders/glsl/ui_shaders_330.glsl"),
        GlslFlavor::Es300 => include_str!("../shaders/glsl/ui_shaders_es3.glsl"),
    };
    let vertex = body.replacen(
        "// __SOFUU_PIPELINE_DEFINES__",
        &format!(
            "#define SOFUU_PIPELINE_{pipeline}\n#define SOFUU_VERTEX\n"
        ),
        1,
    );
    let fragment = body.replacen(
        "// __SOFUU_PIPELINE_DEFINES__",
        &format!(
            "#define SOFUU_PIPELINE_{pipeline}\n#define SOFUU_FRAGMENT\n"
        ),
        1,
    );
    (vertex, fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_layout_constants_match_rust_structs() {
        assert_eq!(std::mem::size_of::<QuadInstance>(), QUAD_STRIDE);
        assert_eq!(std::mem::size_of::<RibbonInstance>(), RIBBON_STRIDE);
        assert_eq!(std::mem::size_of::<StrokeInstance>(), STROKE_STRIDE);
        assert_eq!(std::mem::size_of::<sofuu_ui::GlyphInstance>(), GLYPH_STRIDE);
        assert_eq!(std::mem::size_of::<sofuu_ui::ClipRect>(), 16);
    }

    #[test]
    fn all_pipeline_shader_pairs_define_expected_entry_stages() {
        for flavor in [GlslFlavor::Desktop330, GlslFlavor::Es300] {
            for pipeline in ["QUAD", "RIBBON", "STROKE", "GLYPH"] {
                let (v, f) = shader_sources(flavor, pipeline);
                assert!(v.contains(&format!("#define SOFUU_PIPELINE_{pipeline}")));
                assert!(v.contains("#define SOFUU_VERTEX"));
                assert!(f.contains("#define SOFUU_FRAGMENT"));
                assert!(v.contains("#version"));
                assert!(f.contains("void main()"));
                // The prose header mentions the marker; the actual placeholder line must be gone.
                assert!(!v.lines().any(|l| l.trim() == "// __SOFUU_PIPELINE_DEFINES__"));
                assert!(!f.lines().any(|l| l.trim() == "// __SOFUU_PIPELINE_DEFINES__"));
            }
        }
    }

    #[test]
    fn clip_windows_partition_every_clip_in_order() {
        assert_eq!(clip_windows(0), vec![(0, 0)]);
        assert_eq!(clip_windows(1), vec![(0, 1)]);
        assert_eq!(clip_windows(MAX_CLIPS), vec![(0, MAX_CLIPS)]);
        assert_eq!(
            clip_windows(MAX_CLIPS + 1),
            vec![(0, MAX_CLIPS), (MAX_CLIPS, 1)]
        );

        let total = MAX_CLIPS * 2 + 7;
        let windows = clip_windows(total);
        assert_eq!(windows.len(), 3);
        let mut expected_base = 0;
        for (base, count) in windows {
            assert_eq!(base, expected_base);
            assert!(count > 0 && count <= MAX_CLIPS);
            expected_base += count;
        }
        assert_eq!(expected_base, total);
    }

    #[test]
    fn clip_ids_rebase_into_the_active_window() {
        // Window [256, 512) maps 256..512 to 0..256 and rejects everything else.
        assert_eq!(clip_in_window(255, 256, 256), None);
        assert_eq!(clip_in_window(256, 256, 256), Some(0));
        assert_eq!(clip_in_window(511, 256, 256), Some(255));
        assert_eq!(clip_in_window(512, 256, 256), None);

        // A short tail window still only accepts its own range.
        assert_eq!(clip_in_window(511, 512, 3), None);
        assert_eq!(clip_in_window(514, 512, 3), Some(2));
        assert_eq!(clip_in_window(515, 512, 3), None);
    }

    #[test]
    fn oversized_scene_filters_instances_per_window_without_duplicates() {
        let clip_count = MAX_CLIPS + 3;
        let windows = clip_windows(clip_count);
        let mut seen = vec![0u8; clip_count];
        for (base, count) in windows {
            for clip in 0..clip_count as u32 {
                if let Some(remapped) = clip_in_window(clip, base, count) {
                    assert!(remapped < count as u32);
                    seen[clip as usize] += 1;
                }
            }
        }
        assert!(seen.iter().all(|hits| *hits == 1));
    }
}
