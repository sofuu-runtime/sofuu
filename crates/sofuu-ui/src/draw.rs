//! Packed GPU instance primitives and clip tables for Sofuu's 2D GPU UI compositor.
//!
//! All structs here are `#[repr(C)]` with explicit 16/32/64-byte alignment matching
//! the Metal (MSL), GLSL, and SPIR-V vertex/fragment shaders byte-for-byte.

/// Pack 8-bit sRGBA channels into a little-endian `u32` (`0xAABBGGRR` in memory `[r, g, b, a]`).
#[inline]
pub const fn rgba8(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

/// Construct a packed RGBA8 color from a `0xRRGGBB` hex literal and explicit alpha.
#[inline]
pub const fn hex_rgba(rgb: u32, a: u8) -> u32 {
    let r = ((rgb >> 16) & 0xff) as u8;
    let g = ((rgb >> 8) & 0xff) as u8;
    let b = (rgb & 0xff) as u8;
    rgba8(r, g, b, a)
}

/// Unpack a packed `u32` color into `[r, g, b, a]` bytes.
#[inline]
pub const fn unpack_rgba8(c: u32) -> [u8; 4] {
    [
        (c & 0xff) as u8,
        ((c >> 8) & 0xff) as u8,
        ((c >> 16) & 0xff) as u8,
        ((c >> 24) & 0xff) as u8,
    ]
}

/// Override the alpha byte of a packed RGBA8 color without touching RGB.
#[inline]
pub const fn with_alpha(c: u32, a: u8) -> u32 {
    (c & 0x00ff_ffff) | ((a as u32) << 24)
}

/// Scale the alpha byte of a packed RGBA8 color by `factor` in `0.0..=1.0`.
#[inline]
pub fn scale_alpha(c: u32, factor: f32) -> u32 {
    let f = factor.clamp(0.0, 1.0);
    let a = ((c >> 24) & 0xff) as f32;
    with_alpha(c, (a * f).round().clamp(0.0, 255.0) as u8)
}

/// Linearly interpolate two packed RGBA8 colors for smooth 120Hz spring color transitions.
#[inline]
pub fn lerp_rgba8(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let [ar, ag, ab, aa] = unpack_rgba8(a);
    let [br, bg, bb, ba] = unpack_rgba8(b);
    let mix = |x: u8, y: u8| -> u8 {
        ((x as f32) + ((y as f32) - (x as f32)) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    rgba8(mix(ar, br), mix(ag, bg), mix(ab, bb), mix(aa, ba))
}

/// 80-byte instanced SDF rounded box (card, button, pill, shadow, border, vertical gradient).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadInstance {
    /// `[x, y, width, height]` in logical pixels.
    pub rect: [f32; 4],
    /// Corner radii `[top_left, top_right, bottom_right, bottom_left]` in logical pixels.
    pub radii: [f32; 4],
    /// Top fill color (packed RGBA8).
    pub color_top: u32,
    /// Bottom fill color (packed RGBA8, identical to `color_top` for flat fills).
    pub color_bottom: u32,
    /// Hairline / stroke border color (packed RGBA8).
    pub border_color: u32,
    /// Analytic soft drop shadow color (packed RGBA8).
    pub shadow_color: u32,
    /// `[border_width, shadow_blur, shadow_offset_y, opacity]`.
    pub params: [f32; 4],
    /// Index into `DrawList::clips` (`0` = full viewport).
    pub clip_id: u32,
    pub _pad: [u32; 3],
}

impl Default for QuadInstance {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            radii: [0.0; 4],
            color_top: 0,
            color_bottom: 0,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        }
    }
}

/// 32-byte instanced antialiased vector line capsule (charts, graphs, dividers, connectors).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeInstance {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub width: f32,
    pub feather: f32,
    pub color: u32,
    pub clip_id: u32,
}

/// 48-byte instanced SDF/MSDF glyph quad sampled from the pre-baked single-channel/RGB atlas.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphInstance {
    /// Top-left destination `[x, y]` in logical pixels.
    pub pos: [f32; 2],
    /// Quad `[width, height]` in logical pixels.
    pub size: [f32; 2],
    /// Normalized texture coordinates `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Packed RGBA8 text color.
    pub color: u32,
    /// SDF distance range in atlas pixels (`px_range`).
    pub px_range: f32,
    pub clip_id: u32,
    pub _pad: u32,
}

/// 16-byte scissor / analytic clip rectangle in logical pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl ClipRect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Intersect two clip rectangles. Returns a zero-area rect if disjoint.
    pub fn intersect(self, other: Self) -> Self {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.w).min(other.x + other.w);
        let y1 = (self.y + self.h).min(other.y + other.h);
        Self {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        }
    }
}

/// 32-byte per-frame GPU uniform block shared across all shader passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameUniforms {
    /// Logical viewport `[width, height]`.
    pub viewport_size: [f32; 2],
    /// Atlas dimensions `[width, height]` in pixels.
    pub atlas_size: [f32; 2],
    /// Display DPI scale factor (`1.0`, `2.0` Retina, etc.).
    pub scale_factor: f32,
    /// Elapsed animation time in seconds (optional shader effects).
    pub time_sec: f32,
    pub _pad: [f32; 2],
}

/// Flat GPU command buffer emitted by `UiEngine` and uploaded directly to vertex/instance buffers.
#[derive(Clone, Debug, Default)]
pub struct DrawList {
    pub quads: Vec<QuadInstance>,
    pub strokes: Vec<StrokeInstance>,
    pub glyphs: Vec<GlyphInstance>,
    pub clips: Vec<ClipRect>,
    clip_stack: Vec<u32>,
}

impl DrawList {
    pub fn new() -> Self {
        let mut dl = Self::default();
        dl.reset([1280.0, 800.0]);
        dl
    }

    /// Reset buffers while preserving heap capacity across frames (zero steady-state allocs).
    pub fn reset(&mut self, viewport: [f32; 2]) {
        self.quads.clear();
        self.strokes.clear();
        self.glyphs.clear();
        self.clips.clear();
        self.clip_stack.clear();
        // clip_id 0 is always the full viewport.
        self.clips
            .push(ClipRect::new(0.0, 0.0, viewport[0].max(0.0), viewport[1].max(0.0)));
        self.clip_stack.push(0);
    }

    #[inline]
    pub fn current_clip_id(&self) -> u32 {
        *self.clip_stack.last().unwrap_or(&0)
    }

    #[inline]
    pub fn current_clip(&self) -> ClipRect {
        self.clips[self.current_clip_id() as usize]
    }

    /// Push a clip rectangle intersected with the active clip, returning its `clip_id`.
    pub fn push_clip(&mut self, rect: ClipRect) -> u32 {
        let intersected = self.current_clip().intersect(rect);
        let id = self.clips.len() as u32;
        self.clips.push(intersected);
        self.clip_stack.push(id);
        id
    }

    /// Pop the top clip rectangle (never pops root clip `0`).
    pub fn pop_clip(&mut self) {
        if self.clip_stack.len() > 1 {
            self.clip_stack.pop();
        }
    }

    #[inline]
    pub fn push_quad(&mut self, mut quad: QuadInstance) {
        quad.clip_id = self.current_clip_id();
        self.quads.push(quad);
    }

    #[inline]
    pub fn push_stroke(&mut self, mut stroke: StrokeInstance) {
        stroke.clip_id = self.current_clip_id();
        self.strokes.push(stroke);
    }

    #[inline]
    pub fn push_glyph(&mut self, mut glyph: GlyphInstance) {
        glyph.clip_id = self.current_clip_id();
        self.glyphs.push(glyph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn gpu_struct_sizes_match_shader_layout() {
        assert_eq!(size_of::<QuadInstance>(), 80);
        assert_eq!(size_of::<StrokeInstance>(), 32);
        assert_eq!(size_of::<GlyphInstance>(), 48);
        assert_eq!(size_of::<ClipRect>(), 16);
        assert_eq!(size_of::<FrameUniforms>(), 32);
        assert!(align_of::<QuadInstance>() >= 4);
        assert!(align_of::<StrokeInstance>() >= 4);
        assert!(align_of::<GlyphInstance>() >= 4);
    }

    #[test]
    fn clip_stack_intersects_and_never_underflows() {
        let mut dl = DrawList::new();
        dl.reset([1000.0, 600.0]);
        assert_eq!(dl.current_clip_id(), 0);
        let c1 = dl.push_clip(ClipRect::new(100.0, 100.0, 400.0, 300.0));
        assert_eq!(c1, 1);
        let c2 = dl.push_clip(ClipRect::new(200.0, 50.0, 500.0, 200.0));
        assert_eq!(c2, 2);
        assert_eq!(dl.current_clip(), ClipRect::new(200.0, 100.0, 300.0, 150.0));
        dl.pop_clip();
        assert_eq!(dl.current_clip_id(), 1);
        dl.pop_clip();
        dl.pop_clip(); // extra pop stays at 0
        assert_eq!(dl.current_clip_id(), 0);
    }

    #[test]
    fn color_packing_and_lerp() {
        let c = rgba8(10, 20, 30, 255);
        assert_eq!(unpack_rgba8(c), [10, 20, 30, 255]);
        let h = hex_rgba(0x0b0f14, 255);
        assert_eq!(unpack_rgba8(h), [0x0b, 0x0f, 0x14, 255]);
        let mid = lerp_rgba8(rgba8(0, 0, 0, 0), rgba8(100, 200, 50, 250), 0.5);
        assert_eq!(unpack_rgba8(mid), [50, 100, 25, 125]);
    }
}
