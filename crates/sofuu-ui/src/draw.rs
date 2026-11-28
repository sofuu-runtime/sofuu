//! Packed GPU instance primitives, ordered draw batches, and clip tables for Sofuu's 2D GPU UI compositor.
//!
//! All structs here are `#[repr(C)]` with explicit 16-byte alignment matching
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

/// 80-byte instanced SDF rounded box (card, button, pill, candle body, shadow, border, vertical gradient).
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

/// 48-byte instanced sloped 2D trapezoid/ribbon slice for smooth area fills, stacked bands, and oscillator spreads.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonInstance {
    pub x0: f32,
    pub y0_top: f32,
    pub y0_bot: f32,
    pub x1: f32,
    pub y1_top: f32,
    pub y1_bot: f32,
    /// Fill color at the top boundary (`y*_top`), packed RGBA8.
    pub color_top: u32,
    /// Fill color at the bottom boundary (`y*_bot`), packed RGBA8.
    pub color_bottom: u32,
    pub clip_id: u32,
    pub _pad: [u32; 3],
}

/// 48-byte instanced antialiased vector line capsule with optional GPU dashing (solid when `dash_len <= 0.0`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeInstance {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub width: f32,
    pub feather: f32,
    /// Dash length in logical pixels (`0.0` = solid line).
    pub dash_len: f32,
    /// Gap length between dashes in logical pixels (`0.0` = solid line).
    pub gap_len: f32,
    pub color: u32,
    pub clip_id: u32,
    pub _pad: [u32; 2],
}

impl StrokeInstance {
    #[inline]
    pub const fn solid(p0: [f32; 2], p1: [f32; 2], width: f32, feather: f32, color: u32, clip_id: u32) -> Self {
        Self {
            p0,
            p1,
            width,
            feather,
            dash_len: 0.0,
            gap_len: 0.0,
            color,
            clip_id,
            _pad: [0; 2],
        }
    }

    #[inline]
    pub const fn dashed(
        p0: [f32; 2],
        p1: [f32; 2],
        width: f32,
        feather: f32,
        dash_len: f32,
        gap_len: f32,
        color: u32,
        clip_id: u32,
    ) -> Self {
        Self {
            p0,
            p1,
            width,
            feather,
            dash_len,
            gap_len,
            color,
            clip_id,
            _pad: [0; 2],
        }
    }
}

/// 48-byte instanced SDF glyph quad sampled from the pre-baked single-channel atlas.
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

/// 64-byte instanced textured sprite quad (images, avatars, thumbnails, SDF icons).
///
/// The fragment shader branches on `flags`:
/// - [`SPRITE_FLAG_SDF`]: sample the `R8` icon atlas as a distance field and
///   derive coverage analytically, so one atlas stays crisp at `1x`/`2x`/`3x`.
/// - [`SPRITE_FLAG_PREMULTIPLIED`]: sample an `RGBA8` image atlas page whose
///   texels are already premultiplied.
///
/// `radii` clips rounded avatars/cards directly in the fragment shader; `tint`
/// multiplies the sampled color, which recolors monochrome SDF icons for free.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteInstance {
    /// Destination bounds `[x, y, w, h]` in logical pixels.
    pub rect: [f32; 4],
    /// Atlas or bound texture `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Corner radii `[tl, tr, br, bl]` clipped in the fragment shader.
    pub radii: [f32; 4],
    /// Multiplicative tint (packed RGBA8, `0xFFFFFFFF` = untinted).
    pub tint: u32,
    /// Texture slot (`0` = built-in SDF icon atlas, `1..N` = image atlas pages).
    pub tex_id: u32,
    pub clip_id: u32,
    /// Shader flags: bit 0 = SDF icon mask, bit 1 = premultiplied RGBA image.
    pub flags: u32,
}

/// Treat the sampled `R8` channel as a signed distance field.
pub const SPRITE_FLAG_SDF: u32 = 1 << 0;
/// Treat the sampled `RGBA8` texel as premultiplied alpha.
pub const SPRITE_FLAG_PREMULTIPLIED: u32 = 1 << 1;

impl SpriteInstance {
    /// Opaque, unfiltered, untinted sprite of the whole bound texture.
    pub const fn new(rect: [f32; 4], uv: [f32; 4]) -> Self {
        Self {
            rect,
            uv,
            radii: [0.0; 4],
            tint: 0xffff_ffff,
            tex_id: 0,
            clip_id: 0,
            flags: 0,
        }
    }

    #[inline]
    pub const fn with_radii(mut self, radii: [f32; 4]) -> Self {
        self.radii = radii;
        self
    }

    #[inline]
    pub const fn with_tint(mut self, tint: u32) -> Self {
        self.tint = tint;
        self
    }

    #[inline]
    pub const fn with_tex(mut self, tex_id: u32) -> Self {
        self.tex_id = tex_id;
        self
    }

    #[inline]
    pub const fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }
}

/// 80-byte parameterized UI effect quad (shimmer, ripple, beam, glow, aurora, pulse, orb).
///
/// One fragment shader branches on `kind_and_clip` so every platform compiles a single effect
/// pipeline instead of one state object per effect.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectQuadInstance {
    /// `[x, y, width, height]` in logical pixels.
    pub rect: [f32; 4],
    /// Corner radii `[top_left, top_right, bottom_right, bottom_left]` in logical pixels.
    pub radii: [f32; 4],
    /// Effect-specific parameters (pointer origin, speed, band width, intensity, ...).
    pub params0: [f32; 4],
    /// Secondary parameters (timeline phase, ring radius, amplitude, seed, ...).
    pub params1: [f32; 4],
    pub color0: u32,
    pub color1: u32,
    pub color2: u32,
    /// `[effect_kind: u16, clip_id: u16]` packed little-endian.
    pub kind_and_clip: u32,
}

impl Default for EffectQuadInstance {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            radii: [0.0; 4],
            params0: [0.0; 4],
            params1: [0.0; 4],
            color0: 0,
            color1: 0,
            color2: 0,
            kind_and_clip: 0,
        }
    }
}

/// 32-byte instanced particle (confetti ribbon, spark, ambient dot).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleInstance {
    /// Current center `[x, y]` in logical pixels.
    pub pos: [f32; 2],
    /// `[width, height]` in logical pixels (radius for circles).
    pub size: [f32; 2],
    /// 2D rotation in radians (confetti ribbons).
    pub angle_rad: f32,
    /// Packed RGBA8 with faded alpha.
    pub color: u32,
    /// Lower 4 bits: shape (`0=circle`, `1=ribbon`, `2=star/spark`); upper bits: `clip_id`.
    pub shape_clip: u32,
    pub _pad: u32,
}

impl Default for ParticleInstance {
    fn default() -> Self {
        Self {
            pos: [0.0; 2],
            size: [0.0; 2],
            angle_rad: 0.0,
            color: 0,
            shape_clip: 0,
            _pad: 0,
        }
    }
}

/// 64-byte backdrop partition marker: blur everything painted before this point, then composite
/// the frosted region (tint + specular rim) before normal foreground drawing resumes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropRegion {
    /// `[x, y, width, height]` in logical pixels.
    pub rect: [f32; 4],
    /// Corner radii `[top_left, top_right, bottom_right, bottom_left]` in logical pixels.
    pub radii: [f32; 4],
    /// Surface tint composited over the blurred backdrop.
    pub tint: u32,
    /// Inner 1px specular rim color.
    pub rim: u32,
    /// `[blur_radius, rim_width, grain, opacity]`.
    pub params: [f32; 4],
    pub clip_id: u32,
    pub _pad: u32,
}

impl Default for BackdropRegion {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            radii: [0.0; 4],
            tint: 0,
            rim: 0,
            params: [18.0, 1.0, 0.012, 1.0],
            clip_id: 0,
            _pad: 0,
        }
    }
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

/// Wire version of the `#[repr(C)]` draw structs and their shader contract.
///
/// Bump this whenever the byte layout of [`QuadInstance`], [`RibbonInstance`],
/// [`StrokeInstance`], [`GlyphInstance`], [`SpriteInstance`],
/// [`EffectQuadInstance`], [`ParticleInstance`], [`BackdropRegion`],
/// [`ClipRect`], [`DrawCmd`] or [`FrameUniforms`] changes, or when a
/// `DRAW_KIND_*` id is renumbered. `sofuu-gpu` pins the value with a compile
/// time assertion, and the bundled shader sources in [`crate::shaders`] are
/// the other half of the same contract. Additive kinds/fields still bump it:
/// a host built against the old struct must fail loudly rather than read the
/// wrong stride.
pub const SOFUU_DRAW_ABI_VERSION: u32 = 1;

/// Pipeline kind identifier for ordered GPU command batches (`DrawCmd::kind`).
pub const DRAW_KIND_QUAD: u32 = 0;
pub const DRAW_KIND_RIBBON: u32 = 1;
pub const DRAW_KIND_STROKE: u32 = 2;
pub const DRAW_KIND_GLYPH: u32 = 3;
/// Parameterized `EffectQuadInstance` batch.
pub const DRAW_KIND_EFFECT: u32 = 5;
/// `ParticleInstance` batch.
pub const DRAW_KIND_PARTICLE: u32 = 6;
/// Backdrop partition marker (`start` indexes `DrawList::backdrops`, `count == 1`).
pub const DRAW_KIND_BACKDROP: u32 = 7;
pub const DRAW_KIND_SPRITE: u32 = 4;

/// 16-byte GPU draw batch preserving exact painter's order across shader pipelines.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawCmd {
    pub kind: u32,
    pub start: u32,
    pub count: u32,
    pub _pad: u32,
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
    pub ribbons: Vec<RibbonInstance>,
    pub strokes: Vec<StrokeInstance>,
    pub glyphs: Vec<GlyphInstance>,
    pub sprites: Vec<SpriteInstance>,
    pub effects: Vec<EffectQuadInstance>,
    pub particles: Vec<ParticleInstance>,
    pub backdrops: Vec<BackdropRegion>,
    pub cmds: Vec<DrawCmd>,
    pub clips: Vec<ClipRect>,
    clip_stack: Vec<u32>,
    /// Set by `UiEngine::tick_and_compile`: the host should present another
    /// frame now. A settled list keeps this `false`, which is the 0%-idle
    /// contract every host loop reads instead of reaching into engine state.
    pub needs_next_frame: bool,
    /// Engine-clock seconds at which a *timed* wake is due (caret blink).
    /// `None` means only input/animations can wake the loop.
    pub next_wake_s: Option<f64>,
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
        self.ribbons.clear();
        self.strokes.clear();
        self.glyphs.clear();
        self.sprites.clear();
        self.effects.clear();
        self.particles.clear();
        self.backdrops.clear();
        self.cmds.clear();
        self.clips.clear();
        self.clip_stack.clear();
        // clip_id 0 is always the full viewport.
        self.clips
            .push(ClipRect::new(0.0, 0.0, viewport[0].max(0.0), viewport[1].max(0.0)));
        self.clip_stack.push(0);
    }

    #[inline]
    fn record_cmd(&mut self, kind: u32, start: usize) {
        self.record_batch(kind, start, 1);
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
        let idx = self.quads.len();
        self.quads.push(quad);
        self.record_cmd(DRAW_KIND_QUAD, idx);
    }

    #[inline]
    pub fn push_ribbon(&mut self, mut ribbon: RibbonInstance) {
        ribbon.clip_id = self.current_clip_id();
        let idx = self.ribbons.len();
        self.ribbons.push(ribbon);
        self.record_cmd(DRAW_KIND_RIBBON, idx);
    }

    #[inline]
    pub fn push_stroke(&mut self, mut stroke: StrokeInstance) {
        stroke.clip_id = self.current_clip_id();
        let idx = self.strokes.len();
        self.strokes.push(stroke);
        self.record_cmd(DRAW_KIND_STROKE, idx);
    }

    #[inline]
    pub fn push_glyph(&mut self, mut glyph: GlyphInstance) {
        glyph.clip_id = self.current_clip_id();
        let idx = self.glyphs.len();
        self.glyphs.push(glyph);
        self.record_cmd(DRAW_KIND_GLYPH, idx);
    }

    /// Record a batch of `count` glyphs already appended directly to `self.glyphs`.
    #[inline]
    pub fn record_glyph_batch(&mut self, start: usize, count: usize) {
        self.record_batch(DRAW_KIND_GLYPH, start, count);
    }

    #[inline]
    pub fn push_sprite(&mut self, mut sprite: SpriteInstance) {
        sprite.clip_id = self.current_clip_id();
        let idx = self.sprites.len();
        self.sprites.push(sprite);
        self.record_cmd(DRAW_KIND_SPRITE, idx);
    }

    /// Record a batch of `count` sprites already appended directly to `self.sprites`.
    #[inline]
    pub fn record_sprite_batch(&mut self, start: usize, count: usize) {
        self.record_batch(DRAW_KIND_SPRITE, start, count);
    }

    #[inline]
    pub fn push_effect(&mut self, mut effect: EffectQuadInstance) {
        let clip = self.current_clip_id();
        let kind = (effect.kind_and_clip & 0xffff) as u16;
        effect.kind_and_clip = (kind as u32) | (clip << 16);
        let idx = self.effects.len();
        self.effects.push(effect);
        self.record_cmd(DRAW_KIND_EFFECT, idx);
    }

    #[inline]
    pub fn push_particle(&mut self, mut particle: ParticleInstance) {
        let clip = self.current_clip_id();
        let shape = particle.shape_clip & 0xf;
        particle.shape_clip = shape | (clip << 4);
        let idx = self.particles.len();
        self.particles.push(particle);
        self.record_cmd(DRAW_KIND_PARTICLE, idx);
    }

    /// Append a backdrop partition marker. Unlike instance batches, two adjacent markers must
    /// never merge: each one blurs the framebuffer state at that exact painter position.
    #[inline]
    pub fn push_backdrop(&mut self, mut region: BackdropRegion) {
        region.clip_id = self.current_clip_id();
        let idx = self.backdrops.len();
        self.backdrops.push(region);
        self.cmds.push(DrawCmd {
            kind: DRAW_KIND_BACKDROP,
            start: idx as u32,
            count: 1,
            _pad: 0,
        });
    }

    /// Merge a contiguous run of one pipeline into the previous command when
    /// painter order allows it, avoiding one `DrawCmd` per glyph/sprite.
    #[inline]
    fn record_batch(&mut self, kind: u32, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        if let Some(last) = self.cmds.last_mut() {
            if last.kind == kind && (last.start + last.count) as usize == start {
                last.count += count as u32;
                return;
            }
        }
        self.cmds.push(DrawCmd {
            kind,
            start: start as u32,
            count: count as u32,
            _pad: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn gpu_struct_sizes_match_shader_layout() {
        assert_eq!(size_of::<QuadInstance>(), 80);
        assert_eq!(size_of::<RibbonInstance>(), 48);
        assert_eq!(size_of::<StrokeInstance>(), 48);
        assert_eq!(size_of::<GlyphInstance>(), 48);
        assert_eq!(size_of::<SpriteInstance>(), 64);
        assert_eq!(size_of::<EffectQuadInstance>(), 80);
        assert_eq!(size_of::<ParticleInstance>(), 32);
        assert_eq!(size_of::<BackdropRegion>(), 64);
        assert_eq!(size_of::<ClipRect>(), 16);
        assert_eq!(size_of::<DrawCmd>(), 16);
        assert_eq!(size_of::<FrameUniforms>(), 32);
        assert!(align_of::<QuadInstance>() >= 4);
        assert!(align_of::<RibbonInstance>() >= 4);
        assert!(align_of::<StrokeInstance>() >= 4);
        assert!(align_of::<GlyphInstance>() >= 4);
        assert!(align_of::<SpriteInstance>() >= 4);
    }

    #[test]
    fn sprite_batches_preserve_painter_order_and_merge_runs() {
        let mut dl = DrawList::new();
        dl.reset([400.0, 300.0]);
        dl.push_quad(QuadInstance::default());
        dl.push_sprite(SpriteInstance::new([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]));
        dl.push_sprite(SpriteInstance::new([5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 1.0, 1.0]));
        dl.push_quad(QuadInstance::default());

        assert_eq!(dl.sprites.len(), 2);
        assert_eq!(
            dl.cmds.iter().map(|c| (c.kind, c.start, c.count)).collect::<Vec<_>>(),
            vec![(DRAW_KIND_QUAD, 0, 1), (DRAW_KIND_SPRITE, 0, 2), (DRAW_KIND_QUAD, 1, 1)]
        );
        // The current clip is stamped onto each sprite automatically.
        assert_eq!(dl.sprites[0].clip_id, 0);
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
        dl.pop_clip();
        assert_eq!(dl.current_clip_id(), 0);
    }

    /// Independent of the C header gate: this pins the Rust half of the
    /// draw ABI so a field insertion cannot silently change an instance
    /// stride between `sofuu-ui` and `sofuu-gpu`.
    #[test]
    fn draw_abi_layout_is_pinned() {
        use std::mem::{align_of, size_of};

        assert_eq!(SOFUU_DRAW_ABI_VERSION, 1, "bump the shaders with the ABI");
        assert_eq!(size_of::<FrameUniforms>(), 32);
        assert_eq!(size_of::<ClipRect>(), 16);
        assert_eq!(size_of::<DrawCmd>(), 16);
        assert_eq!(size_of::<QuadInstance>(), 80);
        assert_eq!(size_of::<RibbonInstance>(), 48);
        assert_eq!(size_of::<StrokeInstance>(), 48);
        assert_eq!(size_of::<GlyphInstance>(), 48);
        assert_eq!(size_of::<SpriteInstance>(), 64);
        assert_eq!(size_of::<EffectQuadInstance>(), 80);
        assert_eq!(size_of::<ParticleInstance>(), 32);
        assert_eq!(size_of::<BackdropRegion>(), 64);
        for align in [
            align_of::<FrameUniforms>(),
            align_of::<QuadInstance>(),
            align_of::<SpriteInstance>(),
        ] {
            assert_eq!(align, 4);
        }
        assert_eq!(
            [
                DRAW_KIND_QUAD,
                DRAW_KIND_RIBBON,
                DRAW_KIND_STROKE,
                DRAW_KIND_GLYPH,
                DRAW_KIND_SPRITE,
                DRAW_KIND_EFFECT,
                DRAW_KIND_PARTICLE,
                DRAW_KIND_BACKDROP,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
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
