//! Pre-baked pure-Rust Signed Distance Field (SDF) glyph atlas & text layout engine.
//!
//! Default embedded font: **Geist Mono** (`assets/geist_mono.sfdf`), baked offline at 6x
//! supersampling from `GeistMono-Regular.ttf` into a compact `768x288` `R8` SDF texture atlas.
//! Zero OS font dependencies and zero per-frame CPU rasterization across Metal, OpenGL, and Vulkan.

use crate::draw::{DrawList, GlyphInstance};

const DEFAULT_FONT_SFDF: &[u8] = include_bytes!("../assets/geist_mono.sfdf");

#[derive(Clone, Copy, Debug)]
pub struct GlyphMetrics {
    /// Normalized atlas UVs `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Top-left quad X offset from pen origin as a multiple of `px_size`.
    pub quad_offset_x_em: f32,
    /// Top-left quad Y offset from line top as a multiple of `px_size`.
    pub quad_offset_y_em: f32,
    /// Quad width as a multiple of `px_size`.
    pub quad_w_em: f32,
    /// Quad height as a multiple of `px_size`.
    pub quad_h_em: f32,
    /// Horizontal pen advance as a multiple of `px_size`.
    pub advance_em: f32,
}

/// CPU-owned SDF texture atlas ready to upload once to an `R8Unorm` GPU texture.
#[derive(Clone, Debug)]
pub struct GlyphAtlas {
    pub width: u32,
    pub height: u32,
    /// Row-major `width * height` `R8` SDF bytes (`128` = contour edge, `>128` = inside).
    pub pixels: Vec<u8>,
    pub px_range: f32,
    pub advance_em: f32,
    pub ascender_em: f32,
    pub line_height_em: f32,
    /// Optical cap-height ink top & vertical center in `em` units (derived automatically from `'M'`/`'0'`).
    pub cap_top_em: f32,
    pub cap_center_em: f32,
    /// Metrics for ASCII `0x20..=0x7E` (95 entries) + index `95` fallback `.notdef`.
    metrics: Vec<GlyphMetrics>,
}

impl GlyphAtlas {
    /// Load the built-in default **Geist Mono** SDF atlas (`O(1)` decode from embedded bytes).
    pub fn builtin() -> Self {
        Self::from_sfdf(DEFAULT_FONT_SFDF).expect("embedded geist_mono.sfdf must be valid")
    }

    /// Parse any pre-baked `SFDF` atlas binary blob (allows custom runtime font swaps cleanly).
    pub fn from_sfdf(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 32 || &data[0..4] != b"SFDF" {
            return Err("invalid SFDF header");
        }
        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };
        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };

        let width = read_u32(4);
        let height = read_u32(8);
        let px_range = read_f32(12);
        let glyph_count = read_u32(16) as usize;
        let advance_em = read_f32(20);
        let ascender_em = read_f32(24);
        let line_height_em = read_f32(28);

        let entry_bytes = glyph_count * 32;
        let pixel_bytes = (width as usize) * (height as usize);
        if data.len() < 32 + entry_bytes + pixel_bytes || glyph_count < 96 {
            return Err("truncated SFDF payload");
        }

        let mut metrics = Vec::with_capacity(glyph_count);
        for i in 0..glyph_count {
            let base = 32 + i * 32;
            metrics.push(GlyphMetrics {
                uv: [
                    read_f32(base),
                    read_f32(base + 4),
                    read_f32(base + 8),
                    read_f32(base + 12),
                ],
                quad_offset_x_em: read_f32(base + 16),
                quad_offset_y_em: read_f32(base + 20),
                quad_w_em: read_f32(base + 24),
                quad_h_em: read_f32(base + 28),
                advance_em,
            });
        }

        let pixels = data[32 + entry_bytes..32 + entry_bytes + pixel_bytes].to_vec();
        let (cap_top_em, cap_center_em) =
            measure_cap_optics(width, height, &pixels, &metrics, ascender_em);
        Ok(Self {
            width,
            height,
            pixels,
            px_range,
            advance_em,
            ascender_em,
            line_height_em,
            cap_top_em,
            cap_center_em,
            metrics,
        })
    }

    /// Compute `pen_y` so the optical cap center of `px_size` lands exactly at `rect_y + rect_h * 0.5`.
    #[inline]
    pub fn centered_pen_y(&self, rect_y: f32, rect_h: f32, px_size: f32) -> f32 {
        rect_y + rect_h * 0.5 - self.cap_center_em * px_size
    }

    /// Compute `pen_y` for a secondary smaller font size `sub_px` so its baseline aligns cleanly
    /// with a primary headline rendered at `(primary_y, primary_px)`.
    #[inline]
    pub fn baseline_aligned_y(&self, primary_y: f32, primary_px: f32, sub_px: f32) -> f32 {
        primary_y + (primary_px - sub_px) * self.ascender_em
    }

    /// Draw single-line text centered horizontally and optically centered vertically inside `[x, y, w, h]`.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_centered(
        &self,
        text: &str,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        px_size: f32,
        color: u32,
        mono: bool,
        dl: &mut DrawList,
    ) {
        let (tw, _) = self.measure(text, px_size, mono);
        let tx = x + (w - tw) * 0.5;
        let ty = self.centered_pen_y(y, h, px_size);
        self.layout_dl(text, tx, ty, px_size, color, mono, None, dl);
    }

    #[inline]
    pub fn metrics(&self, ch: char) -> GlyphMetrics {
        let code = ch as u32;
        let idx = if (0x20..=0x7E).contains(&code) {
            (code - 0x20) as usize
        } else {
            95
        };
        let mut m = self.metrics[idx];
        m.advance_em = self.advance_em;
        m
    }

    /// Compatibility alias returning `(em_advance, mono_em_advance)`.
    #[inline]
    pub fn char_advance_em(&self, ch: char, _mono: bool) -> f32 {
        self.metrics(ch).advance_em
    }

    /// Measure single-line or multi-line (`\n`) text block in logical pixels.
    pub fn measure(&self, text: &str, px_size: f32, _mono: bool) -> (f32, f32) {
        let line_h = px_size * self.line_height_em;
        let mut max_w: f32 = 0.0;
        let mut cur_w: f32 = 0.0;
        let mut lines = 1usize;

        for ch in text.chars() {
            if ch == '\n' {
                max_w = max_w.max(cur_w);
                cur_w = 0.0;
                lines += 1;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            cur_w += self.metrics(ch).advance_em * px_size;
        }
        max_w = max_w.max(cur_w);
        (max_w, lines as f32 * line_h)
    }

    /// Emit `GlyphInstance` quads directly into a `DrawList` and record an ordered glyph batch.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_dl(
        &self,
        text: &str,
        x: f32,
        y: f32,
        px_size: f32,
        color: u32,
        mono: bool,
        max_width: Option<f32>,
        dl: &mut DrawList,
    ) {
        let clip = dl.current_clip_id();
        let start = dl.glyphs.len();
        self.layout_into(text, x, y, px_size, color, mono, max_width, &mut dl.glyphs, clip);
        let added = dl.glyphs.len() - start;
        dl.record_glyph_batch(start, added);
    }

    /// Emit `GlyphInstance` quads into `out` starting at top-left `(x, y)` within optional `max_width`.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_into(
        &self,
        text: &str,
        x: f32,
        y: f32,
        px_size: f32,
        color: u32,
        _mono: bool,
        max_width: Option<f32>,
        out: &mut Vec<GlyphInstance>,
        clip_id: u32,
    ) {
        let line_h = px_size * self.line_height_em;
        let mut pen_x = x;
        let mut pen_y = y;

        for ch in text.chars() {
            if ch == '\n' {
                pen_x = x;
                pen_y += line_h;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            let m = self.metrics(ch);
            let adv = m.advance_em * px_size;
            if let Some(limit) = max_width {
                if (pen_x - x) + adv > limit && pen_x > x {
                    pen_x = x;
                    pen_y += line_h;
                }
            }
            if ch != ' ' {
                out.push(GlyphInstance {
                    pos: [
                        pen_x + m.quad_offset_x_em * px_size,
                        pen_y + m.quad_offset_y_em * px_size,
                    ],
                    size: [m.quad_w_em * px_size, m.quad_h_em * px_size],
                    uv: m.uv,
                    color,
                    px_range: self.px_range,
                    clip_id,
                    _pad: 0,
                });
            }
            pen_x += adv;
        }
    }
}

/// Inspect `'M'` (fallback `'0'`) SDF ink bounds once at atlas load so optical vertical centering
/// adapts automatically to any loaded SFDF font without hardcoded magic offsets.
fn measure_cap_optics(
    atlas_w: u32,
    atlas_h: u32,
    pixels: &[u8],
    metrics: &[GlyphMetrics],
    ascender_em: f32,
) -> (f32, f32) {
    let idx = ('M' as usize).saturating_sub(0x20);
    if let Some(m) = metrics.get(idx) {
        let x0 = (m.uv[0] * atlas_w as f32).round() as u32;
        let y0 = (m.uv[1] * atlas_h as f32).round() as u32;
        let x1 = (m.uv[2] * atlas_w as f32).round().min(atlas_w as f32) as u32;
        let y1 = (m.uv[3] * atlas_h as f32).round().min(atlas_h as f32) as u32;
        let cell_h = y1.saturating_sub(y0).max(1) as f32;
        let mut min_row = u32::MAX;
        let mut max_row = 0u32;
        for py in y0..y1 {
            for px in x0..x1 {
                if pixels[(py * atlas_w + px) as usize] >= 128 {
                    let rel = py - y0;
                    min_row = min_row.min(rel);
                    max_row = max_row.max(rel);
                }
            }
        }
        if min_row <= max_row {
            let em_top = m.quad_offset_y_em + (min_row as f32 / cell_h) * m.quad_h_em;
            let em_bot = m.quad_offset_y_em + ((max_row + 1) as f32 / cell_h) * m.quad_h_em;
            return (em_top, (em_top + em_bot) * 0.5);
        }
    }
    let fallback_top = ascender_em * 0.28;
    (fallback_top, (fallback_top + ascender_em) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geist_mono_sfdf_loads_with_valid_sdf_range_and_metrics() {
        let atlas = GlyphAtlas::builtin();
        assert_eq!(atlas.width, 768);
        assert_eq!(atlas.height, 288);
        assert_eq!(atlas.pixels.len(), (768 * 288) as usize);
        assert!((atlas.advance_em - 0.60).abs() < 0.02);

        // Verify glyph 'A' (index 33) has crisp interior (>180) and exterior (<30) SDF contrast.
        let idx = ('A' as u32) - 0x20;
        let ox = (idx % 16) * 48;
        let oy = (idx / 16) * 48;
        let mut min_v = 255u8;
        let mut max_v = 0u8;
        for py in 0..48 {
            for px in 0..48 {
                let v = atlas.pixels[((oy + py) * atlas.width + (ox + px)) as usize];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        assert!(min_v <= 20, "expected outside SDF <= 20, got {min_v}");
        assert!(max_v >= 180, "expected inside SDF >= 180, got {max_v}");

        let (w, h) = atlas.measure("Geist Mono", 16.0, true);
        assert!(w > 80.0 && h > 16.0);

        let mut instances = Vec::new();
        atlas.layout_into(
            "Hello\nWorld",
            10.0,
            20.0,
            14.0,
            0xffffffff,
            true,
            None,
            &mut instances,
            0,
        );
        assert_eq!(instances.len(), 10);
    }
}
