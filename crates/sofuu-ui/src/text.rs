//! Pre-baked pure-Rust Signed Distance Field (SDF) glyph atlas & text layout engine.
//!
//! Zero OS font dependencies and zero per-frame CPU rasterization:
//! At startup (`GlyphAtlas::builtin()`), analytic vector stroke segments for ASCII `0x20..=0x7E`
//! plus a fallback glyph are evaluated once into a compact `512x192` `R8` SDF texture atlas
//! (`96 KB` in GPU memory). At runtime, shaders sample the atlas with `fwidth`-based antialiasing
//! for razor-sharp, resolution-independent typography across Retina, HD, and fractional DPI displays.

use crate::draw::GlyphInstance;

const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 6;
const CELL_SIZE: u32 = 32;
/// Signed distance spread in atlas texels (`px_range` passed directly to GPU shaders).
pub const SDF_PX_RANGE: f32 = 3.5;

#[derive(Clone, Copy, Debug)]
pub struct GlyphMetrics {
    /// Normalized atlas UVs `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Advance width as a multiple of font `px_size` (proportional).
    pub advance: f32,
    /// Fixed advance width for monospace mode as a multiple of `px_size`.
    pub mono_advance: f32,
}

/// CPU-owned SDF texture atlas ready to upload once to an `R8Unorm` GPU texture.
#[derive(Clone, Debug)]
pub struct GlyphAtlas {
    pub width: u32,
    pub height: u32,
    /// Row-major `width * height` `R8` SDF bytes (`128` = contour edge, `>128` = inside).
    pub pixels: Vec<u8>,
    pub px_range: f32,
    /// Metrics for ASCII `0x20..=0x7E` (95 entries) + index `95` fallback box.
    metrics: Vec<GlyphMetrics>,
}

/// Vector primitive in a `[0..10] x [0..14]` design grid (cap height `2..11`, descender `11..13.5`).
#[derive(Clone, Copy)]
enum Prim {
    Line(f32, f32, f32, f32),
    /// Axis-aligned ellipse/circle arc: `(cx, cy, rx, ry, start_deg, end_deg)`.
    Arc(f32, f32, f32, f32, f32, f32),
}

impl GlyphAtlas {
    /// Build the default analytic vector SDF atlas once (~2ms).
    pub fn builtin() -> Self {
        let width = ATLAS_COLS * CELL_SIZE;
        let height = ATLAS_ROWS * CELL_SIZE;
        let mut pixels = vec![0u8; (width * height) as usize];
        let mut metrics = Vec::with_capacity(96);

        let inv_w = 1.0 / width as f32;
        let inv_h = 1.0 / height as f32;

        for idx in 0..96u32 {
            let ch = if idx < 95 {
                (0x20u8 + idx as u8) as char
            } else {
                '\u{FFFD}'
            };
            let col = idx % ATLAS_COLS;
            let row = idx / ATLAS_COLS;
            let ox = col * CELL_SIZE;
            let oy = row * CELL_SIZE;

            let (prims, adv_units) = glyph_vector(ch);
            bake_cell(&mut pixels, width, ox, oy, &prims);

            metrics.push(GlyphMetrics {
                uv: [
                    (ox as f32 + 1.0) * inv_w,
                    (oy as f32 + 1.0) * inv_h,
                    ((ox + CELL_SIZE) as f32 - 1.0) * inv_w,
                    ((oy + CELL_SIZE) as f32 - 1.0) * inv_h,
                ],
                advance: (adv_units / 14.0).max(0.28),
                mono_advance: 0.62,
            });
        }

        Self {
            width,
            height,
            pixels,
            px_range: SDF_PX_RANGE,
            metrics,
        }
    }

    #[inline]
    pub fn metrics(&self, ch: char) -> GlyphMetrics {
        let code = ch as u32;
        let idx = if (0x20..=0x7E).contains(&code) {
            (code - 0x20) as usize
        } else {
            95
        };
        self.metrics[idx]
    }

    /// Measure single-line or multi-line (`\n`) text block in logical pixels.
    pub fn measure(&self, text: &str, px_size: f32, mono: bool) -> (f32, f32) {
        let line_h = px_size * 1.35;
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
            let m = self.metrics(ch);
            let adv = if mono { m.mono_advance } else { m.advance };
            cur_w += adv * px_size;
        }
        max_w = max_w.max(cur_w);
        (max_w, lines as f32 * line_h)
    }

    /// Emit `GlyphInstance` quads into `out` starting at top-left `(x, y)` within optional `max_width`.
    pub fn layout_into(
        &self,
        text: &str,
        x: f32,
        y: f32,
        px_size: f32,
        color: u32,
        mono: bool,
        max_width: Option<f32>,
        out: &mut Vec<GlyphInstance>,
        clip_id: u32,
    ) {
        let quad_h = px_size * 1.25;
        let quad_w = quad_h;
        let line_h = px_size * 1.35;
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
            let adv = (if mono { m.mono_advance } else { m.advance }) * px_size;
            if let Some(limit) = max_width {
                if (pen_x - x) + adv > limit && pen_x > x {
                    pen_x = x;
                    pen_y += line_h;
                }
            }
            if ch != ' ' {
                // Center the square glyph quad horizontally around its proportional advance.
                let glyph_x = pen_x - (quad_w - adv) * 0.35;
                let glyph_y = pen_y - px_size * 0.08;
                out.push(GlyphInstance {
                    pos: [glyph_x, glyph_y],
                    size: [quad_w, quad_h],
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

fn bake_cell(buf: &mut [u8], atlas_w: u32, ox: u32, oy: u32, prims: &[Prim]) {
    // Design box [0..10, 0..14] mapped centered inside [4..28] of the 32x32 cell.
    let margin = 4.0f32;
    let span = CELL_SIZE as f32 - margin * 2.0; // 24 texels
    let scale_x = span / 14.0;
    let scale_y = span / 14.0;
    let stroke_r_texels = 1.40f32; // clean medium-weight geometric stroke (~2.8 texels diameter)

    for py in 0..CELL_SIZE {
        for px in 0..CELL_SIZE {
            let dist_texels = if prims.is_empty() {
                -SDF_PX_RANGE
            } else {
                // Map texel center to design grid coords [ -2..12, 0..14 ].
                let gx = (px as f32 + 0.5 - margin) / scale_x - 2.0;
                let gy = (py as f32 + 0.5 - margin) / scale_y;
                let mut min_d = f32::INFINITY;
                for p in prims {
                    let d = match *p {
                        Prim::Line(x0, y0, x1, y1) => dist_segment(gx, gy, x0, y0, x1, y1),
                        Prim::Arc(cx, cy, rx, ry, a0, a1) => {
                            dist_arc(gx, gy, cx, cy, rx, ry, a0, a1)
                        }
                    };
                    if d < min_d {
                        min_d = d;
                    }
                }
                let d_tex = min_d * scale_y;
                stroke_r_texels - d_tex // positive inside stroke, negative outside
            };

            let norm = (dist_texels / SDF_PX_RANGE) * 0.5 + 0.5;
            let byte = (norm.clamp(0.0, 1.0) * 255.0).round() as u8;
            let dst = ((oy + py) * atlas_w + (ox + px)) as usize;
            buf[dst] = byte;
        }
    }
}

#[inline]
fn dist_segment(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let vx = x1 - x0;
    let vy = y1 - y0;
    let wx = px - x0;
    let wy = py - y0;
    let len_sq = vx * vx + vy * vy;
    let t = if len_sq > 1e-6 {
        ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = px - (x0 + t * vx);
    let dy = py - (y0 + t * vy);
    (dx * dx + dy * dy).sqrt()
}

fn dist_arc(
    px: f32,
    py: f32,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_deg: f32,
    end_deg: f32,
) -> f32 {
    // Subdivide arc into 10 chord segments for exact capsule-strip distance without ellipse roots.
    const SEGS: usize = 10;
    let a0 = start_deg.to_radians();
    let a1 = end_deg.to_radians();
    let mut min_d = f32::INFINITY;
    let mut prev_x = cx + rx * a0.cos();
    let mut prev_y = cy + ry * a0.sin();
    for i in 1..=SEGS {
        let t = i as f32 / SEGS as f32;
        let ang = a0 + (a1 - a0) * t;
        let nx = cx + rx * ang.cos();
        let ny = cy + ry * ang.sin();
        min_d = min_d.min(dist_segment(px, py, prev_x, prev_y, nx, ny));
        prev_x = nx;
        prev_y = ny;
    }
    min_d
}

/// Clean geometric sans-serif / tabular vector glyph definitions on `[0..10] x [0..14]`.
/// Returns `(primitives, advance_in_grid_units)`.
fn glyph_vector(ch: char) -> (Vec<Prim>, f32) {
    use Prim::{Arc as A, Line as L};
    let (p, adv): (Vec<Prim>, f32) = match ch {
        ' ' => (vec![], 5.0),
        '!' => (vec![L(5., 2., 5., 8.5), L(5., 10.6, 5., 11.0)], 4.5),
        '"' => (vec![L(3.6, 2., 3.6, 4.2), L(6.4, 2., 6.4, 4.2)], 6.0),
        '#' => (
            vec![
                L(3.5, 2.5, 2.5, 11.),
                L(7.0, 2.5, 6.0, 11.),
                L(1.5, 5.2, 8.2, 5.2),
                L(1.0, 8.2, 7.7, 8.2),
            ],
            8.5,
        ),
        '$' => (
            vec![
                L(5.0, 1.3, 5.0, 12.0),
                A(5.0, 4.4, 2.5, 2.0, -30.0, -260.0),
                A(5.0, 8.6, 2.5, 2.1, -90.0, 150.0),
            ],
            8.0,
        ),
        '%' => (
            vec![
                L(2.2, 10.8, 7.8, 2.4),
                A(2.8, 3.6, 1.2, 1.2, 0.0, 360.0),
                A(7.2, 9.6, 1.2, 1.2, 0.0, 360.0),
            ],
            9.0,
        ),
        '&' => (
            vec![
                A(4.6, 4.0, 1.9, 1.8, -40.0, 260.0),
                A(4.6, 8.6, 2.6, 2.3, 45.0, 315.0),
                L(3.3, 5.3, 8.0, 11.0),
            ],
            8.5,
        ),
        '\'' => (vec![L(5.0, 2.0, 5.0, 4.4)], 4.2),
        '(' => (vec![A(6.5, 6.8, 3.0, 5.4, 115.0, 245.0)], 5.2),
        ')' => (vec![A(3.5, 6.8, 3.0, 5.4, -65.0, 65.0)], 5.2),
        '*' => (
            vec![
                L(5.0, 3.0, 5.0, 7.4),
                L(3.0, 4.1, 7.0, 6.3),
                L(7.0, 4.1, 3.0, 6.3),
            ],
            7.0,
        ),
        '+' => (vec![L(5.0, 4.0, 5.0, 9.4), L(2.3, 6.7, 7.7, 6.7)], 8.0),
        ',' => (vec![L(5.2, 10.6, 4.4, 12.6)], 4.5),
        '-' => (vec![L(2.4, 6.8, 7.6, 6.8)], 6.5),
        '.' => (vec![L(5.0, 10.7, 5.0, 11.0)], 4.5),
        '/' => (vec![L(7.2, 2.0, 2.8, 11.5)], 6.5),
        '0' => (
            vec![A(5.0, 6.5, 2.8, 4.5, 0.0, 360.0), L(3.2, 9.4, 6.8, 3.6)],
            8.2,
        ),
        '1' => (vec![L(3.4, 3.6, 5.2, 2.0), L(5.2, 2.0, 5.2, 11.0), L(3.0, 11.0, 7.2, 11.0)], 7.2),
        '2' => (
            vec![
                A(5.0, 4.4, 2.6, 2.4, 195.0, 385.0),
                L(7.4, 5.3, 2.3, 11.0),
                L(2.3, 11.0, 7.7, 11.0),
            ],
            8.0,
        ),
        '3' => (
            vec![
                A(4.8, 4.2, 2.6, 2.2, 210.0, 450.0),
                A(4.8, 8.7, 2.8, 2.3, -90.0, 160.0),
            ],
            8.0,
        ),
        '4' => (
            vec![
                L(6.6, 11.0, 6.6, 2.0),
                L(6.6, 2.0, 2.0, 8.2),
                L(2.0, 8.2, 8.2, 8.2),
            ],
            8.2,
        ),
        '5' => (
            vec![
                L(7.4, 2.0, 2.8, 2.0),
                L(2.8, 2.0, 2.6, 5.8),
                A(4.9, 8.3, 2.7, 2.7, -125.0, 155.0),
            ],
            8.0,
        ),
        '6' => (
            vec![
                A(5.2, 6.5, 2.8, 4.5, 110.0, 325.0),
                A(5.0, 8.4, 2.6, 2.6, 0.0, 360.0),
            ],
            8.0,
        ),
        '7' => (vec![L(2.2, 2.0, 7.8, 2.0), L(7.8, 2.0, 3.8, 11.0)], 8.0),
        '8' => (
            vec![
                A(5.0, 4.2, 2.3, 2.2, 0.0, 360.0),
                A(5.0, 8.7, 2.7, 2.3, 0.0, 360.0),
            ],
            8.2,
        ),
        '9' => (
            vec![
                A(5.0, 4.6, 2.6, 2.6, 0.0, 360.0),
                A(4.8, 6.5, 2.8, 4.5, -60.0, 145.0),
            ],
            8.0,
        ),
        ':' => (vec![L(5.0, 5.4, 5.0, 5.8), L(5.0, 10.6, 5.0, 11.0)], 4.5),
        ';' => (vec![L(5.0, 5.4, 5.0, 5.8), L(5.2, 10.6, 4.4, 12.6)], 4.5),
        '<' => (vec![L(7.4, 3.6, 2.6, 6.7), L(2.6, 6.7, 7.4, 9.8)], 8.0),
        '=' => (vec![L(2.2, 5.3, 7.8, 5.3), L(2.2, 8.1, 7.8, 8.1)], 8.0),
        '>' => (vec![L(2.6, 3.6, 7.4, 6.7), L(7.4, 6.7, 2.6, 9.8)], 8.0),
        '?' => (
            vec![
                A(5.0, 4.2, 2.6, 2.2, 185.0, 430.0),
                L(5.0, 6.4, 5.0, 8.2),
                L(5.0, 10.6, 5.0, 11.0),
            ],
            7.6,
        ),
        '@' => (
            vec![
                A(5.0, 6.6, 3.6, 4.2, -20.0, 310.0),
                A(4.9, 6.8, 1.7, 1.9, 0.0, 360.0),
                L(6.6, 5.2, 6.6, 8.5),
            ],
            9.5,
        ),
        'A' => (vec![L(1.8, 11.0, 5.0, 2.0), L(5.0, 2.0, 8.2, 11.0), L(2.9, 8.0, 7.1, 8.0)], 8.8),
        'B' => (
            vec![
                L(2.4, 2.0, 2.4, 11.0),
                L(2.4, 2.0, 5.4, 2.0),
                L(2.4, 6.3, 5.6, 6.3),
                L(2.4, 11.0, 5.6, 11.0),
                A(5.4, 4.15, 2.2, 2.15, -90.0, 90.0),
                A(5.6, 8.65, 2.4, 2.35, -90.0, 90.0),
            ],
            8.6,
        ),
        'C' => (vec![A(5.6, 6.5, 3.4, 4.5, 42.0, 318.0)], 8.6),
        'D' => (
            vec![
                L(2.4, 2.0, 2.4, 11.0),
                L(2.4, 2.0, 4.8, 2.0),
                L(2.4, 11.0, 4.8, 11.0),
                A(4.8, 6.5, 3.4, 4.5, -90.0, 90.0),
            ],
            8.8,
        ),
        'E' => (
            vec![
                L(2.6, 2.0, 2.6, 11.0),
                L(2.6, 2.0, 7.6, 2.0),
                L(2.6, 6.3, 6.8, 6.3),
                L(2.6, 11.0, 7.6, 11.0),
            ],
            8.0,
        ),
        'F' => (
            vec![L(2.6, 2.0, 2.6, 11.0), L(2.6, 2.0, 7.6, 2.0), L(2.6, 6.3, 6.8, 6.3)],
            7.8,
        ),
        'G' => (
            vec![
                A(5.4, 6.5, 3.4, 4.5, 25.0, 325.0),
                L(5.4, 6.8, 8.4, 6.8),
                L(8.4, 6.8, 8.4, 10.6),
            ],
            9.0,
        ),
        'H' => (vec![L(2.2, 2.0, 2.2, 11.0), L(7.8, 2.0, 7.8, 11.0), L(2.2, 6.3, 7.8, 6.3)], 9.0),
        'I' => (vec![L(5.0, 2.0, 5.0, 11.0), L(3.2, 2.0, 6.8, 2.0), L(3.2, 11.0, 6.8, 11.0)], 5.6),
        'J' => (
            vec![
                L(6.8, 2.0, 6.8, 8.6),
                A(4.6, 8.6, 2.2, 2.4, 0.0, 165.0),
            ],
            7.4,
        ),
        'K' => (vec![L(2.4, 2.0, 2.4, 11.0), L(7.8, 2.0, 2.6, 6.8), L(3.8, 5.8, 8.0, 11.0)], 8.6),
        'L' => (vec![L(2.6, 2.0, 2.6, 11.0), L(2.6, 11.0, 7.6, 11.0)], 7.6),
        'M' => (
            vec![
                L(1.6, 11.0, 1.6, 2.0),
                L(1.6, 2.0, 5.0, 7.8),
                L(5.0, 7.8, 8.4, 2.0),
                L(8.4, 2.0, 8.4, 11.0),
            ],
            9.8,
        ),
        'N' => (vec![L(2.2, 11.0, 2.2, 2.0), L(2.2, 2.0, 7.8, 11.0), L(7.8, 11.0, 7.8, 2.0)], 9.0),
        'O' => (vec![A(5.0, 6.5, 3.3, 4.5, 0.0, 360.0)], 9.2),
        'P' => (
            vec![
                L(2.4, 2.0, 2.4, 11.0),
                L(2.4, 2.0, 5.4, 2.0),
                L(2.4, 6.6, 5.4, 6.6),
                A(5.4, 4.3, 2.3, 2.3, -90.0, 90.0),
            ],
            8.2,
        ),
        'Q' => (vec![A(5.0, 6.5, 3.3, 4.5, 0.0, 360.0), L(5.8, 8.4, 8.5, 11.6)], 9.2),
        'R' => (
            vec![
                L(2.4, 2.0, 2.4, 11.0),
                L(2.4, 2.0, 5.4, 2.0),
                L(2.4, 6.5, 5.4, 6.5),
                A(5.4, 4.25, 2.25, 2.25, -90.0, 90.0),
                L(5.2, 6.5, 7.8, 11.0),
            ],
            8.6,
        ),
        'S' => (
            vec![
                A(5.0, 4.2, 2.5, 2.2, -30.0, -260.0),
                A(5.0, 8.7, 2.6, 2.3, -90.0, 150.0),
            ],
            8.2,
        ),
        'T' => (vec![L(1.6, 2.0, 8.4, 2.0), L(5.0, 2.0, 5.0, 11.0)], 8.2),
        'U' => (
            vec![
                L(2.2, 2.0, 2.2, 8.2),
                L(7.8, 2.0, 7.8, 8.2),
                A(5.0, 8.2, 2.8, 2.8, 0.0, 180.0),
            ],
            8.8,
        ),
        'V' => (vec![L(1.8, 2.0, 5.0, 11.0), L(5.0, 11.0, 8.2, 2.0)], 8.6),
        'W' => (
            vec![
                L(1.2, 2.0, 3.1, 11.0),
                L(3.1, 11.0, 5.0, 4.8),
                L(5.0, 4.8, 6.9, 11.0),
                L(6.9, 11.0, 8.8, 2.0),
            ],
            10.2,
        ),
        'X' => (vec![L(2.0, 2.0, 8.0, 11.0), L(8.0, 2.0, 2.0, 11.0)], 8.4),
        'Y' => (
            vec![L(2.0, 2.0, 5.0, 6.6), L(8.0, 2.0, 5.0, 6.6), L(5.0, 6.6, 5.0, 11.0)],
            8.2,
        ),
        'Z' => (
            vec![L(2.0, 2.0, 8.0, 2.0), L(8.0, 2.0, 2.0, 11.0), L(2.0, 11.0, 8.0, 11.0)],
            8.2,
        ),
        '[' => (vec![L(6.6, 1.6, 4.0, 1.6), L(4.0, 1.6, 4.0, 12.0), L(4.0, 12.0, 6.6, 12.0)], 5.6),
        '\\' => (vec![L(2.8, 2.0, 7.2, 11.5)], 6.5),
        ']' => (vec![L(3.4, 1.6, 6.0, 1.6), L(6.0, 1.6, 6.0, 12.0), L(6.0, 12.0, 3.4, 12.0)], 5.6),
        '^' => (vec![L(2.5, 5.6, 5.0, 2.2), L(5.0, 2.2, 7.5, 5.6)], 7.6),
        '_' => (vec![L(1.5, 12.2, 8.5, 12.2)], 7.8),
        '`' => (vec![L(4.2, 1.8, 5.8, 3.6)], 5.0),
        'a' => (
            vec![
                A(4.7, 8.0, 2.5, 3.0, 30.0, 330.0),
                L(7.2, 5.2, 7.2, 11.0),
            ],
            7.8,
        ),
        'b' => (
            vec![L(2.6, 1.8, 2.6, 11.0), A(5.2, 8.0, 2.6, 3.0, -145.0, 145.0)],
            8.0,
        ),
        'c' => (vec![A(5.4, 8.0, 2.8, 3.0, 42.0, 318.0)], 7.4),
        'd' => (
            vec![L(7.4, 1.8, 7.4, 11.0), A(4.8, 8.0, 2.6, 3.0, 35.0, 325.0)],
            8.0,
        ),
        'e' => (
            vec![
                L(2.3, 7.8, 7.6, 7.8),
                A(5.0, 8.0, 2.7, 3.0, 35.0, 360.0),
            ],
            7.8,
        ),
        'f' => (
            vec![
                A(6.3, 3.4, 1.8, 1.8, 180.0, 325.0),
                L(4.5, 3.4, 4.5, 11.0),
                L(2.6, 5.4, 6.8, 5.4),
            ],
            5.8,
        ),
        'g' => (
            vec![
                A(4.8, 7.8, 2.5, 2.8, 0.0, 360.0),
                L(7.3, 5.2, 7.3, 11.2),
                A(5.0, 11.2, 2.3, 2.1, 0.0, 155.0),
            ],
            8.0,
        ),
        'h' => (
            vec![
                L(2.6, 1.8, 2.6, 11.0),
                A(5.0, 7.2, 2.4, 2.2, 180.0, 360.0),
                L(7.4, 7.2, 7.4, 11.0),
            ],
            7.8,
        ),
        'i' => (vec![L(5.0, 2.4, 5.0, 2.8), L(5.0, 5.0, 5.0, 11.0)], 4.2),
        'j' => (
            vec![
                L(6.2, 2.4, 6.2, 2.8),
                L(6.2, 5.0, 6.2, 11.2),
                A(4.4, 11.2, 1.8, 2.0, 0.0, 160.0),
            ],
            4.8,
        ),
        'k' => (vec![L(2.6, 1.8, 2.6, 11.0), L(7.2, 5.0, 2.8, 8.4), L(4.2, 7.4, 7.4, 11.0)], 7.4),
        'l' => (vec![L(5.0, 1.8, 5.0, 11.0)], 4.2),
        'm' => (
            vec![
                L(1.5, 5.0, 1.5, 11.0),
                A(3.25, 6.9, 1.75, 1.9, 180.0, 360.0),
                L(5.0, 6.9, 5.0, 11.0),
                A(6.75, 6.9, 1.75, 1.9, 180.0, 360.0),
                L(8.5, 6.9, 8.5, 11.0),
            ],
            10.2,
        ),
        'n' => (
            vec![
                L(2.6, 5.0, 2.6, 11.0),
                A(5.0, 7.2, 2.4, 2.2, 180.0, 360.0),
                L(7.4, 7.2, 7.4, 11.0),
            ],
            7.8,
        ),
        'o' => (vec![A(5.0, 8.0, 2.7, 3.0, 0.0, 360.0)], 8.0),
        'p' => (
            vec![L(2.6, 5.0, 2.6, 13.4), A(5.2, 8.0, 2.6, 3.0, -145.0, 145.0)],
            8.0,
        ),
        'q' => (
            vec![L(7.4, 5.0, 7.4, 13.4), A(4.8, 8.0, 2.6, 3.0, 35.0, 325.0)],
            8.0,
        ),
        'r' => (vec![L(3.2, 5.0, 3.2, 11.0), A(5.5, 7.0, 2.3, 2.0, 180.0, 330.0)], 6.0),
        's' => (
            vec![
                A(5.0, 6.4, 2.1, 1.5, -25.0, -255.0),
                A(5.0, 9.5, 2.2, 1.6, -90.0, 155.0),
            ],
            7.2,
        ),
        't' => (
            vec![L(4.6, 2.4, 4.6, 9.4), L(2.6, 5.2, 6.8, 5.2), A(5.8, 9.4, 1.2, 1.4, 50.0, 180.0)],
            5.8,
        ),
        'u' => (
            vec![
                L(2.6, 5.0, 2.6, 8.8),
                L(7.4, 5.0, 7.4, 11.0),
                A(5.0, 8.8, 2.4, 2.2, 0.0, 180.0),
            ],
            7.8,
        ),
        'v' => (vec![L(2.2, 5.0, 5.0, 11.0), L(5.0, 11.0, 7.8, 5.0)], 7.4),
        'w' => (
            vec![
                L(1.5, 5.0, 3.2, 11.0),
                L(3.2, 11.0, 5.0, 6.2),
                L(5.0, 6.2, 6.8, 11.0),
                L(6.8, 11.0, 8.5, 5.0),
            ],
            9.6,
        ),
        'x' => (vec![L(2.4, 5.0, 7.6, 11.0), L(7.6, 5.0, 2.4, 11.0)], 7.4),
        'y' => (
            vec![
                L(2.2, 5.0, 5.0, 10.6),
                L(7.8, 5.0, 4.0, 13.4),
            ],
            7.4,
        ),
        'z' => (
            vec![L(2.4, 5.0, 7.6, 5.0), L(7.6, 5.0, 2.4, 11.0), L(2.4, 11.0, 7.6, 11.0)],
            7.4,
        ),
        '{' => (
            vec![
                A(6.4, 3.2, 1.8, 1.6, 180.0, 270.0),
                L(4.6, 3.2, 4.6, 5.8),
                L(4.6, 5.8, 3.2, 6.8),
                L(3.2, 6.8, 4.6, 7.8),
                L(4.6, 7.8, 4.6, 10.4),
                A(6.4, 10.4, 1.8, 1.6, 90.0, 180.0),
            ],
            5.8,
        ),
        '|' => (vec![L(5.0, 1.6, 5.0, 12.2)], 4.4),
        '}' => (
            vec![
                A(3.6, 3.2, 1.8, 1.6, -90.0, 0.0),
                L(5.4, 3.2, 5.4, 5.8),
                L(5.4, 5.8, 6.8, 6.8),
                L(6.8, 6.8, 5.4, 7.8),
                L(5.4, 7.8, 5.4, 10.4),
                A(3.6, 10.4, 1.8, 1.6, 0.0, 90.0),
            ],
            5.8,
        ),
        '~' => (
            vec![
                A(3.5, 6.8, 1.5, 1.1, 190.0, 350.0),
                A(6.5, 6.8, 1.5, 1.1, 10.0, 170.0),
            ],
            7.8,
        ),
        // Fallback box for unknown / non-ASCII codepoints
        _ => (
            vec![
                L(2.4, 2.2, 7.6, 2.2),
                L(7.6, 2.2, 7.6, 10.8),
                L(7.6, 10.8, 2.4, 10.8),
                L(2.4, 10.8, 2.4, 2.2),
            ],
            8.2,
        ),
    };
    (p, adv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_atlas_has_valid_sdf_range_and_metrics() {
        let atlas = GlyphAtlas::builtin();
        assert_eq!(atlas.width, 512);
        assert_eq!(atlas.height, 192);
        assert_eq!(atlas.pixels.len(), (512 * 192) as usize);

        // Verify glyph 'A' (index 33) has both strong interior (>180) and exterior (<60) SDF values.
        let idx = ('A' as u32) - 0x20;
        let ox = (idx % ATLAS_COLS) * CELL_SIZE;
        let oy = (idx / ATLAS_COLS) * CELL_SIZE;
        let mut min_v = 255u8;
        let mut max_v = 0u8;
        for py in 0..CELL_SIZE {
            for px in 0..CELL_SIZE {
                let v = atlas.pixels[((oy + py) * atlas.width + (ox + px)) as usize];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        assert!(min_v <= 50, "expected outside SDF <= 50, got {min_v}");
        assert!(max_v >= 175, "expected inside SDF >= 175, got {max_v}");

        let (w, h) = atlas.measure("Sofuu UI", 16.0, false);
        assert!(w > 40.0 && h > 16.0);

        let mut instances = Vec::new();
        atlas.layout_into("Hello\nWorld", 10.0, 20.0, 14.0, 0xffffffff, false, None, &mut instances, 0);
        assert_eq!(instances.len(), 10); // space skipped in HelloWorld (10 visible glyphs)
    }
}
