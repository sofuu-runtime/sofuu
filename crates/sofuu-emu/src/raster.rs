//! Deterministic software rasterizer for `sofuu_ui::DrawList`.
//!
//! This is the *headless* path: no `NSWindow`, `wl_surface`, `HWND`, swapchain
//! or screenshot permission. It mirrors the five GPU shaders closely enough
//! that PNGs are visually faithful (rounded SDF quads, gradients, borders,
//! soft shadows, ribbons, AA strokes, SDF glyphs) while remaining available in
//! every CI container. Native GPU offscreen backends can be layered on later
//! without changing the screenshot/audit contract.
//!
//! Metadata returned by `screenshot.rs` reports `renderer: "software"` so
//! callers can tell which path produced the pixels.

use sofuu_ui::draw::{
    unpack_rgba8, DrawList, GlyphInstance, QuadInstance, RibbonInstance, SpriteInstance,
    StrokeInstance, DRAW_KIND_GLYPH, DRAW_KIND_QUAD, DRAW_KIND_RIBBON, DRAW_KIND_SPRITE,
    DRAW_KIND_STROKE, SPRITE_FLAG_SDF,
};
use sofuu_ui::icons::IconAtlas;
use sofuu_ui::text::GlyphAtlas;

use crate::image::RgbaImage;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RasterStats {
    pub quads: usize,
    pub ribbons: usize,
    pub strokes: usize,
    pub glyphs: usize,
    pub sprites: usize,
    pub render_ms: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    /// Logical-space offset (e.g. the app viewport inside the device screen).
    pub offset: [f32; 2],
    /// Physical pixels per logical pixel.
    pub scale: f32,
}

impl Transform {
    pub fn identity(scale: f32) -> Self {
        Self {
            offset: [0.0, 0.0],
            scale,
        }
    }
    #[inline]
    fn px(&self, logical: f32, axis: usize) -> f32 {
        (logical + self.offset[axis]) * self.scale
    }
    #[inline]
    fn logical(&self, px: f32, axis: usize) -> f32 {
        px / self.scale - self.offset[axis]
    }
}

/// Rasterize into a fresh image of `viewport * scale`.
pub fn rasterize(
    dl: &DrawList,
    atlas: &GlyphAtlas,
    viewport: [f32; 2],
    scale: f32,
    background: [u8; 4],
) -> (RgbaImage, RasterStats) {
    let scale = scale.max(0.25);
    let width = (viewport[0] * scale).round().max(1.0) as u32;
    let height = (viewport[1] * scale).round().max(1.0) as u32;
    let mut img = RgbaImage::new(width, height, background);
    let stats = rasterize_into(&mut img, dl, atlas, Transform::identity(scale));
    (img, stats)
}

/// Composite a `DrawList` over an existing image.
pub fn rasterize_into(
    img: &mut RgbaImage,
    dl: &DrawList,
    atlas: &GlyphAtlas,
    xf: Transform,
) -> RasterStats {
    let start = std::time::Instant::now();
    let mut stats = RasterStats::default();
    let mut any = false;
    for cmd in &dl.cmds {
        any = true;
        let a = cmd.start as usize;
        let b = (cmd.start + cmd.count) as usize;
        match cmd.kind {
            DRAW_KIND_QUAD => {
                for q in dl.quads.get(a..b).unwrap_or(&[]) {
                    draw_quad(img, dl, q, xf);
                    stats.quads += 1;
                }
            }
            DRAW_KIND_RIBBON => {
                for r in dl.ribbons.get(a..b).unwrap_or(&[]) {
                    draw_ribbon(img, dl, r, xf);
                    stats.ribbons += 1;
                }
            }
            DRAW_KIND_STROKE => {
                for s in dl.strokes.get(a..b).unwrap_or(&[]) {
                    draw_stroke(img, dl, s, xf);
                    stats.strokes += 1;
                }
            }
            DRAW_KIND_GLYPH => {
                for g in dl.glyphs.get(a..b).unwrap_or(&[]) {
                    draw_glyph(img, dl, g, atlas, xf);
                    stats.glyphs += 1;
                }
            }
            DRAW_KIND_SPRITE => {
                for sp in dl.sprites.get(a..b).unwrap_or(&[]) {
                    draw_sprite(img, dl, sp, xf);
                    stats.sprites += 1;
                }
            }
            _ => {}
        }
    }
    if !any {
        for q in &dl.quads {
            draw_quad(img, dl, q, xf);
            stats.quads += 1;
        }
        for r in &dl.ribbons {
            draw_ribbon(img, dl, r, xf);
            stats.ribbons += 1;
        }
        for s in &dl.strokes {
            draw_stroke(img, dl, s, xf);
            stats.strokes += 1;
        }
        for g in &dl.glyphs {
            draw_glyph(img, dl, g, atlas, xf);
            stats.glyphs += 1;
        }
        for sp in &dl.sprites {
            draw_sprite(img, dl, sp, xf);
            stats.sprites += 1;
        }
    }
    stats.render_ms = start.elapsed().as_secs_f64() * 1000.0;
    stats
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 - e0).abs() < 1e-6 {
        return if x < e0 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn sd_rounded_box(p: [f32; 2], half: [f32; 2], radii: [f32; 4]) -> f32 {
    let r_side = if p[0] > 0.0 {
        [radii[1], radii[2]]
    } else {
        [radii[0], radii[3]]
    };
    let mut r = if p[1] > 0.0 { r_side[1] } else { r_side[0] };
    r = r.clamp(0.0, half[0].min(half[1]));
    let q = [p[0].abs() - half[0] + r, p[1].abs() - half[1] + r];
    let outside = [q[0].max(0.0), q[1].max(0.0)];
    q[0].max(q[1]).min(0.0) + (outside[0] * outside[0] + outside[1] * outside[1]).sqrt() - r
}

#[inline]
fn clip_ok(dl: &DrawList, clip_id: u32, lx: f32, ly: f32) -> bool {
    match dl.clips.get(clip_id as usize) {
        Some(c) => lx >= c.x && lx <= c.x + c.w && ly >= c.y && ly <= c.y + c.h,
        None => true,
    }
}

#[inline]
fn pix_range(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    xf: Transform,
    img: &RgbaImage,
) -> (u32, u32, u32, u32) {
    let x0 = xf.px(x, 0).floor().max(0.0) as u32;
    let y0 = xf.px(y, 1).floor().max(0.0) as u32;
    let x1 = xf
        .px(x + w, 0)
        .ceil()
        .min(img.width as f32)
        .max(0.0) as u32;
    let y1 = xf
        .px(y + h, 1)
        .ceil()
        .min(img.height as f32)
        .max(0.0) as u32;
    (x0.min(img.width), y0.min(img.height), x1, y1)
}

fn draw_quad(img: &mut RgbaImage, dl: &DrawList, q: &QuadInstance, xf: Transform) {
    let full = unpack_rgba8(q.color_top);
    let full_b = unpack_rgba8(q.color_bottom);
    let border = unpack_rgba8(q.border_color);
    let shadow = unpack_rgba8(q.shadow_color);
    let opacity = q.params[3].clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return;
    }
    let blur = q.params[1].max(0.0);
    let off_y = q.params[2];
    let margin = if blur > 0.01 {
        blur + off_y.abs() + 1.5
    } else {
        1.0
    };
    let (x0, y0, x1, y1) = pix_range(
        q.rect[0] - margin,
        q.rect[1] - margin,
        q.rect[2] + margin * 2.0,
        q.rect[3] + margin * 2.0,
        xf,
        img,
    );
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let half = [q.rect[2] * 0.5, q.rect[3] * 0.5];
    let center = [q.rect[0] + half[0], q.rect[1] + half[1]];
    let aa = 0.5 / xf.scale.max(1.0);
    let border_w = q.params[0].max(0.0);
    let grad_h = q.rect[3].max(1.0);
    let shadow_a = shadow[3] as f32 / 255.0;
    let border_a = border[3] as f32 / 255.0;

    for py in y0..y1 {
        let ly = xf.logical(py as f32 + 0.5, 1);
        for px in x0..x1 {
            let lx = xf.logical(px as f32 + 0.5, 0);
            if !clip_ok(dl, q.clip_id, lx, ly) {
                continue;
            }
            let p = [lx - center[0], ly - center[1]];
            let dist = sd_rounded_box(p, half, q.radii);

            if blur > 0.01 && shadow_a > 0.001 {
                let sd_s = sd_rounded_box([p[0], p[1] - off_y], half, q.radii);
                let sh = 1.0 - smoothstep(-blur * 0.25, blur, sd_s);
                let a = shadow_a * sh * sh * opacity;
                if a > 0.001 {
                    img.blend(
                        px,
                        py,
                        [
                            shadow[0] as f32 / 255.0,
                            shadow[1] as f32 / 255.0,
                            shadow[2] as f32 / 255.0,
                        ],
                        a,
                    );
                }
            }

            let fill_cov = 1.0 - smoothstep(-aa, aa, dist);
            if fill_cov <= 0.001 {
                continue;
            }
            let t = ((ly - q.rect[1]) / grad_h).clamp(0.0, 1.0);
            let mut rgb = [
                (full[0] as f32 * (1.0 - t) + full_b[0] as f32 * t) / 255.0,
                (full[1] as f32 * (1.0 - t) + full_b[1] as f32 * t) / 255.0,
                (full[2] as f32 * (1.0 - t) + full_b[2] as f32 * t) / 255.0,
            ];
            let fill_a = (full[3] as f32 * (1.0 - t) + full_b[3] as f32 * t) / 255.0;
            let mut a = fill_a * fill_cov;
            if border_w > 0.01 && border_a > 0.001 {
                let inner_cov = 1.0 - smoothstep(-aa, aa, dist + border_w);
                let bc = (fill_cov - inner_cov).clamp(0.0, 1.0) * border_a;
                if bc > 0.001 {
                    rgb = [
                        rgb[0] * (1.0 - bc) + border[0] as f32 / 255.0 * bc,
                        rgb[1] * (1.0 - bc) + border[1] as f32 / 255.0 * bc,
                        rgb[2] * (1.0 - bc) + border[2] as f32 / 255.0 * bc,
                    ];
                    a = a.max(bc);
                }
            }
            img.blend(px, py, rgb, a * opacity);
        }
    }
}

fn draw_ribbon(img: &mut RgbaImage, dl: &DrawList, r: &RibbonInstance, xf: Transform) {
    let top = unpack_rgba8(r.color_top);
    let bot = unpack_rgba8(r.color_bottom);
    let x0 = r.x0.min(r.x1);
    let x1 = r.x0.max(r.x1);
    let ymin = r.y0_top.min(r.y0_bot).min(r.y1_top).min(r.y1_bot);
    let ymax = r.y0_top.max(r.y0_bot).max(r.y1_top).max(r.y1_bot);
    let (px0, py0, px1, py1) = pix_range(x0, ymin, x1 - x0, ymax - ymin, xf, img);
    if px0 >= px1 || py0 >= py1 {
        return;
    }
    let dx = r.x1 - r.x0;
    if dx.abs() < 1e-6 {
        return;
    }
    let aa = 0.75;
    for py in py0..py1 {
        let ly = xf.logical(py as f32 + 0.5, 1);
        for px in px0..px1 {
            let lx = xf.logical(px as f32 + 0.5, 0);
            if !clip_ok(dl, r.clip_id, lx, ly) {
                continue;
            }
            let t = ((lx - r.x0) / dx).clamp(0.0, 1.0);
            let yt = r.y0_top + (r.y1_top - r.y0_top) * t;
            let yb = r.y0_bot + (r.y1_bot - r.y0_bot) * t;
            let (lo, hi) = if yt < yb { (yt, yb) } else { (yb, yt) };
            let edge = smoothstep(lo - aa, lo + aa, ly)
                * (1.0 - smoothstep(hi - aa, hi + aa, ly));
            if edge <= 0.001 {
                continue;
            }
            let g = if (hi - lo).abs() < 1e-6 {
                0.0
            } else {
                ((ly - yt) / (yb - yt)).clamp(0.0, 1.0)
            };
            let rgb = [
                (top[0] as f32 * (1.0 - g) + bot[0] as f32 * g) / 255.0,
                (top[1] as f32 * (1.0 - g) + bot[1] as f32 * g) / 255.0,
                (top[2] as f32 * (1.0 - g) + bot[2] as f32 * g) / 255.0,
            ];
            let a = (top[3] as f32 * (1.0 - g) + bot[3] as f32 * g) / 255.0;
            img.blend(px, py, rgb, a * edge);
        }
    }
}

fn draw_stroke(img: &mut RgbaImage, dl: &DrawList, s: &StrokeInstance, xf: Transform) {
    let half_w = (s.width * 0.5).max(0.35);
    let pad = half_w + s.feather.max(1.0) + 2.0;
    let x0 = s.p0[0].min(s.p1[0]) - pad;
    let y0 = s.p0[1].min(s.p1[1]) - pad;
    let w = (s.p0[0].max(s.p1[0]) + pad) - x0;
    let h = (s.p0[1].max(s.p1[1]) + pad) - y0;
    let (px0, py0, px1, py1) = pix_range(x0, y0, w, h, xf, img);
    if px0 >= px1 || py0 >= py1 {
        return;
    }
    let c = unpack_rgba8(s.color);
    let feather = s.feather.max(0.75 / xf.scale.max(1.0));
    let ba = [s.p1[0] - s.p0[0], s.p1[1] - s.p0[1]];
    let len_sq = (ba[0] * ba[0] + ba[1] * ba[1]).max(1e-6);
    let len = len_sq.sqrt();
    for py in py0..py1 {
        let ly = xf.logical(py as f32 + 0.5, 1);
        for px in px0..px1 {
            let lx = xf.logical(px as f32 + 0.5, 0);
            if !clip_ok(dl, s.clip_id, lx, ly) {
                continue;
            }
            let pa = [lx - s.p0[0], ly - s.p0[1]];
            let hseg = ((pa[0] * ba[0] + pa[1] * ba[1]) / len_sq).clamp(0.0, 1.0);
            let ddx = pa[0] - ba[0] * hseg;
            let ddy = pa[1] - ba[1] * hseg;
            let dist = (ddx * ddx + ddy * ddy).sqrt() - half_w;
            let mut cov = 1.0 - smoothstep(-feather, feather, dist);
            if cov <= 0.001 {
                continue;
            }
            if s.dash_len > 0.01 && s.gap_len > 0.01 {
                let along = hseg * len;
                let period = s.dash_len + s.gap_len;
                let phase = along % period;
                let dash = smoothstep(0.0, 0.75, phase)
                    * (1.0 - smoothstep(s.dash_len - 0.75, s.dash_len + 0.25, phase));
                cov *= dash;
            }
            let a = c[3] as f32 / 255.0 * cov;
            img.blend(
                px,
                py,
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ],
                a,
            );
        }
    }
}

fn draw_glyph(
    img: &mut RgbaImage,
    dl: &DrawList,
    g: &GlyphInstance,
    atlas: &GlyphAtlas,
    xf: Transform,
) {
    if g.size[0] <= 0.1 || g.size[1] <= 0.1 {
        return;
    }
    let (px0, py0, px1, py1) = pix_range(g.pos[0], g.pos[1], g.size[0], g.size[1], xf, img);
    if px0 >= px1 || py0 >= py1 {
        return;
    }
    let c = unpack_rgba8(g.color);
    let du = ((g.uv[2] - g.uv[0]) / g.size[0]).abs().max(1e-6);
    let dv = ((g.uv[3] - g.uv[1]) / g.size[1]).abs().max(1e-6);
    let range = 0.5
        * ((g.px_range / atlas.width.max(1) as f32) / du
            + (g.px_range / atlas.height.max(1) as f32) / dv);
    let range = range.max(1.0);
    for py in py0..py1 {
        let ly = xf.logical(py as f32 + 0.5, 1);
        for px in px0..px1 {
            let lx = xf.logical(px as f32 + 0.5, 0);
            if !clip_ok(dl, g.clip_id, lx, ly) {
                continue;
            }
            let u = g.uv[0]
                + (g.uv[2] - g.uv[0]) * ((lx - g.pos[0]) / g.size[0]).clamp(0.0, 1.0);
            let v = g.uv[1]
                + (g.uv[3] - g.uv[1]) * ((ly - g.pos[1]) / g.size[1]).clamp(0.0, 1.0);
            let sd = sample_atlas(atlas, u, v) / 255.0 - 0.5;
            let cov = smoothstep(-0.48, 0.44, sd * range);
            let a = c[3] as f32 / 255.0 * cov;
            if a > 0.002 {
                img.blend(
                    px,
                    py,
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ],
                    a,
                );
            }
        }
    }
}

/// Software sprite pass: SDF icons sample the built-in icon atlas. Bitmap
/// images (`tex_id >= 1`) are host-uploaded textures, so the headless path
/// leaves them to the GPU renderer instead of inventing pixels.
fn draw_sprite(img: &mut RgbaImage, dl: &DrawList, s: &SpriteInstance, xf: Transform) {
    if s.tex_id != 0 || s.flags & SPRITE_FLAG_SDF == 0 {
        return;
    }
    let atlas = IconAtlas::builtin();
    let (px0, py0, px1, py1) = pix_range(s.rect[0], s.rect[1], s.rect[2], s.rect[3], xf, img);
    if px0 >= px1 || py0 >= py1 {
        return;
    }
    let tint = unpack_rgba8(s.tint);
    let half = [s.rect[2] * 0.5, s.rect[3] * 0.5];
    let center = [s.rect[0] + half[0], s.rect[1] + half[1]];
    let aa = 0.5 / xf.scale.max(1.0);
    let du = ((s.uv[2] - s.uv[0]) / s.rect[2]).abs().max(1e-6);
    let dv = ((s.uv[3] - s.uv[1]) / s.rect[3]).abs().max(1e-6);
    let range = 0.5
        * ((atlas.px_range / atlas.width.max(1) as f32) / du
            + (atlas.px_range / atlas.height.max(1) as f32) / dv);
    let range = range.max(1.0);
    for py in py0..py1 {
        let ly = xf.logical(py as f32 + 0.5, 1);
        for px in px0..px1 {
            let lx = xf.logical(px as f32 + 0.5, 0);
            if !clip_ok(dl, s.clip_id, lx, ly) {
                continue;
            }
            let p = [lx - center[0], ly - center[1]];
            let shape = 1.0 - smoothstep(-aa, aa, sd_rounded_box(p, half, s.radii));
            if shape <= 0.001 {
                continue;
            }
            let u = s.uv[0]
                + (s.uv[2] - s.uv[0]) * ((lx - s.rect[0]) / s.rect[2]).clamp(0.0, 1.0);
            let v = s.uv[1]
                + (s.uv[3] - s.uv[1]) * ((ly - s.rect[1]) / s.rect[3]).clamp(0.0, 1.0);
            let sd = sample_icon(&atlas, u, v) / 255.0 - 0.5;
            let cov = smoothstep(-0.48, 0.44, sd * range) * shape;
            let a = tint[3] as f32 / 255.0 * cov;
            if a > 0.002 {
                img.blend(
                    px,
                    py,
                    [
                        tint[0] as f32 / 255.0,
                        tint[1] as f32 / 255.0,
                        tint[2] as f32 / 255.0,
                    ],
                    a,
                );
            }
        }
    }
}

fn sample_icon(atlas: &IconAtlas, u: f32, v: f32) -> f32 {
    let w = atlas.width.max(1);
    let h = atlas.height.max(1);
    let x = (u.clamp(0.0, 1.0) * (w - 1) as f32).max(0.0);
    let y = (v.clamp(0.0, 1.0) * (h - 1) as f32).max(0.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |xx: u32, yy: u32| atlas.pixels[(yy * w + xx) as usize] as f32;
    let a = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let b = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    a * (1.0 - fy) + b * fy
}

#[inline]
fn sample_atlas(atlas: &GlyphAtlas, u: f32, v: f32) -> f32 {
    let w = atlas.width.max(1);
    let h = atlas.height.max(1);
    let x = (u.clamp(0.0, 1.0) * (w - 1) as f32).max(0.0);
    let y = (v.clamp(0.0, 1.0) * (h - 1) as f32).max(0.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |xx: u32, yy: u32| atlas.pixels[(yy * w + xx) as usize] as f32;
    let a = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let b = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    a * (1.0 - fy) + b * fy
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::draw::{hex_rgba, ClipRect, DrawCmd, DRAW_KIND_QUAD};

    fn dl_with_quad() -> (DrawList, GlyphAtlas) {
        let mut dl = DrawList::new();
        dl.reset([100.0, 60.0]);
        dl.push_quad(QuadInstance {
            rect: [10.0, 10.0, 80.0, 40.0],
            radii: [8.0; 4],
            color_top: hex_rgba(0x3366FF, 255),
            color_bottom: hex_rgba(0x3366FF, 255),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
        (dl, GlyphAtlas::builtin())
    }

    #[test]
    fn rasterizes_a_rounded_quad_at_scale() {
        let (dl, atlas) = dl_with_quad();
        let (img, stats) = rasterize(&dl, &atlas, [100.0, 60.0], 2.0, [0, 0, 0, 255]);
        assert_eq!(img.width, 200);
        assert_eq!(img.height, 120);
        assert_eq!(stats.quads, 1);
        let center = img.get(100, 60);
        assert!(center[2] > 200 && center[0] < 80, "center={center:?}");
        let corner = img.get(22, 22);
        assert!(corner[2] < 40, "corner={corner:?}");
    }

    #[test]
    fn glyphs_use_the_embedded_sdf_atlas() {
        let atlas = GlyphAtlas::builtin();
        let mut dl = DrawList::new();
        dl.reset([80.0, 40.0]);
        atlas.layout_dl(
            "AB",
            8.0,
            8.0,
            18.0,
            hex_rgba(0xFFFFFF, 255),
            false,
            None,
            &mut dl,
        );
        let (img, stats) = rasterize(&dl, &atlas, [80.0, 40.0], 1.0, [0, 0, 0, 255]);
        assert!(stats.glyphs >= 2);
        assert!(img.pixels.chunks_exact(4).any(|p| p[0] > 180));
    }

    #[test]
    fn clips_are_respected() {
        let (mut dl, atlas) = dl_with_quad();
        dl.quads.clear();
        dl.cmds.clear();
        let clip = dl.push_clip(ClipRect::new(0.0, 0.0, 20.0, 20.0));
        let idx = dl.quads.len();
        dl.quads.push(QuadInstance {
            rect: [0.0, 0.0, 100.0, 60.0],
            radii: [0.0; 4],
            color_top: hex_rgba(0xFF0000, 255),
            color_bottom: hex_rgba(0xFF0000, 255),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: clip,
            _pad: [0; 3],
        });
        dl.cmds.push(DrawCmd {
            kind: DRAW_KIND_QUAD,
            start: idx as u32,
            count: 1,
            _pad: 0,
        });
        let (img, _) = rasterize(&dl, &atlas, [100.0, 60.0], 1.0, [0, 0, 0, 255]);
        let inside = img.get(10, 10);
        let outside = img.get(40, 40);
        assert!(inside[0] > 200 && inside[1] < 40);
        assert!(outside[0] < 40 && outside[1] < 40);
    }
}
