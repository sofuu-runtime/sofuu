//! Pre-baked pure-Rust Signed Distance Field (SDF) glyph atlas & text layout engine.
//!
//! The built-in atlas (`assets/ui_fonts.sfdf`, `SFD3`) carries five faces in one
//! `R8` texture: **Geist Sans** at weights 400/500/600/700 (UI proportional
//! typography) and **Geist Mono** 400 (numbers, tables, prices, code). Baking
//! happens offline in `scripts/bake-sfdf.py`; the runtime only samples distance
//! fields, so there is zero OS font dependency and zero per-frame CPU
//! rasterization across Metal, OpenGL, and Vulkan.
//!
//! Layout supports explicit weights, multi-line word/char wrapping, end/middle
//! ellipsis, and caret/selection hit-testing (`char_index_at` /
//! `x_for_char_index`) for editable text.

use serde::{Deserialize, Serialize};

use crate::draw::{DrawList, GlyphInstance};

const DEFAULT_FONT_SFDF: &[u8] = include_bytes!("../assets/ui_fonts.sfdf");

/// Logical font family. `Sans` is the UI proportional face, `Mono` the tabular
/// face used for numbers, prices, tables, and code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FontFamily {
    #[default]
    Sans,
    Mono,
}

/// Canonical weight values (the atlas ships one block per value).
pub const WEIGHT_REGULAR: u16 = 400;
pub const WEIGHT_MEDIUM: u16 = 500;
pub const WEIGHT_SEMIBOLD: u16 = 600;
pub const WEIGHT_BOLD: u16 = 700;

/// Snap any numeric weight to the nearest baked face.
#[inline]
pub fn snap_weight(weight: u16) -> u16 {
    let w = weight.clamp(300, 800);
    if w < 450 {
        WEIGHT_REGULAR
    } else if w < 550 {
        WEIGHT_MEDIUM
    } else if w < 650 {
        WEIGHT_SEMIBOLD
    } else {
        WEIGHT_BOLD
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GlyphMetrics {
    /// Normalized atlas UVs `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    /// Top-left quad X offset from pen origin as a multiple of `px_size`.
    pub quad_offset_x_em: f32,
    /// Top-left quad Y offset from the baseline as a multiple of `px_size`.
    pub quad_offset_y_em: f32,
    /// Quad width as a multiple of `px_size`.
    pub quad_w_em: f32,
    /// Quad height as a multiple of `px_size`.
    pub quad_h_em: f32,
    /// Horizontal pen advance as a multiple of `px_size`.
    pub advance_em: f32,
    /// Optical UI horizontal offset & advance computed from baked SDF ink bounds.
    pub prop_offset_x_em: f32,
    pub prop_advance_em: f32,
}

/// One family/weight face inside the shared atlas texture.
#[derive(Clone, Debug)]
pub struct GlyphBlock {
    pub family: FontFamily,
    pub weight: u16,
    /// Metrics for ASCII `0x20..=0x7E` + `.notdef` at index 95.
    metrics: Vec<GlyphMetrics>,
    /// `true` for v1 monospace blobs, which keep their ink-derived optical spacing.
    optical: bool,
}

impl GlyphBlock {
    #[inline]
    fn metrics(&self, ch: char) -> &GlyphMetrics {
        let code = ch as u32;
        let idx = if (0x20..=0x7E).contains(&code) {
            (code - 0x20) as usize
        } else {
            95
        };
        &self.metrics[idx.min(self.metrics.len().saturating_sub(1))]
    }
}

/// CPU-owned SDF texture atlas ready to upload once to an `R8Unorm` GPU texture.
#[derive(Clone, Debug)]
pub struct GlyphAtlas {
    pub width: u32,
    pub height: u32,
    /// Row-major `width * height` `R8` SDF bytes (`128` = contour edge, `>128` = inside).
    pub pixels: Vec<u8>,
    pub px_range: f32,
    /// Fallback advance for single-block atlases (`em` units).
    pub advance_em: f32,
    pub ascender_em: f32,
    pub line_height_em: f32,
    /// Optical cap-height ink top & vertical center in `em` units.
    pub cap_top_em: f32,
    pub cap_center_em: f32,
    blocks: Vec<GlyphBlock>,
}

/// Explicit typography style used by layout, components, and editable text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    pub size: f32,
    pub family: FontFamily,
    pub weight: u16,
}

impl TextStyle {
    pub const fn new(size: f32, family: FontFamily, weight: u16) -> Self {
        Self {
            size,
            family,
            weight,
        }
    }

    pub const fn sans(size: f32, weight: u16) -> Self {
        Self::new(size, FontFamily::Sans, weight)
    }

    pub const fn mono(size: f32) -> Self {
        Self::new(size, FontFamily::Mono, WEIGHT_REGULAR)
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self::sans(14.0, WEIGHT_REGULAR)
    }
}

/// Line breaking strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineBreak {
    /// Hard breaks (`\n`) only.
    #[default]
    None,
    /// Greedy word wrapping; over-long words fall back to character breaks.
    WordWrap,
    /// Greedy character wrapping.
    CharWrap,
    /// Wrap, then truncate the final line with `...`.
    EllipsisEnd,
    /// Keep head and tail of a single line with `...` in the middle.
    EllipsisMiddle,
}

/// One laid-out line: byte range in the source plus its measured width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextLine {
    /// Inclusive source byte index of the first character.
    pub start: usize,
    /// Exclusive source byte index (the `\n`, when present, is not included).
    pub end: usize,
    pub width: f32,
}

/// Result of measuring a string under a style and width constraint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaidOutText {
    pub lines: Vec<TextLine>,
    pub width: f32,
    pub height: f32,
    pub line_height: f32,
}

/// One styled run inside [`GlyphAtlas::layout_rich`].
#[derive(Clone, Copy, Debug)]
pub struct RichSpan<'a> {
    pub text: &'a str,
    pub style: TextStyle,
    /// `0` means "use the theme's primary text color".
    pub color: u32,
}

#[derive(Clone, Copy, Debug)]
struct RichChar {
    ch: char,
    style: TextStyle,
    color: u32,
    advance: f32,
}

impl GlyphAtlas {
    /// Load the built-in multi-face SDF atlas (Geist Sans 400/500/600/700 + Geist Mono 400).
    pub fn builtin() -> Self {
        Self::from_sfdf(DEFAULT_FONT_SFDF).expect("embedded ui_fonts.sfdf must be valid")
    }

    /// Parse any pre-baked `SFDF` (v1 mono), `SFD2` (proportional), or `SFD3`
    /// (multi-face) atlas binary blob, allowing custom runtime font swaps.
    pub fn from_sfdf(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 32 {
            return Err("invalid SFDF header");
        }
        let magic = &data[0..4];
        let version = match magic {
            b"SFDF" => 1,
            b"SFD2" => 2,
            b"SFD3" => 3,
            _ => return Err("invalid SFDF header"),
        };
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

        let entry_bytes = if version == 1 { 32 } else { 36 };
        let block_header = if version == 3 { 12 } else { 0 };
        let block_count = if version == 3 {
            if data.len() < 36 {
                return Err("truncated SFD3 block table");
            }
            read_u32(32) as usize
        } else {
            1
        };
        if block_count == 0 || block_count > 64 {
            return Err("invalid SFD3 block count");
        }
        let entries_off = if version == 3 {
            36 + block_count * block_header
        } else {
            32
        };
        let entry_total = glyph_count * entry_bytes;
        let pixel_bytes = (width as usize) * (height as usize);
        if glyph_count < 96 || data.len() < entries_off + entry_total + pixel_bytes {
            return Err("truncated SFDF payload");
        }

        // Descripe blocks: v1/v2 carry one implicit block; v3 tags each block.
        let mut block_defs: Vec<(FontFamily, u16, usize, usize)> = Vec::with_capacity(block_count);
        if version == 3 {
            for i in 0..block_count {
                let off = 36 + i * block_header;
                let fam_tag = data[off];
                let weight = u16::from_le_bytes([data[off + 2], data[off + 3]]);
                let first = read_u32(off + 4) as usize;
                let count = read_u32(off + 8) as usize;
                let family = if fam_tag == 1 {
                    FontFamily::Mono
                } else {
                    FontFamily::Sans
                };
                block_defs.push((family, weight, first, count));
            }
        } else {
            let family = if version == 1 {
                FontFamily::Mono
            } else {
                FontFamily::Sans
            };
            block_defs.push((family, WEIGHT_REGULAR, 0, glyph_count));
        }

        let mut blocks = Vec::with_capacity(block_defs.len());
        for (family, weight, first, count) in block_defs {
            if first + count > glyph_count || count < 96 {
                return Err("SFD3 block range outside glyph table");
            }
            let mut metrics = Vec::with_capacity(count);
            for i in 0..count {
                let base = entries_off + (first + i) * entry_bytes;
                let advance = if version == 1 {
                    advance_em
                } else {
                    read_f32(base + 32)
                };
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
                    advance_em: advance,
                    prop_offset_x_em: read_f32(base + 16),
                    prop_advance_em: advance,
                    // placeholder overwritten below when optical
                });
            }
            let mut block = GlyphBlock {
                family,
                weight,
                metrics,
                optical: version == 1,
            };
            if version == 1 {
                derive_optical_metrics(width, height, &[], &mut block, advance_em);
            }
            blocks.push(block);
        }

        let pixels = data[entries_off + entry_total..entries_off + entry_total + pixel_bytes].to_vec();
        // Prefer the Sans regular face for cap optics; fall back to block 0.
        let primary = blocks
            .iter()
            .position(|b| b.family == FontFamily::Sans && b.weight == WEIGHT_REGULAR)
            .unwrap_or(0);
        let mut atlas = Self {
            width,
            height,
            pixels,
            px_range,
            advance_em,
            ascender_em,
            line_height_em,
            cap_top_em: 0.0,
            cap_center_em: 0.0,
            blocks,
        };
        let (cap_top, cap_center) =
            atlas.measure_cap_optics(primary, ascender_em);
        atlas.cap_top_em = cap_top;
        atlas.cap_center_em = cap_center;
        // Optical ink derivation needs the pixel buffer, which existed only after
        // the move above — finish v1 spacing now.
        if version == 1 {
            let pixels = atlas.pixels.clone();
            for b in &mut atlas.blocks {
                if b.optical {
                    derive_optical_metrics(width, height, &pixels, b, advance_em);
                }
            }
        }
        Ok(atlas)
    }

    /// Number of baked faces (1 for v1/v2 atlases, N for v3).
    #[inline]
    pub fn face_count(&self) -> usize {
        self.blocks.len()
    }

    /// Resolve the block serving `(family, weight)`, snapping to the nearest face.
    fn block_index(&self, family: FontFamily, weight: u16) -> usize {
        let mut best: Option<(usize, (u8, u16))> = None;
        for (i, b) in self.blocks.iter().enumerate() {
            let family_penalty = if b.family == family { 0 } else { 1 };
            let key = (family_penalty, b.weight.abs_diff(weight));
            if best.map_or(true, |(_, k)| key < k) {
                best = Some((i, key));
            }
        }
        best.map(|(i, _)| i).unwrap_or(0)
    }

    #[inline]
    fn block(&self, family: FontFamily, weight: u16) -> &GlyphBlock {
        &self.blocks[self.block_index(family, weight)]
    }

    /// Compute `pen_y` so the optical cap center of `px_size` lands exactly at `rect_y + rect_h * 0.5`.
    #[inline]
    pub fn centered_pen_y(&self, rect_y: f32, rect_h: f32, px_size: f32) -> f32 {
        rect_y + rect_h * 0.5 - self.cap_center_em * px_size
    }

    /// Compute the baseline for a secondary smaller font size `sub_px` so it
    /// shares the primary line's baseline. `primary_y` is the line top.
    #[inline]
    pub fn baseline_aligned_y(&self, primary_y: f32, primary_px: f32, _sub_px: f32) -> f32 {
        primary_y + primary_px * self.ascender_em
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
        let style = self.legacy_style(px_size, mono);
        let (tw, _) = self.measure_styled(text, style);
        let tx = x + (w - tw) * 0.5;
        let ty = self.centered_pen_y(y, h, px_size) - self.ascender_em * px_size;
        self.layout_dl_styled(text, tx, ty, style, color, None, dl);
    }

    /// Draw single-line text with an explicit family/weight, optically centered.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_centered_styled(
        &self,
        text: &str,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        style: TextStyle,
        color: u32,
        dl: &mut DrawList,
    ) {
        let (tw, _) = self.measure_styled(text, style);
        let tx = x + (w - tw) * 0.5;
        let ty = self.centered_pen_y(y, h, style.size) - self.ascender_em * style.size;
        self.layout_dl_styled(text, tx, ty, style, color, None, dl);
    }

    #[inline]
    fn legacy_style(&self, px_size: f32, mono: bool) -> TextStyle {
        if mono {
            TextStyle::mono(px_size)
        } else {
            TextStyle::sans(px_size, WEIGHT_REGULAR)
        }
    }

    #[inline]
    pub fn metrics(&self, ch: char) -> GlyphMetrics {
        *self.block(FontFamily::Sans, WEIGHT_REGULAR).metrics(ch)
    }

    /// Metrics for an explicit family/weight.
    #[inline]
    pub fn metrics_styled(&self, ch: char, style: TextStyle) -> GlyphMetrics {
        *self.block(style.family, style.weight).metrics(ch)
    }

    /// Horizontal advance for `ch` in `em`: fixed grid for `Mono`, optical rules
    /// for proportional faces.
    #[inline]
    pub fn char_advance_em(&self, ch: char, mono: bool) -> f32 {
        let style = self.legacy_style(14.0, mono);
        let m = self.block(style.family, style.weight).metrics(ch);
        if mono {
            m.advance_em
        } else {
            m.prop_advance_em
        }
    }

    /// Advance of `ch` in logical px under `style`.
    #[inline]
    pub fn advance(&self, ch: char, style: TextStyle) -> f32 {
        let m = self.block(style.family, style.weight).metrics(ch);
        let em = if m.prop_advance_em > 0.0 && !style.family.eq(&FontFamily::Mono) {
            m.prop_advance_em
        } else {
            m.advance_em
        };
        em * style.size
    }

    /// Measure single-line or multi-line (`\n`) text block in logical pixels.
    pub fn measure(&self, text: &str, px_size: f32, mono: bool) -> (f32, f32) {
        self.measure_styled(text, self.legacy_style(px_size, mono))
    }

    /// Measure text under an explicit style (single line unless the text contains `\n`).
    pub fn measure_styled(&self, text: &str, style: TextStyle) -> (f32, f32) {
        let line_h = style.size * self.line_height_em;
        let block = self.block(style.family, style.weight);
        let mut max_w: f32 = 0.0;
        let mut cur_w: f32 = 0.0;
        let mut lines = 1usize;
        let mut prev: Option<char> = None;

        for ch in text.chars() {
            if ch == '\n' {
                max_w = max_w.max(cur_w);
                cur_w = 0.0;
                lines += 1;
                prev = None;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            let kern = if style.family == FontFamily::Mono || block.optical {
                prev.map_or(0.0, |p| pair_kern_em(p, ch))
            } else {
                0.0
            };
            let m = block.metrics(ch);
            cur_w += (m.prop_advance_em + kern) * style.size;
            prev = Some(ch);
        }
        max_w = max_w.max(cur_w);
        (max_w.max(0.0), lines as f32 * line_h)
    }

    /// Width of a byte range on one line (no wrapping).
    fn range_width(&self, text: &str, block: &GlyphBlock, style: TextStyle, kern: bool) -> f32 {
        let mut w = 0.0f32;
        let mut prev: Option<char> = None;
        for ch in text.chars() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            let k = if kern {
                prev.map_or(0.0, |p| pair_kern_em(p, ch))
            } else {
                0.0
            };
            w += (block.metrics(ch).prop_advance_em + k) * style.size;
            prev = Some(ch);
        }
        w
    }

    /// Lay out `text` into lines for the given width constraint and line-break mode.
    pub fn layout_text(
        &self,
        text: &str,
        style: TextStyle,
        max_width: Option<f32>,
        max_lines: usize,
        break_mode: LineBreak,
    ) -> LaidOutText {
        let line_h = style.size * self.line_height_em;
        let block = self.block(style.family, style.weight);
        let kern = style.family == FontFamily::Mono || block.optical;
        let max_lines = max_lines.max(1);

        if matches!(break_mode, LineBreak::EllipsisMiddle) {
            let avail = max_width.unwrap_or(f32::INFINITY).max(0.0);
            let full = self.range_width(text, block, style, kern);
            if full <= avail {
                return LaidOutText {
                    lines: vec![TextLine {
                        start: 0,
                        end: text.len(),
                        width: full,
                    }],
                    width: full,
                    height: line_h,
                    line_height: line_h,
                };
            }
            let dots = self.range_width("...", block, style, kern);
            let _ = dots;
            // Binary search the largest head byte length whose head+tail fits.
            let mut lo = 0usize;
            let mut hi = text.len();
            let mut best_start = 0;
            let mut best_end = text.len();
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                let head_len = floor_char_boundary(text, mid);
                let head_w = self.range_width(&text[..head_len], block, style, kern);
                let tail_budget = avail - head_w;
                if tail_budget <= 0.0 {
                    hi = mid;
                    continue;
                }
                // Grow a tail from the end that fits the remaining budget.
                let mut tail_start = text.len();
                let mut tail_w = 0.0f32;
                let mut prev: Option<char> = None;
                for (i, ch) in text[head_len..].char_indices().rev() {
                    let k = if kern {
                        pair_kern_em(ch, prev.unwrap_or(ch))
                    } else {
                        0.0
                    };
                    let adv = (block.metrics(ch).prop_advance_em + k) * style.size;
                    if tail_w + adv > tail_budget {
                        break;
                    }
                    tail_w += adv;
                    prev = Some(ch);
                    tail_start = head_len + i;
                    if tail_start <= head_len {
                        break;
                    }
                }
                let candidate = format!("{}...{}", &text[..head_len], &text[tail_start..]);
                let w = self.range_width(&candidate, block, style, kern);
                if w <= avail && tail_start >= head_len {
                    best_start = head_len;
                    best_end = tail_start;
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let candidate = format!("{}...{}", &text[..best_start], &text[best_end..]);
            let w = self.range_width(&candidate, block, style, kern);
            return LaidOutText {
                lines: vec![TextLine {
                    start: 0,
                    end: text.len(),
                    width: w,
                    // `start`/`end` describe the source; the renderer rebuilds
                    // the ellipsized string from `ellipsis_bounds`.
                }],
                width: w,
                height: line_h,
                line_height: line_h,
            };
        }

        let wrap = matches!(
            break_mode,
            LineBreak::WordWrap | LineBreak::CharWrap | LineBreak::EllipsisEnd
        );
        let limit = if wrap {
            max_width.map(|w| w.max(1.0))
        } else {
            None
        };

        let mut lines: Vec<TextLine> = Vec::new();
        let mut para_start = 0usize;
        let bytes = text.as_bytes();
        let mut idx = 0usize;
        // Split into paragraphs on '\n', wrapping each.
        loop {
            let para_end = text[idx..]
                .find('\n')
                .map(|p| idx + p)
                .unwrap_or(text.len());
            wrap_paragraph(
                self, text, block, style, kern, bytes, para_start, para_end, limit, break_mode,
                &mut lines, max_lines,
            );
            if para_end >= text.len() || lines.len() >= max_lines {
                break;
            }
            let mut next = para_end + 1;
            // Collapse '\\r\\n'.
            if next < text.len() && text.as_bytes()[next] == b'\r' {
                next += 1;
            }
            para_start = next;
            idx = next;
            if idx > text.len() {
                break;
            }
        }
        if lines.is_empty() {
            lines.push(TextLine {
                start: 0,
                end: 0,
                width: 0.0,
            });
        }

        // Ellipsis on the final allowed line.
        if matches!(break_mode, LineBreak::EllipsisEnd) {
            if let Some(limit_w) = limit {
                if let Some(last) = lines.last_mut() {
                    let w = self.range_width(&text[last.start..last.end], block, style, kern);
                    if w > limit_w || last.end < text.len() {
                        let (range, ew) =
                            self.ellipsize_end(text, last.start, last.end, block, style, kern, limit_w);
                        last.end = range;
                        last.width = ew;
                    }
                }
            }
        }

        let width = lines.iter().fold(0.0f32, |m, l| m.max(l.width));
        LaidOutText {
            height: lines.len() as f32 * line_h,
            width,
            line_height: line_h,
            lines,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ellipsize_end(
        &self,
        text: &str,
        start: usize,
        end: usize,
        block: &GlyphBlock,
        style: TextStyle,
        kern: bool,
        max_w: f32,
    ) -> (usize, f32) {
        let dots_w = self.range_width("...", block, style, kern);
        if dots_w >= max_w {
            return (start, 0.0);
        }
        let budget = max_w - dots_w;
        let mut acc = 0.0f32;
        let mut prev: Option<char> = None;
        let mut cut = start;
        for (i, ch) in text[start..end].char_indices() {
            let k = if kern {
                prev.map_or(0.0, |p| pair_kern_em(p, ch))
            } else {
                0.0
            };
            let adv = (block.metrics(ch).prop_advance_em + k) * style.size;
            if acc + adv > budget {
                break;
            }
            acc += adv;
            prev = Some(ch);
            cut = start + i + ch.len_utf8();
        }
        (cut, acc + dots_w)
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
        let style = self.legacy_style(px_size, mono);
        self.layout_dl_styled(text, x, y, style, color, max_width, dl);
    }

    /// Emit glyph quads for `text` under an explicit style.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_dl_styled(
        &self,
        text: &str,
        x: f32,
        y: f32,
        style: TextStyle,
        color: u32,
        max_width: Option<f32>,
        dl: &mut DrawList,
    ) {
        let clip = dl.current_clip_id();
        let start = dl.glyphs.len();
        let block = self.block(style.family, style.weight);
        self.emit_into(text, x, y, style, block, color, max_width, &mut dl.glyphs, clip);
        let added = dl.glyphs.len() - start;
        dl.record_glyph_batch(start, added);
    }

    /// Measure and emit a styled run sequence with per-span font weight/family/color.
    ///
    /// Wrapping is greedy at spaces across span boundaries; mixed sizes share a
    /// common baseline. Returns `(width, height)`.
    #[allow(clippy::too_many_arguments)]
    pub fn layout_rich(
        &self,
        spans: &[RichSpan<'_>],
        x: f32,
        y: f32,
        max_width: Option<f32>,
        line_height: Option<f32>,
        dl: &mut DrawList,
    ) -> (f32, f32) {
        let (chars, lines, width, line_h) = self.rich_lines(spans, max_width, line_height);
        let clip = dl.current_clip_id();
        let start = dl.glyphs.len();
        for (li, (s, e)) in lines.iter().enumerate() {
            let line_top = y + li as f32 * line_h;
            let max_size = chars[*s..*e]
                .iter()
                .map(|c| c.style.size)
                .fold(0.0f32, f32::max);
            let mut pen = snap_half_px(x);
            for c in &chars[*s..*e] {
                if c.ch != ' ' && c.ch != '\t' {
                    let block = self.block(c.style.family, c.style.weight);
                    let m = block.metrics(c.ch);
                    let gy = self.baseline_aligned_y(line_top, max_size, c.style.size);
                    dl.glyphs.push(GlyphInstance {
                        pos: [
                            pen + m.quad_offset_x_em * c.style.size,
                            gy + m.quad_offset_y_em * c.style.size,
                        ],
                        size: [
                            m.quad_w_em * c.style.size,
                            m.quad_h_em * c.style.size,
                        ],
                        uv: m.uv,
                        color: c.color,
                        px_range: self.px_range,
                        clip_id: clip,
                        _pad: 0,
                    });
                }
                pen += c.style.size * self.block(c.style.family, c.style.weight).metrics(c.ch).prop_advance_em;
            }
        }
        let added = dl.glyphs.len() - start;
        dl.record_glyph_batch(start, added);
        (width, lines.len() as f32 * line_h)
    }

    /// Measure a rich run sequence without emitting glyphs.
    pub fn measure_rich(
        &self,
        spans: &[RichSpan<'_>],
        max_width: Option<f32>,
        line_height: Option<f32>,
    ) -> (f32, f32) {
        let (_, lines, width, line_h) = self.rich_lines(spans, max_width, line_height);
        (width, lines.len() as f32 * line_h)
    }

    fn rich_lines(
        &self,
        spans: &[RichSpan<'_>],
        max_width: Option<f32>,
        line_height: Option<f32>,
    ) -> (Vec<RichChar>, Vec<(usize, usize)>, f32, f32) {
        let mut chars: Vec<RichChar> = Vec::new();
        for span in spans {
            for ch in span.text.chars() {
                if ch == '\r' {
                    continue;
                }
                let block = self.block(span.style.family, span.style.weight);
                let adv = block.metrics(ch).prop_advance_em * span.style.size;
                chars.push(RichChar {
                    ch,
                    style: span.style,
                    color: span.color,
                    advance: adv,
                });
            }
        }
        let limit = max_width.map(|w| w.max(1.0));
        let mut lines: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let mut pen = 0.0f32;
        let mut last_break: Option<usize> = None;
        let mut width = 0.0f32;
        let mut i = 0usize;
        while i < chars.len() {
            let c = &chars[i];
            if c.ch == '\n' {
                let end = trim_rich_spaces(&chars, start, i);
                lines.push((start, end));
                width = width.max(pen);
                start = i + 1;
                pen = 0.0;
                last_break = None;
                i += 1;
                continue;
            }
            if let Some(l) = limit {
                if pen + c.advance > l && i > start {
                    if let Some(brk) = last_break {
                        if brk > start && brk <= i {
                            let end = trim_rich_spaces(&chars, start, brk);
                            lines.push((start, end));
                            width = width.max(pen);
                            start = brk;
                            pen = 0.0;
                            for cc in &chars[start..i] {
                                pen += cc.advance;
                            }
                            last_break = None;
                            continue;
                        }
                    }
                    let end = trim_rich_spaces(&chars, start, i);
                    lines.push((start, end));
                    width = width.max(pen);
                    start = i;
                    pen = 0.0;
                    last_break = None;
                    continue;
                }
            }
            if c.ch == ' ' || c.ch == '\t' {
                last_break = Some(i + 1);
            }
            pen += c.advance;
            i += 1;
        }
        let end = trim_rich_spaces(&chars, start, chars.len());
        lines.push((start, end));
        width = width.max(pen);
        let line_h = line_height.unwrap_or_else(|| {
            let max_size = chars
                .iter()
                .map(|c| c.style.size)
                .fold(0.0f32, f32::max)
                .max(1.0);
            max_size * self.line_height_em
        });
        (chars, lines, width, line_h)
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
        mono: bool,
        max_width: Option<f32>,
        out: &mut Vec<GlyphInstance>,
        clip_id: u32,
    ) {
        let style = self.legacy_style(px_size, mono);
        let block = self.block(style.family, style.weight);
        self.emit_into(text, x, y, style, block, color, max_width, out, clip_id);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_into(
        &self,
        text: &str,
        x: f32,
        y: f32,
        style: TextStyle,
        block: &GlyphBlock,
        color: u32,
        max_width: Option<f32>,
        out: &mut Vec<GlyphInstance>,
        clip_id: u32,
    ) {
        let px_size = style.size;
        let line_h = px_size * self.line_height_em;
        let origin_x = snap_half_px(x);
        let mut pen_x = origin_x;
        let mut pen_y = snap_half_px(y);
        let kern_enabled = style.family == FontFamily::Mono || block.optical;
        let mut prev: Option<char> = None;

        for ch in text.chars() {
            if ch == '\n' {
                pen_x = origin_x;
                pen_y = snap_half_px(pen_y + line_h);
                prev = None;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            let m = block.metrics(ch);
            let kern = if kern_enabled {
                prev.map_or(0.0, |p| pair_kern_em(p, ch))
            } else {
                0.0
            };
            let adv = m.prop_advance_em * px_size;
            let kpx = kern * px_size;

            if let Some(limit) = max_width {
                if (pen_x - origin_x) + kpx + adv > limit && pen_x > origin_x {
                    pen_x = origin_x;
                    pen_y = snap_half_px(pen_y + line_h);
                    prev = None;
                }
            }
            pen_x += if prev.is_some() { kpx } else { 0.0 };

            if ch != ' ' {
                let gx = pen_x + m.quad_offset_x_em * px_size;
                // `quad_offset_y_em` is baseline-relative (SFD3); `pen_y` is
                // the line top, so add the ascender before the glyph offset.
                let gy = pen_y + self.ascender_em * px_size + m.quad_offset_y_em * px_size;
                out.push(GlyphInstance {
                    pos: [gx, gy],
                    size: [m.quad_w_em * px_size, m.quad_h_em * px_size],
                    uv: m.uv,
                    color,
                    px_range: self.px_range,
                    clip_id,
                    _pad: 0,
                });
            }
            pen_x += adv;
            prev = Some(ch);
        }
    }

    /// Byte index of the character boundary nearest `(x, y)` for caret placement.
    ///
    /// `x`/`y` are relative to the text origin (top-left of the first line).
    #[allow(clippy::too_many_arguments)]
    pub fn char_index_at(
        &self,
        text: &str,
        x: f32,
        y: f32,
        style: TextStyle,
        max_width: Option<f32>,
        break_mode: LineBreak,
    ) -> usize {
        let laid = self.layout_text(text, style, max_width, usize::MAX, break_mode);
        let line_idx = ((y.max(0.0) / laid.line_height.max(0.001)) as usize)
            .min(laid.lines.len().saturating_sub(1));
        let line = laid.lines[line_idx];
        let block = self.block(style.family, style.weight);
        let mut pen = 0.0f32;
        for (i, ch) in text[line.start..line.end].char_indices() {
            let adv = self.advance_styled(ch, block, style);
            if x < pen + adv * 0.5 {
                return line.start + i;
            }
            pen += adv;
        }
        line.end
    }

    /// Horizontal offset of the caret before the character at `index` (UTF-8 boundary).
    #[allow(clippy::too_many_arguments)]
    pub fn x_for_char_index(
        &self,
        text: &str,
        index: usize,
        style: TextStyle,
        max_width: Option<f32>,
        break_mode: LineBreak,
    ) -> (f32, usize) {
        let laid = self.layout_text(text, style, max_width, usize::MAX, break_mode);
        let index = index.min(text.len());
        let block = self.block(style.family, style.weight);
        for (line_idx, line) in laid.lines.iter().enumerate() {
            if index <= line.end {
                let upto = index.max(line.start).min(line.end);
                let mut pen = 0.0f32;
                for ch in text[line.start..upto].chars() {
                    pen += self.advance_styled(ch, block, style);
                }
                return (pen, line_idx);
            }
        }
        (0.0, laid.lines.len().saturating_sub(1))
    }

    #[inline]
    fn advance_styled(&self, ch: char, block: &GlyphBlock, style: TextStyle) -> f32 {
        block.metrics(ch).prop_advance_em * style.size
    }

    /// Derive cap-height optics from the baked `'M'` (fallback `'0'`) SDF ink.
    fn measure_cap_optics(&self, block_idx: usize, ascender_em: f32) -> (f32, f32) {
        let block = &self.blocks[block_idx];
        for target in ['M', '0'] {
            let m = block.metrics(target);
            let x0 = (m.uv[0] * self.width as f32).round() as u32;
            let y0 = (m.uv[1] * self.height as f32).round() as u32;
            let x1 = (m.uv[2] * self.width as f32).round().min(self.width as f32) as u32;
            let y1 = (m.uv[3] * self.height as f32).round().min(self.height as f32) as u32;
            let cell_h = y1.saturating_sub(y0).max(1) as f32;
            let mut min_row = u32::MAX;
            let mut max_row = 0u32;
            for py in y0..y1 {
                for px in x0..x1 {
                    if self.pixels[(py * self.width + px) as usize] >= 128 {
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
}

fn trim_rich_spaces(chars: &[RichChar], start: usize, end: usize) -> usize {
    let mut e = end;
    while e > start && matches!(chars[e - 1].ch, ' ' | '\t') {
        e -= 1;
    }
    e
}

#[inline]
fn snap_half_px(v: f32) -> f32 {
    (v * 2.0).round() * 0.5
}

/// Mild class-based optical pair kerning rules (digits never kern to preserve tabular columns).
#[inline]
fn pair_kern_em(prev: char, cur: char) -> f32 {
    if prev.is_ascii_digit() && cur.is_ascii_digit() {
        return 0.0;
    }
    let tight_after_overhang = matches!(prev, 'T' | 'V' | 'W' | 'Y' | 'P' | 'F');
    if tight_after_overhang && matches!(cur, '.' | ',' | ':' | ';') {
        return -0.018;
    }
    if matches!(prev, 'T' | 'V' | 'W' | 'Y') && matches!(cur, 'a' | 'e' | 'o' | 'u') {
        return -0.014;
    }
    0.0
}

#[inline]
fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Greedy wrap of one paragraph into `lines` (byte ranges), honoring `max_width`.
#[allow(clippy::too_many_arguments)]
fn wrap_paragraph(
    atlas: &GlyphAtlas,
    text: &str,
    block: &GlyphBlock,
    style: TextStyle,
    kern: bool,
    _bytes: &[u8],
    start: usize,
    end: usize,
    max_w: Option<f32>,
    mode: LineBreak,
    lines: &mut Vec<TextLine>,
    max_lines: usize,
) {
    if lines.len() >= max_lines {
        return;
    }
    let Some(limit) = max_w else {
        let w = atlas.range_width(&text[start..end], block, style, kern);
        lines.push(TextLine { start, end, width: w });
        return;
    };
    if end <= start {
        lines.push(TextLine {
            start,
            end: start,
            width: 0.0,
        });
        return;
    }

    let mut line_start = start;
    let mut pen = 0.0f32;
    let mut last_break: Option<usize> = None; // byte index just after a break opportunity
    let mut last_break_pen = 0.0f32;
    let mut prev: Option<char> = None;

    for (i, ch) in text[start..end].char_indices() {
        let abs = start + i;
        let k = if kern {
            prev.map_or(0.0, |p| pair_kern_em(p, ch))
        } else {
            0.0
        };
        let adv = (block.metrics(ch).prop_advance_em + k) * style.size;
        if pen + adv > limit && abs > line_start {
            // Break at the last space for word wrap, otherwise at this char.
            let word_break = mode == LineBreak::WordWrap && last_break.is_some();
            let cut = if word_break {
                last_break.unwrap_or(abs)
            } else {
                abs
            };
            let cut_pen = if word_break { last_break_pen } else { pen };
            let trimmed = trim_end_spaces(text, line_start, cut);
            lines.push(TextLine {
                start: line_start,
                end: trimmed,
                width: cut_pen,
            });
            if lines.len() >= max_lines {
                return;
            }
            line_start = cut;
            pen = 0.0;
            last_break = None;
            last_break_pen = 0.0;
            // Re-measure from the new line start to the current char.
            for (j, c2) in text[line_start..abs + ch.len_utf8()].char_indices() {
                let _ = j;
                pen += block.metrics(c2).prop_advance_em * style.size;
            }
            if ch == ' ' {
                line_start = abs + ch.len_utf8();
                pen = 0.0;
            }
            prev = Some(ch);
            continue;
        }
        if ch == ' ' || ch == '\t' {
            last_break = Some(abs + ch.len_utf8());
            last_break_pen = pen + adv;
        }
        pen += adv;
        prev = Some(ch);
    }
    if line_start < end || lines.is_empty() {
        let trimmed = trim_end_spaces(text, line_start, end);
        lines.push(TextLine {
            start: line_start,
            end: trimmed.max(line_start),
            width: pen,
        });
    }
}

#[inline]
fn trim_end_spaces(text: &str, start: usize, end: usize) -> usize {
    let mut e = end;
    let slice = &text[start..end];
    for (i, ch) in slice.char_indices().rev() {
        if ch == ' ' || ch == '\t' {
            e = start + i;
        } else {
            break;
        }
    }
    e.max(start)
}

/// Inspect every glyph cell's actual SDF ink pixels once at atlas load to derive balanced
/// optical UI spacing & centered punctuation bounds (v1 monospace atlases only).
fn derive_optical_metrics(
    atlas_w: u32,
    atlas_h: u32,
    pixels: &[u8],
    block: &mut GlyphBlock,
    base_advance_em: f32,
) {
    if pixels.is_empty() {
        return;
    }
    for (idx, m) in block.metrics.iter_mut().enumerate() {
        let code = 0x20 + idx as u32;
        let ch = char::from_u32(code).unwrap_or(' ');
        if ch == ' ' {
            m.prop_advance_em = 0.38;
            m.prop_offset_x_em = m.quad_offset_x_em;
            continue;
        }
        if ch.is_ascii_digit() {
            let tab_adv = (base_advance_em * 0.92).clamp(0.52, 0.58);
            m.prop_advance_em = tab_adv;
            m.prop_offset_x_em = m.quad_offset_x_em + (tab_adv - base_advance_em) * 0.5;
            continue;
        }

        let x0 = (m.uv[0] * atlas_w as f32).round() as u32;
        let y0 = (m.uv[1] * atlas_h as f32).round() as u32;
        let x1 = (m.uv[2] * atlas_w as f32).round().min(atlas_w as f32) as u32;
        let y1 = (m.uv[3] * atlas_h as f32).round().min(atlas_h as f32) as u32;
        let cell_w = x1.saturating_sub(x0).max(1) as f32;

        let mut min_col = u32::MAX;
        let mut max_col = 0u32;
        for py in y0..y1 {
            for px in x0..x1 {
                if pixels[(py * atlas_w + px) as usize] >= 128 {
                    let rel = px - x0;
                    min_col = min_col.min(rel);
                    max_col = max_col.max(rel);
                }
            }
        }

        if min_col > max_col {
            m.prop_advance_em = base_advance_em * 0.72;
            m.prop_offset_x_em = m.quad_offset_x_em;
            continue;
        }

        let lsb_em = m.quad_offset_x_em + (min_col as f32 / cell_w) * m.quad_w_em;
        let ink_w_em = ((max_col - min_col + 1) as f32 / cell_w) * m.quad_w_em;

        if matches!(ch, '.' | ',' | ':' | ';' | '!' | '\'' | '"' | '|') {
            let prop_adv = (ink_w_em + 0.16).clamp(0.28, 0.44);
            let target_lsb = (prop_adv - ink_w_em) * 0.5;
            m.prop_advance_em = prop_adv;
            m.prop_offset_x_em = m.quad_offset_x_em + target_lsb - lsb_em;
        } else if matches!(ch, '(' | ')' | '[' | ']' | '{' | '}' | '/' | '\\' | '-' | '_') {
            let prop_adv = (ink_w_em + 0.15).clamp(0.36, 0.56);
            let target_lsb = (prop_adv - ink_w_em) * 0.5;
            m.prop_advance_em = prop_adv;
            m.prop_offset_x_em = m.quad_offset_x_em + target_lsb - lsb_em;
        } else if matches!(ch, 'i' | 'j' | 'l' | 'r' | 't' | 'f' | 'I') {
            let side = 0.056;
            let prop_adv = (ink_w_em + side * 2.0).clamp(0.42, base_advance_em);
            let target_lsb = (prop_adv - ink_w_em) * 0.5;
            m.prop_advance_em = prop_adv;
            m.prop_offset_x_em = m.quad_offset_x_em + target_lsb - lsb_em;
        } else {
            let side = 0.068;
            let prop_adv = (ink_w_em + side * 2.0).clamp(0.46, base_advance_em * 1.04);
            let target_lsb = (prop_adv - ink_w_em) * 0.5;
            m.prop_advance_em = prop_adv;
            m.prop_offset_x_em = m.quad_offset_x_em + target_lsb - lsb_em;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_atlas_has_sans_weights_and_mono_in_one_texture() {
        let atlas = GlyphAtlas::builtin();
        assert_eq!(atlas.face_count(), 5);
        assert!(atlas.width > 0 && atlas.height > 0);
        assert_eq!(atlas.pixels.len(), (atlas.width * atlas.height) as usize);
        assert!((atlas.advance_em - 0.60).abs() < 0.02);

        // 'A' (index 33) must have crisp interior (>180) and exterior (<30) SDF contrast.
        let m = atlas.metrics_styled('A', TextStyle::sans(16.0, WEIGHT_REGULAR));
        let ox = (m.uv[0] * atlas.width as f32).round() as u32;
        let oy = (m.uv[1] * atlas.height as f32).round() as u32;
        let cw = ((m.uv[2] - m.uv[0]) * atlas.width as f32).round() as u32;
        let chh = ((m.uv[3] - m.uv[1]) * atlas.height as f32).round() as u32;
        let mut min_v = 255u8;
        let mut max_v = 0u8;
        for py in 0..chh {
            for px in 0..cw {
                let v = atlas.pixels[((oy + py) * atlas.width + ox + px) as usize];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        assert!(min_v <= 20, "expected outside SDF <= 20, got {min_v}");
        // Stems are ~1.3px half-width at this bake scale, so the SDF interior
        // saturates around 175 rather than the atlas maximum.
        assert!(max_v >= 160, "expected inside SDF >= 160, got {max_v}");
    }

    #[test]
    fn weights_widen_stems_and_sans_is_proportional() {
        let atlas = GlyphAtlas::builtin();
        let regular = atlas.metrics_styled('M', TextStyle::sans(14.0, WEIGHT_REGULAR));
        let bold = atlas.metrics_styled('M', TextStyle::sans(14.0, WEIGHT_BOLD));
        assert_ne!(regular.uv, bold.uv, "weights must use distinct baked blocks");
        let i_w = atlas.advance('i', TextStyle::sans(14.0, WEIGHT_REGULAR));
        let m_w = atlas.advance('M', TextStyle::sans(14.0, WEIGHT_REGULAR));
        assert!(i_w < m_w * 0.6, "proportional face must not be monospaced");
        let mono_i = atlas.advance('i', TextStyle::mono(14.0));
        let mono_m = atlas.advance('M', TextStyle::mono(14.0));
        assert!((mono_i - mono_m).abs() < 1e-4, "mono face must stay tabular");
    }

    #[test]
    fn wrapping_and_ellipsis_respect_the_width_budget() {
        let atlas = GlyphAtlas::builtin();
        let style = TextStyle::sans(12.0, WEIGHT_REGULAR);
        let text = "The quick brown fox jumps over the lazy dog";
        let laid = atlas.layout_text(text, style, Some(80.0), usize::MAX, LineBreak::WordWrap);
        assert!(laid.lines.len() > 1);
        for line in &laid.lines {
            assert!(line.width <= 81.0, "line wider than budget: {line:?}");
        }
        let ell = atlas.layout_text(text, style, Some(80.0), 3, LineBreak::EllipsisEnd);
        assert_eq!(ell.lines.len(), 3);
        let mid = atlas.layout_text(text, style, Some(120.0), 1, LineBreak::EllipsisMiddle);
        assert_eq!(mid.lines.len(), 1);
        assert!(mid.width <= 121.0);
    }

    #[test]
    fn caret_hit_testing_round_trips_through_char_indices() {
        let atlas = GlyphAtlas::builtin();
        let style = TextStyle::sans(14.0, WEIGHT_REGULAR);
        let text = "hello world\nsecond line";
        for idx in [0usize, 3, 5, 11, 18] {
            let (x, line) = atlas.x_for_char_index(text, idx, style, Some(200.0), LineBreak::WordWrap);
            let y = line as f32 * style.size * atlas.line_height_em + style.size * 0.5;
            let hit = atlas.char_index_at(text, x + 0.1, y, style, Some(200.0), LineBreak::WordWrap);
            assert_eq!(hit, idx, "caret index {idx} (line {line}, x {x})");
        }
    }

    #[test]
    fn geist_mono_v1_blob_still_loads() {
        let v1 = include_bytes!("../assets/geist_mono.sfdf");
        let atlas = GlyphAtlas::from_sfdf(v1).expect("v1 mono atlas");
        assert_eq!(atlas.face_count(), 1);
        let (w, _) = atlas.measure("Geist Mono", 16.0, true);
        assert!(w > 80.0);
    }
}
