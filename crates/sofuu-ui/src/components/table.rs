//! Virtualized GPU Data Table (`Table`).
//!
//! Renders only rows visible within the viewport (`O(visible_rows)` instead of `O(total_rows)`),
//! supports sortable column headers, row selection, semantic cell/row tones, smooth kinetic scrolling,
//! and strict column/table `ClipRect` isolation so text never bleeds into adjacent cells or siblings.

use serde::{Deserialize, Serialize};

use crate::components::fit_text;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, Tone};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableSortDir {
    #[default]
    Asc,
    Desc,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableColumn {
    pub key: String,
    pub label: String,
    #[serde(default)]
    pub width: Option<f32>,
    #[serde(default)]
    pub align: TableAlign,
    #[serde(default = "default_true")]
    pub sortable: bool,
}

impl TableColumn {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            width: None,
            align: TableAlign::Left,
            sortable: true,
        }
    }

    pub fn align(mut self, align: TableAlign) -> Self {
        self.align = align;
        self
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(36.0));
        self
    }

    pub fn sortable(mut self, sortable: bool) -> Self {
        self.sortable = sortable;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableCell {
    pub text: String,
    #[serde(default)]
    pub tone: Option<Tone>,
    #[serde(default)]
    pub badge: bool,
}

impl TableCell {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tone: None,
            badge: false,
        }
    }

    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = Some(tone);
        self
    }

    pub fn badge(mut self) -> Self {
        self.badge = true;
        self
    }
}

impl<S: Into<String>> From<S> for TableCell {
    fn from(s: S) -> Self {
        Self::new(s)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableRow {
    pub id: String,
    pub cells: Vec<TableCell>,
    #[serde(default)]
    pub tone: Option<Tone>,
}

impl TableRow {
    pub fn new(id: impl Into<String>, cells: Vec<TableCell>) -> Self {
        Self {
            id: id.into(),
            cells,
            tone: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableData {
    #[serde(default)]
    pub title: Option<String>,
    pub columns: Vec<TableColumn>,
    pub rows: Vec<TableRow>,
    #[serde(default)]
    pub sort_col: Option<usize>,
    #[serde(default)]
    pub sort_dir: TableSortDir,
    #[serde(default)]
    pub selected_row: Option<String>,
    #[serde(default = "default_row_height")]
    pub row_height: f32,
    #[serde(default = "default_true")]
    pub striped: bool,
    #[serde(default)]
    pub empty_text: Option<String>,
    #[serde(default)]
    pub border_width: f32,
    #[serde(default)]
    pub border_color: Option<u32>,
    #[serde(default)]
    pub shadow_blur: f32,
    #[serde(default)]
    pub shadow_offset_y: f32,
}

impl Default for TableData {
    fn default() -> Self {
        Self {
            title: None,
            columns: Vec::new(),
            rows: Vec::new(),
            sort_col: None,
            sort_dir: TableSortDir::Asc,
            selected_row: None,
            row_height: 32.0,
            striped: true,
            empty_text: None,
            border_width: 0.0,
            border_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_row_height() -> f32 {
    32.0
}

pub fn measure_table(data: &TableData, avail_w: f32) -> (f32, f32) {
    let row_h = data.row_height.clamp(24.0, 56.0);
    let visible_rows = data.rows.len().clamp(3, 8) as f32;
    let title_h = if data.title.as_deref().is_some_and(|t| !t.is_empty()) {
        30.0
    } else {
        0.0
    };
    let pref_h = (title_h + 32.0 + visible_rows * row_h + 16.0).clamp(160.0, 360.0);
    (260.0_f32.min(avail_w.max(180.0)), pref_h)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_table(
    id: &str,
    data: &TableData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 80.0 || rect.h < 72.0 {
        return;
    }

    let outer_border = if data.border_width > 0.0 {
        data.border_color.unwrap_or(theme.border)
    } else {
        0
    };
    let outer_shadow = if data.shadow_blur > 0.0 {
        theme.shadow
    } else {
        0
    };
    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [theme.radius_card; 4],
        color_top: theme.surface_1,
        color_bottom: theme.surface_1,
        border_color: outer_border,
        shadow_color: outer_shadow,
        params: [data.border_width, data.shadow_blur, data.shadow_offset_y, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let pad = 12.0;
    let ix = rect.x + pad;
    let iw = (rect.w - pad * 2.0).max(32.0);
    let mut cur_y = rect.y + 10.0;

    if let Some(title) = data.title.as_deref().filter(|t| !t.is_empty()) {
        let fitted = fit_text(atlas, title, 13.5, false, (iw - 76.0).max(24.0));
        atlas.layout_dl(&fitted, ix, cur_y, 13.5, theme.text_primary, false, None, dl);

        let count_lbl = format!("{} rows", data.rows.len());
        let (cw, _) = atlas.measure(&count_lbl, 11.0, false);
        if iw > cw + 60.0 {
            atlas.layout_dl(
                &count_lbl,
                ix + iw - cw,
                cur_y + 2.0,
                11.0,
                theme.text_muted,
                false,
                None,
                dl,
            );
        }
        cur_y += 26.0;
    }

    if data.columns.is_empty() {
        let msg = data.empty_text.as_deref().unwrap_or("No columns configured");
        atlas.layout_centered(
            msg,
            ix,
            cur_y,
            iw,
            (rect.y + rect.h - cur_y - 10.0).max(24.0),
            12.0,
            theme.text_muted,
            false,
            dl,
        );
        dl.pop_clip();
        return;
    }

    let col_widths = resolve_col_widths(&data.columns, iw);
    let header_h = 28.0;

    // Header bar background
    dl.push_quad(QuadInstance {
        rect: [ix, cur_y, iw, header_h],
        radii: [theme.radius_control; 4],
        color_top: theme.surface_0,
        color_bottom: theme.surface_0,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let mut cx = ix;
    for (col_idx, (col, &cw)) in data.columns.iter().zip(col_widths.iter()).enumerate() {
        let th_rect = ComputedRect {
            x: cx,
            y: cur_y,
            w: cw,
            h: header_h,
            content_h: header_h,
            folded: false,
        };
        if col.sortable {
            interaction.push_hit(
                th_rect,
                HitKind::TableHeader {
                    table_id: id.to_string(),
                    col_index: col_idx,
                },
            );
        }
        let th_key = format!("{id}::th::{col_idx}");
        let hovered = col.sortable && interaction.hovered_key.as_deref() == Some(th_key.as_str());
        if hovered {
            dl.push_quad(QuadInstance {
                rect: [cx + 2.0, cur_y + 2.0, (cw - 4.0).max(8.0), header_h - 4.0],
                radii: [theme.radius_control - 2.0; 4],
                color_top: theme.surface_2,
                color_bottom: theme.surface_2,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
        }

        let sort_suffix = if data.sort_col == Some(col_idx) {
            match data.sort_dir {
                TableSortDir::Asc => " ^",
                TableSortDir::Desc => " v",
            }
        } else {
            ""
        };
        let header_lbl = format!("{}{}", col.label, sort_suffix);
        let cell_pad = 8.0;
        let fitted = fit_text(atlas, &header_lbl, 11.0, false, (cw - cell_pad * 2.0).max(8.0));
        let lbl_col = if data.sort_col == Some(col_idx) {
            theme.text_primary
        } else {
            theme.text_secondary
        };
        draw_aligned_text(
            atlas,
            &fitted,
            cx + cell_pad,
            cur_y,
            (cw - cell_pad * 2.0).max(4.0),
            header_h,
            11.0,
            lbl_col,
            col.align,
            dl,
        );
        cx += cw;
    }

    cur_y += header_h + 4.0;
    let body_h = (rect.y + rect.h - 10.0 - cur_y).max(20.0);

    if data.rows.is_empty() {
        let msg = data.empty_text.as_deref().unwrap_or("No rows");
        atlas.layout_centered(msg, ix, cur_y, iw, body_h, 12.0, theme.text_muted, false, dl);
        dl.pop_clip();
        return;
    }

    let row_h = data.row_height.clamp(24.0, 56.0);
    let total_rows_h = data.rows.len() as f32 * row_h;
    let max_scroll = (total_rows_h - body_h).max(0.0);

    let scroll_key = format!("table::{id}");
    if max_scroll > 0.0 {
        interaction.push_hit(
            ComputedRect {
                x: ix,
                y: cur_y,
                w: iw,
                h: body_h,
                content_h: total_rows_h,
                folded: false,
            },
            HitKind::ScrollContainer {
                id: scroll_key.clone(),
                max_scroll_y: max_scroll,
            },
        );
    }
    let scroll_y = {
        let s = interaction.scrolls.entry(scroll_key).or_default();
        s.max_y = max_scroll;
        s.offset_y = s.offset_y.clamp(0.0, max_scroll);
        s.offset_y
    };

    dl.push_clip(ClipRect::new(ix, cur_y, iw, body_h));

    // Virtualization window: pack only visible rows
    let start_idx = ((scroll_y / row_h).floor() as usize).min(data.rows.len());
    let end_idx = (((scroll_y + body_h) / row_h).ceil() as usize + 1).min(data.rows.len());

    for row_idx in start_idx..end_idx {
        let row = &data.rows[row_idx];
        let ry = cur_y + row_idx as f32 * row_h - scroll_y;
        let row_rect = ComputedRect {
            x: ix,
            y: ry,
            w: iw,
            h: row_h,
            content_h: row_h,
            folded: false,
        };
        interaction.push_hit(
            row_rect,
            HitKind::TableRow {
                table_id: id.to_string(),
                row_id: row.id.clone(),
            },
        );

        let tr_key = format!("{id}::tr::{}", row.id);
        let hovered = interaction.hovered_key.as_deref() == Some(tr_key.as_str());
        let selected = data.selected_row.as_deref() == Some(row.id.as_str());
        let hover_t = springs.animate(
            &tr_key,
            "row_hover",
            if hovered { 1.0 } else { 0.0 },
            SpringConfig::SNAPPY,
        );

        let base_bg = if selected {
            scale_alpha(theme.accent, 0.16)
        } else if data.striped && row_idx % 2 == 1 {
            scale_alpha(theme.surface_0, 0.72)
        } else {
            theme.surface_1
        };
        let row_bg = if hover_t > 0.01 && !selected {
            lerp_rgba8(base_bg, theme.surface_2, hover_t)
        } else {
            base_bg
        };

        dl.push_quad(QuadInstance {
            rect: [ix, ry + 1.0, iw, (row_h - 2.0).max(16.0)],
            radii: [theme.radius_control - 2.0; 4],
            color_top: row_bg,
            color_bottom: row_bg,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });

        if let Some(rtone) = row.tone.filter(|&t| t != Tone::Neutral) {
            let tc = theme.tone_color(rtone);
            dl.push_quad(QuadInstance {
                rect: [ix + 2.0, ry + 6.0, 2.5, (row_h - 12.0).max(8.0)],
                radii: [1.25; 4],
                color_top: tc,
                color_bottom: tc,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
        }

        let mut cell_x = ix;
        for (col_idx, (col, &cw)) in data.columns.iter().zip(col_widths.iter()).enumerate() {
            if let Some(cell) = row.cells.get(col_idx) {
                let cell_pad = 8.0;
                let inner_cw = (cw - cell_pad * 2.0).max(8.0);
                let cell_tone = cell.tone.unwrap_or(Tone::Neutral);
                let fg = if cell_tone == Tone::Neutral {
                    theme.text_primary
                } else {
                    theme.tone_color(cell_tone)
                };
                if cell.badge && !cell.text.is_empty() {
                    let fitted = fit_text(atlas, &cell.text, 10.5, false, (inner_cw - 10.0).max(6.0));
                    let (tw, _) = atlas.measure(&fitted, 10.5, false);
                    let pill_w = (tw + 12.0).min(cw - 6.0).max(18.0);
                    let pill_h = (row_h - 10.0).clamp(16.0, 22.0);
                    let pill_x = match col.align {
                        TableAlign::Left => cell_x + cell_pad - 2.0,
                        TableAlign::Center => cell_x + (cw - pill_w) * 0.5,
                        TableAlign::Right => cell_x + cw - cell_pad - pill_w + 2.0,
                    };
                    let pill_y = ry + (row_h - pill_h) * 0.5;
                    let pill_bg = if cell_tone == Tone::Neutral {
                        theme.surface_2
                    } else {
                        scale_alpha(fg, 0.18)
                    };
                    dl.push_quad(QuadInstance {
                        rect: [pill_x, pill_y, pill_w, pill_h],
                        radii: [pill_h * 0.5; 4],
                        color_top: pill_bg,
                        color_bottom: pill_bg,
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                    atlas.layout_centered(&fitted, pill_x, pill_y, pill_w, pill_h, 10.5, fg, false, dl);
                } else {
                    let fitted = fit_text(atlas, &cell.text, 12.0, false, inner_cw);
                    draw_aligned_text(
                        atlas,
                        &fitted,
                        cell_x + cell_pad,
                        ry,
                        inner_cw,
                        row_h,
                        12.0,
                        fg,
                        col.align,
                        dl,
                    );
                }
            }
            cell_x += cw;
        }
    }

    // Subtle overlay scrollbar thumb
    if max_scroll > 1.0 {
        let track_h = (body_h - 8.0).max(20.0);
        let thumb_h = ((body_h / total_rows_h) * track_h).clamp(20.0, track_h);
        let thumb_y = cur_y + 4.0 + (scroll_y / max_scroll) * (track_h - thumb_h);
        dl.push_quad(QuadInstance {
            rect: [ix + iw - 4.0, thumb_y, 3.0, thumb_h],
            radii: [1.5; 4],
            color_top: scale_alpha(theme.text_muted, 0.5),
            color_bottom: scale_alpha(theme.text_muted, 0.5),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }

    dl.pop_clip();
    dl.pop_clip();
}

/// Sort rows in-place numerically when both cell texts parse cleanly as finite `f64`, else lexicographically.
pub fn sort_table_rows(data: &mut TableData, col_index: usize) {
    if col_index >= data.columns.len() {
        return;
    }
    if data.sort_col == Some(col_index) {
        data.sort_dir = match data.sort_dir {
            TableSortDir::Asc => TableSortDir::Desc,
            TableSortDir::Desc => TableSortDir::Asc,
        };
    } else {
        data.sort_col = Some(col_index);
        data.sort_dir = TableSortDir::Asc;
    }
    let asc = data.sort_dir == TableSortDir::Asc;
    data.rows.sort_by(|a, b| {
        let ta = a.cells.get(col_index).map(|c| c.text.trim()).unwrap_or("");
        let tb = b.cells.get(col_index).map(|c| c.text.trim()).unwrap_or("");
        let ord = match (parse_numeric_cell(ta), parse_numeric_cell(tb)) {
            (Some(na), Some(nb)) => na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal),
            _ => ta.cmp(tb),
        };
        if asc {
            ord
        } else {
            ord.reverse()
        }
    });
}

fn parse_numeric_cell(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|&c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+'))
        .collect();
    if cleaned.is_empty() || cleaned == "-" || cleaned == "+" || cleaned == "." {
        return None;
    }
    cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
}

fn resolve_col_widths(cols: &[TableColumn], total_w: f32) -> Vec<f32> {
    let mut fixed_sum = 0.0f32;
    let mut auto_count = 0usize;
    for c in cols {
        if let Some(w) = c.width {
            fixed_sum += w.max(24.0);
        } else {
            auto_count += 1;
        }
    }
    if fixed_sum > total_w && fixed_sum > 0.0 {
        let scale = total_w / fixed_sum;
        return cols
            .iter()
            .map(|c| c.width.map(|w| w * scale).unwrap_or(24.0 * scale))
            .collect();
    }
    let rem = (total_w - fixed_sum).max(0.0);
    let auto_w = if auto_count > 0 {
        rem / auto_count as f32
    } else {
        0.0
    };
    let mut out: Vec<f32> = cols
        .iter()
        .map(|c| c.width.unwrap_or(auto_w))
        .collect();
    // Redistribute leftover space if all columns specified widths < total_w
    let used: f32 = out.iter().sum();
    if used > 0.0 && used < total_w {
        let scale = total_w / used;
        for w in &mut out {
            *w *= scale;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn draw_aligned_text(
    atlas: &GlyphAtlas,
    text: &str,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    px: f32,
    color: u32,
    align: TableAlign,
    dl: &mut DrawList,
) {
    if text.is_empty() {
        return;
    }
    let (tw, _) = atlas.measure(text, px, false);
    let tx = match align {
        TableAlign::Left => x,
        TableAlign::Center => x + ((w - tw) * 0.5).max(0.0),
        TableAlign::Right => x + (w - tw).max(0.0),
    };
    let ty = atlas.centered_pen_y(y, h, px);
    atlas.layout_dl(text, tx, ty, px, color, false, None, dl);
}
