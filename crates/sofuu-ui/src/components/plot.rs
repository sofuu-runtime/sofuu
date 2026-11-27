//! `Plot` component: GPU-accelerated, multi-pane synchronized charting engine.
//!
//! Supports 6 series types (`Line`, `Area` via smooth sloped ribbons, `StackedArea`,
//! dual-curve `Band`, TradingView-proportioned OHLC `Candles`, and directional `Histogram`),
//! dashed reference thresholds, synchronized multi-pane stacks, scroll-zoom & drag-pan,
//! interactive clickable legends, and zero-occlusion HeaderBar + Axis-Pill crosshairs.

use serde::{Deserialize, Serialize};

use crate::components::{fit_text, format_number};
use crate::draw::{
    lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance, RibbonInstance, StrokeInstance,
};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::Theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotSeriesKind {
    #[default]
    Line,
    Area,
    StackedArea,
    /// Dual-curve spread fill between `point.y` and `point.y2` (falls back to baseline if `y2` is `None`).
    Band,
    Candles,
    Histogram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadoutMode {
    /// Non-occluding TradingView-style OHLC/series HUD bar at top of pane + axis pills (default).
    #[default]
    HeaderBar,
    /// Floating edge-clamped tooltip card inside plot bounds.
    Floating,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotPoint {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub y2: Option<f64>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandlePoint {
    pub x: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotThreshold {
    pub value: f64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub color: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotSeries {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub kind: PlotSeriesKind,
    #[serde(default)]
    pub color: Option<u32>,
    #[serde(default)]
    pub secondary_color: Option<u32>,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub dashed: bool,
    #[serde(default)]
    pub stroke_width: Option<f32>,
    #[serde(default)]
    pub points: Vec<PlotPoint>,
    #[serde(default)]
    pub candles: Vec<CandlePoint>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotPane {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default = "default_weight")]
    pub height_weight: f32,
    #[serde(default)]
    pub y_prefix: String,
    #[serde(default)]
    pub y_suffix: String,
    #[serde(default)]
    pub y_precision: Option<usize>,
    #[serde(default)]
    pub stacked_percent: bool,
    #[serde(default)]
    pub thresholds: Vec<PlotThreshold>,
    #[serde(default)]
    pub series: Vec<PlotSeries>,
}

impl Default for PlotPane {
    fn default() -> Self {
        Self {
            id: "main".to_string(),
            label: None,
            height_weight: 1.0,
            y_prefix: String::new(),
            y_suffix: String::new(),
            y_precision: None,
            stacked_percent: false,
            thresholds: Vec::new(),
            series: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotData {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub subtitle: Option<String>,
    #[serde(default)]
    pub series: Vec<PlotSeries>,
    /// Optional synchronized multi-pane layout. If non-empty, renders stacked panes sharing X-axis & crosshair.
    #[serde(default)]
    pub panes: Vec<PlotPane>,
    #[serde(default)]
    pub thresholds: Vec<PlotThreshold>,
    #[serde(default = "default_true")]
    pub show_grid: bool,
    #[serde(default = "default_true")]
    pub show_legend: bool,
    #[serde(default = "default_true")]
    pub crosshair: bool,
    #[serde(default)]
    pub readout_mode: ReadoutMode,
    #[serde(default)]
    pub stacked_percent: bool,
    #[serde(default)]
    pub y_prefix: String,
    #[serde(default)]
    pub y_suffix: String,
    #[serde(default)]
    pub y_precision: Option<usize>,
    /// Optional `[x_min_frac, x_max_frac]` normalized `0.0..=1.0` zoom window (managed automatically on wheel/drag).
    #[serde(default)]
    pub view_range: Option<[f64; 2]>,
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

const fn default_true() -> bool {
    true
}

const fn default_weight() -> f32 {
    1.0
}

impl Default for PlotData {
    fn default() -> Self {
        Self {
            title: None,
            subtitle: None,
            series: Vec::new(),
            panes: Vec::new(),
            thresholds: Vec::new(),
            show_grid: true,
            show_legend: true,
            crosshair: true,
            readout_mode: ReadoutMode::HeaderBar,
            stacked_percent: false,
            y_prefix: String::new(),
            y_suffix: String::new(),
            y_precision: None,
            view_range: None,
            empty_text: None,
            border_width: 0.0,
            border_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
        }
    }
}

pub fn measure_plot(_data: &PlotData, avail_w: f32) -> (f32, f32) {
    (280.0_f32.min(avail_w.max(180.0)), 300.0)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_plot(
    id: &str,
    data: &PlotData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 96.0 || rect.h < 96.0 {
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

    let pad = if rect.w < 340.0 { 12.0 } else { 16.0 };
    let ix = rect.x + pad;
    let iy = rect.y + pad;
    let iw = (rect.w - pad * 2.0).max(24.0);
    let ih = (rect.h - pad * 2.0).max(24.0);

    // Resolve unified pane list (explicit multi-pane OR synthesized single main pane)
    let fallback_pane;
    let panes: &[PlotPane] = if !data.panes.is_empty() {
        &data.panes
    } else {
        fallback_pane = [PlotPane {
            id: "main".to_string(),
            label: None,
            height_weight: 1.0,
            y_prefix: data.y_prefix.clone(),
            y_suffix: data.y_suffix.clone(),
            y_precision: data.y_precision,
            stacked_percent: data.stacked_percent,
            thresholds: data.thresholds.clone(),
            series: data.series.clone(),
        }];
        &fallback_pane
    };

    // Header row: title + subtitle + primary pane legend chips
    let primary_series = panes.first().map(|p| p.series.as_slice()).unwrap_or(&[]);
    let has_title = data.title.as_deref().is_some_and(|t| !t.is_empty());
    let has_sub = data.subtitle.as_deref().is_some_and(|s| !s.is_empty()) && rect.w >= 460.0;
    let show_legend = data.show_legend && !primary_series.is_empty() && rect.h >= 150.0;

    let mut header_h = 0.0f32;
    if has_title || show_legend {
        let title_px = if rect.w < 340.0 { 13.0 } else { 14.5 };
        let mut title_w_used = 0.0f32;
        if let Some(title) = &data.title {
            if !title.is_empty() {
                let max_t = if show_legend && rect.w >= 460.0 {
                    iw * 0.48
                } else {
                    iw
                };
                let fitted = fit_text(atlas, title, title_px, false, max_t);
                let (tw, _) = atlas.measure(&fitted, title_px, false);
                atlas.layout_dl(
                    &fitted,
                    ix,
                    atlas.centered_pen_y(iy, 24.0, title_px),
                    title_px,
                    theme.text_primary,
                    false,
                    None,
                    dl,
                );
                title_w_used = tw + 14.0;
                header_h = 24.0;
            }
        }
        let sub_budget = (iw * 0.65 - title_w_used).max(0.0);
        if has_sub && rect.w >= 560.0 && sub_budget >= 60.0 {
            if let Some(sub) = &data.subtitle {
                let sub_fit = fit_text(atlas, sub, 11.0, false, sub_budget);
                if !sub_fit.is_empty() {
                    atlas.layout_dl(
                        &sub_fit,
                        ix + title_w_used,
                        atlas.centered_pen_y(iy, 24.0, 11.0),
                        11.0,
                        theme.text_muted,
                        false,
                        None,
                        dl,
                    );
                }
            }
        }
        if show_legend {
            let legend_same_row = rect.w >= 460.0 || !has_title;
            let ly = if legend_same_row { iy } else { iy + 24.0 };
            header_h = if !legend_same_row { 46.0 } else { header_h.max(24.0) };
            emit_legend_chips(id, primary_series, ix, ly, iw, legend_same_row, theme, atlas, interaction, dl);
        }
        header_h += 6.0;
    }

    // Compute global X bounds across all visible series in all panes
    let Some((full_x_min, full_x_max, x_labels)) = compute_global_x_bounds(panes) else {
        let cy = iy + header_h + (ih - header_h) * 0.5;
        dl.push_stroke(StrokeInstance::solid([ix, cy], [ix + iw, cy], 1.0, 0.75, theme.border, 0));
        let empty_msg = data.empty_text.as_deref().unwrap_or("No data");
        let msg = fit_text(atlas, empty_msg, 12.5, false, iw);
        let (mw, mh) = atlas.measure(&msg, 12.5, false);
        atlas.layout_dl(
            &msg,
            ix + (iw - mw) * 0.5,
            cy - mh - 8.0,
            12.5,
            theme.text_muted,
            false,
            None,
            dl,
        );
        dl.pop_clip();
        return;
    };

    let [vf0, vf1] = data
        .view_range
        .unwrap_or([0.0, 1.0]);
    let vf0 = vf0.clamp(0.0, 0.96);
    let vf1 = vf1.clamp(vf0 + 0.04, 1.0);
    let full_x_span = (full_x_max - full_x_min).max(1e-9);
    let view_x_min = full_x_min + vf0 * full_x_span;
    let view_x_max = full_x_min + vf1 * full_x_span;

    // Determine unified right Y-axis gutter width across all panes so vertical grids stay aligned
    let mut max_gutter = 46.0f32;
    let x_axis_h = if rect.h >= 150.0 { 26.0 } else { 0.0 };
    let pane_gap = if panes.len() > 1 { 6.0 } else { 0.0 };
    let panes_avail_h = (ih - header_h - x_axis_h - pane_gap * (panes.len().saturating_sub(1)) as f32).max(40.0);
    let weight_sum: f32 = panes.iter().map(|p| p.height_weight.max(0.1)).sum();

    // Hover state & snapped crosshair X
    let hover_pt = if data.crosshair {
        interaction.plot_hover.get(id).copied()
    } else {
        None
    };

    // Pre-pass pane Y-bounds + tick string widths
    struct PaneLayout<'a> {
        pane: &'a PlotPane,
        y: f32,
        h: f32,
        y_min: f64,
        y_max: f64,
        ticks: Vec<(f32, String)>,
    }

    let mut pane_layouts = Vec::with_capacity(panes.len());
    let mut cur_y = iy + header_h;

    for (p_idx, pane) in panes.iter().enumerate() {
        let ph = (panes_avail_h * (pane.height_weight.max(0.1) / weight_sum)).max(44.0);
        let (raw_ymin, raw_ymax) = compute_pane_y_bounds(pane, view_x_min, view_x_max);
        let span_raw = (raw_ymax - raw_ymin).abs();
        let pad_y = if pane.stacked_percent {
            0.0
        } else if span_raw < 1e-6 {
            raw_ymax.abs().max(1.0) * 0.08
        } else {
            span_raw * 0.08
        };
        let t_ymin = (raw_ymin - pad_y) as f32;
        let t_ymax = (raw_ymax + pad_y) as f32;

        let spring_ns = format!("plot::{id}");
        let y_min = if pane.stacked_percent {
            0.0
        } else {
            springs.animate(&spring_ns, &format!("p{p_idx}_ymin"), t_ymin, SpringConfig::UI_DEFAULT) as f64
        };
        let y_max = if pane.stacked_percent {
            100.0
        } else {
            springs.animate(&spring_ns, &format!("p{p_idx}_ymax"), t_ymax, SpringConfig::UI_DEFAULT) as f64
        };
        let y_range = (y_max - y_min).max(1e-6);

        let tick_count: usize = if ph < 110.0 {
            3
        } else if ph < 220.0 {
            4
        } else {
            5
        };
        let mut ticks = Vec::with_capacity(tick_count);
        for t in 0..tick_count {
            let frac = t as f64 / (tick_count - 1).max(1) as f64;
            let val = y_min + frac * y_range;
            let txt = format!(
                "{}{}{}",
                pane.y_prefix,
                format_number(val, pane.y_precision),
                pane.y_suffix
            );
            let (tw, _) = atlas.measure(&txt, 10.5, false);
            max_gutter = max_gutter.max(tw + 14.0);
            ticks.push((frac as f32, txt));
        }

        pane_layouts.push(PaneLayout {
            pane,
            y: cur_y,
            h: ph,
            y_min,
            y_max,
            ticks,
        });
        cur_y += ph + pane_gap;
    }

    let gutter_r = max_gutter.min(iw * 0.25);
    let plot_x = ix;
    let plot_w = (iw - gutter_r - 4.0).max(32.0);
    let inner_pad_x = 10.0;
    let draw_x = plot_x + inner_pad_x;
    let draw_w = (plot_w - inner_pad_x * 2.0).max(8.0);
    let view_x_span = (view_x_max - view_x_min).max(1e-9);
    let to_px_x = |xv: f64| -> f32 { draw_x + (((xv - view_x_min) / view_x_span) as f32) * draw_w };

    // Shared outer plot hit area covering all stacked panes for zoom/pan/crosshair
    let all_panes_rect = ComputedRect {
        x: plot_x,
        y: iy + header_h,
        w: plot_w,
        h: panes_avail_h,
        content_h: panes_avail_h,
        folded: false,
    };
    if data.crosshair {
        interaction.push_hit(all_panes_rect, HitKind::PlotArea { id: id.to_string() });
    }

    // Determine hovered X value cleanly once across all panes
    let hovered_x_val = hover_pt.and_then(|[hx, hy]| {
        if hx >= plot_x && hx <= plot_x + plot_w && hy >= all_panes_rect.y && hy <= all_panes_rect.y + all_panes_rect.h {
            Some(view_x_min + (((hx - draw_x) / draw_w).clamp(0.0, 1.0) as f64) * view_x_span)
        } else {
            None
        }
    });

    // Vertical X gridlines across every pane (clean subtle ticks)
    let x_tick_count: usize = ((plot_w / 110.0).floor() as usize).clamp(3, 9);
    let mut x_grid_xs = Vec::with_capacity(x_tick_count);
    for t in 0..x_tick_count {
        let frac = t as f64 / (x_tick_count - 1).max(1) as f64;
        let xv = view_x_min + frac * view_x_span;
        let gx = to_px_x(xv);
        let lbl = resolve_x_label(&x_labels, xv);
        x_grid_xs.push((gx, lbl));
    }

    // Render each pane cleanly with deterministic Z-ordering via DrawList::cmds!
    let mut hovered_snap_x: Option<f32> = None;
    let mut hovered_snap_lbl: Option<String> = None;

    for (p_idx, pl) in pane_layouts.iter().enumerate() {
        let pane_rect = ComputedRect {
            x: plot_x,
            y: pl.y,
            w: plot_w,
            h: pl.h,
            content_h: pl.h,
            folded: false,
        };

        // 1. Clean borderless pane background well
        dl.push_quad(QuadInstance {
            rect: [pane_rect.x, pane_rect.y, pane_rect.w, pane_rect.h],
            radii: [8.0; 4],
            color_top: theme.surface_0,
            color_bottom: theme.surface_0,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });

        let draw_y = pl.y + (if data.readout_mode == ReadoutMode::HeaderBar { 26.0 } else { 12.0 });
        let draw_h = (pl.y + pl.h - 14.0 - draw_y).max(16.0);
        let y_range = (pl.y_max - pl.y_min).max(1e-6);
        let to_px_y = |yv: f64| -> f32 {
            draw_y + draw_h - (((yv - pl.y_min) / y_range) as f32).clamp(-0.02, 1.02) * draw_h
        };

        // 2. Horizontal + vertical subtle gridlines (inside pane, behind all chart geometry)
        dl.push_clip(ClipRect::new(pane_rect.x, pane_rect.y, pane_rect.w, pane_rect.h));
        if data.show_grid {
            let grid_col = scale_alpha(theme.border, 0.45);
            for &(gx, _) in &x_grid_xs {
                dl.push_stroke(StrokeInstance::solid(
                    [gx, pane_rect.y + 4.0],
                    [gx, pane_rect.y + pane_rect.h - 4.0],
                    1.0,
                    0.5,
                    grid_col,
                    0,
                ));
            }
            for (frac, label) in &pl.ticks {
                let gy = draw_y + draw_h - (*frac * draw_h);
                dl.push_stroke(StrokeInstance::solid(
                    [pane_rect.x + 4.0, gy],
                    [pane_rect.x + pane_rect.w - 4.0, gy],
                    1.0,
                    0.5,
                    grid_col,
                    0,
                ));
                // Right gutter label (drawn cleanly outside pane clip? Wait: draw after popping clip)
                let _ = label;
            }
        }

        // Reference dashed threshold lines (e.g. RSI 70/30, Stoch 80/20, target price alert lines)
        for th in &pl.pane.thresholds {
            if th.value >= pl.y_min && th.value <= pl.y_max {
                let ty = to_px_y(th.value);
                let col = th.color.unwrap_or(scale_alpha(theme.text_muted, 0.55));
                dl.push_stroke(StrokeInstance::dashed(
                    [pane_rect.x + 6.0, ty],
                    [pane_rect.x + pane_rect.w - 6.0, ty],
                    1.0,
                    0.75,
                    4.0,
                    4.0,
                    col,
                    0,
                ));
            }
        }

        // 3. Draw StackedArea + Band + Area smooth sloped Ribbons (behind candles & crisp strokes!)
        let stacked_series: Vec<(usize, &PlotSeries)> = pl
            .pane
            .series
            .iter()
            .enumerate()
            .filter(|(_, s)| s.visible && s.kind == PlotSeriesKind::StackedArea)
            .collect();
        if !stacked_series.is_empty() {
            emit_stacked_areas(
                &stacked_series,
                pl.pane.stacked_percent,
                view_x_min,
                view_x_max,
                draw_x,
                draw_w,
                &to_px_x,
                &to_px_y,
                theme,
                dl,
            );
        }

        for (s_idx, s) in pl.pane.series.iter().enumerate() {
            if !s.visible {
                continue;
            }
            let col = s.color.unwrap_or_else(|| palette_color(theme, s_idx + p_idx));
            match s.kind {
                PlotSeriesKind::Area => {
                    emit_area_ribbon(
                        &s.points,
                        view_x_min,
                        view_x_max,
                        draw_w,
                        draw_y + draw_h,
                        &to_px_x,
                        &to_px_y,
                        col,
                        dl,
                    );
                }
                PlotSeriesKind::Band => {
                    let col_b = s.secondary_color.unwrap_or_else(|| palette_color(theme, s_idx + 1));
                    emit_band_ribbon(
                        &s.points,
                        view_x_min,
                        view_x_max,
                        draw_w,
                        draw_y + draw_h,
                        &to_px_x,
                        &to_px_y,
                        col,
                        col_b,
                        dl,
                    );
                }
                _ => {}
            }
        }

        // 4. Draw Histogram + Candles + Lines cleanly ON TOP of ribbons
        for (s_idx, s) in pl.pane.series.iter().enumerate() {
            if !s.visible {
                continue;
            }
            let col = s.color.unwrap_or_else(|| palette_color(theme, s_idx + p_idx));
            match s.kind {
                PlotSeriesKind::Histogram => {
                    emit_histogram(&s.points, view_x_min, view_x_max, draw_w, draw_y, draw_h, &to_px_x, &to_px_y, col, theme, dl);
                }
                PlotSeriesKind::Candles => {
                    emit_candles(&s.candles, view_x_min, view_x_max, draw_w, &to_px_x, &to_px_y, theme, dl);
                }
                PlotSeriesKind::Line | PlotSeriesKind::Area => {
                    emit_line_stroke(
                        &s.points,
                        view_x_min,
                        view_x_max,
                        draw_w,
                        &to_px_x,
                        &to_px_y,
                        col,
                        s.stroke_width.unwrap_or(1.8),
                        s.dashed,
                        dl,
                    );
                }
                PlotSeriesKind::Band => {
                    emit_line_stroke(&s.points, view_x_min, view_x_max, draw_w, &to_px_x, &to_px_y, col, 1.5, false, dl);
                    // Secondary y2 stroke
                    let pts_y2: Vec<PlotPoint> = s
                        .points
                        .iter()
                        .filter_map(|p| p.y2.map(|v| PlotPoint { x: p.x, y: v, y2: None, label: p.label.clone() }))
                        .collect();
                    if !pts_y2.is_empty() {
                        let col_b = s.secondary_color.unwrap_or_else(|| palette_color(theme, s_idx + 1));
                        emit_line_stroke(&pts_y2, view_x_min, view_x_max, draw_w, &to_px_x, &to_px_y, col_b, 1.5, false, dl);
                    }
                }
                PlotSeriesKind::StackedArea => {}
            }
        }

        // 5. Current live value dotted horizontal line across pane + active hover dots
        let latest_y = latest_pane_y(pl.pane);
        let live_col = primary_pane_color(pl.pane, theme, p_idx);
        if let Some(ly) = latest_y {
            let py = to_px_y(ly).clamp(pane_rect.y + 8.0, pane_rect.y + pane_rect.h - 8.0);
            dl.push_stroke(StrokeInstance::dashed(
                [pane_rect.x + 6.0, py],
                [pane_rect.x + pane_rect.w - 4.0, py],
                1.0,
                0.75,
                2.5,
                3.5,
                scale_alpha(live_col, 0.72),
                0,
            ));
        }

        // Collect live or hovered header HUD readout strings (zero overlap over chart body!)
        let readout_lines = collect_pane_readout(pl.pane, hovered_x_val, theme, p_idx);

        // Draw snapped data point dot & horizontal crosshair inside this pane when hovered
        if let Some(target_x) = hovered_x_val {
            let snap_x = compute_pane_snap_x(pl.pane, target_x, &to_px_x).unwrap_or_else(|| to_px_x(target_x));
            if hovered_snap_x.is_none() {
                hovered_snap_x = Some(snap_x);
                hovered_snap_lbl = Some(resolve_x_label(&x_labels, target_x));
            }
            dl.push_stroke(StrokeInstance::dashed(
                [snap_x, pane_rect.y + 4.0],
                [snap_x, pane_rect.y + pane_rect.h - 4.0],
                1.0,
                0.75,
                3.5,
                3.5,
                scale_alpha(theme.text_secondary, 0.55),
                0,
            ));
            if let Some([_, hy]) = hover_pt {
                if hy >= pane_rect.y && hy <= pane_rect.y + pane_rect.h {
                    dl.push_stroke(StrokeInstance::dashed(
                        [pane_rect.x + 4.0, hy],
                        [pane_rect.x + pane_rect.w - 4.0, hy],
                        1.0,
                        0.75,
                        3.5,
                        3.5,
                        scale_alpha(theme.text_secondary, 0.45),
                        0,
                    ));
                }
            }
            // Highlight dots on active snapped points
            for (s_idx, s) in pl.pane.series.iter().enumerate() {
                if !s.visible || s.kind == PlotSeriesKind::StackedArea {
                    continue;
                }
                let col = s.color.unwrap_or_else(|| palette_color(theme, s_idx + p_idx));
                if let Some(c) = nearest_candle(&s.candles, target_x) {
                    let cy = to_px_y(c.close);
                    dl.push_quad(QuadInstance {
                        rect: [snap_x - 3.5, cy - 3.5, 7.0, 7.0],
                        radii: [3.5; 4],
                        color_top: theme.text_primary,
                        color_bottom: theme.text_primary,
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                } else if let Some(p) = nearest_point(&s.points, target_x) {
                    let py = to_px_y(p.y);
                    dl.push_quad(QuadInstance {
                        rect: [snap_x - 3.5, py - 3.5, 7.0, 7.0],
                        radii: [3.5; 4],
                        color_top: col,
                        color_bottom: col,
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                }
            }
        }

        // Non-occluding Top-Left Pane Readout Bar (`HeaderBar` mode — TradingView style!)
        if data.readout_mode == ReadoutMode::HeaderBar && !readout_lines.is_empty() {
            let mut rx = pane_rect.x + 10.0;
            let bar_h = 24.0;
            let ry_txt = atlas.centered_pen_y(pane_rect.y, bar_h, 10.5);
            if let Some(plabel) = &pl.pane.label {
                let pf = fit_text(atlas, plabel, 10.5, false, 120.0);
                let (pw, _) = atlas.measure(&pf, 10.5, false);
                atlas.layout_dl(&pf, rx, ry_txt, 10.5, theme.text_muted, false, None, dl);
                rx += pw + 12.0;
            }
            for (seg_txt, seg_col) in &readout_lines {
                let avail = (pane_rect.x + pane_rect.w - rx - 14.0).max(0.0);
                if avail < 40.0 {
                    break;
                }
                dl.push_quad(QuadInstance {
                    rect: [rx, pane_rect.y + (bar_h - 5.0) * 0.5, 5.0, 5.0],
                    radii: [2.5; 4],
                    color_top: *seg_col,
                    color_bottom: *seg_col,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
                rx += 9.0;
                let fitted = fit_text(atlas, seg_txt, 10.5, false, avail);
                let (sw, _) = atlas.measure(&fitted, 10.5, false);
                atlas.layout_dl(&fitted, rx, ry_txt, 10.5, *seg_col, false, None, dl);
                rx += sw + 12.0;
            }
        }

        dl.pop_clip(); // finish pane clip before painting right gutter Y-axis ticks & badges

        // 6. Right Y-axis gutter with strict priority collision suppression (no overlapping text!)
        let mut occupied_y: Vec<[f32; 2]> = Vec::new();
        let pill_h = 18.0f32;
        let overlaps = |occ: &[[f32; 2]], top: f32, bot: f32, pad: f32| -> bool {
            occ.iter().any(|&[a, b]| top < b + pad && bot > a - pad)
        };

        // Priority 1: Hovered cursor Y-axis pill
        let mut hover_badge: Option<(f32, String)> = None;
        if let Some([_, hy]) = hover_pt {
            if hy >= pl.y && hy <= pl.y + pl.h {
                let clamped_hy = hy.clamp(pl.y + pill_h * 0.5, pl.y + pl.h - pill_h * 0.5);
                let cursor_y_val =
                    pl.y_max - (((hy - draw_y) / draw_h).clamp(0.0, 1.0) as f64) * y_range;
                let ytxt = format!(
                    "{}{}{}",
                    pl.pane.y_prefix,
                    format_number(cursor_y_val, pl.pane.y_precision),
                    pl.pane.y_suffix
                );
                occupied_y.push([clamped_hy - pill_h * 0.5, clamped_hy + pill_h * 0.5]);
                hover_badge = Some((clamped_hy, ytxt));
            }
        }

        // Priority 2: Live right-edge price pill (render if not occluded by hover pill)
        if let Some(ly) = latest_y {
            let py = to_px_y(ly).clamp(pl.y + pill_h * 0.5, pl.y + pl.h - pill_h * 0.5);
            let top = py - pill_h * 0.5;
            let bot = py + pill_h * 0.5;
            if !overlaps(&occupied_y, top, bot, 2.0) {
                occupied_y.push([top, bot]);
                let btxt = format!(
                    "{}{}{}",
                    pl.pane.y_prefix,
                    format_number(ly, pl.pane.y_precision),
                    pl.pane.y_suffix
                );
                let fitted_b = fit_text(atlas, &btxt, 10.5, false, gutter_r - 8.0);
                let (bw, _) = atlas.measure(&fitted_b, 10.5, false);
                let pill_w = (bw + 10.0).min(gutter_r + 2.0);
                dl.push_quad(QuadInstance {
                    rect: [plot_x + plot_w + 4.0, top, pill_w, pill_h],
                    radii: [4.0; 4],
                    color_top: live_col,
                    color_bottom: live_col,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
                atlas.layout_centered(
                    &fitted_b,
                    plot_x + plot_w + 4.0,
                    top,
                    pill_w,
                    pill_h,
                    10.5,
                    theme.bg,
                    false,
                    dl,
                );
            }
        }

        // Priority 3: Threshold badges (`80`, `20`, `50`)
        for th in &pl.pane.thresholds {
            if th.value >= pl.y_min && th.value <= pl.y_max {
                let th_h = 16.0f32;
                let ty = to_px_y(th.value).clamp(pl.y + th_h * 0.5, pl.y + pl.h - th_h * 0.5);
                let top = ty - th_h * 0.5;
                let bot = ty + th_h * 0.5;
                if overlaps(&occupied_y, top, bot, 2.0) {
                    continue;
                }
                occupied_y.push([top, bot]);
                let ttxt = th
                    .label
                    .clone()
                    .unwrap_or_else(|| format_number(th.value, pl.pane.y_precision));
                let fitted_t = fit_text(atlas, &ttxt, 10.0, false, gutter_r - 8.0);
                let (tw, _) = atlas.measure(&fitted_t, 10.0, false);
                let pill_w = (tw + 10.0).min(gutter_r);
                let badge_col = th.color.unwrap_or(theme.surface_3);
                dl.push_quad(QuadInstance {
                    rect: [plot_x + plot_w + 4.0, top, pill_w, th_h],
                    radii: [4.0; 4],
                    color_top: badge_col,
                    color_bottom: badge_col,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
                atlas.layout_centered(
                    &fitted_t,
                    plot_x + plot_w + 4.0,
                    top,
                    pill_w,
                    th_h,
                    10.0,
                    theme.text_primary,
                    false,
                    dl,
                );
            }
        }

        // Priority 4: Grid tick labels (suppressed cleanly if intersecting any active badge!)
        if data.show_grid {
            for (frac, label) in &pl.ticks {
                let gy = draw_y + draw_h - (*frac * draw_h);
                let top = gy - 7.5;
                let bot = gy + 7.5;
                if overlaps(&occupied_y, top, bot, 4.5) {
                    continue;
                }
                let fitted_y = fit_text(atlas, label, 10.5, false, gutter_r - 6.0);
                atlas.layout_dl(
                    &fitted_y,
                    plot_x + plot_w + 8.0,
                    atlas.centered_pen_y(gy - 8.0, 16.0, 10.5),
                    10.5,
                    theme.text_muted,
                    false,
                    None,
                    dl,
                );
            }
        }

        // Paint hovered cursor badge cleanly on top
        if let Some((clamped_hy, ytxt)) = hover_badge {
            let top = clamped_hy - pill_h * 0.5;
            let fitted_y = fit_text(atlas, &ytxt, 10.5, false, gutter_r - 8.0);
            let (yw, _) = atlas.measure(&fitted_y, 10.5, false);
            let pill_w = (yw + 10.0).min(gutter_r + 2.0);
            dl.push_quad(QuadInstance {
                rect: [plot_x + plot_w + 4.0, top, pill_w, pill_h],
                radii: [4.0; 4],
                color_top: theme.surface_3,
                color_bottom: theme.surface_3,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            atlas.layout_centered(
                &fitted_y,
                plot_x + plot_w + 4.0,
                top,
                pill_w,
                pill_h,
                10.5,
                theme.text_primary,
                false,
                dl,
            );
        }
    }

    // 7. Shared bottom X-axis ruler + hovered X-axis timestamp pill
    if x_axis_h > 0.0 {
        let axis_y = iy + header_h + panes_avail_h + 6.0;
        let axis_h = 18.0f32;
        let mut prev_right = plot_x - 20.0;
        for (gx, lbl) in &x_grid_xs {
            let (lw, _) = atlas.measure(lbl, 10.5, false);
            let lx = (gx - lw * 0.5).clamp(plot_x, (plot_x + plot_w - lw).max(plot_x));
            if lx > prev_right + 14.0 {
                atlas.layout_dl(
                    lbl,
                    lx,
                    atlas.centered_pen_y(axis_y, axis_h, 10.5),
                    10.5,
                    theme.text_muted,
                    false,
                    None,
                    dl,
                );
                prev_right = lx + lw;
            }
        }

        if let (Some(sx), Some(slbl)) = (hovered_snap_x, hovered_snap_lbl) {
            let (sw, _) = atlas.measure(&slbl, 10.5, false);
            let pw = sw + 12.0;
            let px = (sx - pw * 0.5).clamp(plot_x, (plot_x + plot_w - pw).max(plot_x));
            dl.push_quad(QuadInstance {
                rect: [px, axis_y, pw, axis_h],
                radii: [4.0; 4],
                color_top: theme.surface_3,
                color_bottom: theme.surface_3,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            atlas.layout_centered(&slbl, px, axis_y, pw, axis_h, 10.5, theme.text_primary, false, dl);
        }
    }

    dl.pop_clip(); // outer plot clip
}

#[allow(clippy::too_many_arguments)]
fn emit_legend_chips(
    plot_id: &str,
    series: &[PlotSeries],
    ix: f32,
    y: f32,
    iw: f32,
    right_aligned: bool,
    theme: &Theme,
    atlas: &GlyphAtlas,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    let mut chips: Vec<(usize, String, f32, u32, bool)> = Vec::new();
    let mut total_w = 0.0f32;
    for (idx, s) in series.iter().enumerate() {
        let name = fit_text(atlas, &s.name, 11.0, false, 120.0);
        if name.is_empty() {
            continue;
        }
        let (nw, _) = atlas.measure(&name, 11.0, false);
        let cw = nw + 22.0;
        if total_w + cw > iw * (if right_aligned { 0.54 } else { 1.0 }) && !chips.is_empty() {
            break;
        }
        total_w += cw + 6.0;
        chips.push((idx, name, cw, s.color.unwrap_or_else(|| palette_color(theme, idx)), s.visible));
    }

    let mut cx = if right_aligned {
        ix + iw - (total_w - 6.0).max(0.0)
    } else {
        ix
    };

    for (idx, name, cw, col, vis) in chips {
        let chip_rect = ComputedRect {
            x: cx,
            y,
            w: cw,
            h: 22.0,
            content_h: 22.0,
            folded: false,
        };
        interaction.push_hit(
            chip_rect,
            HitKind::PlotLegendChip {
                plot_id: plot_id.to_string(),
                series_index: idx,
            },
        );
        let hovered = interaction.hovered_key.as_deref()
            == Some(format!("{plot_id}::legend::{idx}").as_str());
        dl.push_quad(QuadInstance {
            rect: [cx, y, cw, 22.0],
            radii: [6.0; 4],
            color_top: if hovered { theme.surface_3 } else { theme.surface_2 },
            color_bottom: if hovered { theme.surface_3 } else { theme.surface_2 },
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, if vis { 1.0 } else { 0.45 }],
            clip_id: 0,
            _pad: [0; 3],
        });
        dl.push_quad(QuadInstance {
            rect: [cx + 7.0, y + 7.5, 7.0, 7.0],
            radii: [3.5; 4],
            color_top: col,
            color_bottom: col,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, if vis { 1.0 } else { 0.35 }],
            clip_id: 0,
            _pad: [0; 3],
        });
        atlas.layout_centered(
            &name,
            cx + 16.0,
            y,
            (cw - 20.0).max(8.0),
            22.0,
            11.0,
            if vis { theme.text_primary } else { theme.text_muted },
            false,
            dl,
        );
        cx += cw + 6.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_area_ribbon(
    points: &[PlotPoint],
    view_x_min: f64,
    view_x_max: f64,
    draw_w: f32,
    base_y: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    color: u32,
    dl: &mut DrawList,
) {
    let pts = visible_decimated_points(points, view_x_min, view_x_max, draw_w, to_px_x, to_px_y);
    for win in pts.windows(2) {
        let [x0, y0] = win[0];
        let [x1, y1] = win[1];
        dl.push_ribbon(RibbonInstance {
            x0,
            y0_top: y0,
            y0_bot: base_y,
            x1,
            y1_top: y1,
            y1_bot: base_y,
            color_top: scale_alpha(color, 0.28),
            color_bottom: scale_alpha(color, 0.02),
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_band_ribbon(
    points: &[PlotPoint],
    view_x_min: f64,
    view_x_max: f64,
    _draw_w: f32,
    base_y: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    color_a: u32,
    color_b: u32,
    dl: &mut DrawList,
) {
    let vis: Vec<&PlotPoint> = points
        .iter()
        .filter(|p| p.x >= view_x_min - 1e-6 && p.x <= view_x_max + 1e-6)
        .collect();
    for win in vis.windows(2) {
        let p0 = win[0];
        let p1 = win[1];
        let x0 = to_px_x(p0.x);
        let x1 = to_px_x(p1.x);
        let y0a = to_px_y(p0.y);
        let y1a = to_px_y(p1.y);
        let y0b = p0.y2.map(to_px_y).unwrap_or(base_y);
        let y1b = p1.y2.map(to_px_y).unwrap_or(base_y);
        let tint = if p1.y >= p1.y2.unwrap_or(0.0) { color_a } else { color_b };
        dl.push_ribbon(RibbonInstance {
            x0,
            y0_top: y0a,
            y0_bot: y0b,
            x1,
            y1_top: y1a,
            y1_bot: y1b,
            color_top: scale_alpha(tint, 0.28),
            color_bottom: scale_alpha(tint, 0.16),
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_stacked_areas(
    stacked: &[(usize, &PlotSeries)],
    percent_mode: bool,
    view_x_min: f64,
    view_x_max: f64,
    _draw_x: f32,
    _draw_w: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    theme: &Theme,
    dl: &mut DrawList,
) {
    let base_pts: Vec<&PlotPoint> = stacked[0]
        .1
        .points
        .iter()
        .filter(|p| p.x >= view_x_min - 1e-6 && p.x <= view_x_max + 1e-6)
        .collect();
    if base_pts.len() < 2 {
        return;
    }

    let n_pts = base_pts.len();
    let mut prev_top_y: Vec<f64> = vec![0.0; n_pts];
    let mut totals: Vec<f64> = vec![0.0; n_pts];
    if percent_mode {
        for (i, bp) in base_pts.iter().enumerate() {
            let mut sum = 0.0;
            for &(_, s) in stacked {
                if let Some(p) = nearest_point(&s.points, bp.x) {
                    sum += p.y.max(0.0);
                }
            }
            totals[i] = sum.max(1e-9);
        }
    }

    for &(s_idx, s) in stacked {
        let col = s.color.unwrap_or_else(|| palette_color(theme, s_idx));
        let mut next_top_y = Vec::with_capacity(n_pts);
        for (i, bp) in base_pts.iter().enumerate() {
            let raw = nearest_point(&s.points, bp.x).map(|p| p.y.max(0.0)).unwrap_or(0.0);
            let add = if percent_mode {
                (raw / totals[i]) * 100.0
            } else {
                raw
            };
            next_top_y.push(prev_top_y[i] + add);
        }

        for i in 0..(n_pts - 1) {
            let x0 = to_px_x(base_pts[i].x);
            let x1 = to_px_x(base_pts[i + 1].x);
            let y0_bot = to_px_y(prev_top_y[i]);
            let y1_bot = to_px_y(prev_top_y[i + 1]);
            let y0_top = to_px_y(next_top_y[i]);
            let y1_top = to_px_y(next_top_y[i + 1]);
            dl.push_ribbon(RibbonInstance {
                x0,
                y0_top,
                y0_bot,
                x1,
                y1_top,
                y1_bot,
                color_top: scale_alpha(col, 0.90),
                color_bottom: scale_alpha(col, 0.80),
                clip_id: 0,
                _pad: [0; 3],
            });
        }
        prev_top_y = next_top_y;
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_line_stroke(
    points: &[PlotPoint],
    view_x_min: f64,
    view_x_max: f64,
    draw_w: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    color: u32,
    width: f32,
    dashed: bool,
    dl: &mut DrawList,
) {
    let pts = visible_decimated_points(points, view_x_min, view_x_max, draw_w, to_px_x, to_px_y);
    if pts.len() == 1 {
        let [px, py] = pts[0];
        dl.push_quad(QuadInstance {
            rect: [px - 3.5, py - 3.5, 7.0, 7.0],
            radii: [3.5; 4],
            color_top: color,
            color_bottom: color,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
        return;
    }
    for win in pts.windows(2) {
        if dashed {
            dl.push_stroke(StrokeInstance::dashed(win[0], win[1], width, 0.85, 5.0, 3.5, color, 0));
        } else {
            dl.push_stroke(StrokeInstance::solid(win[0], win[1], width, 0.85, color, 0));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_candles(
    candles: &[CandlePoint],
    view_x_min: f64,
    view_x_max: f64,
    draw_w: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    theme: &Theme,
    dl: &mut DrawList,
) {
    let vis: Vec<&CandlePoint> = candles
        .iter()
        .filter(|c| c.x >= view_x_min - 1e-6 && c.x <= view_x_max + 1e-6)
        .collect();
    if vis.is_empty() {
        return;
    }

    let max_bins = ((draw_w / 5.5).floor() as usize).max(16);
    let binned: Vec<CandlePoint>;
    let src: Vec<&CandlePoint> = if vis.len() > max_bins {
        let step = vis.len() as f32 / max_bins as f32;
        let mut out = Vec::with_capacity(max_bins);
        for b in 0..max_bins {
            let s = ((b as f32 * step).floor() as usize).min(vis.len() - 1);
            let e = (((b + 1) as f32 * step).ceil() as usize).clamp(s + 1, vis.len());
            let first = vis[s];
            let last = vis[e - 1];
            let mut hi = first.high;
            let mut lo = first.low;
            for c in &vis[s..e] {
                hi = hi.max(c.high);
                lo = lo.min(c.low);
            }
            out.push(CandlePoint {
                x: (first.x + last.x) * 0.5,
                open: first.open,
                high: hi,
                low: lo,
                close: last.close,
                label: last.label.clone(),
            });
        }
        binned = out;
        binned.iter().collect()
    } else {
        vis
    };

    // Pro TradingView candle body proportions (`2.0..=9.0px` with clean spacing)
    let slot_w = (draw_w / src.len().max(1) as f32).max(2.5);
    let body_w = (slot_w * 0.58).clamp(2.0, 9.0);

    for c in src {
        let cx = to_px_x(c.x);
        let yh = to_px_y(c.high);
        let yl = to_px_y(c.low);
        let yo = to_px_y(c.open);
        let yc = to_px_y(c.close);

        let bull = c.close >= c.open;
        let col = if bull { theme.ok } else { theme.danger };
        let wick_col = lerp_rgba8(col, theme.text_secondary, 0.15);

        dl.push_stroke(StrokeInstance::solid([cx, yh], [cx, yl], 1.1, 0.55, wick_col, 0));

        let top = yo.min(yc);
        let h = (yo - yc).abs().max(1.8);
        dl.push_quad(QuadInstance {
            rect: [cx - body_w * 0.5, top, body_w, h],
            radii: [1.25; 4],
            color_top: col,
            color_bottom: col,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_histogram(
    points: &[PlotPoint],
    view_x_min: f64,
    view_x_max: f64,
    draw_w: f32,
    draw_y: f32,
    draw_h: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
    color: u32,
    theme: &Theme,
    dl: &mut DrawList,
) {
    let vis: Vec<&PlotPoint> = points
        .iter()
        .filter(|p| p.x >= view_x_min - 1e-6 && p.x <= view_x_max + 1e-6)
        .collect();
    if vis.is_empty() {
        return;
    }
    let bar_w = ((draw_w / vis.len().max(1) as f32) * 0.56).clamp(1.8, 10.0);
    let base_y = draw_y + draw_h;
    for (idx, p) in vis.iter().enumerate() {
        let cx = to_px_x(p.x);
        let py = to_px_y(p.y).min(base_y);
        let h = (base_y - py).max(1.5);
        let bar_col = if p.y2.is_some_and(|prev| p.y >= prev) || (idx > 0 && p.y >= vis[idx - 1].y) {
            theme.ok
        } else if idx > 0 && p.y < vis[idx - 1].y {
            theme.danger
        } else {
            color
        };
        dl.push_quad(QuadInstance {
            rect: [cx - bar_w * 0.5, py, bar_w, h],
            radii: [1.5, 1.5, 0.0, 0.0],
            color_top: scale_alpha(bar_col, 0.45),
            color_bottom: scale_alpha(bar_col, 0.16),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

fn visible_decimated_points(
    points: &[PlotPoint],
    view_x_min: f64,
    view_x_max: f64,
    draw_w: f32,
    to_px_x: &dyn Fn(f64) -> f32,
    to_px_y: &dyn Fn(f64) -> f32,
) -> Vec<[f32; 2]> {
    let vis: Vec<&PlotPoint> = points
        .iter()
        .filter(|p| p.x >= view_x_min - 1e-6 && p.x <= view_x_max + 1e-6)
        .collect();
    if vis.is_empty() {
        return Vec::new();
    }
    let max_buckets = (draw_w.round() as usize).clamp(16, 1200);
    if vis.len() <= max_buckets {
        return vis.iter().map(|p| [to_px_x(p.x), to_px_y(p.y)]).collect();
    }
    let step = vis.len() as f32 / max_buckets as f32;
    let mut out = Vec::with_capacity(max_buckets * 2);
    for b in 0..max_buckets {
        let s = ((b as f32 * step).floor() as usize).min(vis.len() - 1);
        let e = (((b + 1) as f32 * step).ceil() as usize).clamp(s + 1, vis.len());
        let mut min_p = vis[s];
        let mut max_p = vis[s];
        for &p in &vis[s..e] {
            if p.y < min_p.y {
                min_p = p;
            }
            if p.y > max_p.y {
                max_p = p;
            }
        }
        if min_p.x <= max_p.x {
            out.push([to_px_x(min_p.x), to_px_y(min_p.y)]);
            if !std::ptr::eq(min_p, max_p) {
                out.push([to_px_x(max_p.x), to_px_y(max_p.y)]);
            }
        } else {
            out.push([to_px_x(max_p.x), to_px_y(max_p.y)]);
            out.push([to_px_x(min_p.x), to_px_y(min_p.y)]);
        }
    }
    out
}

fn collect_pane_readout(
    pane: &PlotPane,
    hovered_x: Option<f64>,
    theme: &Theme,
    pane_idx: usize,
) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    for (s_idx, s) in pane.series.iter().enumerate() {
        if !s.visible {
            continue;
        }
        let col = s.color.unwrap_or_else(|| palette_color(theme, s_idx + pane_idx));
        match s.kind {
            PlotSeriesKind::Candles => {
                let c = match hovered_x {
                    Some(tx) => nearest_candle(&s.candles, tx),
                    None => s.candles.last(),
                };
                if let Some(c) = c {
                    let pct = if c.open.abs() > 1e-9 {
                        ((c.close - c.open) / c.open) * 100.0
                    } else {
                        0.0
                    };
                    let tone_col = if c.close >= c.open { theme.ok } else { theme.danger };
                    out.push((
                        format!(
                            "{}: O {} H {} L {} C {} ({:+.2}%)",
                            s.name,
                            format_number(c.open, pane.y_precision),
                            format_number(c.high, pane.y_precision),
                            format_number(c.low, pane.y_precision),
                            format_number(c.close, pane.y_precision),
                            pct
                        ),
                        tone_col,
                    ));
                }
            }
            PlotSeriesKind::Band => {
                let p = match hovered_x {
                    Some(tx) => nearest_point(&s.points, tx),
                    None => s.points.last(),
                };
                if let Some(p) = p {
                    let y2s = p
                        .y2
                        .map(|v| format!(" / {}", format_number(v, pane.y_precision)))
                        .unwrap_or_default();
                    out.push((
                        format!("{}: {}{}", s.name, format_number(p.y, pane.y_precision), y2s),
                        col,
                    ));
                }
            }
            _ => {
                let p = match hovered_x {
                    Some(tx) => nearest_point(&s.points, tx),
                    None => s.points.last(),
                };
                if let Some(p) = p {
                    out.push((
                        format!(
                            "{}: {}{}{}",
                            s.name,
                            pane.y_prefix,
                            format_number(p.y, pane.y_precision),
                            pane.y_suffix
                        ),
                        col,
                    ));
                }
            }
        }
    }
    out
}

fn compute_pane_snap_x(pane: &PlotPane, target_x: f64, to_px_x: &dyn Fn(f64) -> f32) -> Option<f32> {
    for s in &pane.series {
        if !s.visible {
            continue;
        }
        if let Some(c) = nearest_candle(&s.candles, target_x) {
            return Some(to_px_x(c.x));
        }
        if let Some(p) = nearest_point(&s.points, target_x) {
            return Some(to_px_x(p.x));
        }
    }
    None
}

fn compute_global_x_bounds(panes: &[PlotPane]) -> Option<(f64, f64, Vec<(f64, String)>)> {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut p0_min = f64::INFINITY;
    let mut p0_max = f64::NEG_INFINITY;
    let mut labels = Vec::new();

    for (p_idx, pane) in panes.iter().enumerate() {
        for s in &pane.series {
            if !s.visible {
                continue;
            }
            for p in &s.points {
                if p.x.is_finite() {
                    x_min = x_min.min(p.x);
                    x_max = x_max.max(p.x);
                    if p_idx == 0 {
                        p0_min = p0_min.min(p.x);
                        p0_max = p0_max.max(p.x);
                    }
                    if labels.len() < 1024 {
                        if let Some(lbl) = &p.label {
                            labels.push((p.x, lbl.clone()));
                        }
                    }
                }
            }
            for c in &s.candles {
                if c.x.is_finite() {
                    x_min = x_min.min(c.x);
                    x_max = x_max.max(c.x);
                    if p_idx == 0 {
                        p0_min = p0_min.min(c.x);
                        p0_max = p0_max.max(c.x);
                    }
                    if labels.len() < 1024 {
                        if let Some(lbl) = &c.label {
                            labels.push((c.x, lbl.clone()));
                        }
                    }
                }
            }
        }
    }

    // In synchronized multi-pane layouts, lock active rolling window to primary pane when populated
    // so an inactive or shorter secondary pane never strands data in half the chart.
    if p0_min.is_finite() && p0_max.is_finite() && (p0_max - p0_min) > 1e-9 {
        x_min = p0_min;
        x_max = p0_max;
    }

    if !x_min.is_finite() || !x_max.is_finite() {
        return None;
    }
    if (x_max - x_min).abs() < 1e-9 {
        x_min -= 1.0;
        x_max += 1.0;
    }
    labels.sort_by(|a, b| a.0.total_cmp(&b.0));
    labels.retain(|(x, _)| *x >= x_min - 1e-6 && *x <= x_max + 1e-6);
    Some((x_min, x_max, labels))
}

fn compute_pane_y_bounds(pane: &PlotPane, x_min: f64, x_max: f64) -> (f64, f64) {
    if pane.stacked_percent {
        return (0.0, 100.0);
    }
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    let stacked: Vec<&PlotSeries> = pane
        .series
        .iter()
        .filter(|s| s.visible && s.kind == PlotSeriesKind::StackedArea)
        .collect();
    if !stacked.is_empty() {
        for p in &stacked[0].points {
            if p.x >= x_min - 1e-6 && p.x <= x_max + 1e-6 {
                let mut sum = 0.0;
                for s in &stacked {
                    sum += nearest_point(&s.points, p.x).map(|pt| pt.y.max(0.0)).unwrap_or(0.0);
                }
                y_min = y_min.min(0.0);
                y_max = y_max.max(sum);
            }
        }
    }

    for s in &pane.series {
        if !s.visible || s.kind == PlotSeriesKind::StackedArea {
            continue;
        }
        for p in &s.points {
            if p.x >= x_min - 1e-6 && p.x <= x_max + 1e-6 && p.y.is_finite() {
                y_min = y_min.min(p.y);
                y_max = y_max.max(p.y);
                if let Some(y2) = p.y2 {
                    if y2.is_finite() {
                        y_min = y_min.min(y2);
                        y_max = y_max.max(y2);
                    }
                }
            }
        }
        for c in &s.candles {
            if c.x >= x_min - 1e-6 && c.x <= x_max + 1e-6 && c.low.is_finite() && c.high.is_finite() {
                y_min = y_min.min(c.low);
                y_max = y_max.max(c.high);
            }
        }
    }

    for th in &pane.thresholds {
        if th.value.is_finite() {
            y_min = y_min.min(th.value);
            y_max = y_max.max(th.value);
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        return (0.0, 100.0);
    }
    (y_min, y_max)
}

fn resolve_x_label(labels: &[(f64, String)], target_x: f64) -> String {
    if labels.is_empty() {
        return format_number(target_x, Some(0));
    }
    let idx = labels.partition_point(|(x, _)| *x < target_x);
    if idx == 0 {
        return labels[0].1.clone();
    }
    if idx >= labels.len() {
        return labels[labels.len() - 1].1.clone();
    }
    let l = &labels[idx - 1];
    let r = &labels[idx];
    if (target_x - l.0).abs() <= (r.0 - target_x).abs() {
        l.1.clone()
    } else {
        r.1.clone()
    }
}

fn nearest_point(points: &[PlotPoint], target_x: f64) -> Option<&PlotPoint> {
    if points.is_empty() {
        return None;
    }
    let idx = points.partition_point(|p| p.x < target_x);
    if idx == 0 {
        return Some(&points[0]);
    }
    if idx >= points.len() {
        return Some(&points[points.len() - 1]);
    }
    let left = &points[idx - 1];
    let right = &points[idx];
    if (target_x - left.x).abs() <= (right.x - target_x).abs() {
        Some(left)
    } else {
        Some(right)
    }
}

fn nearest_candle(candles: &[CandlePoint], target_x: f64) -> Option<&CandlePoint> {
    if candles.is_empty() {
        return None;
    }
    let idx = candles.partition_point(|c| c.x < target_x);
    if idx == 0 {
        return Some(&candles[0]);
    }
    if idx >= candles.len() {
        return Some(&candles[candles.len() - 1]);
    }
    let left = &candles[idx - 1];
    let right = &candles[idx];
    if (target_x - left.x).abs() <= (right.x - target_x).abs() {
        Some(left)
    } else {
        Some(right)
    }
}

fn latest_pane_y(pane: &PlotPane) -> Option<f64> {
    for s in &pane.series {
        if !s.visible || s.kind == PlotSeriesKind::StackedArea {
            continue;
        }
        if let Some(c) = s.candles.last() {
            return Some(c.close);
        }
        if let Some(p) = s.points.last() {
            return Some(p.y);
        }
    }
    None
}

fn primary_pane_color(pane: &PlotPane, theme: &Theme, pane_idx: usize) -> u32 {
    for (idx, s) in pane.series.iter().enumerate() {
        if !s.visible {
            continue;
        }
        if let Some(c) = s.candles.last() {
            return if c.close >= c.open { theme.ok } else { theme.danger };
        }
        return s.color.unwrap_or_else(|| palette_color(theme, idx + pane_idx));
    }
    theme.accent
}

#[inline]
fn palette_color(theme: &Theme, idx: usize) -> u32 {
    match idx % 5 {
        0 => theme.info,
        1 => theme.accent,
        2 => theme.ok,
        3 => theme.warn,
        _ => theme.danger,
    }
}
