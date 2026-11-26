//! `Stat` component: precision live KPI tile with counter springs, semantic delta pill,
//! smooth sloped-ribbon SDF sparkline, range gauge variant, and 3-tier responsive adaptation.

use serde::{Deserialize, Serialize};

use crate::components::{fit_text, format_number};
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance, RibbonInstance, StrokeInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, Tone};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatVariant {
    #[default]
    Sparkline,
    /// Horizontal min..max range gauge (e.g. 24h High/Low band with live marker dot).
    RangeBar,
    /// Minimal high-contrast metric focus tile without chart strip.
    Compact,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StatData {
    pub label: String,
    #[serde(default)]
    pub badge: Option<String>,
    #[serde(default)]
    pub variant: StatVariant,
    pub value: f64,
    #[serde(default)]
    pub value_text: Option<String>,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub precision: Option<usize>,
    #[serde(default)]
    pub delta: Option<String>,
    #[serde(default)]
    pub tone: Tone,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub sparkline: Vec<f32>,
    /// Optional `[low, high]` bounds used by `StatVariant::RangeBar`.
    #[serde(default)]
    pub range_bounds: Option<[f64; 2]>,
    #[serde(default = "default_true")]
    pub animate_value: bool,
    #[serde(default = "default_true")]
    pub interactive: bool,
    /// Explicit opt-in outline width (defaults to `0.0` — no outline).
    #[serde(default)]
    pub border_width: f32,
    #[serde(default)]
    pub border_color: Option<u32>,
    /// Explicit opt-in drop shadow blur (defaults to `0.0` — shadows disabled unless requested).
    #[serde(default)]
    pub shadow_blur: f32,
    #[serde(default)]
    pub shadow_offset_y: f32,
    #[serde(default)]
    pub update_seq: u64,
}

const fn default_true() -> bool {
    true
}

impl Default for StatData {
    fn default() -> Self {
        Self {
            label: String::new(),
            badge: None,
            variant: StatVariant::Sparkline,
            value: 0.0,
            value_text: None,
            prefix: String::new(),
            suffix: String::new(),
            precision: None,
            delta: None,
            tone: Tone::Neutral,
            caption: None,
            sparkline: Vec::new(),
            range_bounds: None,
            animate_value: true,
            interactive: true,
            border_width: 0.0,
            border_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
            update_seq: 0,
        }
    }
}

/// Measure preferred height given available width (automatically adjusts density).
pub fn measure_stat(data: &StatData, avail_w: f32) -> (f32, f32) {
    let has_visual = match data.variant {
        StatVariant::Sparkline => !data.sparkline.is_empty(),
        StatVariant::RangeBar => data.range_bounds.is_some(),
        StatVariant::Compact => false,
    };
    let has_caption = data.caption.as_deref().is_some_and(|c| !c.is_empty());
    let pref_h = if avail_w < 190.0 {
        if has_visual { 120.0 } else { 92.0 }
    } else if has_visual && has_caption {
        154.0
    } else if has_visual || has_caption {
        124.0
    } else {
        92.0
    };
    (avail_w.clamp(170.0, 230.0), pref_h)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_stat(
    id: &str,
    data: &StatData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 40.0 || rect.h < 32.0 {
        return;
    }

    let hit_key = format!("stat::{id}");
    let hover_t = if data.interactive {
        let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
        let pressed = interaction.pressed_key.as_deref() == Some(hit_key.as_str());
        let target = if pressed {
            1.0
        } else if hovered {
            0.65
        } else {
            0.0
        };
        interaction.push_hit(rect, HitKind::Stat { id: id.to_string() });
        springs.animate(&hit_key, "stat_hover", target, SpringConfig::SNAPPY)
    } else {
        0.0
    };

    let last_seq_key = format!("stat_seq::{id}");
    let prev_seq = springs.get(&last_seq_key, "seq", data.update_seq as f32);
    let pulse_target = if (data.update_seq as f32 - prev_seq).abs() >= 0.5 {
        springs.animate(&last_seq_key, "seq", data.update_seq as f32, SpringConfig::SNAPPY);
        1.0
    } else {
        0.0
    };
    let pulse_t = springs.animate(&hit_key, "stat_pulse", pulse_target, SpringConfig::GENTLE);

    let tone_col = if data.tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(data.tone)
    };

    let bg_top = lerp_rgba8(
        lerp_rgba8(theme.surface_1, theme.surface_2, hover_t * 0.85),
        scale_alpha(tone_col, 0.14),
        pulse_t * 0.25,
    );
    let bg_bot = lerp_rgba8(theme.surface_1, theme.surface_2, hover_t * 0.70);
    let border_col = if data.border_width > 0.0 {
        data.border_color.unwrap_or(theme.border)
    } else {
        0
    };
    let shadow_col = if data.shadow_blur > 0.0 {
        theme.shadow
    } else {
        0
    };

    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [theme.radius_card; 4],
        color_top: bg_top,
        color_bottom: bg_bot,
        border_color: border_col,
        shadow_color: shadow_col,
        params: [data.border_width, data.shadow_blur, data.shadow_offset_y, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let pad_x = if rect.w < 200.0 { 14.0 } else { 18.0 };
    let pad_y = if rect.h < 90.0 { 12.0 } else { 16.0 };
    let ix = rect.x + pad_x;
    let iy = rect.y + pad_y;
    let iw = (rect.w - pad_x * 2.0).max(16.0);
    let ih = (rect.h - pad_y * 2.0).max(16.0);

    let disp_val = if data.animate_value && data.value_text.is_none() && data.value.is_finite() {
        springs.animate(id, "stat_val", data.value as f32, SpringConfig::UI_DEFAULT) as f64
    } else {
        data.value
    };

    let formatted_core = match &data.value_text {
        Some(t) => t.clone(),
        None => format_number(disp_val, data.precision),
    };
    let hero_str = format!("{}{}{}", data.prefix, formatted_core, data.suffix);

    if rect.h < 80.0 {
        let label_s = fit_text(atlas, &data.label, 12.0, true, iw * 0.45);
        atlas.layout_dl(
            &label_s,
            ix,
            atlas.centered_pen_y(iy, ih, 12.0),
            12.0,
            theme.text_secondary,
            true,
            None,
            dl,
        );
        let (vw, _) = atlas.measure(&hero_str, 17.0, true);
        atlas.layout_dl(
            &hero_str,
            ix + iw - vw,
            atlas.centered_pen_y(iy, ih, 17.0),
            17.0,
            theme.text_primary,
            true,
            None,
            dl,
        );
        dl.pop_clip();
        return;
    }

    let header_row_h = 20.0f32;
    let mut right_used_w = 0.0f32;
    if let Some(delta) = &data.delta {
        if !delta.is_empty() {
            let (dw, _) = atlas.measure(delta, 11.5, true);
            let pill_w = (dw + 14.0).min(iw * 0.55);
            let pill_x = ix + iw - pill_w;
            let pill_y = iy;
            dl.push_quad(QuadInstance {
                rect: [pill_x, pill_y, pill_w, header_row_h],
                radii: [theme.radius_pill; 4],
                color_top: scale_alpha(tone_col, 0.14),
                color_bottom: scale_alpha(tone_col, 0.14),
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            let fitted_d = fit_text(atlas, delta, 11.5, true, pill_w - 8.0);
            atlas.layout_centered(
                &fitted_d,
                pill_x,
                pill_y,
                pill_w,
                header_row_h,
                11.5,
                tone_col,
                true,
                dl,
            );
            right_used_w += pill_w + 6.0;
        }
    }

    if let Some(badge) = &data.badge {
        if !badge.is_empty() && rect.w - pad_x * 2.0 - right_used_w > 95.0 {
            let (bw, _) = atlas.measure(badge, 10.5, true);
            let pill_w = bw + 12.0;
            let pill_x = ix + iw - right_used_w - pill_w;
            let pill_y = iy;
            dl.push_quad(QuadInstance {
                rect: [pill_x, pill_y, pill_w, header_row_h],
                radii: [5.0; 4],
                color_top: theme.surface_2,
                color_bottom: theme.surface_2,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            atlas.layout_centered(
                badge,
                pill_x,
                pill_y,
                pill_w,
                header_row_h,
                10.5,
                theme.text_muted,
                true,
                dl,
            );
            right_used_w += pill_w + 6.0;
        }
    }

    let label_max_w = (iw - right_used_w).max(24.0);
    let label_str = fit_text(atlas, &data.label, 12.5, true, label_max_w);
    atlas.layout_dl(
        &label_str,
        ix,
        atlas.centered_pen_y(iy, header_row_h, 12.5),
        12.5,
        theme.text_secondary,
        true,
        None,
        dl,
    );

    let mut hero_px = if rect.w >= 460.0 && rect.h >= 230.0 {
        38.0
    } else if rect.w >= 320.0 && rect.h >= 180.0 {
        30.0
    } else if rect.w < 200.0 || rect.h < 104.0 {
        20.0
    } else {
        24.0
    };
    while hero_px > 14.0 && atlas.measure(&hero_str, hero_px, true).0 > iw {
        hero_px -= 1.5;
    }
    let fitted_hero = fit_text(atlas, &hero_str, hero_px, true, iw);
    let hero_y = iy + if rect.h >= 210.0 { 30.0 } else { 25.0 };
    let (_, hero_h) = atlas.measure(&fitted_hero, hero_px, true);
    atlas.layout_dl(
        &fitted_hero,
        ix,
        hero_y,
        hero_px,
        theme.text_primary,
        true,
        None,
        dl,
    );

    let mut cursor_y = hero_y + hero_h + if rect.h >= 210.0 { 12.0 } else { 6.0 };
    let caption_h = if let Some(cap) = &data.caption {
        if !cap.is_empty() && rect.h >= 114.0 {
            18.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let bottom_limit = iy + ih - caption_h;
    let visual_avail_h = (bottom_limit - cursor_y).max(0.0);
    let max_spark_h = if rect.h >= 210.0 {
        visual_avail_h.min(220.0)
    } else {
        visual_avail_h.min(34.0)
    };

    match data.variant {
        StatVariant::Sparkline => {
            if !data.sparkline.is_empty() && visual_avail_h >= 18.0 {
                let spark_h = max_spark_h;
                emit_sparkline(&data.sparkline, ix, cursor_y, iw, spark_h, tone_col, theme, dl);
                cursor_y += spark_h + 6.0;
            }
        }
        StatVariant::RangeBar => {
            if let Some([lo, hi]) = data.range_bounds {
                if visual_avail_h >= 18.0 {
                    let bar_h = visual_avail_h.min(28.0);
                    emit_range_bar(
                        lo,
                        hi,
                        disp_val,
                        &data.prefix,
                        &data.suffix,
                        data.precision,
                        ix,
                        cursor_y,
                        iw,
                        bar_h,
                        tone_col,
                        theme,
                        atlas,
                        dl,
                    );
                    cursor_y += bar_h + 4.0;
                }
            }
        }
        StatVariant::Compact => {}
    }

    if caption_h > 0.0 {
        if let Some(cap) = &data.caption {
            let cap_y = atlas
                .centered_pen_y(iy + ih - 14.0, 14.0, 11.5)
                .max(cursor_y + 2.0);
            let fitted_cap = fit_text(atlas, cap, 11.5, true, iw);
            atlas.layout_dl(
                &fitted_cap,
                ix,
                cap_y,
                11.5,
                theme.text_muted,
                true,
                None,
                dl,
            );
        }
    }

    dl.pop_clip();
}

#[allow(clippy::too_many_arguments)]
fn emit_sparkline(
    vals: &[f32],
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: u32,
    theme: &Theme,
    dl: &mut DrawList,
) {
    let base_y = y + h - 2.0;
    dl.push_stroke(StrokeInstance::solid(
        [x, base_y],
        [x + w, base_y],
        1.0,
        0.75,
        scale_alpha(theme.border, 0.5),
        0,
    ));

    if vals.len() == 1 {
        dl.push_quad(QuadInstance {
            rect: [x + w * 0.5 - 3.0, y + h * 0.5 - 3.0, 6.0, 6.0],
            radii: [3.0; 4],
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

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in vals {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        return;
    }
    let span = (max_v - min_v).max(1e-5);
    let inner_y = y + 3.0;
    let inner_h = (h - 7.0).max(4.0);

    let max_pts = (w * 0.75).max(16.0) as usize;
    let step = if vals.len() > max_pts {
        vals.len() as f32 / max_pts as f32
    } else {
        1.0
    };

    let count = ((vals.len() as f32) / step).floor().max(2.0) as usize;
    let sample_at = |idx: usize| -> f32 {
        let src = ((idx as f32 * step).floor() as usize).min(vals.len() - 1);
        let v = vals[src];
        let norm = if v.is_finite() {
            ((v - min_v) / span).clamp(0.0, 1.0)
        } else {
            0.5
        };
        inner_y + inner_h * (1.0 - norm)
    };

    let mut pts = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        pts.push([x + t * w, sample_at(i)]);
    }

    // Pass 1: Smooth sloped trapezoid gradient fill under sparkline (zero stair-steps)
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
            color_top: scale_alpha(color, 0.24),
            color_bottom: scale_alpha(color, 0.0),
            clip_id: 0,
            _pad: [0; 3],
        });
    }

    // Pass 2: Crisp antialiased line curve on top of area wash
    for win in pts.windows(2) {
        dl.push_stroke(StrokeInstance::solid(win[0], win[1], 1.75, 0.9, color, 0));
    }

    if let Some(&[lx, ly]) = pts.last() {
        dl.push_quad(QuadInstance {
            rect: [lx - 3.0, ly - 3.0, 6.0, 6.0],
            radii: [3.0; 4],
            color_top: color,
            color_bottom: color,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_range_bar(
    lo: f64,
    hi: f64,
    val: f64,
    prefix: &str,
    suffix: &str,
    precision: Option<usize>,
    x: f32,
    y: f32,
    w: f32,
    _h: f32,
    color: u32,
    theme: &Theme,
    atlas: &GlyphAtlas,
    dl: &mut DrawList,
) {
    let span = (hi - lo).abs().max(1e-9);
    let frac = ((val - lo) / span).clamp(0.0, 1.0) as f32;
    let track_y = y + 4.0;
    let track_h = 5.0;

    dl.push_quad(QuadInstance {
        rect: [x, track_y, w, track_h],
        radii: [2.5; 4],
        color_top: theme.surface_2,
        color_bottom: theme.surface_2,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let fill_w = (w * frac).max(track_h);
    dl.push_quad(QuadInstance {
        rect: [x, track_y, fill_w, track_h],
        radii: [2.5; 4],
        color_top: scale_alpha(color, 0.85),
        color_bottom: scale_alpha(color, 0.85),
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let knob_x = (x + w * frac - 4.5).clamp(x, x + w - 9.0);
    dl.push_quad(QuadInstance {
        rect: [knob_x, track_y - 2.0, 9.0, 9.0],
        radii: [4.5; 4],
        color_top: color,
        color_bottom: color,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let lo_txt = format!("{}{}{}", prefix, format_number(lo, precision), suffix);
    let hi_txt = format!("{}{}{}", prefix, format_number(hi, precision), suffix);
    let (hw, _) = atlas.measure(&hi_txt, 10.5, true);
    atlas.layout_dl(&lo_txt, x, track_y + 11.0, 10.5, theme.text_muted, true, None, dl);
    atlas.layout_dl(&hi_txt, x + w - hw, track_y + 11.0, 10.5, theme.text_muted, true, None, dl);
}
