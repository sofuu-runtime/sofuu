//! Independent interactive control components (`ThemeToggle`, `Segmented`, `Button`, `Tabs`).
//!
//! Each control is self-contained, clips strictly to its own resolved `ComputedRect`,
//! and maintains isolated spring & hit-test namespaces so controls never conflict with
//! neighboring components or each other.

use crate::components::fit_text;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance, StrokeInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::ir::TabItem;
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, ThemeMode, Tone};

/// Theme Toggle icon button: displays Sun icon in Light mode and Crescent Moon icon in Dark mode.
pub fn emit_theme_toggle(
    id: &str,
    mode: ThemeMode,
    rect: ComputedRect,
    theme: &Theme,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 12.0 || rect.h < 12.0 {
        return;
    }
    let hit_key = format!("toggle::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let pressed = interaction.pressed_key.as_deref() == Some(hit_key.as_str());
    let hover_target = if pressed {
        1.0
    } else if hovered {
        0.7
    } else {
        0.0
    };
    let hover_t = springs.animate(&hit_key, "hover", hover_target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::ThemeToggle { id: id.to_string() });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let btn_bg = lerp_rgba8(theme.surface_1, theme.surface_2, hover_t);
    let radius = rect.w.min(rect.h) * 0.5;
    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [radius; 4],
        color_top: btn_bg,
        color_bottom: btn_bg,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let cx = rect.x + rect.w * 0.5;
    let cy = rect.y + rect.h * 0.5;
    let unit = (rect.w.min(rect.h) / 36.0).clamp(0.6, 1.8);

    match mode {
        ThemeMode::Light => {
            let sun_col = theme.warn;
            let sun_r = 4.5 * unit;
            dl.push_quad(QuadInstance {
                rect: [cx - sun_r, cy - sun_r, sun_r * 2.0, sun_r * 2.0],
                radii: [sun_r; 4],
                color_top: sun_col,
                color_bottom: sun_col,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            for step in 0..8 {
                let ang = (step as f32 * 45.0).to_radians();
                let r0 = 6.7 * unit;
                let r1 = 9.0 * unit;
                dl.push_stroke(StrokeInstance::solid(
                    [cx + r0 * ang.cos(), cy + r0 * ang.sin()],
                    [cx + r1 * ang.cos(), cy + r1 * ang.sin()],
                    1.6 * unit,
                    0.75,
                    sun_col,
                    0,
                ));
            }
        }
        ThemeMode::Dark => {
            let moon_col = theme.text_primary;
            let moon_r = 7.4 * unit;
            dl.push_quad(QuadInstance {
                rect: [cx - moon_r, cy - moon_r, moon_r * 2.0, moon_r * 2.0],
                radii: [moon_r; 4],
                color_top: moon_col,
                color_bottom: moon_col,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            let bite_r = 6.0 * unit;
            let bite_cx = cx + 3.0 * unit;
            let bite_cy = cy - 2.3 * unit;
            dl.push_quad(QuadInstance {
                rect: [bite_cx - bite_r, bite_cy - bite_r, bite_r * 2.0, bite_r * 2.0],
                radii: [bite_r; 4],
                color_top: btn_bg,
                color_bottom: btn_bg,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
        }
    }

    dl.pop_clip();
}

/// Borderless, shadowless animated pill switcher control.
#[allow(clippy::too_many_arguments)]
pub fn emit_segmented(
    id: &str,
    active: usize,
    options: &[String],
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 24.0 || rect.h < 18.0 {
        return;
    }
    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [theme.radius_control; 4],
        color_top: theme.surface_1,
        color_bottom: theme.surface_1,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    if options.is_empty() {
        dl.pop_clip();
        return;
    }

    let pad = 3.0;
    let inner_w = (rect.w - pad * 2.0).max(16.0);
    let item_h = (rect.h - pad * 2.0).max(12.0);
    let active_idx = active.min(options.len() - 1);

    let mut widths = Vec::with_capacity(options.len());
    let mut raw_sum = 0.0f32;
    for o in options {
        let w = atlas.measure(o, 12.0, false).0 + 18.0;
        widths.push(w);
        raw_sum += w;
    }
    let scale = if raw_sum > 0.0 { inner_w / raw_sum } else { 1.0 };
    let mut xs = Vec::with_capacity(options.len());
    let mut cur_x = rect.x + pad;
    for w in &mut widths {
        *w *= scale;
        xs.push(cur_x);
        cur_x += *w;
    }

    let seg_ns = format!("seg::{id}");
    let anim_x = springs.animate(&seg_ns, "x", xs[active_idx], SpringConfig::SNAPPY);
    let anim_w = springs.animate(&seg_ns, "w", widths[active_idx], SpringConfig::SNAPPY);

    dl.push_quad(QuadInstance {
        rect: [anim_x, rect.y + pad, anim_w, item_h],
        radii: [theme.radius_control - 2.0; 4],
        color_top: theme.surface_3,
        color_bottom: theme.surface_3,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    for (idx, opt) in options.iter().enumerate() {
        let bx = xs[idx];
        let bw = widths[idx];
        let btn_rect = ComputedRect {
            x: bx,
            y: rect.y + pad,
            w: bw,
            h: item_h,
            content_h: item_h,
            folded: false,
        };
        interaction.push_hit(
            btn_rect,
            HitKind::SegmentedItem {
                seg_id: id.to_string(),
                index: idx,
            },
        );
        let col = if idx == active_idx {
            theme.text_primary
        } else {
            theme.text_secondary
        };
        atlas.layout_centered(opt, bx, rect.y + pad, bw, item_h, 12.0, col, false, dl);
    }

    dl.pop_clip();
}

/// Borderless, shadowless pill action button.
#[allow(clippy::too_many_arguments)]
pub fn emit_button(
    id: &str,
    label: &str,
    tone: Tone,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 16.0 || rect.h < 16.0 {
        return;
    }
    let hit_key = format!("btn::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let pressed = interaction.pressed_key.as_deref() == Some(hit_key.as_str());
    let hover_target = if pressed {
        1.0
    } else if hovered {
        0.7
    } else {
        0.0
    };
    let hover_t = springs.animate(&hit_key, "hover", hover_target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::Button { id: id.to_string() });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let (base_bg, hover_bg, fg) = if tone == Tone::Neutral {
        (theme.surface_1, theme.surface_2, theme.text_primary)
    } else {
        let tc = theme.tone_color(tone);
        (scale_alpha(tc, 0.16), scale_alpha(tc, 0.28), tc)
    };
    let bg = lerp_rgba8(base_bg, hover_bg, hover_t);
    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [theme.radius_control; 4],
        color_top: bg,
        color_bottom: bg,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    atlas.layout_centered(label, rect.x, rect.y, rect.w, rect.h, 12.0, fg, false, dl);

    dl.pop_clip();
}

/// Tab bar header emission; returns index of active tab item to render below.
#[allow(clippy::too_many_arguments)]
pub fn emit_tabs_bar(
    id: &str,
    active: usize,
    items: &[TabItem],
    ix: f32,
    iy: f32,
    iw: f32,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) -> Option<usize> {
    let bar_h = 38.0;
    dl.push_clip(ClipRect::new(ix, iy, iw, bar_h));

    dl.push_quad(QuadInstance {
        rect: [ix, iy, iw, bar_h],
        radii: [theme.radius_control; 4],
        color_top: theme.surface_0,
        color_bottom: theme.surface_0,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    if items.is_empty() {
        dl.pop_clip();
        return None;
    }

    let count = items.len();
    let pad = 4.0;
    let pill_w = ((iw - pad * 2.0) / count as f32).max(20.0);
    let active_clamped = active.min(count - 1);
    let target_pill_x = ix + pad + active_clamped as f32 * pill_w;
    let tab_ns = format!("tabs::{id}");
    let anim_pill_x = springs.animate(&tab_ns, "pill_x", target_pill_x, SpringConfig::UI_DEFAULT);

    dl.push_quad(QuadInstance {
        rect: [anim_pill_x, iy + pad, pill_w, bar_h - pad * 2.0],
        radii: [theme.radius_control - 2.0; 4],
        color_top: theme.surface_2,
        color_bottom: theme.surface_2,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    for (idx, item) in items.iter().enumerate() {
        let btn_x = ix + pad + idx as f32 * pill_w;
        let btn_rect = ComputedRect {
            x: btn_x,
            y: iy + pad,
            w: pill_w,
            h: bar_h - pad * 2.0,
            content_h: bar_h,
            folded: false,
        };
        interaction.push_hit(
            btn_rect,
            HitKind::TabButton {
                tabs_id: id.to_string(),
                index: idx,
            },
        );
        let color = if idx == active_clamped {
            theme.text_primary
        } else {
            theme.text_secondary
        };
        atlas.layout_centered(
            &item.label,
            btn_x + 4.0,
            iy + pad,
            (pill_w - 8.0).max(8.0),
            bar_h - pad * 2.0,
            13.0,
            color,
            false,
            dl,
        );
    }

    dl.pop_clip();
    Some(active_clamped)
}

/// Borderless, shadowless animated pill toggle switch with optional label.
#[allow(clippy::too_many_arguments)]
pub fn emit_switch(
    id: &str,
    checked: bool,
    label: Option<&str>,
    tone: Tone,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 28.0 || rect.h < 18.0 {
        return;
    }
    let hit_key = format!("switch::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let on_t = springs.animate(
        &hit_key,
        "checked",
        if checked { 1.0 } else { 0.0 },
        SpringConfig::SNAPPY,
    );
    let hover_t = springs.animate(
        &hit_key,
        "hover",
        if hovered { 1.0 } else { 0.0 },
        SpringConfig::SNAPPY,
    );
    interaction.push_hit(rect, HitKind::Switch { id: id.to_string() });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let track_h = rect.h.clamp(18.0, 22.0);
    let track_w = (track_h * 1.82).min(rect.w);
    let tx = rect.x;
    let ty = rect.y + (rect.h - track_h) * 0.5;

    let active_col = if tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(tone)
    };
    let off_col = lerp_rgba8(theme.surface_2, theme.surface_3, hover_t * 0.7);
    let track_col = lerp_rgba8(off_col, active_col, on_t);

    dl.push_quad(QuadInstance {
        rect: [tx, ty, track_w, track_h],
        radii: [track_h * 0.5; 4],
        color_top: track_col,
        color_bottom: track_col,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let pad = 3.0;
    let knob_d = (track_h - pad * 2.0).max(8.0);
    let kx = tx + pad + (track_w - knob_d - pad * 2.0) * on_t;
    let ky = ty + pad;
    let knob_col = lerp_rgba8(theme.surface_0, 0xFF_FF_FF_FF, on_t * 0.85);
    dl.push_quad(QuadInstance {
        rect: [kx, ky, knob_d, knob_d],
        radii: [knob_d * 0.5; 4],
        color_top: knob_col,
        color_bottom: knob_col,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    if let Some(lbl) = label.filter(|l| !l.is_empty()) {
        let lx = tx + track_w + 8.0;
        let lw = (rect.x + rect.w - lx).max(0.0);
        if lw > 12.0 {
            let fitted = fit_text(atlas, lbl, 12.0, false, lw);
            let py = atlas.centered_pen_y(rect.y, rect.h, 12.0);
            atlas.layout_dl(&fitted, lx, py, 12.0, theme.text_primary, false, None, dl);
        }
    }

    dl.pop_clip();
}

/// Interactive single-line monospace text input field with focus ring & live caret.
#[allow(clippy::too_many_arguments)]
pub fn emit_input(
    id: &str,
    value: &str,
    placeholder: &str,
    tone: Tone,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 36.0 || rect.h < 22.0 {
        return;
    }
    let hit_key = format!("input::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let focused = interaction.focused_key.as_deref() == Some(hit_key.as_str());
    let state_target = if focused {
        1.0
    } else if hovered {
        0.55
    } else {
        0.0
    };
    let state_t = springs.animate(&hit_key, "focus", state_target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::Input { id: id.to_string() });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));

    let focus_col = if tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(tone)
    };
    let bg = lerp_rgba8(theme.surface_1, theme.surface_2, state_t * 0.7);
    dl.push_quad(QuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii: [theme.radius_control; 4],
        color_top: bg,
        color_bottom: bg,
        border_color: if focused { focus_col } else { 0 },
        shadow_color: 0,
        params: [if focused { 1.2 } else { 0.0 }, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let pad_x = 10.0;
    let inner_w = (rect.w - pad_x * 2.0).max(8.0);
    let px = 12.5;
    let py = atlas.centered_pen_y(rect.y, rect.h, px);

    if value.is_empty() {
        let fitted_ph = fit_text(atlas, placeholder, px, false, inner_w);
        atlas.layout_dl(
            &fitted_ph,
            rect.x + pad_x,
            py,
            px,
            theme.text_muted,
            false,
            None,
            dl,
        );
        if focused {
            let caret_h = (rect.h - 14.0).clamp(12.0, 18.0);
            let caret_y = rect.y + (rect.h - caret_h) * 0.5;
            dl.push_quad(QuadInstance {
                rect: [rect.x + pad_x, caret_y, 1.6, caret_h],
                radii: [0.8; 4],
                color_top: focus_col,
                color_bottom: focus_col,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
        }
    } else {
        // Keep trailing characters visible when value exceeds field width so active typing stays in view
        let caret_reserve = if focused { 4.0 } else { 0.0 };
        let visible = fit_tail_text(atlas, value, px, false, (inner_w - caret_reserve).max(4.0));
        atlas.layout_dl(
            &visible,
            rect.x + pad_x,
            py,
            px,
            theme.text_primary,
            false,
            None,
            dl,
        );
        if focused {
            let (vw, _) = atlas.measure(&visible, px, false);
            let caret_h = (rect.h - 14.0).clamp(12.0, 18.0);
            let caret_y = rect.y + (rect.h - caret_h) * 0.5;
            dl.push_quad(QuadInstance {
                rect: [rect.x + pad_x + vw + 1.0, caret_y, 1.6, caret_h],
                radii: [0.8; 4],
                color_top: focus_col,
                color_bottom: focus_col,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
        }
    }

    dl.pop_clip();
}

fn fit_tail_text(atlas: &GlyphAtlas, text: &str, px: f32, mono: bool, max_w: f32) -> String {
    let (w, _) = atlas.measure(text, px, mono);
    if w <= max_w {
        return text.to_string();
    }
    let mut chars: Vec<char> = text.chars().collect();
    let mut acc = 0.0f32;
    let mut start = chars.len();
    for i in (0..chars.len()).rev() {
        let cw = atlas.char_advance_em(chars[i], mono) * px;
        if acc + cw > max_w {
            break;
        }
        acc += cw;
        start = i;
    }
    chars.drain(start..).collect()
}
