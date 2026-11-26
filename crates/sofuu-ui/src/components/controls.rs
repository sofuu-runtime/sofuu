//! Independent interactive control components (`ThemeToggle`, `Segmented`, `Button`, `Tabs`).
//!
//! Each control is self-contained, clips strictly to its own resolved `ComputedRect`,
//! and maintains isolated spring & hit-test namespaces so controls never conflict with
//! neighboring components or each other.

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
        let w = atlas.measure(o, 12.0, true).0 + 18.0;
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
        atlas.layout_centered(opt, bx, rect.y + pad, bw, item_h, 12.0, col, true, dl);
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
    atlas.layout_centered(label, rect.x, rect.y, rect.w, rect.h, 12.0, fg, true, dl);

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
            true,
            dl,
        );
    }

    dl.pop_clip();
    Some(active_clamped)
}
