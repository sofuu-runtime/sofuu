//! Independent interactive control components (`ThemeToggle`, `Segmented`, `Button`, `Tabs`).
//!
//! Each control is self-contained, clips strictly to its own resolved `ComputedRect`,
//! and maintains isolated spring & hit-test namespaces so controls never conflict with
//! neighboring components or each other.

use crate::components::fit_text;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance, StrokeInstance};
use crate::hit_test::{HitKind, InteractionState, TextEditState};
use crate::ir::TabItem;
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::{GlyphAtlas, LineBreak, TextStyle, WEIGHT_REGULAR};
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
#[allow(clippy::too_many_arguments)]
pub fn emit_input(
    id: &str,
    value: &str,
    placeholder: &str,
    tone: Tone,
    prefix_icon: Option<&str>,
    clearable: bool,
    error: Option<&str>,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    icons: &crate::icons::IconAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    // The caption lives inside the node box, under the chrome.
    let caption_h = if error.is_some_and(|e| !e.is_empty()) {
        16.0
    } else {
        0.0
    };
    let field = ComputedRect {
        content_h: (rect.h - caption_h).max(20.0),
        folded: rect.folded,
        ..rect
    };
    let rect = field;
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

    let mut pad_x = 10.0;
    let px = 12.5;
    let py = atlas.centered_pen_y(rect.y, rect.h, px);

    // Leading icon occupies a 22pt gutter; without one the text starts flush.
    if let Some(name) = prefix_icon.filter(|n| !n.is_empty()) {
        let gutter = 24.0;
        crate::components::media::emit_icon(
            icons,
            name,
            ComputedRect {
                x: rect.x + 8.0,
                y: rect.y + (rect.h - gutter) * 0.5,
                w: gutter,
                h: gutter,
                ..rect
            },
            16.0,
            theme.text_muted,
            dl,
        );
        pad_x += gutter - 4.0;
    }
    let show_clear = clearable && !value.is_empty();
    let clear_w = if show_clear { 24.0 } else { 0.0 };
    let inner_w = (rect.w - pad_x - 10.0 - clear_w).max(8.0);
    if show_clear {
        let box_rect = ComputedRect {
            x: rect.x + rect.w - 10.0 - clear_w,
            y: rect.y + (rect.h - clear_w) * 0.5,
            w: clear_w,
            h: clear_w,
            ..rect
        };
        // Pushed after the field hit region so it wins inside its own box.
        interaction.push_hit(box_rect, HitKind::InputClear { id: id.to_string() });
        crate::components::media::emit_icon(
            icons,
            "x",
            box_rect,
            14.0,
            theme.text_muted,
            dl,
        );
    }

    let caret_reserve = if focused { 4.0 } else { 0.0 };
    let (shown_start, visible) = tail_window(
        atlas,
        value,
        px,
        false,
        (inner_w - caret_reserve).max(4.0),
    );
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
    } else {
        // Keep trailing characters visible when value exceeds field width so active typing stays in view
        atlas.layout_dl(
            visible,
            rect.x + pad_x,
            py,
            px,
            theme.text_primary,
            false,
            None,
            dl,
        );
    }
    if focused {
        emit_edit_affordances(
            value,
            visible,
            shown_start,
            rect.x + pad_x,
            py,
            inner_w,
            px,
            interaction.edit.as_ref(),
            &hit_key,
            value.len(),
            focus_col,
            interaction.caret_visible,
            atlas,
            dl,
        );
    }

    dl.pop_clip();

    if caption_h > 0.0 {
        let msg = error.unwrap_or_default();
        let fitted = fit_text(atlas, msg, 11.5, false, rect.w);
        atlas.layout_dl(
            &fitted,
            rect.x + 2.0,
            rect.y + rect.h + 2.0,
            11.5,
            theme.tone_color(Tone::Danger),
            false,
            None,
            dl,
        );
    }
}

/// Visible tail of a one-line value: `(start_byte, slice)`.
fn tail_window<'a>(
    atlas: &GlyphAtlas,
    text: &'a str,
    px: f32,
    mono: bool,
    max_w: f32,
) -> (usize, &'a str) {
    let (w, _) = atlas.measure(text, px, mono);
    if w <= max_w {
        return (0, text);
    }
    let mut acc = 0.0f32;
    let mut start = text.len();
    for (i, ch) in text.char_indices().rev() {
        let cw = atlas.char_advance_em(ch, mono) * px;
        if acc + cw > max_w {
            break;
        }
        acc += cw;
        start = i;
    }
    (start, &text[start..])
}

/// Draw the selection highlight and caret for a focused editable.
///
/// `shown` is what the renderer actually paints (a tail window for one-line
/// inputs, bullets for secure text areas); `shown_start` maps its first byte
/// back into `raw`. `fallback_caret` keeps a caret visible when focus came
/// from somewhere other than a click (programmatic focus, Tab).
#[allow(clippy::too_many_arguments)]
fn emit_edit_affordances(
    raw: &str,
    shown: &str,
    shown_start: usize,
    origin_x: f32,
    origin_y: f32,
    max_w: f32,
    px: f32,
    edit: Option<&TextEditState>,
    hit_key: &str,
    fallback_caret: usize,
    focus_col: u32,
    caret_visible: bool,
    atlas: &GlyphAtlas,
    dl: &mut DrawList,
) {
    let style = TextStyle::sans(px, WEIGHT_REGULAR);
    let line_h = px * atlas.line_height_em;
    let (caret_raw, selection) = match edit.filter(|e| e.id == hit_key) {
        Some(e) => (e.caret.min(raw.len()), Some(e.selection())),
        None => (fallback_caret.min(raw.len()), None),
    };
    let to_local = |idx: usize| -> usize {
        if shown_start > 0 {
            return floor_byte(shown, idx.saturating_sub(shown_start).min(shown.len()));
        }
        if raw.len() == shown.len() {
            return floor_byte(shown, idx.min(shown.len()));
        }
        // Masked rendering keeps the character count but changes byte widths.
        let chars = raw[..floor_byte(raw, idx)].chars().count();
        shown
            .char_indices()
            .nth(chars)
            .map(|(i, _)| i)
            .unwrap_or(shown.len())
    };

    if let Some((a, b)) = selection {
        let a = to_local(a);
        let b = to_local(b);
        if a < b {
            let laid = atlas.layout_text(
                shown,
                style,
                Some(max_w),
                usize::MAX,
                LineBreak::CharWrap,
            );
            let bg = scale_alpha(focus_col, 0.32);
            for (li, line) in laid.lines.iter().enumerate() {
                if b <= line.start || a > line.end {
                    continue;
                }
                let s = a.max(line.start);
                let e = b.min(line.end);
                let x0 = if s <= line.start {
                    0.0
                } else {
                    atlas
                        .x_for_char_index(shown, s, style, Some(max_w), LineBreak::CharWrap)
                        .0
                };
                let x1 = if e >= line.end {
                    // Extend a few px over the newline so a full-line selection
                    // reads as continuous.
                    line.width + if b > line.end { 4.0 } else { 0.0 }
                } else {
                    atlas
                        .x_for_char_index(shown, e, style, Some(max_w), LineBreak::CharWrap)
                        .0
                };
                if x1 > x0 {
                    dl.push_quad(QuadInstance {
                        rect: [origin_x + x0, origin_y + li as f32 * line_h + 1.0, x1 - x0, line_h - 2.0],
                        radii: [2.0; 4],
                        color_top: bg,
                        color_bottom: bg,
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                }
            }
        }
    }

    if !caret_visible {
        return;
    }
    let (cx, line_idx) = atlas.x_for_char_index(
        shown,
        to_local(caret_raw),
        style,
        Some(max_w),
        LineBreak::CharWrap,
    );
    let cy = origin_y + line_idx as f32 * line_h;
    dl.push_quad(QuadInstance {
        rect: [origin_x + cx, cy + 1.0, 1.6, (line_h - 2.0).max(10.0)],
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

#[inline]
fn floor_byte(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Horizontal slider with a filled track, thumb, optional label and value readout.
#[allow(clippy::too_many_arguments)]
pub fn emit_slider(
    id: &str,
    data: &crate::ir::SliderData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 60.0 || rect.h < 18.0 {
        return;
    }
    let accent = if data.tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(data.tone)
    };
    let hit_key = format!("slider::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let active = interaction.slider_drag.as_deref() == Some(id);
    let target = if active {
        1.0
    } else if hovered {
        0.6
    } else {
        0.0
    };
    let t = springs.animate(&hit_key, "hover", target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::Slider { id: id.to_string() });

    let label_h = if data.label.is_some() || data.show_value {
        18.0
    } else {
        0.0
    };
    /* Labels sit on the glyph baseline box that can extend a hair above
     * `rect.y`; draw them before the track clip so the ascender is never cut. */
    if let Some(label) = data.label.as_deref().filter(|l| !l.is_empty()) {
        atlas.layout_dl(
            label,
            rect.x,
            rect.y,
            12.0,
            theme.text_secondary,
            false,
            None,
            dl,
        );
    }
    let value_text = format_slider_value(data);
    if data.show_value {
        let (vw, _) = atlas.measure(&value_text, 12.0, false);
        atlas.layout_dl(
            &value_text,
            rect.x + rect.w - vw,
            rect.y,
            12.0,
            theme.text_primary,
            false,
            None,
            dl,
        );
    }
    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
    let track_y = rect.y + label_h + (rect.h - label_h) * 0.5 - 2.0;
    let track_h = 4.0 + t * 1.5;
    let knob_r = (7.0 + t * 1.5).min(rect.h * 0.5);
    let x0 = rect.x + knob_r;
    let x1 = rect.x + rect.w - knob_r;
    let span = (data.max - data.min).abs().max(1e-6);
    let frac = ((data.value - data.min) / span).clamp(0.0, 1.0);
    let knob_x = x0 + (x1 - x0) * frac;

    dl.push_quad(QuadInstance {
        rect: [x0, track_y, (x1 - x0).max(1.0), track_h],
        radii: [track_h * 0.5; 4],
        color_top: theme.surface_3,
        color_bottom: theme.surface_3,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    dl.push_quad(QuadInstance {
        rect: [x0, track_y, (knob_x - x0).max(0.0), track_h],
        radii: [track_h * 0.5; 4],
        color_top: accent,
        color_bottom: accent,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    dl.push_quad(QuadInstance {
        rect: [
            knob_x - knob_r,
            track_y + track_h * 0.5 - knob_r,
            knob_r * 2.0,
            knob_r * 2.0,
        ],
        radii: [knob_r; 4],
        color_top: theme.surface_1,
        color_bottom: theme.surface_1,
        border_color: accent,
        shadow_color: 0,
        params: [2.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    dl.pop_clip();
}

fn format_slider_value(data: &crate::ir::SliderData) -> String {
    if data.step >= 1.0 {
        format!("{}", data.value.round() as i64)
    } else {
        format!("{:.2}", data.value)
    }
}

/// Apply one pointer x position to a slider's value (step-snapped and clamped).
pub fn slider_value_at(data: &crate::ir::SliderData, rect: ComputedRect, x: f32) -> f32 {
    let knob_r = 7.0;
    let x0 = rect.x + knob_r;
    let x1 = rect.x + rect.w - knob_r;
    let frac = ((x - x0) / (x1 - x0).max(1.0)).clamp(0.0, 1.0);
    let span = data.max - data.min;
    let mut v = data.min + span * frac;
    if data.step > 0.0 {
        v = (v / data.step).round() * data.step;
    }
    v.clamp(data.min.min(data.max), data.max.max(data.min))
}

/// Dual-handle range slider: one track, two thumbs, inclusive span highlighted.
#[allow(clippy::too_many_arguments)]
pub fn emit_range_slider(
    id: &str,
    data: &crate::ir::RangeSliderData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 60.0 || rect.h < 18.0 {
        return;
    }
    let accent = if data.tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(data.tone)
    };
    let hit_key = format!("range::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let active = interaction
        .range_drag
        .as_ref()
        .is_some_and(|(rid, _)| rid == id);
    let target = if active {
        1.0
    } else if hovered {
        0.6
    } else {
        0.0
    };
    let t = springs.animate(&hit_key, "hover", target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::RangeSlider { id: id.to_string() });

    let label_h = if data.label.is_some() || data.show_value {
        18.0
    } else {
        0.0
    };
    if let Some(label) = data.label.as_deref().filter(|l| !l.is_empty()) {
        atlas.layout_dl(
            label,
            rect.x,
            rect.y,
            12.0,
            theme.text_secondary,
            false,
            None,
            dl,
        );
    }
    let (low, high) = data.clamped();
    let value_text = format_range_value(data, low, high);
    if data.show_value {
        let (vw, _) = atlas.measure(&value_text, 12.0, false);
        atlas.layout_dl(
            &value_text,
            rect.x + rect.w - vw,
            rect.y,
            12.0,
            theme.text_primary,
            false,
            None,
            dl,
        );
    }
    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
    let track_y = rect.y + label_h + (rect.h - label_h) * 0.5 - 2.0;
    let track_h = 4.0 + t * 1.5;
    let knob_r = (7.0 + t * 1.5).min(rect.h * 0.5);
    let x0 = rect.x + knob_r;
    let x1 = rect.x + rect.w - knob_r;
    let (lo_x, hi_x) = range_thumb_x(data, rect, low, high);

    dl.push_quad(QuadInstance {
        rect: [x0, track_y, (x1 - x0).max(1.0), track_h],
        radii: [track_h * 0.5; 4],
        color_top: theme.surface_3,
        color_bottom: theme.surface_3,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    dl.push_quad(QuadInstance {
        rect: [lo_x, track_y, (hi_x - lo_x).max(0.0), track_h],
        radii: [track_h * 0.5; 4],
        color_top: accent,
        color_bottom: accent,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });
    for thumb_x in [lo_x, hi_x] {
        dl.push_quad(QuadInstance {
            rect: [
                thumb_x - knob_r,
                track_y + track_h * 0.5 - knob_r,
                knob_r * 2.0,
                knob_r * 2.0,
            ],
            radii: [knob_r; 4],
            color_top: theme.surface_1,
            color_bottom: theme.surface_1,
            border_color: accent,
            shadow_color: 0,
            params: [2.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }
    dl.pop_clip();
}

fn format_range_value(data: &crate::ir::RangeSliderData, low: f32, high: f32) -> String {
    if data.step >= 1.0 {
        format!("{} – {}", low.round() as i64, high.round() as i64)
    } else {
        format!("{low:.2} – {high:.2}")
    }
}

/// Screen x of the two thumbs, in `(low, high)` order.
fn range_thumb_x(
    data: &crate::ir::RangeSliderData,
    rect: ComputedRect,
    low: f32,
    high: f32,
) -> (f32, f32) {
    let knob_r = 7.0;
    let x0 = rect.x + knob_r;
    let x1 = rect.x + rect.w - knob_r;
    let span = (data.max - data.min).abs().max(1e-6);
    let lo = ((low - data.min) / span).clamp(0.0, 1.0);
    let hi = ((high - data.min) / span).clamp(0.0, 1.0);
    (
        x0 + (x1 - x0) * lo,
        x0 + (x1 - x0) * hi.max(lo),
    )
}

/// Which thumb a pointer at `x` should grab, plus the value it maps to.
///
/// Ties go to the nearer thumb; a click outside the pair moves the closer one.
pub fn range_thumb_at(
    data: &crate::ir::RangeSliderData,
    rect: ComputedRect,
    x: f32,
) -> (u8, f32) {
    let (low, high) = data.clamped();
    let (lo_x, hi_x) = range_thumb_x(data, rect, low, high);
    let value = |x: f32| -> f32 {
        let knob_r = 7.0;
        let x0 = rect.x + knob_r;
        let x1 = rect.x + rect.w - knob_r;
        let frac = ((x - x0) / (x1 - x0).max(1.0)).clamp(0.0, 1.0);
        let span = data.max - data.min;
        crate::ir::snap(data.min + span * frac, data.min, data.step)
            .clamp(data.min.min(data.max), data.max.max(data.min))
    };
    let v = value(x);
    let d_lo = (x - lo_x).abs();
    let d_hi = (x - hi_x).abs();
    // Outside the span there is only one sensible answer.
    if v < low {
        return (0, v);
    }
    if v > high {
        return (1, v);
    }
    if d_lo <= d_hi {
        (0, v.min(high))
    } else {
        (1, v.max(low))
    }
}

/// Progress meter: a linear bar or circular ring, deterministic or sweeping.
#[allow(clippy::too_many_arguments)]
pub fn emit_progress(
    data: &crate::ir::ProgressData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    now_s: f64,
    dl: &mut DrawList,
) {
    if rect.w < 24.0 || rect.h < 4.0 {
        return;
    }
    let accent = if data.tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(data.tone)
    };
    let frac = data.fraction();
    let label = data.label.clone().filter(|l| !l.is_empty());
    let label_h = if label.is_some() { 18.0 } else { 0.0 };
    if let Some(text) = label.as_deref() {
        atlas.layout_dl(
            text,
            rect.x,
            rect.y,
            12.0,
            theme.text_secondary,
            false,
            None,
            dl,
        );
        if !data.indeterminate {
            let pct = format!("{}%", (frac * 100.0).round() as i64);
            let (pw, _) = atlas.measure(&pct, 12.0, false);
            atlas.layout_dl(
                &pct,
                rect.x + rect.w - pw,
                rect.y,
                12.0,
                theme.text_primary,
                false,
                None,
                dl,
            );
        }
    }

    match data.variant {
        crate::ir::ProgressVariant::Linear => {
            let track_h = 6.0_f32.min(rect.h - label_h).max(3.0);
            let track_y = rect.y + label_h + (rect.h - label_h - track_h) * 0.5;
            dl.push_quad(QuadInstance {
                rect: [rect.x, track_y, rect.w, track_h],
                radii: [track_h * 0.5; 4],
                color_top: theme.surface_3,
                color_bottom: theme.surface_3,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            let (fx, fw) = if data.indeterminate {
                // 28% sweep sliding across the track on a 1.6 s loop.
                let span = rect.w * 0.72;
                let phase = ((now_s / 1.6) % 1.0) as f32;
                (rect.x + span * phase, rect.w * 0.28)
            } else {
                (rect.x, rect.w * frac)
            };
            if fw > 0.5 {
                dl.push_quad(QuadInstance {
                    rect: [fx, track_y, fw, track_h],
                    radii: [track_h * 0.5; 4],
                    color_top: accent,
                    color_bottom: accent,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
            }
        }
        crate::ir::ProgressVariant::Circular => {
            let avail_h = (rect.h - label_h).max(4.0);
            let side = avail_h.min(rect.w).max(4.0);
            let cx = rect.x + rect.w * 0.5;
            let cy = rect.y + label_h + avail_h * 0.5;
            let r = (side * 0.5 - 2.0).max(2.0);
            let thickness = (side * 0.11).clamp(2.0, 6.0);
            arc(dl, cx, cy, r, thickness, 0.0, 1.0, theme.surface_3);
            if data.indeterminate {
                // A quarter-turn arc chasing around the ring on a 1.2 s loop.
                let phase = ((now_s / 1.2) % 1.0) as f32;
                arc(dl, cx, cy, r, thickness, phase, 0.25, accent);
            } else if frac > 0.001 {
                arc(dl, cx, cy, r, thickness, 0.0, frac, accent);
            }
        }
    }
}

/// Stroke a circular arc as short chords. `from`/`span` are fractions of a turn,
/// starting at 12 o'clock and sweeping clockwise.
fn arc(
    dl: &mut DrawList,
    cx: f32,
    cy: f32,
    r: f32,
    width: f32,
    from: f32,
    span: f32,
    color: u32,
) {
    let span = span.clamp(0.0, 1.0);
    if span <= 0.0 || r <= 0.0 {
        return;
    }
    // One chord per ~12 degrees keeps the ring smooth at 3x without many quads.
    let steps = ((span * 30.0).ceil() as usize).max(2);
    let step = span / steps as f32;
    let angle = |t: f32| {
        let a = -std::f32::consts::FRAC_PI_2 + t * std::f32::consts::TAU;
        (cx + a.cos() * r, cy + a.sin() * r)
    };
    for k in 0..steps {
        let t0 = from + step * k as f32;
        let t1 = t0 + step;
        let (x0, y0) = angle(t0);
        let (x1, y1) = angle(t1);
        dl.push_stroke(StrokeInstance::solid(
            [x0, y0],
            [x1, y1],
            width,
            0.7,
            color,
            0,
        ));
    }
}

/// Multi-line editable text area with focus ring, caret and placeholder.
#[allow(clippy::too_many_arguments)]
pub fn emit_textarea(
    id: &str,
    data: &crate::ir::TextAreaData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 40.0 || rect.h < 24.0 {
        return;
    }
    let hit_key = format!("textarea::{id}");
    let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
    let focused = interaction.focused_key.as_deref() == Some(hit_key.as_str());
    let target = if focused {
        1.0
    } else if hovered {
        0.5
    } else {
        0.0
    };
    let t = springs.animate(&hit_key, "focus", target, SpringConfig::SNAPPY);
    interaction.push_hit(rect, HitKind::TextArea { id: id.to_string() });

    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
    let focus_col = if data.tone == Tone::Neutral {
        theme.accent
    } else {
        theme.tone_color(data.tone)
    };
    let bg = lerp_rgba8(theme.surface_1, theme.surface_2, t * 0.65);
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

    let pad = 10.0;
    let px = 12.5;
    let inner_w = (rect.w - pad * 2.0).max(8.0);
    let _inner_h = (rect.h - pad * 2.0).max(8.0);
    let shown = if data.secure {
        "•".repeat(data.value.chars().count())
    } else {
        data.value.clone()
    };
    let body = if shown.is_empty() {
        data.placeholder.clone()
    } else {
        shown.clone()
    };
    let color = if shown.is_empty() {
        theme.text_muted
    } else {
        theme.text_primary
    };
    atlas.layout_dl(
        &body,
        rect.x + pad,
        rect.y + pad,
        px,
        color,
        false,
        Some(inner_w),
        dl,
    );
    if focused {
        emit_edit_affordances(
            &data.value,
            &shown,
            0,
            rect.x + pad,
            rect.y + pad,
            inner_w,
            px,
            interaction.edit.as_ref(),
            &hit_key,
            data.value.len(),
            focus_col,
            interaction.caret_visible,
            atlas,
            dl,
        );
    }
    dl.pop_clip();
}
