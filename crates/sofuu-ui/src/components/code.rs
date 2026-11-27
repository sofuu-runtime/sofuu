//! Virtualized GPU Monospace Code / Structured Output Viewer (`CodeText`).
//!
//! Renders syntax-tinted monospace code or logs in `Geist Mono` with an aligned line-number
//! gutter, per-line diagnostic highlight tones, kinetic vertical scrolling, viewport virtualization,
//! and strict per-component `ClipRect` containment.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::components::fit_text;
use crate::draw::{scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::SpringBank;
use crate::text::GlyphAtlas;
use crate::theme::{Theme, Tone};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeHighlight {
    /// 1-based line number to highlight.
    pub line: usize,
    pub tone: Tone,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeData {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    pub content: String,
    #[serde(default = "default_true")]
    pub show_line_numbers: bool,
    #[serde(default)]
    pub highlights: Vec<CodeHighlight>,
    #[serde(default)]
    pub border_width: f32,
    #[serde(default)]
    pub border_color: Option<u32>,
    #[serde(default)]
    pub shadow_blur: f32,
    #[serde(default)]
    pub shadow_offset_y: f32,
}

impl Default for CodeData {
    fn default() -> Self {
        Self {
            title: None,
            language: None,
            content: String::new(),
            show_line_numbers: true,
            highlights: Vec::new(),
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

pub fn measure_code(data: &CodeData, avail_w: f32) -> (f32, f32) {
    let lines = data.content.lines().count().clamp(4, 14) as f32;
    let has_hdr = data.title.as_deref().is_some_and(|t| !t.is_empty())
        || data.language.as_deref().is_some_and(|l| !l.is_empty());
    let hdr_h = if has_hdr { 28.0 } else { 0.0 };
    let pref_h = (hdr_h + lines * 18.0 + 22.0).clamp(130.0, 340.0);
    (260.0_f32.min(avail_w.max(180.0)), pref_h)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_code(
    id: &str,
    data: &CodeData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    _springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 80.0 || rect.h < 56.0 {
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
    let iw = (rect.w - pad * 2.0).max(24.0);
    let mut cur_y = rect.y + 10.0;

    let lines: Vec<&str> = if data.content.is_empty() {
        vec![""]
    } else {
        data.content.split('\n').map(|l| l.trim_end_matches('\r')).collect()
    };

    let has_hdr = data.title.as_deref().is_some_and(|t| !t.is_empty())
        || data.language.as_deref().is_some_and(|l| !l.is_empty());
    if has_hdr && rect.h >= 92.0 {
        let mut right_edge = ix + iw;
        let count_lbl = format!("{} ln", lines.len());
        let (cw, _) = atlas.measure(&count_lbl, 10.5, false);
        if iw > cw + 90.0 {
            right_edge -= cw + 8.0;
            atlas.layout_dl(
                &count_lbl,
                right_edge + 8.0,
                cur_y + 2.0,
                10.5,
                theme.text_muted,
                false,
                None,
                dl,
            );
        }

        if let Some(lang) = data.language.as_deref().filter(|l| !l.is_empty()) {
            let fitted_l = fit_text(atlas, lang, 10.0, false, 76.0);
            let (lw, _) = atlas.measure(&fitted_l, 10.0, false);
            if lw > 0.0 && right_edge - (lw + 16.0) > ix + 24.0 {
                let pw = lw + 12.0;
                let ph = 18.0;
                let px = right_edge - pw;
                dl.push_quad(QuadInstance {
                    rect: [px, cur_y, pw, ph],
                    radii: [9.0; 4],
                    color_top: scale_alpha(theme.accent, 0.16),
                    color_bottom: scale_alpha(theme.accent, 0.16),
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
                atlas.layout_centered(
                    &fitted_l,
                    px,
                    cur_y,
                    pw,
                    ph,
                    10.0,
                    theme.accent,
                    false,
                    dl,
                );
                right_edge = px - 8.0;
            }
        }

        if let Some(title) = data.title.as_deref().filter(|t| !t.is_empty()) {
            let fitted_t = fit_text(atlas, title, 13.0, false, (right_edge - ix).max(16.0));
            atlas.layout_dl(&fitted_t, ix, cur_y + 1.0, 13.0, theme.text_primary, false, None, dl);
        }
        cur_y += 26.0;
    }

    let body_h = (rect.y + rect.h - pad - cur_y).max(20.0);
    dl.push_quad(QuadInstance {
        rect: [ix, cur_y, iw, body_h],
        radii: [theme.radius_control; 4],
        color_top: theme.surface_0,
        color_bottom: theme.surface_0,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    let line_h = 18.5f32;
    let total_h = lines.len() as f32 * line_h + 12.0;
    let max_scroll = (total_h - body_h).max(0.0);
    let scroll_key = format!("code::{id}");
    if max_scroll > 0.0 {
        interaction.push_hit(
            ComputedRect {
                x: ix,
                y: cur_y,
                w: iw,
                h: body_h,
                content_h: total_h,
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

    let mut hl_map: HashMap<usize, Tone> = HashMap::with_capacity(data.highlights.len());
    for h in &data.highlights {
        hl_map.insert(h.line, h.tone);
    }

    let show_gutter = data.show_line_numbers && iw >= 140.0;
    let digits = lines.len().max(1).to_string().len().max(2);
    let gutter_w = if show_gutter {
        digits as f32 * 7.6 + 18.0
    } else {
        0.0
    };
    let code_x = ix + gutter_w + 8.0;
    let code_avail_w = (ix + iw - 10.0 - code_x).max(16.0);

    dl.push_clip(ClipRect::new(ix, cur_y, iw, body_h));

    if show_gutter {
        dl.push_quad(QuadInstance {
            rect: [ix, cur_y, gutter_w, body_h],
            radii: [theme.radius_control, 0.0, 0.0, theme.radius_control],
            color_top: scale_alpha(theme.surface_1, 0.45),
            color_bottom: scale_alpha(theme.surface_1, 0.45),
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }

    let start_idx = ((scroll_y / line_h).floor() as usize).min(lines.len());
    let end_idx = (((scroll_y + body_h) / line_h).ceil() as usize + 1).min(lines.len());

    for idx in start_idx..end_idx {
        let line_no = idx + 1;
        let ly = cur_y + 6.0 + idx as f32 * line_h - scroll_y;

        if let Some(&tone) = hl_map.get(&line_no) {
            let tc = if tone == Tone::Neutral {
                theme.accent
            } else {
                theme.tone_color(tone)
            };
            dl.push_quad(QuadInstance {
                rect: [ix, ly, iw, line_h],
                radii: [0.0; 4],
                color_top: scale_alpha(tc, 0.14),
                color_bottom: scale_alpha(tc, 0.14),
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });
            dl.push_quad(QuadInstance {
                rect: [ix, ly + 2.0, 2.5, (line_h - 4.0).max(8.0)],
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

        if show_gutter {
            let num_str = format!("{line_no}");
            let (nw, _) = atlas.measure(&num_str, 11.0, true);
            let num_col = if hl_map.contains_key(&line_no) {
                theme.text_primary
            } else {
                theme.text_muted
            };
            atlas.layout_dl(
                &num_str,
                ix + gutter_w - 8.0 - nw,
                ly + 2.0,
                11.0,
                num_col,
                true,
                None,
                dl,
            );
        }

        let raw_line = lines[idx];
        let fitted = fit_text(atlas, raw_line, 11.5, true, code_avail_w);
        emit_tinted_code_line(atlas, &fitted, code_x, ly + 2.0, 11.5, theme, dl);
    }

    if max_scroll > 1.0 {
        let track_h = (body_h - 8.0).max(20.0);
        let thumb_h = ((body_h / total_h) * track_h).clamp(20.0, track_h);
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

/// Lightweight, language-agnostic token coloring (comments, strings, numbers, punctuation/words).
fn emit_tinted_code_line(
    atlas: &GlyphAtlas,
    line: &str,
    x: f32,
    y: f32,
    px: f32,
    theme: &Theme,
    dl: &mut DrawList,
) {
    if line.is_empty() {
        return;
    }
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("--") {
        atlas.layout_dl(line, x, y, px, theme.text_muted, true, None, dl);
        return;
    }

    let bytes = line.as_bytes();
    let mut i = 0usize;
    let mut pen_x = x;

    while i < bytes.len() {
        let ch = line[i..].chars().next().unwrap_or(' ');
        // Inline comment tail
        if line[i..].starts_with("//") {
            let tail = &line[i..];
            atlas.layout_dl(tail, pen_x, y, px, theme.text_muted, true, None, dl);
            break;
        }
        // Quoted string literal
        if ch == '"' || ch == '\'' || ch == '`' {
            let quote = ch;
            let start = i;
            i += ch.len_utf8();
            let mut esc = false;
            while i < bytes.len() {
                let c2 = line[i..].chars().next().unwrap_or(' ');
                i += c2.len_utf8();
                if esc {
                    esc = false;
                } else if c2 == '\\' {
                    esc = true;
                } else if c2 == quote {
                    break;
                }
            }
            let seg = &line[start..i];
            atlas.layout_dl(seg, pen_x, y, px, theme.ok, true, None, dl);
            pen_x += atlas.measure(seg, px, true).0;
            continue;
        }
        // Word token
        if ch.is_ascii_alphanumeric() || ch == '_' {
            let start = i;
            while i < bytes.len() {
                let c2 = line[i..].chars().next().unwrap_or(' ');
                if c2.is_ascii_alphanumeric() || c2 == '_' || c2 == '.' {
                    i += c2.len_utf8();
                } else {
                    break;
                }
            }
            let tok = &line[start..i];
            let col = if is_common_keyword(tok) {
                theme.accent
            } else if tok.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                theme.warn
            } else {
                theme.text_primary
            };
            atlas.layout_dl(tok, pen_x, y, px, col, true, None, dl);
            pen_x += atlas.measure(tok, px, true).0;
            continue;
        }
        // Whitespace or punctuation run
        let start = i;
        while i < bytes.len() {
            let c2 = line[i..].chars().next().unwrap_or(' ');
            if c2.is_ascii_alphanumeric()
                || matches!(c2, '_' | '"' | '\'' | '`')
                || line[i..].starts_with("//")
            {
                break;
            }
            i += c2.len_utf8();
        }
        let seg = &line[start..i];
        atlas.layout_dl(seg, pen_x, y, px, theme.text_secondary, true, None, dl);
        pen_x += atlas.measure(seg, px, true).0;
    }
}

fn is_common_keyword(tok: &str) -> bool {
    matches!(
        tok,
        "fn" | "let"
            | "const"
            | "var"
            | "pub"
            | "struct"
            | "enum"
            | "impl"
            | "return"
            | "if"
            | "else"
            | "match"
            | "for"
            | "while"
            | "async"
            | "await"
            | "import"
            | "export"
            | "function"
            | "true"
            | "false"
            | "null"
            | "None"
            | "Some"
            | "Ok"
            | "Err"
    )
}
