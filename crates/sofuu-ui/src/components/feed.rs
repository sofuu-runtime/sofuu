//! Adaptive GPU Event / Log Stream (`Feed`).
//!
//! Renders structured chronological items with semantic tone accents, badges,
//! timestamps, wrapped body summaries, live arrival pulses, kinetic scrolling,
//! and strict per-feed `ClipRect` isolation.

use serde::{Deserialize, Serialize};

use crate::components::fit_text;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, Tone};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedItem {
    pub id: String,
    #[serde(default)]
    pub time: String,
    #[serde(default)]
    pub badge: Option<String>,
    #[serde(default)]
    pub tone: Tone,
    pub title: String,
    #[serde(default)]
    pub body: Option<String>,
}

impl FeedItem {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            time: String::new(),
            badge: None,
            tone: Tone::Neutral,
            title: title.into(),
            body: None,
        }
    }

    pub fn time(mut self, t: impl Into<String>) -> Self {
        self.time = t.into();
        self
    }

    pub fn badge(mut self, badge: impl Into<String>) -> Self {
        self.badge = Some(badge.into());
        self
    }

    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }

    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedData {
    #[serde(default)]
    pub title: Option<String>,
    pub items: Vec<FeedItem>,
    #[serde(default = "default_max_items")]
    pub max_items: usize,
    #[serde(default = "default_true")]
    pub newest_first: bool,
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
    #[serde(default)]
    pub update_seq: u64,
}

impl Default for FeedData {
    fn default() -> Self {
        Self {
            title: None,
            items: Vec::new(),
            max_items: 200,
            newest_first: true,
            empty_text: None,
            border_width: 0.0,
            border_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
            update_seq: 0,
        }
    }
}

fn default_max_items() -> usize {
    200
}

fn default_true() -> bool {
    true
}

pub fn measure_feed(data: &FeedData, avail_w: f32) -> (f32, f32) {
    let title_h = if data.title.as_deref().is_some_and(|t| !t.is_empty()) {
        28.0
    } else {
        0.0
    };
    let rows = data.items.len().clamp(2, 5) as f32;
    let pref_h = (title_h + rows * 58.0 + 20.0).clamp(150.0, 340.0);
    (250.0_f32.min(avail_w.max(180.0)), pref_h)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_feed(
    id: &str,
    data: &FeedData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 80.0 || rect.h < 64.0 {
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

    if let Some(title) = data.title.as_deref().filter(|t| !t.is_empty()) {
        let fitted = fit_text(atlas, title, 13.5, false, (iw - 60.0).max(24.0));
        atlas.layout_dl(&fitted, ix, cur_y, 13.5, theme.text_primary, false, None, dl);

        let count_txt = format!("{}", data.items.len());
        let (cw, _) = atlas.measure(&count_txt, 11.0, false);
        if iw > cw + 48.0 {
            atlas.layout_dl(
                &count_txt,
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

    let body_h = (rect.y + rect.h - 10.0 - cur_y).max(20.0);
    if data.items.is_empty() {
        let msg = data.empty_text.as_deref().unwrap_or("No events recorded");
        atlas.layout_centered(msg, ix, cur_y, iw, body_h, 12.0, theme.text_muted, false, dl);
        dl.pop_clip();
        return;
    }

    let seq_key = format!("feed_seq::{id}");
    let prev_seq = springs.get(&seq_key, "seq", data.update_seq as f32);
    let pulse_target = if (data.update_seq as f32 - prev_seq).abs() >= 0.5 {
        springs.animate(&seq_key, "seq", data.update_seq as f32, SpringConfig::SNAPPY);
        1.0
    } else {
        0.0
    };
    let pulse_t = springs.animate(&format!("feed::{id}"), "pulse", pulse_target, SpringConfig::GENTLE);

    let item_gap = 6.0;
    let mut item_heights = Vec::with_capacity(data.items.len());
    let mut total_h = 0.0f32;
    for (idx, item) in data.items.iter().enumerate() {
        if idx > 0 {
            total_h += item_gap;
        }
        let has_body = item.body.as_deref().is_some_and(|b| !b.is_empty());
        let h = if has_body && iw >= 180.0 { 56.0 } else { 38.0 };
        item_heights.push(h);
        total_h += h;
    }

    let max_scroll = (total_h - body_h).max(0.0);
    let scroll_key = format!("feed::{id}");
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

    dl.push_clip(ClipRect::new(ix, cur_y, iw, body_h));

    let mut y_cursor = cur_y - scroll_y;
    for (display_idx, (item, &ih)) in data.items.iter().zip(item_heights.iter()).enumerate() {
        if y_cursor + ih >= cur_y && y_cursor <= cur_y + body_h {
            let item_rect = ComputedRect {
                x: ix,
                y: y_cursor,
                w: iw,
                h: ih,
                content_h: ih,
                folded: false,
            };
            interaction.push_hit(
                item_rect,
                HitKind::FeedItem {
                    feed_id: id.to_string(),
                    item_id: item.id.clone(),
                },
            );

            let hit_key = format!("{id}::feed::{}", item.id);
            let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
            let hover_t = springs.animate(
                &hit_key,
                "hover",
                if hovered { 1.0 } else { 0.0 },
                SpringConfig::SNAPPY,
            );
            let tone_col = if item.tone == Tone::Neutral {
                theme.accent
            } else {
                theme.tone_color(item.tone)
            };

            let mut bg = lerp_rgba8(theme.surface_0, theme.surface_2, hover_t * 0.8);
            if display_idx == 0 && pulse_t > 0.01 {
                bg = lerp_rgba8(bg, scale_alpha(tone_col, 0.22), pulse_t);
            }

            dl.push_quad(QuadInstance {
                rect: [ix, y_cursor, iw, ih],
                radii: [theme.radius_control; 4],
                color_top: bg,
                color_bottom: bg,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });

            // Left tone indicator dot/bar
            dl.push_quad(QuadInstance {
                rect: [ix + 6.0, y_cursor + 10.0, 3.0, (ih - 20.0).max(8.0)],
                radii: [1.5; 4],
                color_top: tone_col,
                color_bottom: tone_col,
                border_color: 0,
                shadow_color: 0,
                params: [0.0, 0.0, 0.0, 1.0],
                clip_id: 0,
                _pad: [0; 3],
            });

            let tx = ix + 16.0;
            let mut right_edge = ix + iw - 10.0;

            // Optional right-aligned timestamp
            if !item.time.is_empty() && iw > 150.0 {
                let fitted_time = fit_text(atlas, &item.time, 10.5, false, 76.0);
                let (tw, _) = atlas.measure(&fitted_time, 10.5, false);
                if tw > 0.0 {
                    right_edge -= tw + 8.0;
                    atlas.layout_dl(
                        &fitted_time,
                        right_edge + 8.0,
                        y_cursor + 9.0,
                        10.5,
                        theme.text_muted,
                        false,
                        None,
                        dl,
                    );
                }
            }

            // Optional tone badge pill next to title
            let mut title_x = tx;
            if let Some(badge) = item.badge.as_deref().filter(|b| !b.is_empty()) {
                let fitted_b = fit_text(atlas, badge, 10.0, false, 78.0);
                let (bw, _) = atlas.measure(&fitted_b, 10.0, false);
                if bw > 0.0 && title_x + bw + 20.0 < right_edge {
                    let pill_w = bw + 12.0;
                    let pill_h = 18.0;
                    let pill_y = y_cursor + 7.0;
                    dl.push_quad(QuadInstance {
                        rect: [title_x, pill_y, pill_w, pill_h],
                        radii: [9.0; 4],
                        color_top: scale_alpha(tone_col, 0.18),
                        color_bottom: scale_alpha(tone_col, 0.18),
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                    atlas.layout_centered(
                        &fitted_b,
                        title_x,
                        pill_y,
                        pill_w,
                        pill_h,
                        10.0,
                        tone_col,
                        false,
                        dl,
                    );
                    title_x += pill_w + 8.0;
                }
            }

            let title_avail = (right_edge - title_x).max(16.0);
            let fitted_title = fit_text(atlas, &item.title, 12.0, false, title_avail);
            atlas.layout_dl(
                &fitted_title,
                title_x,
                y_cursor + 9.0,
                12.0,
                theme.text_primary,
                false,
                None,
                dl,
            );

            if ih > 44.0 {
                if let Some(body) = item.body.as_deref().filter(|b| !b.is_empty()) {
                    let body_avail = (ix + iw - 10.0 - tx).max(16.0);
                    let fitted_body = fit_text(atlas, body, 11.0, false, body_avail);
                    atlas.layout_dl(
                        &fitted_body,
                        tx,
                        y_cursor + 31.0,
                        11.0,
                        theme.text_secondary,
                        false,
                        None,
                        dl,
                    );
                }
            }
        }
        y_cursor += ih + item_gap;
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
