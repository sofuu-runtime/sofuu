//! Topological GPU Node-Edge Visualizer (`Graph`).
//!
//! Renders directed or workflow graphs with pure-Rust layered topological auto-layout
//! (when explicit coordinates are omitted), smooth cubic Bezier `StrokeInstance` edges,
//! arrowheads, active-path highlights, selectable/hoverable borderless node cards,
//! and strict per-graph `ClipRect` containment.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::components::fit_text;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance, StrokeInstance};
use crate::hit_test::{HitKind, InteractionState};
use crate::layout::ComputedRect;
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, Tone};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub subtitle: Option<String>,
    #[serde(default)]
    pub tone: Tone,
    /// Optional normalized canvas coordinate `[0..1, 0..1]`; auto-computed when `None`.
    #[serde(default)]
    pub pos: Option<[f32; 2]>,
}

impl GraphNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            subtitle: None,
            tone: Tone::Neutral,
            pos: None,
        }
    }

    pub fn subtitle(mut self, sub: impl Into<String>) -> Self {
        self.subtitle = Some(sub.into());
        self
    }

    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }

    pub fn pos(mut self, nx: f32, ny: f32) -> Self {
        self.pos = Some([nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)]);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub tone: Option<Tone>,
}

impl GraphEdge {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            label: None,
            active: false,
            tone: None,
        }
    }

    pub fn label(mut self, lbl: impl Into<String>) -> Self {
        self.label = Some(lbl.into());
        self
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = Some(tone);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphData {
    #[serde(default)]
    pub title: Option<String>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    #[serde(default)]
    pub selected_node: Option<String>,
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

impl Default for GraphData {
    fn default() -> Self {
        Self {
            title: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            selected_node: None,
            empty_text: None,
            border_width: 0.0,
            border_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
        }
    }
}

pub fn measure_graph(_data: &GraphData, avail_w: f32) -> (f32, f32) {
    (280.0_f32.min(avail_w.max(180.0)), 240.0)
}

#[allow(clippy::too_many_arguments)]
pub fn emit_graph(
    id: &str,
    data: &GraphData,
    rect: ComputedRect,
    theme: &Theme,
    atlas: &GlyphAtlas,
    springs: &mut SpringBank,
    interaction: &mut InteractionState,
    dl: &mut DrawList,
) {
    if rect.w < 96.0 || rect.h < 72.0 {
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
        let fitted = fit_text(atlas, title, 13.5, false, (iw - 76.0).max(24.0));
        atlas.layout_dl(&fitted, ix, cur_y, 13.5, theme.text_primary, false, None, dl);

        let stat_txt = format!("{}n / {}e", data.nodes.len(), data.edges.len());
        let (sw, _) = atlas.measure(&stat_txt, 11.0, false);
        if iw > sw + 60.0 {
            atlas.layout_dl(
                &stat_txt,
                ix + iw - sw,
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

    let canvas_h = (rect.y + rect.h - pad - cur_y).max(32.0);
    let canvas_rect = ComputedRect {
        x: ix,
        y: cur_y,
        w: iw,
        h: canvas_h,
        content_h: canvas_h,
        folded: false,
    };

    // Subtle canvas surface backdrop
    dl.push_quad(QuadInstance {
        rect: [canvas_rect.x, canvas_rect.y, canvas_rect.w, canvas_rect.h],
        radii: [theme.radius_control; 4],
        color_top: theme.surface_0,
        color_bottom: theme.surface_0,
        border_color: 0,
        shadow_color: 0,
        params: [0.0, 0.0, 0.0, 1.0],
        clip_id: 0,
        _pad: [0; 3],
    });

    if data.nodes.is_empty() {
        let msg = data.empty_text.as_deref().unwrap_or("Empty graph");
        atlas.layout_centered(
            msg,
            canvas_rect.x,
            canvas_rect.y,
            canvas_rect.w,
            canvas_rect.h,
            12.0,
            theme.text_muted,
            false,
            dl,
        );
        dl.pop_clip();
        return;
    }

    dl.push_clip(ClipRect::new(
        canvas_rect.x,
        canvas_rect.y,
        canvas_rect.w,
        canvas_rect.h,
    ));

    let positions = solve_graph_layout(&data.nodes, &data.edges);
    let max_cols = positions.values().map(|p| p.layer).max().unwrap_or(0) + 1;
    let max_rows = positions.values().map(|p| p.rows_in_layer).max().unwrap_or(1);

    let node_w = ((canvas_rect.w - 24.0) / (max_cols as f32 * 1.38).max(1.0)).clamp(74.0, 152.0);
    let has_subs = data
        .nodes
        .iter()
        .any(|n| n.subtitle.as_deref().is_some_and(|s| !s.is_empty()));
    let max_nh = ((canvas_rect.h - 20.0) / (max_rows as f32 * 1.22).max(1.0)).clamp(28.0, 56.0);
    let node_h = if has_subs {
        max_nh.min(48.0)
    } else {
        max_nh.min(36.0)
    };

    let pad_x = (node_w * 0.5 + 10.0).min(canvas_rect.w * 0.45);
    let pad_y = (node_h * 0.5 + 10.0).min(canvas_rect.h * 0.45);
    let span_x = (canvas_rect.w - pad_x * 2.0).max(1.0);
    let span_y = (canvas_rect.h - pad_y * 2.0).max(1.0);

    let mut rects_by_id: HashMap<&str, ComputedRect> = HashMap::with_capacity(data.nodes.len());
    for node in &data.nodes {
        let norm = node.pos.unwrap_or_else(|| {
            positions
                .get(node.id.as_str())
                .map(|p| p.norm)
                .unwrap_or([0.5, 0.5])
        });
        let cx = canvas_rect.x + pad_x + norm[0] * span_x;
        let cy = canvas_rect.y + pad_y + norm[1] * span_y;
        rects_by_id.insert(
            node.id.as_str(),
            ComputedRect {
                x: cx - node_w * 0.5,
                y: cy - node_h * 0.5,
                w: node_w,
                h: node_h,
                content_h: node_h,
                folded: false,
            },
        );
    }

    // 1. Draw Bezier edges behind nodes
    for edge in &data.edges {
        let Some(&from_r) = rects_by_id.get(edge.from.as_str()) else {
            continue;
        };
        let Some(&to_r) = rects_by_id.get(edge.to.as_str()) else {
            continue;
        };

        let edge_tone = edge
            .tone
            .unwrap_or(if edge.active { Tone::Info } else { Tone::Neutral });
        let base_col = if edge_tone == Tone::Neutral {
            scale_alpha(theme.text_muted, 0.55)
        } else {
            theme.tone_color(edge_tone)
        };
        let active_key = format!("{id}::edge::{}->{}", edge.from, edge.to);
        let active_t = springs.animate(
            &active_key,
            "active",
            if edge.active { 1.0 } else { 0.0 },
            SpringConfig::GENTLE,
        );
        let stroke_col = if edge.active {
            lerp_rgba8(scale_alpha(base_col, 0.65), base_col, 0.35 + active_t * 0.65)
        } else {
            base_col
        };
        let stroke_w = if edge.active { 2.2 } else { 1.4 };

        let (p0, p1, c1, c2) = if to_r.x >= from_r.x + from_r.w * 0.25 {
            // Left-to-right smooth Bezier
            let sx = from_r.x + from_r.w;
            let sy = from_r.y + from_r.h * 0.5;
            let ex = to_r.x;
            let ey = to_r.y + to_r.h * 0.5;
            let dx = ((ex - sx).abs() * 0.48).max(24.0);
            ([sx, sy], [ex, ey], [sx + dx, sy], [ex - dx, ey])
        } else {
            // Back-edge or vertical link: arch smoothly
            let sx = from_r.x + from_r.w * 0.5;
            let sy = from_r.y;
            let ex = to_r.x + to_r.w * 0.5;
            let ey = to_r.y;
            let lift = 28.0;
            ([sx, sy], [ex, ey], [sx, sy - lift], [ex, ey - lift])
        };

        let segs = 16usize;
        let mut prev = p0;
        for step in 1..=segs {
            let t = step as f32 / segs as f32;
            let pt = cubic_bezier(p0, c1, c2, p1, t);
            dl.push_stroke(StrokeInstance::solid(prev, pt, stroke_w, 0.9, stroke_col, 0));
            prev = pt;
        }

        // Directional arrowhead at p1
        let tan = cubic_bezier(p0, c1, c2, p1, 0.92);
        let dx = p1[0] - tan[0];
        let dy = p1[1] - tan[1];
        let len = (dx * dx + dy * dy).sqrt().max(1e-3);
        let ux = dx / len;
        let uy = dy / len;
        let nx = -uy;
        let ny = ux;
        let ah = 6.0;
        let aw = 3.2;
        let tip = p1;
        let left = [tip[0] - ux * ah + nx * aw, tip[1] - uy * ah + ny * aw];
        let right = [tip[0] - ux * ah - nx * aw, tip[1] - uy * ah - ny * aw];
        dl.push_stroke(StrokeInstance::solid(left, tip, stroke_w, 0.85, stroke_col, 0));
        dl.push_stroke(StrokeInstance::solid(right, tip, stroke_w, 0.85, stroke_col, 0));

        if let Some(lbl) = edge.label.as_deref().filter(|l| !l.is_empty()) {
            let mid = cubic_bezier(p0, c1, c2, p1, 0.5);
            let fitted = fit_text(atlas, lbl, 9.5, false, 72.0);
            let (lw, lh) = atlas.measure(&fitted, 9.5, false);
            if lw > 0.0 && canvas_rect.w >= 240.0 {
                let pw = lw + 10.0;
                let ph = lh + 4.0;
                dl.push_quad(QuadInstance {
                    rect: [mid[0] - pw * 0.5, mid[1] - ph * 0.5, pw, ph],
                    radii: [ph * 0.5; 4],
                    color_top: theme.surface_1,
                    color_bottom: theme.surface_1,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
                atlas.layout_centered(
                    &fitted,
                    mid[0] - pw * 0.5,
                    mid[1] - ph * 0.5,
                    pw,
                    ph,
                    9.5,
                    if edge.active { stroke_col } else { theme.text_secondary },
                    false,
                    dl,
                );
            }
        }
    }

    // 2. Draw interactive node cards in foreground
    for node in &data.nodes {
        let Some(&nr) = rects_by_id.get(node.id.as_str()) else {
            continue;
        };
        interaction.push_hit(
            nr,
            HitKind::GraphNode {
                graph_id: id.to_string(),
                node_id: node.id.clone(),
            },
        );

        let hit_key = format!("{id}::node::{}", node.id);
        let hovered = interaction.hovered_key.as_deref() == Some(hit_key.as_str());
        let selected = data.selected_node.as_deref() == Some(node.id.as_str());
        let hover_t = springs.animate(
            &hit_key,
            "hover",
            if hovered { 1.0 } else { 0.0 },
            SpringConfig::SNAPPY,
        );
        let tone_col = if node.tone == Tone::Neutral {
            theme.accent
        } else {
            theme.tone_color(node.tone)
        };

        let bg = if selected {
            lerp_rgba8(theme.surface_2, scale_alpha(tone_col, 0.24), 0.55)
        } else {
            lerp_rgba8(theme.surface_1, theme.surface_2, hover_t * 0.85)
        };

        dl.push_quad(QuadInstance {
            rect: [nr.x, nr.y, nr.w, nr.h],
            radii: [theme.radius_control; 4],
            color_top: bg,
            color_bottom: bg,
            border_color: if selected { tone_col } else { 0 },
            shadow_color: 0,
            params: [if selected { 1.25 } else { 0.0 }, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });

        // Status indicator dot
        let dot_r = 3.5;
        let dot_y = if has_subs && nr.h >= 40.0 {
            nr.y + 13.0
        } else {
            nr.y + nr.h * 0.5
        };
        dl.push_quad(QuadInstance {
            rect: [nr.x + 8.0, dot_y - dot_r, dot_r * 2.0, dot_r * 2.0],
            radii: [dot_r; 4],
            color_top: tone_col,
            color_bottom: tone_col,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });

        let text_x = nr.x + 20.0;
        let text_w = (nr.w - 26.0).max(12.0);
        let fitted_lbl = fit_text(atlas, &node.label, 11.5, false, text_w);
        if nr.h >= 40.0 {
            if let Some(sub) = node.subtitle.as_deref().filter(|s| !s.is_empty()) {
                atlas.layout_dl(
                    &fitted_lbl,
                    text_x,
                    nr.y + 7.0,
                    11.5,
                    theme.text_primary,
                    false,
                    None,
                    dl,
                );
                let fitted_sub = fit_text(atlas, sub, 9.5, false, text_w);
                atlas.layout_dl(
                    &fitted_sub,
                    text_x,
                    nr.y + 24.0,
                    9.5,
                    theme.text_muted,
                    false,
                    None,
                    dl,
                );
                continue;
            }
        }
        let py = atlas.centered_pen_y(nr.y, nr.h, 11.5);
        atlas.layout_dl(&fitted_lbl, text_x, py, 11.5, theme.text_primary, false, None, dl);
    }

    dl.pop_clip();
    dl.pop_clip();
}

struct NodeLayerPos {
    norm: [f32; 2],
    layer: usize,
    rows_in_layer: usize,
}

fn solve_graph_layout<'a>(
    nodes: &'a [GraphNode],
    edges: &[GraphEdge],
) -> HashMap<&'a str, NodeLayerPos> {
    let mut id_to_idx: HashMap<&str, usize> = HashMap::with_capacity(nodes.len());
    for (idx, n) in nodes.iter().enumerate() {
        id_to_idx.insert(n.id.as_str(), idx);
    }

    let n = nodes.len();
    let mut indeg = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in edges {
        if let (Some(&u), Some(&v)) = (id_to_idx.get(e.from.as_str()), id_to_idx.get(e.to.as_str()))
        {
            if u != v {
                adj[u].push(v);
                indeg[v] += 1;
            }
        }
    }

    let mut depth = vec![0usize; n];
    let mut queue = VecDeque::new();
    for (i, &d) in indeg.iter().enumerate() {
        if d == 0 {
            queue.push_back(i);
        }
    }

    let mut visited = 0usize;
    while let Some(u) = queue.pop_front() {
        visited += 1;
        for &v in &adj[u] {
            depth[v] = depth[v].max(depth[u] + 1);
            indeg[v] = indeg[v].saturating_sub(1);
            if indeg[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    // Cycle fallback: relax remaining edges safely up to `n` steps
    if visited < n {
        for _ in 0..n {
            let mut changed = false;
            for (u, nbrs) in adj.iter().enumerate() {
                for &v in nbrs {
                    let cand = (depth[u] + 1).min(n.saturating_sub(1));
                    if depth[v] < cand {
                        depth[v] = cand;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    let max_depth = depth.iter().copied().max().unwrap_or(0);
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for (idx, &d) in depth.iter().enumerate() {
        layers[d].push(idx);
    }

    let mut out = HashMap::with_capacity(n);
    for (layer_idx, group) in layers.iter().enumerate() {
        let nx = if max_depth == 0 {
            0.5
        } else {
            layer_idx as f32 / max_depth as f32
        };
        let count = group.len().max(1);
        for (slot, &node_idx) in group.iter().enumerate() {
            let ny = if count == 1 {
                0.5
            } else {
                (slot as f32 + 0.5) / count as f32
            };
            out.insert(
                nodes[node_idx].id.as_str(),
                NodeLayerPos {
                    norm: [nx, ny],
                    layer: layer_idx,
                    rows_in_layer: count,
                },
            );
        }
    }
    out
}

#[inline]
fn cubic_bezier(p0: [f32; 2], c1: [f32; 2], c2: [f32; 2], p1: [f32; 2], t: f32) -> [f32; 2] {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    let uuu = uu * u;
    let ttt = tt * t;
    [
        uuu * p0[0] + 3.0 * uu * t * c1[0] + 3.0 * u * tt * c2[0] + ttt * p1[0],
        uuu * p0[1] + 3.0 * uu * t * c1[1] + 3.0 * u * tt * c2[1] + ttt * p1[1],
    ]
}
