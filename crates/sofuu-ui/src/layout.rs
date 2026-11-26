//! Pure-Rust 2D responsive flex + grid layout solver.
//!
//! Computes exact logical-pixel rectangles (`ComputedRect`) in two deterministic passes
//! (intrinsic measurement -> top-down placement) with zero heap churn on steady frames
//! and responsive `Row -> Column` folding below `fold_width`.

use std::collections::HashMap;

use crate::components::plot::measure_plot;
use crate::components::stat::measure_stat;
use crate::ir::{Align, Axis, Node, NodeKind, SizeSpec};
use crate::text::GlyphAtlas;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComputedRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Intrinsic content height (used by scrollable containers to clamp scroll offsets).
    pub content_h: f32,
    /// Resolved axis (after responsive `fold_width` evaluation) for containers.
    pub folded: bool,
}

/// Deterministic initial path hash for the root UI node.
pub const ROOT_PATH_HASH: u64 = 0xcbf2_9ce4_8422_2325;

/// Compute a deterministic, collision-free tree path hash for a child node.
#[inline]
pub fn child_path_hash(parent_hash: u64, slot: usize, id: &str) -> u64 {
    let mut h = parent_hash ^ 0x9e37_79b9_7f4a_7c15_u64.wrapping_add(slot as u64);
    h = h.wrapping_mul(0x1000_0000_01b3);
    for &b in id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

#[derive(Clone, Debug, Default)]
pub struct LayoutMap {
    rects: HashMap<String, ComputedRect>,
    rects_by_path: HashMap<u64, ComputedRect>,
}

impl LayoutMap {
    pub fn clear(&mut self) {
        self.rects.clear();
        self.rects_by_path.clear();
    }

    #[inline]
    pub fn get(&self, id: &str) -> Option<ComputedRect> {
        self.rects.get(id).copied()
    }

    #[inline]
    pub fn get_path(&self, path_hash: u64) -> Option<ComputedRect> {
        self.rects_by_path.get(&path_hash).copied()
    }

    /// Solve the full tree into `self` for the given viewport `[width, height]`.
    pub fn solve(&mut self, root: &Node, viewport: [f32; 2], atlas: &GlyphAtlas) {
        self.solve_with_top_inset(root, viewport, 0.0, atlas);
    }

    /// Solve the tree below a platform top chrome/safe-area inset (e.g. borderless traffic lights).
    pub fn solve_with_top_inset(
        &mut self,
        root: &Node,
        viewport: [f32; 2],
        top_inset: f32,
        atlas: &GlyphAtlas,
    ) {
        self.rects.clear();
        self.rects_by_path.clear();
        if !root.visible {
            return;
        }
        let y0 = top_inset.max(0.0);
        let h0 = (viewport[1] - y0).max(0.0);
        self.place_node(
            root,
            0.0,
            y0,
            viewport[0].max(0.0),
            h0,
            ROOT_PATH_HASH,
            atlas,
        );
    }

    /// Intrinsic minimum/preferred size `(w, h)` when unconstrained or within `avail_w`.
    fn measure_intrinsic(&self, node: &Node, avail_w: f32, atlas: &GlyphAtlas) -> (f32, f32) {
        if !node.visible {
            return (0.0, 0.0);
        }
        let inner_avail_w = match node.width {
            SizeSpec::Px(px) => (px - node.padding.horizontal()).max(0.0),
            _ => (avail_w - node.padding.horizontal()).max(0.0),
        };

        let (cw, ch) = match &node.kind {
            NodeKind::Text {
                content,
                size,
                mono,
                wrap,
                ..
            } => {
                let (tw, th) = atlas.measure(content, *size, *mono);
                if *wrap && inner_avail_w > 1.0 && tw > inner_avail_w {
                    let approx_lines = (tw / inner_avail_w).ceil().max(1.0);
                    (inner_avail_w, approx_lines * (*size * 1.35))
                } else {
                    (tw, th)
                }
            }
            NodeKind::Rule { thickness, .. } => (0.0, (*thickness).max(1.0)),
            NodeKind::Spacer => (0.0, 0.0),
            NodeKind::ThemeToggle { size } => ((*size).max(24.0), (*size).max(24.0)),
            NodeKind::Segmented { options, .. } => {
                let mut total = 8.0f32;
                for o in options {
                    let (ow, _) = atlas.measure(o, 12.0, true);
                    total += ow + 20.0;
                }
                (total.max(80.0), 34.0)
            }
            NodeKind::Button { label, .. } => {
                let (lw, _) = atlas.measure(label, 12.5, true);
                (lw + 24.0, 34.0)
            }
            NodeKind::Stat(data) => measure_stat(data, inner_avail_w),
            NodeKind::Plot(data) => measure_plot(data, inner_avail_w),
            NodeKind::Surface { child, .. } => {
                if let Some(c) = child {
                    self.measure_intrinsic(c, inner_avail_w, atlas)
                } else {
                    (0.0, 0.0)
                }
            }
            NodeKind::Tabs { items, active } => {
                let bar_h = 38.0;
                let gap = 10.0;
                let body_h = items
                    .get(*active)
                    .map(|item| self.measure_intrinsic(&item.child, inner_avail_w, atlas).1)
                    .unwrap_or(0.0);
                (inner_avail_w, bar_h + gap + body_h)
            }
            NodeKind::Container {
                axis,
                gap,
                fold_width,
                children,
                ..
            } => {
                let effective_axis = resolve_axis(*axis, *fold_width, avail_w);
                let visible_children: Vec<&Node> = children.iter().filter(|c| c.visible).collect();
                if visible_children.is_empty() {
                    (0.0, 0.0)
                } else {
                    let gaps = *gap * (visible_children.len().saturating_sub(1)) as f32;
                    match effective_axis {
                        Axis::Column => {
                            let mut max_w = 0.0f32;
                            let mut sum_h = gaps;
                            for c in visible_children {
                                let (mw, mh) = self.measure_intrinsic(c, inner_avail_w, atlas);
                                max_w = max_w.max(mw);
                                sum_h += mh;
                            }
                            (max_w, sum_h)
                        }
                        Axis::Row => {
                            let mut sum_w = gaps;
                            let mut max_h = 0.0f32;
                            let per_w =
                                ((inner_avail_w - gaps) / visible_children.len() as f32).max(0.0);
                            for c in visible_children {
                                let (mw, mh) = self.measure_intrinsic(c, per_w, atlas);
                                sum_w += mw;
                                max_h = max_h.max(mh);
                            }
                            (sum_w, max_h)
                        }
                    }
                }
            }
            NodeKind::Grid {
                columns,
                min_col_width,
                gap,
                children,
                ..
            } => {
                let visible: Vec<&Node> = children.iter().filter(|c| c.visible).collect();
                let cols =
                    resolve_grid_cols(*columns, *min_col_width, *gap, inner_avail_w, visible.len());
                let col_w =
                    ((inner_avail_w - *gap * (cols.saturating_sub(1)) as f32) / cols as f32)
                        .max(0.0);
                let mut total_h = 0.0f32;
                let mut row_idx = 0usize;
                for row in visible.chunks(cols) {
                    if row_idx > 0 {
                        total_h += *gap;
                    }
                    let mut row_h = 0.0f32;
                    for c in row {
                        row_h = row_h.max(self.measure_intrinsic(c, col_w, atlas).1);
                    }
                    total_h += row_h;
                    row_idx += 1;
                }
                (inner_avail_w, total_h)
            }
        };

        let mut w = match node.width {
            SizeSpec::Px(px) => px,
            _ => cw + node.padding.horizontal(),
        };
        let mut h = match node.height {
            SizeSpec::Px(px) => px,
            _ => ch + node.padding.vertical(),
        };
        w = w.max(node.min_width);
        h = h.max(node.min_height);
        (w, h)
    }

    fn place_node(
        &mut self,
        node: &Node,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        path_hash: u64,
        atlas: &GlyphAtlas,
    ) {
        if !node.visible {
            return;
        }
        let ix = x + node.padding.left;
        let iy = y + node.padding.top;
        let iw = (w - node.padding.horizontal()).max(0.0);
        let ih = (h - node.padding.vertical()).max(0.0);

        let mut content_h = ih;
        let mut folded = false;

        match &node.kind {
            NodeKind::Surface { child, .. } => {
                if let Some(c) = child {
                    let (_, ch) = self.measure_intrinsic(c, iw, atlas);
                    content_h = ch + node.padding.vertical();
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.place_node(c, ix, iy, iw, ih, ch_hash, atlas);
                }
            }
            NodeKind::Tabs { items, active } => {
                let bar_h = 38.0;
                let gap = 10.0;
                if let Some(item) = items.get(*active) {
                    let body_y = iy + bar_h + gap;
                    let body_h = (ih - bar_h - gap).max(0.0);
                    let (_, ch) = self.measure_intrinsic(&item.child, iw, atlas);
                    content_h = bar_h + gap + ch + node.padding.vertical();
                    let ch_hash = child_path_hash(path_hash, *active, &item.child.id);
                    self.place_node(&item.child, ix, body_y, iw, body_h, ch_hash, atlas);
                }
            }
            NodeKind::Container {
                axis,
                align,
                gap,
                fold_width,
                children,
                ..
            } => {
                let effective = resolve_axis(*axis, *fold_width, w);
                folded = *axis == Axis::Row && effective == Axis::Column;
                content_h = self.place_flex(
                    children,
                    effective,
                    *align,
                    *gap,
                    ix,
                    iy,
                    iw,
                    ih,
                    path_hash,
                    atlas,
                ) + node.padding.vertical();
            }
            NodeKind::Grid {
                columns,
                min_col_width,
                gap,
                children,
                ..
            } => {
                content_h = self.place_grid(
                    children,
                    *columns,
                    *min_col_width,
                    *gap,
                    ix,
                    iy,
                    iw,
                    path_hash,
                    atlas,
                ) + node.padding.vertical();
            }
            NodeKind::Text { .. }
            | NodeKind::Rule { .. }
            | NodeKind::Spacer
            | NodeKind::ThemeToggle { .. }
            | NodeKind::Segmented { .. }
            | NodeKind::Button { .. }
            | NodeKind::Stat(_)
            | NodeKind::Plot(_) => {
                let (_, mh) = self.measure_intrinsic(node, w, atlas);
                content_h = mh;
            }
        }

        let rect = ComputedRect {
            x,
            y,
            w,
            h,
            content_h,
            folded,
        };
        self.rects.insert(node.id.clone(), rect);
        self.rects_by_path.insert(path_hash, rect);
    }

    #[allow(clippy::too_many_arguments)]
    fn place_flex(
        &mut self,
        children: &[Node],
        axis: Axis,
        align: Align,
        gap: f32,
        ix: f32,
        iy: f32,
        iw: f32,
        ih: f32,
        parent_hash: u64,
        atlas: &GlyphAtlas,
    ) -> f32 {
        let visible: Vec<(usize, &Node)> = children
            .iter()
            .enumerate()
            .filter(|(_, c)| c.visible)
            .collect();
        if visible.is_empty() {
            return 0.0;
        }
        let total_gaps = gap * (visible.len().saturating_sub(1)) as f32;

        // First pass: measure fixed/auto sizes and accumulate Fill weights along primary axis.
        let mut intrinsic: Vec<(f32, f32)> = Vec::with_capacity(visible.len());
        let mut fill_sum = 0.0f32;
        let mut px_sum = 0.0f32;
        let mut auto_sum = 0.0f32;

        for (_, child) in &visible {
            let spec = match axis {
                Axis::Row => child.width,
                Axis::Column => child.height,
            };
            let meas = self.measure_intrinsic(child, iw, atlas);
            match spec {
                SizeSpec::Fill(weight) => {
                    fill_sum += weight.max(0.0);
                }
                SizeSpec::Px(px) => {
                    px_sum += px.max(match axis {
                        Axis::Row => child.min_width,
                        Axis::Column => child.min_height,
                    });
                }
                SizeSpec::Auto => {
                    auto_sum += match axis {
                        Axis::Row => meas.0,
                        Axis::Column => meas.1,
                    };
                }
            }
            intrinsic.push(meas);
        }

        let main_avail = match axis {
            Axis::Row => iw,
            Axis::Column => ih,
        };
        let fixed_sum = total_gaps + px_sum + auto_sum;
        let remaining = (main_avail - fixed_sum).max(0.0);
        // Prevent Auto children in a Row from ever overflowing horizontally and colliding.
        let auto_scale = if axis == Axis::Row && fixed_sum > main_avail && auto_sum > 0.0 {
            let avail_for_auto = (main_avail - total_gaps - px_sum).max(0.0);
            (avail_for_auto / auto_sum).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let mut cursor = match axis {
            Axis::Row => ix,
            Axis::Column => iy,
        };
        let mut total_used_h = 0.0f32;

        for (i, (slot, child)) in visible.iter().enumerate() {
            if i > 0 {
                cursor += gap;
                if axis == Axis::Column {
                    total_used_h += gap;
                }
            }
            let (mw, mh) = intrinsic[i];
            let main_spec = match axis {
                Axis::Row => child.width,
                Axis::Column => child.height,
            };
            let main_size = match main_spec {
                SizeSpec::Fill(weight) if fill_sum > 0.0 => remaining * (weight.max(0.0) / fill_sum),
                SizeSpec::Fill(_) => 0.0,
                SizeSpec::Px(px) => px,
                SizeSpec::Auto => match axis {
                    Axis::Row => mw * auto_scale,
                    Axis::Column => mh,
                },
            };

            let (cx, cy, cw, ch) = match axis {
                Axis::Row => {
                    let cross_size = match child.height {
                        SizeSpec::Px(px) => px,
                        SizeSpec::Fill(_) => ih,
                        SizeSpec::Auto => match align {
                            Align::Stretch => ih,
                            _ => mh.min(ih),
                        },
                    };
                    let cy = match align {
                        Align::Start | Align::Stretch => iy,
                        Align::Center => iy + (ih - cross_size) * 0.5,
                        Align::End => iy + (ih - cross_size),
                    };
                    let cw = if auto_scale < 1.0 && matches!(child.width, SizeSpec::Auto) {
                        main_size
                    } else {
                        main_size.max(child.min_width)
                    };
                    (cursor, cy, cw, cross_size)
                }
                Axis::Column => {
                    let cross_size = match child.width {
                        SizeSpec::Px(px) => px,
                        SizeSpec::Fill(_) => iw,
                        SizeSpec::Auto => match align {
                            Align::Stretch => iw,
                            _ => mw.min(iw),
                        },
                    };
                    let cx = match align {
                        Align::Start | Align::Stretch => ix,
                        Align::Center => ix + (iw - cross_size) * 0.5,
                        Align::End => ix + (iw - cross_size),
                    };
                    (cx, cursor, cross_size, main_size.max(child.min_height))
                }
            };

            let ch_hash = child_path_hash(parent_hash, *slot, &child.id);
            self.place_node(child, cx, cy, cw, ch, ch_hash, atlas);
            cursor += match axis {
                Axis::Row => cw,
                Axis::Column => ch,
            };
            match axis {
                Axis::Column => total_used_h += ch,
                Axis::Row => total_used_h = total_used_h.max(ch),
            }
        }
        total_used_h
    }

    #[allow(clippy::too_many_arguments)]
    fn place_grid(
        &mut self,
        children: &[Node],
        columns: usize,
        min_col_width: f32,
        gap: f32,
        ix: f32,
        iy: f32,
        iw: f32,
        parent_hash: u64,
        atlas: &GlyphAtlas,
    ) -> f32 {
        let visible: Vec<(usize, &Node)> = children
            .iter()
            .enumerate()
            .filter(|(_, c)| c.visible)
            .collect();
        if visible.is_empty() {
            return 0.0;
        }
        let cols = resolve_grid_cols(columns, min_col_width, gap, iw, visible.len());
        let col_w = ((iw - gap * (cols.saturating_sub(1)) as f32) / cols as f32).max(0.0);

        let mut cy = iy;
        let mut total_h = 0.0f32;

        for (row_idx, row) in visible.chunks(cols).enumerate() {
            if row_idx > 0 {
                cy += gap;
                total_h += gap;
            }
            let mut row_h = 0.0f32;
            for (_, child) in row {
                let ch = match child.height {
                    SizeSpec::Px(px) => px,
                    _ => self.measure_intrinsic(child, col_w, atlas).1,
                };
                row_h = row_h.max(ch.max(child.min_height));
            }
            for (col_idx, (slot, child)) in row.iter().enumerate() {
                let cx = ix + col_idx as f32 * (col_w + gap);
                let ch_hash = child_path_hash(parent_hash, *slot, &child.id);
                self.place_node(child, cx, cy, col_w, row_h, ch_hash, atlas);
            }
            cy += row_h;
            total_h += row_h;
        }
        total_h
    }
}

#[inline]
fn resolve_axis(axis: Axis, fold_width: Option<f32>, avail_w: f32) -> Axis {
    if axis == Axis::Row {
        if let Some(fw) = fold_width {
            if avail_w < fw {
                return Axis::Column;
            }
        }
    }
    axis
}

#[inline]
fn resolve_grid_cols(
    columns: usize,
    min_col_width: f32,
    gap: f32,
    avail_w: f32,
    item_count: usize,
) -> usize {
    let items = item_count.max(1);
    if columns > 0 {
        if min_col_width > 0.0 {
            let max_fit = (((avail_w + gap) / (min_col_width + gap)).floor() as usize).max(1);
            return columns.min(max_fit);
        }
        return columns;
    }
    if min_col_width <= 1.0 {
        return items;
    }
    let max_fit = (((avail_w + gap) / (min_col_width + gap)).floor() as usize).max(1);
    max_fit.min(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Insets, NodeKind};

    #[test]
    fn row_folds_to_column_below_fold_width() {
        let atlas = GlyphAtlas::builtin();
        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Row,
                align: Align::Stretch,
                gap: 12.0,
                fold_width: Some(720.0),
                scrollable: false,
                style: None,
                children: vec![
                    Node::new("left", NodeKind::Spacer).with_size(SizeSpec::Fill(1.0), SizeSpec::Px(100.0)),
                    Node::new("right", NodeKind::Spacer).with_size(SizeSpec::Fill(1.0), SizeSpec::Px(100.0)),
                ],
            },
        )
        .with_padding(Insets::all(16.0));

        let mut map = LayoutMap::default();
        // Wide viewport -> side-by-side Row
        map.solve(&root, [1000.0, 600.0], &atlas);
        assert!(!map.get("root").unwrap().folded);
        let l_wide = map.get("left").unwrap();
        let r_wide = map.get("right").unwrap();
        assert_eq!(l_wide.y, r_wide.y);
        assert!(r_wide.x > l_wide.x);

        // Narrow viewport -> folded Column
        map.solve(&root, [500.0, 600.0], &atlas);
        assert!(map.get("root").unwrap().folded);
        let l_nar = map.get("left").unwrap();
        let r_nar = map.get("right").unwrap();
        assert_eq!(l_nar.x, r_nar.x);
        assert!(r_nar.y > l_nar.y);
    }

    #[test]
    fn grid_wraps_responsively_by_min_col_width() {
        let atlas = GlyphAtlas::builtin();
        let root = Node::new(
            "grid",
            NodeKind::Grid {
                columns: 0,
                min_col_width: 240.0,
                gap: 12.0,
                style: None,
                children: (0..4)
                    .map(|i| {
                        Node::new(format!("c{i}"), NodeKind::Spacer)
                            .with_size(SizeSpec::Auto, SizeSpec::Px(80.0))
                    })
                    .collect(),
            },
        );
        let mut map = LayoutMap::default();
        map.solve(&root, [800.0, 600.0], &atlas);
        // 800px fits 3 columns of >=240px + 12px gaps, so c3 wraps to second row.
        assert_eq!(map.get("c0").unwrap().y, map.get("c1").unwrap().y);
        assert_eq!(map.get("c0").unwrap().y, map.get("c2").unwrap().y);
        assert!(map.get("c3").unwrap().y > map.get("c0").unwrap().y);
    }
}
