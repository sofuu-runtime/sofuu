//! Pure-Rust 2D responsive flex + grid layout solver.
//!
//! Computes exact logical-pixel rectangles (`ComputedRect`) in two deterministic passes
//! (intrinsic measurement -> top-down placement) with zero heap churn on steady frames
//! and responsive `Row -> Column` folding below `fold_width`.

use std::collections::HashMap;

use crate::components::code::measure_code;
use crate::components::feed::measure_feed;
use crate::components::graph::measure_graph;
use crate::components::plot::measure_plot;
use crate::components::stat::measure_stat;
use crate::components::table::measure_table;
use crate::ir::{
    Align, Axis, Insets, ListViewData, Node, NodeKind, RichTextData, SizeSpec,
};
use crate::text::{FontFamily, GlyphAtlas, RichSpan, TextStyle};

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
    /// Persistent measured row heights for virtualized lists (survives frame clears).
    item_heights: HashMap<String, f32>,
    /// Live scroll offsets supplied by the engine for this solve.
    scrolls: HashMap<String, f32>,
    /// Platform safe-area insets added to every `SafeArea` node.
    extra_insets: Insets,
    /// Visible item slot range `[start, end)` per virtualized list.
    list_ranges: HashMap<String, (usize, usize)>,
}

impl LayoutMap {
    pub fn clear(&mut self) {
        self.rects.clear();
        self.rects_by_path.clear();
        self.list_ranges.clear();
    }

    #[inline]
    pub fn get(&self, id: &str) -> Option<ComputedRect> {
        self.rects.get(id).copied()
    }

    #[inline]
    pub fn get_path(&self, path_hash: u64) -> Option<ComputedRect> {
        self.rects_by_path.get(&path_hash).copied()
    }

    /// Visible item slots `[start, end)` for a virtualized list this frame.
    #[inline]
    pub fn list_range(&self, id: &str) -> Option<(usize, usize)> {
        self.list_ranges.get(id).copied()
    }

    /// Cached measured row height for a virtualized list item.
    #[inline]
    pub fn item_height(&self, list_id: &str, item_id: &str) -> Option<f32> {
        self.item_heights.get(&format!("{list_id}::{item_id}")).copied()
    }

    /// Test-only injection of a rectangle for intentionally malformed layout
    /// (audit tests need overlapping siblings, which the solver never emits).
    #[cfg(test)]
    pub(crate) fn set_path_for_test(&mut self, path_hash: u64, rect: ComputedRect) {
        self.rects_by_path.insert(path_hash, rect);
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
        self.solve_full(
            root,
            viewport,
            top_inset,
            atlas,
            &HashMap::new(),
            Insets::ZERO,
        );
    }

    /// Solve with live kinetic-scroll offsets (virtualized lists only compile the
    /// visible window) and extra platform safe-area insets applied to every
    /// `SafeArea` node.
    pub fn solve_full(
        &mut self,
        root: &Node,
        viewport: [f32; 2],
        top_inset: f32,
        atlas: &GlyphAtlas,
        scrolls: &HashMap<String, f32>,
        extra_insets: Insets,
    ) {
        self.rects.clear();
        self.rects_by_path.clear();
        self.list_ranges.clear();
        if !root.visible {
            return;
        }
        self.scrolls.clear();
        self.scrolls.extend(scrolls.iter().map(|(k, v)| (k.clone(), *v)));
        self.extra_insets = extra_insets;
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
    fn measure_intrinsic(&mut self, node: &Node, avail_w: f32, atlas: &GlyphAtlas) -> (f32, f32) {
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
                    let (ow, _) = atlas.measure(o, 12.0, false);
                    total += ow + 20.0;
                }
                (total.max(80.0), 34.0)
            }
            NodeKind::Button { label, .. } => {
                let (lw, _) = atlas.measure(label, 12.5, false);
                (lw + 24.0, 34.0)
            }
            NodeKind::Switch { label, .. } => {
                let lw = label
                    .as_deref()
                    .filter(|l| !l.is_empty())
                    .map(|l| atlas.measure(l, 12.0, false).0 + 10.0)
                    .unwrap_or(0.0);
                ((42.0 + lw).min(inner_avail_w.max(42.0)), 28.0)
            }
            NodeKind::Input {
                value,
                placeholder,
                prefix_icon,
                clearable,
                error,
                ..
            } => {
                let txt = if value.is_empty() { placeholder } else { value };
                let (tw, _) = atlas.measure(txt, 12.5, false);
                let gutter = if prefix_icon.is_some() { 24.0 } else { 0.0 };
                let clear = if *clearable && !value.is_empty() {
                    24.0
                } else {
                    0.0
                };
                let caption = if error.is_some() { 18.0 } else { 0.0 };
                (
                    (tw + 28.0 + gutter + clear).clamp(160.0, 300.0),
                    48.0 + caption,
                )
            }
            NodeKind::Stat(data) => measure_stat(data, inner_avail_w),
            NodeKind::Plot(data) => measure_plot(data, inner_avail_w),
            NodeKind::Table(data) => measure_table(data, inner_avail_w),
            NodeKind::Feed(data) => measure_feed(data, inner_avail_w),
            NodeKind::Graph(data) => measure_graph(data, inner_avail_w),
            NodeKind::CodeText(data) => measure_code(data, inner_avail_w),
            NodeKind::RichText(data) => {
                let spans = rich_spans(data);
                let max_w = if data.wrap && inner_avail_w > 1.0 {
                    Some(inner_avail_w)
                } else {
                    None
                };
                let (w, h) = atlas.measure_rich(&spans, max_w, data.line_height);
                (if data.wrap { inner_avail_w } else { w }, h)
            }
            NodeKind::Icon { size, .. } => {
                let s = (*size).max(8.0);
                (s, s)
            }
            NodeKind::Image { .. } => {
                let w = if inner_avail_w > 1.0 { inner_avail_w } else { 64.0 };
                (w, w)
            }
            NodeKind::Stack {
                children,
                align: _,
                gap,
                ..
            } => {
                let mut max_w = 0.0f32;
                let mut max_h = 0.0f32;
                for c in children.iter().filter(|c| c.visible) {
                    let (mut cw, ch) = self.measure_intrinsic(c, inner_avail_w, atlas);
                    if let Some(inset) = c.layout.inset {
                        if inset.left.is_some() && inset.right.is_some() {
                            cw = (inner_avail_w
                                - inset.left.unwrap_or(0.0)
                                - inset.right.unwrap_or(0.0))
                            .max(0.0);
                        }
                    }
                    max_w = max_w.max(cw);
                    max_h = max_h.max(ch);
                }
                (max_w + gap * 2.0, max_h + gap * 2.0)
            }
            NodeKind::SafeArea {
                insets,
                keyboard_avoidance,
                keyboard_height,
                child,
            } => {
                let pad = self.safe_padding(*insets, *keyboard_avoidance, *keyboard_height);
                let (cw, ch) = child
                    .as_ref()
                    .map(|c| self.measure_intrinsic(c, (inner_avail_w - pad.horizontal()).max(0.0), atlas))
                    .unwrap_or((0.0, 0.0));
                (cw + pad.horizontal(), ch + pad.vertical())
            }
            NodeKind::Overlay { child, .. } => child
                .as_ref()
                .map(|c| self.measure_intrinsic(c, inner_avail_w, atlas))
                .unwrap_or((inner_avail_w, inner_avail_w * 0.6)),
            NodeKind::NavStack { routes, active, .. } => routes
                .get(*active)
                .map(|r| self.measure_intrinsic(&r.child, inner_avail_w, atlas))
                .unwrap_or((inner_avail_w, 0.0)),
            NodeKind::ListView(data) => {
                let total = self.list_total_height(&node.id, data, inner_avail_w, atlas);
                (inner_avail_w, total)
            }
            NodeKind::TextArea(data) => {
                let rows = data.rows.max(1) as f32;
                let (tw, _) = atlas.measure(&data.placeholder, 12.5, false);
                ((tw + 28.0).clamp(180.0, inner_avail_w.max(180.0)), (rows * 20.0 + 18.0).max(44.0))
            }
            NodeKind::Slider(data) => {
                let label_h = data
                    .label
                    .as_deref()
                    .filter(|l| !l.is_empty())
                    .map(|l| atlas.measure(l, 12.0, false).1 + 6.0)
                    .unwrap_or(0.0);
                (inner_avail_w.max(180.0), 28.0 + label_h)
            }
            NodeKind::RangeSlider(data) => {
                let label_h = data
                    .label
                    .as_deref()
                    .filter(|l| !l.is_empty())
                    .map(|l| atlas.measure(l, 12.0, false).1 + 6.0)
                    .unwrap_or(0.0);
                (inner_avail_w.max(180.0), 28.0 + label_h)
            }
            NodeKind::Progress(data) => {
                let label_h = data
                    .label
                    .as_deref()
                    .filter(|l| !l.is_empty())
                    .map(|_| 18.0)
                    .unwrap_or(0.0);
                match data.variant {
                    crate::ir::ProgressVariant::Linear => (inner_avail_w.max(120.0), 8.0 + label_h),
                    crate::ir::ProgressVariant::Circular => (48.0 + label_h, 48.0 + label_h),
                }
            }
            NodeKind::Pressable { child, .. } | NodeKind::Draggable { child, .. } => child
                .as_ref()
                .map(|c| self.measure_intrinsic(c, inner_avail_w, atlas))
                .unwrap_or((24.0, 24.0)),
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
        if let Some(r) = node.layout.aspect_ratio.filter(|r| *r > 0.0) {
            match (node.width, node.height) {
                (SizeSpec::Auto, SizeSpec::Auto) => {
                    if h > 0.0 {
                        w = h * r;
                    } else if w > 0.0 {
                        h = w / r;
                    }
                }
                (_, SizeSpec::Auto) => h = w / r,
                (SizeSpec::Auto, _) => w = h * r,
                _ => {}
            }
        }
        if let Some(max) = node.layout.max_width {
            w = w.min(max.max(0.0));
        }
        if let Some(max) = node.layout.max_height {
            h = h.min(max.max(0.0));
        }
        (w.max(0.0), h.max(0.0))
    }

    /// Node insets plus engine-level safe-area insets and the soft keyboard.
    fn safe_padding(&self, insets: Insets, keyboard_avoidance: bool, keyboard_height: f32) -> Insets {
        let mut p = Insets {
            top: insets.top + self.extra_insets.top,
            right: insets.right + self.extra_insets.right,
            bottom: insets.bottom + self.extra_insets.bottom,
            left: insets.left + self.extra_insets.left,
        };
        if keyboard_avoidance {
            p.bottom += keyboard_height.max(0.0);
        }
        p
    }

    /// Estimated total scroll height for a virtualized list (O(items) additions only).
    fn list_total_height(
        &self,
        _list_id: &str,
        data: &ListViewData,
        _avail_w: f32,
        _atlas: &GlyphAtlas,
    ) -> f32 {
        let spacing = data.spacing.max(0.0);
        let mut total = 0.0f32;
        for (i, item) in data.items.iter().enumerate() {
            total += item
                .height
                .unwrap_or(data.estimated_height.max(1.0))
                .max(1.0);
            if i + 1 < data.items.len() {
                total += spacing;
            }
        }
        total
    }

    /// Height for one list row: explicit, cached measurement, estimate, or a
    /// real intrinsic measurement when the row is inside the visible window.
    fn list_row_height(
        &mut self,
        list_id: &str,
        item: &crate::ir::ListItem,
        avail_w: f32,
        atlas: &GlyphAtlas,
        measure: bool,
    ) -> f32 {
        if let Some(h) = item.height {
            return h.max(1.0);
        }
        let key = format!("{list_id}::{}", item.id);
        if let Some(h) = self.item_heights.get(&key) {
            return *h;
        }
        if measure {
            let h = self
                .measure_intrinsic(&item.child, avail_w, atlas)
                .1
                .max(1.0);
            self.item_heights.insert(key, h);
            h
        } else {
            item.estimated_or(56.0)
        }
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
            NodeKind::Stack {
                align,
                gap,
                children,
                ..
            } => {
                let pad = (*gap).max(0.0);
                let sx = ix + pad;
                let sy = iy + pad;
                let sw = (iw - pad * 2.0).max(0.0);
                let sh = (ih - pad * 2.0).max(0.0);
                let mut order: Vec<usize> = children
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.visible)
                    .map(|(i, _)| i)
                    .collect();
                order.sort_by_key(|&i| children[i].layout.z);
                let mut max_bottom = iy;
                for slot in order {
                    let c = &children[slot];
                    let (mw, mh) = self.measure_intrinsic(c, sw, atlas);
                    let (mut cw, mut ch) = match c.width {
                        SizeSpec::Px(px) => (px, mh),
                        SizeSpec::Fill(_) => (sw, mh),
                        SizeSpec::Auto => (mw, mh),
                    };
                    match c.height {
                        SizeSpec::Px(px) => ch = px,
                        SizeSpec::Fill(_) => ch = sh,
                        SizeSpec::Auto => {}
                    }
                    let mut cx;
                    let mut cy;
                    if let Some(inset) = c.layout.inset {
                        match (inset.left, inset.right) {
                            (Some(l), Some(r)) => {
                                if matches!(c.width, SizeSpec::Auto) {
                                    cw = (sw - l - r).max(0.0);
                                }
                                cx = sx + l;
                            }
                            (Some(l), None) => cx = sx + l,
                            (None, Some(r)) => cx = sx + sw - cw - r,
                            (None, None) => cx = align.place([sx, sy, sw, sh], [cw, ch])[0],
                        }
                        match (inset.top, inset.bottom) {
                            (Some(t), Some(b)) => {
                                if matches!(c.height, SizeSpec::Auto) {
                                    ch = (sh - t - b).max(0.0);
                                }
                                cy = sy + t;
                            }
                            (Some(t), None) => cy = sy + t,
                            (None, Some(b)) => cy = sy + sh - ch - b,
                            (None, None) => cy = align.place([sx, sy, sw, sh], [cw, ch])[1],
                        }
                    } else {
                        let p = align.place([sx, sy, sw, sh], [cw, ch]);
                        cx = p[0];
                        cy = p[1];
                    }
                    if let Some(max) = c.layout.max_width {
                        cw = cw.min(max.max(0.0));
                    }
                    if let Some(max) = c.layout.max_height {
                        ch = ch.min(max.max(0.0));
                    }
                    cw = cw.max(c.min_width);
                    ch = ch.max(c.min_height);
                    cy = cy.min(sy + sh - ch).max(sy);
                    cx = cx.min(sx + sw - cw).max(sx);
                    let ch_hash = child_path_hash(path_hash, slot, &c.id);
                    self.place_node(c, cx, cy, cw, ch, ch_hash, atlas);
                    max_bottom = max_bottom.max(cy + ch);
                }
                content_h = (max_bottom - y) + node.padding.bottom;
            }
            NodeKind::SafeArea {
                insets,
                keyboard_avoidance,
                keyboard_height,
                child,
            } => {
                let pad = self.safe_padding(*insets, *keyboard_avoidance, *keyboard_height);
                let cx = ix + pad.left;
                let cy = iy + pad.top;
                let cw = (iw - pad.horizontal()).max(0.0);
                let chh = (ih - pad.vertical()).max(0.0);
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    let (_, measured) = self.measure_intrinsic(c, cw, atlas);
                    self.place_node(c, cx, cy, cw, chh, ch_hash, atlas);
                    content_h = pad.vertical() + measured;
                } else {
                    content_h = pad.vertical() + chh;
                }
            }
            NodeKind::Overlay {
                align, child, ..
            } => {
                if let Some(c) = child {
                    let (mw, mh) = self.measure_intrinsic(c, iw, atlas);
                    let cw = mw.min(iw).max(c.min_width);
                    let chh = mh.min(ih).max(c.min_height);
                    let p = align.place([ix, iy, iw, ih], [cw, chh]);
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.place_node(c, p[0], p[1], cw, chh, ch_hash, atlas);
                    content_h = chh + node.padding.vertical();
                }
            }
            NodeKind::NavStack {
                routes,
                active,
                progress,
                ..
            } => {
                let p = (*progress).clamp(0.0, 1.0);
                if p > 0.0 && *active > 0 {
                    if let Some(prev) = routes.get(*active - 1) {
                        let h = child_path_hash(path_hash, *active - 1, &prev.child.id);
                        self.place_node(&prev.child, ix - p * iw, iy, iw, ih, h, atlas);
                    }
                }
                if let Some(route) = routes.get(*active) {
                    let h = child_path_hash(path_hash, *active, &route.child.id);
                    self.place_node(&route.child, ix + (1.0 - p) * iw, iy, iw, ih, h, atlas);
                }
                content_h = ih;
            }
            NodeKind::ListView(data) => {
                let total = self.list_total_height(&node.id, data, iw, atlas);
                let scroll = self
                    .scrolls
                    .get(&node.id)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, (total - ih).max(0.0));
                let overscan = data.overscan.max(0.0) * ih;
                let spacing = data.spacing.max(0.0);
                let mut cy = iy - scroll;
                let window_top = iy - overscan;
                let window_bottom = iy + ih + overscan;
                let mut first_visible = usize::MAX;
                let mut last_visible = 0usize;
                for (slot, item) in data.items.iter().enumerate() {
                    let est = item.estimated_or(data.estimated_height);
                    let visible = cy + est >= window_top && cy <= window_bottom;
                    let h = self.list_row_height(&node.id, item, iw, atlas, visible);
                    if visible && cy + h > window_top && cy < window_bottom {
                        let h_hash = child_path_hash(path_hash, slot, &item.child.id);
                        self.place_node(&item.child, ix, cy, iw, h, h_hash, atlas);
                        first_visible = first_visible.min(slot);
                        last_visible = last_visible.max(slot + 1);
                    }
                    cy += h + spacing;
                }
                if first_visible != usize::MAX {
                    self.list_ranges
                        .insert(node.id.clone(), (first_visible, last_visible));
                }
                content_h = total + node.padding.vertical();
            }
            NodeKind::Pressable { child, .. } => {
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.place_node(c, ix, iy, iw, ih, ch_hash, atlas);
                }
                content_h = ih;
            }
            NodeKind::Draggable {
                offset, child, ..
            } => {
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.place_node(c, ix + offset[0], iy + offset[1], iw, ih, ch_hash, atlas);
                }
                content_h = ih;
            }
            NodeKind::RichText { .. }
            | NodeKind::Icon { .. }
            | NodeKind::Image { .. }
            | NodeKind::TextArea(_)
            | NodeKind::Slider(_)
            | NodeKind::RangeSlider(_)
            | NodeKind::Progress(_)
            | NodeKind::Text { .. }
            | NodeKind::Rule { .. }
            | NodeKind::Spacer
            | NodeKind::ThemeToggle { .. }
            | NodeKind::Segmented { .. }
            | NodeKind::Button { .. }
            | NodeKind::Switch { .. }
            | NodeKind::Input { .. }
            | NodeKind::Stat(_)
            | NodeKind::Plot(_)
            | NodeKind::Table(_)
            | NodeKind::Feed(_)
            | NodeKind::Graph(_)
            | NodeKind::CodeText(_) => {
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

/// Build styled runs for rich text layout/drawing.
pub fn rich_spans(data: &RichTextData) -> Vec<RichSpan<'_>> {
    data.spans
        .iter()
        .map(|span| RichSpan {
            text: &span.text,
            style: TextStyle::new(
                span.size.unwrap_or(data.size),
                if span.mono {
                    FontFamily::Mono
                } else {
                    FontFamily::Sans
                },
                span.weight,
            ),
            color: span.color.unwrap_or(0),
        })
        .collect()
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
    use crate::ir::{Insets, NodeKind, StackAlign};

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
    fn stack_centers_children_and_honors_insets_and_aspect_ratio() {
        let atlas = GlyphAtlas::builtin();
        let root = Node::new(
            "stack",
            NodeKind::Stack {
                align: StackAlign::Center,
                gap: 0.0,
                style: None,
                children: vec![
                    Node::new("bg", NodeKind::Spacer)
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
                    Node::new("badge", NodeKind::Spacer)
                        .with_size(SizeSpec::Px(20.0), SizeSpec::Px(20.0)),
                    Node::new("pinned", NodeKind::Spacer)
                        .with_size(SizeSpec::Px(20.0), SizeSpec::Px(20.0))
                        .with_inset(crate::ir::InsetHint {
                            top: Some(4.0),
                            right: Some(8.0),
                            ..Default::default()
                        }),
                ],
            },
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));
        let mut map = LayoutMap::default();
        map.solve(&root, [200.0, 100.0], &atlas);
        let badge = map.get("badge").unwrap();
        assert_eq!((badge.x, badge.y), (90.0, 40.0));
        let pinned = map.get("pinned").unwrap();
        assert_eq!((pinned.x, pinned.y), (172.0, 4.0));

        let ratio = Node::new(
            "ratio_root",
            NodeKind::Stack {
                align: StackAlign::TopLeft,
                gap: 0.0,
                style: None,
                children: vec![Node::new("ratio", NodeKind::Spacer)
                    .with_size(SizeSpec::Px(80.0), SizeSpec::Auto)
                    .with_aspect_ratio(2.0)],
            },
        );
        let mut map2 = LayoutMap::default();
        map2.solve(&ratio, [400.0, 300.0], &atlas);
        let r = map2.get("ratio").unwrap();
        assert_eq!((r.w, r.h), (80.0, 40.0));
    }

    #[test]
    fn safe_area_applies_insets_and_keyboard_padding() {
        let atlas = GlyphAtlas::builtin();
        let root = Node::new(
            "safe",
            NodeKind::SafeArea {
                insets: Insets {
                    top: 10.0,
                    right: 7.0,
                    bottom: 3.0,
                    left: 5.0,
                },
                keyboard_avoidance: true,
                keyboard_height: 120.0,
                child: Some(Box::new(
                    Node::new("body", NodeKind::Spacer)
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
                )),
            },
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));
        let mut map = LayoutMap::default();
        map.solve(&root, [200.0, 300.0], &atlas);
        let body = map.get("body").unwrap();
        assert_eq!((body.x, body.y), (5.0, 10.0));
        assert_eq!((body.w, body.h), (188.0, 167.0));
    }

    #[test]
    fn virtualized_list_places_only_the_visible_window() {
        let atlas = GlyphAtlas::builtin();
        let items: Vec<crate::ir::ListItem> = (0..1000)
            .map(|i| crate::ir::ListItem {
                id: format!("row{i}"),
                height: Some(20.0),
                child: Node::new(format!("item{i}"), NodeKind::Spacer),
            })
            .collect();
        let root = Node::new(
            "list",
            NodeKind::ListView(ListViewData {
                items,
                estimated_height: 20.0,
                spacing: 0.0,
                overscan: 1.0,
                style: None,
            }),
        )
        .with_size(SizeSpec::Px(200.0), SizeSpec::Px(200.0));
        let mut map = LayoutMap::default();
        map.solve(&root, [200.0, 200.0], &atlas);
        let (start, end) = map.list_range("list").expect("window tracked");
        assert_eq!(start, 0);
        // 200px viewport + 200px overscan / 20px rows.
        assert_eq!(end, 20);
        assert!(map.get("item0").is_some());
        assert!(map.get("item19").is_some());
        assert!(map.get("item900").is_none());

        // Scrolling moves the window without placing every row.
        let mut scrolls = std::collections::HashMap::new();
        scrolls.insert("list".to_string(), 5000.0);
        map.solve_full(&root, [200.0, 200.0], 0.0, &atlas, &scrolls, Insets::ZERO);
        let (start2, end2) = map.list_range("list").unwrap();
        assert_eq!(start2, 240);
        // Fully overscanned: one viewport above + viewport + one below = 30 rows.
        assert_eq!(end2 - start2, 30);
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
