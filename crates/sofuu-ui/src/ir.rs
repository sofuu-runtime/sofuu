//! Domain-agnostic Scene IR (retained node tree + incremental patches).
//!
//! Layout containers (`Container`, `Grid`, `Tabs`), base structural surfaces (`Surface`, `Text`,
//! `Rule`, `Spacer`), and patch commands sit here without any baked-in domain schemas.

use serde::{Deserialize, Serialize};

use crate::components::code::{CodeData, CodeHighlight};
use crate::components::feed::{FeedData, FeedItem};
use crate::components::graph::{GraphData, GraphEdge, GraphNode};
use crate::components::plot::{
    CandlePoint, PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind, PlotThreshold,
};
use crate::components::stat::StatData;
use crate::components::table::{reapply_table_sort, sort_table_rows, TableData, TableRow};
use crate::theme::{ThemeMode, Tone};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    #[default]
    Column,
    Row,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Align {
    #[default]
    Stretch,
    Start,
    Center,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizeSpec {
    #[default]
    Auto,
    Px(f32),
    /// Grow share inside a flex container (e.g., `1.0`).
    Fill(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Insets {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Insets {
    pub const ZERO: Self = Self {
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    };

    pub const fn all(v: f32) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }

    pub const fn xy(x: f32, y: f32) -> Self {
        Self {
            top: y,
            right: x,
            bottom: y,
            left: x,
        }
    }

    #[inline]
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    #[inline]
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Visual surface styling options for base boxes/panels.
///
/// Borders and drop shadows are **disabled by default** (`border = None`, `shadow_blur = 0.0`)
/// and only rendered when explicitly configured. `bg = None` resolves dynamically from the active `Theme`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BoxStyle {
    pub bg: Option<u32>,
    pub bg_bottom: Option<u32>,
    pub hover_bg: Option<u32>,
    pub border: Option<u32>,
    pub border_width: f32,
    pub radius: [f32; 4],
    pub shadow_color: Option<u32>,
    pub shadow_blur: f32,
    pub shadow_offset_y: f32,
    pub opacity: f32,
}

impl Default for BoxStyle {
    fn default() -> Self {
        Self {
            bg: None,
            bg_bottom: None,
            hover_bg: None,
            border: None,
            border_width: 0.0,
            radius: [0.0; 4],
            shadow_color: None,
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
            opacity: 1.0,
        }
    }
}

impl BoxStyle {
    /// Theme-adaptive elevated card surface with NO border and NO shadow by default.
    pub fn card(radius: f32) -> Self {
        Self {
            radius: [radius; 4],
            ..Self::default()
        }
    }

    pub fn flat(bg: u32) -> Self {
        Self {
            bg: Some(bg),
            ..Self::default()
        }
    }

    pub fn with_bg(mut self, bg: u32) -> Self {
        self.bg = Some(bg);
        self
    }

    pub fn with_border(mut self, color: Option<u32>, width: f32) -> Self {
        self.border = color;
        self.border_width = width.max(0.0);
        self
    }

    pub fn with_shadow(mut self, blur: f32, offset_y: f32, color: Option<u32>) -> Self {
        self.shadow_blur = blur.max(0.0);
        self.shadow_offset_y = offset_y;
        self.shadow_color = color;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TabItem {
    pub label: String,
    pub child: Node,
}

const fn default_toggle_size() -> f32 {
    36.0
}

const fn default_node_priority() -> u8 {
    100
}

const fn default_node_visible() -> bool {
    true
}

/// Core retained Scene IR node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(default = "default_node_priority")]
    pub priority: u8,
    #[serde(default = "default_node_visible")]
    pub visible: bool,
    #[serde(default)]
    pub width: SizeSpec,
    #[serde(default)]
    pub height: SizeSpec,
    #[serde(default)]
    pub min_width: f32,
    #[serde(default)]
    pub min_height: f32,
    #[serde(default)]
    pub padding: Insets,
    pub kind: NodeKind,
}

impl Node {
    pub fn new(id: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id: id.into(),
            priority: 100,
            visible: true,
            width: SizeSpec::Auto,
            height: SizeSpec::Auto,
            min_width: 0.0,
            min_height: 0.0,
            padding: Insets::ZERO,
            kind,
        }
    }

    pub fn with_size(mut self, width: SizeSpec, height: SizeSpec) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_padding(mut self, padding: Insets) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Recursively visit all nodes mutably to apply targeted incremental patches.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut Node> {
        if self.id == id {
            return Some(self);
        }
        match &mut self.kind {
            NodeKind::Container { children, .. } | NodeKind::Grid { children, .. } => {
                for c in children {
                    if let Some(found) = c.find_mut(id) {
                        return Some(found);
                    }
                }
            }
            NodeKind::Tabs { items, .. } => {
                for item in items {
                    if let Some(found) = item.child.find_mut(id) {
                        return Some(found);
                    }
                }
            }
            NodeKind::Surface { child, .. } => {
                if let Some(c) = child {
                    return c.find_mut(id);
                }
            }
            NodeKind::Text { .. }
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
            | NodeKind::CodeText(_) => {}
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeKind {
    /// Flex container supporting automatic responsive folding (`Row` -> `Column` below `fold_width`).
    Container {
        axis: Axis,
        align: Align,
        gap: f32,
        /// Optional viewport/parent width threshold under which `Axis::Row` folds into `Axis::Column`.
        fold_width: Option<f32>,
        scrollable: bool,
        style: Option<BoxStyle>,
        children: Vec<Node>,
    },
    /// Responsive multi-column grid. Computes column count automatically from `min_col_width` when `columns == 0`.
    Grid {
        columns: usize,
        min_col_width: f32,
        gap: f32,
        style: Option<BoxStyle>,
        children: Vec<Node>,
    },
    /// Interactive segmented tab view with animated active pill indicator.
    Tabs {
        active: usize,
        items: Vec<TabItem>,
    },
    /// Elevated or interactive card/panel box (supports hover/click springs when `interactive = true`).
    Surface {
        style: BoxStyle,
        interactive: bool,
        child: Option<Box<Node>>,
    },
    /// SDF typography primitive.
    Text {
        content: String,
        size: f32,
        color: Option<u32>,
        mono: bool,
        wrap: bool,
    },
    /// Hairline horizontal separator.
    Rule {
        thickness: f32,
        color: Option<u32>,
    },
    /// Flexible empty spacer.
    Spacer,
    /// Borderless, shadowless single icon showing Sun in Light mode and Crescent Moon in Dark mode.
    ThemeToggle {
        #[serde(default = "default_toggle_size")]
        size: f32,
    },
    /// Animated pill option group (e.g. timeframes, view variants).
    Segmented {
        active: usize,
        options: Vec<String>,
    },
    /// Borderless, shadowless pill action button.
    Button {
        label: String,
        #[serde(default)]
        tone: Tone,
    },
    /// Borderless animated pill toggle switch with optional label.
    Switch {
        checked: bool,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        tone: Tone,
    },
    /// Interactive single-line monospace text input field.
    Input {
        value: String,
        #[serde(default)]
        placeholder: String,
        #[serde(default)]
        tone: Tone,
    },
    /// Adaptive KPI metric tile with counter spring, delta pill, sparkline/range bar.
    Stat(StatData),
    /// Multi-series / multi-pane GPU chart (Line, Area, StackedArea, Band, Candles, Histogram).
    Plot(PlotData),
    /// Virtualized GPU data table with sortable columns and row selection.
    Table(TableData),
    /// Chronological event/alert stream with arrival highlights and kinetic scroll.
    Feed(FeedData),
    /// Topological node-edge DAG visualizer with auto-layout and active Bezier edges.
    Graph(GraphData),
    /// Monospace syntax-tinted code/log viewer with line gutter and highlights.
    CodeText(CodeData),
}

/// Complete scene root held by `UiEngine`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub title: String,
    pub theme: ThemeMode,
    pub root: Node,
}

impl Scene {
    pub fn new(title: impl Into<String>, theme: ThemeMode, root: Node) -> Self {
        Self {
            title: title.into(),
            theme,
            root,
        }
    }

    pub fn apply_patch(&mut self, patch: ScenePatch) -> bool {
        match patch {
            ScenePatch::ReplaceScene(scene) => {
                *self = scene;
                true
            }
            ScenePatch::SetTheme { mode } => {
                if self.theme != mode {
                    self.theme = mode;
                    true
                } else {
                    false
                }
            }
            ScenePatch::SetText { id, content } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Text { content: old, .. } = &mut node.kind {
                        if *old != content {
                            *old = content;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SetVisible { id, visible } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if node.visible != visible {
                        node.visible = visible;
                        return true;
                    }
                }
                false
            }
            ScenePatch::SetTab { id, active } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Tabs {
                        active: cur,
                        items,
                    } = &mut node.kind
                    {
                        if !items.is_empty() {
                            let clamped = active.min(items.len() - 1);
                            if *cur != clamped {
                                *cur = clamped;
                                return true;
                            }
                        }
                    }
                }
                false
            }
            ScenePatch::ReplaceNode { id, node } => {
                if let Some(target) = self.root.find_mut(&id) {
                    *target = node;
                    true
                } else {
                    false
                }
            }
            ScenePatch::UpdateStat {
                id,
                value,
                value_text,
                delta,
                tone,
                caption,
                push_sparkline,
                sparkline_max_len,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Stat(stat) = &mut node.kind {
                        if let Some(v) = value {
                            stat.value = v;
                        }
                        if let Some(vt) = value_text {
                            stat.value_text = vt;
                        }
                        if let Some(d) = delta {
                            stat.delta = d;
                        }
                        if let Some(t) = tone {
                            stat.tone = t;
                        }
                        if let Some(c) = caption {
                            stat.caption = c;
                        }
                        if let Some(sp) = push_sparkline {
                            stat.sparkline.push(sp);
                            let cap = sparkline_max_len.unwrap_or(64).max(4);
                            if stat.sparkline.len() > cap {
                                let excess = stat.sparkline.len() - cap;
                                stat.sparkline.drain(0..excess);
                            }
                        }
                        stat.update_seq = stat.update_seq.wrapping_add(1);
                        return true;
                    }
                }
                false
            }
            ScenePatch::AppendPlotPoint {
                id,
                series_id,
                point,
                max_points,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        let cap = max_points.unwrap_or(2000).max(8);
                        if let Some(s) = find_plot_series_mut(plot, &series_id) {
                            s.points.push(point);
                            if s.points.len() > cap {
                                let excess = s.points.len() - cap;
                                s.points.drain(0..excess);
                            }
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::UpsertCandle {
                id,
                series_id,
                candle,
                max_candles,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        let cap = max_candles.unwrap_or(500).max(8);
                        if let Some(s) = find_plot_series_mut(plot, &series_id) {
                            if let Some(last) = s.candles.last_mut() {
                                if (last.x - candle.x).abs() < 1e-6 {
                                    *last = candle;
                                    return true;
                                }
                            }
                            s.candles.push(candle);
                            if s.candles.len() > cap {
                                let excess = s.candles.len() - cap;
                                s.candles.drain(0..excess);
                            }
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::TogglePlotSeries { id, series_index } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        let target = if let Some(first_pane) = plot.panes.first_mut() {
                            first_pane.series.get_mut(series_index)
                        } else {
                            plot.series.get_mut(series_index)
                        };
                        if let Some(s) = target {
                            s.visible = !s.visible;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SetSegmented { id, active } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Segmented { active: cur, options } = &mut node.kind {
                        if !options.is_empty() {
                            let clamped = active.min(options.len() - 1);
                            if *cur != clamped {
                                *cur = clamped;
                                return true;
                            }
                        }
                    }
                }
                false
            }
            ScenePatch::ZoomPlot {
                id,
                delta_y,
                anchor_frac,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        let [f0, f1] = plot.view_range.unwrap_or([0.0, 1.0]);
                        let span = (f1 - f0).clamp(0.05, 1.0);
                        let zoom = (1.0 - (delta_y as f64) * 0.002).clamp(0.65, 1.45);
                        let new_span = (span * zoom).clamp(0.05, 1.0);
                        let anchor = anchor_frac.clamp(0.0, 1.0) as f64;
                        let center = f0 + span * anchor;
                        let nf0 = (center - new_span * anchor).clamp(0.0, 1.0 - new_span);
                        if new_span >= 0.995 {
                            plot.view_range = None;
                        } else {
                            plot.view_range = Some([nf0, nf0 + new_span]);
                        }
                        return true;
                    }
                }
                false
            }
            ScenePatch::PanPlot { id, delta_frac } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        if let Some([f0, f1]) = plot.view_range {
                            let span = f1 - f0;
                            let nf0 = (f0 + delta_frac as f64 * span).clamp(0.0, 1.0 - span);
                            plot.view_range = Some([nf0, nf0 + span]);
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::ResetPlotView { id } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        if plot.view_range.take().is_some() {
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SetSwitch { id, checked } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Switch { checked: cur, .. } = &mut node.kind {
                        if *cur != checked {
                            *cur = checked;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SetInput { id, value } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Input { value: cur, .. } = &mut node.kind {
                        if *cur != value {
                            *cur = value;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SortTable { id, col_index } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Table(table) = &mut node.kind {
                        if col_index < table.columns.len() {
                            sort_table_rows(table, col_index);
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SelectTableRow { id, row_id } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Table(table) = &mut node.kind {
                        if table.selected_row != row_id {
                            table.selected_row = row_id;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::AppendFeedItem {
                id,
                item,
                max_items,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Feed(feed) = &mut node.kind {
                        let cap = max_items.unwrap_or(feed.max_items).max(1);
                        if feed.newest_first {
                            feed.items.insert(0, item);
                            if feed.items.len() > cap {
                                feed.items.truncate(cap);
                            }
                        } else {
                            feed.items.push(item);
                            if feed.items.len() > cap {
                                let excess = feed.items.len() - cap;
                                feed.items.drain(0..excess);
                            }
                        }
                        feed.update_seq = feed.update_seq.wrapping_add(1);
                        return true;
                    }
                }
                false
            }
            ScenePatch::SelectGraphNode { id, node_id } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Graph(graph) = &mut node.kind {
                        if graph.selected_node != node_id {
                            graph.selected_node = node_id;
                            return true;
                        }
                    }
                }
                false
            }
            ScenePatch::SetGraphNodeTone { id, node_id, tone } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Graph(graph) = &mut node.kind {
                        if let Some(gn) = graph.nodes.iter_mut().find(|n| n.id == node_id) {
                            if gn.tone != tone {
                                gn.tone = tone;
                                return true;
                            }
                        }
                    }
                }
                false
            }
            ScenePatch::SetGraphEdgeActive {
                id,
                from,
                to,
                active,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Graph(graph) = &mut node.kind {
                        let mut changed = false;
                        for edge in &mut graph.edges {
                            if edge.from == from && edge.to == to && edge.active != active {
                                edge.active = active;
                                changed = true;
                            }
                        }
                        return changed;
                    }
                }
                false
            }
            ScenePatch::SetCode {
                id,
                content,
                highlights,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::CodeText(code) = &mut node.kind {
                        code.content = content;
                        if let Some(hl) = highlights {
                            code.highlights = hl;
                        }
                        return true;
                    }
                }
                false
            }
            ScenePatch::SetTableRows {
                id,
                rows,
                selected_row,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Table(table) = &mut node.kind {
                        let mut changed = table.rows != rows;
                        if changed {
                            table.rows = rows;
                            // Preserve the user's sort selection: re-order the
                            // new rows by the current column/direction instead
                            // of resetting `sort_col`/`sort_dir`.
                            reapply_table_sort(table);
                        }
                        if let Some(sel) = selected_row {
                            if table.selected_row != sel {
                                table.selected_row = sel;
                                changed = true;
                            }
                        }
                        return changed;
                    }
                }
                false
            }
            ScenePatch::SetPlotData {
                id,
                title,
                subtitle,
                series,
                panes,
                thresholds,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Plot(plot) = &mut node.kind {
                        let old_shape = plot_shape(plot);
                        if let Some(t) = title {
                            plot.title = t;
                        }
                        if let Some(st) = subtitle {
                            plot.subtitle = st;
                        }
                        if let Some(s) = series {
                            plot.series = s;
                        }
                        if let Some(p) = panes {
                            plot.panes = p;
                        }
                        if let Some(th) = thresholds {
                            plot.thresholds = th;
                        }
                        // The node's bounds are untouched, so a compatible
                        // data swap keeps the user's zoom window. Only a
                        // structural change (series/pane count or candle vs
                        // point layout) clears it.
                        if old_shape != plot_shape(plot) {
                            plot.view_range = None;
                        }
                        return true;
                    }
                }
                false
            }
            ScenePatch::SetGraph {
                id,
                title,
                nodes,
                edges,
                selected_node,
            } => {
                if let Some(node) = self.root.find_mut(&id) {
                    if let NodeKind::Graph(graph) = &mut node.kind {
                        let mut changed = false;
                        if let Some(t) = title {
                            if graph.title != t {
                                graph.title = t;
                                changed = true;
                            }
                        }
                        if graph.nodes != nodes {
                            graph.nodes = nodes;
                            changed = true;
                        }
                        if graph.edges != edges {
                            graph.edges = edges;
                            changed = true;
                        }
                        if let Some(sel) = selected_node {
                            if graph.selected_node != sel {
                                graph.selected_node = sel;
                                changed = true;
                            }
                        }
                        return changed;
                    }
                }
                false
            }
            ScenePatch::SetChildren { id, children } => {
                if let Some(node) = self.root.find_mut(&id) {
                    match &mut node.kind {
                        NodeKind::Container {
                            children: current, ..
                        }
                        | NodeKind::Grid {
                            children: current, ..
                        } => {
                            if *current != children {
                                *current = children;
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
                false
            }
        }
    }
}

/// Structural fingerprint used by [`ScenePatch::SetPlotData`] to decide whether
/// the existing zoom window is still valid (`PlotSeriesKind`, series count,
/// pane count, and whether each series carries points or candles).
fn plot_shape(plot: &PlotData) -> (usize, usize, Vec<(PlotSeriesKind, bool, bool)>) {
    fn push(out: &mut Vec<(PlotSeriesKind, bool, bool)>, series: &[PlotSeries]) {
        for s in series {
            out.push((s.kind, !s.points.is_empty(), !s.candles.is_empty()));
        }
    }
    let mut out = Vec::new();
    push(&mut out, &plot.series);
    for pane in &plot.panes {
        push(&mut out, &pane.series);
    }
    (out.len(), plot.panes.len(), out)
}

fn find_plot_series_mut<'a>(
    plot: &'a mut PlotData,
    series_id: &str,
) -> Option<&'a mut crate::components::plot::PlotSeries> {
    if let Some(s) = plot.series.iter_mut().find(|s| s.id == series_id) {
        return Some(s);
    }
    for pane in &mut plot.panes {
        if let Some(s) = pane.series.iter_mut().find(|s| s.id == series_id) {
            return Some(s);
        }
    }
    None
}

/// Partial state mutations dispatched from background threads / JS runtime without rebuilding unchanged trees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ScenePatch {
    ReplaceScene(Scene),
    SetTheme { mode: ThemeMode },
    SetText { id: String, content: String },
    SetVisible { id: String, visible: bool },
    SetTab { id: String, active: usize },
    SetSegmented { id: String, active: usize },
    SetSwitch { id: String, checked: bool },
    SetInput { id: String, value: String },
    SortTable { id: String, col_index: usize },
    SelectTableRow { id: String, row_id: Option<String> },
    AppendFeedItem {
        id: String,
        item: FeedItem,
        #[serde(default)]
        max_items: Option<usize>,
    },
    SelectGraphNode { id: String, node_id: Option<String> },
    SetGraphNodeTone { id: String, node_id: String, tone: Tone },
    SetGraphEdgeActive { id: String, from: String, to: String, active: bool },
    SetCode {
        id: String,
        content: String,
        #[serde(default)]
        highlights: Option<Vec<CodeHighlight>>,
    },
    ReplaceNode { id: String, node: Node },
    /// Replace a table's rows while preserving `sort_col`/`sort_dir` and the
    /// engine's kinetic scroll state. `selected_row` omitted = keep selection.
    SetTableRows {
        id: String,
        rows: Vec<TableRow>,
        #[serde(default)]
        selected_row: Option<Option<String>>,
    },
    /// Swap a plot's data while keeping the outer node bounds and (when the
    /// series/pane shape is compatible) the user's zoom window.
    SetPlotData {
        id: String,
        #[serde(default)]
        title: Option<Option<String>>,
        #[serde(default)]
        subtitle: Option<Option<String>>,
        #[serde(default)]
        series: Option<Vec<PlotSeries>>,
        #[serde(default)]
        panes: Option<Vec<PlotPane>>,
        #[serde(default)]
        thresholds: Option<Vec<PlotThreshold>>,
    },
    /// Replace a graph's nodes/edges while preserving its style and bounds.
    SetGraph {
        id: String,
        #[serde(default)]
        title: Option<Option<String>>,
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        #[serde(default)]
        selected_node: Option<Option<String>>,
    },
    /// Replace the children of a container or grid, keeping its id, gap,
    /// alignment, fold width, padding and style untouched.
    SetChildren { id: String, children: Vec<Node> },
    UpdateStat {
        id: String,
        #[serde(default)]
        value: Option<f64>,
        #[serde(default)]
        value_text: Option<Option<String>>,
        #[serde(default)]
        delta: Option<Option<String>>,
        #[serde(default)]
        tone: Option<Tone>,
        #[serde(default)]
        caption: Option<Option<String>>,
        #[serde(default)]
        push_sparkline: Option<f32>,
        #[serde(default)]
        sparkline_max_len: Option<usize>,
    },
    AppendPlotPoint {
        id: String,
        series_id: String,
        point: PlotPoint,
        #[serde(default)]
        max_points: Option<usize>,
    },
    UpsertCandle {
        id: String,
        series_id: String,
        candle: CandlePoint,
        #[serde(default)]
        max_candles: Option<usize>,
    },
    TogglePlotSeries {
        id: String,
        series_index: usize,
    },
    ZoomPlot {
        id: String,
        delta_y: f32,
        anchor_frac: f32,
    },
    PanPlot {
        id: String,
        delta_frac: f32,
    },
    ResetPlotView {
        id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_node_json_deserializes_with_defaults() {
        let json = r#"{
            "title": "Boot",
            "theme": "light",
            "root": {
                "id": "root",
                "kind": {
                    "type": "container",
                    "axis": "column",
                    "align": "stretch",
                    "gap": 10.0,
                    "fold_width": null,
                    "scrollable": false,
                    "style": null,
                    "children": [
                        {
                            "id": "lbl",
                            "kind": {
                                "type": "text",
                                "content": "Hello",
                                "size": 14.0,
                                "color": null,
                                "mono": false,
                                "wrap": false
                            }
                        }
                    ]
                }
            }
        }"#;
        let scene: Scene = serde_json::from_str(json).expect("deserialize minimal scene");
        assert_eq!(scene.root.priority, 100);
        assert!(scene.root.visible);
        assert_eq!(scene.root.width, SizeSpec::Auto);
    }

    fn container(children: Vec<Node>) -> Node {
        Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children,
            },
        )
    }

    fn row_text(id: &str, text: &str) -> TableRow {
        TableRow::new(id, vec![crate::components::table::TableCell::new(text)])
    }

    #[test]
    fn incremental_table_rows_reapply_existing_sort_without_resetting_it() {
        use crate::components::table::{TableColumn, TableSortDir};

        let table = Node::new(
            "tbl",
            NodeKind::Table(TableData {
                columns: vec![TableColumn::new("n", "N")],
                rows: vec![row_text("a", "2"), row_text("b", "10")],
                sort_col: Some(0),
                sort_dir: TableSortDir::Desc,
                ..TableData::default()
            }),
        );
        let mut scene = Scene::new("T", ThemeMode::Light, container(vec![table]));

        let patch: ScenePatch = serde_json::from_str(
            r#"{"op":"set_table_rows","id":"tbl","rows":[{"id":"c","cells":[{"text":"5"}]},{"id":"d","cells":[{"text":"1"}]},{"id":"e","cells":[{"text":"9"}]}]}"#,
        )
        .expect("set_table_rows json");
        assert!(scene.apply_patch(patch));

        let NodeKind::Table(table) = &scene.root.find_mut("tbl").unwrap().kind else {
            panic!("table missing");
        };
        assert_eq!(table.sort_col, Some(0));
        assert_eq!(table.sort_dir, TableSortDir::Desc);
        let values: Vec<&str> = table.rows.iter().map(|r| r.cells[0].text.as_str()).collect();
        assert_eq!(values, vec!["9", "5", "1"]);

        // Selected-row is only touched when the patch asks for it.
        assert_eq!(table.selected_row, None);
        let select: ScenePatch =
            serde_json::from_str(r#"{"op":"set_table_rows","id":"tbl","rows":[],"selected_row":"d"}"#)
                .expect("selected_row json");
        assert!(scene.apply_patch(select));
        let NodeKind::Table(table) = &scene.root.find_mut("tbl").unwrap().kind else {
            panic!("table missing");
        };
        assert_eq!(table.selected_row.as_deref(), Some("d"));
    }

    fn line_series(id: &str) -> PlotSeries {
        PlotSeries {
            id: id.to_string(),
            name: id.to_string(),
            kind: PlotSeriesKind::Line,
            color: None,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: None,
            points: vec![PlotPoint {
                x: 0.0,
                y: 1.0,
                y2: None,
                label: None,
            }],
            candles: Vec::new(),
        }
    }

    #[test]
    fn incremental_plot_data_keeps_compatible_zoom_and_resets_on_shape_change() {
        let plot = Node::new(
            "p1",
            NodeKind::Plot(PlotData {
                series: vec![line_series("price")],
                view_range: Some([0.2, 0.8]),
                ..PlotData::default()
            }),
        );
        let mut scene = Scene::new("T", ThemeMode::Light, container(vec![plot]));

        let compatible: ScenePatch = serde_json::from_str(
            r#"{"op":"set_plot_data","id":"p1","title":"Updated","series":[{"id":"price","name":"Price","kind":"line","points":[{"x":0,"y":2}]}]}"#,
        )
        .expect("set_plot_data json");
        assert!(scene.apply_patch(compatible));
        let NodeKind::Plot(plot) = &scene.root.find_mut("p1").unwrap().kind else {
            panic!("plot missing");
        };
        assert_eq!(plot.title.as_deref(), Some("Updated"));
        assert_eq!(plot.view_range, Some([0.2, 0.8]));

        let incompatible: ScenePatch = serde_json::from_str(
            r#"{"op":"set_plot_data","id":"p1","series":[{"id":"price","name":"Price","kind":"line","points":[{"x":0,"y":2}]},{"id":"vol","name":"Volume","kind":"histogram","points":[{"x":0,"y":3}]}]}"#,
        )
        .expect("set_plot_data shape json");
        assert!(scene.apply_patch(incompatible));
        let NodeKind::Plot(plot) = &scene.root.find_mut("p1").unwrap().kind else {
            panic!("plot missing");
        };
        assert_eq!(plot.view_range, None);
        assert_eq!(plot.series.len(), 2);
    }

    #[test]
    fn incremental_set_children_preserves_grid_geometry() {
        let grid = Node::new(
            "g",
            NodeKind::Grid {
                columns: 3,
                min_col_width: 150.0,
                gap: 7.0,
                style: None,
                children: vec![Node::new("old", NodeKind::Spacer)],
            },
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(220.0));
        let mut scene = Scene::new("T", ThemeMode::Light, container(vec![grid]));

        let patch: ScenePatch = serde_json::from_str(
            r#"{"op":"set_children","id":"g","children":[{"id":"n1","kind":{"type":"spacer"}}]}"#,
        )
        .expect("set_children json");
        assert!(scene.apply_patch(patch));
        let NodeKind::Grid { gap, columns, min_col_width, children, .. } =
            &scene.root.find_mut("g").unwrap().kind
        else {
            panic!("grid missing");
        };
        assert_eq!(*gap, 7.0);
        assert_eq!(*columns, 3);
        assert_eq!(*min_col_width, 150.0);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "n1");
    }

    #[test]
    fn incremental_graph_swap_keeps_style_and_title_control() {
        let graph = Node::new(
            "dag",
            NodeKind::Graph(GraphData {
                title: Some("Old".into()),
                nodes: vec![GraphNode::new("a", "A")],
                edges: Vec::new(),
                ..GraphData::default()
            }),
        );
        let mut scene = Scene::new("T", ThemeMode::Light, container(vec![graph]));
        let patch: ScenePatch = serde_json::from_str(
            r#"{"op":"set_graph","id":"dag","title":null,"nodes":[{"id":"b","label":"B"}],"edges":[{"from":"b","to":"a"}]}"#,
        )
        .expect("set_graph json");
        assert!(scene.apply_patch(patch));
        let NodeKind::Graph(graph) = &scene.root.find_mut("dag").unwrap().kind else {
            panic!("graph missing");
        };
        // `title: null` deserializes to "leave unchanged" under Option<Option<_>>.
        assert_eq!(graph.title.as_deref(), Some("Old"));
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].id, "b");
        assert_eq!(graph.edges.len(), 1);
    }
}
