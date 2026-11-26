//! Domain-agnostic Scene IR (retained node tree + incremental patches).
//!
//! Layout containers (`Container`, `Grid`, `Tabs`), base structural surfaces (`Surface`, `Text`,
//! `Rule`, `Spacer`), and patch commands sit here without any baked-in domain schemas.

use serde::{Deserialize, Serialize};

use crate::components::plot::{CandlePoint, PlotData, PlotPoint};
use crate::components::stat::StatData;
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

/// Core retained Scene IR node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub priority: u8,
    pub visible: bool,
    pub width: SizeSpec,
    pub height: SizeSpec,
    pub min_width: f32,
    pub min_height: f32,
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
            | NodeKind::Stat(_)
            | NodeKind::Plot(_) => {}
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
    /// Adaptive KPI metric tile with counter spring, delta pill, sparkline/range bar.
    Stat(StatData),
    /// Multi-series / multi-pane GPU chart (Line, Area, StackedArea, Band, Candles, Histogram).
    Plot(PlotData),
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
            ScenePatch::SetTheme(mode) => {
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
        }
    }
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
    SetTheme(ThemeMode),
    SetText { id: String, content: String },
    SetVisible { id: String, visible: bool },
    SetTab { id: String, active: usize },
    SetSegmented { id: String, active: usize },
    ReplaceNode { id: String, node: Node },
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
