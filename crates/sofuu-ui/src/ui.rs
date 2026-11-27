//! Ultra-simple, adaptive fluent builder API (`sofuu_ui::ui`).
//!
//! Lets developers construct single-component windows (e.g. price-only or chart-only) or
//! multi-pane dashboards in a few lines with zero layout boilerplate. Defaults to `ThemeMode::Light`
//! unless `.dark()` is explicitly selected, and auto-adapts sizing when few or solo components exist.

use crate::components::code::{CodeData, CodeHighlight};
use crate::components::feed::{FeedData, FeedItem};
use crate::components::graph::{GraphData, GraphEdge, GraphNode};
use crate::components::plot::{
    CandlePoint, PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind, PlotThreshold,
    ReadoutMode,
};
use crate::components::stat::{StatData, StatVariant};
use crate::components::table::{TableCell, TableColumn, TableData, TableRow};
use crate::ir::{Align, Axis, Insets, Node, NodeKind, Scene, SizeSpec};
use crate::theme::{ThemeMode, Tone};

/// Entry point to build an adaptive `Scene` (defaults to `ThemeMode::Light`).
pub fn app(title: impl Into<String>) -> AppBuilder {
    AppBuilder {
        title: title.into(),
        theme: ThemeMode::Light,
        header: None,
        stats: Vec::new(),
        children: Vec::new(),
        padding: 16.0,
        gap: 12.0,
    }
}

pub struct AppBuilder {
    title: String,
    theme: ThemeMode,
    header: Option<Node>,
    stats: Vec<Node>,
    children: Vec<Node>,
    padding: f32,
    gap: f32,
}

impl AppBuilder {
    /// Opt in to Dark theme explicitly (Light is the default).
    pub fn dark(mut self) -> Self {
        self.theme = ThemeMode::Dark;
        self
    }

    /// Explicitly pick Light theme (already the default).
    pub fn light(mut self) -> Self {
        self.theme = ThemeMode::Light;
        self
    }

    pub fn theme(mut self, mode: ThemeMode) -> Self {
        self.theme = mode;
        self
    }

    pub fn padding(mut self, pad: f32) -> Self {
        self.padding = pad.max(0.0);
        self
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap.max(0.0);
        self
    }

    /// Attach an optional top header row (e.g. title, segmented switchers, `ui::theme_toggle`).
    pub fn header(mut self, node: impl Into<Node>) -> Self {
        self.header = Some(node.into());
        self
    }

    /// Add one or more `Stat` cards cleanly into an auto-adapting responsive grid strip.
    pub fn stats<I, N>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<Node>,
    {
        for item in items {
            self.stats.push(item.into());
        }
        self
    }

    /// Add a component (`stat`, `plot`, `row`, `col`, `button`, etc.).
    pub fn add(mut self, node: impl Into<Node>) -> Self {
        self.children.push(node.into());
        self
    }

    /// Build the adaptive `Scene`:
    /// - Solo `Stat` or `Plot` without explicit height automatically expands to fill the canvas.
    /// - Single/few stats automatically span full width cleanly.
    /// - Plots placed alongside stats automatically fill remaining vertical space.
    pub fn build(mut self) -> Scene {
        let mut root_items = Vec::new();

        if let Some(h) = self.header.take() {
            root_items.push(h);
        }

        let only_one_stat =
            root_items.is_empty() && self.stats.len() == 1 && self.children.is_empty();
        if only_one_stat {
            let mut solo = self.stats.remove(0);
            if solo.height == SizeSpec::Auto {
                solo.height = SizeSpec::Fill(1.0);
            }
            if solo.width == SizeSpec::Auto {
                solo.width = SizeSpec::Fill(1.0);
            }
            root_items.push(solo);
        } else if !self.stats.is_empty() {
            let no_other_body = self.children.is_empty();
            let mut grid = Node::new(
                "auto_stats_grid",
                NodeKind::Grid {
                    columns: 0,
                    min_col_width: 220.0,
                    gap: self.gap,
                    style: None,
                    children: self.stats,
                },
            );
            if no_other_body && grid.height == SizeSpec::Auto {
                grid.height = SizeSpec::Fill(1.0);
            }
            root_items.push(grid);
        }

        let solo_content_child = self.children.len() == 1
            && matches!(
                self.children[0].kind,
                NodeKind::Stat(_)
                    | NodeKind::Table(_)
                    | NodeKind::Feed(_)
                    | NodeKind::Graph(_)
                    | NodeKind::CodeText(_)
            );

        for mut c in self.children {
            if matches!(c.kind, NodeKind::Plot(_)) && c.height == SizeSpec::Auto {
                c.height = SizeSpec::Fill(1.0);
            } else if solo_content_child && c.height == SizeSpec::Auto {
                c.height = SizeSpec::Fill(1.0);
            }
            root_items.push(c);
        }

        let root = if root_items.len() == 1 {
            let mut single = root_items.remove(0);
            if single.width == SizeSpec::Auto {
                single.width = SizeSpec::Fill(1.0);
            }
            if single.height == SizeSpec::Auto {
                single.height = SizeSpec::Fill(1.0);
            }
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: self.gap,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![single],
                },
            )
            .with_padding(Insets::all(self.padding))
        } else {
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: self.gap,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: root_items,
                },
            )
            .with_padding(Insets::all(self.padding))
        };

        Scene::new(self.title, self.theme, root)
    }
}

// ── Quick Component Constructors ───────────────────────────────────────────────────────────────

pub fn text(id: impl Into<String>, content: impl Into<String>) -> TextBuilder {
    TextBuilder {
        id: id.into(),
        content: content.into(),
        size: 14.0,
        color: None,
        wrap: false,
        fill_w: false,
    }
}

pub fn title(content: impl Into<String>) -> TextBuilder {
    text("title", content).size(18.0).fill()
}

pub struct TextBuilder {
    id: String,
    content: String,
    size: f32,
    color: Option<u32>,
    wrap: bool,
    fill_w: bool,
}

impl TextBuilder {
    pub fn size(mut self, px: f32) -> Self {
        self.size = px;
        self
    }

    pub fn color(mut self, c: u32) -> Self {
        self.color = Some(c);
        self
    }

    pub fn wrap(mut self) -> Self {
        self.wrap = true;
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_w = true;
        self
    }
}

impl From<TextBuilder> for Node {
    fn from(b: TextBuilder) -> Self {
        let mut n = Node::new(
            b.id,
            NodeKind::Text {
                content: b.content,
                size: b.size,
                color: b.color,
                mono: false,
                wrap: b.wrap,
            },
        );
        if b.fill_w {
            n.width = SizeSpec::Fill(1.0);
        }
        n
    }
}

pub fn row(id: impl Into<String>) -> ContainerBuilder {
    ContainerBuilder {
        id: id.into(),
        axis: Axis::Row,
        align: Align::Center,
        gap: 10.0,
        fold_width: Some(680.0),
        fill_h: false,
        children: Vec::new(),
    }
}

pub fn col(id: impl Into<String>) -> ContainerBuilder {
    ContainerBuilder {
        id: id.into(),
        axis: Axis::Column,
        align: Align::Stretch,
        gap: 12.0,
        fold_width: None,
        fill_h: false,
        children: Vec::new(),
    }
}

pub struct ContainerBuilder {
    id: String,
    axis: Axis,
    align: Align,
    gap: f32,
    fold_width: Option<f32>,
    fill_h: bool,
    children: Vec<Node>,
}

impl ContainerBuilder {
    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    pub fn fold_below(mut self, width: f32) -> Self {
        self.fold_width = Some(width);
        self
    }

    pub fn align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_h = true;
        self.align = Align::Stretch;
        self
    }

    pub fn add(mut self, child: impl Into<Node>) -> Self {
        self.children.push(child.into());
        self
    }
}

impl From<ContainerBuilder> for Node {
    fn from(b: ContainerBuilder) -> Self {
        let mut n = Node::new(
            b.id,
            NodeKind::Container {
                axis: b.axis,
                align: b.align,
                gap: b.gap,
                fold_width: b.fold_width,
                scrollable: false,
                style: None,
                children: b.children,
            },
        );
        if b.fill_h {
            n.height = SizeSpec::Fill(1.0);
        }
        n
    }
}

pub fn segmented(
    id: impl Into<String>,
    active: usize,
    options: impl IntoIterator<Item = impl Into<String>>,
) -> Node {
    Node::new(
        id,
        NodeKind::Segmented {
            active,
            options: options.into_iter().map(Into::into).collect(),
        },
    )
}

pub fn button(id: impl Into<String>, label: impl Into<String>) -> Node {
    button_toned(id, label, Tone::Neutral)
}

pub fn button_toned(id: impl Into<String>, label: impl Into<String>, tone: Tone) -> Node {
    Node::new(
        id,
        NodeKind::Button {
            label: label.into(),
            tone,
        },
    )
}

pub fn theme_toggle(id: impl Into<String>) -> Node {
    Node::new(id, NodeKind::ThemeToggle { size: 34.0 })
}

pub fn switch(id: impl Into<String>, checked: bool, label: Option<&str>) -> Node {
    Node::new(
        id,
        NodeKind::Switch {
            checked,
            label: label.map(ToString::to_string),
            tone: Tone::Neutral,
        },
    )
}

pub fn input(id: impl Into<String>, placeholder: impl Into<String>) -> Node {
    let mut n = Node::new(
        id,
        NodeKind::Input {
            value: String::new(),
            placeholder: placeholder.into(),
            tone: Tone::Neutral,
        },
    );
    n.height = SizeSpec::Px(34.0);
    n
}

// ── Fluent Stat Builder ────────────────────────────────────────────────────────────────────────

pub fn stat(id: impl Into<String>, label: impl Into<String>, value: f64) -> StatBuilder {
    StatBuilder {
        id: id.into(),
        data: StatData {
            label: label.into(),
            value,
            ..StatData::default()
        },
    }
}

pub struct StatBuilder {
    id: String,
    data: StatData,
}

impl StatBuilder {
    pub fn badge(mut self, badge: impl Into<String>) -> Self {
        self.data.badge = Some(badge.into());
        self
    }

    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.data.prefix = prefix.into();
        self
    }

    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.data.suffix = suffix.into();
        self
    }

    pub fn precision(mut self, decimals: usize) -> Self {
        self.data.precision = Some(decimals);
        self
    }

    pub fn delta(mut self, delta: impl Into<String>) -> Self {
        self.data.delta = Some(delta.into());
        self
    }

    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.data.caption = Some(caption.into());
        self
    }

    pub fn spark(mut self, vals: impl Into<Vec<f32>>) -> Self {
        self.data.variant = StatVariant::Sparkline;
        self.data.sparkline = vals.into();
        self
    }

    pub fn range(mut self, low: f64, high: f64) -> Self {
        self.data.variant = StatVariant::RangeBar;
        self.data.range_bounds = Some([low, high]);
        self
    }

    pub fn compact(mut self) -> Self {
        self.data.variant = StatVariant::Compact;
        self
    }

    pub fn ok(mut self) -> Self {
        self.data.tone = Tone::Ok;
        self
    }

    pub fn info(mut self) -> Self {
        self.data.tone = Tone::Info;
        self
    }

    pub fn warn(mut self) -> Self {
        self.data.tone = Tone::Warn;
        self
    }

    pub fn danger(mut self) -> Self {
        self.data.tone = Tone::Danger;
        self
    }

    pub fn shadow(mut self, blur: f32, offset_y: f32) -> Self {
        self.data.shadow_blur = blur.max(0.0);
        self.data.shadow_offset_y = offset_y;
        self
    }
}

impl From<StatBuilder> for Node {
    fn from(b: StatBuilder) -> Self {
        Node::new(b.id, NodeKind::Stat(b.data))
    }
}

// ── Fluent Plot & Pane Builders ────────────────────────────────────────────────────────────────

pub fn plot(id: impl Into<String>) -> PlotBuilder {
    PlotBuilder {
        id: id.into(),
        data: PlotData::default(),
    }
}

pub fn pane(id: impl Into<String>) -> PaneBuilder {
    PaneBuilder {
        pane: PlotPane {
            id: id.into(),
            ..PlotPane::default()
        },
    }
}

pub struct PlotBuilder {
    id: String,
    data: PlotData,
}

impl PlotBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.data.title = Some(title.into());
        self
    }

    pub fn subtitle(mut self, sub: impl Into<String>) -> Self {
        self.data.subtitle = Some(sub.into());
        self
    }

    pub fn y_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.data.y_prefix = prefix.into();
        self
    }

    pub fn y_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.data.y_suffix = suffix.into();
        self
    }

    pub fn precision(mut self, decimals: usize) -> Self {
        self.data.y_precision = Some(decimals);
        self
    }

    pub fn percent(mut self) -> Self {
        self.data.stacked_percent = true;
        self.data.y_suffix = "%".into();
        self
    }

    pub fn readout(mut self, mode: ReadoutMode) -> Self {
        self.data.readout_mode = mode;
        self
    }

    pub fn candles(mut self, id: impl Into<String>, name: impl Into<String>, candles: Vec<CandlePoint>) -> Self {
        self.data.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Candles,
            color: None,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: None,
            points: vec![],
            candles,
        });
        self
    }

    pub fn line(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: Option<u32>) -> Self {
        self.data.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Line,
            color,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: Some(1.7),
            points,
            candles: vec![],
        });
        self
    }

    pub fn area(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: Option<u32>) -> Self {
        self.data.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Area,
            color,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: Some(1.7),
            points,
            candles: vec![],
        });
        self
    }

    pub fn stacked(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: u32) -> Self {
        self.data.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::StackedArea,
            color: Some(color),
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: None,
            points,
            candles: vec![],
        });
        self
    }

    pub fn pane(mut self, pane: impl Into<PlotPane>) -> Self {
        self.data.panes.push(pane.into());
        self
    }

    pub fn threshold(mut self, value: f64, label: Option<String>, color: Option<u32>) -> Self {
        self.data.thresholds.push(PlotThreshold { value, label, color });
        self
    }

    pub fn shadow(mut self, blur: f32, offset_y: f32) -> Self {
        self.data.shadow_blur = blur.max(0.0);
        self.data.shadow_offset_y = offset_y;
        self
    }
}

impl From<PlotBuilder> for Node {
    fn from(b: PlotBuilder) -> Self {
        Node::new(b.id, NodeKind::Plot(b.data)).with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0))
    }
}

pub struct PaneBuilder {
    pane: PlotPane,
}

impl PaneBuilder {
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.pane.label = Some(label.into());
        self
    }

    pub fn weight(mut self, weight: f32) -> Self {
        self.pane.height_weight = weight.max(0.1);
        self
    }

    pub fn y_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.pane.y_prefix = prefix.into();
        self
    }

    pub fn precision(mut self, decimals: usize) -> Self {
        self.pane.y_precision = Some(decimals);
        self
    }

    pub fn threshold(mut self, value: f64, label: impl Into<String>, color: Option<u32>) -> Self {
        self.pane.thresholds.push(PlotThreshold {
            value,
            label: Some(label.into()),
            color,
        });
        self
    }

    pub fn candles(mut self, id: impl Into<String>, name: impl Into<String>, candles: Vec<CandlePoint>) -> Self {
        self.pane.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Candles,
            color: None,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: None,
            points: vec![],
            candles,
        });
        self
    }

    pub fn line(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: Option<u32>) -> Self {
        self.pane.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Line,
            color,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: Some(1.6),
            points,
            candles: vec![],
        });
        self
    }

    pub fn area(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: Option<u32>) -> Self {
        self.pane.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Area,
            color,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: Some(1.6),
            points,
            candles: vec![],
        });
        self
    }

    pub fn band(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        points: Vec<PlotPoint>,
        color_a: Option<u32>,
        color_b: Option<u32>,
    ) -> Self {
        self.pane.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Band,
            color: color_a,
            secondary_color: color_b,
            visible: true,
            dashed: false,
            stroke_width: None,
            points,
            candles: vec![],
        });
        self
    }

    pub fn histogram(mut self, id: impl Into<String>, name: impl Into<String>, points: Vec<PlotPoint>, color: Option<u32>) -> Self {
        self.pane.series.push(PlotSeries {
            id: id.into(),
            name: name.into(),
            kind: PlotSeriesKind::Histogram,
            color,
            secondary_color: None,
            visible: true,
            dashed: false,
            stroke_width: None,
            points,
            candles: vec![],
        });
        self
    }
}

impl From<PaneBuilder> for PlotPane {
    fn from(b: PaneBuilder) -> Self {
        b.pane
    }
}

// ── Fluent Table Builder ───────────────────────────────────────────────────────────────────────

pub fn table(id: impl Into<String>) -> TableBuilder {
    TableBuilder {
        id: id.into(),
        data: TableData::default(),
        fill_both: false,
    }
}

pub struct TableBuilder {
    id: String,
    data: TableData,
    fill_both: bool,
}

impl TableBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.data.title = Some(title.into());
        self
    }

    pub fn col(mut self, col: TableColumn) -> Self {
        self.data.columns.push(col);
        self
    }

    pub fn columns(mut self, cols: impl IntoIterator<Item = TableColumn>) -> Self {
        self.data.columns.extend(cols);
        self
    }

    pub fn row(
        mut self,
        id: impl Into<String>,
        cells: impl IntoIterator<Item = TableCell>,
    ) -> Self {
        self.data
            .rows
            .push(TableRow::new(id, cells.into_iter().collect()));
        self
    }

    pub fn rows(mut self, rows: impl IntoIterator<Item = TableRow>) -> Self {
        self.data.rows.extend(rows);
        self
    }

    pub fn empty_text(mut self, msg: impl Into<String>) -> Self {
        self.data.empty_text = Some(msg.into());
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_both = true;
        self
    }
}

impl From<TableBuilder> for Node {
    fn from(b: TableBuilder) -> Self {
        let mut n = Node::new(b.id, NodeKind::Table(b.data));
        if b.fill_both {
            n.width = SizeSpec::Fill(1.0);
            n.height = SizeSpec::Fill(1.0);
        }
        n
    }
}

// ── Fluent Feed Builder ────────────────────────────────────────────────────────────────────────

pub fn feed(id: impl Into<String>) -> FeedBuilder {
    FeedBuilder {
        id: id.into(),
        data: FeedData::default(),
        fill_both: false,
    }
}

pub struct FeedBuilder {
    id: String,
    data: FeedData,
    fill_both: bool,
}

impl FeedBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.data.title = Some(title.into());
        self
    }

    pub fn item(mut self, item: FeedItem) -> Self {
        self.data.items.push(item);
        self
    }

    pub fn items(mut self, items: impl IntoIterator<Item = FeedItem>) -> Self {
        self.data.items.extend(items);
        self
    }

    pub fn max_items(mut self, cap: usize) -> Self {
        self.data.max_items = cap.max(1);
        self
    }

    pub fn empty_text(mut self, msg: impl Into<String>) -> Self {
        self.data.empty_text = Some(msg.into());
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_both = true;
        self
    }
}

impl From<FeedBuilder> for Node {
    fn from(b: FeedBuilder) -> Self {
        let mut n = Node::new(b.id, NodeKind::Feed(b.data));
        if b.fill_both {
            n.width = SizeSpec::Fill(1.0);
            n.height = SizeSpec::Fill(1.0);
        }
        n
    }
}

// ── Fluent Graph Builder ───────────────────────────────────────────────────────────────────────

pub fn graph(id: impl Into<String>) -> GraphBuilder {
    GraphBuilder {
        id: id.into(),
        data: GraphData::default(),
        fill_both: false,
    }
}

pub struct GraphBuilder {
    id: String,
    data: GraphData,
    fill_both: bool,
}

impl GraphBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.data.title = Some(title.into());
        self
    }

    pub fn node(mut self, node: GraphNode) -> Self {
        self.data.nodes.push(node);
        self
    }

    pub fn edge(mut self, edge: GraphEdge) -> Self {
        self.data.edges.push(edge);
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_both = true;
        self
    }
}

impl From<GraphBuilder> for Node {
    fn from(b: GraphBuilder) -> Self {
        let mut n = Node::new(b.id, NodeKind::Graph(b.data));
        if b.fill_both {
            n.width = SizeSpec::Fill(1.0);
            n.height = SizeSpec::Fill(1.0);
        }
        n
    }
}

// ── Fluent CodeText Builder ────────────────────────────────────────────────────────────────────

pub fn code(id: impl Into<String>, content: impl Into<String>) -> CodeBuilder {
    CodeBuilder {
        id: id.into(),
        data: CodeData {
            content: content.into(),
            ..CodeData::default()
        },
        fill_both: false,
    }
}

pub struct CodeBuilder {
    id: String,
    data: CodeData,
    fill_both: bool,
}

impl CodeBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.data.title = Some(title.into());
        self
    }

    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.data.language = Some(lang.into());
        self
    }

    pub fn line_numbers(mut self, show: bool) -> Self {
        self.data.show_line_numbers = show;
        self
    }

    pub fn highlight(mut self, line: usize, tone: Tone) -> Self {
        self.data.highlights.push(CodeHighlight { line, tone });
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill_both = true;
        self
    }
}

impl From<CodeBuilder> for Node {
    fn from(b: CodeBuilder) -> Self {
        let mut n = Node::new(b.id, NodeKind::CodeText(b.data));
        if b.fill_both {
            n.width = SizeSpec::Fill(1.0);
            n.height = SizeSpec::Fill(1.0);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::UiEngine;
    use crate::draw::DrawList;

    #[test]
    fn default_theme_is_light_unless_dark_specified() {
        let light_scene = app("Light App")
            .add(stat("s", "Price", 100.0).prefix("$").ok())
            .build();
        assert_eq!(light_scene.theme, ThemeMode::Light);

        let dark_scene = app("Dark App")
            .dark()
            .add(stat("s", "Price", 100.0).prefix("$").ok())
            .build();
        assert_eq!(dark_scene.theme, ThemeMode::Dark);
    }

    #[test]
    fn solo_stat_and_solo_plot_auto_fill_full_window() {
        let solo_stat = app("Price Only")
            .add(stat("s", "Price", 248.79).prefix("$").spark(vec![240.0, 245.0, 248.8]))
            .build();
        let mut engine = UiEngine::new(solo_stat, [800.0, 480.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        let r = engine.layout.get("s").unwrap();
        assert!(r.w > 750.0 && r.h > 420.0, "solo stat must expand cleanly to fill window: {r:?}");

        let solo_plot = app("Chart Only")
            .add(
                plot("p")
                    .title("Price")
                    .line("l", "Feed", vec![PlotPoint { x: 0.0, y: 10.0, y2: None, label: None }, PlotPoint { x: 1.0, y: 12.0, y2: None, label: None }], None),
            )
            .build();
        let mut p_eng = UiEngine::new(solo_plot, [800.0, 480.0]);
        let _ = p_eng.tick_and_compile(1.0 / 120.0, &mut dl);
        let pr = p_eng.layout.get("p").unwrap();
        assert!(pr.w > 750.0 && pr.h > 420.0, "solo plot must fill window: {pr:?}");
    }
}
