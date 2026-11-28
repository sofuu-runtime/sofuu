//! Geometry, accessibility & contrast audit engine for `UiEngine` scenes.
//!
//! Every frame already produces exact `ComputedRect`s (layout) and a flat
//! `DrawList` (quads / ribbons / strokes / glyphs). This module reads those
//! two products — it never re-measures text or re-runs layout — and reports
//! structured, tagged violations that AI agents can consume as JSON.
//!
//! The checks are deliberately domain-agnostic. Nothing here knows what an
//! app is for; it only knows IR kinds, rectangles, safe areas and emitted
//! colors:
//!
//! 1. `overlap_siblings`
//! 2. `clipped_or_overflow`
//! 3. `zero_area_visible_node`
//! 4. `safe_area_collision`
//! 5. `keyboard_occluded`
//! 6. `touch_target_small`
//! 7. `contrast_low`

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::draw::{unpack_rgba8, DrawList, QuadInstance};
use crate::ir::{Insets, Node, NodeKind, Scene};
use crate::layout::{child_path_hash, ComputedRect, LayoutMap, ROOT_PATH_HASH};
use crate::text::GlyphAtlas;

/// Platform rules applied to one audit run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditRules {
    /// `android`, `ios`, `macos`, `windows`, `linux`, or a custom id.
    #[serde(default)]
    pub platform: String,
    /// Logical app viewport `[width, height]` (the engine's coordinate space).
    #[serde(default)]
    pub viewport: [f32; 2],
    /// Whether this profile is touch-first (enables touch-target checks).
    #[serde(default)]
    pub touch: bool,
    /// Minimum interactive target (`44` ios, `48` android); `0` disables.
    #[serde(default)]
    pub min_touch: f32,
    /// Reserved insets in the same logical coordinate space as the scene.
    #[serde(default)]
    pub safe: Insets,
    #[serde(default)]
    pub keyboard_open: bool,
    /// Height of the software IME in logical px.
    #[serde(default)]
    pub keyboard_height: f32,
    /// WCAG normal-text minimum (4.5 by default).
    #[serde(default = "default_min_contrast")]
    pub min_contrast: f32,
    /// WCAG large-text minimum (3.0 by default).
    #[serde(default = "default_min_large_contrast")]
    pub min_large_contrast: f32,
    /// Text at or above this size is "large" for contrast purposes.
    #[serde(default = "default_large_text_px")]
    pub large_text_px: f32,
}

fn default_min_contrast() -> f32 {
    4.5
}
fn default_min_large_contrast() -> f32 {
    3.0
}
fn default_large_text_px() -> f32 {
    18.0
}

impl Default for AuditRules {
    fn default() -> Self {
        Self {
            platform: "custom".to_string(),
            viewport: [1280.0, 800.0],
            touch: false,
            min_touch: 0.0,
            safe: Insets::ZERO,
            keyboard_open: false,
            keyboard_height: 0.0,
            min_contrast: 4.5,
            min_large_contrast: 3.0,
            large_text_px: 18.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl Severity {
    pub fn is_error(self) -> bool {
        matches!(self, Severity::Error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    OverlapSiblings,
    ClippedOrOverflow,
    ZeroAreaVisibleNode,
    SafeAreaCollision,
    KeyboardOccluded,
    TouchTargetSmall,
    ContrastLow,
}

impl IssueKind {
    pub fn as_str(self) -> &'static str {
        match self {
            IssueKind::OverlapSiblings => "overlap_siblings",
            IssueKind::ClippedOrOverflow => "clipped_or_overflow",
            IssueKind::ZeroAreaVisibleNode => "zero_area_visible_node",
            IssueKind::SafeAreaCollision => "safe_area_collision",
            IssueKind::KeyboardOccluded => "keyboard_occluded",
            IssueKind::TouchTargetSmall => "touch_target_small",
            IssueKind::ContrastLow => "contrast_low",
        }
    }
}

/// One violation. `tag` matches the numbered overlay box in `annotated` mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditIssue {
    pub tag: u32,
    pub severity: Severity,
    pub kind: IssueKind,
    pub node_id: String,
    /// `[x, y, w, h]` in logical app-viewport pixels.
    pub rect: [f32; 4],
    pub detail: String,
}

/// A tagged node in the companion `nodes` manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditNode {
    pub tag: u32,
    pub id: String,
    pub kind: String,
    pub rect: [f32; 4],
    pub clipped: bool,
    pub interactive: bool,
    pub visible: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditSummary {
    pub errors: usize,
    pub warnings: usize,
    pub interactive_nodes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditPerf {
    pub quads: usize,
    pub ribbons: usize,
    pub strokes: usize,
    pub glyphs: usize,
    pub sprites: usize,
    #[serde(default)]
    pub render_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    /// Filled by the capture layer (`screenshot.rs`); audits are also useful
    /// without a PNG.
    #[serde(default)]
    pub capture_png: Option<String>,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub device: String,
    /// Logical app viewport `[w, h]`.
    #[serde(default)]
    pub viewport_px: [f32; 2],
    #[serde(default)]
    pub scale_factor: f32,
    #[serde(default)]
    pub theme: String,
    #[serde(default)]
    pub keyboard_open: bool,
    #[serde(default)]
    pub settled: bool,
    #[serde(default)]
    pub frames_stepped: u32,
    /// Which renderer produced the pixels (`metal`, `software`, ...). Filled
    /// by the capture layer; empty when the audit ran without a PNG.
    #[serde(default)]
    pub renderer: String,
    #[serde(default)]
    pub perf: AuditPerf,
    #[serde(default)]
    pub summary: AuditSummary,
    #[serde(default)]
    pub issues: Vec<AuditIssue>,
    #[serde(default)]
    pub nodes: Vec<AuditNode>,
}

impl AuditReport {
    pub fn has_errors(&self) -> bool {
        self.summary.errors > 0
    }
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn from_computed(r: ComputedRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        }
    }

    fn x2(self) -> f32 {
        self.x + self.w
    }
    fn y2(self) -> f32 {
        self.y + self.h
    }

    fn intersection_area(self, other: Self) -> f32 {
        let w = (self.x2().min(other.x2()) - self.x.max(other.x)).max(0.0);
        let h = (self.y2().min(other.y2()) - self.y.max(other.y)).max(0.0);
        w * h
    }

    /// `inner` is fully inside `self` (1 px tolerance for sub-pixel layout).
    fn contains(self, inner: Self) -> bool {
        inner.x >= self.x - 1.0
            && inner.y >= self.y - 1.0
            && inner.x2() <= self.x2() + 1.0
            && inner.y2() <= self.y2() + 1.0
    }

    fn overlaps_band_top(self, band: f32) -> bool {
        band > 0.0 && self.y < band - 0.5
    }
    fn overlaps_band_bottom(self, band: f32, viewport_h: f32) -> bool {
        band > 0.0 && self.y2() > viewport_h - band + 0.5
    }
    fn overlaps_band_left(self, band: f32) -> bool {
        band > 0.0 && self.x < band - 0.5
    }
    fn overlaps_band_right(self, band: f32, viewport_w: f32) -> bool {
        band > 0.0 && self.x2() > viewport_w - band + 0.5
    }

    fn as_array(self) -> [f32; 4] {
        [round2(self.x), round2(self.y), round2(self.w), round2(self.h)]
    }
}

fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClipKind {
    Scroll,
    Surface,
}

#[derive(Clone)]
struct Walked {
    id: String,
    kind: String,
    rect: Rect,
    interactive: bool,
    visible: bool,
    /// The node is (partly) outside a hard clip that is supposed to contain it.
    clipped: bool,
    text_size: Option<f32>,
    text_wrap: bool,
    text: Option<(String, f32, bool)>,
}

struct Walker<'a> {
    layout: &'a LayoutMap,
    scrolls: &'a HashMap<String, f32>,
    nodes: Vec<Walked>,
    issues: Vec<AuditIssue>,
    rules: &'a AuditRules,
    atlas: &'a GlyphAtlas,
}

impl<'a> Walker<'a> {
    fn new(
        layout: &'a LayoutMap,
        scrolls: &'a HashMap<String, f32>,
        rules: &'a AuditRules,
        atlas: &'a GlyphAtlas,
    ) -> Self {
        Self {
            layout,
            scrolls,
            nodes: Vec::new(),
            issues: Vec::new(),
            rules,
            atlas,
        }
    }

    fn walk(
        &mut self,
        node: &Node,
        path_hash: u64,
        clip: Option<(Rect, ClipKind)>,
        scroll_y: f32,
    ) {
        if !node.visible {
            return;
        }
        let Some(computed) = self.layout.get_path(path_hash) else {
            return;
        };
        let mut rect = Rect::from_computed(computed);
        rect.y -= scroll_y;

        let clipped = match clip {
            Some((c, kind)) if kind != ClipKind::Scroll => !c.contains(rect),
            _ => false,
        };

        let interactive = is_interactive(node);
        let text_size = match &node.kind {
            NodeKind::Text { size, .. } => Some(*size),
            _ => None,
        };
        let text_wrap = matches!(&node.kind, NodeKind::Text { wrap: true, .. });
        let text = match &node.kind {
            NodeKind::Text { content, size, mono, .. } => {
                Some((content.clone(), *size, *mono))
            }
            _ => None,
        };

        self.nodes.push(Walked {
            id: node.id.clone(),
            kind: kind_name(&node.kind).to_string(),
            rect,
            interactive,
            visible: true,
            clipped,
            text_size,
            text_wrap,
            text,
        });

        match &node.kind {
            NodeKind::Container {
                scrollable,
                children,
                ..
            } => {
                let child_clip = if *scrollable {
                    Some((rect, ClipKind::Scroll))
                } else {
                    clip
                };
                let next_scroll = if *scrollable {
                    scroll_y + self.scrolls.get(&node.id).copied().unwrap_or(0.0)
                } else {
                    scroll_y
                };
                for (slot, child) in children.iter().enumerate() {
                    let h = child_path_hash(path_hash, slot, &child.id);
                    self.walk(child, h, child_clip, next_scroll);
                }
                self.check_siblings(children, rect, *scrollable);
            }
            NodeKind::Grid { children, .. } => {
                for (slot, child) in children.iter().enumerate() {
                    let h = child_path_hash(path_hash, slot, &child.id);
                    self.walk(child, h, clip, scroll_y);
                }
                self.check_siblings(children, rect, false);
            }
            NodeKind::Surface { child, .. } => {
                if let Some(c) = child {
                    let h = child_path_hash(path_hash, 0, &c.id);
                    self.walk(c, h, Some((rect, ClipKind::Surface)), scroll_y);
                }
            }
            NodeKind::Tabs { active, items } => {
                if let Some(item) = items.get(*active) {
                    let h = child_path_hash(path_hash, *active, &item.child.id);
                    self.walk(&item.child, h, clip, scroll_y);
                }
            }
            NodeKind::Stack { children, .. } => {
                for (slot, child) in children.iter().enumerate() {
                    let h = child_path_hash(path_hash, slot, &child.id);
                    self.walk(child, h, clip, scroll_y);
                }
            }
            NodeKind::Pressable {
                child: Some(c), ..
            } => {
                let h = child_path_hash(path_hash, 0, &c.id);
                self.walk(c, h, clip, scroll_y);
            }
            NodeKind::ListView(data) => {
                for (slot, item) in data.items.iter().enumerate() {
                    let h = child_path_hash(path_hash, slot, &item.child.id);
                    self.walk(&item.child, h, Some((rect, ClipKind::Scroll)), scroll_y);
                }
            }
            NodeKind::NavStack { routes, active, .. } => {
                if let Some(route) = routes.get(*active) {
                    let h = child_path_hash(path_hash, *active, &route.child.id);
                    self.walk(&route.child, h, Some((rect, ClipKind::Scroll)), scroll_y);
                }
            }
            NodeKind::SafeArea { child, .. } => {
                if let Some(c) = child {
                    let h = child_path_hash(path_hash, 0, &c.id);
                    self.walk(c, h, Some((rect, ClipKind::Surface)), scroll_y);
                }
            }
            NodeKind::Overlay { child, .. } => {
                if let Some(c) = child {
                    let h = child_path_hash(path_hash, 0, &c.id);
                    self.walk(c, h, Some((rect, ClipKind::Surface)), scroll_y);
                }
            }
            NodeKind::Draggable { child, .. } => {
                if let Some(c) = child {
                    let h = child_path_hash(path_hash, 0, &c.id);
                    self.walk(c, h, clip, scroll_y);
                }
            }
            _ => {}
        }
    }

    fn check_siblings(&mut self, children: &[Node], parent: Rect, scrollable: bool) {
        // Overflow of a child outside its non-scrollable parent.
        if !scrollable {
            for child in children.iter().filter(|c| c.visible) {
                let Some(r) = self.node_rect(&child.id) else {
                    continue;
                };
                if !parent.contains(r) && !matches!(child.kind, NodeKind::Spacer | NodeKind::Rule { .. }) {
                    self.push(
                        IssueKind::ClippedOrOverflow,
                        Severity::Warning,
                        &child.id,
                        r,
                        format!(
                            "child escapes parent bounds (child {:.0}x{:.0} at {:.0},{:.0}; parent {:.0}x{:.0})",
                            r.w, r.h, r.x, r.y, parent.w, parent.h
                        ),
                    );
                }
            }
        }

        let mut rects: Vec<(&Node, Rect)> = Vec::new();
        for child in children {
            if !child.visible || matches!(child.kind, NodeKind::Spacer | NodeKind::Rule { .. }) {
                continue;
            }
            if let Some(r) = self.node_rect(&child.id) {
                if r.w >= 0.5 || r.h >= 0.5 {
                    rects.push((child, r));
                }
            }
        }
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                let (a, ar) = rects[i];
                let (b, br) = rects[j];
                let overlap = ar.intersection_area(br);
                if overlap <= 1.0 {
                    continue;
                }
                // A card/surface intentionally contains its own child; siblings
                // are only a problem when neither fully contains the other.
                if ar.contains(br) || br.contains(ar) {
                    continue;
                }
                self.push(
                    IssueKind::OverlapSiblings,
                    Severity::Error,
                    &b.id,
                    br,
                    format!(
                        "sibling `{}` overlaps `{}` by {:.0}px² (layer `{}`)",
                        b.id,
                        a.id,
                        overlap,
                        parent_kind_label(a)
                    ),
                );
            }
        }
    }

    fn node_rect(&self, id: &str) -> Option<Rect> {
        self.nodes.iter().find(|n| n.id == id).map(|n| n.rect)
    }

    fn push(
        &mut self,
        kind: IssueKind,
        severity: Severity,
        node_id: &str,
        rect: Rect,
        detail: String,
    ) {
        // One issue per (kind,node) keeps annotated overlays readable.
        if self
            .issues
            .iter()
            .any(|i| i.kind == kind && i.node_id == node_id)
        {
            return;
        }
        self.issues.push(AuditIssue {
            tag: 0,
            severity,
            kind,
            node_id: node_id.to_string(),
            rect: rect.as_array(),
            detail,
        });
    }

    fn finish(mut self, draw: &DrawList, scene: &Scene) -> AuditReport {
        // 3. zero-area visible nodes (Spacer/Rule are intentionally thin).
        for n in self.nodes.clone() {
            if n.kind == "spacer" || n.kind == "rule" {
                continue;
            }
            if n.rect.w < 1.0 || n.rect.h < 1.0 {
                self.push(
                    IssueKind::ZeroAreaVisibleNode,
                    Severity::Warning,
                    &n.id,
                    n.rect,
                    format!(
                        "visible `{}` collapsed to {:.2}x{:.2}px during layout",
                        n.id, n.rect.w, n.rect.h
                    ),
                );
            }
        }

        // 4/5/6. Platform rules.
        let vw = self.rules.viewport[0];
        let vh = self.rules.viewport[1];
        for n in self.nodes.clone() {
            if !n.interactive {
                continue;
            }
            if n.rect.overlaps_band_top(self.rules.safe.top)
                || n.rect.overlaps_band_left(self.rules.safe.left)
                || n.rect.overlaps_band_right(self.rules.safe.right, vw)
            {
                let detail = if n.rect.overlaps_band_top(self.rules.safe.top) {
                    format!(
                        "top edge {:.0} is inside the {:.0}px top safe area",
                        n.rect.y, self.rules.safe.top
                    )
                } else if n.rect.overlaps_band_left(self.rules.safe.left) {
                    format!(
                        "left edge {:.0} is inside the {:.0}px left safe area",
                        n.rect.x, self.rules.safe.left
                    )
                } else {
                    format!(
                        "right edge {:.0} is inside the {:.0}px right safe area",
                        n.rect.x2(),
                        self.rules.safe.right
                    )
                };
                self.push(
                    IssueKind::SafeAreaCollision,
                    Severity::Error,
                    &n.id,
                    n.rect,
                    detail,
                );
            }

            if self.rules.keyboard_open && self.rules.keyboard_height > 0.0 {
                let kb_top = vh - self.rules.keyboard_height;
                if n.rect.y2() > kb_top + 0.5 && n.rect.y < vh {
                    self.push(
                        IssueKind::KeyboardOccluded,
                        Severity::Error,
                        &n.id,
                        n.rect,
                        format!(
                            "interactive `{}` (y2={:.0}) sits under the open keyboard (kb_top={:.0})",
                            n.id,
                            n.rect.y2(),
                            kb_top
                        ),
                    );
                }
            } else if n.rect.overlaps_band_bottom(self.rules.safe.bottom, vh) {
                self.push(
                    IssueKind::SafeAreaCollision,
                    Severity::Error,
                    &n.id,
                    n.rect,
                    format!(
                        "bottom edge {:.0} is inside the {:.0}px bottom safe area",
                        n.rect.y2(),
                        self.rules.safe.bottom
                    ),
                );
            }

            if self.rules.touch && self.rules.min_touch > 0.0 {
                let min = self.rules.min_touch;
                if n.rect.w < min - 0.5 || n.rect.h < min - 0.5 {
                    self.push(
                        IssueKind::TouchTargetSmall,
                        Severity::Warning,
                        &n.id,
                        n.rect,
                        format!(
                            "{} minimum touch target is {:.0}x{:.0} (actual {:.0}x{:.0})",
                            platform_label(&self.rules.platform),
                            min,
                            min,
                            n.rect.w,
                            n.rect.h
                        ),
                    );
                }
            }
        }

        // 7. Contrast from actually emitted glyphs + quads.
        for n in self.nodes.clone() {
            if n.kind != "text" {
                continue;
            }
            let Some(size) = n.text_size else { continue };
            if let Some(ratio) = worst_text_contrast(n.rect, draw, self.rules.viewport) {
                let required = if size >= self.rules.large_text_px {
                    self.rules.min_large_contrast
                } else {
                    self.rules.min_contrast
                };
                if ratio < required {
                    self.push(
                        IssueKind::ContrastLow,
                        Severity::Warning,
                        &n.id,
                        n.rect,
                        format!(
                            "WCAG contrast {:.2}:1 is below the {:.1}:1 minimum for {:.0}px text",
                            ratio, required, size
                        ),
                    );
                }
            }
        }

        // Truncation: measure the node's own string at its own font size.
        // Scanning every glyph by vertical band (the old detector) reported
        // false positives whenever a neighbouring glyph's quad overshot the
        // line box by a fraction of a pixel; measuring the string is exact
        // and node-scoped.
        for n in self.nodes.clone() {
            if n.kind != "text" || n.text_wrap {
                continue;
            }
            let Some((content, size, mono)) = &n.text else { continue };
            if content.is_empty() {
                continue;
            }
            let (text_w, _) = self.atlas.measure(content, *size, *mono);
            if text_w > n.rect.w + 2.0 {
                self.push(
                    IssueKind::ClippedOrOverflow,
                    Severity::Warning,
                    &n.id,
                    n.rect,
                    format!(
                        "text is {:.0}px wide in a {:.0}px node and is clipped (wrap=false)",
                        text_w, n.rect.w
                    ),
                );
            }
        }

        // Assign tags: interactive nodes first (traversal order), then any
        // static node referenced by an issue.
        let mut tags: HashMap<String, u32> = HashMap::new();
        let mut next = 1u32;
        for n in self.nodes.iter().filter(|n| n.interactive) {
            tags.entry(n.id.clone()).or_insert_with(|| {
                let t = next;
                next += 1;
                t
            });
        }
        let mut issue_order: Vec<usize> = (0..self.issues.len()).collect();
        issue_order.sort_by_key(|&i| {
            let sev = match self.issues[i].severity {
                Severity::Error => 0,
                Severity::Warning => 1,
                Severity::Info => 2,
            };
            (sev, i)
        });
        let mut sorted_issues = Vec::with_capacity(self.issues.len());
        for i in issue_order {
            let mut issue = self.issues[i].clone();
            let tag = *tags.entry(issue.node_id.clone()).or_insert_with(|| {
                let t = next;
                next += 1;
                t
            });
            issue.tag = tag;
            sorted_issues.push(issue);
        }

        let mut nodes: Vec<AuditNode> = Vec::new();
        for n in &self.nodes {
            let Some(&tag) = tags.get(&n.id) else { continue };
            nodes.push(AuditNode {
                tag,
                id: n.id.clone(),
                kind: n.kind.clone(),
                rect: n.rect.as_array(),
                clipped: n.clipped,
                interactive: n.interactive,
                visible: n.visible,
            });
        }
        nodes.sort_by_key(|n| n.tag);

        let errors = sorted_issues
            .iter()
            .filter(|i| i.severity.is_error())
            .count();
        let warnings = sorted_issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .count();

        AuditReport {
            platform: self.rules.platform.clone(),
            viewport_px: self.rules.viewport,
            keyboard_open: self.rules.keyboard_open,
            perf: AuditPerf {
                quads: draw.quads.len(),
                ribbons: draw.ribbons.len(),
                strokes: draw.strokes.len(),
                glyphs: draw.glyphs.len(),
                sprites: draw.sprites.len(),
                render_ms: 0.0,
            },
            summary: AuditSummary {
                errors,
                warnings,
                interactive_nodes: tags.len(),
            },
            issues: sorted_issues,
            nodes,
            ..AuditReport::default()
        }
        .with_scene_meta(scene)
    }
}

impl AuditReport {
    fn with_scene_meta(mut self, scene: &Scene) -> Self {
        self.theme = match scene.theme {
            crate::theme::ThemeMode::Dark => "dark".to_string(),
            crate::theme::ThemeMode::Light => "light".to_string(),
        };
        self
    }
}

fn parent_kind_label(_n: &Node) -> &'static str {
    "layout"
}

fn platform_label(p: &str) -> &'static str {
    match p {
        "ios" => "iOS",
        "android" => "Android",
        "macos" => "macOS",
        "windows" => "Windows",
        "linux" => "Linux",
        _ => "Platform",
    }
}

fn kind_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Container { .. } => "container",
        NodeKind::Grid { .. } => "grid",
        NodeKind::Tabs { .. } => "tabs",
        NodeKind::Surface { .. } => "surface",
        NodeKind::Text { .. } => "text",
        NodeKind::Rule { .. } => "rule",
        NodeKind::Spacer => "spacer",
        NodeKind::ThemeToggle { .. } => "theme_toggle",
        NodeKind::Segmented { .. } => "segmented",
        NodeKind::Button { .. } => "button",
        NodeKind::Switch { .. } => "switch",
        NodeKind::Input { .. } => "input",
        NodeKind::Stat(_) => "stat",
        NodeKind::Plot(_) => "plot",
        NodeKind::Table(_) => "table",
        NodeKind::Feed(_) => "feed",
        NodeKind::Graph(_) => "graph",
        NodeKind::CodeText(_) => "code_text",
        NodeKind::RichText(_) => "rich_text",
        NodeKind::Icon { .. } => "icon",
        NodeKind::Image { .. } => "image",
        NodeKind::Stack { .. } => "stack",
        NodeKind::SafeArea { .. } => "safe_area",
        NodeKind::Overlay { .. } => "overlay",
        NodeKind::NavStack { .. } => "nav_stack",
        NodeKind::ListView(_) => "list_view",
        NodeKind::TextArea(_) => "text_area",
        NodeKind::Slider(_) => "slider",
        NodeKind::RangeSlider(_) => "range_slider",
        NodeKind::Progress(_) => "progress",
        NodeKind::Pressable { .. } => "pressable",
        NodeKind::Draggable { .. } => "draggable",
    }
}

fn is_interactive(node: &Node) -> bool {
    match &node.kind {
        NodeKind::Button { .. }
        | NodeKind::Switch { .. }
        | NodeKind::Input { .. }
        | NodeKind::Segmented { .. }
        | NodeKind::ThemeToggle { .. }
        | NodeKind::Tabs { .. }
        | NodeKind::Table(_)
        | NodeKind::Feed(_)
        | NodeKind::Graph(_)
        | NodeKind::Slider(_)
        | NodeKind::RangeSlider(_)
        | NodeKind::TextArea(_)
        | NodeKind::ListView(_)
        | NodeKind::Draggable { .. } => true,
        NodeKind::Pressable { disabled, .. } => !*disabled,
        NodeKind::Overlay { dismissable, .. } => *dismissable,
        NodeKind::Surface { interactive, .. } => *interactive,
        _ => false,
    }
}

/// Run all seven audit checks against a compiled scene.
pub fn audit_scene(
    scene: &Scene,
    layout: &LayoutMap,
    draw: &DrawList,
    rules: &AuditRules,
) -> AuditReport {
    audit_scene_with_scrolls(scene, layout, draw, rules, &HashMap::new())
}

/// Audit with the engine's live kinetic-scroll offsets so off-screen rows are
/// measured where the frame actually painted them.
pub fn audit_scene_with_scrolls(
    scene: &Scene,
    layout: &LayoutMap,
    draw: &DrawList,
    rules: &AuditRules,
    scrolls: &HashMap<String, f32>,
) -> AuditReport {
    let atlas = GlyphAtlas::builtin();
    audit_scene_with_atlas(scene, layout, draw, rules, scrolls, &atlas)
}

/// Audit reusing the engine's already-loaded glyph atlas (no second SFD parse
/// on every frame).
pub fn audit_scene_with_atlas(
    scene: &Scene,
    layout: &LayoutMap,
    draw: &DrawList,
    rules: &AuditRules,
    scrolls: &HashMap<String, f32>,
    atlas: &GlyphAtlas,
) -> AuditReport {
    let mut walker = Walker::new(layout, scrolls, rules, atlas);
    walker.walk(&scene.root, ROOT_PATH_HASH, None, 0.0);
    walker.finish(draw, scene)
}

/// Worst (lowest) WCAG contrast for any glyph in `rect`, comparing the emitted
/// glyph color with the composite of emitted quads behind it.
fn worst_text_contrast(rect: Rect, draw: &DrawList, viewport: [f32; 2]) -> Option<f32> {
    let mut worst = f32::INFINITY;
    let mut seen = false;
    for g in &draw.glyphs {
        let cx = g.pos[0] + g.size[0] * 0.5;
        let cy = g.pos[1] + g.size[1] * 0.5;
        if cx < rect.x || cx > rect.x2() || cy < rect.y || cy > rect.y2() {
            continue;
        }
        // Text scrolled or clipped out of view is not read by anyone: sampling
        // it would score the caret against whatever happens to sit underneath
        // (usually nothing) and report a contrast failure for invisible copy.
        if !clip_contains(draw, g.clip_id, cx, cy) {
            continue;
        }
        if viewport[0] > 0.0
            && (cx < 0.0 || cy < 0.0 || cx > viewport[0] || cy > viewport[1])
        {
            continue;
        }
        // Glyph quads can be sparse; ignore pure whitespace cells.
        if g.size[0] <= 0.1 || g.size[1] <= 0.1 {
            continue;
        }
        seen = true;
        let bg = composite_at(draw, cx, cy);
        let fg = unpack_rgba8(g.color);
        let ratio = contrast_ratio(blend_over(bg, fg), bg);
        if ratio < worst {
            worst = ratio;
        }
    }
    if seen {
        Some(worst)
    } else {
        None
    }
}

fn composite_at(draw: &DrawList, x: f32, y: f32) -> [f32; 3] {
    // Start from the frame background; the first full-viewport quad is
    // normally the root. If none is found, use opaque black.
    let mut out = [0.0f32, 0.0, 0.0];
    for q in &draw.quads {
        if !quad_contains(q, x, y) {
            continue;
        }
        if !clip_contains(draw, q.clip_id, x, y) {
            continue;
        }
        let (c, a) = quad_color_at(q, y);
        out = [
            c[0] * a + out[0] * (1.0 - a),
            c[1] * a + out[1] * (1.0 - a),
            c[2] * a + out[2] * (1.0 - a),
        ];
    }
    out
}

fn quad_contains(q: &QuadInstance, x: f32, y: f32) -> bool {
    let [qx, qy, qw, qh] = q.rect;
    // Include the shadow/blur margin so soft shadows tint the sampled base.
    let margin = if q.params[1] > 0.01 {
        q.params[1] + q.params[2].abs()
    } else {
        0.0
    };
    x >= qx - margin && x <= qx + qw + margin && y >= qy - margin && y <= qy + qh + margin
}

fn clip_contains(draw: &DrawList, clip_id: u32, x: f32, y: f32) -> bool {
    match draw.clips.get(clip_id as usize) {
        Some(c) => x >= c.x && x <= c.x + c.w && y >= c.y && y <= c.y + c.h,
        None => true,
    }
}

fn quad_color_at(q: &QuadInstance, y: f32) -> ([f32; 3], f32) {
    let top = unpack_rgba8(q.color_top);
    let bot = unpack_rgba8(q.color_bottom);
    let h = q.rect[3].max(1.0);
    let t = ((y - q.rect[1]) / h).clamp(0.0, 1.0);
    let a = (top[3] as f32 / 255.0) * (1.0 - t) + (bot[3] as f32 / 255.0) * t;
    let mix = |i: usize| {
        (top[i] as f32 / 255.0) * (1.0 - t) + (bot[i] as f32 / 255.0) * t
    };
    ([mix(0), mix(1), mix(2)], a)
}

/// Composite a straight-alpha sRGB color over a straight-alpha background.
fn blend_over(bg: [f32; 3], fg: [u8; 4]) -> [f32; 3] {
    let a = fg[3] as f32 / 255.0;
    [
        (fg[0] as f32 / 255.0) * a + bg[0] * (1.0 - a),
        (fg[1] as f32 / 255.0) * a + bg[1] * (1.0 - a),
        (fg[2] as f32 / 255.0) * a + bg[2] * (1.0 - a),
    ]
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn relative_luminance(c: [f32; 3]) -> f32 {
    0.2126 * srgb_to_linear(c[0].clamp(0.0, 1.0))
        + 0.7152 * srgb_to_linear(c[1].clamp(0.0, 1.0))
        + 0.0722 * srgb_to_linear(c[2].clamp(0.0, 1.0))
}

fn contrast_ratio(a: [f32; 3], b: [f32; 3]) -> f32 {
    let la = relative_luminance(a);
    let lb = relative_luminance(b);
    let (hi, lo) = if la > lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Axis, Align, SizeSpec};
    use crate::theme::ThemeMode;

    fn text(id: &str, content: &str) -> Node {
        Node::new(
            id,
            NodeKind::Text {
                content: content.to_string(),
                size: 14.0,
                color: None,
                mono: false,
                wrap: false,
            },
        )
    }

    fn button(id: &str, label: &str) -> Node {
        Node::new(
            id,
            NodeKind::Button {
                label: label.to_string(),
                tone: crate::theme::Tone::Neutral,
            },
        )
    }

    fn col(id: &str, gap: f32, children: Vec<Node>) -> Node {
        Node::new(
            id,
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap,
                fold_width: None,
                scrollable: false,
                style: None,
                children,
            },
        )
    }

    fn compile(scene: &Scene) -> (LayoutMap, DrawList) {
        let mut layout = LayoutMap::default();
        layout.solve(&scene.root, [393.0, 852.0], &crate::text::GlyphAtlas::builtin());
        let mut engine = crate::compiler::UiEngine::new(scene.clone(), [393.0, 852.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(0.016, &mut dl);
        (engine.layout, dl)
    }

    fn rules() -> AuditRules {
        AuditRules {
            platform: "ios".to_string(),
            viewport: [393.0, 852.0],
            touch: true,
            min_touch: 44.0,
            ..AuditRules::default()
        }
    }

    #[test]
    fn detects_overlapping_siblings() {
        let mut a = button("a", "A");
        a.width = SizeSpec::Fill(1.0);
        a.height = SizeSpec::Px(44.0);
        let mut b = button("b", "B");
        b.width = SizeSpec::Fill(1.0);
        b.height = SizeSpec::Px(44.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 0.0, vec![a, b]));
        let (mut layout, dl) = compile(&scene);
        // The solver never overlaps siblings, so force one deliberately.
        layout.set_path_for_test(
            child_path_hash(ROOT_PATH_HASH, 0, "a"),
            ComputedRect {
                x: 10.0,
                y: 10.0,
                w: 200.0,
                h: 44.0,
                ..ComputedRect::default()
            },
        );
        layout.set_path_for_test(
            child_path_hash(ROOT_PATH_HASH, 1, "b"),
            ComputedRect {
                x: 60.0,
                y: 20.0,
                w: 200.0,
                h: 44.0,
                ..ComputedRect::default()
            },
        );
        let r = audit_scene(&scene, &layout, &dl, &rules());
        let issue = r
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::OverlapSiblings)
            .expect("overlapping siblings");
        assert_eq!(issue.node_id, "b");
        assert_eq!(issue.severity, Severity::Error);
    }

    #[test]
    fn detects_zero_area_visible_node() {
        let mut b = button("tiny", "T");
        b.width = SizeSpec::Px(0.0);
        b.height = SizeSpec::Px(0.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 8.0, vec![b]));
        let (layout, dl) = compile(&scene);
        let r = audit_scene(&scene, &layout, &dl, &rules());
        assert!(r
            .issues
            .iter()
            .any(|i| i.kind == IssueKind::ZeroAreaVisibleNode && i.node_id == "tiny"));
    }

    #[test]
    fn detects_touch_target_small() {
        let mut b = button("small", "x");
        b.width = SizeSpec::Px(30.0);
        b.height = SizeSpec::Px(30.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 8.0, vec![b]));
        let (layout, dl) = compile(&scene);
        let r = audit_scene(&scene, &layout, &dl, &rules());
        let issue = r
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::TouchTargetSmall)
            .expect("small target");
        assert_eq!(issue.node_id, "small");
        assert_eq!(issue.severity, Severity::Warning);
    }

    #[test]
    fn detects_keyboard_occlusion() {
        let mut top = text("top", "top");
        top.height = SizeSpec::Px(30.0);
        let mut spacer = Node::new("push", NodeKind::Spacer);
        spacer.height = SizeSpec::Fill(1.0);
        let mut bottom = button("bottom_btn", "Pay");
        bottom.height = SizeSpec::Px(48.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 8.0, vec![top, spacer, bottom]));
        let (layout, dl) = compile(&scene);
        let mut r = rules();
        r.keyboard_open = true;
        r.keyboard_height = 336.0;
        let report = audit_scene(&scene, &layout, &dl, &r);
        let issue = report
            .issues
            .iter()
            .find(|i| i.kind == IssueKind::KeyboardOccluded)
            .expect("keyboard occlusion");
        assert_eq!(issue.node_id, "bottom_btn");
        assert!(issue.detail.contains("kb_top="));
    }

    #[test]
    fn detects_safe_area_collision() {
        let mut b = button("top_btn", "Top");
        b.height = SizeSpec::Px(44.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 0.0, vec![b]));
        let (layout, dl) = compile(&scene);
        let mut r = rules();
        r.safe.top = 59.0;
        let report = audit_scene(&scene, &layout, &dl, &r);
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == IssueKind::SafeAreaCollision && i.node_id == "top_btn"));
    }

    #[test]
    fn detects_contrast_low_from_emitted_colors() {
        // A white text on a near-white custom surface quad.
        let text_node = Node::new(
            "low_contrast",
            NodeKind::Text {
                content: "Hello".to_string(),
                size: 14.0,
                color: Some(crate::draw::hex_rgba(0xFFFFFF, 255)),
                mono: false,
                wrap: false,
            },
        );
        let surface = Node::new(
            "card",
            NodeKind::Surface {
                style: crate::ir::BoxStyle::flat(crate::draw::hex_rgba(0xF8F8F8, 255)),
                interactive: false,
                child: Some(Box::new(text_node)),
            },
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(60.0));
        let scene = Scene::new(
            "t",
            ThemeMode::Light,
            col("root", 0.0, vec![surface]),
        );
        let (layout, dl) = compile(&scene);
        let report = audit_scene(&scene, &layout, &dl, &rules());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == IssueKind::ContrastLow && i.node_id == "low_contrast"));
    }

    #[test]
    fn detects_overflowing_text_clip() {
        let mut t = Node::new(
            "long",
            NodeKind::Text {
                content: "A very long label that cannot possibly fit in one hundred pixels".to_string(),
                size: 14.0,
                color: None,
                mono: false,
                wrap: false,
            },
        );
        t.width = SizeSpec::Px(100.0);
        t.height = SizeSpec::Px(20.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 0.0, vec![t]));
        let (layout, dl) = compile(&scene);
        let report = audit_scene(&scene, &layout, &dl, &rules());
        assert!(report.issues.iter().any(|i| {
            i.kind == IssueKind::ClippedOrOverflow && i.node_id == "long"
        }));
    }

    #[test]
    fn report_tags_interactive_nodes_against_issues() {
        let mut b = button("small", "x");
        b.width = SizeSpec::Px(30.0);
        b.height = SizeSpec::Px(30.0);
        let scene = Scene::new("t", ThemeMode::Light, col("root", 8.0, vec![b]));
        let (layout, dl) = compile(&scene);
        let r = audit_scene(&scene, &layout, &dl, &rules());
        let node = r.nodes.iter().find(|n| n.id == "small").expect("node");
        let issue = r
            .issues
            .iter()
            .find(|i| i.node_id == "small")
            .expect("issue");
        assert_eq!(node.tag, issue.tag);
        assert!(node.interactive);
    }

    #[test]
    fn glyphs_outside_their_clip_are_not_scored_for_contrast() {
        use crate::draw::{ClipRect, GlyphInstance};

        // A 400x200 window over copy that runs to y=1200: the tail is clipped
        // away, so sampling it would compare the text against the empty
        // backdrop and report a bogus ~1.18:1 failure.
        let mut dl = DrawList::new();
        dl.push_quad(QuadInstance {
            rect: [0.0, 0.0, 400.0, 1200.0],
            radii: [0.0; 4],
            color_top: 0xFFFF_FFFF,
            color_bottom: 0xFFFF_FFFF,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
        let clip = dl.push_clip(ClipRect::new(0.0, 0.0, 400.0, 200.0));
        let glyph = |y: f32| GlyphInstance {
            pos: [20.0, y],
            size: [6.0, 12.0],
            uv: [0.0, 0.0, 0.0, 0.0],
            color: 0xFF1A1A1A,
            px_range: 4.0,
            clip_id: clip,
            _pad: 0,
        };
        dl.glyphs.push(glyph(40.0)); // visible
        dl.glyphs.push(glyph(900.0)); // clipped away
        dl.pop_clip();

        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 400.0,
            h: 1200.0,
        };
        let ratio = worst_text_contrast(rect, &dl, [400.0, 1200.0])
            .expect("the visible glyph is scored");
        assert!(ratio > 10.0, "dark on white should be high contrast, got {ratio}");
    }

    #[test]
    fn overlays_safe_areas_drags_and_wrapped_copy_audit_clean_320_to_2560() {
        use crate::ir::{
            BoxStyle, DragAxis, Insets, ListViewData, RichTextData, StackAlign, TextSpan,
        };

        fn surface(id: &str, w: SizeSpec, h: SizeSpec, label: &str) -> Node {
            Node::new(
                id,
                NodeKind::Surface {
                    style: BoxStyle::default(),
                    interactive: false,
                    child: Some(Box::new(text(&format!("{id}_copy"), label))),
                },
            )
            .with_size(w, h)
        }

        for width in [320.0f32, 375.0, 768.0, 1024.0, 1440.0, 2560.0] {
            let height = (width * 1.6).clamp(480.0, 1600.0);
            let mut cta = button("cta", "Continue");
            cta.width = SizeSpec::Fill(1.0);
            cta.height = SizeSpec::Px(48.0);

            // Two layers occupy the same Stack box: that is the point of a
            // z-container, so the sibling-overlap rule must not fire for it.
            let stack = Node::new(
                "stack",
                NodeKind::Stack {
                    align: StackAlign::Center,
                    gap: 0.0,
                    style: None,
                    children: vec![
                        surface("layer_base", SizeSpec::Fill(1.0), SizeSpec::Px(132.0), "Base layer"),
                        surface("layer_top", SizeSpec::Px(160.0), SizeSpec::Px(48.0), "Chip"),
                    ],
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(132.0));

            let safe = Node::new(
                "safe",
                NodeKind::SafeArea {
                    insets: Insets {
                        top: 47.0,
                        right: 0.0,
                        bottom: 34.0,
                        left: 0.0,
                    },
                    keyboard_avoidance: true,
                    keyboard_height: 0.0,
                    child: Some(Box::new(stack)),
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Auto);

            let drag_card = surface("drag_card", SizeSpec::Fill(1.0), SizeSpec::Px(56.0), "Drag me");
            let draggable = Node::new(
                "drag",
                NodeKind::Draggable {
                    axis: DragAxis::Both,
                    offset: [0.0, 0.0],
                    inertia: true,
                    child: Some(Box::new(drag_card)),
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(56.0));

            let overlay = Node::new(
                "sheet",
                NodeKind::Overlay {
                    scrim: Some(0x99000000),
                    dismissable: true,
                    align: StackAlign::Center,
                    style: None,
                    child: Some(Box::new(cta)),
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(96.0));

            let rich = Node::new(
                "rich",
                NodeKind::RichText(RichTextData {
                    spans: vec![TextSpan {
                        text: "Wrapped copy that must never clip, ellipsize, or overlap its                                neighbouring rows at any supported width. "
                            .repeat(6),
                        color: None,
                        weight: 400,
                        mono: false,
                        size: None,
                    }],
                    size: 14.0,
                    wrap: true,
                    line_height: None,
                }),
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Auto);

            let mut list_items = Vec::new();
            for i in 0..30 {
                list_items.push(crate::ir::ListItem {
                    id: format!("row{i}"),
                    height: Some(48.0),
                    child: surface(
                        &format!("row{i}_card"),
                        SizeSpec::Fill(1.0),
                        SizeSpec::Px(48.0),
                        &format!("Row {i}"),
                    ),
                });
            }
            let list = Node::new(
                "list",
                NodeKind::ListView(ListViewData {
                    items: list_items,
                    estimated_height: 48.0,
                    spacing: 2.0,
                    overscan: 0.0,
                    style: None,
                }),
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(220.0));

            let mut root = col("root", 12.0, vec![safe, draggable, overlay, rich, list]);
            root.padding = Insets::all(12.0);
            let scene = Scene::new("Responsive", ThemeMode::Light, root);

            let atlas = crate::text::GlyphAtlas::builtin();
            let mut engine = crate::compiler::UiEngine::new(scene.clone(), [width, height]);
            let mut dl = DrawList::new();
            engine.tick_and_compile(1.0 / 60.0, &mut dl);

            let rules = AuditRules {
                platform: "ios".to_string(),
                viewport: [width, height],
                touch: true,
                min_touch: 44.0,
                ..AuditRules::default()
            };
            let report = audit_scene_with_atlas(
                &scene,
                &engine.layout,
                &dl,
                &rules,
                &std::collections::HashMap::new(),
                &atlas,
            );
            let errors: Vec<_> = report
                .issues
                .iter()
                .filter(|i| i.severity == Severity::Error)
                .collect();
            assert!(
                errors.is_empty(),
                "width {width}: audit errors {errors:?}"
            );
        }
    }
}
