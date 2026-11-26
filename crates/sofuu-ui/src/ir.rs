//! Domain-agnostic Scene IR (retained node tree + incremental patches).
//!
//! Layout containers (`Container`, `Grid`, `Tabs`), base structural surfaces (`Surface`, `Text`,
//! `Rule`, `Spacer`), and patch commands sit here without any baked-in domain schemas.

use serde::{Deserialize, Serialize};

use crate::theme::ThemeMode;

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
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct BoxStyle {
    pub bg: Option<u32>,
    pub bg_bottom: Option<u32>,
    pub hover_bg: Option<u32>,
    pub border: Option<u32>,
    pub border_width: f32,
    pub radius: [f32; 4],
    pub shadow_blur: f32,
    pub shadow_offset_y: f32,
    pub opacity: f32,
}

impl BoxStyle {
    pub fn card(bg: u32, border: u32, radius: f32) -> Self {
        Self {
            bg: Some(bg),
            bg_bottom: None,
            hover_bg: None,
            border: Some(border),
            border_width: 1.0,
            radius: [radius; 4],
            shadow_blur: 14.0,
            shadow_offset_y: 4.0,
            opacity: 1.0,
        }
    }

    pub fn flat(bg: u32) -> Self {
        Self {
            bg: Some(bg),
            bg_bottom: None,
            hover_bg: None,
            border: None,
            border_width: 0.0,
            radius: [0.0; 4],
            shadow_blur: 0.0,
            shadow_offset_y: 0.0,
            opacity: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TabItem {
    pub label: String,
    pub child: Node,
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
            NodeKind::Text { .. } | NodeKind::Rule { .. } | NodeKind::Spacer => {}
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
        }
    }
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
    ReplaceNode { id: String, node: Node },
}
