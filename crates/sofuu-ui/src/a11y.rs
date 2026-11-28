//! Accessibility semantic tree (`A11yNode`) for native screen readers.
//!
//! The renderer never owns platform accessibility objects: this module exports a
//! compact JSON tree that hosts map onto `NSAccessibility` / `UIAccessibility` /
//! Android `AccessibilityNodeInfo`. Rects come from the last solved
//! [`crate::layout::LayoutMap`], so VoiceOver/TalkBack hitboxes always agree with
//! what is painted.

use serde::{Deserialize, Serialize};

use crate::hit_test::InteractionState;
use crate::ir::{ListViewData, Node, NodeKind, Scene};
use crate::layout::LayoutMap;

/// One node in the exported accessibility tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct A11yNode {
    pub id: String,
    /// Platform-agnostic role: `button`, `text_field`, `slider`, `image`, …
    pub role: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// `[x, y, w, h]` in logical pixels, matching the painted rect.
    pub rect: [f32; 4],
    /// Soft-keyboard type for editable nodes (`text`, `email`, `number`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    pub enabled: bool,
    pub focusable: bool,
    pub focused: bool,
    pub children: Vec<A11yNode>,
}

impl A11yNode {
    fn group(id: &str, role: &str, label: impl Into<String>, rect: [f32; 4]) -> Self {
        Self {
            id: id.to_string(),
            role: role.to_string(),
            label: label.into(),
            value: None,
            hint: None,
            rect,
            input_type: None,
            enabled: true,
            focusable: false,
            focused: false,
            children: Vec::new(),
        }
    }
}

/// Build the semantic tree for the current scene/layout/interaction state.
pub fn scene_tree(scene: &Scene, layout: &LayoutMap, interaction: &InteractionState) -> A11yNode {
    build(&scene.root, layout, interaction)
}

/// Serialize [`scene_tree`] for the C ABI (`serde_json` object).
pub fn scene_tree_json(
    scene: &Scene,
    layout: &LayoutMap,
    interaction: &InteractionState,
) -> String {
    serde_json::to_string(&scene_tree(scene, layout, interaction)).unwrap_or_else(|_| "{}".into())
}

fn rect_of(id: &str, layout: &LayoutMap) -> [f32; 4] {
    layout
        .get(id)
        .map(|r| [r.x, r.y, r.w, r.h])
        .unwrap_or([0.0; 4])
}

fn focused_id(interaction: &InteractionState) -> Option<&str> {
    interaction
        .focused_key
        .as_deref()
        .and_then(|k| k.split_once("::").map(|(_, id)| id))
}

fn build(node: &Node, layout: &LayoutMap, interaction: &InteractionState) -> A11yNode {
    let rect = rect_of(&node.id, layout);
    let focused = focused_id(interaction) == Some(node.id.as_str());
    let mut a = match &node.kind {
        NodeKind::Container { children, .. }
        | NodeKind::Grid { children, .. }
        | NodeKind::Stack { children, .. } => {
            let mut n = A11yNode::group(&node.id, "group", "", rect);
            n.children = children
                .iter()
                .filter(|c| c.visible)
                .map(|c| build(c, layout, interaction))
                .collect();
            n
        }
        NodeKind::Surface {
            interactive, child, ..
        } => {
            let mut n = A11yNode::group(
                &node.id,
                if *interactive { "button" } else { "group" },
                collect_text(node),
                rect,
            );
            n.focusable = *interactive;
            if let Some(c) = child {
                n.children.push(build(c, layout, interaction));
            }
            n
        }
        NodeKind::Tabs { items, active } => {
            let label = items
                .get(*active)
                .map(|i| i.label.clone())
                .unwrap_or_default();
            A11yNode::group(&node.id, "tab_list", label, rect)
        }
        NodeKind::Text {
            content, wrap: _, ..
        } => A11yNode::group(&node.id, "text", content.clone(), rect),
        NodeKind::RichText(data) => A11yNode::group(
            &node.id,
            "text",
            data.spans.iter().map(|s| s.text.as_str()).collect::<String>(),
            rect,
        ),
        NodeKind::CodeText(data) => A11yNode::group(&node.id, "text", data.content.clone(), rect),
        NodeKind::Rule { .. } | NodeKind::Spacer => {
            let mut n = A11yNode::group(&node.id, "separator", "", rect);
            n.enabled = false;
            n
        }
        NodeKind::ThemeToggle { .. } => {
            let mut n = A11yNode::group(&node.id, "button", "Toggle theme", rect);
            n.focusable = true;
            n
        }
        NodeKind::Segmented { options, active } => {
            let label = options.get(*active).cloned().unwrap_or_default();
            A11yNode::group(&node.id, "segmented_control", label, rect)
        }
        NodeKind::Button { label, .. } => {
            let mut n = A11yNode::group(&node.id, "button", label.clone(), rect);
            n.focusable = true;
            n
        }
        NodeKind::Switch {
            checked, label, ..
        } => {
            let mut n = A11yNode::group(
                &node.id,
                "switch",
                label.clone().unwrap_or_default(),
                rect,
            );
            n.value = Some(if *checked { "on" } else { "off" }.into());
            n.focusable = true;
            n
        }
        NodeKind::Input {
            value,
            placeholder,
            keyboard,
            ..
        } => {
            let mut n = A11yNode::group(&node.id, "text_field", placeholder.clone(), rect);
            n.value = Some(value.clone());
            n.input_type = Some(keyboard.as_str().into());
            n.focusable = true;
            n.focused = focused;
            n
        }
        NodeKind::TextArea(data) => {
            let mut n = A11yNode::group(&node.id, "text_area", data.placeholder.clone(), rect);
            n.value = Some(if data.secure {
                "•".repeat(data.value.chars().count())
            } else {
                data.value.clone()
            });
            n.input_type = Some(data.keyboard.as_str().into());
            n.focusable = true;
            n.focused = focused;
            n
        }
        NodeKind::Slider(data) => {
            let mut n = A11yNode::group(
                &node.id,
                "slider",
                data.label.clone().unwrap_or_default(),
                rect,
            );
            n.value = Some(format!("{}", data.value));
            n.focusable = true;
            n.focused = focused;
            n
        }
        NodeKind::RangeSlider(data) => {
            let (low, high) = data.clamped();
            let mut n = A11yNode::group(
                &node.id,
                "range_slider",
                data.label.clone().unwrap_or_default(),
                rect,
            );
            n.value = Some(format!("{low} – {high}"));
            n.focusable = true;
            n.focused = focused;
            n
        }
        NodeKind::Progress(data) => {
            let mut n = A11yNode::group(
                &node.id,
                "progress_bar",
                data.label.clone().unwrap_or_default(),
                rect,
            );
            n.value = Some(if data.indeterminate {
                "indeterminate".to_string()
            } else {
                format!("{}%", (data.fraction() * 100.0).round() as i64)
            });
            n
        }
        NodeKind::Stat(data) => {
            let mut n = A11yNode::group(&node.id, "text", data.label.clone(), rect);
            n.value = Some(
                data.value_text
                    .clone()
                    .unwrap_or_else(|| format!("{}{}{}", data.prefix, data.value, data.suffix)),
            );
            n
        }
        NodeKind::Plot(data) => {
            let label = data.title.clone().unwrap_or_else(|| "Chart".into());
            A11yNode::group(&node.id, "image", label, rect)
        }
        NodeKind::Table(data) => {
            let label = data.title.clone().unwrap_or_else(|| "Table".into());
            A11yNode::group(&node.id, "table", label, rect)
        }
        NodeKind::Feed(data) => {
            let label = data.title.clone().unwrap_or_else(|| "Feed".into());
            A11yNode::group(&node.id, "list", label, rect)
        }
        NodeKind::Graph(data) => {
            let label = data.title.clone().unwrap_or_else(|| "Graph".into());
            A11yNode::group(&node.id, "image", label, rect)
        }
        NodeKind::Icon { name, .. } => {
            let mut n = A11yNode::group(&node.id, "image", name.clone(), rect);
            n.enabled = false;
            n
        }
        NodeKind::Image { src, alt, .. } => {
            let label = if alt.is_empty() { src.clone() } else { alt.clone() };
            A11yNode::group(&node.id, "image", label, rect)
        }
        NodeKind::SafeArea { child, .. } => {
            let mut n = A11yNode::group(&node.id, "group", "", rect);
            if let Some(c) = child {
                n.children.push(build(c, layout, interaction));
            }
            n
        }
        NodeKind::Overlay {
            dismissable, child, ..
        } => {
            let mut n = A11yNode::group(&node.id, "dialog", collect_text(node), rect);
            n.enabled = *dismissable;
            if let Some(c) = child {
                n.children.push(build(c, layout, interaction));
            }
            n
        }
        NodeKind::NavStack { routes, active, .. } => {
            let label = routes
                .get(*active)
                .map(|r| r.title.clone())
                .unwrap_or_default();
            let mut n = A11yNode::group(&node.id, "navigation", label, rect);
            if let Some(route) = routes.get(*active) {
                n.children.push(build(&route.child, layout, interaction));
            }
            n
        }
        NodeKind::ListView(data) => {
            let mut n = A11yNode::group(&node.id, "list", "", rect);
            n.children = list_items(data, layout, interaction);
            n
        }
        NodeKind::Pressable {
            disabled, child, ..
        } => {
            let mut n = A11yNode::group(&node.id, "button", collect_text(node), rect);
            n.enabled = !*disabled;
            n.focusable = !*disabled;
            if let Some(c) = child {
                n.children.push(build(c, layout, interaction));
            }
            n
        }
        NodeKind::Draggable { child, .. } => {
            let mut n = A11yNode::group(&node.id, "adjustable", collect_text(node), rect);
            n.focusable = true;
            if let Some(c) = child {
                n.children.push(build(c, layout, interaction));
            }
            n
        }
    };
    if node.min_width > 0.0 || node.min_height > 0.0 {
        a.hint = Some(format!(
            "min {}x{}",
            node.min_width.round(),
            node.min_height.round()
        ));
    }
    a
}

fn list_items(
    data: &ListViewData,
    layout: &LayoutMap,
    interaction: &InteractionState,
) -> Vec<A11yNode> {
    data.items
        .iter()
        .filter_map(|item| {
            let r = layout.get(&item.child.id)?;
            if r.h <= 0.0 {
                return None;
            }
            let mut n = build(&item.child, layout, interaction);
            n.id = item.id.clone();
            n.role = "list_item".into();
            Some(n)
        })
        .collect()
}

/// Best-effort label for composite nodes: concatenate descendant text.
fn collect_text(node: &Node) -> String {
    let mut out = String::new();
    collect_text_into(node, &mut out);
    out
}

fn collect_text_into(node: &Node, out: &mut String) {
    match &node.kind {
        NodeKind::Text { content, .. } => push_part(out, content),
        NodeKind::RichText(data) => {
            for span in &data.spans {
                push_part(out, &span.text);
            }
        }
        NodeKind::Button { label, .. } => push_part(out, label),
        NodeKind::Switch { label, .. } => {
            if let Some(l) = label {
                push_part(out, l);
            }
        }
        NodeKind::Input { placeholder, .. } => push_part(out, placeholder),
        NodeKind::TextArea(data) => push_part(out, &data.placeholder),
        NodeKind::Slider(data) => {
            if let Some(l) = &data.label {
                push_part(out, l);
            }
        }
        NodeKind::RangeSlider(data) => {
            if let Some(l) = &data.label {
                push_part(out, l);
            }
        }
        NodeKind::Progress(data) => {
            if let Some(l) = &data.label {
                push_part(out, l);
            }
        }
        NodeKind::Icon { name, .. } => push_part(out, name),
        NodeKind::Image { alt, .. } => push_part(out, alt),
        NodeKind::Container { children, .. }
        | NodeKind::Grid { children, .. }
        | NodeKind::Stack { children, .. } => {
            for c in children {
                collect_text_into(c, out);
            }
        }
        NodeKind::Tabs { items, active } => {
            if let Some(i) = items.get(*active) {
                push_part(out, &i.label);
            }
        }
        NodeKind::NavStack { routes, active, .. } => {
            if let Some(r) = routes.get(*active) {
                push_part(out, &r.title);
            }
        }
        NodeKind::ListView(data) => {
            for i in &data.items {
                collect_text_into(&i.child, out);
            }
        }
        NodeKind::Surface { child, .. }
        | NodeKind::SafeArea { child, .. }
        | NodeKind::Overlay { child, .. }
        | NodeKind::Pressable { child, .. }
        | NodeKind::Draggable { child, .. } => {
            if let Some(c) = child {
                collect_text_into(c, out);
            }
        }
        _ => {}
    }
}

fn push_part(out: &mut String, part: &str) {
    if part.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(part);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Axis, Insets, TextAreaData};
    use crate::text::GlyphAtlas;
    use crate::theme::ThemeMode;

    fn sample() -> Scene {
        Scene::new(
            "A11y",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: crate::ir::Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![
                        Node::new(
                            "save",
                            NodeKind::Button {
                                label: "Save".into(),
                                tone: crate::theme::Tone::Neutral,
                            },
                        ),
                        Node::new(
                            "name",
                            NodeKind::Input {
                                value: "Ada".into(),
                                placeholder: "Name".into(),
                                tone: crate::theme::Tone::Neutral,
                                keyboard: crate::ir::KeyboardHint::Text,
                                prefix_icon: None,
                                clearable: false,
                                error: None,
                            },
                        ),
                        Node::new(
                            "note",
                            NodeKind::TextArea(TextAreaData {
                                value: "hello".into(),
                                placeholder: "Note".into(),
                                rows: 2,
                                tone: crate::theme::Tone::Neutral,
                                secure: false,
                                max_len: 0,
                                keyboard: crate::ir::KeyboardHint::Text,
                            }),
                        ),
                        Node::new(
                            "vol",
                            NodeKind::Slider(crate::ir::SliderData {
                                value: 30.0,
                                min: 0.0,
                                max: 100.0,
                                step: 10.0,
                                tone: crate::theme::Tone::Neutral,
                                label: Some("Volume".into()),
                                show_value: true,
                            }),
                        ),
                    ],
                },
            )
            .with_padding(Insets::all(12.0)),
        )
    }

    #[test]
    fn roles_values_focus_and_rects_match_the_scene() {
        let scene = sample();
        let atlas = GlyphAtlas::builtin();
        let mut layout = LayoutMap::default();
        layout.solve(&scene.root, [400.0, 500.0], &atlas);
        let mut interaction = InteractionState::default();
        interaction.focused_key = Some("input::name".into());

        let tree = scene_tree(&scene, &layout, &interaction);
        assert_eq!(tree.role, "group");
        assert_eq!(tree.children.len(), 4);

        let save = &tree.children[0];
        assert_eq!((save.role.as_str(), save.label.as_str()), ("button", "Save"));
        assert!(save.focusable);
        assert!(save.rect[2] > 0.0);

        let name = &tree.children[1];
        assert_eq!(name.role, "text_field");
        assert_eq!(name.value.as_deref(), Some("Ada"));
        assert!(name.focused);

        let note = &tree.children[2];
        assert_eq!(note.role, "text_area");
        assert_eq!(note.value.as_deref(), Some("hello"));

        let vol = &tree.children[3];
        assert_eq!(vol.role, "slider");
        assert_eq!(vol.label, "Volume");
        assert_eq!(vol.value.as_deref(), Some("30"));

        let json = scene_tree_json(&scene, &layout, &interaction);
        assert!(json.contains("\"role\":\"slider\""));
        assert!(json.contains("\"focused\":true"));
    }

    #[test]
    fn secure_fields_never_export_plaintext() {
        let mut scene = sample();
        if let Some(node) = scene.root.find_mut("note") {
            if let NodeKind::TextArea(data) = &mut node.kind {
                data.secure = true;
                data.value = "secret".into();
            }
        }
        let atlas = GlyphAtlas::builtin();
        let mut layout = LayoutMap::default();
        layout.solve(&scene.root, [400.0, 500.0], &atlas);
        let tree = scene_tree(&scene, &layout, &InteractionState::default());
        let note = &tree.children[2];
        assert_eq!(note.value.as_deref(), Some("••••••"));
        assert!(!serde_json::to_string(note).unwrap().contains("secret"));
    }

    #[test]
    fn hidden_nodes_are_skipped() {
        let mut scene = sample();
        if let Some(node) = scene.root.find_mut("save") {
            node.visible = false;
        }
        let atlas = GlyphAtlas::builtin();
        let mut layout = LayoutMap::default();
        layout.solve(&scene.root, [400.0, 500.0], &atlas);
        let tree = scene_tree(&scene, &layout, &InteractionState::default());
        assert!(tree.children.iter().all(|c| c.id != "save"));
    }
}
