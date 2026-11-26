//! Scene compiler: turns Scene IR + Layout + 120Hz Springs into flat GPU `DrawList` buffers.

use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::hit_test::{HitKind, InputEvent, InteractionState, KineticScroll, UIEvent};
use crate::ir::{BoxStyle, Node, NodeKind, Scene, ScenePatch};
use crate::layout::{ComputedRect, LayoutMap};
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::Theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileStatus {
    /// Whether `DrawList` was rebuilt this tick and needs GPU presentation.
    pub compiled: bool,
    /// Whether any spring or kinetic scroll is still in motion (requests next vsync tick).
    pub animating: bool,
}

/// Core 2D GPU UI engine state container (independent of OS windowing & GPU backend).
pub struct UiEngine {
    pub scene: Scene,
    pub theme: Theme,
    pub atlas: GlyphAtlas,
    pub layout: LayoutMap,
    pub springs: SpringBank,
    pub interaction: InteractionState,
    viewport: [f32; 2],
    dirty: bool,
}

impl UiEngine {
    pub fn new(scene: Scene, viewport: [f32; 2]) -> Self {
        let theme = Theme::for_mode(scene.theme);
        Self {
            scene,
            theme,
            atlas: GlyphAtlas::builtin(),
            layout: LayoutMap::default(),
            springs: SpringBank::new(),
            interaction: InteractionState::default(),
            viewport,
            dirty: true,
        }
    }

    pub fn viewport(&self) -> [f32; 2] {
        self.viewport
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty_or_animating(&self) -> bool {
        self.dirty
            || self.springs.has_active()
            || self
                .interaction
                .scrolls
                .values()
                .any(|s| s.velocity_y.abs() >= 4.0)
    }

    pub fn set_scene(&mut self, scene: Scene) {
        self.theme = Theme::for_mode(scene.theme);
        self.scene = scene;
        self.dirty = true;
    }

    pub fn apply_patch(&mut self, patch: ScenePatch) -> bool {
        let changed = self.scene.apply_patch(patch);
        if changed {
            self.theme = Theme::for_mode(self.scene.theme);
            self.dirty = true;
        }
        changed
    }

    /// Route raw OS input into interaction springs, kinetic scrolls, and semantic `UIEvent`s.
    pub fn handle_input(&mut self, event: InputEvent) -> Vec<UIEvent> {
        let mut out = Vec::new();
        match event {
            InputEvent::Resize { width, height, .. } => {
                let w = width.max(1.0);
                let h = height.max(1.0);
                if (self.viewport[0] - w).abs() > 0.25 || (self.viewport[1] - h).abs() > 0.25 {
                    self.viewport = [w, h];
                    self.dirty = true;
                }
            }
            InputEvent::PointerMove { x, y } => {
                self.interaction.pointer = [x, y];
                let next_hover = self
                    .interaction
                    .hit_topmost_interactive(x, y)
                    .map(|(k, _)| k);
                if next_hover != self.interaction.hovered_key {
                    self.interaction.hovered_key = next_hover;
                    self.dirty = true;
                }
            }
            InputEvent::PointerDown { x, y, button } => {
                self.interaction.pointer = [x, y];
                if button == 0 {
                    let hit = self
                        .interaction
                        .hit_topmost_interactive(x, y)
                        .map(|(k, _)| k);
                    if hit != self.interaction.pressed_key {
                        self.interaction.pressed_key = hit;
                        self.dirty = true;
                    }
                }
            }
            InputEvent::PointerUp { x, y, button } => {
                self.interaction.pointer = [x, y];
                if button == 0 {
                    let pressed = self.interaction.pressed_key.take();
                    if pressed.is_some() {
                        self.dirty = true;
                    }
                    if let Some(pressed_key) = pressed {
                        if let Some((hit_key, kind)) = self.interaction.hit_topmost_interactive(x, y)
                        {
                            if hit_key == pressed_key {
                                match kind.clone() {
                                    HitKind::Surface { id } => {
                                        out.push(UIEvent::Click { id });
                                    }
                                    HitKind::TabButton { tabs_id, index } => {
                                        let _ = self.apply_patch(ScenePatch::SetTab {
                                            id: tabs_id.clone(),
                                            active: index,
                                        });
                                        out.push(UIEvent::TabChanged { id: tabs_id, index });
                                    }
                                    HitKind::ScrollContainer { .. } => {}
                                }
                            }
                        }
                    }
                }
            }
            InputEvent::Scroll { x, y, delta_y, .. } => {
                if let Some((scroll_id, max_y)) = self.interaction.hit_topmost_scroll(x, y) {
                    let entry = self
                        .interaction
                        .scrolls
                        .entry(scroll_id.clone())
                        .or_insert_with(|| KineticScroll {
                            offset_y: 0.0,
                            velocity_y: 0.0,
                            max_y,
                        });
                    entry.max_y = max_y;
                    entry.impulse(delta_y);
                    self.dirty = true;
                    out.push(UIEvent::Scrolled {
                        id: scroll_id,
                        offset_y: entry.offset_y,
                    });
                }
            }
            InputEvent::CloseRequested => {
                out.push(UIEvent::WindowClosed);
            }
            InputEvent::KeyDown { .. } | InputEvent::TextInput { .. } => {}
        }
        out
    }

    /// Advance 120Hz springs & kinetic scrolls and compile `DrawList` when dirty or animating.
    pub fn tick_and_compile(&mut self, dt: f32, draw_list: &mut DrawList) -> CompileStatus {
        let scroll_moving = self.interaction.step_scrolls(dt);
        let spring_moving = self.springs.step_all(dt);

        if !self.dirty && !scroll_moving && !spring_moving {
            return CompileStatus {
                compiled: false,
                animating: false,
            };
        }

        self.dirty = false;
        self.layout.solve(&self.scene.root, self.viewport, &self.atlas);
        self.interaction.clear_hits();
        self.springs.begin_frame();
        draw_list.reset(self.viewport);

        // Paint root window background quad.
        draw_list.push_quad(QuadInstance {
            rect: [0.0, 0.0, self.viewport[0], self.viewport[1]],
            radii: [0.0; 4],
            color_top: self.theme.bg,
            color_bottom: self.theme.bg,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });

        let root_node = self.scene.root.clone();
        self.emit_node(&root_node, 0.0, 0.0, draw_list);
        self.springs.prune_unvisited();

        let still_animating = self.springs.has_active()
            || self
                .interaction
                .scrolls
                .values()
                .any(|s| s.velocity_y.abs() >= 4.0);

        CompileStatus {
            compiled: true,
            animating: still_animating,
        }
    }

    fn emit_node(&mut self, node: &Node, ox: f32, oy: f32, dl: &mut DrawList) {
        if !node.visible {
            return;
        }
        let Some(mut rect) = self.layout.get(&node.id) else {
            return;
        };
        rect.x += ox;
        rect.y += oy;

        match &node.kind {
            NodeKind::Container {
                scrollable,
                style,
                children,
                ..
            } => {
                if let Some(s) = style {
                    self.emit_box(rect, s, 0.0, dl);
                }
                let max_scroll = if *scrollable {
                    (rect.content_h - rect.h).max(0.0)
                } else {
                    0.0
                };
                if *scrollable {
                    self.interaction.push_hit(
                        rect,
                        HitKind::ScrollContainer {
                            id: node.id.clone(),
                            max_scroll_y: max_y_clamp(max_scroll),
                        },
                    );
                }
                let scroll_y = if *scrollable {
                    let s = self
                        .interaction
                        .scrolls
                        .entry(node.id.clone())
                        .or_default();
                    s.max_y = max_scroll;
                    s.offset_y = s.offset_y.clamp(0.0, max_scroll);
                    s.offset_y
                } else {
                    0.0
                };

                if *scrollable {
                    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                }
                for child in children {
                    self.emit_node(child, ox, oy - scroll_y, dl);
                }
                if *scrollable {
                    // Subtle overlay scrollbar thumb when content overflows.
                    if max_scroll > 1.0 && rect.content_h > rect.h {
                        let track_h = (rect.h - 12.0).max(24.0);
                        let thumb_h = ((rect.h / rect.content_h) * track_h).clamp(24.0, track_h);
                        let thumb_t = scroll_y / max_scroll;
                        let thumb_y = rect.y + 6.0 + thumb_t * (track_h - thumb_h);
                        dl.push_quad(QuadInstance {
                            rect: [rect.x + rect.w - 6.0, thumb_y, 3.5, thumb_h],
                            radii: [2.0; 4],
                            color_top: scale_alpha(self.theme.text_muted, 0.55),
                            color_bottom: scale_alpha(self.theme.text_muted, 0.55),
                            border_color: 0,
                            shadow_color: 0,
                            params: [0.0, 0.0, 0.0, 1.0],
                            clip_id: 0,
                            _pad: [0; 3],
                        });
                    }
                    dl.pop_clip();
                }
            }
            NodeKind::Grid {
                style, children, ..
            } => {
                if let Some(s) = style {
                    self.emit_box(rect, s, 0.0, dl);
                }
                for child in children {
                    self.emit_node(child, ox, oy, dl);
                }
            }
            NodeKind::Surface {
                style,
                interactive,
                child,
            } => {
                let hover_t = if *interactive {
                    let is_hovered = self.interaction.hovered_key.as_deref() == Some(node.id.as_str());
                    let is_pressed = self.interaction.pressed_key.as_deref() == Some(node.id.as_str());
                    let target = if is_pressed {
                        1.0
                    } else if is_hovered {
                        0.65
                    } else {
                        0.0
                    };
                    self.springs
                        .animate(&node.id, "hover", target, SpringConfig::SNAPPY)
                } else {
                    0.0
                };
                self.emit_box(rect, style, hover_t, dl);
                if *interactive {
                    self.interaction.push_hit(
                        rect,
                        HitKind::Surface {
                            id: node.id.clone(),
                        },
                    );
                }
                if let Some(c) = child {
                    self.emit_node(c, ox, oy, dl);
                }
            }
            NodeKind::Tabs { active, items } => {
                let ix = rect.x + node.padding.left;
                let iy = rect.y + node.padding.top;
                let iw = (rect.w - node.padding.horizontal()).max(0.0);
                let bar_h = 38.0;

                // Tab bar background rail
                dl.push_quad(QuadInstance {
                    rect: [ix, iy, iw, bar_h],
                    radii: [self.theme.radius_control; 4],
                    color_top: self.theme.surface_0,
                    color_bottom: self.theme.surface_0,
                    border_color: self.theme.border,
                    shadow_color: 0,
                    params: [1.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });

                if !items.is_empty() {
                    let count = items.len();
                    let pad = 4.0;
                    let pill_w = ((iw - pad * 2.0) / count as f32).max(20.0);
                    let active_clamped = (*active).min(count - 1);
                    let target_pill_x = ix + pad + active_clamped as f32 * pill_w;
                    let anim_pill_x = self.springs.animate(
                        &node.id,
                        "tab_pill_x",
                        target_pill_x,
                        SpringConfig::UI_DEFAULT,
                    );

                    // Animated active pill
                    dl.push_quad(QuadInstance {
                        rect: [anim_pill_x, iy + pad, pill_w, bar_h - pad * 2.0],
                        radii: [self.theme.radius_control - 2.0; 4],
                        color_top: self.theme.surface_2,
                        color_bottom: self.theme.surface_2,
                        border_color: self.theme.border_focus,
                        shadow_color: 0,
                        params: [1.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });

                    for (idx, item) in items.iter().enumerate() {
                        let btn_x = ix + pad + idx as f32 * pill_w;
                        let btn_rect = ComputedRect {
                            x: btn_x,
                            y: iy + pad,
                            w: pill_w,
                            h: bar_h - pad * 2.0,
                            content_h: bar_h,
                            folded: false,
                        };
                        self.interaction.push_hit(
                            btn_rect,
                            HitKind::TabButton {
                                tabs_id: node.id.clone(),
                                index: idx,
                            },
                        );

                        let (tw, th) = self.atlas.measure(&item.label, 13.0, false);
                        let tx = btn_x + (pill_w - tw) * 0.5;
                        let ty = iy + (bar_h - th) * 0.5 + 1.0;
                        let color = if idx == active_clamped {
                            self.theme.text_primary
                        } else {
                            self.theme.text_secondary
                        };
                        let clip = dl.current_clip_id();
                        self.atlas.layout_into(
                            &item.label,
                            tx,
                            ty,
                            13.0,
                            color,
                            false,
                            Some(pill_w - 8.0),
                            &mut dl.glyphs,
                            clip,
                        );
                    }

                    if let Some(active_item) = items.get(active_clamped) {
                        self.emit_node(&active_item.child, ox, oy, dl);
                    }
                }
            }
            NodeKind::Text {
                content,
                size,
                color,
                mono,
                wrap,
            } => {
                let tx = rect.x + node.padding.left;
                let ty = rect.y + node.padding.top;
                let max_w = if *wrap {
                    Some((rect.w - node.padding.horizontal()).max(16.0))
                } else {
                    None
                };
                let clip = dl.current_clip_id();
                self.atlas.layout_into(
                    content,
                    tx,
                    ty,
                    *size,
                    color.unwrap_or(self.theme.text_primary),
                    *mono,
                    max_w,
                    &mut dl.glyphs,
                    clip,
                );
            }
            NodeKind::Rule { thickness, color } => {
                let c = color.unwrap_or(self.theme.border);
                dl.push_quad(QuadInstance {
                    rect: [rect.x, rect.y, rect.w, (*thickness).max(1.0)],
                    radii: [0.0; 4],
                    color_top: c,
                    color_bottom: c,
                    border_color: 0,
                    shadow_color: 0,
                    params: [0.0, 0.0, 0.0, 1.0],
                    clip_id: 0,
                    _pad: [0; 3],
                });
            }
            NodeKind::Spacer => {}
        }
    }

    fn emit_box(&self, rect: ComputedRect, style: &BoxStyle, hover_t: f32, dl: &mut DrawList) {
        let base_top = style.bg.unwrap_or(self.theme.surface_1);
        let base_bot = style.bg_bottom.unwrap_or(base_top);
        let hover_target = style.hover_bg.unwrap_or(self.theme.surface_2);
        let top = if hover_t > 0.0 {
            lerp_rgba8(base_top, hover_target, hover_t)
        } else {
            base_top
        };
        let bot = if hover_t > 0.0 {
            lerp_rgba8(base_bot, hover_target, hover_t)
        } else {
            base_bot
        };
        let border = if hover_t > 0.0 {
            lerp_rgba8(
                style.border.unwrap_or(self.theme.border),
                self.theme.border_focus,
                hover_t * 0.8,
            )
        } else {
            style.border.unwrap_or(0)
        };
        let shadow = if style.shadow_blur > 0.0 {
            self.theme.shadow
        } else {
            0
        };
        dl.push_quad(QuadInstance {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radii: style.radius,
            color_top: top,
            color_bottom: bot,
            border_color: border,
            shadow_color: shadow,
            params: [
                style.border_width,
                style.shadow_blur,
                style.shadow_offset_y,
                if style.opacity <= 0.0 { 1.0 } else { style.opacity },
            ],
            clip_id: 0,
            _pad: [0; 3],
        });
    }
}

#[inline]
fn max_y_clamp(v: f32) -> f32 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Align, Axis, Insets, SizeSpec, TabItem};
    use crate::theme::ThemeMode;

    fn sample_scene() -> Scene {
        let t = Theme::dark();
        Scene::new(
            "Sofuu Base",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 12.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![
                        Node::new(
                            "title",
                            NodeKind::Text {
                                content: "Sofuu UI Base".into(),
                                size: 20.0,
                                color: None,
                                mono: false,
                                wrap: false,
                            },
                        ),
                        Node::new(
                            "tabs",
                            NodeKind::Tabs {
                                active: 0,
                                items: vec![
                                    TabItem {
                                        label: "Overview".into(),
                                        child: Node::new(
                                            "card_1",
                                            NodeKind::Surface {
                                                style: BoxStyle::card(
                                                    t.surface_1,
                                                    t.border,
                                                    t.radius_card,
                                                ),
                                                interactive: true,
                                                child: Some(Box::new(Node::new(
                                                    "card_text",
                                                    NodeKind::Text {
                                                        content: "Ready".into(),
                                                        size: 14.0,
                                                        color: None,
                                                        mono: false,
                                                        wrap: false,
                                                    },
                                                ))),
                                            },
                                        )
                                        .with_padding(Insets::all(16.0))
                                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(120.0)),
                                    },
                                    TabItem {
                                        label: "Events".into(),
                                        child: Node::new(
                                            "tab2_text",
                                            NodeKind::Text {
                                                content: "Second Tab".into(),
                                                size: 14.0,
                                                color: None,
                                                mono: false,
                                                wrap: false,
                                            },
                                        ),
                                    },
                                ],
                            },
                        )
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
                    ],
                },
            )
            .with_padding(Insets::all(16.0)),
        )
    }

    #[test]
    fn engine_sleeps_when_idle_and_wakes_on_tab_click_with_spring_animation() {
        let mut engine = UiEngine::new(sample_scene(), [960.0, 640.0]);
        let mut dl = DrawList::new();

        // First frame compiles full draw list and settles cleanly.
        let s1 = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(s1.compiled);
        assert!(!s1.animating);
        assert!(!dl.quads.is_empty());
        assert!(!dl.glyphs.is_empty());

        // Idle second frame skips work completely (0% CPU/GPU).
        let s2 = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(!s2.compiled);
        assert!(!s2.animating);

        // Click the second tab in the tab bar (around x = 700, y = 65).
        let _ = engine.handle_input(InputEvent::PointerDown {
            x: 700.0,
            y: 68.0,
            button: 0,
        });
        let events = engine.handle_input(InputEvent::PointerUp {
            x: 700.0,
            y: 68.0,
            button: 0,
        });
        assert_eq!(
            events,
            vec![UIEvent::TabChanged {
                id: "tabs".into(),
                index: 1
            }]
        );

        // Spring animation is now active for the tab pill!
        let s3 = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(s3.compiled);
        assert!(s3.animating);

        // Run until spring settles back to 0% idle.
        let mut settled = false;
        for _ in 0..240 {
            let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
            if !st.animating {
                settled = true;
                break;
            }
        }
        assert!(settled);
    }

    #[test]
    fn incremental_patch_updates_text_without_losing_structure() {
        let mut engine = UiEngine::new(sample_scene(), [800.0, 500.0]);
        assert!(engine.apply_patch(ScenePatch::SetText {
            id: "card_text".into(),
            content: "Patched OK".into(),
        }));
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
    }
}
