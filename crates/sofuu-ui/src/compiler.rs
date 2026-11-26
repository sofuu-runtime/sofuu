//! Scene compiler: turns Scene IR + Layout + 120Hz Springs into flat GPU `DrawList` buffers.

use crate::components::controls::{emit_button, emit_segmented, emit_tabs_bar, emit_theme_toggle};
use crate::components::plot::emit_plot;
use crate::components::stat::emit_stat;
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::hit_test::{HitKind, InputEvent, InteractionState, KineticScroll, UIEvent};
use crate::ir::{BoxStyle, Node, NodeKind, Scene, ScenePatch};
use crate::layout::{child_path_hash, ComputedRect, LayoutMap, ROOT_PATH_HASH};
use crate::spring::{SpringBank, SpringConfig};
use crate::text::GlyphAtlas;
use crate::theme::{Theme, ThemeMode};

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
    top_inset: f32,
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
            top_inset: 0.0,
            dirty: true,
        }
    }

    /// Reserve top safe-area clearance (e.g. frameless OS traffic-light buttons) while keeping
    /// the background surface full-bleed from `(0, 0)`.
    pub fn set_top_inset(&mut self, top_inset: f32) {
        let next = top_inset.max(0.0);
        if (self.top_inset - next).abs() > 0.1 {
            self.top_inset = next;
            self.dirty = true;
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
                if let Some((pid, last_x, plot_w)) = self.interaction.dragging_plot.as_mut() {
                    let dx = x - *last_x;
                    if dx.abs() > 0.5 && *plot_w > 1.0 {
                        let pid_c = pid.clone();
                        let delta_frac = -dx / *plot_w;
                        *last_x = x;
                        let _ = self.apply_patch(ScenePatch::PanPlot {
                            id: pid_c,
                            delta_frac,
                        });
                    }
                }
                let next_hover = self
                    .interaction
                    .hit_topmost_interactive(x, y)
                    .map(|(k, _)| k);
                if next_hover != self.interaction.hovered_key {
                    self.interaction.hovered_key = next_hover;
                    self.dirty = true;
                }
                let hit_plot = self.interaction.hit_plot_area(x, y);
                match hit_plot {
                    Some((pid, _)) => {
                        let prev = self.interaction.plot_hover.insert(pid.clone(), [x, y]);
                        self.interaction.plot_hover.retain(|k, _| k == &pid);
                        if prev != Some([x, y]) {
                            self.dirty = true;
                        }
                    }
                    None => {
                        if !self.interaction.plot_hover.is_empty() {
                            self.interaction.plot_hover.clear();
                            self.dirty = true;
                        }
                    }
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
                        self.interaction.pressed_key = hit.clone();
                        self.dirty = true;
                    }
                    if hit.is_none() {
                        if let Some((pid, prect)) = self.interaction.hit_plot_area(x, y) {
                            self.interaction.dragging_plot = Some((pid, x, prect.w));
                        }
                    }
                } else if button == 1 {
                    if let Some((pid, _)) = self.interaction.hit_plot_area(x, y) {
                        let _ = self.apply_patch(ScenePatch::ResetPlotView { id: pid });
                    }
                }
            }
            InputEvent::PointerUp { x, y, button } => {
                self.interaction.pointer = [x, y];
                if button == 0 {
                    self.interaction.dragging_plot = None;
                    let pressed = self.interaction.pressed_key.take();
                    if pressed.is_some() {
                        self.dirty = true;
                    }
                    if let Some(pressed_key) = pressed {
                        if let Some((hit_key, kind)) = self.interaction.hit_topmost_interactive(x, y)
                        {
                            if hit_key == pressed_key {
                                match kind.clone() {
                                    HitKind::Surface { id }
                                    | HitKind::Stat { id }
                                    | HitKind::Button { id } => {
                                        out.push(UIEvent::Click { id });
                                    }
                                    HitKind::TabButton { tabs_id, index } => {
                                        let _ = self.apply_patch(ScenePatch::SetTab {
                                            id: tabs_id.clone(),
                                            active: index,
                                        });
                                        out.push(UIEvent::TabChanged { id: tabs_id, index });
                                    }
                                    HitKind::SegmentedItem { seg_id, index } => {
                                        let _ = self.apply_patch(ScenePatch::SetSegmented {
                                            id: seg_id.clone(),
                                            active: index,
                                        });
                                        out.push(UIEvent::SegmentedChanged { id: seg_id, index });
                                    }
                                    HitKind::ThemeToggle { id } => {
                                        let next = match self.scene.theme {
                                            ThemeMode::Dark => ThemeMode::Light,
                                            ThemeMode::Light => ThemeMode::Dark,
                                        };
                                        let _ = self.apply_patch(ScenePatch::SetTheme(next));
                                        out.push(UIEvent::ThemeChanged { id, mode: next });
                                    }
                                    HitKind::PlotLegendChip {
                                        plot_id,
                                        series_index,
                                    } => {
                                        let _ = self.apply_patch(ScenePatch::TogglePlotSeries {
                                            id: plot_id.clone(),
                                            series_index,
                                        });
                                        out.push(UIEvent::SeriesToggled {
                                            id: plot_id,
                                            series_index,
                                        });
                                    }
                                    HitKind::PlotArea { .. } | HitKind::ScrollContainer { .. } => {}
                                }
                            }
                        }
                    }
                }
            }
            InputEvent::Scroll { x, y, delta_y, .. } => {
                if let Some((pid, prect)) = self.interaction.hit_plot_area(x, y) {
                    let anchor_frac = ((x - prect.x) / prect.w.max(1.0)).clamp(0.0, 1.0);
                    let _ = self.apply_patch(ScenePatch::ZoomPlot {
                        id: pid,
                        delta_y,
                        anchor_frac,
                    });
                } else if let Some((scroll_id, max_y)) = self.interaction.hit_topmost_scroll(x, y) {
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
        self.layout
            .solve_with_top_inset(&self.scene.root, self.viewport, self.top_inset, &self.atlas);
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
        self.emit_node(&root_node, ROOT_PATH_HASH, 0.0, 0.0, draw_list);
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

    fn emit_node(
        &mut self,
        node: &Node,
        path_hash: u64,
        ox: f32,
        oy: f32,
        dl: &mut DrawList,
    ) {
        if !node.visible {
            return;
        }
        let Some(mut rect) = self.layout.get_path(path_hash) else {
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
                for (slot, child) in children.iter().enumerate() {
                    if !child.visible {
                        continue;
                    }
                    let ch_hash = child_path_hash(path_hash, slot, &child.id);
                    self.emit_node(child, ch_hash, ox, oy - scroll_y, dl);
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
                for (slot, child) in children.iter().enumerate() {
                    if !child.visible {
                        continue;
                    }
                    let ch_hash = child_path_hash(path_hash, slot, &child.id);
                    self.emit_node(child, ch_hash, ox, oy, dl);
                }
            }
            NodeKind::Surface {
                style,
                interactive,
                child,
            } => {
                let hit_key = format!("surf::{}", node.id);
                let hover_t = if *interactive {
                    let is_hovered =
                        self.interaction.hovered_key.as_deref() == Some(hit_key.as_str());
                    let is_pressed =
                        self.interaction.pressed_key.as_deref() == Some(hit_key.as_str());
                    let target = if is_pressed {
                        1.0
                    } else if is_hovered {
                        0.65
                    } else {
                        0.0
                    };
                    self.springs
                        .animate(&hit_key, "hover", target, SpringConfig::SNAPPY)
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
                    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                    dl.pop_clip();
                }
            }
            NodeKind::Tabs { active, items } => {
                let ix = rect.x + node.padding.left;
                let iy = rect.y + node.padding.top;
                let iw = (rect.w - node.padding.horizontal()).max(0.0);
                if let Some(active_clamped) = emit_tabs_bar(
                    &node.id,
                    *active,
                    items,
                    ix,
                    iy,
                    iw,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                ) {
                    if let Some(active_item) = items.get(active_clamped) {
                        let ch_hash = child_path_hash(path_hash, active_clamped, &active_item.child.id);
                        self.emit_node(&active_item.child, ch_hash, ox, oy, dl);
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
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                let tx = rect.x + node.padding.left;
                let inner_h = (rect.h - node.padding.vertical()).max(0.0);
                let ty = if !*wrap && !content.contains('\n') && inner_h > *size {
                    self.atlas
                        .centered_pen_y(rect.y + node.padding.top, inner_h, *size)
                } else {
                    rect.y + node.padding.top
                };
                let max_w = if *wrap {
                    Some((rect.w - node.padding.horizontal()).max(16.0))
                } else {
                    None
                };
                self.atlas.layout_dl(
                    content,
                    tx,
                    ty,
                    *size,
                    color.unwrap_or(self.theme.text_primary),
                    *mono,
                    max_w,
                    dl,
                );
                dl.pop_clip();
            }
            NodeKind::Rule { thickness, color } => {
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
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
                dl.pop_clip();
            }
            NodeKind::Spacer => {}
            NodeKind::ThemeToggle { .. } => {
                emit_theme_toggle(
                    &node.id,
                    self.scene.theme,
                    rect,
                    &self.theme,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Segmented { active, options } => {
                emit_segmented(
                    &node.id,
                    *active,
                    options,
                    rect,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Button { label, tone } => {
                emit_button(
                    &node.id,
                    label,
                    *tone,
                    rect,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Stat(data) => {
                emit_stat(
                    &node.id,
                    data,
                    rect,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Plot(data) => {
                emit_plot(
                    &node.id,
                    data,
                    rect,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
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
        // Zero outline unless user explicitly opted in (`border` + `border_width > 0`)
        let border = if style.border_width > 0.0 {
            style.border.unwrap_or(self.theme.border)
        } else {
            0
        };
        // Zero shadow unless user explicitly opted in (`shadow_blur > 0`)
        let shadow = if style.shadow_blur > 0.0 {
            style.shadow_color.unwrap_or(self.theme.shadow)
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
                                                style: BoxStyle::card(t.radius_card),
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

    #[test]
    fn stat_and_plot_adapt_to_extreme_sizes_cardinalities_and_streaming_patches() {
        use crate::components::plot::{CandlePoint, PlotData, PlotPoint, PlotSeries, PlotSeriesKind};
        use crate::components::stat::StatData;
        use crate::theme::Tone;

        // Build a 10,000-point stream to verify bounded min/max decimation.
        let dense_pts: Vec<PlotPoint> = (0..10_000)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 100.0 + ((i as f64) * 0.05).sin() * 15.0,
                y2: None,
                label: Some(format!("T{i}")),
            })
            .collect();

        let scene = Scene::new(
            "Adaptive Quantitative Monitor",
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
                            "kpi_main",
                            NodeKind::Stat(StatData {
                                label: "Primary Index Value With Very Long Label".into(),
                                badge: Some("LIVE".into()),
                                value: 94210.50,
                                prefix: "$".into(),
                                delta: Some("+2.84%".into()),
                                tone: Tone::Ok,
                                caption: Some("Rolling 24h window".into()),
                                sparkline: vec![91.0, 92.5, 92.0, 93.8, 94.2],
                                ..StatData::default()
                            }),
                        )
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Auto),
                        Node::new(
                            "chart",
                            NodeKind::Plot(PlotData {
                                title: Some("Dense Stream + OHLC".into()),
                                subtitle: Some("Adaptive decimation".into()),
                                y_prefix: "$".into(),
                                series: vec![
                                    PlotSeries {
                                        id: "candles".into(),
                                        name: "Price".into(),
                                        kind: PlotSeriesKind::Candles,
                                        color: None,
                                        secondary_color: None,
                                        visible: true,
                                        dashed: false,
                                        stroke_width: None,
                                        points: vec![],
                                        candles: vec![
                                            CandlePoint {
                                                x: 0.0,
                                                open: 95.0,
                                                high: 108.0,
                                                low: 93.0,
                                                close: 104.0,
                                                label: Some("00:00".into()),
                                            },
                                            CandlePoint {
                                                x: 5000.0,
                                                open: 104.0,
                                                high: 115.0,
                                                low: 100.0,
                                                close: 111.0,
                                                label: Some("12:00".into()),
                                            },
                                        ],
                                    },
                                    PlotSeries {
                                        id: "signal".into(),
                                        name: "Signal".into(),
                                        kind: PlotSeriesKind::Line,
                                        color: None,
                                        secondary_color: None,
                                        visible: true,
                                        dashed: false,
                                        stroke_width: None,
                                        points: dense_pts,
                                        candles: vec![],
                                    },
                                ],
                                ..PlotData::default()
                            }),
                        )
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
                    ],
                },
            )
            .with_padding(Insets::all(12.0)),
        );

        let mut engine = UiEngine::new(scene, [1200.0, 760.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        // Decimated 10k-point line + candles stays cleanly under 4,000 stroke instances.
        assert!(dl.strokes.len() < 4000 && dl.strokes.len() > 20);

        // Compact phone viewport: Stat compacts cleanly and Plot adapts axes without panic.
        let _ = engine.handle_input(InputEvent::Resize {
            width: 340.0,
            height: 480.0,
            scale_factor: 2.0,
        });
        let st_narrow = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st_narrow.compiled);

        // Hover near right edge of plot to test edge-clamped crosshair readout box.
        let _ = engine.handle_input(InputEvent::PointerMove { x: 250.0, y: 300.0 });
        let st_hover = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st_hover.compiled);

        // Streaming Stat + Plot updates drive animations & settle back cleanly.
        assert!(engine.apply_patch(ScenePatch::UpdateStat {
            id: "kpi_main".into(),
            value: Some(95120.25),
            value_text: None,
            delta: Some(Some("+3.10%".into())),
            tone: Some(Tone::Ok),
            caption: None,
            push_sparkline: Some(95.1),
            sparkline_max_len: Some(32),
        }));
        assert!(engine.apply_patch(ScenePatch::UpsertCandle {
            id: "chart".into(),
            series_id: "candles".into(),
            candle: CandlePoint {
                x: 9999.0,
                open: 111.0,
                high: 118.0,
                low: 109.0,
                close: 116.5,
                label: Some("23:59".into()),
            },
            max_candles: Some(100),
        }));
        let st_stream = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st_stream.compiled);
        assert!(st_stream.animating);

        // Theme toggles cleanly between Dark and Light at runtime.
        assert!(engine.apply_patch(ScenePatch::SetTheme(ThemeMode::Light)));
        let st_light = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st_light.compiled);
        // Zero default borders and zero default shadows on cards/plots/pills unless explicitly opted in.
        assert!(
            dl.quads
                .iter()
                .all(|q| q.border_color == 0 && q.shadow_color == 0),
            "default Stat/Plot UI quads must have zero border and zero shadow unless explicitly added"
        );
    }

    #[test]
    fn theme_toggle_renders_moon_in_dark_and_sun_in_light_mode() {
        let scene = Scene::new(
            "Theme Icon Test",
            ThemeMode::Dark,
            Node::new("theme_btn", NodeKind::ThemeToggle { size: 36.0 }),
        );
        let mut engine = UiEngine::new(scene, [200.0, 100.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        // Dark mode draws crescent moon only (0 sun ray strokes)
        assert_eq!(dl.strokes.len(), 0);
        assert!(dl.quads.iter().all(|q| q.border_color == 0 && q.shadow_color == 0));

        // Click inside the 36x36 ThemeToggle button
        let _ = engine.handle_input(InputEvent::PointerDown {
            x: 18.0,
            y: 18.0,
            button: 0,
        });
        let events = engine.handle_input(InputEvent::PointerUp {
            x: 18.0,
            y: 18.0,
            button: 0,
        });
        assert_eq!(
            events,
            vec![UIEvent::ThemeChanged {
                id: "theme_btn".into(),
                mode: ThemeMode::Light
            }]
        );
        assert_eq!(engine.scene.theme, ThemeMode::Light);

        // Light mode draws Sun core + 8 symmetrical rays
        let st_light = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st_light.compiled);
        assert_eq!(dl.strokes.len(), 8);
    }

    #[test]
    fn multipane_stacked_area_band_and_zoom_emit_clean_ordered_ribbons() {
        use crate::components::plot::{
            PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind, PlotThreshold,
        };

        let pts_a: Vec<PlotPoint> = (0..20)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 45.0 + (i as f64 * 0.4).sin() * 12.0,
                y2: Some(38.0 + (i as f64 * 0.25).cos() * 10.0),
                label: Some(format!("D{i}")),
            })
            .collect();
        let pts_s1: Vec<PlotPoint> = (0..20)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 40.0 + (i as f64 * 0.3).sin() * 5.0,
                y2: None,
                label: None,
            })
            .collect();
        let pts_s2: Vec<PlotPoint> = (0..20)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 35.0,
                y2: None,
                label: None,
            })
            .collect();

        let scene = Scene::new(
            "MultiPane Test",
            ThemeMode::Dark,
            Node::new(
                "mp",
                NodeKind::Plot(PlotData {
                    title: Some("Sync Panes".into()),
                    panes: vec![
                        PlotPane {
                            id: "osc".into(),
                            label: Some("Band Spread".into()),
                            height_weight: 1.0,
                            thresholds: vec![PlotThreshold {
                                value: 50.0,
                                label: Some("MID".into()),
                                color: None,
                            }],
                            series: vec![PlotSeries {
                                id: "band".into(),
                                name: "Osc".into(),
                                kind: PlotSeriesKind::Band,
                                color: None,
                                secondary_color: None,
                                visible: true,
                                dashed: false,
                                stroke_width: None,
                                points: pts_a,
                                candles: vec![],
                            }],
                            ..PlotPane::default()
                        },
                        PlotPane {
                            id: "alloc".into(),
                            label: Some("Stacked 100%".into()),
                            height_weight: 1.0,
                            stacked_percent: true,
                            series: vec![
                                PlotSeries {
                                    id: "s1".into(),
                                    name: "S1".into(),
                                    kind: PlotSeriesKind::StackedArea,
                                    color: None,
                                    secondary_color: None,
                                    visible: true,
                                    dashed: false,
                                    stroke_width: None,
                                    points: pts_s1,
                                    candles: vec![],
                                },
                                PlotSeries {
                                    id: "s2".into(),
                                    name: "S2".into(),
                                    kind: PlotSeriesKind::StackedArea,
                                    color: None,
                                    secondary_color: None,
                                    visible: true,
                                    dashed: false,
                                    stroke_width: None,
                                    points: pts_s2,
                                    candles: vec![],
                                },
                            ],
                            ..PlotPane::default()
                        },
                    ],
                    ..PlotData::default()
                }),
            ),
        );

        let mut engine = UiEngine::new(scene, [900.0, 600.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        assert!(!dl.ribbons.is_empty());
        assert!(!dl.cmds.is_empty());
        assert!(engine.apply_patch(ScenePatch::ZoomPlot {
            id: "mp".into(),
            delta_y: 120.0,
            anchor_frac: 0.5,
        }));
    }

    #[test]
    fn multipane_locks_rolling_x_window_to_primary_pane() {
        use crate::components::plot::{PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind};

        // Primary pane rolled forward to x = 500..=510; secondary pane still holds older x = 0..=10.
        let p0_pts: Vec<PlotPoint> = (500..=510)
            .map(|x| PlotPoint {
                x: x as f64,
                y: 250.0 + (x - 500) as f64,
                y2: None,
                label: Some(format!("T{x}")),
            })
            .collect();
        let p1_pts: Vec<PlotPoint> = (0..=10)
            .map(|x| PlotPoint {
                x: x as f64,
                y: 50.0,
                y2: None,
                label: Some(format!("Old{x}")),
            })
            .collect();

        let scene = Scene::new(
            "Rolling X Lock",
            ThemeMode::Light,
            Node::new(
                "mp_lock",
                NodeKind::Plot(PlotData {
                    panes: vec![
                        PlotPane {
                            id: "p0".into(),
                            series: vec![PlotSeries {
                                id: "s0".into(),
                                name: "S0".into(),
                                kind: PlotSeriesKind::Line,
                                color: None,
                                secondary_color: None,
                                visible: true,
                                dashed: false,
                                stroke_width: None,
                                points: p0_pts,
                                candles: vec![],
                            }],
                            ..PlotPane::default()
                        },
                        PlotPane {
                            id: "p1".into(),
                            series: vec![PlotSeries {
                                id: "s1".into(),
                                name: "S1".into(),
                                kind: PlotSeriesKind::Line,
                                color: None,
                                secondary_color: None,
                                visible: true,
                                dashed: false,
                                stroke_width: None,
                                points: p1_pts,
                                candles: vec![],
                            }],
                            ..PlotPane::default()
                        },
                    ],
                    ..PlotData::default()
                }),
            ),
        );

        let mut engine = UiEngine::new(scene, [1000.0, 600.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        // Primary pane's 10 segments MUST span across the full plot width (from ~26px to ~880px),
        // NOT compressed into the right 2% of the chart.
        let line_xs: Vec<f32> = dl
            .strokes
            .iter()
            .filter(|s| (s.p0[1] - s.p1[1]).abs() > 0.5) // sloped line segments (not flat gridlines)
            .flat_map(|s| [s.p0[0], s.p1[0]])
            .collect();
        let min_x = line_xs.iter().copied().fold(f32::INFINITY, f32::min);
        let max_x = line_xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_x < 100.0 && max_x > 800.0, "expected full width span, got [{min_x}, {max_x}]");
    }

    #[test]
    fn row_components_never_overflow_or_collide() {
        use crate::components::plot::PlotPoint;
        use crate::ui;

        let p_pts: Vec<PlotPoint> = (0..10)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 100.0 + i as f64,
                y2: None,
                label: None,
            })
            .collect();

        let s1: Node = ui::stat("s1", "Metric A", 100.0).into();
        let s2: Node = ui::stat("s2", "Metric B", 200.0).into();
        let p1: Node = ui::plot("p1")
            .pane(ui::pane("pane1").line("l", "L", p_pts, None))
            .into();
        let b1 = ui::button("b1", "Reset");

        let root_node: Node = ui::row("row_root")
            .fold_below(0.0)
            .add(s1)
            .add(s2)
            .add(p1)
            .add(b1)
            .into();
        let scene = Scene::new("Row Isolation", ThemeMode::Light, root_node);

        // Total unshrunk intrinsic width = 170..230 + 170..230 + 280 + ~80 + gaps > 720
        let vw = 720.0;
        let mut engine = UiEngine::new(scene, [vw, 500.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);

        let r_s1 = engine.layout.get("s1").unwrap();
        let r_s2 = engine.layout.get("s2").unwrap();
        let r_p1 = engine.layout.get("p1").unwrap();
        let r_b1 = engine.layout.get("b1").unwrap();

        let rects = [r_s1, r_s2, r_p1, r_b1];
        for pair in rects.windows(2) {
            assert!(
                pair[0].x + pair[0].w <= pair[1].x + 0.01,
                "components overlapped in row: {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(
            r_b1.x + r_b1.w <= vw + 0.01,
            "row overflowed viewport width {vw}: {:?}",
            r_b1
        );
    }

    #[test]
    fn duplicate_component_ids_isolate_layout_and_hits() {
        use crate::components::stat::StatData;
        use crate::ui;

        // Even if a user accidentally passes identical IDs ("dup") to two distinct components,
        // path-keyed layout + namespaced hit regions prevent overwrite or cross-hover leakage.
        let stat_node = Node::new(
            "dup",
            NodeKind::Stat(StatData {
                label: "Metric".into(),
                value: 42.0,
                interactive: true,
                ..StatData::default()
            }),
        );
        let btn_node = ui::button("dup", "Action");
        let root_node: Node = ui::row("dup_row")
            .fold_below(0.0)
            .add(stat_node)
            .add(btn_node)
            .into();
        let scene = Scene::new("Duplicate ID Isolation", ThemeMode::Light, root_node);

        let mut engine = UiEngine::new(scene, [800.0, 400.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);

        let h0 = child_path_hash(ROOT_PATH_HASH, 0, "dup");
        let h1 = child_path_hash(ROOT_PATH_HASH, 1, "dup");
        let r0 = engine.layout.get_path(h0).unwrap();
        let r1 = engine.layout.get_path(h1).unwrap();
        assert!(r0.x + r0.w < r1.x, "duplicate IDs collided: {r0:?} vs {r1:?}");

        // Hover over the stat -> hit key is namespaced ("stat::dup"), NOT "btn::dup"
        let _ = engine.handle_input(InputEvent::PointerMove {
            x: r0.x + r0.w * 0.5,
            y: r0.y + r0.h * 0.5,
        });
        assert_eq!(engine.interaction.hovered_key.as_deref(), Some("stat::dup"));

        // Hover over the button -> hit key switches cleanly to "btn::dup"
        let _ = engine.handle_input(InputEvent::PointerMove {
            x: r1.x + r1.w * 0.5,
            y: r1.y + r1.h * 0.5,
        });
        assert_eq!(engine.interaction.hovered_key.as_deref(), Some("btn::dup"));
    }

    #[test]
    fn side_by_side_plots_have_independent_crosshairs() {
        use crate::components::plot::PlotPoint;
        use crate::ui;

        let pts: Vec<PlotPoint> = (0..12)
            .map(|i| PlotPoint {
                x: i as f64,
                y: 50.0 + (i as f64) * 2.0,
                y2: None,
                label: None,
            })
            .collect();

        let left_plot: Node = ui::plot("plot_left")
            .pane(ui::pane("lp").line("l", "Left", pts.clone(), None))
            .into();
        let right_plot: Node = ui::plot("plot_right")
            .pane(ui::pane("rp").line("r", "Right", pts, None))
            .into();

        let root_node: Node = ui::row("plots_row")
            .fold_below(0.0)
            .add(left_plot)
            .add(right_plot)
            .into();
        let scene = Scene::new("Multi-Plot Isolation", ThemeMode::Light, root_node);

        let mut engine = UiEngine::new(scene, [1000.0, 500.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);

        let rl = engine.layout.get("plot_left").unwrap();
        let rr = engine.layout.get("plot_right").unwrap();
        assert!(rl.x + rl.w < rr.x, "side-by-side plots overlapped");

        // Move pointer into center of left plot only
        let _ = engine.handle_input(InputEvent::PointerMove {
            x: rl.x + rl.w * 0.5,
            y: rl.y + rl.h * 0.5,
        });
        assert!(engine.interaction.plot_hover.contains_key("plot_left"));
        assert!(!engine.interaction.plot_hover.contains_key("plot_right"));
    }
}
