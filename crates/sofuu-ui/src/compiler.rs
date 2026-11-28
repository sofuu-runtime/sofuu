//! Scene compiler: turns Scene IR + Layout + 120Hz Springs into flat GPU `DrawList` buffers.

use crate::components::code::emit_code;
use std::collections::HashMap;

use crate::components::controls::{
    emit_button, emit_input, emit_progress, emit_range_slider, emit_segmented, emit_slider,
    emit_switch, emit_tabs_bar, emit_textarea, emit_theme_toggle, slider_value_at,
};
use crate::components::list as vlist;
use crate::components::media::{emit_icon, emit_image};
use crate::components::feed::emit_feed;
use crate::components::graph::emit_graph;
use crate::components::plot::emit_plot;
use crate::components::stat::emit_stat;
use crate::components::table::{emit_table, TableSortDir};
use crate::draw::{lerp_rgba8, scale_alpha, ClipRect, DrawList, QuadInstance};
use crate::fx::{DrawMark, FxEngine, NodeFx};
use crate::hit_test::{
    DragSession, HitKind, InputEvent, InteractionState, KineticScroll, NavSwipe, ScrollPhysics,
    TextEditState, UIEvent, MOD_CTRL, MOD_META, MOD_SHIFT,
};
use crate::icons::IconAtlas;
use crate::images::{key_for, ImageAtlas};
use crate::ir::{BoxStyle, Insets, KeyboardHint, Node, NodeKind, Scene, ScenePatch};
use crate::layout::{child_path_hash, rich_spans, ComputedRect, LayoutMap, ROOT_PATH_HASH};
use crate::spring::{SpringBank, SpringConfig};
use crate::text::{GlyphAtlas, LineBreak, TextStyle, WEIGHT_REGULAR};
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
    /// Opt-in motion/effect scheduler (`sofuu.fx`).
    pub fx: FxEngine,
    /// Process-wide procedural SDF icon atlas.
    pub icons: std::sync::Arc<IconAtlas>,
    /// LRU-packed bitmap image atlas (decoded lazily by the host).
    pub images: ImageAtlas,
    /// Platform safe-area insets (notch, rounded corners, gesture bar).
    pub safe_insets: Insets,
    /// Monotonic seconds accumulated from `tick_and_compile` deltas. Drives
    /// caret blinking, indeterminate progress sweeps and tap-multiplicity
    /// detection without the engine ever reading the wall clock.
    pub now_s: f64,
    viewport: [f32; 2],
    top_inset: f32,
    dirty: bool,
    /// Last observed `(field, anchor, caret)`; a change restarts the blink.
    caret_sig: Option<(String, usize, usize)>,
    /// Draw a live touch reticle at the pointer while it is down. Live emulator
    /// only; headless captures never enable it.
    touch_reticle: bool,
    /// Draw a thin outline around every interactive hit region. Live emulator
    /// only (strip "Bounds"); headless captures never enable it.
    show_bounds: bool,
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
            fx: FxEngine::new(),
            icons: IconAtlas::builtin(),
            images: ImageAtlas::new(crate::images::DEFAULT_MAX_VRAM_MB),
            safe_insets: Insets::ZERO,
            now_s: 0.0,
            viewport,
            top_inset: 0.0,
            dirty: true,
            caret_sig: None,
            touch_reticle: false,
            show_bounds: false,
        }
    }

    /// Update platform insets (notch, rounded corners, home indicator). Marks the
    /// scene dirty so the next tick re-solves layout.
    pub fn set_safe_insets(&mut self, insets: Insets) {
        if self.safe_insets != insets {
            self.safe_insets = insets;
            self.dirty = true;
        }
    }

    /// Image sources referenced by visible `Image` nodes and not yet decoded.
    pub fn pending_images(&self) -> Vec<String> {
        let mut out = Vec::new();
        collect_pending_images(&self.scene.root, &self.images, &mut out);
        out
    }

    /// Decode/insert a bitmap for `src`; the next tick uploads and draws it.
    pub fn set_image(
        &mut self,
        src: &str,
        image: crate::images::DecodedImage,
    ) -> Result<crate::images::ImageHandle, crate::images::ImageError> {
        let handle = self.images.insert(key_for(src), image)?;
        self.dirty = true;
        Ok(handle)
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

    /// Enable the live touch reticle (emulator interaction parity). Captures
    /// leave this off so the PNG stays clean.
    pub fn set_touch_reticle(&mut self, on: bool) {
        if self.touch_reticle != on {
            self.touch_reticle = on;
            self.dirty = true;
        }
    }

    /// Enable the live layout-bounds overlay (emulator strip "Bounds").
    pub fn set_show_bounds(&mut self, on: bool) {
        if self.show_bounds != on {
            self.show_bounds = on;
            self.dirty = true;
        }
    }

    pub fn viewport(&self) -> [f32; 2] {
        self.viewport
    }

    /// Scroll offsets as the renderer sees them: clamped offset plus the
    /// platform overscroll displacement.
    fn visual_scrolls(&self) -> HashMap<String, f32> {
        self.interaction
            .scrolls
            .iter()
            .map(|(id, s)| (id.clone(), s.visual_offset()))
            .collect()
    }

    /// Choose the platform's edge behaviour (`iOS`/`macOS` rubber band,
    /// `Android` stretch, desktop clamp + scrollbar).
    pub fn set_scroll_physics(&mut self, physics: crate::hit_test::ScrollPhysics) {
        if self.interaction.scroll_physics != physics {
            self.interaction.scroll_physics = physics;
            self.dirty = true;
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty_or_animating(&self) -> bool {
        self.dirty
            || self.springs.has_active()
            || self.fx.is_animating()
            || self
                .interaction
                .scrolls
                .values()
                .any(|s| s.velocity_y.abs() >= 4.0 || s.overscroll.abs() >= 0.35)
    }

    /// Restart the blink whenever the caret or selection moved since the last
    /// tick, so typing always leaves a solid caret behind.
    fn note_caret_activity(&mut self) {
        let sig = self
            .interaction
            .edit
            .as_ref()
            .map(|e| (e.id.clone(), e.anchor, e.caret));
        if sig != self.caret_sig {
            self.caret_sig = sig;
            let now = self.now_s;
            self.interaction.reset_caret_blink(now);
        }
    }

    /// Advance the 500 ms caret blink. A toggle marks the frame dirty (one
    /// compile) and the engine then sleeps again, so a focused field costs one
    /// wake-up per second instead of a 60/120 Hz present loop.
    fn step_caret_blink(&mut self) {
        if self.interaction.edit.is_none() || self.interaction.focused_key.is_none() {
            if !self.interaction.caret_visible {
                self.interaction.caret_visible = true;
                self.dirty = true;
            }
            return;
        }
        let phase = (self.now_s - self.interaction.caret_epoch_s).rem_euclid(1.0);
        let visible = phase < 0.5;
        if visible != self.interaction.caret_visible {
            self.interaction.caret_visible = visible;
            self.dirty = true;
        }
    }

    /// Milliseconds until the host must wake the render thread again, or
    /// `None` when nothing is scheduled (the window can sleep indefinitely).
    ///
    /// Today this is the caret blink half-period; ambient effects and springs
    /// report through [`Self::is_dirty_or_animating`] instead.
    pub fn next_wake_ms(&self) -> Option<f64> {
        self.interaction
            .caret_blink_deadline(self.now_s)
            .map(|secs| (secs * 1000.0).max(1.0))
    }

    /// OS / app "reduce motion" preference. Ambient effects freeze; one-shots still settle.
    pub fn set_reduced_motion(&mut self, enabled: bool) {
        if self.fx.reduced_motion() != enabled {
            self.fx.set_reduced_motion(enabled);
            self.dirty = true;
        }
    }

    /// Pause continuous ambient effects while the window is unfocused/backgrounded.
    pub fn set_ambient_paused(&mut self, paused: bool) {
        if self.fx.ambient_paused() != paused {
            self.fx.set_ambient_paused(paused);
            self.dirty = true;
        }
    }

    /// Audit the most recently compiled layout + draw list. Call
    /// [`Self::tick_and_compile`] first (or use [`Self::audit_now`]).
    pub fn audit(
        &self,
        draw_list: &DrawList,
        rules: &crate::audit::AuditRules,
    ) -> crate::audit::AuditReport {
        let scrolls = self.visual_scrolls();
        crate::audit::audit_scene_with_atlas(
            &self.scene,
            &self.layout,
            draw_list,
            rules,
            &scrolls,
            &self.atlas,
        )
    }

    /// Compile one frame (if needed) and audit it immediately.
    pub fn audit_now(
        &mut self,
        draw_list: &mut DrawList,
        rules: &crate::audit::AuditRules,
    ) -> crate::audit::AuditReport {
        let _ = self.tick_and_compile(0.016, draw_list);
        self.audit(draw_list, rules)
    }

    pub fn set_scene(&mut self, scene: Scene) {
        self.theme = Theme::for_mode(scene.theme);
        self.scene = scene;
        self.dirty = true;
    }

    pub fn apply_patch(&mut self, patch: ScenePatch) -> bool {
        let changed = match patch {
            ScenePatch::FxTrigger {
                target,
                kind,
                count,
                x,
                y,
                seed,
            } => {
                let origin = match (x, y) {
                    (Some(x), Some(y)) => Some([x, y]),
                    _ => None,
                };
                self.fx
                    .trigger(target, kind, count.unwrap_or(48), origin, seed.unwrap_or(1));
                true
            }
            ScenePatch::ReplaceScene(scene) => self.replace_scene_preserving(scene),
            other => self.scene.apply_patch(other),
        };
        if changed {
            self.theme = Theme::for_mode(self.scene.theme);
            self.dirty = true;
        }
        changed
    }

    /// Hot-reload entry point: swap the scene graph while carrying live app
    /// state (input text, switch/slider values, active route, theme) across
    /// matching node ids. Scroll offsets and focus live in `interaction` and
    /// therefore survive untouched.
    pub fn replace_scene_preserving(&mut self, mut scene: Scene) -> bool {
        let old = &self.scene;
        scene.theme = old.theme;
        preserve_scene_state(&old.root, &mut scene.root);
        if old.title == scene.title && old.theme == scene.theme && old.root == scene.root {
            return false;
        }
        self.scene = scene;
        true
    }

    /// Soft-keyboard type for the focused editable, if any (`SOFUU_KEYBOARD_*`).
    pub fn focused_keyboard_hint(&self) -> Option<KeyboardHint> {
        let focused = self.interaction.focused_key.as_deref()?;
        let (kind, id) = focused.split_once("::")?;
        let node = self.scene.root.find(id)?;
        match (&node.kind, kind) {
            (NodeKind::Input { keyboard, .. }, "input") => Some(*keyboard),
            (NodeKind::TextArea(data), "textarea") => Some(data.keyboard),
            _ => None,
        }
    }

    /// Resolve the focused field's caret/selection, creating it lazily.
    fn edit_for(&mut self, focus_key: &str, value_len: usize) -> &mut TextEditState {
        let needs_reset = self
            .interaction
            .edit
            .as_ref()
            .map(|e| e.id != focus_key)
            .unwrap_or(true);
        if needs_reset {
            self.interaction.edit = Some(TextEditState::new(focus_key, value_len));
        }
        self.interaction.edit.as_mut().expect("edit state installed above")
    }

    fn editable_value(&self, id: &str) -> Option<(String, bool)> {
        match &self.scene.root.find(id)?.kind {
            NodeKind::Input { value, .. } => Some((value.clone(), false)),
            NodeKind::TextArea(data) => Some((data.value.clone(), true)),
            _ => None,
        }
    }

    fn set_editable_value(&mut self, id: &str, value: String) {
        if let Some(node) = self.scene.root.find_mut(id) {
            match &mut node.kind {
                NodeKind::Input { value: v, .. } => *v = value,
                NodeKind::TextArea(data) => data.value = value,
                _ => {}
            }
        }
    }

    /// Apply one edit `[start, end) -> insert` and return `(new_value, caret)`.
    fn apply_edit_range(
        &mut self,
        id: &str,
        start: usize,
        end: usize,
        insert: &str,
    ) -> Option<(String, usize)> {
        let (mut value, _) = self.editable_value(id)?;
        let start = floor_boundary(&value, start.min(value.len()));
        let end = floor_boundary(&value, end.min(value.len())).max(start);
        self.record_edit(id, &value, start, end, insert);
        value.replace_range(start..end, insert);
        let caret = start + insert.len();
        self.set_editable_value(id, value.clone());
        Some((value, caret))
    }

    /// Push a pre-edit checkpoint so `Cmd+Z` can restore it.
    fn record_edit(&mut self, id: &str, value: &str, start: usize, end: usize, insert: &str) {
        let focus_key = self
            .interaction
            .focused_key
            .clone()
            .unwrap_or_else(|| format!("input::{id}"));
        let (anchor, caret) = self
            .interaction
            .edit
            .as_ref()
            .map(|e| (e.anchor, e.caret))
            .unwrap_or((start, end));
        let kind = if insert.is_empty() {
            crate::hit_test::EditOp::Delete
        } else {
            crate::hit_test::EditOp::Insert
        };
        let snapshot = crate::hit_test::EditSnapshot {
            value: value.to_string(),
            anchor,
            caret,
        };
        let inserted = insert.chars().count();
        self.interaction
            .histories
            .entry(focus_key)
            .or_default()
            .record(snapshot, kind, start, end, inserted);
    }

    /// Restore the previous value/selection for `id`. `redo` walks forward.
    fn step_history(&mut self, id: &str, redo: bool) -> Option<Vec<UIEvent>> {
        let focus_key = self
            .interaction
            .focused_key
            .clone()
            .unwrap_or_else(|| format!("input::{id}"));
        let (value, _) = self.editable_value(id)?;
        let (anchor, caret) = self
            .interaction
            .edit
            .as_ref()
            .map(|e| (e.anchor, e.caret))
            .unwrap_or((value.len(), value.len()));
        let current = crate::hit_test::EditSnapshot {
            value: value.clone(),
            anchor,
            caret,
        };
        let hist = self.interaction.histories.entry(focus_key.clone()).or_default();
        let next = if redo {
            hist.pop_redo(current)
        } else {
            hist.pop_undo(current)
        }?;
        let caret = next.caret.min(next.value.len());
        let anchor = next.anchor.min(next.value.len());
        self.set_editable_value(id, next.value.clone());
        self.interaction.edit = Some(TextEditState {
            id: focus_key,
            anchor,
            caret,
        });
        self.dirty = true;
        Some(vec![UIEvent::TextChanged {
            id: id.to_string(),
            value: next.value,
        }])
    }

    /// The wrap width used by the field's renderer (`None` for one-line inputs).
    fn editable_wrap_width(&self, id: &str) -> Option<f32> {
        let rect = self.layout.get(id)?;
        match &self.scene.root.find(id)?.kind {
            NodeKind::TextArea(_) => Some((rect.w - 20.0).max(8.0)),
            _ => None,
        }
    }

    /// Consecutive-tap counter: `1` = caret, `2` = word, `3` = line.
    ///
    /// Taps further apart than 320 ms or 24 pt are a new gesture, so a
    /// deliberate second click does not turn into a word selection.
    fn tap_count(&mut self, key: &str, x: f32, y: f32) -> u8 {
        const GAP_S: f64 = 0.32;
        const SLOP: f32 = 24.0;
        let next = match &self.interaction.tap {
            Some(t)
                if t.key == key
                    && (self.now_s - t.at_s) <= GAP_S
                    && (x - t.x).abs() <= SLOP
                    && (y - t.y).abs() <= SLOP =>
            {
                (t.count + 1).min(3)
            }
            _ => 1,
        };
        self.interaction.tap = Some(crate::hit_test::TapRecord {
            key: key.to_string(),
            x,
            y,
            at_s: self.now_s,
            count: next,
        });
        next
    }

    /// Byte offset of the caret for a click at `(x, y)` inside an editable.
    fn caret_from_pointer(&self, key: &str, x: f32, y: f32) -> Option<usize> {
        let (_, id) = key.split_once("::")?;
        let rect = self.layout.get(id)?;
        let style = TextStyle::sans(EDIT_PX, WEIGHT_REGULAR);
        match &self.scene.root.find(id)?.kind {
            NodeKind::Input { value, .. } => {
                let inner = (rect.w - 20.0).max(8.0);
                let (full_w, _) = self.atlas.measure(value, EDIT_PX, false);
                // The one-line field renders its tail when the value overflows;
                // clicking there keeps the caret at the end rather than
                // pretending the hidden head is visible.
                if full_w > (inner - 4.0).max(4.0) {
                    return Some(value.len());
                }
                let x_rel = (x - (rect.x + 10.0)).max(0.0);
                Some(
                    self.atlas
                        .char_index_at(value, x_rel, 0.0, style, None, LineBreak::None),
                )
            }
            NodeKind::TextArea(data) => {
                let inner = (rect.w - 20.0).max(8.0);
                let x_rel = (x - (rect.x + 10.0)).max(0.0);
                let y_rel = (y - (rect.y + 10.0)).max(0.0);
                Some(self.atlas.char_index_at(
                    &data.value,
                    x_rel,
                    y_rel,
                    style,
                    Some(inner),
                    LineBreak::CharWrap,
                ))
            }
            _ => None,
        }
    }

    /// Start/end byte offsets of the visual line containing `caret`.
    fn line_bounds(&self, value: &str, caret: usize, wrap_w: Option<f32>) -> (usize, usize) {
        let style = TextStyle::sans(EDIT_PX, WEIGHT_REGULAR);
        let laid = self
            .atlas
            .layout_text(value, style, wrap_w, usize::MAX, LineBreak::CharWrap);
        for line in &laid.lines {
            if caret <= line.end {
                return (line.start, line.end);
            }
        }
        let last = laid.lines.last().copied().unwrap_or(crate::text::TextLine {
            start: 0,
            end: value.len(),
            width: 0.0,
        });
        (last.start, last.end)
    }

    /// Keyboard editing for the focused field: caret navigation, shift
    /// selection, select-all, copy/cut, and selection-aware insertion/deletion.
    fn handle_editable_key(
        &mut self,
        id: String,
        is_area: bool,
        key: &str,
        modifiers: u32,
    ) -> Vec<UIEvent> {
        let mut out = Vec::new();
        let shift = modifiers & MOD_SHIFT != 0;
        let cmd = modifiers & (MOD_CTRL | MOD_META) != 0;
        let Some((value, _)) = self.editable_value(&id) else {
            return out;
        };
        let focus_key = self
            .interaction
            .focused_key
            .clone()
            .unwrap_or_else(|| format!("input::{id}"));

        if matches!(key, "keycode:53" | "Escape") {
            self.interaction.focused_key = None;
            self.interaction.edit = None;
            self.dirty = true;
            return out;
        }

        let (selection, caret, anchor) = {
            let edit = self.edit_for(&focus_key, value.len());
            let caret = edit.caret.min(value.len());
            (edit.selection(), caret, edit.anchor.min(value.len()))
        };

        let undo = key == "Undo" || (cmd && !shift && matches!(key, "z" | "Z" | "keycode:6"));
        let redo = key == "Redo"
            || (cmd && shift && matches!(key, "z" | "Z" | "keycode:6"))
            || (cmd && matches!(key, "y" | "Y" | "keycode:16"));
        if undo || redo {
            return self.step_history(&id, redo).unwrap_or_default();
        }

        let select_all = key == "SelectAll" || (cmd && matches!(key, "a" | "A" | "keycode:0"));
        if select_all {
            self.interaction.edit = Some(TextEditState {
                id: focus_key,
                anchor: 0,
                caret: value.len(),
            });
            self.dirty = true;
            return out;
        }

        let copy = key == "Copy" || (cmd && matches!(key, "c" | "C" | "keycode:8"));
        let cut = key == "Cut" || (cmd && matches!(key, "x" | "X" | "keycode:7"));
        if copy || cut {
            let (a, b) = selection;
            if a != b {
                out.push(UIEvent::CopyRequested {
                    id: id.clone(),
                    text: value[a..b].to_string(),
                });
                if cut {
                    if let Some((new_value, caret)) = self.apply_edit_range(&id, a, b, "") {
                        self.interaction.edit = Some(TextEditState {
                            id: focus_key,
                            anchor: caret,
                            caret,
                        });
                        self.dirty = true;
                        out.push(UIEvent::TextChanged { id, value: new_value });
                    }
                }
            }
            return out;
        }

        let is_backspace = matches!(key, "keycode:51" | "Backspace" | "\u{8}" | "\u{7f}");
        let is_delete = matches!(key, "Delete" | "keycode:117");
        if is_backspace || is_delete {
            let (start, end) = if selection.0 != selection.1 {
                selection
            } else if is_backspace {
                (prev_boundary(&value, caret), caret)
            } else {
                (caret, next_boundary(&value, caret))
            };
            if start != end {
                if let Some((new_value, new_caret)) = self.apply_edit_range(&id, start, end, "") {
                    self.interaction.edit = Some(TextEditState {
                        id: focus_key,
                        anchor: new_caret,
                        caret: new_caret,
                    });
                    self.dirty = true;
                    out.push(UIEvent::TextChanged { id, value: new_value });
                }
            }
            return out;
        }

        let is_enter = matches!(
            key,
            "keycode:36" | "keycode:76" | "Enter" | "Return" | "\r" | "\n"
        );
        if is_enter {
            if is_area {
                if let Some(events) = self.insert_text_at_caret(&id, "\n") {
                    out.extend(events);
                }
            } else {
                out.push(UIEvent::InputSubmitted { id, value });
            }
            return out;
        }

        let is_left = matches!(key, "ArrowLeft" | "keycode:123");
        let is_right = matches!(key, "ArrowRight" | "keycode:124");
        let is_up = matches!(key, "ArrowUp" | "keycode:126");
        let is_down = matches!(key, "ArrowDown" | "keycode:125");
        let is_home = matches!(key, "Home" | "keycode:115");
        let is_end = matches!(key, "End" | "keycode:119");
        if is_left || is_right || is_up || is_down || is_home || is_end {
            let mut next = caret;
            if is_left {
                next = if !shift && selection.0 != selection.1 {
                    selection.0
                } else {
                    prev_boundary(&value, caret)
                };
            } else if is_right {
                next = if !shift && selection.0 != selection.1 {
                    selection.1
                } else {
                    next_boundary(&value, caret)
                };
            } else if is_home || is_end {
                let (line_start, line_end) = self.line_bounds(
                    &value,
                    caret,
                    self.editable_wrap_width(&id),
                );
                next = if is_home { line_start } else { line_end };
            } else if is_area {
                let wrap_w = self.editable_wrap_width(&id);
                let style = TextStyle::sans(EDIT_PX, WEIGHT_REGULAR);
                let (x, line_idx) = self.atlas.x_for_char_index(
                    &value,
                    caret,
                    style,
                    wrap_w,
                    LineBreak::CharWrap,
                );
                let line_h = style.size * self.atlas.line_height_em;
                let target_y = if is_up {
                    (line_idx as f32 - 1.0) * line_h + 0.5
                } else {
                    (line_idx as f32 + 1.0) * line_h + 0.5
                };
                next = self.atlas.char_index_at(
                    &value,
                    x,
                    target_y.max(0.0),
                    style,
                    wrap_w,
                    LineBreak::CharWrap,
                );
            }
            let next_anchor = if shift { anchor } else { next };
            if next != caret || next_anchor != anchor {
                self.interaction.edit = Some(TextEditState {
                    id: focus_key.clone(),
                    anchor: next_anchor,
                    caret: next,
                });
                if let Some(h) = self.interaction.histories.get_mut(&focus_key) {
                    h.break_run(next);
                }
                self.dirty = true;
            }
            return out;
        }

        // Printable characters arrive through `TextInput`, not `KeyDown`.
        out
    }

    /// Insert `text` at the caret (replacing any selection) and report the change.
    fn insert_text_at_caret(&mut self, id: &str, text: &str) -> Option<Vec<UIEvent>> {
        let (value, _) = self.editable_value(id)?;
        let focus_key = self
            .interaction
            .focused_key
            .clone()
            .unwrap_or_else(|| format!("input::{id}"));
        let edit = self
            .interaction
            .edit
            .clone()
            .unwrap_or_else(|| TextEditState::new(focus_key.clone(), value.len()));
        let (a, b) = edit.selection();
        let (new_value, caret) = self.apply_edit_range(id, a, b, text)?;
        self.interaction.edit = Some(TextEditState {
            id: focus_key,
            anchor: caret,
            caret,
        });
        self.dirty = true;
        Some(vec![UIEvent::TextChanged {
            id: id.to_string(),
            value: new_value,
        }])
    }

    /// Route raw OS input into interaction springs, kinetic scrolls, and semantic `UIEvent`s.
    /// Scroll a `ScrollContainer` by `delta_y` logical pixels. Used by
    /// headless `--actions scroll` replay, where there is no pointer gesture.
    /// Returns `false` when `id` is not a scroll container in the current frame.
    pub fn scroll_by(&mut self, id: &str, delta_y: f32) -> bool {
        let max_y = self.interaction.hits.iter().find_map(|hit| match &hit.kind {
            HitKind::ScrollContainer {
                id: hit_id,
                max_scroll_y,
            } if hit_id == id => Some(*max_scroll_y),
            _ => None,
        });
        let Some(max_y) = max_y else {
            return false;
        };
        let entry = self
            .interaction
            .scrolls
            .entry(id.to_string())
            .or_insert_with(|| KineticScroll {
                offset_y: 0.0,
                velocity_y: 0.0,
                max_y,
                overscroll: 0.0,
            });
        entry.max_y = max_y;
        entry.impulse(delta_y, ScrollPhysics::default());
        // Headless replay is deterministic: no fling tail after the gesture.
        entry.velocity_y = 0.0;
        self.dirty = true;
        true
    }

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
                if let Some(sid) = self.interaction.slider_drag.clone() {
                    if let Some(rect) = self.layout.get(&sid) {
                        if let Some(node) = self.scene.root.find_mut(&sid) {
                            if let NodeKind::Slider(data) = &mut node.kind {
                                let v = slider_value_at(data, rect, x);
                                if (data.value - v).abs() > 1e-6 {
                                    data.value = v;
                                    self.dirty = true;
                                    out.push(UIEvent::ValueChanged {
                                        id: sid.clone(),
                                        value: v,
                                    });
                                }
                            }
                        }
                    }
                }
                if let Some((rid, thumb)) = self.interaction.range_drag.clone() {
                    if let Some(rect) = self.layout.get(&rid) {
                        let moved = self.scene.root.find_mut(&rid).and_then(|node| {
                            let NodeKind::RangeSlider(data) = &mut node.kind else {
                                return None;
                            };
                            let (_, v) = crate::components::controls::range_thumb_at(data, rect, x);
                            let (mut low, mut high) = data.clamped();
                            if thumb == 0 {
                                low = v.min(high);
                            } else {
                                high = v.max(low);
                            }
                            if (data.low - low).abs() > 1e-6 || (data.high - high).abs() > 1e-6 {
                                data.low = low;
                                data.high = high;
                                Some((low, high))
                            } else {
                                None
                            }
                        });
                        if let Some((low, high)) = moved {
                            self.dirty = true;
                            out.push(UIEvent::RangeChanged {
                                id: rid.clone(),
                                low,
                                high,
                            });
                        }
                    }
                }
                if let Some(key) = self.interaction.pressed_key.clone() {
                    if key.starts_with("input::") || key.starts_with("textarea::") {
                        if let Some(caret) = self.caret_from_pointer(&key, x, y) {
                            let mut changed = false;
                            if let Some(edit) = self.interaction.edit.as_mut() {
                                if edit.id == key && edit.caret != caret {
                                    edit.caret = caret;
                                    changed = true;
                                }
                            }
                            if changed {
                                self.dirty = true;
                            }
                        }
                    }
                }
                let drag_update = self.interaction.drag.as_ref().and_then(|s| {
                    let dx = x - s.last[0];
                    let dy = y - s.last[1];
                    (dx.abs() > 0.01 || dy.abs() > 0.01).then(|| (s.id.clone(), dx, dy))
                });
                if let Some((did, dx, dy)) = drag_update {
                    if let Some(session) = self.interaction.drag.as_mut() {
                        session.last = [x, y];
                        session.total[0] += dx;
                        session.total[1] += dy;
                    }
                    let next = self.scene.root.find_mut(&did).and_then(|n| match &n.kind {
                        NodeKind::Draggable { axis, offset, .. } => {
                            let (ddx, ddy) = match axis {
                                crate::ir::DragAxis::Both => (dx, dy),
                                crate::ir::DragAxis::Horizontal => (dx, 0.0),
                                crate::ir::DragAxis::Vertical => (0.0, dy),
                            };
                            Some([offset[0] + ddx, offset[1] + ddy])
                        }
                        _ => None,
                    });
                    if let Some(offset) = next {
                        let _ = self.apply_patch(ScenePatch::SetDraggable {
                            id: did.clone(),
                            offset,
                        });
                        out.push(UIEvent::Dragged { id: did, dx, dy });
                    }
                }
                if let Some(swipe) = self.interaction.nav_swipe.as_mut() {
                    let width = swipe.width.max(1.0);
                    swipe.progress = ((x - swipe.start_x) / width).clamp(0.0, 1.0);
                    let id = swipe.id.clone();
                    let progress = swipe.progress;
                    let _ = self.apply_patch(ScenePatch::SetNavProgress { id, progress });
                }
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
                    self.interaction.pointer_down = true;
                    let hit = self
                        .interaction
                        .hit_topmost_interactive(x, y)
                        .map(|(k, kind)| {
                            (
                                k,
                                matches!(kind, HitKind::Input { .. } | HitKind::TextArea { .. }),
                            )
                        });
                    /* Auxiliary controls (on-screen keyboard keys) opt out of
                     * focus stealing so typing keeps the field focused. */
                    let preserves = self
                        .interaction
                        .hit_topmost_interactive(x, y)
                        .map(|(_, kind)| match kind {
                            HitKind::Button { id } | HitKind::Pressable { id } => self
                                .scene
                                .root
                                .find(id)
                                .map(|n| n.preserve_focus)
                                .unwrap_or(false),
                            _ => false,
                        })
                        .unwrap_or(false);
                    let new_focus = match &hit {
                        Some((k, true)) => Some(k.clone()),
                        _ if preserves => self.interaction.focused_key.clone(),
                        _ => None,
                    };
                    if self.interaction.focused_key != new_focus {
                        self.interaction.focused_key = new_focus;
                        self.dirty = true;
                    }
                    let hit_key = hit.map(|(k, _)| k);
                    if hit_key != self.interaction.pressed_key {
                        self.interaction.pressed_key = hit_key.clone();
                        self.dirty = true;
                    }
                    // Place the caret where the pointer landed (or focus the
                    // field at its end when the text is wider than the box).
                    match hit_key.as_deref() {
                        Some(key)
                            if key.starts_with("input::") || key.starts_with("textarea::") =>
                        {
                            let caret = self.caret_from_pointer(key, x, y).unwrap_or(0);
                            let taps = self.tap_count(key, x, y);
                            let (id, is_area) = key
                                .split_once("::")
                                .map(|(kind, id)| (id.to_string(), kind == "textarea"))
                                .unwrap_or_else(|| (key.to_string(), false));
                            let (anchor, caret) = match (taps, self.editable_value(&id)) {
                                (2, Some((value, _))) => word_range(&value, caret),
                                (3, Some((value, _))) if is_area => self.line_bounds(
                                    &value,
                                    caret,
                                    self.editable_wrap_width(&id),
                                ),
                                // A one-line field has no lines to select.
                                (3, Some((value, _))) => (0, value.len()),
                                _ => (caret, caret),
                            };
                            self.interaction.edit = Some(TextEditState {
                                id: key.to_string(),
                                anchor,
                                caret,
                            });
                            if let Some(h) = self.interaction.histories.get_mut(key) {
                                h.break_run(caret);
                            }
                            self.dirty = true;
                        }
                        _ => {
                            if self.interaction.edit.take().is_some() {
                                self.dirty = true;
                            }
                        }
                    }
                    if hit_key.is_none() {
                        if let Some((pid, prect)) = self.interaction.hit_plot_area(x, y) {
                            self.interaction.dragging_plot = Some((pid, x, prect.w));
                        }
                    }
                    let down_kind = self
                        .interaction
                        .hit_topmost_interactive(x, y)
                        .map(|(_, k)| k.clone());
                    match down_kind {
                        Some(HitKind::Slider { id }) => {
                            self.interaction.slider_drag = Some(id.clone());
                            if let Some(rect) = self.layout.get(&id) {
                                if let Some(node) = self.scene.root.find_mut(&id) {
                                    if let NodeKind::Slider(data) = &mut node.kind {
                                        let v = slider_value_at(data, rect, x);
                                        if (data.value - v).abs() > 1e-6 {
                                            data.value = v;
                                            self.dirty = true;
                                            out.push(UIEvent::ValueChanged {
                                                id: id.clone(),
                                                value: v,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        Some(HitKind::RangeSlider { id }) => {
                            if let Some(rect) = self.layout.get(&id) {
                                let grabbed = self.scene.root.find_mut(&id).and_then(|node| {
                                    let NodeKind::RangeSlider(data) = &mut node.kind else {
                                        return None;
                                    };
                                    let (thumb, v) =
                                        crate::components::controls::range_thumb_at(data, rect, x);
                                    let (mut low, mut high) = data.clamped();
                                    if thumb == 0 {
                                        low = v.min(high);
                                    } else {
                                        high = v.max(low);
                                    }
                                    data.low = low;
                                    data.high = high;
                                    Some((thumb, low, high))
                                });
                                if let Some((thumb, low, high)) = grabbed {
                                    self.interaction.range_drag = Some((id.clone(), thumb));
                                    self.dirty = true;
                                    out.push(UIEvent::RangeChanged {
                                        id: id.clone(),
                                        low,
                                        high,
                                    });
                                }
                            }
                        }
                        Some(HitKind::Draggable { id }) => {
                            self.interaction.drag = Some(DragSession {
                                id,
                                last: [x, y],
                                total: [0.0, 0.0],
                            });
                        }
                        Some(HitKind::NavEdge { id }) => {
                            let width = self
                                .layout
                                .get(&id)
                                .map(|r| r.w)
                                .unwrap_or(self.viewport[0]);
                            self.interaction.nav_swipe = Some(NavSwipe {
                                id,
                                start_x: x,
                                width,
                                progress: 0.0,
                            });
                        }
                        _ => {}
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
                    self.interaction.pointer_down = false;
                    self.interaction.dragging_plot = None;
                    self.interaction.slider_drag = None;
                    self.interaction.range_drag = None;
                    self.interaction.drag = None;
                    if let Some(swipe) = self.interaction.nav_swipe.take() {
                        let id = swipe.id.clone();
                        let route = self
                            .scene
                            .root
                            .find_mut(&id)
                            .and_then(|n| match &n.kind {
                                NodeKind::NavStack { routes, active, .. } => {
                                    routes.get(*active).map(|r| r.id.clone())
                                }
                                _ => None,
                            })
                            .unwrap_or_default();
                        if crate::components::nav::should_commit_back(swipe.progress, 0.4) {
                            let _ = self.apply_patch(ScenePatch::NavPop { id: id.clone() });
                            out.push(UIEvent::BackRequested { id, route });
                        } else {
                            let _ = self.apply_patch(ScenePatch::SetNavProgress {
                                id,
                                progress: 0.0,
                            });
                        }
                    }
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
                                        let _ = self.apply_patch(ScenePatch::SetTheme { mode: next });
                                        out.push(UIEvent::ThemeChanged { id, mode: next });
                                    }
                                    HitKind::Switch { id } => {
                                        if let Some(node) = self.scene.root.find_mut(&id) {
                                            if let NodeKind::Switch { checked, .. } = &mut node.kind
                                            {
                                                let next = !*checked;
                                                *checked = next;
                                                self.dirty = true;
                                                out.push(UIEvent::SwitchToggled {
                                                    id,
                                                    checked: next,
                                                });
                                            }
                                        }
                                    }
                                    HitKind::Input { .. } => {}
                                    HitKind::TableHeader {
                                        table_id,
                                        col_index,
                                    } => {
                                        if self.apply_patch(ScenePatch::SortTable {
                                            id: table_id.clone(),
                                            col_index,
                                        }) {
                                            let asc = self
                                                .scene
                                                .root
                                                .find_mut(&table_id)
                                                .and_then(|n| match &n.kind {
                                                    NodeKind::Table(t) => {
                                                        Some(t.sort_dir == TableSortDir::Asc)
                                                    }
                                                    _ => None,
                                                })
                                                .unwrap_or(true);
                                            out.push(UIEvent::TableSorted {
                                                id: table_id,
                                                col_index,
                                                asc,
                                            });
                                        }
                                    }
                                    HitKind::TableRow { table_id, row_id } => {
                                        let _ = self.apply_patch(ScenePatch::SelectTableRow {
                                            id: table_id.clone(),
                                            row_id: Some(row_id.clone()),
                                        });
                                        out.push(UIEvent::RowSelected {
                                            id: table_id,
                                            row_id,
                                        });
                                    }
                                    HitKind::FeedItem { feed_id, item_id } => {
                                        out.push(UIEvent::FeedItemClicked {
                                            id: feed_id,
                                            item_id,
                                        });
                                    }
                                    HitKind::GraphNode { graph_id, node_id } => {
                                        let _ = self.apply_patch(ScenePatch::SelectGraphNode {
                                            id: graph_id.clone(),
                                            node_id: Some(node_id.clone()),
                                        });
                                        out.push(UIEvent::GraphNodeClicked {
                                            id: graph_id,
                                            node_id,
                                        });
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
                                    HitKind::ListItem {
                                        list_id,
                                        index,
                                        item_id,
                                    } => {
                                        out.push(UIEvent::ListItemClicked {
                                            id: list_id,
                                            index,
                                            item_id,
                                        });
                                    }
                                    HitKind::Pressable { id } => {
                                        out.push(UIEvent::Click { id });
                                    }
                                    HitKind::OverlayBackdrop { id } => {
                                        out.push(UIEvent::Click { id });
                                    }
                                    HitKind::Slider { id } => {
                                        if let Some(node) = self.scene.root.find_mut(&id) {
                                            if let NodeKind::Slider(data) = &node.kind {
                                                out.push(UIEvent::ValueChanged {
                                                    id,
                                                    value: data.value,
                                                });
                                            }
                                        }
                                    }
                                    HitKind::InputClear { id } => {
                                        if self.apply_patch(ScenePatch::SetInput {
                                            id: id.clone(),
                                            value: String::new(),
                                        }) {
                                            self.interaction.edit = Some(TextEditState::new(
                                                format!("input::{id}"),
                                                0,
                                            ));
                                            out.push(UIEvent::TextChanged {
                                                id,
                                                value: String::new(),
                                            });
                                        }
                                    }
                                    HitKind::RangeSlider { .. }
                                    | HitKind::TextArea { .. }
                                    | HitKind::Draggable { .. }
                                    | HitKind::NavEdge { .. }
                                    | HitKind::PlotArea { .. }
                                    | HitKind::ScrollContainer { .. } => {}
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
                            overscroll: 0.0,
                        });
                    entry.max_y = max_y;
                    entry.impulse(delta_y, ScrollPhysics::default());
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
            InputEvent::KeyDown { key, modifiers } => {
                if let Some((id, is_area)) = focused_editable(&self.interaction.focused_key) {
                    out.extend(self.handle_editable_key(id, is_area, &key, modifiers));
                }
            }
            InputEvent::TextInput { text } | InputEvent::Paste { text } => {
                if let Some((id, is_area)) = focused_editable(&self.interaction.focused_key) {
                    let clean: String = text
                        .chars()
                        .filter(|c| (*c == '\n' && is_area) || !c.is_control())
                        .collect();
                    if !clean.is_empty() {
                        if let Some(events) = self.insert_text_at_caret(&id, &clean) {
                            self.dirty = true;
                            out.extend(events);
                        }
                    }
                }
            }
            InputEvent::Back => {
                if let Some((id, route)) = first_active_nav(&self.scene.root) {
                    let _ = self.apply_patch(ScenePatch::NavPop { id: id.clone() });
                    out.push(UIEvent::BackRequested { id, route });
                }
            }
        }
        out
    }

    /// Advance 120Hz springs & kinetic scrolls and compile `DrawList` when dirty or animating.
    pub fn tick_and_compile(&mut self, dt: f32, draw_list: &mut DrawList) -> CompileStatus {
        self.now_s += dt.clamp(0.0, 1.0) as f64;
        self.tick_inner(dt, draw_list)
    }

    /// Present one frame after the host slept to [`Self::next_wake_ms`].
    ///
    /// `anim_dt` stays clamped for spring/scroll stability, while `elapsed_s`
    /// is the real wall time that passed. Advancing the engine clock by the
    /// wall time is what makes a 500 ms caret-deadline sleep toggle the caret
    /// exactly once instead of ten times (the loop clamps `anim_dt` to 50 ms).
    pub fn tick_with_elapsed(
        &mut self,
        anim_dt: f32,
        elapsed_s: f64,
        draw_list: &mut DrawList,
    ) -> CompileStatus {
        self.now_s += elapsed_s.clamp(0.0, 1.0);
        self.tick_inner(anim_dt, draw_list)
    }

    fn tick_inner(&mut self, dt: f32, draw_list: &mut DrawList) -> CompileStatus {
        self.note_caret_activity();
        self.step_caret_blink();
        let fx_was_active = self.fx.is_animating();
        self.fx.step(dt);
        let scroll_moving = self.interaction.step_scrolls(dt);
        let spring_moving = self.springs.step_all(dt);

        if !self.dirty
            && !scroll_moving
            && !spring_moving
            && !fx_was_active
            && !self.fx.is_animating()
        {
            self.publish_frame_schedule(draw_list, false);
            return CompileStatus {
                compiled: false,
                animating: false,
            };
        }

        self.dirty = false;
        let scrolls = self.visual_scrolls();
        self.layout.solve_full(
            &self.scene.root,
            self.viewport,
            self.top_inset,
            &self.atlas,
            &scrolls,
            self.safe_insets,
        );
        self.interaction.clear_hits();
        self.springs.begin_frame();
        self.fx.begin_compile();
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

        // Move the root out instead of deep-cloning the whole scene: `emit_node`
        // borrows it immutably while mutating the engine, and a clone would copy
        // every virtualized list row on every frame (O(n) in the 50k-row gate).
        let root_node = std::mem::replace(
            &mut self.scene.root,
            Node::new("__compiling__", NodeKind::Spacer),
        );
        self.emit_node(&root_node, ROOT_PATH_HASH, 0.0, 0.0, draw_list);
        self.scene.root = root_node;
        self.fx.emit_particles(draw_list);
        if self.touch_reticle && self.interaction.pointer_down {
            self.emit_touch_reticle(draw_list);
        }
        if self.show_bounds {
            self.emit_bounds_overlay(draw_list);
        }
        self.springs.prune_unvisited();
        self.fx.end_frame();

        let still_animating = self.springs.has_active()
            || self.fx.is_animating()
            || self
                .interaction
                .scrolls
                .values()
                .any(|s| s.velocity_y.abs() >= 4.0);
        self.publish_frame_schedule(draw_list, still_animating);

        CompileStatus {
            compiled: true,
            animating: still_animating,
        }
    }

    /// Publish the scheduling contract on the draw list itself, so a host
    /// (including one in another repository) needs no engine internals to
    /// decide between "present now", "wake in N ms" and "sleep".
    fn publish_frame_schedule(&self, draw_list: &mut DrawList, animating: bool) {
        let wake_ms = self.next_wake_ms();
        draw_list.needs_next_frame = animating || self.is_dirty_or_animating() || wake_ms.is_some();
        draw_list.next_wake_s = wake_ms.map(|ms| self.now_s + ms / 1000.0);
    }

    /// Live-only touch indicator: a translucent ring plus a solid dot at the
    /// pointer. Drawn after the scene so it is never occluded, and never part
    /// of the scene tree (so layout/audit are unaffected).
    /// Live-only layout inspector: a thin outline around every interactive
    /// region, drawn after the scene so it is never occluded. Not part of the
    /// scene tree, so layout/audit are unaffected.
    fn emit_bounds_overlay(&mut self, dl: &mut DrawList) {
        for hit in &self.interaction.hits {
            let r = hit.rect;
            if r.w <= 0.0 || r.h <= 0.0 {
                continue;
            }
            dl.push_quad(QuadInstance {
                rect: [r.x, r.y, r.w, r.h],
                radii: [0.0; 4],
                color_top: 0,
                color_bottom: 0,
                border_color: 0x00E5FFCC,
                params: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            });
        }
    }

    fn emit_touch_reticle(&mut self, dl: &mut DrawList) {
        let [x, y] = self.interaction.pointer;
        let ring = 22.0;
        dl.push_quad(QuadInstance {
            rect: [x - ring, y - ring, ring * 2.0, ring * 2.0],
            radii: [ring; 4],
            color_top: 0xFFFFFF66,
            color_bottom: 0xFFFFFF66,
            border_color: 0x00000055,
            params: [1.5, 0.0, 0.0, 1.0],
            ..Default::default()
        });
        let dot = 7.0;
        dl.push_quad(QuadInstance {
            rect: [x - dot, y - dot, dot * 2.0, dot * 2.0],
            radii: [dot; 4],
            color_top: 0xFFFFFFCC,
            color_bottom: 0xFFFFFFCC,
            ..Default::default()
        });
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

        // Effects never change layout: the motion layer rewrites only the primitives this node
        // emits, while hit-testing/audit keep observing the untouched `ComputedRect`.
        let pressed = self
            .interaction
            .pressed_key
            .as_deref()
            .map(|k| k.rsplit("::").next() == Some(node.id.as_str()))
            .unwrap_or(false);
        let hovered = self
            .interaction
            .hovered_key
            .as_deref()
            .map(|k| k.rsplit("::").next() == Some(node.id.as_str()))
            .unwrap_or(false);
        let node_fx = if node.fx.is_empty() {
            // Impulse bursts can target any node: resolve here so the origin is the laid-out
            // rect even when the target carries no fx specs of its own.
            if self.fx.has_pending_burst(&node.id) {
                self.fx.resolve_burst(&node.id, rect);
            }
            NodeFx::for_rect(rect)
        } else {
            self.fx.node_fx(&node.id, rect, &node.fx, pressed, self.interaction.pointer)
        };
        let mark = DrawMark::capture(dl);
        if !node.fx.is_empty() {
            self.fx.emit_backdrop(&node.fx, rect, &self.theme, dl);
        }

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
                let animated = if node.fx.is_empty() {
                    None
                } else {
                    self.fx.text_override(&node.id, &node.fx, content)
                };
                let content = animated.as_deref().unwrap_or(content);
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                let tx = rect.x + node.padding.left;
                let inner_h = (rect.h - node.padding.vertical()).max(0.0);
                let ty = if !*wrap && !content.contains('\n') && inner_h > *size {
                    self.atlas
                        .centered_pen_y(rect.y + node.padding.top, inner_h, *size)
                        - self.atlas.ascender_em * *size
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
            NodeKind::Switch {
                checked,
                label,
                tone,
            } => {
                emit_switch(
                    &node.id,
                    *checked,
                    label.as_deref(),
                    *tone,
                    rect,
                    &self.theme,
                    &self.atlas,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Input {
                value,
                placeholder,
                tone,
                prefix_icon,
                clearable,
                error,
                ..
            } => {
                emit_input(
                    &node.id,
                    value,
                    placeholder,
                    *tone,
                    prefix_icon.as_deref(),
                    *clearable,
                    error.as_deref(),
                    rect,
                    &self.theme,
                    &self.atlas,
                    &self.icons,
                    &mut self.springs,
                    &mut self.interaction,
                    dl,
                );
            }
            NodeKind::Stat(data) => {
                let owned = if node.fx.is_empty() {
                    None
                } else {
                    self.fx
                        .number_roll_value(&node.id, &node.fx, data.value)
                        .map(|(value, text)| {
                            let mut d = data.clone();
                            d.value = value;
                            d.value_text = Some(text);
                            d
                        })
                };
                emit_stat(
                    &node.id,
                    owned.as_ref().unwrap_or(data),
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
            NodeKind::Table(data) => {
                emit_table(
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
            NodeKind::Feed(data) => {
                emit_feed(
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
            NodeKind::Graph(data) => {
                emit_graph(
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
            NodeKind::CodeText(data) => {
                emit_code(
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
            NodeKind::RichText(data) => {
                let spans = rich_spans(data);
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                let max_w = if data.wrap {
                    Some((rect.w - node.padding.horizontal()).max(16.0))
                } else {
                    None
                };
                self.atlas.layout_rich(
                    &spans,
                    rect.x + node.padding.left,
                    rect.y + node.padding.top,
                    max_w,
                    data.line_height,
                    dl,
                );
                dl.pop_clip();
            }
            NodeKind::Icon { name, size, color } => {
                let tint = color.unwrap_or(self.theme.text_primary);
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                emit_icon(&self.icons, name, rect, *size, tint, dl);
                dl.pop_clip();
            }
            NodeKind::Image {
                src,
                fit,
                radius,
                tint,
                placeholder,
                alt: _,
            } => {
                let handle = self.images.get(key_for(src));
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                emit_image(
                    handle,
                    rect,
                    *fit,
                    *radius,
                    *tint,
                    *placeholder,
                    &self.theme,
                    dl,
                );
                dl.pop_clip();
            }
            NodeKind::Stack {
                style, children, ..
            } => {
                if let Some(s) = style {
                    self.emit_box(rect, s, 0.0, dl);
                }
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                let mut order: Vec<usize> = children
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.visible)
                    .map(|(i, _)| i)
                    .collect();
                order.sort_by_key(|&i| children[i].layout.z);
                for slot in order {
                    let c = &children[slot];
                    let ch_hash = child_path_hash(path_hash, slot, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                }
                dl.pop_clip();
            }
            NodeKind::SafeArea { child, .. } => {
                if let Some(c) = child {
                    dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                    dl.pop_clip();
                }
            }
            NodeKind::Overlay {
                scrim,
                dismissable,
                child,
                ..
            } => {
                if let Some(color) = scrim {
                    dl.push_quad(QuadInstance {
                        rect: [rect.x, rect.y, rect.w, rect.h],
                        radii: [0.0; 4],
                        color_top: *color,
                        color_bottom: *color,
                        border_color: 0,
                        shadow_color: 0,
                        params: [0.0, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                }
                if *dismissable {
                    self.interaction.push_hit(
                        rect,
                        HitKind::OverlayBackdrop {
                            id: node.id.clone(),
                        },
                    );
                }
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                }
            }
            NodeKind::NavStack {
                routes,
                active,
                progress,
                swipe_back,
            } => {
                let p = progress.clamp(0.0, 1.0);
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                if p > 0.0 && *active > 0 {
                    if let Some(prev) = routes.get(*active - 1) {
                        let h = child_path_hash(path_hash, *active - 1, &prev.child.id);
                        self.emit_node(&prev.child, h, ox, oy, dl);
                    }
                }
                if let Some(route) = routes.get(*active) {
                    let h = child_path_hash(path_hash, *active, &route.child.id);
                    self.emit_node(&route.child, h, ox, oy, dl);
                }
                dl.pop_clip();
                if *swipe_back && *active > 0 {
                    let edge = ComputedRect {
                        x: rect.x,
                        y: rect.y,
                        w: 24.0f32.min(rect.w),
                        h: rect.h,
                        content_h: rect.h,
                        folded: false,
                    };
                    self.interaction.push_hit(
                        edge,
                        HitKind::NavEdge {
                            id: node.id.clone(),
                        },
                    );
                }
            }
            NodeKind::ListView(data) => {
                if let Some(s) = &data.style {
                    self.emit_box(rect, s, 0.0, dl);
                }
                let max_scroll = vlist::max_scroll(data, rect.h);
                self.interaction.push_hit(
                    rect,
                    HitKind::ScrollContainer {
                        id: node.id.clone(),
                        max_scroll_y: max_y_clamp(max_scroll),
                    },
                );
                let scroll = {
                    let s = self
                        .interaction
                        .scrolls
                        .entry(node.id.clone())
                        .or_default();
                    s.max_y = max_scroll;
                    s.offset_y = s.offset_y.clamp(0.0, max_scroll);
                    s.offset_y
                };
                dl.push_clip(ClipRect::new(rect.x, rect.y, rect.w, rect.h));
                let (start, end) = self
                    .layout
                    .list_range(&node.id)
                    .unwrap_or((0, data.items.len()));
                for slot in start..end.min(data.items.len()) {
                    let item = &data.items[slot];
                    let h_hash = child_path_hash(path_hash, slot, &item.child.id);
                    if let Some(ir) = self.layout.get_path(h_hash) {
                        let mut row = ir;
                        row.y -= scroll;
                        self.interaction.push_hit(
                            row,
                            HitKind::ListItem {
                                list_id: node.id.clone(),
                                index: slot,
                                item_id: item.id.clone(),
                            },
                        );
                    }
                    self.emit_node(&item.child, h_hash, ox, oy - scroll, dl);
                }
                if max_scroll > 1.0 {
                    let track_h = (rect.h - 12.0).max(24.0);
                    let content_h = vlist::total_height(data);
                    let thumb_h = ((rect.h / content_h) * track_h).clamp(24.0, track_h);
                    let thumb_t = scroll / max_scroll;
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
            NodeKind::RangeSlider(data) => {
                emit_range_slider(
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
            NodeKind::Progress(data) => {
                emit_progress(
                    data,
                    rect,
                    &self.theme,
                    &self.atlas,
                    self.now_s,
                    dl,
                );
            }
            NodeKind::TextArea(data) => {
                emit_textarea(
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
            NodeKind::Slider(data) => {
                emit_slider(
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
            NodeKind::Pressable {
                hit_slop,
                disabled,
                child,
            } => {
                let hit_key = format!("press::{}", node.id);
                let hovered = self.interaction.hovered_key.as_deref() == Some(hit_key.as_str());
                let pressed = self.interaction.pressed_key.as_deref() == Some(hit_key.as_str());
                let target = if *disabled {
                    0.0
                } else if pressed {
                    1.0
                } else if hovered {
                    0.55
                } else {
                    0.0
                };
                let t = self
                    .springs
                    .animate(&hit_key, "hover", target, SpringConfig::SNAPPY);
                if t > 0.01 {
                    let glow = lerp_rgba8(self.theme.accent, 0x00000000, 1.0 - t * 0.25);
                    dl.push_quad(QuadInstance {
                        rect: [rect.x, rect.y, rect.w, rect.h],
                        radii: [self.theme.radius_control; 4],
                        color_top: 0,
                        color_bottom: 0,
                        border_color: glow,
                        shadow_color: 0,
                        params: [1.5, 0.0, 0.0, 1.0],
                        clip_id: 0,
                        _pad: [0; 3],
                    });
                }
                if !*disabled {
                    let slop = hit_slop.max(0.0);
                    let hit_rect = ComputedRect {
                        x: rect.x - slop,
                        y: rect.y - slop,
                        w: rect.w + slop * 2.0,
                        h: rect.h + slop * 2.0,
                        content_h: rect.content_h,
                        folded: rect.folded,
                    };
                    self.interaction.push_hit(
                        hit_rect,
                        HitKind::Pressable {
                            id: node.id.clone(),
                        },
                    );
                }
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                }
            }
            NodeKind::Draggable {
                axis: _,
                offset: _,
                inertia: _,
                child,
            } => {
                self.interaction.push_hit(
                    rect,
                    HitKind::Draggable {
                        id: node.id.clone(),
                    },
                );
                if let Some(c) = child {
                    let ch_hash = child_path_hash(path_hash, 0, &c.id);
                    self.emit_node(c, ch_hash, ox, oy, dl);
                }
            }
        }

        if !node.fx.is_empty() {
            self.fx
                .apply_text_shimmer(&node.fx, dl, mark.glyphs, rect);
            self.fx.emit_effects(
                &node.id,
                rect,
                &node.fx,
                &self.theme,
                self.interaction.pointer,
                hovered,
                dl,
            );
            mark.apply_fx(dl, &node_fx);
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

/// Copy user-edited values from `old` onto matching ids in `new`.
fn preserve_scene_state(old: &Node, new: &mut Node) {
    if let Some(prev) = old.find(&new.id) {
        match (&prev.kind, &mut new.kind) {
            (
                NodeKind::Input { value: ov, .. },
                NodeKind::Input { value: nv, .. },
            ) => *nv = ov.clone(),
            (
                NodeKind::TextArea(crate::ir::TextAreaData { value: ov, .. }),
                NodeKind::TextArea(crate::ir::TextAreaData { value: nv, .. }),
            ) => *nv = ov.clone(),
            (
                NodeKind::Switch { checked: oc, .. },
                NodeKind::Switch { checked: nc, .. },
            ) => *nc = *oc,
            (
                NodeKind::Slider(crate::ir::SliderData { value: ov, .. }),
                NodeKind::Slider(crate::ir::SliderData { value: nv, .. }),
            ) => *nv = *ov,
            (
                NodeKind::Tabs {
                    active: oa,
                    items: oi,
                },
                NodeKind::Tabs {
                    active: na,
                    items: ni,
                },
            ) => {
                if *oa < oi.len() && *oa < ni.len() {
                    *na = *oa;
                }
            }
            (
                NodeKind::NavStack {
                    routes: or,
                    active: oa,
                    ..
                },
                NodeKind::NavStack {
                    routes: nr,
                    active: na,
                    ..
                },
            ) => {
                if let Some(active_route) = or.get(*oa) {
                    if let Some(idx) = nr.iter().position(|r| r.id == active_route.id) {
                        *na = idx;
                    }
                }
            }
            _ => {}
        }
    }
    match &mut new.kind {
        NodeKind::Container { children, .. }
        | NodeKind::Grid { children, .. }
        | NodeKind::Stack { children, .. } => {
            for c in children {
                preserve_scene_state(old, c);
            }
        }
        NodeKind::Tabs { items, .. } => {
            for item in items {
                preserve_scene_state(old, &mut item.child);
            }
        }
        NodeKind::NavStack { routes, .. } => {
            for route in routes {
                preserve_scene_state(old, &mut route.child);
            }
        }
        NodeKind::ListView(data) => {
            for item in &mut data.items {
                preserve_scene_state(old, &mut item.child);
            }
        }
        NodeKind::Surface { child, .. }
        | NodeKind::SafeArea { child, .. }
        | NodeKind::Overlay { child, .. }
        | NodeKind::Pressable { child, .. }
        | NodeKind::Draggable { child, .. } => {
            if let Some(c) = child {
                preserve_scene_state(old, c);
            }
        }
        _ => {}
    }
}

/// Font size used by `Input`/`TextArea` in `components::controls`.
const EDIT_PX: f32 = 12.5;

/// Largest UTF-8 boundary at or below `index`.
#[inline]
fn floor_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Byte range of the word (letters, digits, `_`) around `index`.
///
/// Whitespace and punctuation select their own character, matching the native
/// `double-click` behaviour on AppKit/UIKit/Android/Win32.
fn word_range(text: &str, index: usize) -> (usize, usize) {
    let index = floor_boundary(text, index.min(text.len()));
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let at = text[index..].chars().next();
    match at {
        Some(c) if is_word(c) => {
            let mut start = index;
            for (i, ch) in text[..index].char_indices().rev() {
                if !is_word(ch) {
                    break;
                }
                start = i;
            }
            let mut end = index + c.len_utf8();
            for ch in text[end..].chars() {
                if !is_word(ch) {
                    break;
                }
                end += ch.len_utf8();
            }
            (start, end)
        }
        Some(c) => (index, index + c.len_utf8()),
        None => (index, index),
    }
}

/// Byte offset of the character before `index` (0 at the start).
#[inline]
fn prev_boundary(text: &str, index: usize) -> usize {
    let index = floor_boundary(text, index);
    text[..index]
        .char_indices()
        .next_back()
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Byte offset of the character after `index` (len at the end).
#[inline]
fn next_boundary(text: &str, index: usize) -> usize {
    let index = floor_boundary(text, index);
    text[index..]
        .chars()
        .next()
        .map(|c| index + c.len_utf8())
        .unwrap_or(text.len())
}

/// Split a focused interaction key into `(node_id, is_multiline)`.
#[inline]
fn focused_editable(key: &Option<String>) -> Option<(String, bool)> {
    let key = key.as_deref()?;
    if let Some(id) = key.strip_prefix("input::") {
        Some((id.to_string(), false))
    } else {
        key.strip_prefix("textarea::")
            .map(|id| (id.to_string(), true))
    }
}

/// Append undecoded image sources in paint order.
fn collect_pending_images(node: &Node, atlas: &ImageAtlas, out: &mut Vec<String>) {
    match &node.kind {
        NodeKind::Image { src, .. } => {
            if !out.iter().any(|s| s == src) && atlas.peek(key_for(src)).is_none() {
                out.push(src.clone());
            }
        }
        NodeKind::Container { children, .. }
        | NodeKind::Grid { children, .. }
        | NodeKind::Stack { children, .. } => {
            for c in children {
                collect_pending_images(c, atlas, out);
            }
        }
        NodeKind::Tabs { items, active } => {
            if let Some(item) = items.get(*active) {
                collect_pending_images(&item.child, atlas, out);
            }
        }
        NodeKind::NavStack { routes, active, .. } => {
            if let Some(route) = routes.get(*active) {
                collect_pending_images(&route.child, atlas, out);
            }
        }
        NodeKind::ListView(data) => {
            for item in &data.items {
                collect_pending_images(&item.child, atlas, out);
            }
        }
        NodeKind::Surface { child, .. }
        | NodeKind::SafeArea { child, .. }
        | NodeKind::Overlay { child, .. }
        | NodeKind::Pressable { child, .. }
        | NodeKind::Draggable { child, .. } => {
            if let Some(c) = child {
                collect_pending_images(c, atlas, out);
            }
        }
        _ => {}
    }
}

/// First `NavStack` below its root route: `(nav_id, active_route_id)`.
fn first_active_nav(node: &Node) -> Option<(String, String)> {
    match &node.kind {
        NodeKind::NavStack { routes, active, .. } => {
            if *active > 0 {
                if let Some(route) = routes.get(*active) {
                    return Some((node.id.clone(), route.id.clone()));
                }
            }
            routes
                .get(*active)
                .and_then(|r| first_active_nav(&r.child))
        }
        NodeKind::Container { children, .. }
        | NodeKind::Grid { children, .. }
        | NodeKind::Stack { children, .. } => {
            children.iter().find_map(first_active_nav)
        }
        NodeKind::Tabs { items, active } => items
            .get(*active)
            .and_then(|i| first_active_nav(&i.child)),
        NodeKind::ListView(data) => data.items.iter().find_map(|i| first_active_nav(&i.child)),
        NodeKind::Surface { child, .. }
        | NodeKind::SafeArea { child, .. }
        | NodeKind::Overlay { child, .. }
        | NodeKind::Pressable { child, .. }
        | NodeKind::Draggable { child, .. } => child.as_deref().and_then(first_active_nav),
        _ => None,
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
    fn set_table_rows_preserves_kinetic_scroll_interaction_state() {
        use crate::components::table::{TableColumn, TableCell, TableData, TableRow, TableSortDir};
        use crate::hit_test::KineticScroll;

        let table = Node::new(
            "tbl",
            NodeKind::Table(TableData {
                columns: vec![TableColumn::new("n", "N")],
                rows: vec![TableRow::new("a", vec![TableCell::new("1")])],
                sort_col: Some(0),
                sort_dir: TableSortDir::Desc,
                ..TableData::default()
            }),
        );
        let scene = Scene::new(
            "T",
            ThemeMode::Light,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![table],
                },
            ),
        );
        let mut engine = UiEngine::new(scene, [640.0, 480.0]);
        engine.interaction.scrolls.insert(
            "tbl".into(),
            KineticScroll {
                offset_y: 42.0,
                velocity_y: 0.0,
                max_y: 300.0,
                ..KineticScroll::default()
            },
        );

        assert!(engine.apply_patch(ScenePatch::SetTableRows {
            id: "tbl".into(),
            rows: vec![
                TableRow::new("a", vec![TableCell::new("2")]),
                TableRow::new("b", vec![TableCell::new("10")]),
            ],
            selected_row: None,
        }));

        assert_eq!(
            engine.interaction.scrolls.get("tbl").map(|s| s.offset_y),
            Some(42.0)
        );
        let NodeKind::Table(table) = &engine.scene.root.find_mut("tbl").unwrap().kind else {
            panic!("table missing");
        };
        assert_eq!(table.sort_dir, TableSortDir::Desc);
        let values: Vec<&str> = table.rows.iter().map(|r| r.cells[0].text.as_str()).collect();
        assert_eq!(values, vec!["10", "2"]);
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
        assert!(engine.apply_patch(ScenePatch::SetTheme { mode: ThemeMode::Light }));
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

    #[test]
    fn table_virtualizes_large_rows_and_sorts_and_selects() {
        use crate::components::table::{TableCell, TableColumn, TableRow};
        use crate::ui;

        let rows: Vec<TableRow> = (0..5_000)
            .map(|i| TableRow::new(format!("r{i}"), vec![TableCell::new(format!("{}", 5_000 - i))]))
            .collect();
        let scene = ui::app("Table Test")
            .add(
                ui::table("tbl")
                    .title("Queue")
                    .columns([TableColumn::new("val", "Val")])
                    .rows(rows),
            )
            .build();

        let mut engine = UiEngine::new(scene, [600.0, 360.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        // Virtualized: 5,000 rows must emit only visible viewport rows (< 60 quads total)
        assert!(dl.quads.len() < 60, "expected virtualized quads < 60, got {}", dl.quads.len());

        // Find column header hit rect & click to sort Asc
        let hdr_rect = engine
            .interaction
            .hits
            .iter()
            .find_map(|h| match &h.kind {
                HitKind::TableHeader { col_index: 0, .. } => Some(h.rect),
                _ => None,
            })
            .unwrap();
        let hx = hdr_rect.x + hdr_rect.w * 0.5;
        let hy = hdr_rect.y + hdr_rect.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x: hx, y: hy, button: 0 });
        let evs = engine.handle_input(InputEvent::PointerUp { x: hx, y: hy, button: 0 });
        assert!(matches!(
            evs.as_slice(),
            [UIEvent::TableSorted { id, col_index: 0, asc: true }] if id == "tbl"
        ));

        // First row after ascending numeric sort must be "1" (r4999)
        if let Some(node) = engine.scene.root.find_mut("tbl") {
            if let NodeKind::Table(t) = &node.kind {
                assert_eq!(t.rows[0].cells[0].text, "1");
            }
        }
    }

    #[test]
    fn feed_appends_and_clicks_items() {
        use crate::components::feed::FeedItem;
        use crate::ui;

        let scene = ui::app("Feed Test")
            .add(
                ui::feed("fd")
                    .title("Events")
                    .item(FeedItem::new("i1", "Initial event").badge("OK").tone(crate::theme::Tone::Ok)),
            )
            .build();
        let mut engine = UiEngine::new(scene, [520.0, 360.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);

        assert!(engine.apply_patch(ScenePatch::AppendFeedItem {
            id: "fd".into(),
            item: FeedItem::new("i2", "Newest event").body("Details"),
            max_items: Some(50),
        }));
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);

        let item_rect = engine
            .interaction
            .hits
            .iter()
            .find_map(|h| match &h.kind {
                HitKind::FeedItem { item_id, .. } if item_id == "i2" => Some(h.rect),
                _ => None,
            })
            .unwrap();
        let x = item_rect.x + 12.0;
        let y = item_rect.y + 12.0;
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let evs = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert!(matches!(
            evs.as_slice(),
            [UIEvent::FeedItemClicked { id, item_id }] if id == "fd" && item_id == "i2"
        ));
    }

    #[test]
    fn graph_auto_layouts_dag_and_emits_node_click() {
        use crate::components::graph::{GraphEdge, GraphNode};
        use crate::ui;

        let scene = ui::app("Graph Test")
            .add(
                ui::graph("g1")
                    .title("DAG")
                    .node(GraphNode::new("a", "Start"))
                    .node(GraphNode::new("b", "Step"))
                    .edge(GraphEdge::new("a", "b").active(true)),
            )
            .build();
        let mut engine = UiEngine::new(scene, [640.0, 360.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(dl.strokes.len() >= 18, "expected Bezier curve + arrow strokes");

        let nb_rect = engine
            .interaction
            .hits
            .iter()
            .find_map(|h| match &h.kind {
                HitKind::GraphNode { node_id, .. } if node_id == "b" => Some(h.rect),
                _ => None,
            })
            .unwrap();
        let x = nb_rect.x + nb_rect.w * 0.5;
        let y = nb_rect.y + nb_rect.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let evs = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert!(matches!(
            evs.as_slice(),
            [UIEvent::GraphNodeClicked { id, node_id }] if id == "g1" && node_id == "b"
        ));
    }

    #[test]
    fn code_text_renders_gutter_highlights_and_updates() {
        use crate::ui;

        let scene = ui::app("Code Test")
            .add(
                ui::code("c1", "let x = 42;\nreturn x;")
                    .title("main.js")
                    .language("JS")
                    .highlight(1, crate::theme::Tone::Ok),
            )
            .build();
        let mut engine = UiEngine::new(scene, [580.0, 320.0]);
        let mut dl = DrawList::new();
        let st = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(st.compiled);
        assert!(!dl.glyphs.is_empty());

        assert!(engine.apply_patch(ScenePatch::SetCode {
            id: "c1".into(),
            content: "const ok = true;".into(),
            highlights: None,
        }));
    }

    #[test]
    fn switch_toggles_and_input_accepts_typing_backspace_and_submit() {
        use crate::ui;

        let row_n: Node = ui::row("ctrl_row")
            .fold_below(0.0)
            .add(ui::switch("sw1", false, Some("Enable")))
            .add(ui::input("in1", "Type here"))
            .into();
        let scene = Scene::new("Controls", ThemeMode::Light, row_n);

        let mut engine = UiEngine::new(scene, [600.0, 200.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);

        // Toggle switch
        let sw_r = engine.layout.get("sw1").unwrap();
        let sx = sw_r.x + 12.0;
        let sy = sw_r.y + sw_r.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x: sx, y: sy, button: 0 });
        let sw_ev = engine.handle_input(InputEvent::PointerUp { x: sx, y: sy, button: 0 });
        assert!(matches!(
            sw_ev.as_slice(),
            [UIEvent::SwitchToggled { id, checked: true }] if id == "sw1"
        ));

        // Focus Input -> Type "ab" -> Backspace -> Enter
        let in_r = engine.layout.get("in1").unwrap();
        let ix = in_r.x + in_r.w * 0.5;
        let iy = in_r.y + in_r.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x: ix, y: iy, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x: ix, y: iy, button: 0 });
        assert_eq!(engine.interaction.focused_key.as_deref(), Some("input::in1"));

        let t_ev = engine.handle_input(InputEvent::TextInput { text: "ab".into() });
        assert!(matches!(
            t_ev.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "in1" && value == "ab"
        ));
        let bs_ev = engine.handle_input(InputEvent::KeyDown {
            key: "keycode:51".into(),
            modifiers: 0,
        });
        assert!(matches!(
            bs_ev.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "in1" && value == "a"
        ));
        let sub_ev = engine.handle_input(InputEvent::KeyDown {
            key: "keycode:36".into(),
            modifiers: 0,
        });
        assert!(matches!(
            sub_ev.as_slice(),
            [UIEvent::InputSubmitted { id, value }] if id == "in1" && value == "a"
        ));
    }

    #[test]
    fn preserve_focus_button_keeps_input_focused() {
        use crate::ui;

        let mut key = ui::button("emu_key", "Q");
        key.preserve_focus = true;
        let scene = ui::app("Keyboard")
            .add(ui::col("body").add(ui::input("field", "Type")).add(key))
            .build();
        let mut engine = UiEngine::new(scene, [360.0, 240.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);

        let field = engine.layout.get("field").unwrap();
        let fx = field.x + field.w * 0.5;
        let fy = field.y + field.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x: fx, y: fy, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x: fx, y: fy, button: 0 });
        assert_eq!(engine.interaction.focused_key.as_deref(), Some("input::field"));

        let key_r = engine.layout.get("emu_key").unwrap();
        let kx = key_r.x + key_r.w * 0.5;
        let ky = key_r.y + key_r.h * 0.5;
        let _ = engine.handle_input(InputEvent::PointerDown { x: kx, y: ky, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x: kx, y: ky, button: 0 });
        assert_eq!(
            engine.interaction.focused_key.as_deref(),
            Some("input::field"),
            "an on-screen key must not blur the field it types into"
        );

        let typed = engine.handle_input(InputEvent::TextInput { text: "q".into() });
        assert!(matches!(
            typed.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "field" && value == "q"
        ));
    }

    #[test]
    fn scroll_by_moves_scroll_container() {
        use crate::ui;

        let rows: Vec<Node> = (0..30)
            .map(|i| ui::button(format!("row_{i}"), format!("Row {i}")))
            .collect();
        let list = Node::new(
            "list",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 6.0,
                fold_width: None,
                scrollable: true,
                style: None,
                children: rows,
            },
        );
        let scene = ui::app("Scroll").add(list).build();
        let mut engine = UiEngine::new(scene, [320.0, 400.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let before = engine
            .interaction
            .scrolls
            .get("list")
            .map(|s| s.offset_y)
            .unwrap_or(0.0);
        assert!(engine.scroll_by("list", -240.0));
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let after = engine.interaction.scrolls.get("list").unwrap().offset_y;
        assert!(
            (after - before - 240.0).abs() < 1.0,
            "scroll_by should move content by 240 px (before {before}, after {after})"
        );
        assert!(!engine.scroll_by("missing", -10.0));
    }

    #[test]
    fn touch_reticle_only_draws_while_pointer_is_down() {
        use crate::ui;

        let scene = ui::app("Reticle").add(ui::button("btn", "Tap")).build();
        let mut engine = UiEngine::new(scene, [320.0, 400.0]);
        let mut dl = DrawList::new();
        engine.set_touch_reticle(true);
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            !dl.quads.iter().any(|q| q.color_top == 0xFFFFFF66),
            "reticle must not draw before a pointer down"
        );

        let _ = engine.handle_input(InputEvent::PointerDown {
            x: 100.0,
            y: 120.0,
            button: 0,
        });
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            dl.quads.iter().any(|q| q.color_top == 0xFFFFFF66),
            "reticle ring should draw while the pointer is down"
        );

        let _ = engine.handle_input(InputEvent::PointerUp {
            x: 100.0,
            y: 120.0,
            button: 0,
        });
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            !dl.quads.iter().any(|q| q.color_top == 0xFFFFFF66),
            "reticle must disappear after pointer up"
        );
    }

    #[test]
    fn bounds_overlay_draws_interactive_outlines_only_when_enabled() {
        use crate::ui;

        let scene = ui::app("Bounds").add(ui::button("btn", "Tap")).build();
        let mut engine = UiEngine::new(scene, [320.0, 400.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            !dl.quads.iter().any(|q| q.border_color == 0x00E5FFCC),
            "bounds overlay must be off by default"
        );

        engine.set_show_bounds(true);
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            dl.quads.iter().any(|q| q.border_color == 0x00E5FFCC),
            "bounds overlay should outline interactive regions"
        );

        engine.set_show_bounds(false);
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(
            !dl.quads.iter().any(|q| q.border_color == 0x00E5FFCC),
            "bounds overlay must disappear when disabled"
        );
    }

    fn new_surface_scene() -> Scene {
        Scene::new(
            "Surfaces",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Row,
                    align: Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![
                        Node::new(
                            "stack",
                            NodeKind::Stack {
                                align: crate::ir::StackAlign::Center,
                                gap: 4.0,
                                style: None,
                                children: vec![
                                    Node::new(
                                        "icon",
                                        NodeKind::Icon {
                                            name: "search".into(),
                                            size: 18.0,
                                            color: Some(0xff00ff00),
                                        },
                                    )
                                    .with_size(SizeSpec::Px(36.0), SizeSpec::Px(36.0)),
                                    Node::new(
                                        "pic",
                                        NodeKind::Image {
                                            src: "asset:missing".into(),
                                            fit: crate::images::ImageFit::Cover,
                                            radius: [8.0; 4],
                                            tint: None,
                                            placeholder: None,
                                            alt: "missing".into(),
                                        },
                                    )
                                    .with_size(SizeSpec::Px(64.0), SizeSpec::Px(64.0)),
                                ],
                            },
                        )
                        .with_size(SizeSpec::Px(160.0), SizeSpec::Px(120.0)),
                        Node::new(
                            "rich",
                            NodeKind::RichText(crate::ir::RichTextData {
                                spans: vec![
                                    crate::ir::TextSpan {
                                        text: "Hello ".into(),
                                        color: Some(0xffffffff),
                                        weight: 600,
                                        mono: false,
                                        size: None,
                                    },
                                    crate::ir::TextSpan {
                                        text: "world".into(),
                                        color: Some(0xff00aaff),
                                        weight: 400,
                                        mono: true,
                                        size: None,
                                    },
                                ],
                                size: 14.0,
                                wrap: true,
                                line_height: None,
                            }),
                        )
                        .with_size(SizeSpec::Px(200.0), SizeSpec::Px(60.0)),
                    ],
                },
            )
            .with_padding(Insets::all(10.0)),
        )
    }

    #[test]
    fn icons_images_stacks_and_rich_text_emit_sprite_and_glyph_primitives() {
        let mut engine = UiEngine::new(new_surface_scene(), [640.0, 400.0]);
        let mut dl = DrawList::new();
        let status = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(status.compiled);
        // Icon samples the process-wide SDF atlas through the sprite pipeline.
        let icon = dl
            .sprites
            .iter()
            .find(|s| s.tex_id == crate::images::ICON_TEX_ID)
            .expect("icon sprite");
        assert_ne!(icon.flags & crate::draw::SPRITE_FLAG_SDF, 0);
        assert_eq!(icon.tint, 0xff00ff00);
        // Undecoded image falls back to a rounded placeholder quad.
        assert!(dl.quads.iter().any(|q| q.radii == [8.0; 4]));
        // Rich text produced per-span glyphs (spaces emit no quad).
        assert!(dl.glyphs.len() >= "Helloworld".len());
        assert!(engine.layout.get("pic").is_some());
        assert_eq!(engine.pending_images(), vec!["asset:missing".to_string()]);
    }

    #[test]
    fn slider_scrubbing_updates_scene_and_emits_value_events() {
        let scene = Scene::new(
            "Slider",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 0.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![Node::new(
                        "vol",
                        NodeKind::Slider(crate::ir::SliderData {
                            value: 0.0,
                            min: 0.0,
                            max: 100.0,
                            step: 10.0,
                            tone: crate::theme::Tone::Neutral,
                            label: Some("Volume".into()),
                            show_value: true,
                        }),
                    )],
                },
            ),
        );
        let mut engine = UiEngine::new(scene, [400.0, 300.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let r = engine.layout.get("vol").unwrap();
        let y = r.y + r.h * 0.5;
        let down = engine.handle_input(InputEvent::PointerDown {
            x: r.x + r.w * 0.5,
            y,
            button: 0,
        });
        assert!(matches!(
            down.as_slice(),
            [UIEvent::ValueChanged { id, value }] if id == "vol" && (*value - 50.0).abs() < 0.001
        ));
        let moved = engine.handle_input(InputEvent::PointerMove { x: r.x + r.w * 0.8, y });
        assert!(matches!(
            moved.as_slice(),
            [UIEvent::ValueChanged { id, value }] if id == "vol" && (*value - 80.0).abs() < 0.001
        ));
        let _ = engine.handle_input(InputEvent::PointerUp {
            x: r.x + r.w * 0.8,
            y,
            button: 0,
        });
        assert!(engine.interaction.slider_drag.is_none());
        if let Some(node) = engine.scene.root.find_mut("vol") {
            if let NodeKind::Slider(data) = &node.kind {
                assert!((data.value - 80.0).abs() < 0.001);
            }
        }
    }

    #[test]
    fn list_view_virtualizes_and_reports_semantic_row_clicks() {
        let items: Vec<crate::ir::ListItem> = (0..1000)
            .map(|i| crate::ir::ListItem {
                id: format!("msg{i}"),
                height: Some(40.0),
                child: Node::new(format!("msg_child{i}"), NodeKind::Spacer),
            })
            .collect();
        let scene = Scene::new(
            "List",
            ThemeMode::Dark,
            Node::new(
                "list",
                NodeKind::ListView(crate::ir::ListViewData {
                    items,
                    estimated_height: 40.0,
                    spacing: 0.0,
                    overscan: 1.0,
                    style: None,
                }),
            )
            .with_size(SizeSpec::Px(320.0), SizeSpec::Px(300.0)),
        );
        let mut engine = UiEngine::new(scene, [320.0, 300.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let (start, end) = engine.layout.list_range("list").expect("virtual window");
        assert_eq!(start, 0);
        assert!(end < 40, "only the visible window compiles, got {end}");
        assert!(engine.layout.get("msg_child900").is_none());

        let r = engine.layout.get("list").unwrap();
        let x = r.x + r.w * 0.5;
        let y = r.y + 2.0 * 40.0 + 10.0;
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let up = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert!(matches!(
            up.as_slice(),
            [UIEvent::ListItemClicked { id, index, item_id }]
                if id == "list" && *index == 2 && item_id == "msg2"
        ));
    }

    #[test]
    fn nav_stack_pushes_pops_and_handles_back() {
        let scene = Scene::new(
            "Nav",
            ThemeMode::Dark,
            Node::new(
                "nav",
                NodeKind::NavStack {
                    routes: vec![crate::ir::NavRoute {
                        id: "home".into(),
                        title: "Home".into(),
                        child: Node::new("home_body", NodeKind::Spacer),
                    }],
                    active: 0,
                    progress: 0.0,
                    swipe_back: true,
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
        );
        let mut engine = UiEngine::new(scene, [400.0, 600.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(engine
            .apply_patch(ScenePatch::NavPush {
                id: "nav".into(),
                route: crate::ir::NavRoute {
                    id: "detail".into(),
                    title: "Detail".into(),
                    child: Node::new("detail_body", NodeKind::Spacer),
                },
            }));
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(engine.layout.get("detail_body").is_some());
        let events = engine.handle_input(InputEvent::Back);
        assert!(matches!(
            events.as_slice(),
            [UIEvent::BackRequested { id, route }] if id == "nav" && route == "detail"
        ));
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(engine.layout.get("detail_body").is_none());
        assert!(engine.layout.get("home_body").is_some());
    }

    #[test]
    fn nav_stack_edge_swipe_pops_route() {
        let scene = Scene::new(
            "Nav",
            ThemeMode::Dark,
            Node::new(
                "nav",
                NodeKind::NavStack {
                    routes: vec![
                        crate::ir::NavRoute {
                            id: "home".into(),
                            title: "Home".into(),
                            child: Node::new("home_body", NodeKind::Spacer),
                        },
                        crate::ir::NavRoute {
                            id: "detail".into(),
                            title: "Detail".into(),
                            child: Node::new("detail_body", NodeKind::Spacer),
                        },
                    ],
                    active: 1,
                    progress: 0.0,
                    swipe_back: true,
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
        );
        let mut engine = UiEngine::new(scene, [400.0, 600.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);

        // Drag from the left edge past the 40% commit threshold.
        let _ = engine.handle_input(InputEvent::PointerDown {
            x: 4.0,
            y: 300.0,
            button: 0,
        });
        let _ = engine.handle_input(InputEvent::PointerMove { x: 260.0, y: 300.0 });
        let events = engine.handle_input(InputEvent::PointerUp {
            x: 260.0,
            y: 300.0,
            button: 0,
        });
        assert!(matches!(
            events.as_slice(),
            [UIEvent::BackRequested { id, route }] if id == "nav" && route == "detail"
        ));
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(engine.layout.get("detail_body").is_none());
        assert!(engine.layout.get("home_body").is_some());
    }

    #[test]
    fn textarea_focus_types_pastes_and_enters_newlines() {
        let scene = Scene::new(
            "Compose",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 0.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![Node::new(
                        "note",
                        NodeKind::TextArea(crate::ir::TextAreaData {
                            value: String::new(),
                            placeholder: "Write".into(),
                            rows: 3,
                            tone: crate::theme::Tone::Neutral,
                            secure: false,
                            max_len: 0,
                            keyboard: crate::ir::KeyboardHint::Text,
                        }),
                    )],
                },
            ),
        );
        let mut engine = UiEngine::new(scene, [400.0, 300.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let r = engine.layout.get("note").unwrap();
        let (x, y) = (r.x + r.w * 0.5, r.y + r.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert_eq!(
            engine.interaction.focused_key.as_deref(),
            Some("textarea::note")
        );
        let typed = engine.handle_input(InputEvent::TextInput { text: "hi".into() });
        assert!(matches!(
            typed.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "note" && value == "hi"
        ));
        let pasted = engine.handle_input(InputEvent::Paste { text: " there".into() });
        assert!(matches!(
            pasted.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "note" && value == "hi there"
        ));
        let _ = engine.handle_input(InputEvent::KeyDown {
            key: "keycode:36".into(),
            modifiers: 0,
        });
        let _ = engine.handle_input(InputEvent::TextInput { text: "again".into() });
        if let Some(node) = engine.scene.root.find_mut("note") {
            if let NodeKind::TextArea(data) = &node.kind {
                assert_eq!(data.value, "hi there\nagain");
            }
        }
    }

    fn reload_scene(title: &str, theme: ThemeMode, extra_field: bool) -> Scene {
        let mut children = vec![
            Node::new(
                "name",
                NodeKind::Input {
                    value: String::new(),
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
                NodeKind::TextArea(crate::ir::TextAreaData {
                    value: String::new(),
                    placeholder: "Note".into(),
                    rows: 3,
                    tone: crate::theme::Tone::Neutral,
                    secure: false,
                    max_len: 0,
                    keyboard: crate::ir::KeyboardHint::Text,
                }),
            ),
            Node::new(
                "notify",
                NodeKind::Switch {
                    checked: false,
                    label: Some("Notify".into()),
                    tone: crate::theme::Tone::Neutral,
                },
            ),
            Node::new(
                "vol",
                NodeKind::Slider(crate::ir::SliderData {
                    value: 20.0,
                    min: 0.0,
                    max: 100.0,
                    step: 1.0,
                    tone: crate::theme::Tone::Neutral,
                    label: Some("Volume".into()),
                    show_value: true,
                }),
            ),
            Node::new(
                "tabs",
                NodeKind::Tabs {
                    active: 0,
                    items: vec![
                        TabItem {
                            label: "One".into(),
                            child: Node::new(
                                "tab_one",
                                NodeKind::Text {
                                    content: "tab one".into(),
                                    size: 13.0,
                                    color: None,
                                    mono: false,
                                    wrap: false,
                                },
                            ),
                        },
                        TabItem {
                            label: "Two".into(),
                            child: Node::new(
                                "tab_two",
                                NodeKind::Text {
                                    content: "tab two".into(),
                                    size: 13.0,
                                    color: None,
                                    mono: false,
                                    wrap: false,
                                },
                            ),
                        },
                    ],
                },
            ),
            Node::new(
                "nav",
                NodeKind::NavStack {
                    routes: vec![
                        crate::ir::NavRoute {
                            id: "home".into(),
                            title: "Home".into(),
                            child: Node::new(
                                "home_body",
                                NodeKind::Text {
                                    content: "home".into(),
                                    size: 13.0,
                                    color: None,
                                    mono: false,
                                    wrap: false,
                                },
                            ),
                        },
                        crate::ir::NavRoute {
                            id: "detail".into(),
                            title: "Detail".into(),
                            child: Node::new(
                                "detail_body",
                                NodeKind::Text {
                                    content: "detail".into(),
                                    size: 13.0,
                                    color: None,
                                    mono: false,
                                    wrap: false,
                                },
                            ),
                        },
                    ],
                    active: 0,
                    progress: 0.0,
                    swipe_back: true,
                },
            ),
        ];
        if extra_field {
            children.push(Node::new(
                "email",
                NodeKind::Input {
                    value: String::new(),
                    placeholder: "Email".into(),
                    tone: crate::theme::Tone::Neutral,
                    keyboard: crate::ir::KeyboardHint::Email,
                    prefix_icon: None,
                    clearable: false,
                    error: None,
                },
            ));
        }
        Scene::new(
            title,
            theme,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 16.0,
                    fold_width: None,
                    scrollable: true,
                    style: None,
                    children,
                },
            ),
        )
    }

    #[test]
    fn hot_reload_carries_live_state_across_matching_ids() {
        let mut engine = UiEngine::new(reload_scene("v1", ThemeMode::Dark, false), [420.0, 320.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);

        // Type into the single-line field, then the multi-line one.
        let name = engine.layout.get("name").unwrap();
        let (nx, ny) = (name.x + name.w * 0.5, name.y + name.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x: nx, y: ny, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x: nx, y: ny, button: 0 });
        let typed = engine.handle_input(InputEvent::TextInput { text: "Ada".into() });
        assert!(matches!(typed.as_slice(), [UIEvent::TextChanged { id, .. }] if id == "name"));

        let note = engine.layout.get("note").unwrap();
        let (tx, ty) = (note.x + note.w * 0.5, note.y + note.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x: tx, y: ty, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x: tx, y: ty, button: 0 });
        let _ = engine.handle_input(InputEvent::TextInput { text: "memo".into() });

        // Toggle the switch and drag the slider (real input, not scene edits).
        let notify = engine.layout.get("notify").unwrap();
        let (sx, sy) = (notify.x + notify.w * 0.5, notify.y + notify.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x: sx, y: sy, button: 0 });
        let toggled = engine.handle_input(InputEvent::PointerUp { x: sx, y: sy, button: 0 });
        assert!(matches!(toggled.as_slice(), [UIEvent::SwitchToggled { checked: true, .. }]));

        let vol = engine.layout.get("vol").unwrap();
        let (vx, vy) = (vol.x + vol.w * 0.75, vol.y + vol.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x: vx, y: vy, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerMove { x: vx, y: vy });
        let _ = engine.handle_input(InputEvent::PointerUp { x: vx, y: vy, button: 0 });

        // Tab switch + nav push + scroll, via the public patch/input surface.
        engine.apply_patch(ScenePatch::SetTab { id: "tabs".into(), active: 1 });
        engine.apply_patch(ScenePatch::NavPush {
            id: "nav".into(),
            route: crate::ir::NavRoute {
                id: "detail".into(),
                title: "Detail".into(),
                child: Node::new(
                    "detail_body",
                    NodeKind::Text {
                        content: "pushed".into(),
                        size: 13.0,
                        color: None,
                        mono: false,
                        wrap: false,
                    },
                ),
            },
        });
        let _ = engine.handle_input(InputEvent::Scroll {
            x: 210.0,
            y: 160.0,
            delta_x: 0.0,
            delta_y: 60.0,
        });
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let scroll_before = engine
            .interaction
            .scrolls
            .get("root")
            .map(|s| s.offset_y)
            .unwrap_or(0.0);

        let (old_name, old_note, old_vol) = (
            input_value(&engine, "name"),
            textarea_value(&engine, "note"),
            slider_value(&engine, "vol"),
        );
        assert_eq!(old_name, "Ada");
        assert_eq!(old_note, "memo");
        assert!(old_vol > 20.0);

        // Hot reload: a new build with restyled/added nodes, but the same ids.
        let changed = engine.apply_patch(ScenePatch::ReplaceScene(reload_scene(
            "v2",
            ThemeMode::Light,
            true,
        )));
        assert!(changed, "a structurally different scene must recompile");
        assert_eq!(engine.scene.title, "v2");
        assert_eq!(engine.scene.theme, ThemeMode::Dark, "host theme survives reload");
        assert_eq!(input_value(&engine, "name"), old_name);
        assert_eq!(input_value(&engine, "email"), "");
        assert_eq!(textarea_value(&engine, "note"), old_note);
        assert_eq!(slider_value(&engine, "vol"), old_vol);
        match &engine.scene.root.find("notify").unwrap().kind {
            NodeKind::Switch { checked, .. } => assert!(*checked),
            other => panic!("unexpected kind {other:?}"),
        }
        match &engine.scene.root.find("tabs").unwrap().kind {
            NodeKind::Tabs { active, .. } => assert_eq!(*active, 1),
            other => panic!("unexpected kind {other:?}"),
        }
        match &engine.scene.root.find("nav").unwrap().kind {
            NodeKind::NavStack { active, routes, .. } => {
                assert_eq!(routes[*active].id, "detail");
            }
            other => panic!("unexpected kind {other:?}"),
        }
        let scroll_after = engine
            .interaction
            .scrolls
            .get("root")
            .map(|s| s.offset_y)
            .unwrap_or(0.0);
        assert_eq!(scroll_after, scroll_before, "scroll offset survives reload");

        // Idempotent reload: same scene bytes do not dirty the engine.
        let same = reload_scene("v2", ThemeMode::Dark, true);
        assert!(!engine.apply_patch(ScenePatch::ReplaceScene(same)));
    }

    fn focus_input(engine: &mut UiEngine, id: &str) {
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        let r = engine.layout.get(id).expect("field laid out");
        let (x, y) = (r.x + 12.0, r.y + r.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
    }

    fn type_text(engine: &mut UiEngine, text: &str) {
        for ch in text.chars() {
            let _ = engine.handle_input(InputEvent::TextInput {
                text: ch.to_string(),
            });
        }
    }

    fn field_value(engine: &UiEngine, id: &str) -> String {
        match &engine.scene.root.find(id).expect("field").kind {
            NodeKind::Input { value, .. } => value.clone(),
            NodeKind::TextArea(data) => data.value.clone(),
            other => panic!("not editable: {other:?}"),
        }
    }

    #[test]
    fn input_clear_button_empties_the_field_and_error_caption_adds_height() {
        use crate::ir::{Align, Axis};

        let field = |error: Option<&str>| {
            Node::new(
                "q",
                NodeKind::Input {
                    value: "draft".into(),
                    placeholder: "Search".into(),
                    tone: crate::theme::Tone::Neutral,
                    keyboard: crate::ir::KeyboardHint::Search,
                    prefix_icon: Some("search".into()),
                    clearable: true,
                    error: error.map(ToString::to_string),
                },
            )
        };
        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![field(None)],
            },
        );
        let mut engine = UiEngine::new(Scene::new("clear", ThemeMode::Light, root), [420.0, 200.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        let plain_h = engine.layout.get("q").expect("field").h;

        // The caption is part of the node box, so layout makes room for it.
        let replaced = engine.apply_patch(ScenePatch::ReplaceNode {
            id: "q".into(),
            node: field(Some("That name is taken")),
        });
        assert!(replaced, "node replaced");
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        let with_error = engine.layout.get("q").expect("field").h;
        assert!(
            with_error > plain_h,
            "caption must grow the box ({plain_h} -> {with_error})"
        );
        let a11y = crate::a11y::scene_tree_json(
            &engine.scene,
            &engine.layout,
            &engine.interaction,
        );
        assert!(a11y.contains("text_field"), "{a11y}");

        // Click the trailing `×`: the value empties and the change is reported.
        let r = engine.layout.get("q").expect("field");
        let (x, y) = (r.x + r.w - 20.0, r.y + 22.0);
        engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let events = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert_eq!(field_value(&engine, "q"), "", "clear button empties the value");
        assert!(
            events.iter().any(
                |e| matches!(e, UIEvent::TextChanged { id, value } if id == "q" && value.is_empty())
            ),
            "clear must emit TextChanged: {events:?}"
        );
    }

    #[test]
    fn double_click_selects_a_word_and_triple_click_selects_the_line() {
        use crate::ir::{Align, Axis, TextAreaData};

        fn cta(id: &str, kind: NodeKind) -> Node {
            Node::new(id, kind)
        }

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 10.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![
                    cta(
                        "one",
                        NodeKind::Input {
                            value: "alpha beta gamma".into(),
                            placeholder: String::new(),
                            tone: crate::theme::Tone::Neutral,
                            keyboard: crate::ir::KeyboardHint::Text,
                            prefix_icon: None,
                            clearable: false,
                            error: None,
                        },
                    ),
                    cta(
                        "multi",
                        NodeKind::TextArea(TextAreaData {
                            value: "first line\nsecond line".into(),
                            placeholder: String::new(),
                            rows: 2,
                            tone: crate::theme::Tone::Neutral,
                            secure: false,
                            max_len: 0,
                            keyboard: crate::ir::KeyboardHint::Text,
                        }),
                    ),
                ],
            },
        );
        let mut engine = UiEngine::new(Scene::new("taps", ThemeMode::Light, root), [460.0, 260.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);

        // Double-click the word "beta": tap twice inside one 320 ms window.
        let r = engine.layout.get("one").expect("input laid out");
        let (x, y) = (r.x + 60.0, r.y + r.h * 0.5);
        for _ in 0..2 {
            let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
            let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        }
        let edit = engine.interaction.edit.clone().expect("caret installed");
        let value = field_value(&engine, "one");
        let selected = &value[edit.anchor.min(edit.caret)..edit.anchor.max(edit.caret)];
        assert_eq!(
            selected, "beta",
            "double click should select the word under the pointer"
        );

        // A pause breaks the run, so the next click is a single tap again.
        engine.tick_and_compile(0.5, &mut dl);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        let edit = engine.interaction.edit.clone().expect("caret installed");
        assert_eq!(edit.anchor, edit.caret, "a slow click only places the caret");

        // Triple-click the second line of a text area selects that line.
        let r = engine.layout.get("multi").expect("textarea laid out");
        let (tx, ty) = (r.x + 30.0, r.y + 36.0);
        for _ in 0..3 {
            let _ = engine.handle_input(InputEvent::PointerDown {
                x: tx,
                y: ty,
                button: 0,
            });
            let _ = engine.handle_input(InputEvent::PointerUp {
                x: tx,
                y: ty,
                button: 0,
            });
        }
        let edit = engine.interaction.edit.clone().expect("caret installed");
        let value = field_value(&engine, "multi");
        let selected = value[edit.anchor.min(edit.caret)..edit.anchor.max(edit.caret)].to_string();
        assert!(
            selected.contains("second line") && !selected.contains("first"),
            "triple click should select one line, got {selected:?}"
        );
    }

    #[test]
    fn undo_and_redo_walk_the_edit_stack_without_losing_the_caret() {
        use crate::hit_test::{MOD_META, MOD_SHIFT};
        use crate::ir::{Align, Axis};

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![Node::new(
                    "q",
                    NodeKind::Input {
                        value: String::new(),
                        placeholder: "Question".into(),
                        tone: crate::theme::Tone::Neutral,
                        keyboard: crate::ir::KeyboardHint::Text,
                        prefix_icon: None,
                        clearable: false,
                        error: None,
                    },
                )],
            },
        );
        let mut engine = UiEngine::new(Scene::new("undo", ThemeMode::Light, root), [420.0, 200.0]);
        focus_input(&mut engine, "q");

        // A typed run collapses into one undoable edit.
        type_text(&mut engine, "hello");
        assert_eq!(field_value(&engine, "q"), "hello");

        let undo = |e: &mut UiEngine, shift: bool| {
            let mut mods = MOD_META;
            if shift {
                mods |= MOD_SHIFT;
            }
            let _ = e.handle_input(InputEvent::KeyDown {
                key: "z".into(),
                modifiers: mods,
            });
        };
        undo(&mut engine, false);
        assert_eq!(field_value(&engine, "q"), "", "one undo clears the run");

        undo(&mut engine, true);
        assert_eq!(field_value(&engine, "q"), "hello", "redo restores it");
        assert_eq!(
            engine.interaction.edit.as_ref().map(|e| e.caret),
            Some(5),
            "caret lands at the end of the restored text"
        );

        // A second run is a separate entry: backspaces coalesce on their own.
        type_text(&mut engine, "!");
        for _ in 0..2 {
            let _ = engine.handle_input(InputEvent::KeyDown {
                key: "Backspace".into(),
                modifiers: 0,
            });
        }
        assert_eq!(field_value(&engine, "q"), "hell");
        undo(&mut engine, false);
        assert_eq!(field_value(&engine, "q"), "hello!", "backspace run undone");
        undo(&mut engine, false);
        assert_eq!(field_value(&engine, "q"), "hello", "then the '!' insert");
        undo(&mut engine, false);
        assert_eq!(field_value(&engine, "q"), "", "then the typed run");
        assert!(
            !engine.interaction.histories["input::q"].can_undo(),
            "stack exhausted at the field's origin"
        );
    }

    #[test]
    fn caret_blinks_on_a_half_second_deadline_without_a_present_loop() {
        use crate::ir::{Align, Axis};

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 10.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![Node::new(
                    "note",
                    NodeKind::Input {
                        value: "hello".into(),
                        placeholder: "Note".into(),
                        tone: crate::theme::Tone::Neutral,
                        keyboard: crate::ir::KeyboardHint::Text,
                        prefix_icon: None,
                        clearable: false,
                        error: None,
                    },
                )],
            },
        );
        let mut engine = UiEngine::new(Scene::new("blink", ThemeMode::Light, root), [420.0, 240.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);

        // Nothing focused: the window may sleep forever, and the draw list
        // says so on its own (the host loop reads only this contract).
        assert_eq!(engine.next_wake_ms(), None);
        assert!(!dl.needs_next_frame);
        assert_eq!(dl.next_wake_s, None);

        let r = engine.layout.get("note").expect("input laid out");
        let (x, y) = (r.x + 20.0, r.y + r.h * 0.5);
        engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        assert!(engine.interaction.caret_visible, "a fresh caret is solid");
        let wake = engine.next_wake_ms().expect("blink deadline while focused");
        assert!(
            (400.0..=500.0).contains(&wake),
            "expected a sub-half-second wake, got {wake}"
        );
        assert!(dl.needs_next_frame, "focus schedules a wake");
        let due = dl.next_wake_s.expect("absolute deadline published");
        assert!(
            (due - (engine.now_s + wake / 1000.0)).abs() < 1e-9,
            "deadline must be engine-clock based"
        );

        // Let the focus spring settle first, then count compiles across a full
        // blink cycle: one wake-up, not one per frame.
        for _ in 0..200 {
            engine.tick_and_compile(0.02, &mut dl);
            if !engine.is_dirty_or_animating() {
                break;
            }
        }
        assert!(
            !engine.is_dirty_or_animating(),
            "focus spring should settle before the blink window"
        );
        let mut toggles = 0;
        for _ in 0..50 {
            let status = engine.tick_and_compile(0.02, &mut dl);
            if status.compiled {
                toggles += 1;
            }
        }
        assert!(
            (1..=2).contains(&toggles),
            "a 1 s window must cost 1-2 compiles, not {toggles}"
        );
        assert!(!engine.is_dirty_or_animating(), "settles between blinks");
        // A non-toggle tick still reports the upcoming deadline: the host
        // sleeps until it, instead of polling at 120 Hz.
        let mut saw_idle_tick = false;
        for _ in 0..100 {
            let status = engine.tick_and_compile(0.02, &mut dl);
            if !status.compiled {
                saw_idle_tick = true;
                assert!(dl.needs_next_frame);
                assert!(dl.next_wake_s.is_some());
                assert!(!engine.is_dirty_or_animating());
                break;
            }
        }
        assert!(saw_idle_tick, "a blink cycle has idle ticks between toggles");

        // Typing makes the caret solid again and restarts the deadline.
        engine.handle_input(InputEvent::TextInput {
            text: "!".to_string(),
        });
        engine.tick_and_compile(0.02, &mut dl);
        assert!(engine.interaction.caret_visible);
        let wake = engine.next_wake_ms().expect("deadline after typing");
        assert!(wake > 400.0, "typing restarts the blink: {wake}");
    }

    #[test]
    fn wall_clock_elapsed_can_exceed_the_clamped_animation_step() {
        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: crate::ir::Axis::Column,
                align: crate::ir::Align::Stretch,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![Node::new(
                    "label",
                    NodeKind::Text {
                        content: "clock".into(),
                        size: 14.0,
                        color: None,
                        mono: false,
                        wrap: false,
                    },
                )],
            },
        );
        let mut engine = UiEngine::new(Scene::new("clock", ThemeMode::Light, root), [320.0, 200.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);

        let before = engine.now_s;
        engine.tick_with_elapsed(1.0 / 60.0, 0.5, &mut dl);
        assert!(
            (engine.now_s - before - 0.5).abs() < 1e-9,
            "a slept deadline must advance the engine clock by 0.5s, got {}",
            engine.now_s - before
        );

        let before = engine.now_s;
        engine.tick_and_compile(0.25, &mut dl);
        assert!((engine.now_s - before - 0.25).abs() < 1e-9);
    }

    #[test]
    fn keyboard_hints_follow_focus_and_reach_a11y() {
        use crate::ir::{Align, Axis, KeyboardHint, TextAreaData};

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 10.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![
                    Node::new(
                        "mail",
                        NodeKind::Input {
                            value: String::new(),
                            placeholder: "Email".into(),
                            tone: crate::theme::Tone::Neutral,
                            keyboard: KeyboardHint::Email,
                            prefix_icon: None,
                            clearable: false,
                            error: None,
                        },
                    ),
                    Node::new(
                        "pin",
                        NodeKind::TextArea(TextAreaData {
                            value: String::new(),
                            placeholder: "PIN".into(),
                            rows: 1,
                            tone: crate::theme::Tone::Neutral,
                            secure: true,
                            max_len: 8,
                            keyboard: KeyboardHint::Password,
                        }),
                    ),
                ],
            },
        );
        let mut engine = UiEngine::new(Scene::new("keys", ThemeMode::Light, root), [420.0, 260.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        assert_eq!(engine.focused_keyboard_hint(), None);

        let r = engine.layout.get("mail").expect("input laid out");
        let (x, y) = (r.x + r.w * 0.5, r.y + r.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert_eq!(engine.focused_keyboard_hint(), Some(KeyboardHint::Email));

        let tree = crate::a11y::scene_tree_json(&engine.scene, &engine.layout, &engine.interaction);
        assert!(tree.contains("\"input_type\":\"email\""), "{tree}");

        let r = engine.layout.get("pin").expect("textarea laid out");
        let (x, y) = (r.x + 12.0, r.y + r.h * 0.5);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        assert_eq!(engine.focused_keyboard_hint(), Some(KeyboardHint::Password));
        let tree = crate::a11y::scene_tree_json(&engine.scene, &engine.layout, &engine.interaction);
        assert!(tree.contains("\"input_type\":\"password\""), "{tree}");
    }

    #[test]
    fn large_hot_reload_swap_is_under_fifty_ms_and_settles_to_idle() {
        use std::time::Instant;

        fn reload_big(rev: &str) -> Scene {
            let mut rows = Vec::new();
            for i in 0..400 {
                let content = if rev == "v1" {
                    format!("row {i} of the first build")
                } else {
                    format!("row {i} of the hot-reloaded build")
                };
                rows.push(
                    Node::new(
                        format!("r{i}"),
                        NodeKind::Text {
                            content,
                            size: 13.0,
                            color: None,
                            mono: false,
                            wrap: false,
                        },
                    )
                    .with_size(crate::ir::SizeSpec::Fill(1.0), crate::ir::SizeSpec::Px(22.0)),
                );
            }
            rows.push(Node::new(
                "knob",
                NodeKind::Switch {
                    checked: false,
                    label: Some("Hot".into()),
                    tone: crate::theme::Tone::Neutral,
                },
            ));
            let list = Node::new(
                "rows",
                NodeKind::Container {
                    axis: crate::ir::Axis::Column,
                    align: Align::Stretch,
                    gap: 2.0,
                    fold_width: None,
                    scrollable: true,
                    style: None,
                    children: rows,
                },
            )
            .with_size(crate::ir::SizeSpec::Fill(1.0), crate::ir::SizeSpec::Fill(1.0));
            let root = Node::new(
                "root",
                NodeKind::Container {
                    axis: crate::ir::Axis::Column,
                    align: Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![list],
                },
            );
            Scene::new(rev, ThemeMode::Dark, root)
        }

        let viewport = [900.0, 800.0];
        let mut engine = UiEngine::new(reload_big("v1"), viewport);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 60.0, &mut dl);

        let t0 = Instant::now();
        assert!(engine.apply_patch(ScenePatch::ReplaceScene(reload_big("v2"))));
        engine.tick_and_compile(1.0 / 60.0, &mut dl);
        let swap_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("hot reload: 401 nodes swapped + recompiled in {swap_ms:.2} ms");
        assert!(
            swap_ms < 50.0,
            "scene swap + recompile took {swap_ms:.2} ms (target < 50 ms)"
        );

        // Settle every spring/scroll, then prove the next frame is a no-op so
        // an idle window can stop rendering entirely (0% GPU when idle).
        for _ in 0..240 {
            if !engine.is_dirty_or_animating() {
                break;
            }
            engine.tick_and_compile(1.0 / 60.0, &mut dl);
        }
        assert!(!engine.is_dirty_or_animating(), "engine must settle to idle");
        let idle = engine.tick_and_compile(1.0 / 60.0, &mut dl);
        assert!(!idle.compiled && !idle.animating, "idle tick must not recompile");
    }

    #[test]
    fn fifty_thousand_row_list_compiles_under_one_ms() {
        use std::time::Instant;
        let mut items = Vec::with_capacity(50_000);
        for i in 0..50_000 {
            let h = match i % 3 {
                0 => 44.0,
                1 => 56.0,
                _ => 68.0,
            };
            items.push(crate::ir::ListItem {
                id: format!("r{i}"),
                height: Some(h),
                child: Node::new(
                    format!("c{i}"),
                    NodeKind::Text {
                        content: format!("row {i}"),
                        size: 13.0,
                        color: None,
                        mono: false,
                        wrap: false,
                    },
                ),
            });
        }
        let scene = Scene::new(
            "50k",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 0.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![Node::new(
                        "list",
                        NodeKind::ListView(crate::ir::ListViewData {
                            items,
                            estimated_height: 56.0,
                            spacing: 2.0,
                            overscan: 0.0,
                            style: None,
                        }),
                    )
                    .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0))],
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
        );
        let mut engine = UiEngine::new(scene, [420.0, 800.0]);
        let mut dl = DrawList::new();
        let first = Instant::now();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let first_ms = first.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(dl.sprites.len(), 0);
        let (start, end) = engine.layout.list_range("list").unwrap();
        assert!(
            end - start < 40,
            "virtualization must only place the visible window (got {start}..{end})"
        );

        // Scrolling dirties the frame every tick; measure the steady-state cost.
        let frames = 30;
        let mut total = 0.0;
        for _ in 0..frames {
            let _ = engine.handle_input(InputEvent::Scroll {
                x: 210.0,
                y: 400.0,
                delta_x: 0.0,
                delta_y: 120.0,
            });
            let t = Instant::now();
            engine.tick_and_compile(1.0 / 120.0, &mut dl);
            total += t.elapsed().as_secs_f64();
        }
        let avg_ms = total * 1000.0 / frames as f64;
        println!("50k list: first={first_ms:.3} ms, scroll avg={avg_ms:.3} ms/frame");
        assert!(
            avg_ms < 1.0,
            "50k mixed-height rows must compile < 1.0 ms/frame (got {avg_ms:.3} ms)"
        );
    }

    #[test]
    fn textarea_selection_clipboard_and_shift_navigation() {
        let scene = Scene::new(
            "Editor",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 8.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![Node::new(
                        "note",
                        NodeKind::TextArea(crate::ir::TextAreaData {
                            value: "hello world".into(),
                            placeholder: "Write".into(),
                            rows: 3,
                            tone: crate::theme::Tone::Neutral,
                            secure: false,
                            max_len: 0,
                            keyboard: crate::ir::KeyboardHint::Text,
                        }),
                    )],
                },
            ),
        );
        let mut engine = UiEngine::new(scene, [420.0, 240.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let r = engine.layout.get("note").unwrap();

        // Click at the left edge -> caret at 0, anchor at 0.
        let (x, y) = (r.x + 11.0, r.y + 15.0);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        let edit = engine.interaction.edit.clone().unwrap();
        assert_eq!((edit.anchor, edit.caret), (0, 0));

        // Shift+Right x5 selects "hello".
        for _ in 0..5 {
            let _ = engine.handle_input(InputEvent::KeyDown {
                key: "ArrowRight".into(),
                modifiers: crate::hit_test::MOD_SHIFT,
            });
        }
        let edit = engine.interaction.edit.clone().unwrap();
        assert_eq!(edit.selection(), (0, 5));

        // Copy reports the selected slice and does not mutate the value.
        let copied = engine.handle_input(InputEvent::KeyDown {
            key: "c".into(),
            modifiers: crate::hit_test::MOD_META,
        });
        assert!(matches!(
            copied.as_slice(),
            [UIEvent::CopyRequested { id, text }] if id == "note" && text == "hello"
        ));

        // Typing replaces the selection at the caret.
        let typed = engine.handle_input(InputEvent::TextInput { text: "hi ".into() });
        assert!(matches!(
            typed.as_slice(),
            [UIEvent::TextChanged { id, value }] if id == "note" && value == "hi  world"
        ));
        let edit = engine.interaction.edit.clone().unwrap();
        assert_eq!((edit.anchor, edit.caret), (3, 3));

        // End jumps to the line end; Backspace removes the last character.
        let _ = engine.handle_input(InputEvent::KeyDown {
            key: "End".into(),
            modifiers: 0,
        });
        assert_eq!(engine.interaction.edit.as_ref().unwrap().caret, 9);
        let bs = engine.handle_input(InputEvent::KeyDown {
            key: "Backspace".into(),
            modifiers: 0,
        });
        assert!(matches!(
            bs.as_slice(),
            [UIEvent::TextChanged { value, .. }] if value == "hi  worl"
        ));

        // Cmd+A then typing replaces the whole field.
        let _ = engine.handle_input(InputEvent::KeyDown {
            key: "a".into(),
            modifiers: crate::hit_test::MOD_META,
        });
        let edit = engine.interaction.edit.clone().unwrap();
        assert_eq!(edit.selection(), (0, 8));
        let _ = engine.handle_input(InputEvent::TextInput { text: "fresh".into() });
        assert_eq!(textarea_value(&engine, "note"), "fresh");

        // Cut emits CopyRequested + TextChanged and clears the field.
        let _ = engine.handle_input(InputEvent::KeyDown {
            key: "a".into(),
            modifiers: crate::hit_test::MOD_META,
        });
        let cut = engine.handle_input(InputEvent::KeyDown {
            key: "x".into(),
            modifiers: crate::hit_test::MOD_META,
        });
        assert_eq!(cut.len(), 2);
        assert!(matches!(&cut[0], UIEvent::CopyRequested { text, .. } if text == "fresh"));
        assert!(matches!(&cut[1], UIEvent::TextChanged { value, .. } if value.is_empty()));
        assert_eq!(textarea_value(&engine, "note"), "");
    }

    #[test]
    fn secure_textarea_hides_selection_text_from_copy() {
        let scene = Scene::new(
            "Vault",
            ThemeMode::Dark,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 0.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: vec![Node::new(
                        "pin",
                        NodeKind::TextArea(crate::ir::TextAreaData {
                            value: "1234".into(),
                            placeholder: "PIN".into(),
                            rows: 1,
                            tone: crate::theme::Tone::Neutral,
                            secure: true,
                            max_len: 8,
                            keyboard: crate::ir::KeyboardHint::Password,
                        }),
                    )],
                },
            ),
        );
        let mut engine = UiEngine::new(scene, [320.0, 160.0]);
        let mut dl = DrawList::new();
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let r = engine.layout.get("pin").unwrap();
        let (x, y) = (r.x + 12.0, r.y + 12.0);
        let _ = engine.handle_input(InputEvent::PointerDown { x, y, button: 0 });
        let _ = engine.handle_input(InputEvent::PointerUp { x, y, button: 0 });
        // A11y masks the value, and the painted selection stays on screen.
        let tree = crate::a11y::scene_tree(&engine.scene, &engine.layout, &engine.interaction);
        let json = serde_json::to_string(&tree).unwrap();
        assert!(!json.contains("1234"));
        let _ = engine.handle_input(InputEvent::KeyDown {
            key: "SelectAll".into(),
            modifiers: 0,
        });
        assert_eq!(engine.interaction.edit.as_ref().unwrap().selection(), (0, 4));
    }

    fn input_value(engine: &UiEngine, id: &str) -> String {
        match &engine.scene.root.find(id).unwrap().kind {
            NodeKind::Input { value, .. } => value.clone(),
            other => panic!("{id} is not an input: {other:?}"),
        }
    }

    fn textarea_value(engine: &UiEngine, id: &str) -> String {
        match &engine.scene.root.find(id).unwrap().kind {
            NodeKind::TextArea(data) => data.value.clone(),
            other => panic!("{id} is not a textarea: {other:?}"),
        }
    }

    fn slider_value(engine: &UiEngine, id: &str) -> f32 {
        match &engine.scene.root.find(id).unwrap().kind {
            NodeKind::Slider(data) => data.value,
            other => panic!("{id} is not a slider: {other:?}"),
        }
    }
}
