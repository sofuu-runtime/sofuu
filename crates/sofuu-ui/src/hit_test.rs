//! Spatial hit-testing, pointer/keyboard input routing, and kinetic scroll physics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::layout::ComputedRect;
use crate::theme::ThemeMode;

/// Raw OS input events dispatched from `sofuu-gpu` (Thread 0) into `UiEngine`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputEvent {
    PointerMove { x: f32, y: f32 },
    PointerDown { x: f32, y: f32, button: u8 },
    PointerUp { x: f32, y: f32, button: u8 },
    Scroll { x: f32, y: f32, delta_x: f32, delta_y: f32 },
    KeyDown { key: String, modifiers: u32 },
    TextInput { text: String },
    /// Host clipboard paste (bridged from AppKit/UIKit/Android/Win32/Wayland).
    Paste { text: String },
    /// Hardware/system Back button or gesture.
    Back,
    Resize { width: f32, height: f32, scale_factor: f32 },
    CloseRequested,
}

/// High-level semantic events emitted back to the Sofuu agent / host application.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UIEvent {
    Click { id: String },
    TabChanged { id: String, index: usize },
    SegmentedChanged { id: String, index: usize },
    ThemeChanged { id: String, mode: ThemeMode },
    SwitchToggled { id: String, checked: bool },
    TextChanged { id: String, value: String },
    InputSubmitted { id: String, value: String },
    TableSorted { id: String, col_index: usize, asc: bool },
    RowSelected { id: String, row_id: String },
    FeedItemClicked { id: String, item_id: String },
    GraphNodeClicked { id: String, node_id: String },
    SeriesToggled { id: String, series_index: usize },
    Scrolled { id: String, offset_y: f32 },
    /// Slider / range value changed (already clamped and step-snapped).
    ValueChanged { id: String, value: f32 },
    /// Dual-handle range slider moved; `low <= high` always holds.
    RangeChanged { id: String, low: f32, high: f32 },
    /// Virtualized list row tapped.
    ListItemClicked { id: String, index: usize, item_id: String },
    /// `NavStack` popped (edge swipe, back button, or host request).
    BackRequested { id: String, route: String },
    /// Pull-to-refresh threshold crossed.
    RefreshRequested { id: String },
    /// Headless drag primitive moved by `dx`/`dy` since the previous event.
    Dragged { id: String, dx: f32, dy: f32 },
    /// Copy requested; the host should write `text` to the system clipboard.
    CopyRequested { id: String, text: String },
    WindowClosed,
}

#[derive(Clone, Debug)]
pub enum HitKind {
    Surface { id: String },
    Stat { id: String },
    TabButton { tabs_id: String, index: usize },
    SegmentedItem { seg_id: String, index: usize },
    Button { id: String },
    ThemeToggle { id: String },
    Switch { id: String },
    Input { id: String },
    TableHeader { table_id: String, col_index: usize },
    TableRow { table_id: String, row_id: String },
    FeedItem { feed_id: String, item_id: String },
    GraphNode { graph_id: String, node_id: String },
    PlotArea { id: String },
    PlotLegendChip { plot_id: String, series_index: usize },
    ScrollContainer { id: String, max_scroll_y: f32 },
    Slider { id: String },
    RangeSlider { id: String },
    /// The `×` affordance inside an `Input`.
    InputClear { id: String },
    TextArea { id: String },
    ListItem { list_id: String, index: usize, item_id: String },
    Pressable { id: String },
    Draggable { id: String },
    NavEdge { id: String },
    OverlayBackdrop { id: String },
}

#[derive(Clone, Debug)]
pub struct HitRegion {
    pub rect: ComputedRect,
    pub kind: HitKind,
}

impl HitRegion {
    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.rect.x
            && x <= self.rect.x + self.rect.w
            && y >= self.rect.y
            && y <= self.rect.y + self.rect.h
    }
}

/// Overscroll behaviour when a scroll hits its bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollPhysics {
    /// Hard stop, plus a desktop scrollbar (`Windows`/`Linux`).
    #[default]
    Clamped,
    /// Apple rubber band: the content follows the finger past the edge with
    /// heavy resistance and springs back on release (`iOS`/`macOS`).
    RubberBand,
    /// Android stretch: a smaller displacement that detaches and springs back
    /// faster, so the edge reads as elastic rather than draggable.
    Stretch,
}

impl ScrollPhysics {
    /// Platform id (`ios`, `android`, `macos`, `windows`, `linux`) → physics.
    pub fn for_platform(platform: &str) -> Self {
        match platform {
            "ios" | "macos" => ScrollPhysics::RubberBand,
            "android" => ScrollPhysics::Stretch,
            _ => ScrollPhysics::Clamped,
        }
    }

    /// Fraction of an out-of-range drag that becomes visible displacement.
    pub const fn resistance(self) -> f32 {
        match self {
            ScrollPhysics::Clamped => 0.0,
            ScrollPhysics::RubberBand => 0.55,
            ScrollPhysics::Stretch => 0.32,
        }
    }

    /// How quickly the displaced edge returns home (larger = snappier).
    pub const fn rebound(self) -> f32 {
        match self {
            ScrollPhysics::Clamped => 0.0,
            ScrollPhysics::RubberBand => 9.0,
            ScrollPhysics::Stretch => 16.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KineticScroll {
    pub offset_y: f32,
    pub velocity_y: f32,
    pub max_y: f32,
    /// Displacement drawn *past* a bound (negative = above the top). Never
    /// part of `offset_y`, so virtualization and clamping stay exact.
    pub overscroll: f32,
}

impl KineticScroll {
    /// Offset the renderer should translate content by.
    #[inline]
    pub fn visual_offset(&self) -> f32 {
        self.offset_y + self.overscroll
    }

    pub fn impulse(&mut self, delta_y: f32, physics: ScrollPhysics) {
        let raw = self.offset_y - delta_y;
        if raw < 0.0 || raw > self.max_y {
            let excess = if raw < 0.0 { raw } else { raw - self.max_y };
            self.overscroll += excess * physics.resistance();
            // Pulling past the edge kills translational speed; the stored
            // overscroll is what the bounce returns from.
            self.velocity_y = 0.0;
        } else {
            self.velocity_y = (-delta_y * 14.0).clamp(-3600.0, 3600.0);
            // Dragging back from an overscrolled edge reclaims it first.
            if self.overscroll != 0.0 {
                let used = self.overscroll.clamp(-self.max_y, self.max_y);
                self.overscroll -= used;
            }
        }
        self.offset_y = raw.clamp(0.0, self.max_y);
    }

    pub fn step(&mut self, dt: f32, physics: ScrollPhysics) -> bool {
        let dt = dt.clamp(0.0, 0.05);
        self.offset_y = self.offset_y.clamp(0.0, self.max_y);

        // A displaced edge springs home independently of the kinetic velocity.
        let mut bouncing = false;
        if self.overscroll != 0.0 {
            let k = physics.rebound();
            if k <= 0.0 {
                self.overscroll = 0.0;
            } else {
                self.overscroll *= (1.0 - k * dt).max(0.0);
                if self.overscroll.abs() < 0.35 {
                    self.overscroll = 0.0;
                } else {
                    bouncing = true;
                }
                self.max_y = self.max_y.max(0.0);
            }
        }

        if self.velocity_y.abs() < 4.0 {
            self.velocity_y = 0.0;
            return bouncing;
        }
        let next = self.offset_y + self.velocity_y * dt;
        if next < 0.0 || next > self.max_y {
            // A fling into the edge converts into overscroll and stops.
            let excess = if next < 0.0 { next } else { next - self.max_y };
            self.overscroll += excess * physics.resistance();
            self.velocity_y = 0.0;
            self.offset_y = next.clamp(0.0, self.max_y);
            return bouncing || self.overscroll.abs() >= 0.35;
        }
        self.offset_y = next;
        // Exponential friction damping (~0.90 per 16.6ms frame)
        let decay = (0.0018_f32).powf(dt);
        self.velocity_y *= decay;
        bouncing
    }
}

/// Keyboard modifier bits carried by [`InputEvent::KeyDown`], matching the
/// C ABI's `SOFUU_MOD_*` mask.
pub const MOD_SHIFT: u32 = 1 << 0;
pub const MOD_CTRL: u32 = 1 << 1;
pub const MOD_ALT: u32 = 1 << 2;
pub const MOD_META: u32 = 1 << 3;

/// Caret + selection for the focused single-line or multi-line text field.
///
/// `anchor`/`caret` are byte offsets into the field's `value`, always on UTF-8
/// character boundaries. `anchor` is where the selection began and `caret` is
/// the moving end; `anchor == caret` means "no selection". Clicking places
/// both at the pointer, Shift+arrow/Shift+click moves only `caret`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEditState {
    /// Focus key: `input::<id>` or `textarea::<id>`.
    pub id: String,
    pub anchor: usize,
    pub caret: usize,
}

/// One checkpoint of an editable field (value + selection) for undo/redo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditSnapshot {
    pub value: String,
    pub anchor: usize,
    pub caret: usize,
}

/// Undo/redo stack for one editable field.
///
/// Consecutive single-character insertions (and backspaces) collapse into one
/// entry, so `Cmd+Z` removes the run a user thinks of as one edit rather than
/// one keystroke at a time.
#[derive(Clone, Debug, Default)]
pub struct EditHistory {
    undo: Vec<EditSnapshot>,
    redo: Vec<EditSnapshot>,
    last: Option<EditOp>,
    /// Caret position the previous edit landed on (for run coalescing).
    run_end: usize,
}

/// Kind of the most recent edit, used only for coalescing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditOp {
    Insert,
    Delete,
}

impl EditHistory {
    pub const LIMIT: usize = 200;

    /// Record the state *before* an edit. `start`/`end` are the replaced byte
    /// range and `inserted` is the number of characters written.
    pub fn record(
        &mut self,
        snapshot: EditSnapshot,
        kind: EditOp,
        start: usize,
        end: usize,
        inserted: usize,
    ) {
        // A run continues while the edit picks up exactly where the previous
        // one left the caret: the next character typed, or the next backspace.
        let continues = match kind {
            EditOp::Insert => start == self.run_end,
            EditOp::Delete => end == self.run_end,
        };
        let coalesce =
            self.last == Some(kind) && inserted <= 1 && continues && !self.undo.is_empty();
        self.redo.clear();
        if !coalesce {
            self.undo.push(snapshot);
            if self.undo.len() > Self::LIMIT {
                self.undo.remove(0);
            }
        }
        self.last = Some(kind);
        self.run_end = start + inserted;
    }

    /// Note that the caret moved, which breaks the current typing run.
    pub fn break_run(&mut self, caret: usize) {
        self.last = None;
        self.run_end = caret;
    }

    pub fn pop_undo(&mut self, current: EditSnapshot) -> Option<EditSnapshot> {
        let prev = self.undo.pop()?;
        self.redo.push(current);
        self.last = None;
        self.run_end = prev.caret;
        Some(prev)
    }

    pub fn pop_redo(&mut self, current: EditSnapshot) -> Option<EditSnapshot> {
        let next = self.redo.pop()?;
        self.undo.push(current);
        self.last = None;
        self.run_end = next.caret;
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

impl TextEditState {
    pub fn new(id: impl Into<String>, caret: usize) -> Self {
        Self {
            id: id.into(),
            anchor: caret,
            caret,
        }
    }

    /// Normalized `(start, end)` byte range.
    #[inline]
    pub fn selection(&self) -> (usize, usize) {
        (self.anchor.min(self.caret), self.anchor.max(self.caret))
    }

    #[inline]
    pub fn has_selection(&self) -> bool {
        self.anchor != self.caret
    }

    /// Move the caret; `extend` keeps the anchor (Shift held).
    pub fn move_caret(&mut self, caret: usize, extend: bool) {
        self.caret = caret;
        if !extend {
            self.anchor = caret;
        }
    }
}

#[derive(Clone, Debug)]
pub struct InteractionState {
    pub pointer: [f32; 2],
    /// True between `PointerDown` and `PointerUp`; drives the live touch reticle.
    pub pointer_down: bool,
    pub hovered_key: Option<String>,
    pub pressed_key: Option<String>,
    pub focused_key: Option<String>,
    /// Caret + selection of the focused text field (survives scene reloads).
    pub edit: Option<TextEditState>,
    pub hits: Vec<HitRegion>,
    pub scrolls: HashMap<String, KineticScroll>,
    pub plot_hover: HashMap<String, [f32; 2]>,
    pub dragging_plot: Option<(String, f32, f32)>,
    /// Slider being scrubbed, keyed by node id.
    pub slider_drag: Option<String>,
    /// Range slider being scrubbed: `(node id, thumb 0=low 1=high)`.
    pub range_drag: Option<(String, u8)>,
    /// Engine-clock reading of the last caret move, in seconds. The caret is
    /// drawn for 500 ms after every edit, then blinks at 1 Hz.
    pub caret_epoch_s: f64,
    /// Visible half of the caret blink cycle (true = draw the caret).
    pub caret_visible: bool,
    /// Per-field undo/redo stacks, keyed by focus key (`input::<id>`).
    pub histories: HashMap<String, EditHistory>,
    /// Most recent tap, used to detect double/triple clicks.
    pub tap: Option<TapRecord>,
    /// Edge behaviour when a scroll container hits its bound.
    pub scroll_physics: ScrollPhysics,
    /// Active headless drag session.
    pub drag: Option<DragSession>,
    /// Active `NavStack` edge-swipe session.
    pub nav_swipe: Option<NavSwipe>,
}

/// Last tap on a field, for double/triple-click selection.
#[derive(Clone, Debug)]
pub struct TapRecord {
    pub key: String,
    pub x: f32,
    pub y: f32,
    pub at_s: f64,
    /// Consecutive taps at this spot (`1` = single, `2` = word, `3` = line).
    pub count: u8,
}

/// Pointer drag session for a `Draggable` node.
#[derive(Clone, Debug)]
pub struct DragSession {
    pub id: String,
    pub last: [f32; 2],
    pub total: [f32; 2],
}

/// Interactive edge-swipe-back session for a `NavStack`.
#[derive(Clone, Debug)]
pub struct NavSwipe {
    pub id: String,
    pub start_x: f32,
    pub width: f32,
    /// Current scrub progress `0..=1`.
    pub progress: f32,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            pointer: [0.0; 2],
            pointer_down: false,
            hovered_key: None,
            pressed_key: None,
            focused_key: None,
            edit: None,
            hits: Vec::new(),
            scrolls: HashMap::new(),
            plot_hover: HashMap::new(),
            dragging_plot: None,
            slider_drag: None,
            range_drag: None,
            caret_epoch_s: 0.0,
            // Nothing is focused yet, so the caret is "visible"; the first
            // focus/edit stamps the epoch and starts the blink.
            caret_visible: true,
            histories: HashMap::new(),
            tap: None,
            scroll_physics: ScrollPhysics::default(),
            drag: None,
            nav_swipe: None,
        }
    }
}

impl InteractionState {
    /// Restart the caret blink, keeping it solid for the next 500 ms.
    pub fn reset_caret_blink(&mut self, now_s: f64) {
        self.caret_epoch_s = now_s;
        self.caret_visible = true;
    }

    /// Seconds until the caret should toggle visibility, if a caret is live.
    pub fn caret_blink_deadline(&self, now_s: f64) -> Option<f64> {
        self.edit.as_ref()?;
        self.focused_key.as_ref()?;
        let phase = (now_s - self.caret_epoch_s).rem_euclid(1.0);
        Some(if phase < 0.5 { 0.5 - phase } else { 1.5 - phase })
    }

    pub fn clear_hits(&mut self) {
        self.hits.clear();
    }

    pub fn push_hit(&mut self, rect: ComputedRect, kind: HitKind) {
        self.hits.push(HitRegion { rect, kind });
    }

    /// Key string identifying the topmost interactive target under `(x, y)`.
    pub fn hit_topmost_interactive(&self, x: f32, y: f32) -> Option<(String, &HitKind)> {
        // Iterate in reverse paint order so top/child nodes win over parent containers.
        for region in self.hits.iter().rev() {
            if !region.contains(x, y) {
                continue;
            }
            match &region.kind {
                HitKind::Surface { id } => return Some((format!("surf::{id}"), &region.kind)),
                HitKind::Stat { id } => return Some((format!("stat::{id}"), &region.kind)),
                HitKind::TabButton { tabs_id, index } => {
                    return Some((format!("{tabs_id}::tab::{index}"), &region.kind))
                }
                HitKind::SegmentedItem { seg_id, index } => {
                    return Some((format!("{seg_id}::seg::{index}"), &region.kind))
                }
                HitKind::Button { id } => return Some((format!("btn::{id}"), &region.kind)),
                HitKind::ThemeToggle { id } => return Some((format!("toggle::{id}"), &region.kind)),
                HitKind::Switch { id } => return Some((format!("switch::{id}"), &region.kind)),
                HitKind::Input { id } => return Some((format!("input::{id}"), &region.kind)),
                HitKind::TableHeader {
                    table_id,
                    col_index,
                } => return Some((format!("{table_id}::th::{col_index}"), &region.kind)),
                HitKind::TableRow { table_id, row_id } => {
                    return Some((format!("{table_id}::tr::{row_id}"), &region.kind))
                }
                HitKind::FeedItem { feed_id, item_id } => {
                    return Some((format!("{feed_id}::feed::{item_id}"), &region.kind))
                }
                HitKind::GraphNode { graph_id, node_id } => {
                    return Some((format!("{graph_id}::node::{node_id}"), &region.kind))
                }
                HitKind::PlotLegendChip {
                    plot_id,
                    series_index,
                } => return Some((format!("{plot_id}::legend::{series_index}"), &region.kind)),
                HitKind::Slider { id } => return Some((format!("slider::{id}"), &region.kind)),
                HitKind::RangeSlider { id } => {
                    return Some((format!("range::{id}"), &region.kind))
                }
                HitKind::InputClear { id } => {
                    return Some((format!("clear::{id}"), &region.kind))
                }
                HitKind::TextArea { id } => return Some((format!("textarea::{id}"), &region.kind)),
                HitKind::ListItem { list_id, index, .. } => {
                    return Some((format!("{list_id}::li::{index}"), &region.kind))
                }
                HitKind::Pressable { id } => return Some((format!("press::{id}"), &region.kind)),
                HitKind::Draggable { id } => return Some((format!("drag::{id}"), &region.kind)),
                HitKind::NavEdge { id } => return Some((format!("nav::{id}"), &region.kind)),
                HitKind::OverlayBackdrop { id } => {
                    return Some((format!("scrim::{id}"), &region.kind))
                }
                HitKind::PlotArea { .. } | HitKind::ScrollContainer { .. } => {}
            }
        }
        None
    }

    pub fn hit_plot_area(&self, x: f32, y: f32) -> Option<(String, ComputedRect)> {
        for region in self.hits.iter().rev() {
            if region.contains(x, y) {
                if let HitKind::PlotArea { id } = &region.kind {
                    return Some((id.clone(), region.rect));
                }
            }
        }
        None
    }

    pub fn hit_topmost_scroll(&self, x: f32, y: f32) -> Option<(String, f32)> {
        for region in self.hits.iter().rev() {
            if !region.contains(x, y) {
                continue;
            }
            if let HitKind::ScrollContainer { id, max_scroll_y } = &region.kind {
                if *max_scroll_y > 0.0 {
                    return Some((id.clone(), *max_scroll_y));
                }
            }
        }
        None
    }

    pub fn step_scrolls(&mut self, dt: f32) -> bool {
        let mut moving = false;
        let physics = self.scroll_physics;
        for scroll in self.scrolls.values_mut() {
            if scroll.step(dt, physics) {
                moving = true;
            }
        }
        moving
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overscroll_resists_per_platform_and_springs_home() {
        // Drag 100px past the top of a container that is already at 0.
        let drag = |physics: ScrollPhysics| {
            let mut s = KineticScroll {
                offset_y: 0.0,
                velocity_y: 0.0,
                max_y: 400.0,
                overscroll: 0.0,
            };
            // A downward drag (positive delta) at the top edge pulls past it.
            s.impulse(100.0, physics);
            s
        };

        let clamped = drag(ScrollPhysics::Clamped);
        assert_eq!(clamped.overscroll, 0.0, "desktop does not move past the edge");
        assert_eq!(clamped.offset_y, 0.0);

        let mut apple = drag(ScrollPhysics::RubberBand);
        assert!(
            (apple.overscroll.abs() - 55.0).abs() < 0.01,
            "rubber band takes 55% of the drag, got {}",
            apple.overscroll
        );
        assert_eq!(apple.offset_y, 0.0, "the offset itself never leaves bounds");
        assert!(
            (apple.visual_offset() + 55.0).abs() < 0.01,
            "pulling down at the top shifts content down (negative offset)"
        );

        let android = drag(ScrollPhysics::Stretch);
        assert!(
            android.overscroll.abs() < apple.overscroll.abs(),
            "Android stretch should resist harder than the Apple band"
        );
        assert!(android.overscroll.abs() > 0.0);

        // The spring home is monotonic and finishes; Android returns faster.
        let mut apple = apple;
        let mut android = android;
        let mut apple_frames = 0;
        while apple.overscroll != 0.0 && apple_frames < 600 {
            apple.step(1.0 / 120.0, ScrollPhysics::RubberBand);
            apple_frames += 1;
        }
        let mut android_frames = 0;
        while android.overscroll != 0.0 && android_frames < 600 {
            android.step(1.0 / 120.0, ScrollPhysics::Stretch);
            android_frames += 1;
        }
        assert!(apple_frames > 0 && apple_frames < 600, "rubber band settles");
        assert!(
            android_frames < apple_frames,
            "Android settles faster ({android_frames} vs {apple_frames} frames)"
        );
        assert_eq!(apple.overscroll, 0.0);
        assert!(!apple.step(1.0 / 120.0, ScrollPhysics::RubberBand), "idles");
    }

    #[test]
    fn a_fling_into_the_edge_turns_into_overscroll_and_stops() {
        let mut s = KineticScroll {
            offset_y: 390.0,
            velocity_y: 2200.0,
            max_y: 400.0,
            overscroll: 0.0,
        };
        let moving = s.step(1.0 / 60.0, ScrollPhysics::RubberBand);
        assert_eq!(s.offset_y, 400.0, "clamped to the bottom");
        assert!(s.overscroll > 0.0, "the remaining travel becomes a bounce");
        assert_eq!(s.velocity_y, 0.0, "the fling is spent on the edge");
        assert!(moving, "a live bounce keeps the frame loop awake");

        // Settles back to a resting bottom edge.
        for _ in 0..600 {
            if !s.step(1.0 / 120.0, ScrollPhysics::RubberBand) {
                break;
            }
        }
        assert_eq!(s.overscroll, 0.0);
        assert_eq!(s.offset_y, 400.0);
    }
}
