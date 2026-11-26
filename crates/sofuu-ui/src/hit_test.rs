//! Spatial hit-testing, pointer/keyboard input routing, and kinetic scroll physics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::layout::ComputedRect;

/// Raw OS input events dispatched from `sofuu-gpu` (Thread 0) into `UiEngine`.
#[derive(Clone, Debug, PartialEq)]
pub enum InputEvent {
    PointerMove { x: f32, y: f32 },
    PointerDown { x: f32, y: f32, button: u8 },
    PointerUp { x: f32, y: f32, button: u8 },
    Scroll { x: f32, y: f32, delta_x: f32, delta_y: f32 },
    KeyDown { key: String, modifiers: u32 },
    TextInput { text: String },
    Resize { width: f32, height: f32, scale_factor: f32 },
    CloseRequested,
}

/// High-level semantic events emitted back to the Sofuu agent / host application.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UIEvent {
    Click { id: String },
    TabChanged { id: String, index: usize },
    Scrolled { id: String, offset_y: f32 },
    WindowClosed,
}

#[derive(Clone, Debug)]
pub enum HitKind {
    Surface { id: String },
    TabButton { tabs_id: String, index: usize },
    ScrollContainer { id: String, max_scroll_y: f32 },
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

#[derive(Clone, Copy, Debug, Default)]
pub struct KineticScroll {
    pub offset_y: f32,
    pub velocity_y: f32,
    pub max_y: f32,
}

impl KineticScroll {
    pub fn impulse(&mut self, delta_y: f32) {
        self.offset_y = (self.offset_y - delta_y).clamp(0.0, self.max_y);
        self.velocity_y = (-delta_y * 14.0).clamp(-3600.0, 3600.0);
    }

    pub fn step(&mut self, dt: f32) -> bool {
        self.offset_y = self.offset_y.clamp(0.0, self.max_y);
        if self.velocity_y.abs() < 4.0 {
            self.velocity_y = 0.0;
            return false;
        }
        let dt = dt.clamp(0.0, 0.05);
        self.offset_y = (self.offset_y + self.velocity_y * dt).clamp(0.0, self.max_y);
        // Exponential friction damping (~0.90 per 16.6ms frame)
        let decay = (0.0018_f32).powf(dt);
        self.velocity_y *= decay;
        if self.offset_y <= 0.0 || self.offset_y >= self.max_y {
            self.velocity_y *= 0.25;
        }
        self.velocity_y.abs() >= 4.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct InteractionState {
    pub pointer: [f32; 2],
    pub hovered_key: Option<String>,
    pub pressed_key: Option<String>,
    pub hits: Vec<HitRegion>,
    pub scrolls: HashMap<String, KineticScroll>,
}

impl InteractionState {
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
                HitKind::Surface { id } => return Some((id.clone(), &region.kind)),
                HitKind::TabButton { tabs_id, index } => {
                    return Some((format!("{tabs_id}::tab::{index}"), &region.kind))
                }
                HitKind::ScrollContainer { .. } => {}
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
        for scroll in self.scrolls.values_mut() {
            if scroll.step(dt) {
                moving = true;
            }
        }
        moving
    }
}
