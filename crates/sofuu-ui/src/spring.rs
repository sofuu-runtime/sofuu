//! 120Hz critically-damped spring interpolation engine.
//!
//! Gives every state change (hover glow, tab indicator, card bounds, scroll, values)
//! fluid physical motion while snapping cleanly to `settled = true` so idle windows
//! consume 0% CPU and 0% GPU.

use std::collections::HashMap;

/// Tuning parameters for a harmonic spring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringConfig {
    /// Spring stiffness (`omega^2` equivalent). Higher = faster response.
    pub stiffness: f32,
    /// Viscous damping (`2 * zeta * omega`). `>= 2*sqrt(stiffness)` is critically damped.
    pub damping: f32,
    /// Position & velocity threshold below which the spring snaps and sleeps.
    pub epsilon: f32,
}

impl SpringConfig {
    /// Snappy UI transition (~180ms settle, critically damped).
    pub const UI_DEFAULT: Self = Self {
        stiffness: 320.0,
        damping: 35.8,
        epsilon: 0.002,
    };

    /// Micro-interaction (hover/press states, ~90ms settle).
    pub const SNAPPY: Self = Self {
        stiffness: 620.0,
        damping: 49.8,
        epsilon: 0.002,
    };

    /// Smooth layout / drawer glide (~280ms settle).
    pub const GENTLE: Self = Self {
        stiffness: 180.0,
        damping: 26.8,
        epsilon: 0.002,
    };
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self::UI_DEFAULT
    }
}

/// Scalar spring integrator.
#[derive(Clone, Copy, Debug)]
pub struct SpringState {
    pub value: f32,
    pub target: f32,
    pub velocity: f32,
    pub settled: bool,
}

impl SpringState {
    pub const fn new(initial: f32) -> Self {
        Self {
            value: initial,
            target: initial,
            velocity: 0.0,
            settled: true,
        }
    }

    pub fn set_target(&mut self, target: f32, epsilon: f32) {
        if (self.target - target).abs() > epsilon * 0.25 {
            self.target = target;
            self.settled = false;
        } else if (self.value - target).abs() > epsilon || self.velocity.abs() > epsilon {
            self.target = target;
            self.settled = false;
        }
    }

    pub fn snap(&mut self, value: f32) {
        self.value = value;
        self.target = value;
        self.velocity = 0.0;
        self.settled = true;
    }

    /// Advance the spring by `dt` seconds. Sub-steps at <= 1/120s for unconditional stability.
    /// Returns `true` if still animating (`!settled`).
    pub fn step(&mut self, dt: f32, cfg: SpringConfig) -> bool {
        if self.settled {
            return false;
        }
        let clamped_dt = dt.clamp(0.0, 0.05);
        let max_substep = 1.0 / 120.0;
        let steps = (clamped_dt / max_substep).ceil().max(1.0) as u32;
        let h = clamped_dt / steps as f32;

        for _ in 0..steps {
            let disp = self.value - self.target;
            let accel = -cfg.stiffness * disp - cfg.damping * self.velocity;
            self.velocity += accel * h;
            self.value += self.velocity * h;
        }

        if (self.value - self.target).abs() <= cfg.epsilon
            && self.velocity.abs() <= cfg.epsilon * 8.0
        {
            self.value = self.target;
            self.velocity = 0.0;
            self.settled = true;
            return false;
        }
        true
    }
}

/// Stable 64-bit FNV-1a key from node ID and property tag (`"hover"`, `"active"`, `"tab_x"`, etc.).
#[inline]
pub fn spring_key(node_id: &str, prop: &str) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for b in node_id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h ^= 0xff;
    h = h.wrapping_mul(PRIME);
    for b in prop.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Manages active per-node springs without polluting the immutable Scene IR.
#[derive(Clone, Debug, Default)]
pub struct SpringBank {
    springs: HashMap<u64, (SpringState, SpringConfig, u64)>,
    epoch: u64,
}

impl SpringBank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the frame epoch (used to prune springs whose nodes disappeared).
    pub fn begin_frame(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Drive or initialize a named spring property for `node_id`, returning its current value.
    pub fn animate(&mut self, node_id: &str, prop: &str, target: f32, cfg: SpringConfig) -> f32 {
        let key = spring_key(node_id, prop);
        let epoch = self.epoch;
        let entry = self
            .springs
            .entry(key)
            .or_insert_with(|| (SpringState::new(target), cfg, epoch));
        entry.1 = cfg;
        entry.2 = epoch;
        entry.0.set_target(target, cfg.epsilon);
        entry.0.value
    }

    /// Read current animated value without retargeting, or `default_val` if untracked.
    pub fn get(&self, node_id: &str, prop: &str, default_val: f32) -> f32 {
        let key = spring_key(node_id, prop);
        self.springs
            .get(&key)
            .map(|(s, _, _)| s.value)
            .unwrap_or(default_val)
    }

    /// Step all springs by `dt` seconds. Returns `true` if any spring is still moving.
    pub fn step_all(&mut self, dt: f32) -> bool {
        let mut any_active = false;
        for (state, cfg, _) in self.springs.values_mut() {
            if state.step(dt, *cfg) {
                any_active = true;
            }
        }
        any_active
    }

    /// Remove springs that were not touched in the last frame (keeps memory tight across patches).
    pub fn prune_unvisited(&mut self) {
        let epoch = self.epoch;
        self.springs.retain(|_, (_, _, last_epoch)| *last_epoch == epoch);
    }

    pub fn has_active(&self) -> bool {
        self.springs.values().any(|(s, _, _)| !s.settled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_converges_and_sleeps() {
        let mut s = SpringState::new(0.0);
        s.set_target(1.0, SpringConfig::UI_DEFAULT.epsilon);
        assert!(!s.settled);

        let dt = 1.0 / 120.0;
        let mut ticks = 0;
        while s.step(dt, SpringConfig::UI_DEFAULT) && ticks < 240 {
            ticks += 1;
        }
        assert!(s.settled, "spring should settle cleanly within 2 seconds");
        assert!((s.value - 1.0).abs() < 1e-4);
        assert_eq!(s.velocity, 0.0);
    }

    #[test]
    fn spring_bank_retains_progress_and_prunes() {
        let mut bank = SpringBank::new();
        bank.begin_frame();
        let v0 = bank.animate("btn_1", "hover", 1.0, SpringConfig::SNAPPY);
        // First initialization snaps immediately to initial value (1.0) unless retargeted.
        assert_eq!(v0, 1.0);

        // Retarget from 0.0 -> 1.0:
        bank.animate("btn_2", "hover", 0.0, SpringConfig::SNAPPY);
        bank.animate("btn_2", "hover", 1.0, SpringConfig::SNAPPY);
        assert!(bank.step_all(1.0 / 120.0));
        let mid = bank.get("btn_2", "hover", 0.0);
        assert!(mid > 0.0 && mid < 1.0);

        bank.begin_frame();
        let _ = bank.animate("btn_2", "hover", 1.0, SpringConfig::SNAPPY);
        bank.prune_unvisited();
        assert_eq!(bank.get("btn_1", "hover", -1.0), -1.0);
    }
}
