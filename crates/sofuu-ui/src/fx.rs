//! GPU UI effects & choreography (`sofuu.fx`).
//!
//! Two cheap layers sit on top of the retained scene IR:
//!
//! * **Motion** — CPU springs/timelines evaluate a node's opacity, translation and scale and
//!   rewrite the primitives the node emitted. Layout, hit-testing and `audit.rs` keep seeing the
//!   untouched `ComputedRect`.
//! * **Effect metadata** — compact `EffectQuadInstance` / `ParticleInstance` records and
//!   `BackdropRegion` markers are appended to the ordered [`DrawList`]. All pixel math (SDF
//!   masks, shimmer sweeps, aurora noise, particles, Kawase blur) executes on the GPU.
//!
//! Continuous ambient effects opt out on `reduced_motion` or when the host pauses ambient
//! playback; one-shots (ripples, shards, shake, text reveals) always settle and drop to 0% idle.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::draw::{
    lerp_rgba8, rgba8, scale_alpha, with_alpha, BackdropRegion, DrawList, EffectQuadInstance,
    ParticleInstance,
};
use crate::ir::Node;
use crate::layout::ComputedRect;
use crate::spring::SpringConfig;
use crate::theme::{Theme, Tone};

/// Effect mode ids written into [`EffectQuadInstance::kind_and_clip`].
pub const EFFECT_SHIMMER: u16 = 0;
pub const EFFECT_TOUCH_RIPPLE: u16 = 1;
pub const EFFECT_BORDER_BEAM: u16 = 2;
pub const EFFECT_CURSOR_GLOW: u16 = 3;
pub const EFFECT_AURORA: u16 = 4;
pub const EFFECT_PULSE_RING: u16 = 5;
pub const EFFECT_VOICE_ORB: u16 = 6;

/// Particle shapes packed into the low four bits of [`ParticleInstance::shape_clip`].
pub const PARTICLE_CIRCLE: u32 = 0;
pub const PARTICLE_RIBBON: u32 = 1;
pub const PARTICLE_SPARK: u32 = 2;

/// Op-chosen animation spring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FxSpring {
    /// Micro-interaction (~90 ms settle).
    #[default]
    Snappy,
    /// Standard UI transition (~180 ms settle).
    Default,
    /// Gentle glide (~280 ms settle).
    Gentle,
}

impl FxSpring {
    pub const fn config(self) -> SpringConfig {
        match self {
            FxSpring::Snappy => SpringConfig::SNAPPY,
            FxSpring::Default => SpringConfig::UI_DEFAULT,
            FxSpring::Gentle => SpringConfig::GENTLE,
        }
    }
}

/// One declarative effect or choreography preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FxKind {
    // ── Motion / choreography ────────────────────────────────────────────────
    /// Enter preset: `fade`, `slide`, `scale_pop` or `fade_slide`.
    Enter,
    /// Opacity-only enter.
    Fade,
    /// Translation-only enter (`slide_x`, `slide_y`).
    Slide,
    /// Scale-up enter (`scale` is the starting scale).
    ScalePop,
    /// Horizontal damped shake impulse.
    Shake,
    /// Odometer interpolation for text/stat values.
    NumberRoll,
    /// Shared-element bounds morph keyed by `shared_key`.
    Shared,
    // ── Surface / burst effects ──────────────────────────────────────────────
    Shimmer,
    TouchRipple,
    BorderBeam,
    CursorGlow,
    Aurora,
    PulseRing,
    VoiceOrb,
    BackdropBlur,
    TextShimmer,
    TextScramble,
    Typewriter,
    Confetti,
    SparkBurst,
}

impl FxKind {
    /// Ambient modes request frames continuously until the host pauses them.
    pub const fn is_ambient(self) -> bool {
        matches!(
            self,
            FxKind::Shimmer
                | FxKind::BorderBeam
                | FxKind::CursorGlow
                | FxKind::Aurora
                | FxKind::PulseRing
                | FxKind::VoiceOrb
                | FxKind::TextShimmer
        )
    }

    /// One-shots are consumed automatically on mount/press unless `auto = false`.
    pub const fn auto_trigger(self) -> bool {
        !matches!(self, FxKind::Shake | FxKind::Confetti | FxKind::SparkBurst)
    }
}

/// Declarative effect attached to a scene node (`Node::fx`).
///
/// Fields are deliberately flat: the JS bridge authors an object literal, serde keeps omitted
/// values at their defaults, and the compiler only reads the fields relevant to `kind`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct FxSpec {
    pub kind: FxKind,
    /// Enter geometry preset: `fade`, `slide`, `slide_up`, `scale_pop`, `fade_slide`.
    pub preset: Option<String>,
    pub spring: FxSpring,
    /// Delay before the timeline starts (stagger).
    pub delay_ms: f32,
    /// Explicit duration in ms (`0` = per-kind default).
    pub duration_ms: f32,
    pub slide_x: f32,
    pub slide_y: f32,
    /// Starting scale for `scale_pop` (`1.0` means no change).
    pub scale: f32,
    /// Optional numeric source/target for `number_roll`.
    pub from: Option<f64>,
    pub to: Option<f64>,
    pub precision: Option<usize>,
    pub prefix: String,
    pub suffix: String,
    pub tone: Option<Tone>,
    pub color0: Option<u32>,
    pub color1: Option<u32>,
    pub color2: Option<u32>,
    /// Loop speed multiplier / sweep speed (ambient effects).
    pub speed: f32,
    /// Alpha / glow strength.
    pub intensity: f32,
    /// Glow radius, pulse radius or shimmer band width (logical px).
    pub radius: f32,
    /// Border/beam width in logical px.
    pub width: f32,
    /// Reactive scalar for `voice_orb` (`0.0..=1.0`).
    pub amplitude: f32,
    /// Ambient effect enabled (`false` keeps the node clean).
    pub active: bool,
    /// Particle count for `confetti` / `spark_burst`.
    pub count: usize,
    /// Auto-trigger one-shots on mount (defaults to `false` for impulse bursts).
    pub auto: bool,
    /// Matched-geometry key for `shared`.
    pub shared_key: Option<String>,
    /// Kawase blur radius for `backdrop_blur` (logical px).
    pub blur: f32,
    /// Inner specular rim width for `backdrop_blur`.
    pub rim: f32,
    /// Deterministic seed for scramble/particles.
    pub seed: u32,
    /// Run an exit timeline when the node is hidden (fade/slide).
    pub exit: bool,
}

impl Default for FxSpec {
    fn default() -> Self {
        Self {
            kind: FxKind::Shimmer,
            preset: None,
            spring: FxSpring::Snappy,
            delay_ms: 0.0,
            duration_ms: 0.0,
            slide_x: 0.0,
            slide_y: 0.0,
            scale: 0.92,
            from: None,
            to: None,
            precision: None,
            prefix: String::new(),
            suffix: String::new(),
            tone: None,
            color0: None,
            color1: None,
            color2: None,
            speed: 1.0,
            intensity: 1.0,
            radius: 24.0,
            width: 1.5,
            amplitude: 0.0,
            active: true,
            count: 48,
            auto: false,
            shared_key: None,
            blur: 18.0,
            rim: 1.0,
            seed: 1,
            exit: false,
        }
    }
}

impl FxSpec {
    pub fn new(kind: FxKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    pub fn with_tone(mut self, tone: Tone) -> Self {
        self.tone = Some(tone);
        self
    }

    pub fn with_delay_ms(mut self, delay_ms: f32) -> Self {
        self.delay_ms = delay_ms.max(0.0);
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: f32) -> Self {
        self.duration_ms = duration_ms.max(0.0);
        self
    }

    pub fn with_slide(mut self, x: f32, y: f32) -> Self {
        self.slide_x = x;
        self.slide_y = y;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_shared_key(mut self, key: impl Into<String>) -> Self {
        self.shared_key = Some(key.into());
        self
    }

    fn duration_secs(&self) -> f32 {
        if self.duration_ms > 0.0 {
            return self.duration_ms / 1000.0;
        }
        match self.kind {
            FxKind::NumberRoll => 0.65,
            FxKind::TextScramble | FxKind::Typewriter => 0.70,
            FxKind::Shake => 0.45,
            FxKind::Confetti => 1.45,
            FxKind::SparkBurst => 0.80,
            FxKind::TouchRipple | FxKind::PulseRing => 0.75,
            _ => 0.28,
        }
    }

    fn tone_color(&self, theme: &Theme) -> u32 {
        theme.tone_color(self.tone.unwrap_or(Tone::Info))
    }

    fn primary(&self, theme: &Theme) -> u32 {
        self.color0.unwrap_or_else(|| self.tone_color(theme))
    }
}

// ── Rust authoring helpers (parity with the `sofuu.fx` JS decorators) ────────────────────────

/// Attach one or more effect specs to a node.
pub fn attach(mut node: Node, specs: impl IntoIterator<Item = FxSpec>) -> Node {
    node.fx.extend(specs);
    node
}

/// `fx.enter(node, { preset })` — fade/slide/scale entrance timeline.
pub fn enter(node: Node, preset: impl Into<String>) -> Node {
    attach(
        node,
        [FxSpec {
            preset: Some(preset.into()),
            ..FxSpec::new(FxKind::Enter)
        }],
    )
}

/// `fx.backdropBlur(node, { blur, rim })` — half-res Kawase frosted-glass marker.
pub fn backdrop_blur(node: Node, blur: f32, rim: f32) -> Node {
    attach(
        node,
        [FxSpec {
            blur,
            rim,
            ..FxSpec::new(FxKind::BackdropBlur)
        }],
    )
}

/// Apply a staggered `enter` cadence to siblings (`fx.stagger(items, { delayMs, slideY })`).
pub fn stagger<'a>(nodes: impl IntoIterator<Item = &'a mut Node>, delay_ms: f32, slide_y: f32) {
    for (index, node) in nodes.into_iter().enumerate() {
        node.fx.push(FxSpec {
            delay_ms: index as f32 * delay_ms,
            slide_y,
            ..FxSpec::new(FxKind::Enter)
        });
    }
}

/// Per-node transform produced by the motion layer. Coordinates are logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeFx {
    pub opacity: f32,
    pub translate: [f32; 2],
    pub scale: f32,
    pub scale_origin: [f32; 2],
    pub shake: [f32; 2],
}

impl NodeFx {
    pub fn for_rect(rect: ComputedRect) -> Self {
        Self {
            opacity: 1.0,
            translate: [0.0, 0.0],
            scale: 1.0,
            scale_origin: [rect.x + rect.w * 0.5, rect.y + rect.h * 0.5],
            shake: [0.0, 0.0],
        }
    }

    fn total_translate(&self) -> [f32; 2] {
        [
            self.translate[0] + self.shake[0],
            self.translate[1] + self.shake[1],
        ]
    }

    fn map_point(&self, x: f32, y: f32) -> [f32; 2] {
        let t = self.total_translate();
        [
            self.scale_origin[0] + (x - self.scale_origin[0]) * self.scale + t[0],
            self.scale_origin[1] + (y - self.scale_origin[1]) * self.scale + t[1],
        ]
    }
}

/// Start offsets into every flat GPU buffer, used to rewrite exactly the primitives a node
/// emitted without touching siblings or parent content.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawMark {
    pub quads: usize,
    pub ribbons: usize,
    pub strokes: usize,
    pub glyphs: usize,
    pub effects: usize,
    pub backdrops: usize,
    pub clips: usize,
}

impl DrawMark {
    pub fn capture(dl: &DrawList) -> Self {
        Self {
            quads: dl.quads.len(),
            ribbons: dl.ribbons.len(),
            strokes: dl.strokes.len(),
            glyphs: dl.glyphs.len(),
            effects: dl.effects.len(),
            backdrops: dl.backdrops.len(),
            clips: dl.clips.len(),
        }
    }

    /// Translate/scale/alpha-rewrite every primitive recorded after this mark.
    pub fn apply_fx(self, dl: &mut DrawList, fx: &NodeFx) {
        let opacity = fx.opacity.clamp(0.0, 1.0);
        let alpha = |c: u32| scale_alpha(c, opacity);

        for q in &mut dl.quads[self.quads..] {
            let (x, y) = (q.rect[0], q.rect[1]);
            let (w, h) = (q.rect[2], q.rect[3]);
            let p0 = fx.map_point(x, y);
            let p1 = fx.map_point(x + w, y + h);
            q.rect = [
                p0[0],
                p0[1],
                (p1[0] - p0[0]).max(0.0),
                (p1[1] - p0[1]).max(0.0),
            ];
            for r in &mut q.radii {
                *r = (*r * fx.scale.abs()).max(0.0);
            }
            q.color_top = alpha(q.color_top);
            q.color_bottom = alpha(q.color_bottom);
            q.border_color = alpha(q.border_color);
            q.shadow_color = alpha(q.shadow_color);
        }

        for r in &mut dl.ribbons[self.ribbons..] {
            let p0 = fx.map_point(r.x0, r.y0_top);
            let p1 = fx.map_point(r.x1, r.y1_top);
            let p2 = fx.map_point(r.x0, r.y0_bot);
            let p3 = fx.map_point(r.x1, r.y1_bot);
            r.x0 = p0[0];
            r.y0_top = p0[1];
            r.x1 = p1[0];
            r.y1_top = p1[1];
            r.y0_bot = p2[1];
            r.y1_bot = p3[1];
            r.color_top = alpha(r.color_top);
            r.color_bottom = alpha(r.color_bottom);
        }

        for s in &mut dl.strokes[self.strokes..] {
            let p0 = fx.map_point(s.p0[0], s.p0[1]);
            let p1 = fx.map_point(s.p1[0], s.p1[1]);
            s.p0 = p0;
            s.p1 = p1;
            s.width = (s.width * fx.scale.abs()).max(0.0);
            s.color = alpha(s.color);
        }

        for g in &mut dl.glyphs[self.glyphs..] {
            let p = fx.map_point(g.pos[0], g.pos[1]);
            g.pos = p;
            g.size = [
                (g.size[0] * fx.scale.abs()).max(0.0),
                (g.size[1] * fx.scale.abs()).max(0.0),
            ];
            g.color = alpha(g.color);
        }

        // Effect overlays and backdrop markers are emitted at the node's resting rect, so a
        // sliding/scaling node must carry them along or the beam/shimmer detaches for a frame.
        for e in &mut dl.effects[self.effects..] {
            let (x, y) = (e.rect[0], e.rect[1]);
            let (w, h) = (e.rect[2], e.rect[3]);
            let p0 = fx.map_point(x, y);
            let p1 = fx.map_point(x + w, y + h);
            e.rect = [
                p0[0],
                p0[1],
                (p1[0] - p0[0]).max(0.0),
                (p1[1] - p0[1]).max(0.0),
            ];
            for r in &mut e.radii {
                *r = (*r * fx.scale.abs()).max(0.0);
            }
            e.color0 = alpha(e.color0);
            e.color1 = alpha(e.color1);
            e.color2 = alpha(e.color2);
        }

        for b in &mut dl.backdrops[self.backdrops..] {
            let (x, y) = (b.rect[0], b.rect[1]);
            let (w, h) = (b.rect[2], b.rect[3]);
            let p0 = fx.map_point(x, y);
            let p1 = fx.map_point(x + w, y + h);
            b.rect = [
                p0[0],
                p0[1],
                (p1[0] - p0[0]).max(0.0),
                (p1[1] - p0[1]).max(0.0),
            ];
            for r in &mut b.radii {
                *r = (*r * fx.scale.abs()).max(0.0);
            }
            b.tint = alpha(b.tint);
            b.rim = alpha(b.rim);
        }

        for c in &mut dl.clips[self.clips..] {
            let p0 = fx.map_point(c.x, c.y);
            let p1 = fx.map_point(c.x + c.w, c.y + c.h);
            c.x = p0[0];
            c.y = p0[1];
            c.w = (p1[0] - p0[0]).max(0.0);
            c.h = (p1[1] - p0[1]).max(0.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Motion {
    started: bool,
    elapsed: f32,
    delay: f32,
    duration: f32,
    value: f32,
    velocity: f32,
    target: f32,
    spring: SpringConfig,
    settled: bool,
    from_rect: Option<ComputedRect>,
    to_rect: Option<ComputedRect>,
    source: u64,
}

impl Motion {
    fn new(spec: &FxSpec, duration: f32) -> Self {
        Self {
            started: false,
            elapsed: 0.0,
            delay: (spec.delay_ms / 1000.0).max(0.0),
            duration,
            value: 0.0,
            velocity: 0.0,
            target: 1.0,
            spring: spec.spring.config(),
            settled: false,
            from_rect: None,
            to_rect: None,
            source: 0,
        }
    }

    fn restart(&mut self) {
        self.started = true;
        self.elapsed = 0.0;
        self.value = 0.0;
        self.velocity = 0.0;
        self.target = 1.0;
        self.settled = false;
    }

    fn step(&mut self, dt: f32) {
        if self.delay > 0.0 {
            self.delay = (self.delay - dt).max(0.0);
            return;
        }
        if !self.started {
            self.started = true;
        }
        self.elapsed += dt;
        if self.settled {
            return;
        }
        let clamped = dt.clamp(0.0, 0.05);
        let steps = (clamped / (1.0 / 120.0)).ceil().max(1.0) as u32;
        let h = clamped / steps as f32;
        for _ in 0..steps {
            let disp = self.value - self.target;
            let accel = -self.spring.stiffness * disp - self.spring.damping * self.velocity;
            self.velocity += accel * h;
            self.value += self.velocity * h;
        }
        if (self.value - self.target).abs() <= self.spring.epsilon
            && self.velocity.abs() <= self.spring.epsilon * 8.0
        {
            self.value = self.target;
            self.velocity = 0.0;
            self.settled = true;
        }
        // Spring presets settle a little after the nominal preset duration; snap once the
        // timeline budget is spent so one-shots have an exact end state (and exact idle).
        if self.progress() >= 1.0 && !self.settled {
            self.value = self.target;
            self.velocity = 0.0;
            self.settled = true;
        }
    }

    fn progress(&self) -> f32 {
        (self.elapsed / self.duration.max(0.001)).clamp(0.0, 1.0)
    }

    fn done(&self) -> bool {
        (self.settled || self.progress() >= 1.0) && self.delay <= 0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Ripple {
    node: u64,
    start: f32,
    origin: [f32; 2],
    seed: u32,
}

#[derive(Clone, Copy, Debug)]
struct ShakeState {
    start: f32,
    amplitude: f32,
    duration: f32,
}

#[derive(Clone, Debug)]
struct PendingBurst {
    target: Option<String>,
    kind: FxKind,
    origin: Option<[f32; 2]>,
    count: usize,
    seed: u32,
    age: f32,
}

#[derive(Clone, Debug)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    size: [f32; 2],
    angle: f32,
    spin: f32,
    color: u32,
    shape: u32,
    life: f32,
    max_life: f32,
    drag: f32,
    gravity: f32,
}

#[derive(Clone, Copy, Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Stateful motion, one-shot and particle scheduler.
#[derive(Debug, Default)]
pub struct FxEngine {
    motions: HashMap<(String, usize), Motion>,
    ripples: Vec<Ripple>,
    shakes: HashMap<String, ShakeState>,
    particles: Vec<Particle>,
    pending: Vec<PendingBurst>,
    prev_seen: HashSet<String>,
    cur_seen: HashSet<String>,
    pressed: HashSet<String>,
    shared_prev: HashMap<String, ComputedRect>,
    shared_now: Vec<(String, ComputedRect)>,
    ambient: bool,
    reduced_motion: bool,
    ambient_paused: bool,
    time: f32,
}

impl FxEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// OS / app accessibility preference. Ambient effects freeze into static refined styling;
    /// one-shot transitions still run so navigation remains understandable.
    pub fn set_reduced_motion(&mut self, enabled: bool) {
        self.reduced_motion = enabled;
    }

    pub fn reduced_motion(&self) -> bool {
        self.reduced_motion
    }

    /// Host hint: window unfocused/backgrounded or OS low-power mode.
    pub fn set_ambient_paused(&mut self, paused: bool) {
        self.ambient_paused = paused;
    }

    pub fn ambient_paused(&self) -> bool {
        self.ambient_paused
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    /// Trigger an impulse burst or shake by id. `target = None` uses the supplied origin.
    pub fn trigger(
        &mut self,
        target: Option<String>,
        kind: FxKind,
        count: usize,
        origin: Option<[f32; 2]>,
        seed: u32,
    ) {
        if kind == FxKind::Shake {
            if let Some(id) = target {
                self.shakes.insert(
                    id,
                    ShakeState {
                        start: self.time,
                        amplitude: 10.0,
                        duration: 0.42,
                    },
                );
            }
            return;
        }
        if !matches!(kind, FxKind::Confetti | FxKind::SparkBurst) {
            return;
        }
        self.pending.push(PendingBurst {
            target,
            kind,
            origin,
            count: count.max(1).min(512),
            seed: if seed == 0 { 1 } else { seed },
            age: 0.0,
        });
    }

    /// Trigger a horizontal damped shake for a node.
    pub fn shake(&mut self, node_id: &str) {
        self.shakes.insert(
            node_id.to_string(),
            ShakeState {
                start: self.time,
                amplitude: 10.0,
                duration: 0.42,
            },
        );
    }

    /// True while a triggered burst is waiting for its target node to be compiled this frame.
    pub fn has_pending_burst(&self, node_id: &str) -> bool {
        self.pending
            .iter()
            .any(|b| b.target.as_deref() == Some(node_id))
    }

    /// Resolve a targeted burst once its node rect is known. The compiler calls this for every
    /// node, so `fx.burst('confetti', doneBtn)` also works on a target without `fx` specs.
    pub fn resolve_burst(&mut self, node_id: &str, rect: ComputedRect) {
        let Some(index) = self
            .pending
            .iter()
            .position(|b| b.target.as_deref() == Some(node_id))
        else {
            return;
        };
        let burst = self.pending.remove(index);
        let origin = burst
            .origin
            .unwrap_or([rect.x + rect.w * 0.5, rect.y + rect.h * 0.5]);
        self.spawn_particles(burst.kind, origin, burst.count, burst.seed);
    }

    /// Called once before a compile pass.
    pub fn begin_compile(&mut self) {
        self.ambient = false;
        self.cur_seen.clear();
        self.shared_now.clear();
    }

    /// Called after the compile pass has emitted every node.
    pub fn end_frame(&mut self) {
        self.prev_seen.clear();
        self.prev_seen.extend(self.cur_seen.iter().cloned());
        self.cur_seen.clear();
        for (key, rect) in self.shared_now.drain(..) {
            self.shared_prev.insert(key, rect);
        }
        let still_pressed = &self.pressed;
        self.shakes
            .retain(|id, _| still_pressed.contains(id) || self.cur_seen.contains(id));
    }

    /// Advance timelines, ripples, particles and pending bursts.
    pub fn step(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.05);
        self.time = (self.time + dt).rem_euclid(86_400.0);

        for m in self.motions.values_mut() {
            m.step(dt);
        }

        for r in &mut self.ripples {
            let _ = r;
        }
        self.ripples.retain(|r| self.time - r.start < 0.8);

        self.shakes
            .retain(|_, s| self.time - s.start < s.duration + 0.1);

        for p in &mut self.particles {
            p.life -= dt;
            if p.life <= 0.0 {
                continue;
            }
            p.vel[1] += p.gravity * dt;
            let damp = (1.0 - p.drag * dt).clamp(0.0, 1.0);
            p.vel[0] *= damp;
            p.vel[1] *= damp;
            p.pos[0] += p.vel[0] * dt;
            p.pos[1] += p.vel[1] * dt;
            p.angle += p.spin * dt;
        }
        self.particles.retain(|p| p.life > 0.0);

        let mut pending = std::mem::take(&mut self.pending);
        for mut burst in pending.drain(..) {
            burst.age += dt;
            if burst.target.is_none() {
                let origin = burst.origin.unwrap_or([0.0, 0.0]);
                self.spawn_particles(burst.kind, origin, burst.count, burst.seed);
            } else if burst.age < 0.35 {
                self.pending.push(burst);
            }
        }
        // Keep the original queue (minus consumed immediate bursts) at the front.
        pending.append(&mut self.pending);
        self.pending = pending;
    }

    /// True while any timeline, one-shot, particle or enabled ambient effect needs another frame.
    pub fn is_animating(&self) -> bool {
        if self.has_live_motion() || !self.particles.is_empty() || !self.ripples.is_empty() {
            return true;
        }
        if self
            .shakes
            .values()
            .any(|s| self.time - s.start < s.duration)
        {
            return true;
        }
        if !self.pending.is_empty() {
            return true;
        }
        self.ambient && !self.reduced_motion && !self.ambient_paused
    }

    fn has_live_motion(&self) -> bool {
        self.motions
            .values()
            .any(|m| !m.done() || (m.started && m.delay > 0.0))
    }

    fn spawn_particles(&mut self, kind: FxKind, origin: [f32; 2], count: usize, seed: u32) {
        let mut rng = Rng::new(seed as u64);
        let palette = [
            rgba8(59, 130, 246, 255),
            rgba8(56, 189, 248, 255),
            rgba8(34, 197, 94, 255),
            rgba8(245, 158, 11, 255),
            rgba8(239, 68, 68, 255),
            rgba8(168, 85, 247, 255),
            rgba8(248, 250, 252, 255),
        ];
        let confetti = kind == FxKind::Confetti;
        for i in 0..count {
            let angle = rng.range(-std::f32::consts::PI, 0.0);
            let speed = if confetti {
                rng.range(120.0, 430.0)
            } else {
                rng.range(90.0, 520.0)
            };
            let dir = [angle.cos(), angle.sin()];
            let shape = if confetti {
                if i % 3 == 0 {
                    PARTICLE_CIRCLE
                } else {
                    PARTICLE_RIBBON
                }
            } else if i % 4 == 0 {
                PARTICLE_RIBBON
            } else {
                PARTICLE_SPARK
            };
            let base = palette[(rng.next_u32() as usize) % palette.len()];
            let color = if confetti {
                base
            } else {
                lerp_rgba8(base, rgba8(255, 255, 255, 255), 0.35)
            };
            let size = if confetti {
                [rng.range(4.0, 9.0), rng.range(6.0, 13.0)]
            } else {
                [rng.range(1.6, 4.2), rng.range(1.6, 4.2)]
            };
            let life = if confetti {
                // Keep the whole burst inside the 1.2 s idle gate.
                rng.range(0.6, 1.1)
            } else {
                rng.range(0.35, 0.85)
            };
            self.particles.push(Particle {
                pos: [
                    origin[0] + rng.range(-4.0, 4.0),
                    origin[1] + rng.range(-4.0, 4.0),
                ],
                vel: [dir[0] * speed, dir[1] * speed],
                size,
                angle: rng.range(0.0, std::f32::consts::TAU),
                spin: rng.range(-8.0, 8.0),
                color,
                shape,
                life,
                max_life: life,
                drag: if confetti { 1.4 } else { 2.6 },
                gravity: if confetti { 620.0 } else { 140.0 },
            });
        }
    }

    /// Evaluate motion specs for a node and start enter/shared/one-shot timelines.
    pub fn node_fx(
        &mut self,
        node_id: &str,
        rect: ComputedRect,
        specs: &[FxSpec],
        pressed: bool,
        pointer: [f32; 2],
    ) -> NodeFx {
        let mut out = NodeFx::for_rect(rect);
        if specs.is_empty() {
            return out;
        }
        self.cur_seen.insert(node_id.to_string());
        let is_new = !self.prev_seen.contains(node_id);
        let key_hash = hash_id(node_id);

        for (index, spec) in specs.iter().enumerate() {
            if !spec.active {
                continue;
            }
            match spec.kind {
                FxKind::Enter | FxKind::Fade | FxKind::Slide | FxKind::ScalePop => {
                    let key = (node_id.to_string(), index);
                    let duration = spec.duration_secs();
                    let motion = self
                        .motions
                        .entry(key)
                        .or_insert_with(|| Motion::new(spec, duration));
                    if !motion.started || (is_new && motion.settled) {
                        motion.restart();
                    }
                    Self::apply_enter(spec, motion, &mut out);
                }
                FxKind::Shared => {
                    let Some(shared) = spec.shared_key.as_deref() else {
                        continue;
                    };
                    self.shared_now.push((shared.to_string(), rect));
                    let key = (node_id.to_string(), index);
                    let mut motion = self.motions.remove(&key).unwrap_or_else(|| {
                        let mut m = Motion::new(spec, 0.34);
                        m.from_rect = self.shared_prev.get(shared).copied();
                        m.to_rect = Some(rect);
                        m
                    });
                    if motion.to_rect != Some(rect) && motion.done() {
                        motion.from_rect = motion.to_rect.or(self.shared_prev.get(shared).copied());
                        motion.to_rect = Some(rect);
                        motion.restart();
                    }
                    let t = smoothstep(motion.progress());
                    if let (Some(from), Some(to)) = (motion.from_rect, motion.to_rect) {
                        let from_c = [from.x + from.w * 0.5, from.y + from.h * 0.5];
                        let to_c = [to.x + to.w * 0.5, to.y + to.h * 0.5];
                        let sx = (from.w.max(1.0) / to.w.max(1.0)).clamp(0.2, 5.0);
                        let sy = (from.h.max(1.0) / to.h.max(1.0)).clamp(0.2, 5.0);
                        let s = (sx + sy) * 0.5;
                        out.scale *= (1.0 - t) * s + t;
                        out.translate[0] += (1.0 - t) * (from_c[0] - to_c[0]);
                        out.translate[1] += (1.0 - t) * (from_c[1] - to_c[1]);
                    }
                    self.motions.insert(key, motion);
                }
                FxKind::Shake => {
                    if let Some(s) = self.shakes.get(node_id) {
                        let t = (self.time - s.start).max(0.0);
                        if t < s.duration {
                            let env = (-5.5 * t).exp();
                            let phase = 38.0 * t;
                            out.shake[0] += s.amplitude * env * phase.sin();
                        }
                    }
                }
                FxKind::TouchRipple => {
                    if pressed && !self.pressed.contains(node_id) {
                        self.pressed.insert(node_id.to_string());
                        if spec.kind.auto_trigger() {
                            self.ripples.push(Ripple {
                                node: key_hash,
                                start: self.time,
                                origin: [
                                    (pointer[0] - rect.x).clamp(0.0, rect.w),
                                    (pointer[1] - rect.y).clamp(0.0, rect.h),
                                ],
                                seed: spec.seed,
                            });
                        }
                    }
                }
                FxKind::Confetti | FxKind::SparkBurst => {
                    if spec.auto && is_new {
                        self.trigger(
                            Some(node_id.to_string()),
                            spec.kind,
                            spec.count,
                            None,
                            spec.seed,
                        );
                    }
                    let mut found = None;
                    for (i, b) in self.pending.iter().enumerate() {
                        if b.target.as_deref() == Some(node_id) && b.kind == spec.kind {
                            found = Some(i);
                            break;
                        }
                    }
                    if let Some(i) = found {
                        let burst = self.pending.remove(i);
                        let origin = burst
                            .origin
                            .unwrap_or([rect.x + rect.w * 0.5, rect.y + rect.h * 0.5]);
                        self.spawn_particles(burst.kind, origin, burst.count, burst.seed);
                    }
                }
                _ => {}
            }
        }

        if !pressed {
            self.pressed.remove(node_id);
        }
        out.opacity = out.opacity.clamp(0.0, 1.0);
        out.scale = out.scale.clamp(0.05, 4.0);
        out
    }

    fn apply_enter(spec: &FxSpec, motion: &Motion, out: &mut NodeFx) {
        let t = (motion.value.max(0.0)).clamp(0.0, 1.35);
        let clamped = t.min(1.0);
        let preset = spec.preset.as_deref().unwrap_or(match spec.kind {
            FxKind::Fade => "fade",
            FxKind::Slide => "slide",
            FxKind::ScalePop => "scale_pop",
            _ => "fade_slide",
        });
        let (fade, slide, scale) = match preset {
            "fade" => (true, false, false),
            "slide" | "slide_up" | "slide_down" => (false, true, false),
            "scale_pop" | "scalePop" | "scale" | "pop" => (false, false, true),
            _ => (true, true, false),
        };
        if fade {
            out.opacity *= clamped;
        }
        if slide {
            let factor = 1.0 - clamped;
            let (dx, dy) = if spec.slide_x == 0.0 && spec.slide_y == 0.0 {
                (0.0, 12.0)
            } else {
                (spec.slide_x, spec.slide_y)
            };
            out.translate[0] += dx * factor;
            out.translate[1] += dy * factor;
        }
        if scale {
            let start = if spec.scale > 0.0 { spec.scale } else { 0.92 };
            out.scale *= start + (1.0 - start) * t;
        }
    }

    /// Override a text node's content for number-roll / typewriter / scramble.
    pub fn text_override(&mut self, node_id: &str, specs: &[FxSpec], base: &str) -> Option<String> {
        let (index, spec) = specs.iter().enumerate().find(|(_, s)| {
            matches!(
                s.kind,
                FxKind::NumberRoll | FxKind::Typewriter | FxKind::TextScramble
            )
        })?;
        if !spec.active {
            return None;
        }
        let key = (node_id.to_string(), index);
        let source = hash_text(base);
        let duration = spec.duration_secs();
        let motion = self
            .motions
            .entry(key)
            .or_insert_with(|| Motion::new(spec, duration));
        if motion.source != source && motion.done() {
            motion.source = source;
            motion.restart();
        }
        motion.source = source;
        match spec.kind {
            FxKind::NumberRoll => {
                let target = spec.to.or_else(|| parse_number(base)).unwrap_or(0.0);
                let start = spec.from.unwrap_or(0.0);
                let t = ease_out_cubic(motion.progress());
                let value = start + (target - start) * t as f64;
                let body = if let Some(precision) = spec.precision {
                    format!("{:.*}", precision, value)
                } else if (value.fract()).abs() < 1e-6 {
                    format!("{:.0}", value)
                } else {
                    format!("{:.2}", value)
                };
                Some(format!("{}{}{}", spec.prefix, body, spec.suffix))
            }
            FxKind::Typewriter => {
                let t = motion.progress();
                let total = base.chars().count();
                let shown = (total as f32 * t).ceil() as usize;
                Some(base.chars().take(shown).collect())
            }
            _ => {
                let t = motion.progress();
                let total = base.chars().count();
                let shown = (total as f32 * t).ceil() as usize;
                if shown >= total {
                    return Some(base.to_string());
                }
                let tick = (self.time * 18.0) as u32;
                let mut out = String::with_capacity(base.len());
                for (i, ch) in base.chars().enumerate() {
                    if i < shown || ch == ' ' || ch == '\n' || ch == '\t' {
                        out.push(ch);
                    } else {
                        let h = hash3(spec.seed, i as u32, tick);
                        out.push(CHARSET[(h as usize) % CHARSET.len()] as char);
                    }
                }
                Some(out)
            }
        }
    }

    /// Interpolated value + formatted text for a `Stat` node.
    pub fn number_roll_value(
        &mut self,
        node_id: &str,
        specs: &[FxSpec],
        base: f64,
    ) -> Option<(f64, String)> {
        let (index, spec) = specs
            .iter()
            .enumerate()
            .find(|(_, s)| s.kind == FxKind::NumberRoll && s.active)?;
        let key = (node_id.to_string(), index);
        let duration = spec.duration_secs();
        let motion = self
            .motions
            .entry(key)
            .or_insert_with(|| Motion::new(spec, duration));
        let target = spec.to.unwrap_or(base);
        let start = spec.from.unwrap_or(base);
        let t = ease_out_cubic(motion.progress());
        let value = start + (target - start) * t as f64;
        let body = if let Some(precision) = spec.precision {
            format!("{:.*}", precision, value)
        } else {
            format!("{:.2}", value)
        };
        Some((value, format!("{}{}{}", spec.prefix, body, spec.suffix)))
    }

    /// Append effect quads for one node.
    pub fn emit_effects(
        &mut self,
        node_id: &str,
        rect: ComputedRect,
        specs: &[FxSpec],
        theme: &Theme,
        pointer: [f32; 2],
        hovered: bool,
        dl: &mut DrawList,
    ) {
        if specs.is_empty() {
            return;
        }
        let node_hash = hash_id(node_id);
        let t = self.time;
        let reduced = self.reduced_motion;

        for spec in specs {
            if !spec.active {
                continue;
            }
            let primary = spec.primary(theme);
            let secondary = spec
                .color1
                .unwrap_or_else(|| with_alpha(theme.text_primary, 60));
            let accent = spec.color2.unwrap_or(theme.surface_3);
            match spec.kind {
                FxKind::Shimmer => {
                    if !reduced {
                        self.ambient = true;
                    }
                    let speed = if reduced { 0.0 } else { spec.speed.max(0.05) };
                    let phase = if reduced {
                        0.45
                    } else {
                        (t * speed * 0.55).fract()
                    };
                    dl.push_effect(effect(
                        EFFECT_SHIMMER,
                        rect,
                        [0.0; 4],
                        [spec.radius.max(12.0), spec.intensity, speed, 0.0],
                        [phase, 0.0, 0.0, 0.0],
                        with_alpha(theme.text_primary, 110),
                        secondary,
                        primary,
                    ));
                }
                FxKind::TouchRipple => {
                    for r in self.ripples.iter().filter(|r| r.node == node_hash) {
                        let p = ((t - r.start) / 0.55).clamp(0.0, 1.0);
                        if p >= 1.0 {
                            continue;
                        }
                        let max_r = (rect.w.max(rect.h) * 0.75).max(28.0);
                        dl.push_effect(effect(
                            EFFECT_TOUCH_RIPPLE,
                            rect,
                            [0.0; 4],
                            [r.origin[0], r.origin[1], max_r, spec.radius.max(6.0)],
                            [p, r.seed as f32, 0.0, 0.0],
                            with_alpha(primary, 150),
                            secondary,
                            accent,
                        ));
                    }
                }
                FxKind::BorderBeam => {
                    if !reduced {
                        self.ambient = true;
                    }
                    let speed = if reduced { 0.0 } else { spec.speed.max(0.05) };
                    dl.push_effect(effect(
                        EFFECT_BORDER_BEAM,
                        rect,
                        [0.0; 4],
                        [speed, spec.width.max(1.0), spec.intensity, 0.0],
                        [t, 0.0, 0.0, 0.0],
                        primary,
                        secondary,
                        accent,
                    ));
                }
                FxKind::CursorGlow => {
                    if !hovered {
                        continue;
                    }
                    if !reduced {
                        self.ambient = true;
                    }
                    let local = [pointer[0] - rect.x, pointer[1] - rect.y];
                    dl.push_effect(effect(
                        EFFECT_CURSOR_GLOW,
                        rect,
                        [0.0; 4],
                        [local[0], local[1], spec.radius.max(20.0), spec.intensity],
                        [0.0, 0.0, 0.0, 0.0],
                        with_alpha(primary, 44),
                        secondary,
                        accent,
                    ));
                }
                FxKind::Aurora => {
                    if !reduced {
                        self.ambient = true;
                    }
                    let phase = if reduced {
                        0.0
                    } else {
                        t * spec.speed.max(0.05) * 0.18
                    };
                    dl.push_effect(effect(
                        EFFECT_AURORA,
                        rect,
                        [0.0; 4],
                        [
                            phase,
                            spec.intensity,
                            spec.radius.max(40.0),
                            spec.seed as f32,
                        ],
                        [t, 0.0, 0.0, 0.0],
                        with_alpha(primary, 140),
                        with_alpha(accent_color(theme, spec), 120),
                        with_alpha(theme.info, 90),
                    ));
                }
                FxKind::PulseRing => {
                    if !reduced {
                        self.ambient = true;
                    }
                    let speed = if reduced { 0.0 } else { spec.speed.max(0.05) };
                    let p = if reduced {
                        0.62
                    } else {
                        (t * speed * 0.7 + spec.seed as f32 * 0.13).fract()
                    };
                    let max_r = (rect.w.min(rect.h) * 0.5).max(spec.radius.max(20.0));
                    dl.push_effect(effect(
                        EFFECT_PULSE_RING,
                        rect,
                        [0.0; 4],
                        [p, max_r, spec.width.max(1.2), spec.intensity],
                        [rect.w * 0.5, rect.h * 0.5, 0.0, 0.0],
                        with_alpha(primary, 150),
                        secondary,
                        accent,
                    ));
                    if !reduced {
                        let p2 = (p + 0.45).fract();
                        dl.push_effect(effect(
                            EFFECT_PULSE_RING,
                            rect,
                            [0.0; 4],
                            [p2, max_r * 0.82, spec.width.max(1.0), spec.intensity * 0.55],
                            [rect.w * 0.5, rect.h * 0.5, 0.0, 0.0],
                            with_alpha(primary, 100),
                            secondary,
                            accent,
                        ));
                    }
                }
                FxKind::VoiceOrb => {
                    if !reduced {
                        self.ambient = true;
                    }
                    let amp = if reduced {
                        spec.amplitude.max(0.28)
                    } else {
                        spec.amplitude.clamp(0.0, 1.0)
                    };
                    let phase = if reduced {
                        0.0
                    } else {
                        t * spec.speed.max(0.05)
                    };
                    dl.push_effect(effect(
                        EFFECT_VOICE_ORB,
                        rect,
                        [0.0; 4],
                        [amp, phase, spec.intensity, spec.seed as f32],
                        [0.0, 0.0, 0.0, 0.0],
                        with_alpha(primary, 220),
                        with_alpha(theme.info, 190),
                        with_alpha(theme.text_primary, 210),
                    ));
                }
                _ => {}
            }
        }
    }

    /// Append a backdrop-blur partition marker before the node's own content.
    pub fn emit_backdrop(
        &mut self,
        specs: &[FxSpec],
        rect: ComputedRect,
        theme: &Theme,
        dl: &mut DrawList,
    ) {
        let Some(spec) = specs
            .iter()
            .find(|s| s.kind == FxKind::BackdropBlur && s.active)
        else {
            return;
        };
        let tint = spec
            .color0
            .unwrap_or_else(|| with_alpha(theme.surface_1, 208));
        let rim = spec
            .color1
            .unwrap_or_else(|| with_alpha(theme.text_primary, 34));
        dl.push_backdrop(BackdropRegion {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radii: [theme.radius_card; 4],
            tint,
            rim,
            params: [
                spec.blur.max(1.0),
                spec.rim.max(0.0),
                0.012,
                spec.intensity.clamp(0.0, 1.0),
            ],
            clip_id: dl.current_clip_id(),
            _pad: 0,
        });
    }

    /// Tint glyphs emitted by a `TextShimmer` node.
    pub fn apply_text_shimmer(
        &mut self,
        specs: &[FxSpec],
        dl: &mut DrawList,
        start: usize,
        rect: ComputedRect,
    ) {
        if self.reduced_motion {
            return;
        }
        let Some(spec) = specs
            .iter()
            .find(|s| s.kind == FxKind::TextShimmer && s.active)
        else {
            return;
        };
        self.ambient = true;
        let band = spec.width.max(24.0);
        let speed = spec.speed.max(0.05);
        let phase = (self.time * speed * 0.5).fract();
        let cx = rect.x - band + phase * (rect.w + band * 2.0);
        let highlight = spec.color0.unwrap_or(rgba8(255, 255, 255, 255));
        for g in dl.glyphs.iter_mut().skip(start) {
            let gx = g.pos[0] + g.size[0] * 0.5;
            let d = ((gx - cx).abs() / band).clamp(0.0, 1.0);
            let a = (1.0 - d).powi(2) * spec.intensity.clamp(0.0, 1.0);
            if a > 0.002 {
                let src_alpha = (g.color >> 24) as u8;
                g.color = lerp_rgba8(g.color, with_alpha(highlight, src_alpha), a);
            }
        }
    }

    /// Emit the particle layer (called once, after the scene content).
    pub fn emit_particles(&mut self, dl: &mut DrawList) {
        if self.particles.is_empty() {
            return;
        }
        let clip = dl.current_clip_id();
        for p in &self.particles {
            let fade = (p.life / p.max_life).clamp(0.0, 1.0);
            dl.push_particle(ParticleInstance {
                pos: p.pos,
                size: p.size,
                angle_rad: p.angle,
                color: scale_alpha(p.color, fade),
                shape_clip: p.shape | (clip << 4),
                _pad: 0,
            });
        }
    }

    /// Number of queued GPU effect instances (diagnostics / tests).
    pub fn active_effect_count(&self) -> usize {
        self.ripples.len()
    }

    pub fn active_particle_count(&self) -> usize {
        self.particles.len()
    }
}

fn effect(
    kind: u16,
    rect: ComputedRect,
    radii: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    color0: u32,
    color1: u32,
    color2: u32,
) -> EffectQuadInstance {
    EffectQuadInstance {
        rect: [rect.x, rect.y, rect.w, rect.h],
        radii,
        params0,
        params1,
        color0,
        color1,
        color2,
        kind_and_clip: kind as u32,
    }
}

fn accent_color(theme: &Theme, spec: &FxSpec) -> u32 {
    match spec.tone {
        Some(Tone::Ok) => theme.info,
        Some(Tone::Danger) => theme.warn,
        Some(Tone::Info) => theme.accent,
        _ => theme.accent,
    }
}

fn parse_number(text: &str) -> Option<f64> {
    let cleaned: String = text
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
        .collect();
    cleaned.parse().ok()
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

fn hash_id(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn hash_text(text: &str) -> u64 {
    hash_id(text)
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    let mut h = 0x9e37_79b9u32 ^ a;
    h = (h ^ b).wrapping_mul(0x85eb_ca6b);
    h = (h ^ c).wrapping_mul(0xc2b2_ae35);
    h ^= h >> 13;
    h
}

const CHARSET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#@$%&*+=<>/";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    fn rect() -> ComputedRect {
        ComputedRect {
            x: 20.0,
            y: 30.0,
            w: 200.0,
            h: 80.0,
            ..ComputedRect::default()
        }
    }

    fn tick(engine: &mut FxEngine, frame: &mut impl FnMut(&mut FxEngine), seconds: f32) -> u32 {
        let dt = 1.0 / 120.0;
        let mut steps = 0;
        while steps < (seconds * 120.0) as u32 {
            engine.step(dt);
            frame(engine);
            steps += 1;
            if !engine.is_animating() {
                break;
            }
        }
        steps
    }

    #[test]
    fn enter_motion_starts_transparent_and_settles_under_idle_budget() {
        let mut engine = FxEngine::new();
        let spec = FxSpec::new(FxKind::Enter);
        let specs = [spec];
        let mut steps = 0;
        let dt = 1.0 / 120.0;
        let last = loop {
            engine.step(dt);
            engine.begin_compile();
            let fx = engine.node_fx("card", rect(), &specs, false, [0.0, 0.0]);
            engine.end_frame();
            steps += 1;
            if !engine.is_animating() || steps > 240 {
                break fx;
            }
        };
        assert!(
            steps <= (1.2 * 120.0) as u32,
            "enter must settle within 1.2s, took {steps} frames"
        );
        assert!((last.opacity - 1.0).abs() < 1e-3, "{last:?}");
        assert!((last.scale - 1.0).abs() < 1e-3, "{last:?}");
        assert!(!engine.is_animating());
    }

    #[test]
    fn stagger_delays_later_items() {
        let mut engine = FxEngine::new();
        let a = FxSpec::new(FxKind::Fade);
        let mut b = FxSpec::new(FxKind::Fade);
        b.delay_ms = 120.0;
        let specs_a = [a];
        let specs_b = [b];
        engine.begin_compile();
        let _ = engine.node_fx("a", rect(), &specs_a, false, [0.0, 0.0]);
        let _ = engine.node_fx("b", rect(), &specs_b, false, [0.0, 0.0]);
        engine.end_frame();
        for _ in 0..8 {
            engine.step(1.0 / 120.0);
        }
        engine.begin_compile();
        let fx_a = engine.node_fx("a", rect(), &specs_a, false, [0.0, 0.0]);
        let fx_b = engine.node_fx("b", rect(), &specs_b, false, [0.0, 0.0]);
        engine.end_frame();
        assert!(fx_a.opacity > fx_b.opacity, "{fx_a:?} vs {fx_b:?}");
        assert!(fx_b.opacity < 0.01, "delayed item must not have started");
    }

    #[test]
    fn reduced_motion_freezes_ambient_but_keeps_one_shots_settling() {
        let mut engine = FxEngine::new();
        let specs = [FxSpec::new(FxKind::Shimmer)];
        let theme = Theme::dark();
        let mut dl = DrawList::new();
        engine.begin_compile();
        let _ = engine.node_fx("skel", rect(), &specs, false, [0.0, 0.0]);
        engine.emit_effects("skel", rect(), &specs, &theme, [0.0, 0.0], false, &mut dl);
        engine.end_frame();
        assert!(engine.is_animating(), "ambient shimmer requests frames");
        assert_eq!(dl.effects.len(), 1);

        engine.set_reduced_motion(true);
        dl.reset([1280.0, 800.0]);
        engine.begin_compile();
        let _ = engine.node_fx("skel", rect(), &specs, false, [0.0, 0.0]);
        engine.emit_effects("skel", rect(), &specs, &theme, [0.0, 0.0], false, &mut dl);
        engine.end_frame();
        assert!(!engine.is_animating(), "reduced motion must stop ambient");
        assert_eq!(dl.effects.len(), 1, "static refined effect remains");
        assert_eq!(dl.effects[0].params0[2], 0.0, "speed is frozen");
    }

    #[test]
    fn touch_ripple_is_a_one_shot_that_drains_to_idle() {
        let mut engine = FxEngine::new();
        let specs = [FxSpec::new(FxKind::TouchRipple)];
        engine.begin_compile();
        let _ = engine.node_fx("btn", rect(), &specs, true, [0.0, 0.0]);
        let _ = engine.node_fx("btn", rect(), &specs, false, [0.0, 0.0]);
        engine.end_frame();
        assert_eq!(engine.active_effect_count(), 1);
        assert!(engine.is_animating());
        let steps = tick(
            &mut engine,
            &mut |e| {
                e.begin_compile();
                let _ = e.node_fx("btn", rect(), &specs, false, [0.0, 0.0]);
                e.end_frame();
            },
            1.2,
        );
        assert!(steps < 120, "ripple must drain within 1s, took {steps}");
        assert!(!engine.is_animating());
        assert_eq!(engine.active_effect_count(), 0);
    }

    #[test]
    fn shake_is_bounded_and_decays() {
        let mut engine = FxEngine::new();
        let specs = [FxSpec::new(FxKind::Shake)];
        engine.shake("form");
        let mut max = 0.0f32;
        for _ in 0..60 {
            engine.step(1.0 / 120.0);
            engine.begin_compile();
            let fx = engine.node_fx("form", rect(), &specs, false, [0.0, 0.0]);
            engine.end_frame();
            max = max.max(fx.shake[0].abs());
        }
        assert!(max > 0.5 && max <= 10.1, "shake amplitude {max}");
        assert!(!engine.is_animating());
    }

    #[test]
    fn number_roll_interpolates_and_formats() {
        let mut engine = FxEngine::new();
        let mut spec = FxSpec::new(FxKind::NumberRoll);
        spec.from = Some(0.0);
        spec.to = Some(100.0);
        spec.duration_ms = 200.0;
        spec.precision = Some(1);
        let specs = [spec];
        let first = engine.text_override("kpi", &specs, "0").unwrap();
        assert_eq!(first, "0.0");
        for _ in 0..8 {
            engine.step(1.0 / 120.0);
        }
        let mid = engine
            .text_override("kpi", &specs, "0")
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!(mid > 0.0 && mid < 100.0, "{mid}");
        let steps = tick(
            &mut engine,
            &mut |e| {
                let _ = e.text_override("kpi", &specs, "0");
            },
            1.0,
        );
        assert!(steps < 60, "number roll should settle, took {steps}");
        assert_eq!(
            engine.text_override("kpi", &specs, "0").as_deref(),
            Some("100.0")
        );
    }

    #[test]
    fn typewriter_reveals_a_prefix_then_finishes() {
        let mut engine = FxEngine::new();
        let mut spec = FxSpec::new(FxKind::Typewriter);
        spec.duration_ms = 120.0;
        let specs = [spec];
        let base = "agent output";
        assert_eq!(
            engine.text_override("log", &specs, base).as_deref(),
            Some("")
        );
        engine.step(1.0 / 120.0);
        let mid = engine.text_override("log", &specs, base).unwrap();
        assert!(base.starts_with(&mid) && mid.len() < base.len(), "{mid:?}");
        for _ in 0..30 {
            engine.step(1.0 / 120.0);
        }
        assert_eq!(
            engine.text_override("log", &specs, base).as_deref(),
            Some(base)
        );
    }

    #[test]
    fn confetti_burst_emits_particles_and_drains() {
        let mut engine = FxEngine::new();
        engine.trigger(None, FxKind::Confetti, 40, Some([10.0, 10.0]), 7);
        engine.step(1.0 / 120.0);
        assert_eq!(engine.active_particle_count(), 40);
        let mut dl = DrawList::new();
        engine.emit_particles(&mut dl);
        assert_eq!(dl.particles.len(), 40);
        assert!(dl.particles.iter().all(|p| (p.shape_clip & 0xf) <= 2));
        let steps = tick(&mut engine, &mut |_| {}, 3.0);
        assert!(steps < 240, "confetti should drain, took {steps}");
        assert_eq!(engine.active_particle_count(), 0);
        assert!(!engine.is_animating());
    }

    #[test]
    fn shared_geometry_morphs_from_the_previous_rect() {
        let mut engine = FxEngine::new();
        let mut spec = FxSpec::new(FxKind::Shared);
        spec.shared_key = Some("hero".into());
        let specs = [spec];
        let old = ComputedRect {
            x: 10.0,
            y: 10.0,
            w: 80.0,
            h: 80.0,
            ..ComputedRect::default()
        };
        let new = ComputedRect {
            x: 200.0,
            y: 150.0,
            w: 300.0,
            h: 200.0,
            ..ComputedRect::default()
        };
        engine.begin_compile();
        let _ = engine.node_fx("thumb", old, &specs, false, [0.0, 0.0]);
        engine.end_frame();
        engine.step(1.0 / 120.0);
        engine.begin_compile();
        let fx = engine.node_fx("detail", new, &specs, false, [0.0, 0.0]);
        engine.end_frame();
        assert!(fx.scale < 1.0, "must start small like the old rect: {fx:?}");
        assert!(
            fx.translate[0] < 0.0,
            "must start left of the new rect: {fx:?}"
        );
    }

    #[test]
    fn transform_rewrites_only_the_marked_range_and_preserves_counts() {
        let mut dl = DrawList::new();
        dl.reset([640.0, 480.0]);
        dl.push_quad(crate::draw::QuadInstance {
            rect: [0.0, 0.0, 20.0, 20.0],
            color_top: rgba8(255, 255, 255, 255),
            color_bottom: rgba8(255, 255, 255, 255),
            ..Default::default()
        });
        let mark = DrawMark::capture(&dl);
        dl.push_quad(crate::draw::QuadInstance {
            rect: [100.0, 100.0, 20.0, 20.0],
            color_top: rgba8(255, 255, 255, 255),
            color_bottom: rgba8(255, 255, 255, 255),
            ..Default::default()
        });
        dl.push_effect(crate::draw::EffectQuadInstance {
            rect: [100.0, 100.0, 40.0, 20.0],
            radii: [4.0; 4],
            color0: rgba8(255, 255, 255, 255),
            color1: rgba8(255, 255, 255, 128),
            color2: rgba8(10, 20, 30, 255),
            ..Default::default()
        });
        dl.push_backdrop(crate::draw::BackdropRegion {
            rect: [100.0, 100.0, 40.0, 20.0],
            radii: [4.0; 4],
            tint: rgba8(0, 0, 0, 200),
            rim: rgba8(255, 255, 255, 255),
            params: [18.0, 1.0, 0.005, 1.0],
            clip_id: 0,
            _pad: 0,
        });
        let fx = NodeFx {
            opacity: 0.5,
            translate: [10.0, 5.0],
            scale: 1.0,
            scale_origin: [110.0, 110.0],
            shake: [0.0, 0.0],
        };
        eprintln!(
            "pre effects={:?} backdrops={:?}",
            dl.effects.iter().map(|e| e.rect).collect::<Vec<_>>(),
            dl.backdrops.iter().map(|b| b.rect).collect::<Vec<_>>()
        );
        mark.apply_fx(&mut dl, &fx);
        eprintln!(
            "post effects={:?} backdrops={:?}",
            dl.effects.iter().map(|e| e.rect).collect::<Vec<_>>(),
            dl.backdrops.iter().map(|b| b.rect).collect::<Vec<_>>()
        );
        assert_eq!(dl.quads[0].rect, [0.0, 0.0, 20.0, 20.0]);
        assert_eq!(dl.quads[1].rect, [110.0, 105.0, 20.0, 20.0]);
        assert_eq!(dl.quads[1].color_top >> 24, 128);
        assert_eq!(dl.effects[0].rect, [110.0, 105.0, 40.0, 20.0]);
        assert_eq!(dl.effects[0].color0 >> 24, 128);
        assert_eq!(dl.effects[0].color1 >> 24, 64);
        assert_eq!(dl.backdrops[0].rect, [110.0, 105.0, 40.0, 20.0]);
        assert_eq!(dl.backdrops[0].tint >> 24, 100);
        assert_eq!(dl.backdrops[0].rim >> 24, 128);
    }
}

#[cfg(test)]
mod audit_compat_tests {
    use super::*;
    use crate::audit::AuditRules;
    use crate::ir::{Align, Axis, Node, NodeKind, Scene, SizeSpec};
    use crate::theme::ThemeMode;
    use crate::UiEngine;

    #[test]
    fn wrapped_button_keeps_layout_hit_testing_and_audit_clean() {
        let mut button = Node::new(
            "cta",
            NodeKind::Button {
                label: "Run".into(),
                tone: Tone::Info,
            },
        );
        button.width = SizeSpec::Px(160.0);
        button.height = SizeSpec::Px(40.0);
        button.fx = vec![FxSpec::new(FxKind::Enter), FxSpec::new(FxKind::BorderBeam)];

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Start,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![button],
            },
        )
        .with_padding(crate::ir::Insets::all(16.0));
        let mut scene = Scene::new("fx", ThemeMode::Dark, root);
        scene.root.height = SizeSpec::Fill(1.0);

        let mut engine = UiEngine::new(scene, [480.0, 320.0]);
        let mut dl = DrawList::new();
        for _ in 0..40 {
            let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        }
        let rect = engine.layout.get("cta").expect("button layout rect");
        assert!(rect.w > 0.0 && rect.h > 0.0);

        let hits = engine.handle_input(crate::hit_test::InputEvent::PointerDown {
            x: rect.x + 12.0,
            y: rect.y + 12.0,
            button: 0,
        });
        let _ = hits;
        let events = engine.handle_input(crate::hit_test::InputEvent::PointerUp {
            x: rect.x + 12.0,
            y: rect.y + 12.0,
            button: 0,
        });
        assert!(events
            .iter()
            .any(|e| matches!(e, crate::hit_test::UIEvent::Click { id } if id == "cta")));

        let report = engine.audit(&dl, &AuditRules::default());
        assert!(
            report.summary.errors == 0,
            "effects must not create audit errors: {:?}",
            report.issues
        );
        assert!(dl
            .effects
            .iter()
            .any(|e| (e.kind_and_clip & 0xffff) == EFFECT_BORDER_BEAM as u32));
    }

    #[test]
    fn engine_reaches_zero_idle_within_budget_after_one_shots() {
        let mut button = Node::new(
            "cta",
            NodeKind::Button {
                label: "Run".into(),
                tone: Tone::Info,
            },
        );
        button.width = SizeSpec::Px(160.0);
        button.height = SizeSpec::Px(40.0);
        button.fx = vec![FxSpec::new(FxKind::Enter), FxSpec::new(FxKind::TouchRipple)];

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Start,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![button],
            },
        )
        .with_padding(crate::ir::Insets::all(16.0));
        let mut scene = Scene::new("fx-idle", ThemeMode::Dark, root);
        scene.root.height = SizeSpec::Fill(1.0);

        let mut engine = UiEngine::new(scene, [480.0, 320.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let rect = engine.layout.get("cta").expect("button rect");
        let _ = engine.handle_input(crate::hit_test::InputEvent::PointerDown {
            x: rect.x + 12.0,
            y: rect.y + 12.0,
            button: 0,
        });
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        let _ = engine.handle_input(crate::hit_test::InputEvent::PointerUp {
            x: rect.x + 12.0,
            y: rect.y + 12.0,
            button: 0,
        });

        let mut frames = 0;
        loop {
            let status = engine.tick_and_compile(1.0 / 120.0, &mut dl);
            frames += 1;
            if !status.animating {
                break;
            }
            assert!(
                frames <= 144,
                "one-shots must reach 0% idle within 1.2s, still animating after {frames} frames"
            );
        }
        assert!(!engine.fx.is_animating());
        assert_eq!(engine.fx.active_effect_count(), 0);

        // The next tick must do no work at all (0% CPU/GPU).
        let idle = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(!idle.compiled);
        assert!(!idle.animating);
    }

    #[test]
    fn triggered_burst_lands_on_a_target_without_fx_specs() {
        let mut button = Node::new(
            "plain",
            NodeKind::Button {
                label: "Ship".into(),
                tone: Tone::Info,
            },
        );
        button.width = SizeSpec::Px(120.0);
        button.height = SizeSpec::Px(36.0);

        let root = Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Start,
                gap: 8.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![button],
            },
        )
        .with_padding(crate::ir::Insets::all(16.0));
        let mut scene = Scene::new("fx-target", ThemeMode::Dark, root);
        scene.root.height = SizeSpec::Fill(1.0);

        let mut engine = UiEngine::new(scene, [480.0, 320.0]);
        let mut dl = DrawList::new();
        let _ = engine.tick_and_compile(1.0 / 120.0, &mut dl);

        engine.apply_patch(crate::ir::ScenePatch::FxTrigger {
            target: Some("plain".into()),
            kind: FxKind::Confetti,
            count: Some(24),
            x: None,
            y: None,
            seed: Some(11),
        });
        let status = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        assert!(status.animating, "targeted confetti must start a timeline");
        assert!(
            engine.fx.active_particle_count() > 0,
            "targeted burst must spawn particles without a spec on the target"
        );

        let mut frames = 0;
        loop {
            let status = engine.tick_and_compile(1.0 / 120.0, &mut dl);
            frames += 1;
            if !status.animating {
                break;
            }
            assert!(
                frames <= 144,
                "burst must settle within 1.2s, still animating after {frames} frames"
            );
        }
        assert_eq!(engine.fx.active_particle_count(), 0);
    }
}

#[cfg(test)]
mod authoring_tests {
    use super::*;
    use crate::ui;

    #[test]
    fn helpers_attach_stagger_and_preserve_node_identity() {
        let mut first: Node = ui::text("first", "one").into();
        let mut second: Node = ui::text("second", "two").into();
        stagger([&mut first, &mut second], 40.0, 10.0);
        assert_eq!(first.fx.len(), 1);
        assert_eq!(first.fx[0].delay_ms, 0.0);
        assert_eq!(second.fx[0].delay_ms, 40.0);
        assert_eq!(second.fx[0].slide_y, 10.0);
        assert_eq!(first.id, "first");
        assert_eq!(second.id, "second");

        let card: Node = enter(ui::text("card", "three").into(), "scale_pop");
        assert_eq!(card.fx[0].preset.as_deref(), Some("scale_pop"));

        let glass: Node = backdrop_blur(ui::text("glass", "four").into(), 24.0, 1.5);
        assert_eq!(glass.fx[0].kind, FxKind::BackdropBlur);
        assert_eq!(glass.fx[0].blur, 24.0);
        assert_eq!(glass.fx[0].rim, 1.5);

        let doubled: Node = attach(card, [FxSpec::new(FxKind::Shimmer)]);
        assert_eq!(doubled.fx.len(), 2);
    }
}
