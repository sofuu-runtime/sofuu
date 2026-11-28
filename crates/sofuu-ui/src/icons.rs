//! Built-in vector icon system: **150+** essential UI icons baked to one SDF atlas.
//!
//! Icons are authored as compact 24x24 path strings and converted to a
//! single-channel signed distance field once per process (`OnceLock`). The
//! atlas is uploaded alongside the glyph atlas and sampled through the sprite
//! pipeline (`SPRITE_FLAG_SDF`), so a 16 px icon on a 1x display and the same
//! icon at 3x on a phone both resolve crisply — no icon font, no PNG sprites,
//! no per-DPI re-rasterization.
//!
//! All geometry here is generic UI vocabulary (arrows, media, files, devices);
//! nothing depends on an application domain.
//!
//! # Path DSL
//!
//! ```text
//! M x y        move to
//! L x y        line to
//! H x / V y    horizontal / vertical line
//! O cx cy r    stroked circle
//! R x y w h    stroked rectangle
//! C x1 y1 x2 y2 x y   cubic Bézier (flattened into segments)
//! A cx cy r a0 a1     stroked arc, angles in degrees (0 = +x, 90 = +y)
//! P x y ...    filled polygon (even-odd), closed implicitly
//! Z            close the current sub-path
//! ```
//!
//! Coordinates live in a 24x24 box, so paths read like the icon grids most
//! designers already use.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Cell size (atlas px) for each icon. 48 px keeps the field crisp at 3x.
const CELL: u32 = 48;
/// Padding inside each cell so wide strokes and soft corners never clip.
const PAD: f32 = 4.0;
/// Distance range encoded into the `R8` channel.
pub const ICON_PX_RANGE: f32 = 3.0;
/// Columns in the atlas grid.
const COLS: u32 = 16;
/// Design-grid extent (icons are authored in a 24x24 box).
const DESIGN: f32 = 24.0;
/// Stroke width in design units.
const STROKE: f32 = 2.0;

/// One icon's UV rectangle in the atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IconEntry {
    pub name: &'static str,
    /// Normalized `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
}

/// CPU-owned icon atlas ready to upload once as an `R8Unorm` texture.
#[derive(Clone, Debug)]
pub struct IconAtlas {
    pub width: u32,
    pub height: u32,
    /// Row-major `R8` SDF bytes (`128` = contour edge, `>128` = inside).
    pub pixels: Vec<u8>,
    pub px_range: f32,
    entries: Vec<IconEntry>,
    index: HashMap<&'static str, usize>,
}

impl IconAtlas {
    /// Process-wide built-in atlas (built once, shared cheaply via `Arc`).
    pub fn builtin() -> Arc<IconAtlas> {
        static ATLAS: OnceLock<Arc<IconAtlas>> = OnceLock::new();
        ATLAS
            .get_or_init(|| Arc::new(IconAtlas::from_paths(BUILTIN).expect("built-in icons must parse")))
            .clone()
    }

    /// Build an atlas from `(name, path)` pairs.
    pub fn from_paths(defs: &[(&'static str, &'static str)]) -> Result<Self, String> {
        if defs.is_empty() {
            return Err("icon atlas needs at least one icon".into());
        }
        let rows = (defs.len() as u32).div_ceil(COLS);
        let width = COLS * CELL;
        let height = rows * CELL;
        let mut pixels = vec![0u8; (width * height) as usize];
        let mut entries = Vec::with_capacity(defs.len());
        let mut index = HashMap::with_capacity(defs.len());

        for (i, &(name, path)) in defs.iter().enumerate() {
            if index.contains_key(name) {
                return Err(format!("duplicate icon name `{name}`"));
            }
            let cmds = parse_path(path).map_err(|e| format!("icon `{name}`: {e}"))?;
            let cx0 = (i as u32 % COLS) * CELL;
            let cy0 = (i as u32 / COLS) * CELL;
            rasterize_icon(&cmds, &mut pixels, width, cx0, cy0);
            index.insert(name, i);
            entries.push(IconEntry {
                name,
                uv: [
                    cx0 as f32 / width as f32,
                    cy0 as f32 / height as f32,
                    (cx0 + CELL) as f32 / width as f32,
                    (cy0 + CELL) as f32 / height as f32,
                ],
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
            px_range: ICON_PX_RANGE,
            entries,
            index,
        })
    }

    /// Return a new atlas containing the built-ins plus custom SVG-subset paths.
    ///
    /// Custom paths are rasterized lazily (only when this is called), so the
    /// shipped atlas never pays for icons an app does not use.
    pub fn with_custom(base: &IconAtlas, extra: &[(&'static str, &'static str)]) -> Result<Self, String> {
        if extra.is_empty() {
            return Ok(base.clone());
        }
        let mut defs: Vec<(&'static str, &'static str)> = base.entries.iter().map(|e| {
            let path = BUILTIN
                .iter()
                .find(|(n, _)| *n == e.name)
                .map(|(_, p)| *p)
                .unwrap_or("");
            (e.name, path)
        }).collect();
        defs.extend_from_slice(extra);
        Self::from_paths(&defs)
    }

    /// UV rectangle for `name`, if the icon is present.
    #[inline]
    pub fn uv(&self, name: &str) -> Option<[f32; 4]> {
        self.index.get(name).map(|&i| self.entries[i].uv)
    }

    #[inline]
    pub fn entry(&self, name: &str) -> Option<&IconEntry> {
        self.index.get(name).map(|&i| &self.entries[i])
    }

    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Icon count in this atlas.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All icon names in atlas order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|e| e.name)
    }
}

/// Parsed path command.
#[derive(Clone, Debug)]
enum Cmd {
    Move(f32, f32),
    Line(f32, f32),
    Circle(f32, f32, f32),
    Rect(f32, f32, f32, f32),
    Arc(f32, f32, f32, f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Poly(Vec<(f32, f32)>),
    Close,
}

fn next_num(tokens: &mut std::iter::Peekable<std::str::SplitWhitespace<'_>>) -> Result<f32, String> {
    tokens
        .next()
        .ok_or_else(|| "path ended mid-command".to_string())?
        .parse::<f32>()
        .map_err(|_| "expected a number".to_string())
}

fn parse_path(src: &str) -> Result<Vec<Cmd>, String> {
    let mut cmds = Vec::new();
    let mut tokens = src.split_whitespace().peekable();
    while let Some(tok) = tokens.next() {
        match tok {
            "M" => cmds.push(Cmd::Move(next_num(&mut tokens)?, next_num(&mut tokens)?)),
            "L" => cmds.push(Cmd::Line(next_num(&mut tokens)?, next_num(&mut tokens)?)),
            "H" => {
                let x = next_num(&mut tokens)?;
                cmds.push(Cmd::Line(x, f32::NAN)); // resolved against the running pen
            }
            "V" => {
                let y = next_num(&mut tokens)?;
                cmds.push(Cmd::Line(f32::NAN, y));
            }
            "C" => cmds.push(Cmd::Cubic(
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
            )),
            "O" => cmds.push(Cmd::Circle(
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
            )),
            "R" => cmds.push(Cmd::Rect(
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
            )),
            "A" => cmds.push(Cmd::Arc(
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
                next_num(&mut tokens)?,
            )),
            "P" => {
                let mut pts = Vec::new();
                while let Some(t) = tokens.peek() {
                    if t.parse::<f32>().is_err() {
                        break;
                    }
                    pts.push((next_num(&mut tokens)?, next_num(&mut tokens)?));
                }
                if pts.len() < 3 {
                    return Err("P needs at least 3 x y pairs".into());
                }
                cmds.push(Cmd::Poly(pts));
            }
            "Z" => cmds.push(Cmd::Close),
            other => return Err(format!("unknown command `{other}`")),
        }
    }
    // Resolve H/V against the running pen position in a second pass.
    let mut out = Vec::with_capacity(cmds.len());
    let mut pen = (0.0f32, 0.0f32);
    let mut start = pen;
    for c in cmds {
        let c = match c {
            Cmd::Move(x, y) => {
                pen = (x, y);
                start = pen;
                Cmd::Move(x, y)
            }
            Cmd::Line(x, y) => {
                let p = (
                    if x.is_nan() { pen.0 } else { x },
                    if y.is_nan() { pen.1 } else { y },
                );
                pen = p;
                Cmd::Line(p.0, p.1)
            }
            Cmd::Close => {
                pen = start;
                Cmd::Close
            }
            other => other,
        };
        out.push(c);
    }
    Ok(out)
}

/// Distance from `p` to segment `a-b`.
#[inline]
fn seg_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let ap = (p.0 - a.0, p.1 - a.1);
    let len2 = ab.0 * ab.0 + ab.1 * ab.1;
    let t = if len2 <= 1e-9 {
        0.0
    } else {
        ((ap.0 * ab.0 + ap.1 * ab.1) / len2).clamp(0.0, 1.0)
    };
    let dx = ap.0 - ab.0 * t;
    let dy = ap.1 - ab.1 * t;
    (dx * dx + dy * dy).sqrt()
}

/// Signed distance (positive inside) to a point-in-polygon fill.
#[inline]
fn poly_sdf(p: (f32, f32), pts: &[(f32, f32)]) -> f32 {
    let mut inside = false;
    let mut best = f32::MAX;
    let n = pts.len();
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        best = best.min(seg_dist(p, a, b));
        if (a.1 > p.1) != (b.1 > p.1) {
            let x = (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0;
            if p.0 < x {
                inside = !inside;
            }
        }
    }
    if inside {
        best
    } else {
        -best
    }
}

fn rasterize_icon(cmds: &[Cmd], pixels: &mut [u8], atlas_w: u32, cx0: u32, cy0: u32) {
    // Pre-expand circles/rects/arcs into segment lists once per icon.
    let mut segments: Vec<((f32, f32), (f32, f32))> = Vec::new();
    let mut polys: Vec<&[(f32, f32)]> = Vec::new();
    let mut pen = (0.0f32, 0.0f32);
    let mut start = pen;
    for cmd in cmds {
        match *cmd {
            Cmd::Move(x, y) => {
                pen = (x, y);
                start = pen;
            }
            Cmd::Line(x, y) => {
                segments.push((pen, (x, y)));
                pen = (x, y);
            }
            Cmd::Circle(cx, cy, r) => {
                let n = ((r * 6.0).ceil() as usize).max(16);
                for i in 0..n {
                    let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
                    let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
                    segments.push((
                        (cx + r * a0.cos(), cy + r * a0.sin()),
                        (cx + r * a1.cos(), cy + r * a1.sin()),
                    ));
                }
            }
            Cmd::Rect(x, y, w, h) => {
                let pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
                for i in 0..4 {
                    segments.push((pts[i], pts[(i + 1) % 4]));
                }
            }
            Cmd::Arc(cx, cy, r, a0, a1) => {
                let span = (a1 - a0).abs();
                let n = ((span / 12.0).ceil() as usize).max(4);
                let (s, e) = (a0.to_radians(), a1.to_radians());
                for i in 0..n {
                    let t0 = s + (e - s) * (i as f32 / n as f32);
                    let t1 = s + (e - s) * ((i + 1) as f32 / n as f32);
                    segments.push((
                        (cx + r * t0.cos(), cy + r * t0.sin()),
                        (cx + r * t1.cos(), cy + r * t1.sin()),
                    ));
                }
            }
            Cmd::Cubic(x1, y1, x2, y2, x3, y3) => {
                let mut prev = pen;
                for i in 1..=12 {
                    let t = i as f32 / 12.0;
                    let mt = 1.0 - t;
                    let x = mt * mt * mt * pen.0
                        + 3.0 * mt * mt * t * x1
                        + 3.0 * mt * t * t * x2
                        + t * t * t * x3;
                    let y = mt * mt * mt * pen.1
                        + 3.0 * mt * mt * t * y1
                        + 3.0 * mt * t * t * y2
                        + t * t * t * y3;
                    segments.push((prev, (x, y)));
                    prev = (x, y);
                }
                pen = (x3, y3);
            }
            Cmd::Poly(ref pts) => polys.push(pts),
            Cmd::Close => {
                let (dx, dy) = (pen.0 - start.0, pen.1 - start.1);
                if dx * dx + dy * dy > 1e-6 {
                    segments.push((pen, start));
                }
                pen = start;
            }
        }
    }

    let scale = (CELL as f32 - PAD * 2.0) / DESIGN;
    let half = STROKE * 0.5;
    let row = &mut pixels[(cy0 * atlas_w) as usize..((cy0 + CELL) * atlas_w) as usize];
    for py in 0..CELL {
        for px in 0..CELL {
            let p = (
                (px as f32 + 0.5 - PAD) / scale,
                (py as f32 + 0.5 - PAD) / scale,
            );
            let mut sd = -f32::MAX;
            for (a, b) in &segments {
                // Signed distance to the stroke region (positive inside):
                // the nearest segment wins, so take the max over segments.
                sd = sd.max(half - seg_dist(p, *a, *b));
            }
            for poly in &polys {
                sd = sd.max(poly_sdf(p, poly));
            }
            // d is in design units; scale to atlas px so px_range stays in atlas px.
            let d_atlas = sd * scale;
            let v = 128.0 + 127.0 * (d_atlas / ICON_PX_RANGE).clamp(-1.0, 1.0);
            row[(py * atlas_w + px + cx0) as usize] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// The built-in icon vocabulary (name, 24x24 path).
///
/// Kept alphabetical by group: navigation, actions, status, media, files,
/// devices, people, commerce, editing.
pub const BUILTIN: &[(&str, &str)] = &[
    // ── Navigation & arrows ─────────────────────────────────────────────
    ("arrow-left", "M 15 5 L 8 12 L 15 19"),
    ("arrow-right", "M 9 5 L 16 12 L 9 19"),
    ("arrow-up", "M 5 15 L 12 8 L 19 15"),
    ("arrow-down", "M 5 9 L 12 16 L 19 9"),
    ("arrow-up-left", "M 8 8 L 8 16 M 8 8 L 16 8"),
    ("arrow-up-right", "M 16 8 L 8 8 M 16 8 L 16 16"),
    ("arrow-down-left", "M 8 16 L 16 16 M 8 16 L 8 8"),
    ("arrow-down-right", "M 16 16 L 8 16 M 16 16 L 16 8"),
    ("chevron-left", "M 14 6 L 8 12 L 14 18"),
    ("chevron-right", "M 10 6 L 16 12 L 10 18"),
    ("chevron-up", "M 6 14 L 12 8 L 18 14"),
    ("chevron-down", "M 6 10 L 12 16 L 18 10"),
    ("chevrons-left", "M 12 6 L 6 12 L 12 18 M 19 6 L 13 12 L 19 18"),
    ("chevrons-right", "M 5 6 L 11 12 L 5 18 M 12 6 L 18 12 L 12 18"),
    ("chevrons-up", "M 6 12 L 12 6 L 18 12 M 6 19 L 12 13 L 18 19"),
    ("chevrons-down", "M 6 5 L 12 11 L 18 5 M 6 12 L 12 18 L 18 12"),
    ("menu", "M 4 7 H 20 M 4 12 H 20 M 4 17 H 20"),
    ("home", "M 4 11 L 12 4 L 20 11 M 6 10 V 20 H 18 V 10"),
    ("external-link", "M 14 4 H 20 V 10 M 20 4 L 11 13 M 17 14 V 20 H 4 V 7 H 10"),
    ("corner-down-left", "M 20 5 V 13 H 6 M 10 9 L 6 13 L 10 17"),
    ("corner-up-right", "M 4 19 V 11 H 18 M 14 7 L 18 11 L 14 15"),
    ("compass", "O 12 12 9 P 15 9 13 13 9 15 11 11"),
    ("navigation", "P 12 3 19 20 12 16 5 20"),
    ("map-pin", "O 12 9 3 M 12 21 L 9 12 M 12 21 L 15 12"),
    ("map", "M 3 6 L 9 4 L 15 6 L 21 4 V 18 L 15 20 L 9 18 L 3 20 Z M 9 4 V 18 M 15 6 V 20"),
    // ── Actions ─────────────────────────────────────────────────────────
    ("search", "O 11 11 7 M 16 16 L 21 21"),
    ("close", "M 6 6 L 18 18 M 18 6 L 6 18"),
    ("check", "M 5 12 L 10 17 L 19 7"),
    ("plus", "M 12 5 V 19 M 5 12 H 19"),
    ("minus", "M 5 12 H 19"),
    ("settings", "O 12 12 3.2 O 12 12 6.5 M 12 2 V 4 M 12 20 V 22 M 2 12 H 4 M 20 12 H 22 M 5 5 L 6.5 6.5 M 17.5 17.5 L 19 19 M 19 5 L 17.5 6.5 M 6.5 17.5 L 5 19"),
    ("sliders", "M 4 7 H 20 M 4 12 H 20 M 4 17 H 20 M 8 4 V 10 M 16 9 V 15 M 10 14 V 20"),
    ("refresh", "A 12 12 8 30 330 M 18 4 V 9 H 13"),
    ("rotate", "A 12 12 8 30 330 M 18 4 V 9 H 13"),
    ("undo", "M 5 10 H 14 M 9 6 L 5 10 L 9 14 M 14 10 V 20 H 8"),
    ("redo", "M 19 10 H 10 M 15 6 L 19 10 L 15 14 M 10 10 V 20 H 16"),
    ("filter", "P 3 5 21 5 14 13 14 20 10 18 10 13"),
    ("sort", "M 7 4 V 20 M 3 16 L 7 20 L 11 16 M 17 20 V 4 M 13 8 L 17 4 L 21 8"),
    ("download", "M 12 4 V 15 M 7 10 L 12 15 L 17 10 M 4 19 H 20"),
    ("upload", "M 12 15 V 4 M 7 9 L 12 4 L 17 9 M 4 19 H 20"),
    ("save", "M 4 4 H 16 L 20 8 V 20 H 4 Z M 8 4 V 9 H 16 V 4 M 8 20 V 14 H 16 V 20"),
    ("print", "M 6 9 V 4 H 18 V 9 M 4 9 H 20 V 17 H 17 M 7 17 H 4 V 9 M 7 14 H 17 V 21 H 7 Z"),
    ("share", "O 18 6 2 O 18 18 2 O 6 12 2 M 16 7 L 8 11 M 16 17 L 8 13"),
    ("copy", "R 8 8 12 12 M 4 16 V 4 H 16"),
    ("trash", "M 4 7 H 20 M 9 4 H 15 M 6 7 L 7 21 H 17 L 18 7 M 10 11 V 17 M 14 11 V 17"),
    ("edit", "M 4 20 L 5 15 L 16 4 L 20 8 L 9 19 Z M 14 6 L 18 10"),
    ("cut", "O 6 6 2 O 18 6 2 M 7.5 8 L 18 20 M 16.5 8 L 6 20"),
    ("crop", "M 6 2 V 18 H 22 M 2 6 H 18 V 22"),
    ("link", "O 9 12 3 O 15 12 3 M 10 12 H 14"),
    ("zoom-in", "O 11 11 7 M 16 16 L 21 21 M 8 11 H 14 M 11 8 V 14"),
    ("zoom-out", "O 11 11 7 M 16 16 L 21 21 M 8 11 H 14"),
    ("maximize", "M 4 9 V 4 H 9 M 15 4 H 20 V 9 M 20 15 V 20 H 15 M 9 20 H 4 V 15"),
    ("minimize", "M 9 4 V 9 H 4 M 15 9 V 4 H 20 M 20 15 H 15 V 20 M 4 15 H 9 V 20"),
    ("expand", "M 4 4 L 10 10 M 10 4 H 4 V 10 M 20 20 L 14 14 M 14 20 H 20 V 14"),
    ("move", "M 12 3 V 21 M 3 12 H 21 M 9 6 L 12 3 L 15 6 M 9 18 L 12 21 L 15 18 M 6 9 L 3 12 L 6 15 M 18 9 L 21 12 L 18 15"),
    ("more-horizontal", "O 5 12 1.6 O 12 12 1.6 O 19 12 1.6"),
    ("more-vertical", "O 12 5 1.6 O 12 12 1.6 O 12 19 1.6"),
    ("grip-vertical", "O 8 6 1.4 O 16 6 1.4 O 8 12 1.4 O 16 12 1.4 O 8 18 1.4 O 16 18 1.4"),
    ("drag", "O 8 6 1.4 O 16 6 1.4 O 8 12 1.4 O 16 12 1.4 O 8 18 1.4 O 16 18 1.4 M 12 3 V 21"),
    // ── Status & feedback ───────────────────────────────────────────────
    ("info", "O 12 12 9 M 12 11 V 17 M 12 7.5 V 8"),
    ("warning", "M 12 3 L 22 20 H 2 Z M 12 9 V 14 M 12 17 V 17.4"),
    ("error", "O 12 12 9 M 9 9 L 15 15 M 15 9 L 9 15"),
    ("check-circle", "O 12 12 9 M 8 12 L 11 15 L 16 9"),
    ("clock", "O 12 12 9 M 12 7 V 12 L 15 14"),
    ("bell", "M 6 17 H 18 V 16 C 16 14 16 13 16 11 A 12 11 4 180 360 C 8 13 8 14 6 16 Z M 10 20 A 12 20 2 0 180"),
    ("bell-off", "M 6 17 H 18 V 16 C 16 14 16 13 16 11 M 8 11 A 12 11 4 180 360 C 8 13 8 14 6 16 M 10 20 A 12 20 2 0 180 M 4 4 L 20 20"),
    ("lock", "R 5 11 14 10 M 8 11 V 7 A 12 7 4 180 360 V 11"),
    ("unlock", "R 5 11 14 10 M 8 11 V 7 A 12 7 4 180 270"),
    ("shield", "M 12 3 L 20 6 V 12 C 20 17 16 20 12 21 C 8 20 4 17 4 12 V 6 Z"),
    ("shield-check", "M 12 3 L 20 6 V 12 C 20 17 16 20 12 21 C 8 20 4 17 4 12 V 6 Z M 8 12 L 11 15 L 16 9"),
    ("eye", "M 2 12 C 6 6 18 6 22 12 C 18 18 6 18 2 12 Z O 12 12 3"),
    ("eye-off", "M 2 12 C 6 6 18 6 22 12 C 18 18 6 18 2 12 Z M 4 4 L 20 20"),
    ("heart", "M 12 20 C 4 16 2 11 4 7 C 6 3 10 3 12 7 C 14 3 18 3 20 7 C 22 11 20 16 12 20 Z"),
    ("star", "P 12 3 14.7 9.2 21 9.8 16.2 14 17.6 20.4 12 17 6.4 20.4 7.8 14 3 9.8 9.3 9.2"),
    ("bookmark", "M 6 3 H 18 V 21 L 12 16 L 6 21 Z"),
    ("sparkles", "P 10 3 11.6 7.4 16 9 11.6 10.6 10 15 8.4 10.6 4 9 8.4 7.4 P 18 13 18.9 15.1 21 16 18.9 16.9 18 19 17.1 16.9 15 16 17.1 15.1 P 6 16 6.7 17.6 8 18.3 6.7 19 6 20.6 5.3 19 4 18.3 5.3 17.6"),
    ("zap", "P 13 2 4 14 11 14 10 22 19 10 12 10"),
    ("flame", "M 12 2 C 16 7 19 10 19 14 C 19 18 16 21 12 21 C 8 21 5 18 5 14 C 5 11 7 9 9 6 C 9 9 10 10 11 11 C 11 8 11 5 12 2 Z"),
    ("droplet", "M 12 3 C 8 8 5 11 5 15 C 5 19 8 21 12 21 C 16 21 19 19 19 15 C 19 11 16 8 12 3 Z"),
    ("sun", "O 12 12 4 M 12 2 V 4 M 12 20 V 22 M 2 12 H 4 M 20 12 H 22 M 5 5 L 6.5 6.5 M 17.5 17.5 L 19 19 M 19 5 L 17.5 6.5 M 6.5 17.5 L 5 19"),
    ("moon", "M 20 14 C 18 19 12 21 8 17 C 4 13 6 7 11 5 C 8 10 13 16 20 14 Z"),
    ("cloud", "M 7 18 C 4 18 3 16 3 14 C 3 12 5 11 7 11 C 7 7 10 5 13 6 C 16 4 20 6 20 10 C 22 11 22 14 20 15 C 20 17 18 18 17 18 Z"),
    ("cloud-rain", "M 7 14 C 4 14 3 12 3 10 C 3 8 5 7 7 7 C 7 4 10 3 13 4 C 16 2 20 4 20 8 C 22 9 22 12 20 13 C 20 14 18 14 17 14 Z M 8 17 L 7 21 M 13 17 L 12 21 M 18 17 L 17 21"),
    ("wifi", "A 12 20 9 200 340 M 12 20 A 12 20 5.5 205 335 O 12 20 1.6"),
    ("bluetooth", "M 12 3 L 18 8 L 6 16 M 12 21 L 18 16 L 6 8 M 12 3 V 21"),
    ("battery", "R 3 8 16 8 M 21 10 V 14 M 6 11 H 12"),
    ("signal", "M 4 20 V 16 M 9 20 V 12 M 14 20 V 8 M 19 20 V 4"),
    ("thermometer", "O 12 17 4 R 10 5 4 9"),
    ("wind", "M 3 8 H 16 A 19 8 3 180 360 M 3 12 H 20 M 3 16 H 14 A 17 16 3 180 360"),
    // ── Media ───────────────────────────────────────────────────────────
    ("play", "P 7 4 20 12 7 20"),
    ("pause", "M 8 4 V 20 M 16 4 V 20"),
    ("stop", "R 6 6 12 12"),
    ("skip-back", "P 18 4 8 12 18 20 Z M 6 4 V 20"),
    ("skip-forward", "P 6 4 16 12 6 20 Z M 18 4 V 20"),
    ("volume", "P 4 9 9 9 14 5 14 19 9 15 4 15 Z A 17 12 5 310 50 A 17 12 9 310 50"),
    ("volume-x", "P 4 9 9 9 14 5 14 19 9 15 4 15 Z M 17 9 L 22 14 M 22 9 L 17 14"),
    ("mic", "R 9 3 6 11 A 12 14 3 0 180 M 12 17 V 20 M 8 20 H 16"),
    ("mic-off", "R 9 3 6 11 M 12 17 V 20 M 8 20 H 16 M 4 4 L 20 20"),
    ("headphones", "M 4 14 A 12 14 8 180 360 R 4 14 4 7 R 16 14 4 7"),
    ("music", "O 7 17 2.5 O 17 15 2.5 M 9.5 17 V 5 L 19.5 7 V 15"),
    ("speaker", "R 5 4 14 16 O 12 12 3 O 12 18 1"),
    ("camera", "R 3 7 18 13 M 8 7 L 10 4 H 14 L 16 7 O 12 13 4"),
    ("video", "R 3 6 13 12 M 16 10 L 21 7 V 17 L 16 14"),
    ("image", "R 3 4 18 16 M 3 17 L 9 11 L 14 16 L 17 13 L 21 17 O 16 8 1.6"),
    ("film", "R 3 4 18 16 M 7 4 V 20 M 17 4 V 20 M 3 8 H 7 M 3 12 H 7 M 3 16 H 7 M 17 8 H 21 M 17 12 H 21 M 17 16 H 21"),
    // ── Files & data ────────────────────────────────────────────────────
    ("file", "M 6 3 H 14 L 19 8 V 21 H 6 Z M 14 3 V 8 H 19"),
    ("file-text", "M 6 3 H 14 L 19 8 V 21 H 6 Z M 14 3 V 8 H 19 M 9 12 H 16 M 9 16 H 16"),
    ("folder", "M 3 6 H 9 L 11 9 H 21 V 19 H 3 Z"),
    ("folder-open", "M 3 6 H 9 L 11 9 H 21 L 19 19 H 3 Z M 3 9 V 19"),
    ("archive", "R 3 4 18 5 M 5 9 H 19 V 21 H 5 Z M 9 13 H 15"),
    ("database", "A 12 6 8 0 360 M 4 6 V 18 A 12 18 8 0 180 M 20 6 V 18 A 12 18 8 180 360 M 4 12 A 12 12 8 0 180"),
    ("server", "R 3 3 18 7 R 3 14 18 7 M 7 6.5 H 7.1 M 7 17.5 H 7.1 M 12 6.5 H 16 M 12 17.5 H 16"),
    ("hard-drive", "R 3 13 18 8 M 6 13 L 9 5 H 15 L 18 13 M 7 17 H 7.1 M 12 17 H 17"),
    ("terminal", "R 3 4 18 16 M 7 9 L 10 12 L 7 15 M 12 15 H 17"),
    ("code", "M 9 7 L 3 12 L 9 17 M 15 7 L 21 12 L 15 17"),
    ("braces", "M 9 3 C 6 3 6 6 6 9 C 6 11 4 12 3 12 C 4 12 6 13 6 15 C 6 18 6 21 9 21 M 15 3 C 18 3 18 6 18 9 C 18 11 20 12 21 12 C 20 12 18 13 18 15 C 18 18 18 21 15 21"),
    ("hash", "M 5 9 H 19 M 5 15 H 19 M 10 3 L 8 21 M 16 3 L 14 21"),
    ("git-branch", "O 7 6 2 O 7 18 2 O 17 8 2 M 7 8 V 16 M 7 12 C 12 12 13 11 13 9 V 8"),
    ("git-merge", "O 7 6 2 O 7 18 2 O 17 10 2 M 7 8 V 16 M 9 12 C 12 12 14 11 15 10"),
    ("table", "R 3 4 18 16 M 3 9 H 21 M 3 14 H 21 M 9 4 V 20"),
    ("list", "M 8 6 H 21 M 8 12 H 21 M 8 18 H 21 O 4 6 0.6 O 4 12 0.6 O 4 18 0.6"),
    ("grid", "R 3 3 7 7 R 14 3 7 7 R 3 14 7 7 R 14 14 7 7"),
    ("layout", "R 3 3 18 18 M 3 9 H 21 M 9 9 V 21"),
    ("layers", "P 12 3 21 8 12 13 3 8 M 3 12 L 12 17 L 21 12 M 3 16 L 12 21 L 21 16"),
    ("package", "M 12 3 L 21 7 V 17 L 12 21 L 3 17 V 7 Z M 3 7 L 12 11 L 21 7 M 12 11 V 21"),
    ("box", "M 12 3 L 21 7 V 17 L 12 21 L 3 17 V 7 Z M 3 7 L 12 11 L 21 7 M 12 11 V 21 M 7.5 5 L 16.5 9"),
    // ── Devices ─────────────────────────────────────────────────────────
    ("monitor", "R 3 4 18 12 M 8 20 H 16 M 12 16 V 20"),
    ("smartphone", "R 7 2 10 20 M 10 19 H 14"),
    ("tablet", "R 5 2 14 20 M 10 19 H 14"),
    ("watch", "R 8 7 8 10 M 12 3 V 7 M 12 17 V 21 M 12 9 V 12 H 14"),
    ("cpu", "R 6 6 12 12 M 10 2 V 6 M 14 2 V 6 M 10 18 V 22 M 14 18 V 22 M 2 10 H 6 M 2 14 H 6 M 18 10 H 22 M 18 14 H 22 R 9 9 6 6"),
    ("keyboard", "R 2 6 20 12 M 6 10 H 6.1 M 9 10 H 9.1 M 12 10 H 12.1 M 15 10 H 15.1 M 18 10 H 18.1 M 6 14 H 18"),
    ("mouse", "R 7 2 10 20 M 12 2 V 9"),
    ("printer", "M 6 9 V 3 H 18 V 9 M 4 9 H 20 V 18 H 17 M 7 17 H 4 V 9 M 7 14 H 17 V 21 H 7 Z"),
    ("scanner", "R 3 13 18 8 M 6 13 L 3 4 M 8 17 H 16"),
    ("router", "R 3 13 18 7 M 12 13 V 9 M 8 6 L 12 9 L 16 6 M 7 17 H 7.1 M 11 17 H 11.1"),
    ("battery-charging", "R 3 8 16 8 M 21 10 V 14 M 10 9 L 7 13 H 11 L 8 17"),
    // ── People ──────────────────────────────────────────────────────────
    ("user", "O 12 8 4 M 4 21 C 4 16 8 14 12 14 C 16 14 20 16 20 21"),
    ("users", "O 9 8 3.5 M 2 21 C 2 17 5 15 9 15 C 13 15 16 17 16 21 M 16 5 A 16 8.5 3.5 300 60 M 18 21 C 18 18 17 15 15 14"),
    ("user-plus", "O 10 8 4 M 2 21 C 2 16 6 14 10 14 C 14 14 18 16 18 21 M 19 8 V 14 M 16 11 H 22"),
    ("user-minus", "O 10 8 4 M 2 21 C 2 16 6 14 10 14 C 14 14 18 16 18 21 M 16 11 H 22"),
    ("log-in", "M 14 4 H 20 V 20 H 14 M 4 12 H 15 M 11 8 L 15 12 L 11 16"),
    ("log-out", "M 10 4 H 4 V 20 H 10 M 9 12 H 20 M 16 8 L 20 12 L 16 16"),
    // ── Commerce ────────────────────────────────────────────────────────
    ("credit-card", "R 2 5 20 14 M 2 9 H 22 M 6 15 H 10"),
    ("wallet", "R 3 5 18 14 M 16 9 H 21 V 14 H 16 A 18.5 11.5 2.5 270 90 Z M 3 5 L 16 3 V 5"),
    ("shopping-cart", "M 2 3 H 5 L 8 15 H 19 L 21 7 H 6 O 9 19 1.5 O 17 19 1.5"),
    ("tag", "M 3 3 H 12 L 21 12 L 12 21 H 3 Z O 7 7 1.5"),
    ("gift", "R 3 8 18 13 M 3 12 H 21 M 12 8 V 21 M 12 8 C 9 8 6 7 6 5 C 6 3 9 2 10 4 C 11 5 12 7 12 8 Z M 12 8 C 15 8 18 7 18 5 C 18 3 15 2 14 4 C 13 5 12 7 12 8 Z"),
    ("receipt", "M 5 3 H 19 V 21 L 16 19 L 13 21 L 10 19 L 7 21 L 5 19 Z M 8 7 H 16 M 8 11 H 16 M 8 15 H 13"),
    // ── Editing & text ──────────────────────────────────────────────────
    ("type", "M 4 6 V 3 H 20 V 6 M 12 3 V 21 M 8 21 H 16"),
    ("bold", "M 6 3 H 14 C 18 3 18 13 14 13 H 6 Z M 6 13 H 15 C 19 13 19 21 15 21 H 6 Z"),
    ("italic", "M 10 3 H 19 M 5 21 H 14 M 15 3 L 9 21"),
    ("underline", "M 6 3 V 11 A 12 11 6 0 180 V 3 M 4 21 H 20"),
    ("align-left", "M 3 6 H 21 M 3 10 H 14 M 3 14 H 21 M 3 18 H 14"),
    ("align-center", "M 3 6 H 21 M 6 10 H 18 M 3 14 H 21 M 6 18 H 18"),
    ("align-right", "M 3 6 H 21 M 10 10 H 21 M 3 14 H 21 M 10 18 H 21"),
    ("heading", "M 5 3 V 21 M 19 3 V 21 M 5 12 H 19"),
    ("quote", "P 4 5 9 5 9 12 6 19 3 19 5.5 12 4 12 Z P 13 5 18 5 18 12 15 19 12 19 14.5 12 13 12 Z"),
    ("calendar", "R 3 5 18 16 M 3 10 H 21 M 8 3 V 7 M 16 3 V 7 M 7 14 H 9 M 11 14 H 13 M 15 14 H 17"),
    ("clock-alarm", "O 12 13 8 M 12 9 V 13 L 15 15 M 4 5 L 6 3 M 20 5 L 18 3"),
    // ── Misc UI ─────────────────────────────────────────────────────────
    ("toggle-left", "R 2 7 20 10 O 7 12 2"),
    ("toggle-right", "R 2 7 20 10 O 17 12 2"),
    ("checkbox", "R 3 3 18 18 M 7 12 L 10.5 15.5 L 17 8"),
    ("radio", "O 12 12 9 O 12 12 3"),
    ("circle", "O 12 12 9"),
    ("square", "R 4 4 16 16"),
    ("dot", "O 12 12 3"),
    ("gauge", "M 3 17 A 12 17 9 180 360 M 12 17 L 16 11"),
    ("activity", "M 2 12 H 6 L 9 5 L 13 19 L 16 12 H 22"),
    ("target", "O 12 12 9 O 12 12 5 O 12 12 1.6"),
    ("key", "O 7 12 4 M 10 9 L 21 20 M 17 16 L 20 13 M 14 13 L 17 10"),
    ("mail", "R 2 5 20 14 M 2 7 L 12 14 L 22 7"),
    ("message", "R 3 4 18 13 M 9 17 L 4 21 V 17 M 3 4 H 21 V 17 H 9"),
    ("send", "P 21 3 3 10 10 13 13 20 21 3 M 10 13 L 21 3"),
    ("phone", "M 5 3 H 9 L 11 8 L 8 10 C 9 13 11 15 14 16 L 16 13 L 21 15 V 19 C 21 20 20 21 19 21 C 10 21 3 14 3 5 C 3 4 4 3 5 3 Z"),
    ("globe", "O 12 12 9 M 3 12 H 21 M 12 3 C 16 7 16 17 12 21 C 8 17 8 7 12 3 Z"),
    ("power", "M 12 3 V 12 M 12 13 A 12 13 8 340 560"),
    ("loader", "A 12 12 9 20 300 M 12 3 V 6"),
];


/// Debug helper: `cargo test -p sofuu-ui --lib dump_atlas -- --ignored --nocapture`
/// writes the raw atlas to the temp dir so it can be eyeballed.
#[cfg(test)]
#[test]
#[ignore = "writes a preview file for manual inspection"]
fn dump_atlas_for_inspection() {
    let a = IconAtlas::builtin();
    let dir = std::env::temp_dir();
    std::fs::write(dir.join("sofuu_icon_atlas.r8"), &a.pixels).unwrap();
    std::fs::write(
        dir.join("sofuu_icon_atlas.dim"),
        format!("{} {}", a.width, a.height),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_atlas_ships_at_least_150_crisp_icons() {
        let atlas = IconAtlas::builtin();
        assert!(
            atlas.len() >= 150,
            "plan requires 150 icons, atlas has {}",
            atlas.len()
        );
        // Every entry has a valid, non-degenerate UV rect inside the atlas.
        for e in &atlas.entries {
            assert!(e.uv[0] >= 0.0 && e.uv[2] <= 1.0 + 1e-6);
            assert!(e.uv[1] >= 0.0 && e.uv[3] <= 1.0 + 1e-6);
            assert!(e.uv[2] > e.uv[0] && e.uv[3] > e.uv[1]);
        }
        assert_eq!(atlas.pixels.len(), (atlas.width * atlas.height) as usize);
    }

    #[test]
    fn every_icon_has_ink_and_a_clean_distance_range() {
        let atlas = IconAtlas::builtin();
        for (i, e) in atlas.entries.iter().enumerate() {
            let x0 = (e.uv[0] * atlas.width as f32).round() as u32;
            let y0 = (e.uv[1] * atlas.height as f32).round() as u32;
            let mut max_v = 0u8;
            let mut min_v = 255u8;
            let mut inside = 0usize;
            for py in y0..y0 + CELL {
                for px in x0..x0 + CELL {
                    let v = atlas.pixels[(py * atlas.width + px) as usize];
                    max_v = max_v.max(v);
                    min_v = min_v.min(v);
                    if v >= 132 {
                        inside += 1;
                    }
                }
            }
            assert!(
                min_v <= 20,
                "icon #{} ({}) has no empty pixels (min {min_v}, max {max_v})",
                i,
                e.name
            );
            assert!(
                max_v >= 150,
                "icon #{} ({}) has no interior (max {max_v})",
                i,
                e.name
            );
            assert!(
                inside >= 8,
                "icon #{} ({}) rasterized almost empty ({} px)",
                i,
                e.name,
                inside
            );
        }
    }

    #[test]
    fn custom_icons_extend_without_mutating_the_builtin_atlas() {
        let base = IconAtlas::builtin();
        let custom = IconAtlas::with_custom(&base, &[("my-glyph", "M 4 4 H 20 V 20 H 4 Z")])
            .expect("custom icon");
        assert_eq!(custom.len(), base.len() + 1);
        assert!(custom.contains("my-glyph"));
        assert!(!base.contains("my-glyph"));
        assert!(custom.uv("my-glyph").is_some());
        assert!(IconAtlas::from_paths(&[("bad", "M 1")]).is_err());
    }
}




