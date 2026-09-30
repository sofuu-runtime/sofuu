//! M1: procedural image-caption corpus for the IMG1 projector.
//!
//! Real photos need a decoder-free pipeline here too: scenes render DIRECTLY
//! to RGB pixel buffers (no PNG/JPEG involved), then feed
//! `embedding::image::features_from_rgb`. Deterministic under a fixed seed;
//! eval holds out whole parameter combos (unseen colors/counts/layouts),
//! never just unseen noise draws.

use crate::train::Rng;

pub const SCENE_W: u32 = 128;
pub const SCENE_H: u32 = 128;

const PALETTE: &[(&str, (u8, u8, u8))] = &[
    ("red", (210, 40, 40)),
    ("green", (40, 170, 70)),
    ("blue", (50, 110, 220)),
    ("yellow", (230, 200, 50)),
    ("purple", (140, 70, 200)),
    ("orange", (235, 130, 40)),
    ("teal", (40, 170, 170)),
    ("pink", (235, 130, 190)),
    ("brown", (140, 95, 60)),
    ("gray", (140, 140, 140)),
    ("navy", (30, 50, 120)),
    ("lime", (150, 220, 60)),
];

pub struct Scene {
    pub rgb: Vec<u8>,
    pub caption: String,
    pub family: u8,
    pub tag: String,
}

struct Canvas {
    rgb: Vec<u8>,
}

impl Canvas {
    fn filled(c: (u8, u8, u8)) -> Self {
        Self {
            rgb: vec![c.0, c.1, c.2].repeat((SCENE_W * SCENE_H) as usize),
        }
    }
    fn rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, c: (u8, u8, u8)) {
        for y in y0.min(127)..=y1.min(127) {
            for x in x0.min(127)..=x1.min(127) {
                let o = (y * 128 + x) * 3;
                self.rgb[o] = c.0;
                self.rgb[o + 1] = c.1;
                self.rgb[o + 2] = c.2;
            }
        }
    }
    fn hbar(&mut self, y: usize, c: (u8, u8, u8)) {
        self.rect(0, y, 127, y, c);
    }
}

/// Build `n` scenes from `seed`. Families cycle; parameters jitter.
pub fn generate(seed: u64, n: usize) -> Vec<Scene> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(one_scene(&mut rng, (i % 8) as u8));
    }
    out
}

fn pick<'a>(rng: &mut Rng, items: &'a [(&'static str, (u8, u8, u8))]) -> (&'static str, (u8, u8, u8)) {
    items[(rng.next_u32() as usize) % items.len()]
}

fn jitter(rng: &mut Rng, v: u8, amt: u8) -> u8 {
    let d = (rng.next_u32() % (2 * amt as u32 + 1)) as i32 - amt as i32;
    (v as i32 + d).clamp(0, 255) as u8
}

fn jitc(rng: &mut Rng, c: (u8, u8, u8)) -> (u8, u8, u8) {
    (jitter(rng, c.0, 18), jitter(rng, c.1, 18), jitter(rng, c.2, 18))
}

fn one_scene(rng: &mut Rng, family: u8) -> Scene {
    match family {
        // 0: solid color.
        0 => {
            let (name, c) = pick(rng, PALETTE);
            let c = jitc(rng, c);
            Scene {
                rgb: Canvas::filled(c).rgb,
                caption: format!("solid {} background", name),
                family,
                tag: format!("solid-{name}"),
            }
        }
        // 1: two-tone gradient (horizontal bands).
        1 => {
            let (n1, c1) = pick(rng, PALETTE);
            let (n2, c2) = pick(rng, PALETTE);
            let mut cv = Canvas::filled(jitc(rng, c1));
            let split = 40 + (rng.next_u32() % 48) as usize;
            cv.rect(0, split, 127, 127, jitc(rng, c2));
            Scene {
                rgb: cv.rgb,
                caption: format!("{} to {} gradient", n1, n2),
                family,
                tag: format!("grad-{n1}-{n2}"),
            }
        }
        // 2: dialog box with a button.
        2 => {
            let dark = (30 + (rng.next_u32() % 30) as u8, 32, 40);
            let mut cv = Canvas::filled(dark);
            let (bn, bc) = pick(rng, PALETTE);
            let px = 24 + (rng.next_u32() % 24) as usize;
            let py = 20 + (rng.next_u32() % 20) as usize;
            cv.rect(px, py, px + 56, py + 64, (235, 238, 242));
            let kind = if rng.next_u32() % 2 == 0 { "error" } else { "confirmation" };
            cv.rect(px + 32, py + 48, px + 52, py + 58, jitc(rng, bc));
            // Fake title bar + text lines.
            cv.rect(px + 4, py + 4, px + 52, py + 8, (200, 205, 212));
            for r in 0..3 {
                cv.rect(px + 4, py + 16 + r * 8, px + 40 - r * 6, py + 18 + r * 8, (180, 185, 192));
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("{} dialog with {} button", kind, bn),
                family,
                tag: format!("dlg-{kind}-{bn}"),
            }
        }
        // 3: bar chart.
        3 => {
            let mut cv = Canvas::filled((248, 249, 251));
            let n = 3 + (rng.next_u32() % 5) as usize;
            let (_, bc) = pick(rng, PALETTE);
            cv.hbar(110, (60, 63, 68));
            for i in 0..n {
                let bw = 100 / n;
                let hgt = 20 + (rng.next_u32() % 70) as usize;
                let x = 8 + i * bw;
                cv.rect(x, 110 - hgt, x + bw - 3, 109, jitc(rng, bc));
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("bar chart with {} bars", n),
                family,
                tag: format!("chart-{n}"),
            }
        }
        // 4: document page (text lines).
        4 => {
            let mut cv = Canvas::filled((252, 252, 250));
            let n = 4 + (rng.next_u32() % 7) as usize;
            for r in 0..n {
                let y = 12 + r * 9;
                let w = 80 + (rng.next_u32() % 32) as usize;
                cv.rect(14, y, 14 + w.min(100), y + 2, (45, 48, 55));
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("document page with {} lines of text", n),
                family,
                tag: format!("doc-{n}"),
            }
        }
        // 5: landscape (sky + ground + sun).
        5 => {
            let skies = ["pale blue", "deep blue", "sunset orange", "gray"];
            let grounds = ["green", "brown", "sandy", "snowy"];
            let sky = skies[(rng.next_u32() as usize) % skies.len()];
            let ground = grounds[(rng.next_u32() as usize) % grounds.len()];
            let sky_c: (u8, u8, u8) = match sky {
                "pale blue" => (170, 205, 235),
                "deep blue" => (40, 90, 180),
                "sunset orange" => (235, 140, 60),
                _ => (160, 165, 175),
            };
            let gnd_c: (u8, u8, u8) = match ground {
                "green" => (70, 150, 80),
                "brown" => (130, 100, 70),
                "sandy" => (225, 205, 160),
                _ => (240, 242, 245),
            };
            let mut cv = Canvas::filled(sky_c);
            let horizon = 60 + (rng.next_u32() % 30) as usize;
            cv.rect(0, horizon, 127, 127, gnd_c);
            if rng.next_u32() % 2 == 0 {
                let sx = 20 + (rng.next_u32() % 88) as usize;
                cv.rect(sx, 12, sx + 10, 22, (250, 240, 180));
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("{} landscape photo", sky),
                family,
                tag: format!("land-{sky}"),
            }
        }
        // 6: icon/photo grid.
        6 => {
            let mut cv = Canvas::filled((35, 37, 42));
            let (r, c) = (2 + (rng.next_u32() % 2) as usize, 2 + (rng.next_u32() % 3) as usize);
            for gy in 0..r {
                for gx in 0..c {
                    let (_, cc) = pick(rng, PALETTE);
                    cv.rect(8 + gx * 40, 8 + gy * 40, 8 + gx * 40 + 30, 8 + gy * 40 + 30, jitc(rng, cc));
                }
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("{} by {} photo grid", r, c),
                family,
                tag: format!("grid-{r}x{c}"),
            }
        }
        // 7: textured background (value noise blocks).
        _ => {
            let warm = rng.next_u32() % 2 == 0;
            let base: (u8, u8, u8) = if warm { (190, 150, 110) } else { (110, 150, 190) };
            let mut cv = Canvas::filled(base);
            for _ in 0..40 {
                let x = (rng.next_u32() % 120) as usize;
                let y = (rng.next_u32() % 120) as usize;
                let s = 2 + (rng.next_u32() % 6) as usize;
                let d = (rng.next_u32() % 50) as i32 - 25;
                let c = (
                    (base.0 as i32 + d).clamp(0, 255) as u8,
                    (base.1 as i32 + d).clamp(0, 255) as u8,
                    (base.2 as i32 + d).clamp(0, 255) as u8,
                );
                cv.rect(x, y, x + s, y + s, c);
            }
            Scene {
                rgb: cv.rgb,
                caption: format!("textured {} photo background", if warm { "warm" } else { "cool" }),
                family,
                tag: format!("tex-{}", if warm { "warm" } else { "cool" }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic_and_captioned() {
        let a = generate(7, 64);
        let b = generate(7, 64);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|s| s.rgb.len() == 128 * 128 * 3 && !s.caption.is_empty()));
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.rgb, y.rgb);
            assert_eq!(x.caption, y.caption);
        }
        // All eight families appear in 64 draws.
        let mut fams = [false; 8];
        for s in &a {
            fams[s.family as usize] = true;
        }
        assert!(fams.iter().all(|&f| f));
    }
}
