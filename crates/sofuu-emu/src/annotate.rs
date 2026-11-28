//! Vision-grounding overlays: numbered interactive boxes + violation callouts.
//!
//! Every label and box is drawn into a `DrawList` and rasterized by the same
//! software renderer as the frame, so `annotated` pixels and `audit.json`
//! geometry are guaranteed to share one coordinate system.

use sofuu_ui::audit::{AuditReport, AuditRules, Severity};
use sofuu_ui::draw::{
    hex_rgba, rgba8, scale_alpha, DrawList, QuadInstance, StrokeInstance, DRAW_KIND_QUAD,
    DRAW_KIND_STROKE,
};
use sofuu_ui::text::GlyphAtlas;

use crate::image::RgbaImage;
use crate::raster::{rasterize_into, Transform};

pub const GREEN: u32 = rgba8(0x22, 0xC5, 0x5E, 255);
pub const BLUE: u32 = rgba8(0x3B, 0x82, 0xF6, 255);
pub const RED: u32 = rgba8(0xEF, 0x44, 0x44, 255);
pub const AMBER: u32 = rgba8(0xF5, 0x9E, 0x0B, 255);

pub struct Overlay<'a> {
    pub dl: DrawList,
    pub atlas: &'a GlyphAtlas,
    pub viewport: [f32; 2],
}

impl<'a> Overlay<'a> {
    pub fn new(atlas: &'a GlyphAtlas, viewport: [f32; 2]) -> Self {
        let mut dl = DrawList::new();
        dl.reset(viewport);
        Self {
            dl,
            atlas,
            viewport,
        }
    }

    pub fn quad(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32) {
        self.dl.push_quad(QuadInstance {
            rect: [x, y, w, h],
            radii: [3.0; 4],
            color_top: color,
            color_bottom: color,
            border_color: 0,
            shadow_color: 0,
            params: [0.0, 0.0, 0.0, 1.0],
            clip_id: 0,
            _pad: [0; 3],
        });
    }

    pub fn box_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32, width: f32) {
        let c = color;
        self.line(x, y, x + w, y, width, c);
        self.line(x + w, y, x + w, y + h, width, c);
        self.line(x + w, y + h, x, y + h, width, c);
        self.line(x, y + h, x, y, width, c);
    }

    pub fn dashed_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: u32,
        width: f32,
        dash: f32,
    ) {
        self.dashed_line(x, y, x + w, y, width, color, dash);
        self.dashed_line(x + w, y, x + w, y + h, width, color, dash);
        self.dashed_line(x + w, y + h, x, y + h, width, color, dash);
        self.dashed_line(x, y + h, x, y, width, color, dash);
    }

    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, width: f32, color: u32) {
        self.dl.push_stroke(StrokeInstance::solid(
            [x0, y0],
            [x1, y1],
            width,
            0.75,
            color,
            0,
        ));
    }

    pub fn dashed_line(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        width: f32,
        color: u32,
        dash: f32,
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.5 {
            return;
        }
        let ux = dx / len;
        let uy = dy / len;
        let dash = dash.max(2.0);
        let mut along = 0.0;
        while along < len {
            let end = (along + dash).min(len);
            self.line(
                x0 + ux * along,
                y0 + uy * along,
                x0 + ux * end,
                y0 + uy * end,
                width,
                color,
            );
            along += dash * 1.7;
        }
    }

    /// Draw text with an optional solid background badge. Returns the badge
    /// width so callers can lay out multiple labels.
    pub fn label(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        fg: u32,
        bg: Option<u32>,
    ) -> f32 {
        let (tw, th) = self.atlas.measure(text, size, true);
        let pad = 3.0;
        if let Some(bg) = bg {
            self.quad(x, y - 1.0, tw + pad * 2.0, th + 2.0, bg);
        }
        self.atlas.layout_dl(
            text,
            x + pad,
            y + 1.0,
            size,
            fg,
            true,
            None,
            &mut self.dl,
        );
        tw + pad * 2.0
    }
}

/// Overlay `report` on top of `img`. `app_origin` is the app viewport's
/// logical position inside the rendered device; `scale` is physical/logical.
pub fn annotate_audit(
    img: &mut RgbaImage,
    report: &AuditReport,
    rules: &AuditRules,
    app_origin: [f32; 2],
    app_size: [f32; 2],
    scale: f32,
) {
    let atlas = GlyphAtlas::builtin();
    let mut ov = Overlay::new(&atlas, app_size);
    /* Labels are opaque badges: two violations on one node (or a node hugging
     * the keyboard guide) used to paint their text on top of each other.
     * Claim every badge rectangle and push a colliding one straight down. */
    let mut placed: Vec<[f32; 4]> = Vec::new();
    fn overlaps(placed: &[[f32; 4]], x: f32, y: f32, w: f32, h: f32) -> bool {
        placed
            .iter()
            .any(|r| x < r[0] + r[2] && x + w > r[0] && y < r[1] + r[3] && y + h > r[1])
    }
    /* Search both directions from the preferred spot, so a badge near the top
     * pushes down and one hugging the keyboard pushes up instead of being
     * clamped onto the guide line. */
    fn claim(
        placed: &mut Vec<[f32; 4]>,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        bounds: [f32; 2],
    ) -> (f32, f32) {
        let step = h + 2.0;
        let mut candidates = vec![y];
        for i in 1..=32 {
            candidates.push(y + i as f32 * step);
            candidates.push(y - i as f32 * step);
        }
        for yy in candidates {
            if yy < 0.0 || yy + h > bounds[1] {
                continue;
            }
            if !overlaps(placed, x, yy, w, h) {
                placed.push([x, yy, w, h]);
                return (x, yy);
            }
        }
        let yy = y.clamp(0.0, (bounds[1] - h).max(0.0));
        placed.push([x, yy, w, h]);
        (x, yy)
    }

    // Safe-area / keyboard guides first (behind boxes).
    if rules.safe.left > 0.0 {
        ov.dashed_line(
            rules.safe.left,
            0.0,
            rules.safe.left,
            app_size[1],
            1.0,
            scale_alpha(BLUE, 170.0),
            5.0,
        );
    }
    if rules.safe.right > 0.0 {
        ov.dashed_line(
            app_size[0] - rules.safe.right,
            0.0,
            app_size[0] - rules.safe.right,
            app_size[1],
            1.0,
            scale_alpha(BLUE, 170.0),
            5.0,
        );
    }
    if rules.safe.top > 0.0 {
        ov.dashed_line(
            0.0,
            rules.safe.top,
            app_size[0],
            rules.safe.top,
            1.0,
            scale_alpha(BLUE, 170.0),
            5.0,
        );
    }
    if rules.safe.bottom > 0.0 {
        let y = app_size[1] - rules.safe.bottom;
        ov.dashed_line(0.0, y, app_size[0], y, 1.0, scale_alpha(BLUE, 170.0), 5.0);
        let (tw, th) = ov.atlas.measure("HOME", 8.0, true);
        let (lx, ly) = claim(&mut placed, 6.0, y + 2.0, tw + 6.0, th + 2.0, app_size);
        ov.label("HOME", lx, ly, 8.0, BLUE, Some(rgba8(0, 0, 0, 170)));
    }
    if rules.keyboard_open && rules.keyboard_height > 0.0 {
        let y = (app_size[1] - rules.keyboard_height).max(0.0);
        ov.dashed_line(0.0, y, app_size[0], y, 1.5, scale_alpha(RED, 220.0), 6.0);
        ov.quad(0.0, y, app_size[0], app_size[1] - y, rgba8(239, 68, 68, 22));
        let (tw, th) = ov.atlas.measure("KEYBOARD", 9.0, true);
        let (lx, ly) = claim(&mut placed, 8.0, y + 4.0, tw + 6.0, th + 2.0, app_size);
        ov.label(
            "KEYBOARD",
            lx,
            ly,
            9.0,
            hex_rgba(0xFFFFFF, 255),
            Some(rgba8(0, 0, 0, 190)),
        );
    }

    // Interactive targets (green). Issue nodes are drawn again below with
    // their violation color, so a tag never appears without its box.
    for node in &report.nodes {
        if !node.interactive {
            continue;
        }
        let [x, y, w, h] = node.rect;
        ov.box_rect(x, y, w, h, scale_alpha(GREEN, 235.0), 1.25);
        /* A flagged node already gets a `[tag] SEVERITY KIND · id` callout
         * below; a second green `[tag]` badge was pure duplication and the
         * first thing to collide with the callout. */
        if report.issues.iter().any(|i| i.node_id == node.id) {
            continue;
        }
        let badge = format!("[{}]", node.tag);
        let (tw, th) = ov.atlas.measure(&badge, 9.0, true);
        let bx = (x + 2.0).min((app_size[0] - tw - 10.0).max(0.0));
        let preferred = if y >= th + 4.0 { y - th - 3.0 } else { y + 2.0 };
        let (lx, ly) = claim(&mut placed, bx, preferred, tw + 6.0, th + 2.0, app_size);
        ov.quad(lx, ly, tw + 6.0, th + 2.0, rgba8(0, 0, 0, 175));
        ov.label(
            &badge,
            lx,
            ly,
            9.0,
            GREEN,
            None,
        );
    }

    // Violations: red dashed for errors, amber for warnings.
    for issue in &report.issues {
        let (color, label) = match issue.severity {
            Severity::Error => (RED, "ERROR"),
            _ => (AMBER, "WARN"),
        };
        let [x, y, w, h] = issue.rect;
        ov.dashed_rect(x, y, w, h, scale_alpha(color, 240.0), 1.6, 5.0);
        /* Tag + severity + kind + node id: the same triple an agent reads in
         * audit.json, so the PNG alone is sufficient to name the failing
         * node without guessing coordinates from pixels. */
        let text = format!(
            "[{}] {} {} - {}",
            issue.tag,
            label,
            issue.kind.as_str().to_uppercase(),
            issue.node_id
        );
        let (tw, th) = ov.atlas.measure(&text, 8.0, true);
        let bx = (x + 2.0).min((app_size[0] - tw - 10.0).max(0.0));
        let preferred = if y >= th + 5.0 {
            y - th - 4.0
        } else {
            (y + h + 2.0).min(app_size[1] - th - 2.0)
        };
        let (lx, ly) = claim(&mut placed, bx, preferred, tw + 6.0, th + 2.0, app_size);
        ov.quad(lx, ly, tw + 6.0, th + 2.0, rgba8(0, 0, 0, 205));
        ov.label(&text, lx, ly, 8.0, color, None);
    }

    let _ = rasterize_into(
        img,
        &ov.dl,
        &atlas,
        Transform {
            offset: app_origin,
            scale,
        },
    );
}

/// Draw one text label directly in physical-pixel space (filmstrip headers).
pub fn draw_label(
    img: &mut RgbaImage,
    text: &str,
    x: f32,
    y: f32,
    size: f32,
    fg: u32,
    bg: Option<u32>,
) {
    let atlas = GlyphAtlas::builtin();
    let mut ov = Overlay::new(&atlas, [img.width as f32, img.height as f32]);
    if let Some(bg) = bg {
        let (tw, _) = atlas.measure(text, size, true);
        ov.quad(x - 3.0, y - 2.0, tw + 8.0, size + 5.0, bg);
    }
    atlas.layout_dl(text, x, y, size, fg, true, None, &mut ov.dl);
    let _ = rasterize_into(img, &ov.dl, &atlas, Transform::identity(1.0));
}

/// Add a solid rounded panel behind an area before drawing labels.
pub fn draw_panel(img: &mut RgbaImage, x: f32, y: f32, w: f32, h: f32, color: u32) {
    let atlas = GlyphAtlas::builtin();
    let mut ov = Overlay::new(&atlas, [img.width as f32, img.height as f32]);
    ov.quad(x, y, w, h, color);
    let _ = rasterize_into(img, &ov.dl, &atlas, Transform::identity(1.0));
}

// Keep the primitive-kind constants referenced so a future renderer can
// assert that overlays never emit unsupported commands.
const _: [u32; 2] = [DRAW_KIND_QUAD, DRAW_KIND_STROKE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_rasterized() {
        let mut img = RgbaImage::new(120, 40, [0, 0, 0, 255]);
        draw_label(&mut img, "[1] OK", 6.0, 6.0, 12.0, hex_rgba(0xFFFFFF, 255), None);
        assert!(img.pixels.chunks_exact(4).any(|p| p[0] > 180));
    }
}
