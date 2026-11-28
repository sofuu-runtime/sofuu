//! Tiny RGBA image buffer + PNG encoder.
//!
//! Screenshots are written by the same code path on every OS: the GPU
//! `DrawList` is rasterized into this buffer (see `raster.rs`), overlays are
//! composited here, and `png` encodes the result. No window, display server or
//! screenshot permission is involved.

use std::path::Path;

pub const TRANSPARENT: [u8; 4] = [0, 0, 0, 0];

#[derive(Clone, Debug, PartialEq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// Straight-alpha sRGB RGBA, row-major.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32, fill: [u8; 4]) -> Self {
        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        if fill != TRANSPARENT {
            for px in pixels.chunks_exact_mut(4) {
                px.copy_from_slice(&fill);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        ((y * self.width + x) * 4) as usize
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let i = self.idx(x, y);
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    #[inline]
    pub fn put(&mut self, x: u32, y: u32, c: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = self.idx(x, y);
        self.pixels[i..i + 4].copy_from_slice(&c);
    }

    /// Straight-alpha "source over destination" blend.
    #[inline]
    pub fn blend(&mut self, x: u32, y: u32, rgb: [f32; 3], alpha: f32) {
        if x >= self.width || y >= self.height || alpha <= 0.0 {
            return;
        }
        let a = alpha.clamp(0.0, 1.0);
        let i = self.idx(x, y);
        let inv = 1.0 - a;
        for ch in 0..3 {
            let dst = self.pixels[i + ch] as f32 / 255.0;
            let src = rgb[ch].clamp(0.0, 1.0);
            self.pixels[i + ch] = ((src * a + dst * inv) * 255.0).round() as u8;
        }
        let da = self.pixels[i + 3] as f32 / 255.0;
        self.pixels[i + 3] = ((a + da * inv) * 255.0).round() as u8;
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: [u8; 4]) {
        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = (x + w).min(self.width as i32).max(0) as u32;
        let y1 = (y + h).min(self.height as i32).max(0) as u32;
        for yy in y0..y1 {
            for xx in x0..x1 {
                self.put(xx, yy, c);
            }
        }
    }

    /// Alpha-blended rectangle.
    pub fn blend_rect(&mut self, x: f32, y: f32, w: f32, h: f32, rgb: [f32; 3], alpha: f32) {
        let x0 = x.floor().max(0.0) as u32;
        let y0 = y.floor().max(0.0) as u32;
        let x1 = (x + w).ceil().min(self.width as f32).max(0.0) as u32;
        let y1 = (y + h).ceil().min(self.height as f32).max(0.0) as u32;
        for yy in y0..y1 {
            for xx in x0..x1 {
                self.blend(xx, yy, rgb, alpha);
            }
        }
    }

    /// 1px (or `thickness`px) rectangle outline.
    pub fn stroke_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        thickness: i32,
        rgb: [f32; 3],
        alpha: f32,
    ) {
        let t = thickness.max(1);
        self.blend_rect(x, y, w, t as f32, rgb, alpha);
        self.blend_rect(x, y + h - t as f32, w, t as f32, rgb, alpha);
        self.blend_rect(x, y, t as f32, h, rgb, alpha);
        self.blend_rect(x + w - t as f32, y, t as f32, h, rgb, alpha);
    }

    /// Dashed rectangle outline (used for violations).
    pub fn dashed_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        dash: f32,
        thickness: i32,
        rgb: [f32; 3],
        alpha: f32,
    ) {
        let dash = dash.max(2.0);
        let mut along = 0.0f32;
        while along < w {
            let end = (along + dash * 0.6).min(w);
            self.blend_rect(x + along, y, end - along, thickness as f32, rgb, alpha);
            self.blend_rect(
                x + along,
                y + h - thickness as f32,
                end - along,
                thickness as f32,
                rgb,
                alpha,
            );
            along += dash;
        }
        along = 0.0;
        while along < h {
            let end = (along + dash * 0.6).min(h);
            self.blend_rect(x, y + along, thickness as f32, end - along, rgb, alpha);
            self.blend_rect(
                x + w - thickness as f32,
                y + along,
                thickness as f32,
                end - along,
                rgb,
                alpha,
            );
            along += dash;
        }
    }

    /// Nearest-neighbour blit into this image at `(dx, dy)`.
    pub fn blit(&mut self, src: &RgbaImage, dx: i32, dy: i32) {
        for sy in 0..src.height {
            let ty = dy + sy as i32;
            if ty < 0 || ty >= self.height as i32 {
                continue;
            }
            for sx in 0..src.width {
                let tx = dx + sx as i32;
                if tx < 0 || tx >= self.width as i32 {
                    continue;
                }
                let c = src.get(sx, sy);
                if c[3] == 0 {
                    continue;
                }
                self.put(tx as u32, ty as u32, c);
            }
        }
    }

    /// Downscale with a box filter so filmstrip tiles stay crisp.
    pub fn resize(&self, width: u32, height: u32) -> RgbaImage {
        if width == self.width && height == self.height {
            return self.clone();
        }
        let mut out = RgbaImage::new(width, height, TRANSPARENT);
        let sx = self.width as f32 / width.max(1) as f32;
        let sy = self.height as f32 / height.max(1) as f32;
        for y in 0..height {
            for x in 0..width {
                let x0 = (x as f32 * sx).floor() as u32;
                let y0 = (y as f32 * sy).floor() as u32;
                let x1 = (((x + 1) as f32 * sx).ceil() as u32).min(self.width);
                let y1 = (((y + 1) as f32 * sy).ceil() as u32).min(self.height);
                let mut acc = [0u32; 4];
                let mut n = 0u32;
                for yy in y0..y1.max(y0 + 1).min(self.height) {
                    for xx in x0..x1.max(x0 + 1).min(self.width) {
                        let c = self.get(xx, yy);
                        for i in 0..4 {
                            acc[i] += c[i] as u32;
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    out.put(
                        x,
                        y,
                        [
                            (acc[0] / n) as u8,
                            (acc[1] / n) as u8,
                            (acc[2] / n) as u8,
                            (acc[3] / n) as u8,
                        ],
                    );
                }
            }
        }
        out
    }

    pub fn encode_png(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.width, self.height);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_compression(png::Compression::Fast);
            let mut writer = enc
                .write_header()
                .map_err(|e| format!("png header: {e}"))?;
            writer
                .write_image_data(&self.pixels)
                .map_err(|e| format!("png data: {e}"))?;
        }
        Ok(out)
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let bytes = self.encode_png()?;
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }
        std::fs::write(path.as_ref(), bytes).map_err(|e| format!("{}: {e}", path.as_ref().display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_has_expected_header_and_size() {
        let img = RgbaImage::new(4, 3, [10, 20, 30, 255]);
        let png = img.encode_png().expect("png");
        assert_eq!(&png[1..4], b"PNG");
        let decoder = png::Decoder::new(std::io::Cursor::new(png));
        let mut reader = decoder.read_info().expect("info");
        assert_eq!(reader.info().width, 4);
        assert_eq!(reader.info().height, 3);
        let mut buf = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buf).expect("frame");
        assert_eq!(&buf[0..4], &[10, 20, 30, 255]);
    }

    #[test]
    fn resize_preserves_average_color() {
        let img = RgbaImage::new(8, 8, [100, 50, 25, 255]);
        let small = img.resize(2, 2);
        assert_eq!(small.width, 2);
        assert_eq!(small.get(0, 0), [100, 50, 25, 255]);
    }

    #[test]
    fn blend_is_straight_alpha() {
        let mut img = RgbaImage::new(1, 1, [0, 0, 0, 255]);
        img.blend(0, 0, [1.0, 1.0, 1.0], 0.5);
        let c = img.get(0, 0);
        assert!((c[0] as i32 - 128).abs() <= 1);
        assert_eq!(c[3], 255);
    }
}
