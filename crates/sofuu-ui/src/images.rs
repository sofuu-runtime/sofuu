//! Image decoding + shelf-packed `RGBA8` atlas pages with a bounded LRU cache.
//!
//! The UI core never rasters pixels: decoded bitmaps live in GPU-friendly
//! 2048×2048 `RGBA8` atlas pages and are sampled through the sprite pipeline
//! (`DRAW_KIND_SPRITE`). Decoding is CPU work that callers can move off the
//! render thread with [`decode_async`]; the atlas itself is a plain data
//! structure so it can be owned by whichever thread uploads GPU textures.
//!
//! * **Formats:** PNG (all standard color types) and JPEG (baseline/progressive,
//!   grayscale, RGB and CMYK). WebP is detected and reported as unsupported so
//!   hosts can hand it to a platform decoder instead of mis-parsing it.
//! * **Packing:** shelf packing per page with per-slot reuse after eviction.
//! * **Bounded VRAM:** pages are allocated lazily and the total page footprint
//!   never exceeds `max_vram_mb` (default 64 MB). Least-recently-used images are
//!   evicted first when the budget is reached.
//! * **Fit modes:** `cover`, `contain`, `fill`, `scale_down` math for mapping a
//!   source image into a destination rect without CPU resampling.
//!
//! Everything here is domain-agnostic: it knows about pixels, not about what
//! those pixels depict.

use std::collections::HashMap;
use std::fmt;
use std::io::Cursor;
use std::sync::mpsc::{self, Receiver};

/// Default atlas page edge length in pixels.
pub const DEFAULT_PAGE_SIZE: u32 = 2048;
/// Default VRAM budget for decoded image pages.
pub const DEFAULT_MAX_VRAM_MB: u32 = 64;
/// Reserved texture slot for the built-in SDF icon atlas.
pub const ICON_TEX_ID: u32 = 0;
/// First texture slot used by decoded image pages.
pub const FIRST_IMAGE_TEX_ID: u32 = 1;

/// Stable FNV-1a cache key for an image source string (`https://…`, `asset:…`, …).
#[inline]
pub fn key_for(src: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Typed image-pipeline errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// Zero-length input.
    Empty,
    /// Bytes did not start with PNG, JPEG or WebP magic.
    UnsupportedFormat,
    /// WebP was recognized but no built-in decoder is linked in this build.
    WebpNeedsPlatformDecoder,
    /// PNG decoder rejected the stream.
    Png(String),
    /// JPEG decoder rejected the stream.
    Jpeg(String),
    /// Width/height or buffer length is inconsistent.
    InvalidDimensions { width: u32, height: u32, bytes: usize },
    /// The image cannot fit the configured atlas page even after downscaling.
    ImageTooLarge { width: u32, height: u32, max: u32 },
    /// The decoded bitmap would exceed the configured VRAM budget.
    VramBudgetExceeded { needed: usize, budget: usize },
    /// A `u64` image key was not present in the atlas cache.
    NotFound(u64),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "image payload is empty"),
            Self::UnsupportedFormat => {
                write!(f, "unsupported image format (expected PNG, JPEG or WebP)")
            }
            Self::WebpNeedsPlatformDecoder => write!(
                f,
                "WebP decoding requires a platform codec in this build"
            ),
            Self::Png(e) => write!(f, "PNG decode failed: {e}"),
            Self::Jpeg(e) => write!(f, "JPEG decode failed: {e}"),
            Self::InvalidDimensions {
                width,
                height,
                bytes,
            } => write!(
                f,
                "invalid image dimensions {width}x{height} for {bytes} bytes"
            ),
            Self::ImageTooLarge {
                width,
                height,
                max,
            } => write!(
                f,
                "image {width}x{height} cannot fit a {max}px atlas page"
            ),
            Self::VramBudgetExceeded { needed, budget } => write!(
                f,
                "image needs {needed} bytes but the atlas budget is {budget} bytes"
            ),
            Self::NotFound(key) => write!(f, "image key {key} is not in the atlas"),
        }
    }
}

impl std::error::Error for ImageError {}

/// Fully decoded, tightly packed `RGBA8` pixels (row-major, 4 bytes per pixel).
#[derive(Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl fmt::Debug for DecodedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodedImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.rgba.len())
            .finish()
    }
}

impl DecodedImage {
    /// Validate an already-decoded buffer.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, ImageError> {
        let expected = width as usize * height as usize * 4;
        if width == 0 || height == 0 || rgba.len() != expected {
            return Err(ImageError::InvalidDimensions {
                width,
                height,
                bytes: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Number of decoded bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.rgba.len()
    }

    /// Aspect ratio (`width / height`), safe for zero-height images.
    #[inline]
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }

    /// Downscale so neither edge exceeds `max_dim`, preserving aspect ratio.
    ///
    /// Uses a fractional box filter: every destination pixel is the area-weighted
    /// average of the source region it covers. This is the right primitive for
    /// thumbnails (it both avoids aliasing and does not depend on a GPU).
    pub fn downscale_to_fit(&self, max_dim: u32) -> Result<Self, ImageError> {
        if max_dim == 0 {
            return Err(ImageError::ImageTooLarge {
                width: self.width,
                height: self.height,
                max: max_dim,
            });
        }
        if self.width <= max_dim && self.height <= max_dim {
            return Ok(self.clone());
        }
        let scale = (max_dim as f32 / self.width as f32)
            .min(max_dim as f32 / self.height as f32);
        let dw = ((self.width as f32 * scale).round() as u32).clamp(1, max_dim);
        let dh = ((self.height as f32 * scale).round() as u32).clamp(1, max_dim);
        let rgba = resample_box(&self.rgba, self.width, self.height, dw, dh);
        Self::new(dw, dh, rgba)
    }
}

/// Detect the container format from magic bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
}

/// Sniff an image container without decoding it.
pub fn detect_format(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(ImageFormat::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(ImageFormat::Jpeg)
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some(ImageFormat::Webp)
    } else {
        None
    }
}

/// Decode PNG or JPEG bytes into tight `RGBA8` pixels.
///
/// WebP bytes return [`ImageError::WebpNeedsPlatformDecoder`]; hosts that link
/// `libwebp`/ImageIO can decode them and call [`DecodedImage::new`] instead.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    if bytes.is_empty() {
        return Err(ImageError::Empty);
    }
    match detect_format(bytes) {
        Some(ImageFormat::Png) => decode_png(bytes),
        Some(ImageFormat::Jpeg) => decode_jpeg(bytes),
        Some(ImageFormat::Webp) => Err(ImageError::WebpNeedsPlatformDecoder),
        None => Err(ImageError::UnsupportedFormat),
    }
}

/// Decode on a short-lived worker thread.
///
/// The receiver yields exactly one result. Dropping it early is harmless; the
/// worker still exits after decoding.
pub fn decode_async(bytes: Vec<u8>) -> Receiver<Result<DecodedImage, ImageError>> {
    let (tx, rx) = mpsc::channel();
    let _ = std::thread::Builder::new()
        .name("sofuu-image-decode".into())
        .spawn(move || {
            let _ = tx.send(decode(&bytes));
        });
    rx
}

fn decode_png(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(
        png::Transformations::EXPAND | png::Transformations::STRIP_16,
    );
    let mut reader = decoder
        .read_info()
        .map_err(|e| ImageError::Png(e.to_string()))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| ImageError::Png(e.to_string()))?;
    let (w, h) = (info.width, info.height);
    let data = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => rgb_to_rgba(data),
        png::ColorType::Grayscale => gray_to_rgba(data, false),
        png::ColorType::GrayscaleAlpha => gray_to_rgba(data, true),
        png::ColorType::Indexed => {
            return Err(ImageError::Png(
                "indexed PNG was not expanded by the decoder".into(),
            ))
        }
    };
    DecodedImage::new(w, h, rgba)
}

fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let pixels = decoder
        .decode()
        .map_err(|e| ImageError::Jpeg(e.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| ImageError::Jpeg("JPEG header missing after decode".into()))?;
    let (w, h) = (info.width as u32, info.height as u32);
    let rgba = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => rgb_to_rgba(&pixels),
        jpeg_decoder::PixelFormat::L8 => gray_to_rgba(&pixels, false),
        jpeg_decoder::PixelFormat::L16 => {
            // 16-bit grayscale: keep the high byte.
            let gray: Vec<u8> = pixels.chunks_exact(2).map(|p| p[0]).collect();
            gray_to_rgba(&gray, false)
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            let mut out = Vec::with_capacity(pixels.len() / 4 * 4);
            for px in pixels.chunks_exact(4) {
                let k = 255 - px[3] as u16;
                let r = 255 - ((px[0] as u16 * k) / 255) as u8;
                let g = 255 - ((px[1] as u16 * k) / 255) as u8;
                let b = 255 - ((px[2] as u16 * k) / 255) as u8;
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
    };
    DecodedImage::new(w, h, rgba)
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}

fn gray_to_rgba(gray: &[u8], has_alpha: bool) -> Vec<u8> {
    let step = if has_alpha { 2 } else { 1 };
    let mut out = Vec::with_capacity(gray.len() / step * 4);
    for px in gray.chunks_exact(step) {
        let a = if has_alpha { px[1] } else { 255 };
        out.extend_from_slice(&[px[0], px[0], px[0], a]);
    }
    out
}

/// Area-weighted box resample (fractional edges supported).
fn resample_box(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for ty in 0..dh {
        let y0 = ty as f32 * sy;
        let y1 = (ty + 1) as f32 * sy;
        let iy0 = y0.floor() as u32;
        let iy1 = (y1.ceil() as u32).min(sh);
        for tx in 0..dw {
            let x0 = tx as f32 * sx;
            let x1 = (tx + 1) as f32 * sx;
            let ix0 = x0.floor() as u32;
            let ix1 = (x1.ceil() as u32).min(sw);
            let mut acc = [0f32; 4];
            let mut wsum = 0f32;
            for y in iy0..iy1 {
                let wy = ((y + 1) as f32).min(y1) - (y as f32).max(y0);
                if wy <= 0.0 {
                    continue;
                }
                for x in ix0..ix1 {
                    let wx = ((x + 1) as f32).min(x1) - (x as f32).max(x0);
                    if wx <= 0.0 {
                        continue;
                    }
                    let w = wx * wy;
                    let o = ((y * sw + x) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += src[o + c] as f32 * w;
                    }
                    wsum += w;
                }
            }
            let o = ((ty * dw + tx) * 4) as usize;
            if wsum > 0.0 {
                for c in 0..4 {
                    out[o + c] = (acc[c] / wsum).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// How a source image is mapped into a destination rect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFit {
    /// Fill the rect and crop the overflow (default for avatars/cards).
    #[default]
    Cover,
    /// Preserve the whole image and letterbox inside the rect.
    Contain,
    /// Stretch to the rect, distorting aspect ratio.
    Fill,
    /// Like `Contain`, but never upscale beyond the image's natural size.
    ScaleDown,
}

/// Result of fitting a source image into a destination box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedImage {
    /// Destination rectangle `[x, y, w, h]` in logical px.
    pub rect: [f32; 4],
    /// Source UV crop `[u0, v0, u1, v1]` in normalized texture space.
    pub uv: [f32; 4],
}

/// Compute the destination rect and source UV crop for a fit mode.
pub fn fit_image(
    fit: ImageFit,
    dest: [f32; 4],
    src_w: u32,
    src_h: u32,
) -> FittedImage {
    let [x, y, dw, dh] = dest;
    let (sw, sh) = (src_w.max(1) as f32, src_h.max(1) as f32);
    if dw <= 0.0 || dh <= 0.0 {
        return FittedImage {
            rect: dest,
            uv: [0.0, 0.0, 1.0, 1.0],
        };
    }
    match fit {
        ImageFit::Fill => FittedImage {
            rect: dest,
            uv: [0.0, 0.0, 1.0, 1.0],
        },
        ImageFit::Contain | ImageFit::ScaleDown => {
            let fits = sw <= dw && sh <= dh;
            if fit == ImageFit::ScaleDown && fits {
                let w = sw;
                let h = sh;
                return FittedImage {
                    rect: [x + (dw - w) * 0.5, y + (dh - h) * 0.5, w, h],
                    uv: [0.0, 0.0, 1.0, 1.0],
                };
            }
            let s = (dw / sw).min(dh / sh);
            let w = sw * s;
            let h = sh * s;
            FittedImage {
                rect: [x + (dw - w) * 0.5, y + (dh - h) * 0.5, w, h],
                uv: [0.0, 0.0, 1.0, 1.0],
            }
        }
        ImageFit::Cover => {
            let s = (dw / sw).max(dh / sh);
            let vis_w = (dw / s / sw).clamp(0.0, 1.0);
            let vis_h = (dh / s / sh).clamp(0.0, 1.0);
            let u0 = (1.0 - vis_w) * 0.5;
            let v0 = (1.0 - vis_h) * 0.5;
            FittedImage {
                rect: dest,
                uv: [u0, v0, u0 + vis_w, v0 + vis_h],
            }
        }
    }
}

/// A placement in an atlas page, ready to feed a `SpriteInstance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageHandle {
    /// Caller-provided cache key.
    pub key: u64,
    /// Texture slot: `0` is the icon atlas, `>= 1` maps to `page - 1`.
    pub tex_id: u32,
    /// Normalized `[u0, v0, u1, v1]` inside that texture.
    pub uv: [f32; 4],
    /// Pixels in the atlas slot (after any downscale).
    pub width: u32,
    /// Pixels in the atlas slot (after any downscale).
    pub height: u32,
    /// Bumped whenever the slot is re-packed; stale handles can be discarded.
    pub generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct Shelf {
    y: u32,
    h: u32,
    cursor: u32,
}

/// One `RGBA8` atlas page. `pixels` is row-major, 4 bytes per texel.
#[derive(Clone, Debug)]
pub struct ImagePage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    /// Incremented on every write so GPU uploads can skip clean pages.
    pub revision: u64,
    shelves: Vec<Shelf>,
    slots: Vec<(u64, Slot)>,
}

impl ImagePage {
    fn new(size: u32) -> Self {
        Self {
            width: size,
            height: size,
            pixels: vec![0u8; size as usize * size as usize * 4],
            revision: 1,
            shelves: Vec::new(),
            slots: Vec::new(),
        }
    }

    /// Live slots `(key, [x, y, w, h])` in this page.
    pub fn slots(&self) -> impl Iterator<Item = (u64, [u32; 4])> + '_ {
        self.slots.iter().map(|(k, s)| (*k, [s.x, s.y, s.w, s.h]))
    }

    fn reset(&mut self) {
        self.shelves.clear();
        self.slots.clear();
        self.revision += 1;
    }

    fn try_place(&mut self, w: u32, h: u32) -> Option<Slot> {
        if w > self.width || h > self.height {
            return None;
        }
        for shelf in &mut self.shelves {
            if shelf.h >= h && shelf.cursor + w <= self.width {
                let slot = Slot {
                    x: shelf.cursor,
                    y: shelf.y,
                    w,
                    h,
                    generation: 0,
                };
                shelf.cursor += w;
                return Some(slot);
            }
        }
        let y = self
            .shelves
            .last()
            .map(|s| s.y + s.h)
            .unwrap_or(0);
        if y + h <= self.height {
            self.shelves.push(Shelf {
                y,
                h,
                cursor: w,
            });
            return Some(Slot {
                x: 0,
                y,
                w,
                h,
                generation: 0,
            });
        }
        None
    }

    /// Rebuild this page's shelves from its live slots (sorted tallest first).
    /// Returns false and leaves the page untouched when repacking cannot fit.
    fn repack(&mut self, generation: u64) -> bool {
        let mut live: Vec<(u64, u32, u32)> = self
            .slots
            .iter()
            .map(|(k, s)| (*k, s.w, s.h))
            .collect();
        live.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
        let mut shelves: Vec<Shelf> = Vec::new();
        let mut slots: Vec<(u64, Slot)> = Vec::with_capacity(live.len());
        for (key, w, h) in live {
            let mut placed = None;
            for shelf in &mut shelves {
                if shelf.h >= h && shelf.cursor + w <= self.width {
                    placed = Some(Slot {
                        x: shelf.cursor,
                        y: shelf.y,
                        w,
                        h,
                        generation,
                    });
                    shelf.cursor += w;
                    break;
                }
            }
            if placed.is_none() {
                let y = shelves.last().map(|s| s.y + s.h).unwrap_or(0);
                if y + h > self.height {
                    return false;
                }
                shelves.push(Shelf {
                    y,
                    h,
                    cursor: w,
                });
                placed = Some(Slot {
                    x: 0,
                    y,
                    w,
                    h,
                    generation,
                });
            }
            slots.push((key, placed.unwrap()));
        }
        self.shelves = shelves;
        self.slots = slots;
        self.revision += 1;
        true
    }
}

/// Bounded, shelf-packed image atlas.
///
/// Pages are `page_size² RGBA8`. The atlas guarantees
/// `vram_bytes() <= max_bytes` at all times (assuming `max_bytes` is at least one
/// page). Inserting beyond the budget evicts least-recently-used images.
pub struct ImageAtlas {
    page_size: u32,
    max_bytes: usize,
    pages: Vec<ImagePage>,
    /// key -> (page index, slot)
    entries: HashMap<u64, (usize, Slot)>,
    /// Eviction order, oldest at the front. May contain stale keys after a
    /// `remove`; those are skipped lazily.
    lru: Vec<u64>,
    generation: u64,
}

impl fmt::Debug for ImageAtlas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageAtlas")
            .field("page_size", &self.page_size)
            .field("pages", &self.pages.len())
            .field("images", &self.entries.len())
            .field("vram_bytes", &self.vram_bytes())
            .field("budget", &self.max_bytes)
            .finish()
    }
}

impl Default for ImageAtlas {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_VRAM_MB)
    }
}

impl ImageAtlas {
    /// Build an atlas with the default 2048 px page size.
    pub fn new(max_vram_mb: u32) -> Self {
        Self::with_page_size(DEFAULT_PAGE_SIZE, max_vram_mb)
    }

    /// Build an atlas with an explicit page size (clamped to `64..=8192`).
    pub fn with_page_size(page_size: u32, max_vram_mb: u32) -> Self {
        let page_size = page_size.clamp(64, 8192);
        let page_bytes = page_size as usize * page_size as usize * 4;
        let requested = max_vram_mb as usize * 1024 * 1024;
        Self {
            page_size,
            // Always allow at least one page so tiny budgets are still usable.
            max_bytes: requested.max(page_bytes),
            pages: Vec::new(),
            entries: HashMap::new(),
            lru: Vec::new(),
            generation: 1,
        }
    }

    #[inline]
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    #[inline]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Bytes currently reserved by whole pages (the VRAM the GPU must hold).
    #[inline]
    pub fn vram_bytes(&self) -> usize {
        self.pages.len() * self.page_size as usize * self.page_size as usize * 4
    }

    /// Bytes of live image data (excludes shelf slack).
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|(_, s)| s.w as usize * s.h as usize * 4)
            .sum()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Read-only page access for GPU upload (`tex_id = 1 + index`).
    pub fn page(&self, index: usize) -> Option<&ImagePage> {
        self.pages.get(index)
    }

    pub fn pages(&self) -> &[ImagePage] {
        &self.pages
    }

    /// Insert (or refresh) an image and return its placement.
    ///
    /// Images larger than a page are box-downscaled to fit first. If the atlas
    /// is at its VRAM budget the least-recently-used images are evicted.
    pub fn insert(&mut self, key: u64, image: DecodedImage) -> Result<ImageHandle, ImageError> {
        if image.width == 0 || image.height == 0 {
            return Err(ImageError::InvalidDimensions {
                width: image.width,
                height: image.height,
                bytes: image.rgba.len(),
            });
        }
        if let Some(handle) = self.get(key) {
            return Ok(handle);
        }
        let image = if image.width > self.page_size || image.height > self.page_size {
            image.downscale_to_fit(self.page_size)?
        } else {
            image
        };
        let need = image.rgba.len();

        // Keep used image bytes inside the budget.
        while self.used_bytes() + need > self.max_bytes && self.evict_lru() {}

        let page_bytes = self.page_size as usize * self.page_size as usize * 4;
        let mut repacked = false;
        loop {
            if let Some((page_idx, slot)) = self.try_place(image.width, image.height) {
                self.generation += 1;
                let gen = self.generation;
                let slot = Slot {
                    generation: gen,
                    ..slot
                };
                self.write_slot(page_idx, key, slot, &image.rgba);
                self.entries.insert(key, (page_idx, slot));
                self.touch(key);
                return Ok(self.handle(key, page_idx, slot));
            }
            if !repacked {
                repacked = true;
                let gen = self.generation + 1;
                for page in &mut self.pages {
                    let _ = page.repack(gen);
                }
                self.generation = gen;
                continue;
            }
            // Allocate a page if the VRAM budget has room.
            if (self.pages.len() + 1) * page_bytes <= self.max_bytes {
                self.pages.push(ImagePage::new(self.page_size));
                continue;
            }
            // Otherwise free space from the LRU and try again.
            if !self.evict_lru() {
                return Err(ImageError::VramBudgetExceeded {
                    needed: need,
                    budget: self.max_bytes,
                });
            }
            let gen = self.generation + 1;
            for page in &mut self.pages {
                let _ = page.repack(gen);
            }
            self.generation = gen;
            repacked = true;
        }
    }

    /// Fetch a placement, marking it most-recently-used.
    pub fn get(&mut self, key: u64) -> Option<ImageHandle> {
        let (page_idx, slot) = *self.entries.get(&key)?;
        self.touch(key);
        Some(self.handle(key, page_idx, slot))
    }

    /// Fetch a placement without touching the LRU order.
    pub fn peek(&self, key: u64) -> Option<ImageHandle> {
        let (page_idx, slot) = *self.entries.get(&key)?;
        Some(self.handle(key, page_idx, slot))
    }

    /// Remove an image and release its slot (reclaimed on the next repack).
    pub fn remove(&mut self, key: u64) -> bool {
        let Some((page_idx, slot)) = self.entries.remove(&key) else {
            return false;
        };
        let page = &mut self.pages[page_idx];
        page.slots.retain(|(k, _)| *k != key);
        page.revision += 1;
        if page.slots.is_empty() {
            page.reset();
        }
        let _ = slot;
        true
    }

    /// Drop every decoded image (pages are released too).
    pub fn clear(&mut self) {
        self.pages.clear();
        self.entries.clear();
        self.lru.clear();
    }

    /// Evict until `vram_bytes() <= target` or the atlas is empty.
    pub fn trim_to(&mut self, target_bytes: usize) {
        while self.vram_bytes() > target_bytes {
            if !self.evict_lru() {
                break;
            }
            // Drop trailing empty pages so `vram_bytes` can actually shrink.
            while let Some(last) = self.pages.last() {
                if last.slots.is_empty() {
                    self.pages.pop();
                } else {
                    break;
                }
            }
        }
    }

    fn handle(&self, key: u64, page_idx: usize, slot: Slot) -> ImageHandle {
        let size = self.page_size as f32;
        ImageHandle {
            key,
            tex_id: FIRST_IMAGE_TEX_ID + page_idx as u32,
            uv: [
                slot.x as f32 / size,
                slot.y as f32 / size,
                (slot.x + slot.w) as f32 / size,
                (slot.y + slot.h) as f32 / size,
            ],
            width: slot.w,
            height: slot.h,
            generation: slot.generation,
        }
    }

    fn write_slot(&mut self, page_idx: usize, key: u64, slot: Slot, rgba: &[u8]) {
        let page = &mut self.pages[page_idx];
        let stride = page.width as usize * 4;
        for row in 0..slot.h as usize {
            let src = &rgba[row * slot.w as usize * 4..(row + 1) * slot.w as usize * 4];
            let dst_off = (slot.y as usize + row) * stride + slot.x as usize * 4;
            page.pixels[dst_off..dst_off + src.len()].copy_from_slice(src);
        }
        page.slots.push((key, slot));
        page.revision += 1;
    }

    fn try_place(&mut self, w: u32, h: u32) -> Option<(usize, Slot)> {
        for (idx, page) in self.pages.iter_mut().enumerate() {
            if let Some(slot) = page.try_place(w, h) {
                return Some((idx, slot));
            }
        }
        None
    }

    fn touch(&mut self, key: u64) {
        if let Some(pos) = self.lru.iter().position(|k| *k == key) {
            self.lru.remove(pos);
        }
        self.lru.push(key);
    }

    /// Evict the oldest live entry. Returns false when the atlas is empty.
    fn evict_lru(&mut self) -> bool {
        while let Some(key) = self.lru.first().copied() {
            self.lru.remove(0);
            if self.entries.contains_key(&key) {
                return self.remove(key);
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> DecodedImage {
        let mut px = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..width * height {
            px.extend_from_slice(&rgba);
        }
        DecodedImage::new(width, height, px).unwrap()
    }

    fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, width, height);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(rgba).unwrap();
        }
        out
    }

    #[test]
    fn png_round_trips_exact_pixels() {
        let pixels = vec![
            255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 255, 255, 255, 0,
        ];
        let bytes = png_bytes(2, 2, &pixels);
        assert_eq!(detect_format(&bytes), Some(ImageFormat::Png));
        let img = decode(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(img.rgba, pixels);
    }

    #[test]
    fn grayscale_png_expands_to_rgba() {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 2, 1);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(&[10, 200]).unwrap();
        }
        let img = decode(&out).unwrap();
        assert_eq!(img.rgba, vec![10, 10, 10, 255, 200, 200, 200, 255]);
    }

    /// 1x1 white baseline JPEG.
    const TINY_JPEG: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00,
        0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
        0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B,
        0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31,
        0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF,
        0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00,
        0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x0A, 0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08,
        0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7F, 0xFF, 0xD9,
    ];

    #[test]
    fn jpeg_decodes_to_rgba() {
        assert_eq!(detect_format(TINY_JPEG), Some(ImageFormat::Jpeg));
        let img = decode(TINY_JPEG).unwrap();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba.len(), 4);
        assert!(img.rgba[0] > 200, "expected a light JPEG pixel");
        assert_eq!(img.rgba[3], 255);
    }

    #[test]
    fn empty_unknown_and_webp_are_typed_errors() {
        assert_eq!(decode(&[]), Err(ImageError::Empty));
        assert_eq!(decode(b"hello"), Err(ImageError::UnsupportedFormat));
        assert_eq!(
            decode(b"RIFFxxxxWEBP"),
            Err(ImageError::WebpNeedsPlatformDecoder)
        );
        assert!(matches!(
            decode(b"\x89PNG\r\n\x1a\nnot-really"),
            Err(ImageError::Png(_))
        ));
    }

    #[test]
    fn downscale_keeps_aspect_and_averages_blocks() {
        let img = solid(8, 4, [200, 100, 50, 255]);
        let small = img.downscale_to_fit(4).unwrap();
        assert_eq!((small.width, small.height), (4, 2));
        assert_eq!(small.rgba.len(), 4 * 2 * 4);
        assert_eq!(&small.rgba[0..4], &[200, 100, 50, 255]);
        // Already small enough: unchanged.
        assert_eq!(small.downscale_to_fit(8).unwrap(), small);
    }

    #[test]
    fn atlas_shelf_packs_without_overlap() {
        let mut atlas = ImageAtlas::with_page_size(256, 1);
        let keys = [1, 2, 3];
        for k in keys {
            atlas.insert(k, solid(64, 64, [255, 255, 255, 255])).unwrap();
        }
        assert_eq!(atlas.len(), 3);
        assert_eq!(atlas.page_count(), 1);
        let mut rects = Vec::new();
        for k in keys {
            let h = atlas.get(k).unwrap();
            let x = (h.uv[0] * 256.0).round() as u32;
            let y = (h.uv[1] * 256.0).round() as u32;
            rects.push((x, y, 64, 64));
            assert_eq!(h.tex_id, FIRST_IMAGE_TEX_ID);
        }
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let a = rects[i];
                let b = rects[j];
                let disjoint = a.0 + a.2 <= b.0
                    || b.0 + b.2 <= a.0
                    || a.1 + a.3 <= b.1
                    || b.1 + b.3 <= a.1;
                assert!(disjoint, "atlas slots overlap: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn lru_eviction_keeps_vram_under_the_cap() {
        // 256px page = 256 KB; 1 MB budget = 4 pages.
        let mut atlas = ImageAtlas::with_page_size(256, 1);
        for k in 0..40u64 {
            atlas.insert(k, solid(128, 128, [k as u8, 0, 0, 255])).unwrap();
            assert!(
                atlas.vram_bytes() <= 1024 * 1024,
                "VRAM cap exceeded: {} bytes",
                atlas.vram_bytes()
            );
            assert!(atlas.used_bytes() <= 1024 * 1024);
        }
        assert!(atlas.len() < 40);
        assert!(atlas.peek(0).is_none(), "oldest image must be evicted");
        assert!(atlas.peek(39).is_some(), "newest image must survive");
    }

    #[test]
    fn oversized_images_are_downscaled_into_a_page() {
        let mut atlas = ImageAtlas::with_page_size(256, 1);
        let handle = atlas.insert(7, solid(1200, 600, [1, 2, 3, 255])).unwrap();
        assert!(handle.width <= 256 && handle.height <= 256);
        assert_eq!((handle.width, handle.height), (256, 128));
    }

    #[test]
    fn remove_and_clear_free_data() {
        let mut atlas = ImageAtlas::with_page_size(128, 1);
        atlas.insert(1, solid(32, 32, [0; 4])).unwrap();
        atlas.insert(2, solid(32, 32, [0; 4])).unwrap();
        assert!(atlas.remove(1));
        assert!(!atlas.remove(1));
        assert!(atlas.peek(1).is_none());
        assert_eq!(atlas.len(), 1);
        atlas.clear();
        assert!(atlas.is_empty());
        assert_eq!(atlas.vram_bytes(), 0);
    }

    #[test]
    fn repacking_reuses_evicted_space() {
        let mut atlas = ImageAtlas::with_page_size(128, 1);
        for k in 0..8u64 {
            atlas.insert(k, solid(32, 32, [0; 4])).unwrap();
        }
        // Evict everything, then insert again: still one page, no leak.
        for k in 0..8u64 {
            atlas.remove(k);
        }
        for k in 8..16u64 {
            atlas.insert(k, solid(32, 32, [0; 4])).unwrap();
        }
        assert_eq!(atlas.page_count(), 1);
        assert!(atlas.vram_bytes() <= atlas.max_bytes());
    }

    #[test]
    fn fit_modes_cover_contain_fill_and_scale_down() {
        let dest = [10.0, 20.0, 100.0, 50.0];
        // Source is 2:1, dest is 2:1 -> no crop for cover.
        let cover = fit_image(ImageFit::Cover, dest, 200, 100);
        assert_eq!(cover.rect, dest);
        assert!((cover.uv[0] - 0.0).abs() < 1e-6);
        assert!((cover.uv[2] - 1.0).abs() < 1e-6);

        // Square source into a 2:1 box: half the height is cropped.
        let cover = fit_image(ImageFit::Cover, dest, 100, 100);
        assert!((cover.uv[1] - 0.25).abs() < 1e-6);
        assert!((cover.uv[3] - 0.75).abs() < 1e-6);

        let contain = fit_image(ImageFit::Contain, dest, 100, 100);
        assert_eq!(contain.rect, [35.0, 20.0, 50.0, 50.0]);
        assert_eq!(contain.uv, [0.0, 0.0, 1.0, 1.0]);

        let fill = fit_image(ImageFit::Fill, dest, 100, 100);
        assert_eq!(fill.rect, dest);
        assert_eq!(fill.uv, [0.0, 0.0, 1.0, 1.0]);

        let scale_down = fit_image(ImageFit::ScaleDown, dest, 40, 40);
        assert_eq!(scale_down.rect, [40.0, 25.0, 40.0, 40.0]);

        let scale_down = fit_image(ImageFit::ScaleDown, dest, 400, 400);
        assert_eq!(scale_down.rect, [35.0, 20.0, 50.0, 50.0]);
    }

    #[test]
    fn async_decode_delivers_one_result() {
        let bytes = png_bytes(1, 1, &[9, 8, 7, 255]);
        let rx = decode_async(bytes);
        let img = rx.recv().unwrap().unwrap();
        assert_eq!(img.rgba, vec![9, 8, 7, 255]);
    }
}
