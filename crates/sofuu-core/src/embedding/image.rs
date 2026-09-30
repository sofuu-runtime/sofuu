//! M1: frozen hand-built image features (IMGF1).
//!
//! The projector trains against these features — never against pixels — so
//! this layout is a **frozen contract**: same bytes → same features, every
//! build, every platform. Changing anything below (grid sizes, bin edges,
//! DCT selection, normalization) invalidates every trained artifact and
//! must bump `IMGF_VERSION` plus the artifact format id.
//!
//! Pipeline: sniff magic → decode to RGB8 (png / jpeg-decoder crates) →
//! box-downsample to a 64×64 luminance working buffer (+ full-res chroma
//! accumulators) → fixed 144-value layout, all in [0,1]-ish ranges:
//!
//! ```text
//! [0..64]    luminance 8×8 grid (row-major, 0=black 1=white)
//! [64..96]   chroma Cb 4×4 + Cr 4×4 (0.5 = achromatic, pooled from RGB)
//! [96..108]  color histogram: 4 bins × {R,G,B} (sums to 3.0)
//! [108..124] Sobel edge magnitude 4×4 (pooled from the 8×8 lum grid)
//! [124..140] 8×8 DCT-II of the lum grid: DC + 15 zigzag AC coeffs
//! [140..144] globals: mean lum, contrast (std), saturation, edge density
//! ```
//!
//! Caps (match the vision-input caps philosophy in rt/ai.rs): max side
//! 2048px, max ~8MP total, min 2×2. Anything else refuses — an embedder
//! must never OOM the host on a hostile file.

/// Frozen feature-layout version. Bump with ANY layout change.
pub const IMGF_VERSION: u32 = 1;
/// Fixed feature count. The projector's input dim; never changes for v1.
pub const IMG_FEAT_DIM: usize = 144;
/// Working thumbnail side (luminance buffer).
const THUMB: usize = 64;
/// Luminance analysis grid side.
const GRID: usize = 8;
/// Decoder caps.
const MAX_SIDE: u32 = 2048;
const MAX_PIXELS: u64 = 8 * 1024 * 1024;

/// Decode PNG/JPEG bytes to RGB8. Returns (rgb, width, height).
pub fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), &'static str> {
    if bytes.len() < 8 {
        return Err("image too small to sniff");
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(bytes)
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        decode_jpeg(bytes)
    } else {
        Err("unsupported image format (PNG/JPEG only)")
    }
}

fn check_dims(w: u32, h: u32) -> Result<(), &'static str> {
    if w < 2 || h < 2 {
        return Err("image smaller than 2x2");
    }
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err("image side exceeds 2048px cap");
    }
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err("image exceeds 8MP cap");
    }
    Ok(())
}

fn decode_png(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), &'static str> {
    let dec = png::Decoder::new(bytes);
    let mut reader = dec.read_info().map_err(|_| "png header unreadable")?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|_| "png frame undecodable")?;
    let (w, h) = (info.width, info.height);
    check_dims(w, h)?;
    // Normalize every color type to RGB8.
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => {
            if buf.len() < (w * h * 3) as usize {
                return Err("png rgb buffer short");
            }
            buf[..(w * h * 3) as usize].to_vec()
        }
        png::ColorType::Rgba => {
            if buf.len() < (w * h * 4) as usize {
                return Err("png rgba buffer short");
            }
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            for px in buf[..(w * h * 4) as usize].chunks_exact(4) {
                // Composite over white (screenshots with alpha read sanely).
                let a = px[3] as u32;
                out.push(((px[0] as u32 * a + 255 * (255 - a)) / 255) as u8);
                out.push(((px[1] as u32 * a + 255 * (255 - a)) / 255) as u8);
                out.push(((px[2] as u32 * a + 255 * (255 - a)) / 255) as u8);
            }
            out
        }
        png::ColorType::Grayscale => {
            if buf.len() < (w * h) as usize {
                return Err("png gray buffer short");
            }
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            for &g in &buf[..(w * h) as usize] {
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            if buf.len() < (w * h * 2) as usize {
                return Err("png gray-alpha buffer short");
            }
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            for px in buf[..(w * h * 2) as usize].chunks_exact(2) {
                let a = px[1] as u32;
                let g = ((px[0] as u32 * a + 255 * (255 - a)) / 255) as u8;
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        png::ColorType::Indexed => {
            // Expand via the PLTE chunk the decoder resolved.
            let palette = reader.info().palette.as_ref().ok_or("png indexed without palette")?;
            if buf.len() < (w * h) as usize {
                return Err("png indexed buffer short");
            }
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            for &idx in &buf[..(w * h) as usize] {
                let o = (idx as usize) * 3;
                if o + 3 > palette.len() {
                    return Err("png palette index out of range");
                }
                out.extend_from_slice(&palette[o..o + 3]);
            }
            out
        }
    };
    Ok((rgb, w, h))
}

fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), &'static str> {
    let mut dec = jpeg_decoder::Decoder::new(bytes);
    let rgb = dec.decode().map_err(|_| "jpeg undecodable")?;
    let info = dec.info().ok_or("jpeg info missing")?;
    let (w, h) = (info.width as u32, info.height as u32);
    check_dims(w, h)?;
    if rgb.len() < (w * h * 3) as usize {
        return Err("jpeg rgb buffer short");
    }
    Ok((rgb[..(w * h * 3) as usize].to_vec(), w, h))
}

/// Luminance + chroma of one sRGB pixel (BT.601 luma; Cb/Cr centered).
#[inline]
fn ycc(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (rf, gf, bf) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    (y, 0.5 + (bf - y) * 0.564, 0.5 + (rf - y) * 0.713)
}

/// Box-downsample RGB8 to a 64×64 luminance buffer + full-res accumulators.
fn thumbnail(rgb: &[u8], w: u32, h: u32) -> ([f32; THUMB * THUMB], ImageStats) {
    let (w, h) = (w as usize, h as usize);
    let mut lum = [0f32; THUMB * THUMB];
    let mut stats = ImageStats::default();
    let sx = w as f32 / THUMB as f32;
    let sy = h as f32 / THUMB as f32;
    for ty in 0..THUMB {
        for tx in 0..THUMB {
            // Source box for this thumb cell (clamped, at least 1px).
            let x0 = ((tx as f32) * sx).floor() as usize;
            let x1 = (((tx + 1) as f32) * sx).ceil() as usize;
            let y0 = ((ty as f32) * sy).floor() as usize;
            let y1 = (((ty + 1) as f32) * sy).ceil() as usize;
            let (x0, x1) = (x0.min(w - 1), x1.clamp(x0 + 1, w));
            let (y0, y1) = (y0.min(h - 1), y1.clamp(y0 + 1, h));
            let mut sy_: f64 = 0.0;
            let mut scb = 0.0;
            let mut scr = 0.0;
            let mut n = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let o = (y * w + x) * 3;
                    let (yy, cb, cr) = ycc(rgb[o], rgb[o + 1], rgb[o + 2]);
                    sy_ += yy as f64;
                    scb += cb as f64;
                    scr += cr as f64;
                    stats.push(yy, cb, cr, rgb[o], rgb[o + 1], rgb[o + 2]);
                    n += 1;
                }
            }
            let n = n.max(1) as f32;
            lum[ty * THUMB + tx] = sy_ as f32 / n;
            stats.cb_grid[ty * THUMB + tx] = scb as f32 / n;
            stats.cr_grid[ty * THUMB + tx] = scr as f32 / n;
        }
    }
    (lum, stats)
}

struct ImageStats {
    cb_grid: [f32; THUMB * THUMB],
    cr_grid: [f32; THUMB * THUMB],
    hist: [f32; 12],
    sum_y: f64,
    sum_y2: f64,
    pixels: u64,
}

impl Default for ImageStats {
    fn default() -> Self {
        Self {
            cb_grid: [0.0; THUMB * THUMB],
            cr_grid: [0.0; THUMB * THUMB],
            hist: [0.0; 12],
            sum_y: 0.0,
            sum_y2: 0.0,
            pixels: 0,
        }
    }
}

impl ImageStats {
    fn push(&mut self, y: f32, _cb: f32, _cr: f32, r: u8, g: u8, b: u8) {
        for (ch, v) in [r, g, b].iter().enumerate() {
            let bin = ((*v as usize * 4) / 256).min(3);
            self.hist[ch * 4 + bin] += 1.0;
        }
        self.sum_y += y as f64;
        self.sum_y2 += (y as f64) * (y as f64);
        self.pixels += 1;
    }
    fn finish(&mut self) {
        let n = self.pixels.max(1) as f32;
        for h in self.hist.iter_mut() {
            // Per-channel fractions: each channel's 4 bins sum to 1.0,
            // so the 12 cells sum to 3.0.
            *h = *h / n;
        }
    }
}

/// Downsample a THUMB grid to a smaller square by averaging.
fn pool<const N: usize>(grid: &[f32; THUMB * THUMB]) -> [f32; N] {
    let side = (N as f32).sqrt() as usize;
    debug_assert_eq!(side * side, N);
    let mut out = [0.0f32; N];
    let cell = THUMB / side;
    for oy in 0..side {
        for ox in 0..side {
            let mut s = 0.0;
            for dy in 0..cell {
                for dx in 0..cell {
                    s += grid[(oy * cell + dy) * THUMB + ox * cell + dx];
                }
            }
            out[oy * side + ox] = s / (cell * cell) as f32;
        }
    }
    out
}

/// Sobel magnitude on the 8×8 luminance grid (clamped borders), then the
/// 4×4 pooled magnitudes are written by the caller.
fn sobel8(lum8: &[f32; 64]) -> [f32; 64] {
    let at = |x: isize, y: isize| -> f32 {
        lum8[y.clamp(0, 7) as usize * 8 + x.clamp(0, 7) as usize]
    };
    let mut m = [0f32; 64];
    for y in 0isize..8 {
        for x in 0isize..8 {
            let gx = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)
                - at(x - 1, y - 1)
                - 2.0 * at(x - 1, y)
                - at(x - 1, y + 1);
            let gy = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)
                - at(x - 1, y - 1)
                - 2.0 * at(x, y - 1)
                - at(x + 1, y - 1);
            m[y as usize * 8 + x as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    m
}

/// 8×8 DCT-II of the luminance grid; returns DC + first 15 zigzag AC.
fn dct16(lum8: &[f32; 64]) -> [f32; 16] {
    // Zigzag order indices for an 8×8 block (first 16, DC first).
    const ZIG: [(usize, usize); 16] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
        (4, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 4),
        (0, 5),
    ];
    let cos = |u: usize, x: usize| -> f32 {
        ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos()
    };
    let mut out = [0f32; 16];
    for (i, &(u, v)) in ZIG.iter().enumerate() {
        let mut s = 0.0;
        for y in 0..8 {
            for x in 0..8 {
                s += lum8[y * 8 + x] * cos(u, x) * cos(v, y);
            }
        }
        let cu = if u == 0 { 0.70710678 } else { 1.0 };
        let cv = if v == 0 { 0.70710678 } else { 1.0 };
        out[i] = 0.25 * cu * cv * s;
    }
    out
}

/// Frozen IMGF1 features for decoded RGB8. All cells ≈[0,1] except DCT AC
/// (signed, small) and contrast (≥0).
pub fn features_from_rgb(rgb: &[u8], w: u32, h: u32) -> [f32; IMG_FEAT_DIM] {
    let (lum64, mut stats) = thumbnail(rgb, w, h);
    stats.finish();
    let lum8: [f32; 64] = pool::<64>(&lum64);
    let cb4: [f32; 16] = pool::<16>(&stats.cb_grid);
    let cr4: [f32; 16] = pool::<16>(&stats.cr_grid);
    let edge = sobel8(&lum8);
    // Pool the 8×8 edge map to 4×4 by averaging 2×2 cells.
    let mut edge4 = [0f32; 16];
    for oy in 0..4 {
        for ox in 0..4 {
            edge4[oy * 4 + ox] = (edge[(oy * 2) * 8 + ox * 2]
                + edge[(oy * 2) * 8 + ox * 2 + 1]
                + edge[(oy * 2 + 1) * 8 + ox * 2]
                + edge[(oy * 2 + 1) * 8 + ox * 2 + 1])
                * 0.25;
        }
    }
    let dct = dct16(&lum8);
    let n = stats.pixels.max(1) as f64;
    let mean = (stats.sum_y / n) as f32;
    let var = ((stats.sum_y2 / n) - (stats.sum_y / n).powi(2)).max(0.0) as f32;
    let mut edge_sum = 0.0;
    for e in edge.iter() {
        edge_sum += *e;
    }
    // Saturation proxy: mean |Cb-.5|+|Cr-.5| over the thumb (cheap loop).
    let mut sat = 0.0;
    for i in 0..THUMB * THUMB {
        sat += (stats.cb_grid[i] - 0.5).abs() + (stats.cr_grid[i] - 0.5).abs();
    }
    sat /= (THUMB * THUMB) as f32;

    let mut f = [0f32; IMG_FEAT_DIM];
    f[0..64].copy_from_slice(&lum8);
    f[64..80].copy_from_slice(&cb4);
    f[80..96].copy_from_slice(&cr4);
    f[96..108].copy_from_slice(&stats.hist);
    f[108..124].copy_from_slice(&edge4);
    f[124..140].copy_from_slice(&dct);
    f[140] = mean;
    f[141] = var.sqrt();
    f[142] = sat;
    f[143] = edge_sum / 64.0;
    f
}

/// End-to-end: bytes → IMGF1 features.
pub fn img_features(bytes: &[u8]) -> Result<[f32; IMG_FEAT_DIM], &'static str> {
    let (rgb, w, h) = decode_image(bytes)?;
    Ok(features_from_rgb(&rgb, w, h))
}

// ── IMG1 projector (tiny distilled image model) ─────────────────────
// Architecture mirrors SEM1 deliberately: dense IN→16 tanh →64, int8
// per-row weights + f32 scales/biases, L2-normalized 64-dim output.
// The output space is trained INTO sem2-64 text geometry (CLIP-style
// joint space): space id "img1-64" is documented-compatible with
// "sem2-64" cosine, versioned and pinned — the one exception to the
// one-space-per-index rule, and only because the eval gate (§M1-Q)
// proves text-text recall is unharmed with image vectors present.
//
// Artifact layout (all LE, exact length IMG_BLOB_LEN, no trailing bytes):
//   [0..4]    "IMG1" magic
//   [4..8]    version u32 = 1
//   [8..20]   in_dim, hidden, out_dim u32 (144, 16, 64)
//   [20..]    W1 int8[16*144], W1_scales f32[16], b1 f32[16],
//             W2 int8[64*16], W2_scales f32[64], b2 f32[64]
//   [tail-8..] artifact_id u64 = fnv1a64(all preceding bytes)

/// Projector input dim (= IMG_FEAT_DIM — features are the contract).
pub const IMG_IN: usize = IMG_FEAT_DIM;
/// Bottleneck width (32: cross-modal mapping needs more room than SEM1's
/// 16; artifact stays ~7.4KB of the 32KiB budget).
pub const IMG_HIDDEN: usize = 32;
/// Output dim (matches SEM2-64 text geometry by training design).
pub const IMG_DIM: usize = 64;
/// Stable model id for manifests and space tagging.
pub const MODEL_ID_IMG: &str = "image-projector-v1";
/// Artifact format magic.
pub const MODEL_FORMAT_IMG: &str = "IMG1";
/// Space id: compatible-with-sem2-64 cosine by versioned design.
pub const SPACE_IMG: &str = "img1-64";
/// Artifact format version.
pub const IMG_VERSION: u32 = 1;

const IMG_W1_LEN: usize = IMG_HIDDEN * IMG_IN; // 2304
const IMG_W2_LEN: usize = IMG_DIM * IMG_HIDDEN; // 1024
const IMG_BLOB_LEN: usize = 20
    + IMG_W1_LEN
    + IMG_HIDDEN * 4 // W1 scales
    + IMG_HIDDEN * 4 // b1
    + IMG_W2_LEN
    + IMG_DIM * 4 // W2 scales
    + IMG_DIM * 4 // b2
    + 8; // artifact id

const IMG_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/weights_img.sem"));

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Parsed IMG1 artifact (borrowed views into the baked blob).
pub struct QuantizedImg<'a> {
    pub w1: &'a [i8],
    pub s1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [i8],
    pub s2: &'a [f32],
    pub b2: &'a [f32],
    pub artifact: u64,
}

fn read_f32_le(bytes: &[u8], at: &mut usize) -> Result<f32, &'static str> {
    if *at + 4 > bytes.len() {
        return Err("img1 truncated float");
    }
    let v = f32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
    *at += 4;
    Ok(v)
}

fn read_u32_le(bytes: &[u8], at: &mut usize) -> Result<u32, &'static str> {
    if *at + 4 > bytes.len() {
        return Err("img1 truncated u32");
    }
    let v = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
    *at += 4;
    Ok(v)
}

impl<'a> QuantizedImg<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self, &'static str> {
        if blob.len() != IMG_BLOB_LEN {
            return Err("img1 wrong length");
        }
        if &blob[0..4] != b"IMG1" {
            return Err("img1 bad magic");
        }
        let mut at = 4;
        if read_u32_le(blob, &mut at)? != IMG_VERSION {
            return Err("img1 unsupported version");
        }
        if read_u32_le(blob, &mut at)? as usize != IMG_IN
            || read_u32_le(blob, &mut at)? as usize != IMG_HIDDEN
            || read_u32_le(blob, &mut at)? as usize != IMG_DIM
        {
            return Err("img1 dim mismatch");
        }
        // at == 20 here by construction.
        let w1 = &blob[at..at + IMG_W1_LEN];
        at += IMG_W1_LEN;
        let w1 = unsafe { std::slice::from_raw_parts(w1.as_ptr() as *const i8, IMG_W1_LEN) };
        let mut s1 = Vec::with_capacity(IMG_HIDDEN);
        for _ in 0..IMG_HIDDEN {
            s1.push(read_f32_le(blob, &mut at)?);
        }
        let mut b1 = Vec::with_capacity(IMG_HIDDEN);
        for _ in 0..IMG_HIDDEN {
            b1.push(read_f32_le(blob, &mut at)?);
        }
        let w2b = &blob[at..at + IMG_W2_LEN];
        at += IMG_W2_LEN;
        let w2 = unsafe { std::slice::from_raw_parts(w2b.as_ptr() as *const i8, IMG_W2_LEN) };
        let mut s2 = Vec::with_capacity(IMG_DIM);
        for _ in 0..IMG_DIM {
            s2.push(read_f32_le(blob, &mut at)?);
        }
        let mut b2 = Vec::with_capacity(IMG_DIM);
        for _ in 0..IMG_DIM {
            b2.push(read_f32_le(blob, &mut at)?);
        }
        if at + 8 != blob.len() {
            return Err("img1 trailing bytes");
        }
        let artifact = u64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
        if artifact != fnv1a64(&blob[..at]) {
            return Err("img1 artifact id mismatch");
        }
        if s1.iter().chain(s2.iter()).any(|&s| !s.is_finite() || s <= 0.0)
            || b1.iter().chain(b2.iter()).any(|&b| !b.is_finite())
        {
            return Err("img1 non-finite calibration");
        }
        // Leak the small calibration vecs into 'static (4×80 f32s, once).
        // Keeps the struct borrow-only over the blob with owned scales.
        fn leak(v: Vec<f32>) -> &'static [f32] {
            Box::leak(v.into_boxed_slice())
        }
        Ok(Self {
            w1,
            s1: leak(s1),
            b1: leak(b1),
            w2,
            s2: leak(s2),
            b2: leak(b2),
            artifact,
        })
    }

    /// Forward: features → L2-normalized 64-dim vector. None on degenerate
    /// (non-finite/zero-norm) output — never silent garbage.
    pub fn forward(&self, x: &[f32; IMG_FEAT_DIM]) -> Option<[f32; IMG_DIM]> {
        let mut h = [0f32; IMG_HIDDEN];
        for j in 0..IMG_HIDDEN {
            let row = &self.w1[j * IMG_IN..(j + 1) * IMG_IN];
            let mut s = 0.0;
            for (q, &v) in row.iter().zip(x.iter()) {
                s += *q as f32 * v;
            }
            h[j] = (s * self.s1[j] + self.b1[j]).tanh();
        }
        let mut y = [0f32; IMG_DIM];
        for k in 0..IMG_DIM {
            let row = &self.w2[k * IMG_HIDDEN..(k + 1) * IMG_HIDDEN];
            let mut s = 0.0;
            for (q, &v) in row.iter().zip(h.iter()) {
                s += *q as f32 * v;
            }
            y[k] = s * self.s2[k] + self.b2[k];
        }
        let mut norm = 0.0;
        for v in y.iter() {
            if !v.is_finite() {
                return None;
            }
            norm += *v * *v;
        }
        norm = norm.sqrt();
        if !(norm > 1e-12) {
            return None;
        }
        for v in y.iter_mut() {
            *v /= norm;
        }
        Some(y)
    }
}

/// Baked IMG1 model, or Err while untrained (empty staged blob) / corrupt.
pub fn baked_model_img() -> Result<QuantizedImg<'static>, &'static str> {
    QuantizedImg::parse(IMG_BLOB)
}

/// End-to-end: image bytes → 64-dim unit vector in img1-64 space.
pub fn semantic_img(bytes: &[u8]) -> Option<Vec<f32>> {
    let m = baked_model_img().ok()?;
    let f = img_features(bytes).ok()?;
    m.forward(&f).map(|y| y.to_vec())
}

/// Space manifest JSON for embed-info surfaces.
pub fn model_info_json_img() -> String {
    match baked_model_img() {
        Ok(m) => format!(
            "{{\"id\":\"{}\",\"input\":\"imgf-v1\",\"format\":\"{}\",\"blobVersion\":{},\"dimension\":{},\"hidden\":{},\"space\":\"{}\",\"space_compat\":\"sem2-64\",\"artifact\":\"{:016x}\",\"available\":true}}",
            MODEL_ID_IMG, MODEL_FORMAT_IMG, IMG_VERSION, IMG_DIM, IMG_HIDDEN, SPACE_IMG, m.artifact
        ),
        Err(_) => format!(
            "{{\"id\":\"{}\",\"input\":\"imgf-v1\",\"format\":\"{}\",\"blobVersion\":{},\"dimension\":{},\"hidden\":{},\"space\":\"{}\",\"space_compat\":\"sem2-64\",\"artifact\":\"none\",\"available\":false}}",
            MODEL_ID_IMG, MODEL_FORMAT_IMG, IMG_VERSION, IMG_DIM, IMG_HIDDEN, SPACE_IMG
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny RGB PNG in-memory (png encoder is core API).
    fn make_png(w: u32, h: u32, px: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w8 = enc.write_header().unwrap();
            w8.write_image_data(px).unwrap();
        }
        out
    }

    #[test]
    fn png_round_trip_and_layout_ranges() {
        // 4×4 gradient: R ramps with x, G with y, B fixed.
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                px.extend_from_slice(&[(x * 64) as u8, (y * 64) as u8, 128]);
            }
        }
        let bytes = make_png(4, 4, &px);
        let f = img_features(&bytes).expect("4x4 rgb png must decode");
        assert_eq!(f.len(), IMG_FEAT_DIM);
        for (i, &v) in f.iter().enumerate() {
            assert!(v.is_finite(), "feature {i} not finite: {v}");
        }
        // Red ramps left→right: lum must rise across the first row.
        assert!(f[7] > f[0], "lum row should rise with red ramp");
        // Histogram sums to 3.0 (per-channel fractions ×3).
        let hs: f32 = f[96..108].iter().sum();
        assert!((hs - 3.0).abs() < 1e-3, "hist sums {hs}");
        // Global mean luminance in (0,1) for a mid-tone image.
        assert!(f[140] > 0.1 && f[140] < 0.9, "mean {}", f[140]);
        // DC term ≈ 8× mean for the orthonormal DCT scaling used.
        assert!((f[124] - 8.0 * f[140]).abs() < 1e-2, "dc {} vs mean {}", f[124], f[140]);
    }

    #[test]
    fn solid_color_has_no_edges_but_has_chroma() {
        let px = vec![200u8, 30, 30].repeat(8 * 8);
        let bytes = make_png(8, 8, &px);
        let f = img_features(&bytes).expect("solid png must decode");
        let edge: f32 = f[108..124].iter().sum();
        assert!(edge < 1e-4, "solid color must have ~zero edges, got {edge}");
        // Strong red: Cr high, Cb low-ish.
        let cb: f32 = f[64..80].iter().sum::<f32>() / 16.0;
        let cr: f32 = f[80..96].iter().sum::<f32>() / 16.0;
        assert!(cr > cb, "red must push Cr above Cb ({cr} vs {cb})");
    }

    #[test]
    fn determinism_same_bytes_same_features() {
        let px = vec![10u8, 200, 90].repeat(6 * 6);
        let bytes = make_png(6, 6, &px);
        let a = img_features(&bytes).unwrap();
        let b = img_features(&bytes).unwrap();
        assert_eq!(a, b, "features must be bit-identical across runs");
    }

    #[test]
    fn baked_absent_or_valid_never_garbage() {
        // Pre-training the staged blob is empty: the loader must refuse
        // (None), never return a degenerate vector. Post-training it must
        // parse, forward finitely, and unit-normalize.
        match baked_model_img() {
            Err(_) => {
                let px = vec![50u8, 100, 150].repeat(4 * 4);
                assert!(semantic_img(&make_png(4, 4, &px)).is_none());
            }
            Ok(m) => {
                assert_eq!(m.artifact, fnv1a64(&IMG_BLOB[..IMG_BLOB.len() - 8]));
                let px = vec![50u8, 100, 150].repeat(4 * 4);
                let bytes = make_png(4, 4, &px);
                let v = semantic_img(&bytes).expect("valid model must embed");
                assert_eq!(v.len(), IMG_DIM);
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((n - 1.0).abs() < 1e-4, "img output unit, got {n}");
                let info = model_info_json_img();
                assert!(info.contains("\"available\":true"), "{info}");
            }
        }
    }

    #[test]
    fn golden_artifact_id_and_vector_are_pinned() {
        // Round-1 IMG1 artifact. ANY change (retrain, reqant, layout) must
        // update these pins deliberately — silent drift breaks every brain
        // that stored img1-64 vectors.
        let m = baked_model_img().expect("img1 artifact must be baked to pin goldens");
        assert_eq!(format!("{:016x}", m.artifact), "9b0b7077897360c5");
        // Golden forward vector (first 8 of 64) for a fixed 4×4 input.
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                px.extend_from_slice(&[(x * 64) as u8, (y * 64) as u8, 128]);
            }
        }
        let bytes = make_png(4, 4, &px);
        let v = semantic_img(&bytes).expect("golden input must embed");
        let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.6}")).collect();
        assert_eq!(
            head.join(","),
            "-0.037028,0.091512,-0.049016,0.016268,-0.046195,0.030557,-0.045408,0.001515"
        );
    }

    #[test]
    fn hostile_inputs_refuse() {
        assert!(img_features(b"").is_err());
        assert!(img_features(b"definitely not an image at all....").is_err());
        // PNG magic, then garbage.
        assert!(img_features(b"\x89PNG\r\n\x1a\nGARBAGE").is_err());
        // JPEG magic, then garbage.
        assert!(img_features(b"\xff\xd8\xffGARBAGE").is_err());
        // 1×1 pixel (below 2×2 floor).
        let px = vec![0u8, 0, 0];
        assert!(img_features(&make_png(1, 1, &px)).is_err());
    }
}
