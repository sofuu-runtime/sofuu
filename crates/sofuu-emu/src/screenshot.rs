//! Headless screenshot engine: `clean`, `annotated` and `filmstrip` PNGs.
//!
//! The screenshot path never opens a window. It compiles the app scene at the
//! platform's exact logical viewport, compiles the device chrome around a
//! clone of it, rasterizes the resulting `DrawList` into an RGBA buffer and
//! encodes PNG. The same run produces the geometric `AuditReport` so pixels
//! and JSON can never drift apart.
//!
//! Two app compiles are intentional:
//!
//! * the **audit** engine renders only the app at `app_size`, so tagged
//!   rectangles are app-local and emulator chrome (status bar buttons, home
//!   indicator) never pollutes the report;
//! * the **pixel** engine renders the full device frame, including status bar,
//!   Dynamic Island / punch hole, titlebar, home bar and the IME overlay.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use sofuu_ui::draw::DrawList;
use sofuu_ui::hit_test::ScrollPhysics;
use sofuu_ui::ir::{Scene, ScenePatch};
use sofuu_ui::theme::ThemeMode;
use sofuu_ui::UiEngine;

use crate::annotate::{annotate_audit, draw_label, draw_panel};
use crate::host::{EmulatorConfig, EmulatorHost};
use crate::image::RgbaImage;
use crate::profiles::resolve;
use crate::raster::{rasterize, RasterStats};

/// One of the three AI-oriented output modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    /// True user view: no audit overlay.
    Clean,
    /// True user view + numbered interactive boxes and violation callouts.
    Annotated,
    /// Up to six platform states stitched into one labeled contact sheet.
    Filmstrip,
}

impl CaptureMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CaptureMode::Clean => "clean",
            CaptureMode::Annotated => "annotated",
            CaptureMode::Filmstrip => "filmstrip",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "clean" | "user" => Ok(CaptureMode::Clean),
            "annotated" | "audit" | "overlay" => Ok(CaptureMode::Annotated),
            "filmstrip" | "contact" | "matrix" => Ok(CaptureMode::Filmstrip),
            other => Err(format!(
                "unknown capture mode `{other}` (expected clean, annotated or filmstrip)"
            )),
        }
    }
}

impl Default for CaptureMode {
    fn default() -> Self {
        CaptureMode::Clean
    }
}

/// Headless scroll gesture replayed after the patch stream (E7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScrollAction {
    pub id: String,
    pub delta_y: f32,
}

/// Options shared by every capture mode.
#[derive(Clone, Debug, Default)]
pub struct ShotOptions {
    pub mode: CaptureMode,
    /// PNG destination. `None` uses `.sofu/screenshots/<platform>-<device>-<mode>.png`.
    pub out: Option<PathBuf>,
    /// Optional `audit.json` destination.
    pub audit_json: Option<PathBuf>,
    /// Cap the physical scale (e.g. `1.0` for fast CI captures).
    pub scale_limit: Option<f32>,
    /// Hard cap for filmstrip tiles (the plan calls for six).
    pub max_tiles: usize,
    /// Scroll gestures replayed after the patch stream.
    pub scrolls: Vec<ScrollAction>,
}

impl ShotOptions {
    pub fn new(mode: CaptureMode) -> Self {
        Self {
            mode,
            max_tiles: 6,
            ..Self::default()
        }
    }
}

/// Declarative render job used by the `sofuu-emu` binary and by
/// `sofuu-core`'s dynamic bridge. Keeping it JSON-shaped means the core never
/// links `sofuu-emu` statically.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderJob {
    #[serde(default)]
    pub app: Option<Scene>,
    #[serde(default)]
    pub config: EmulatorConfig,
    #[serde(default)]
    pub mode: CaptureMode,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub audit_json: Option<String>,
    #[serde(default)]
    pub scale_limit: Option<f32>,
    /// Patch stream recorded from `sofuu.ui` (mount + incremental updates).
    /// Replayed through `UiEngine` before the frame is captured.
    #[serde(default)]
    pub patches: Vec<ScenePatch>,
    /// Headless scroll gestures replayed after `patches` (E7).
    #[serde(default)]
    pub scrolls: Vec<ScrollAction>,
    #[serde(default)]
    pub fail_on_error: bool,
    /// Optional multi-state filmstrip. Each tile carries its own config and
    /// app scene so callers can drive different scenarios/keyboards.
    #[serde(default)]
    pub tiles: Vec<RenderTileJob>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderTileJob {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub config: EmulatorConfig,
    pub app: Scene,
    #[serde(default)]
    pub patches: Vec<ScenePatch>,
}

impl RenderJob {
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("render job: {e}"))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TileCapture {
    pub label: String,
    pub platform: String,
    pub device: String,
    pub width: u32,
    pub height: u32,
    pub audit: sofuu_ui::AuditReport,
}

/// CLI/JS-facing result. PNG bytes stay in [`Captured::image`] and are not
/// serialized (agents read paths instead of base64).
#[derive(Clone, Debug, Serialize)]
pub struct Captured {
    pub ok: bool,
    pub capture_png: Option<String>,
    pub mode: String,
    pub platform: String,
    pub device: String,
    pub width: u32,
    pub height: u32,
    /// Always `software` today. Native offscreen GPU backends can report
    /// `metal` / `egl` / `vulkan` without changing this contract.
    pub renderer: String,
    pub capture_ms: f64,
    pub audit: sofuu_ui::AuditReport,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiles: Vec<TileCapture>,
    #[serde(skip)]
    pub image: RgbaImage,
}

impl Captured {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn write_audit_json(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }
        let json = serde_json::to_string_pretty(&self.audit)
            .map_err(|e| format!("audit json: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Output of compiling + rasterizing one frame, before file encoding.
pub struct FrameShot {
    pub image: RgbaImage,
    pub audit: sofuu_ui::AuditReport,
    pub stats: RasterStats,
    pub scale: f32,
    pub frames: u32,
    pub settled: bool,
    /// `metal` / `egl` / `vulkan` when the GPU offscreen path ran, else `software`.
    pub renderer: &'static str,
}

/// Compile a frame and audit it. `mode` only affects metadata (annotation is
/// applied by the caller because filmstrip tiles are always clean).
pub fn render_frame(
    host: &EmulatorHost,
    app: Scene,
    mode: CaptureMode,
    scale_limit: Option<f32>,
) -> Result<FrameShot, String> {
    render_frame_patched(host, app, &[], mode, scale_limit)
}

/// Compile the app, replay every recorded `ScenePatch`, then capture.
pub fn render_frame_patched(
    host: &EmulatorHost,
    app: Scene,
    patches: &[ScenePatch],
    mode: CaptureMode,
    scale_limit: Option<f32>,
) -> Result<FrameShot, String> {
    render_frame_actions(host, app, patches, &[], mode, scale_limit)
}

/// Compile, replay patches and scroll gestures, then capture.
pub fn render_frame_actions(
    host: &EmulatorHost,
    app: Scene,
    patches: &[ScenePatch],
    scrolls: &[ScrollAction],
    mode: CaptureMode,
    scale_limit: Option<f32>,
) -> Result<FrameShot, String> {
    let app_size = host.app_size(false);
    let rules = host.audit_rules(host.keyboard_open);
    let mut app_engine = UiEngine::new(app.clone(), app_size);
    // Platform-exact edge behaviour: iOS/macOS rubber band, Android stretch,
    // desktop clamp + scrollbar.
    app_engine.set_scroll_physics(ScrollPhysics::for_platform(&host.profile.id));
    for patch in patches {
        app_engine.apply_patch(patch.clone());
    }
    if !scrolls.is_empty() {
        // Layout must exist before a scroll container can be resolved.
        let mut warm = DrawList::new();
        let _ = app_engine.tick_and_compile(1.0 / 60.0, &mut warm);
        for scroll in scrolls {
            app_engine.scroll_by(&scroll.id, scroll.delta_y);
        }
    }
    let mut app_dl = DrawList::new();
    let (app_frames, app_settled) = settle(&mut app_engine, &mut app_dl);
    let mut audit = app_engine.audit(&app_dl, &rules);
    audit.mode = mode.as_str().to_string();
    audit.platform = host.profile.id.clone();
    audit.device = host.device.id.clone();
    audit.viewport_px = app_size;
    audit.theme = mode_name(host.theme);
    audit.keyboard_open = host.keyboard_open && host.geometry().keyboard_height > 0.0;
    audit.settled = app_settled;
    audit.frames_stepped = app_frames;

    let g = host.geometry();
    let final_app = app_engine.scene.clone();
    let device_scene = host.wrap_device_scene(final_app, false);
    let mut device_engine = UiEngine::new(device_scene, [g.width, g.height]);
    device_engine.set_scroll_physics(ScrollPhysics::for_platform(&host.profile.id));
    let mut dl = DrawList::new();
    let (device_frames, device_settled) = settle(&mut device_engine, &mut dl);

    let scale = pick_scale([g.width, g.height], g.scale, scale_limit);
    let bg = device_engine.theme.bg;
    let (mut img, stats, renderer) =
        match crate::gpu::render(&dl, &device_engine.atlas, [g.width, g.height], scale) {
            Some((pixels, render_ms)) => {
                let width = (g.width * scale).round().max(1.0) as u32;
                let height = (g.height * scale).round().max(1.0) as u32;
                (
                    RgbaImage {
                        width,
                        height,
                        pixels,
                    },
                    RasterStats {
                        quads: dl.quads.len(),
                        ribbons: dl.ribbons.len(),
                        strokes: dl.strokes.len(),
                        glyphs: dl.glyphs.len(),
                        sprites: dl.sprites.len(),
                        render_ms,
                    },
                    "metal",
                )
            }
            None => {
                let (img, stats) = rasterize(
                    &dl,
                    &device_engine.atlas,
                    [g.width, g.height],
                    scale,
                    unpack(bg),
                );
                (img, stats, "software")
            }
        };

    // The IME is composited, never in layout flow, so a full-height app can
    // still be caught by `keyboard_occluded`.
    if host.keyboard_open && g.keyboard_height > 0.0 {
        if let Some(kb) = render_overlay_part(
            host,
            host.keyboard_node(),
            [g.width, g.keyboard_height.max(160.0)],
            scale,
        ) {
            let y = ((g.height - g.keyboard_height.max(160.0)) * scale).round() as i32;
            img.blit(&kb, 0, y);
        }
    }
    if g.home_bar_height > 0.0 {
        if let Some(bar) = render_overlay_part(
            host,
            host.home_bar_node(),
            [g.width, g.home_bar_height],
            scale,
        ) {
            let y = ((g.height - g.home_bar_height) * scale).round() as i32;
            img.blit(&bar, 0, y);
        }
    }

    audit.scale_factor = scale;
    audit.renderer = renderer.to_string();
    audit.frames_stepped = audit.frames_stepped.max(device_frames);
    audit.settled = audit.settled && device_settled;
    audit.perf = sofuu_ui::AuditPerf {
        quads: stats.quads,
        ribbons: stats.ribbons,
        strokes: stats.strokes,
        glyphs: stats.glyphs,
        sprites: stats.sprites,
        render_ms: stats.render_ms,
    };

    Ok(FrameShot {
        image: img,
        audit,
        stats,
        scale,
        frames: device_frames,
        settled: device_settled,
        renderer,
    })
}

/// Capture a single device frame (clean or annotated).
pub fn capture(
    host: &EmulatorHost,
    app: Scene,
    opts: &ShotOptions,
) -> Result<Captured, String> {
    capture_patched(host, app, &[], opts)
}

/// Capture a single device frame after replaying recorded patches.
pub fn capture_patched(
    host: &EmulatorHost,
    app: Scene,
    patches: &[ScenePatch],
    opts: &ShotOptions,
) -> Result<Captured, String> {
    if opts.mode == CaptureMode::Filmstrip {
        let tile = FilmstripTile {
            label: format!("{} - {}", host.profile.name, host.device.name),
            host: host.clone(),
            app,
            patches: patches.to_vec(),
        };
        return capture_filmstrip(std::slice::from_ref(&tile), opts);
    }
    let start = Instant::now();
    let mut shot = render_frame_actions(
        host,
        app,
        patches,
        &opts.scrolls,
        opts.mode,
        opts.scale_limit,
    )?;
    let out = opts
        .out
        .clone()
        .unwrap_or_else(|| default_out(host, opts.mode));
    if opts.mode == CaptureMode::Annotated {
        annotate_audit(
            &mut shot.image,
            &shot.audit,
            &host.audit_rules(host.keyboard_open),
            host.app_origin(),
            host.app_size(false),
            shot.scale,
        );
    }
    shot.image.save_png(&out)?;
    shot.audit.capture_png = Some(absolute(&out));
    let captured = Captured {
        ok: !shot.audit.has_errors(),
        capture_png: Some(absolute(&out)),
        mode: opts.mode.as_str().to_string(),
        platform: host.profile.id.clone(),
        device: host.device.id.clone(),
        width: shot.image.width,
        height: shot.image.height,
        renderer: shot.renderer.to_string(),
        capture_ms: start.elapsed().as_secs_f64() * 1000.0,
        audit: shot.audit,
        tiles: Vec::new(),
        image: shot.image,
    };
    if let Some(path) = &opts.audit_json {
        captured.write_audit_json(path)?;
    }
    Ok(captured)
}

pub struct FilmstripTile {
    pub label: String,
    pub host: EmulatorHost,
    pub app: Scene,
    #[doc(hidden)]
    pub patches: Vec<ScenePatch>,
}

/// Stitch up to six platform states into a single labeled contact sheet.
pub fn capture_filmstrip(
    tiles: &[FilmstripTile],
    opts: &ShotOptions,
) -> Result<Captured, String> {
    if tiles.is_empty() {
        return Err("filmstrip needs at least one tile".to_string());
    }
    let limit = opts.max_tiles.clamp(1, 6);
    if tiles.len() > limit {
        return Err(format!(
            "filmstrip supports at most {limit} tiles (received {})",
            tiles.len()
        ));
    }
    let start = Instant::now();
    let mut rendered = Vec::with_capacity(tiles.len());
    for tile in tiles {
        let shot = render_frame_actions(
            &tile.host,
            tile.app.clone(),
            &tile.patches,
            &opts.scrolls,
            CaptureMode::Clean,
            Some(1.0),
        )?;
        rendered.push((tile, shot));
    }

    const CELL_W: u32 = 340;
    const CELL_H: u32 = 660;
    const PAD: u32 = 18;
    const TITLE: u32 = 30;
    let n = rendered.len();
    let cols = match n {
        1 => 1,
        2 => 2,
        _ => 3,
    };
    let rows = (n + cols - 1) / cols;
    let cell_w = CELL_W;
    let cell_h = CELL_H;
    let width = PAD + cols as u32 * (cell_w + PAD);
    let mut height = PAD + rows as u32 * (cell_h + TITLE + PAD);

    let mut sheet = RgbaImage::new(width, height, [11, 14, 20, 255]);

    let mut tile_captures = Vec::with_capacity(n);
    let mut merged = sofuu_ui::AuditReport {
        mode: CaptureMode::Filmstrip.as_str().to_string(),
        platform: "filmstrip".to_string(),
        device: format!("{n} tiles"),
        theme: mode_name(tiles[0].host.theme),
        keyboard_open: rendered.iter().any(|(_, s)| s.audit.keyboard_open),
        settled: rendered.iter().all(|(_, s)| s.audit.settled),
        frames_stepped: rendered.iter().map(|(_, s)| s.audit.frames_stepped).max().unwrap_or(0),
        ..sofuu_ui::AuditReport::default()
    };

    for (i, (tile, shot)) in rendered.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = PAD + col as u32 * (cell_w + PAD);
        let y = PAD + row as u32 * (cell_h + TITLE + PAD);
        // Card background + title.
        draw_panel(&mut sheet, x as f32, y as f32, cell_w as f32, (cell_h + TITLE) as f32, 0x141A24FF);
        draw_label(
            &mut sheet,
            &tile.label,
            x as f32 + 10.0,
            y as f32 + 7.0,
            12.0,
            0xFFFFFFFF,
            None,
        );
        let fit_max_w = cell_w - 20;
        let fit_max_h = cell_h - 16;
        let s = (fit_max_w as f32 / shot.image.width.max(1) as f32)
            .min(fit_max_h as f32 / shot.image.height.max(1) as f32)
            .min(1.0);
        let tw = (shot.image.width as f32 * s).round().max(1.0) as u32;
        let th = (shot.image.height as f32 * s).round().max(1.0) as u32;
        let thumb = shot.image.resize(tw, th);
        let dx = x as i32 + ((cell_w - tw) / 2) as i32;
        let dy = y as i32 + TITLE as i32 + ((cell_h - th) / 2) as i32;
        sheet.blit(&thumb, dx, dy);

        if merged.renderer.is_empty() {
            merged.renderer = shot.audit.renderer.clone();
        }
        merged.summary.errors += shot.audit.summary.errors;
        merged.summary.warnings += shot.audit.summary.warnings;
        merged.summary.interactive_nodes += shot.audit.summary.interactive_nodes;
        merged.perf.quads += shot.audit.perf.quads;
        merged.perf.ribbons += shot.audit.perf.ribbons;
        merged.perf.strokes += shot.audit.perf.strokes;
        merged.perf.glyphs += shot.audit.perf.glyphs;
        merged.perf.sprites += shot.audit.perf.sprites;
        merged.perf.render_ms += shot.audit.perf.render_ms;
        merged.issues.extend(shot.audit.issues.iter().cloned().map(|mut issue| {
            issue.detail = format!("[{}] {}", tile.label, issue.detail);
            issue
        }));
        tile_captures.push(TileCapture {
            label: tile.label.clone(),
            platform: tile.host.profile.id.clone(),
            device: tile.host.device.id.clone(),
            width: shot.image.width,
            height: shot.image.height,
            audit: shot.audit.clone(),
        });
    }

    // Keep at least one row even if a caller passes fewer tiles.
    if height == 0 {
        height = PAD;
    }
    let out = opts
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(".sofu/screenshots/filmstrip.png"));
    sheet.save_png(&out)?;
    merged.capture_png = Some(absolute(&out));
    merged.viewport_px = [width as f32, height as f32];
    let captured = Captured {
        ok: !merged.has_errors(),
        capture_png: Some(absolute(&out)),
        mode: CaptureMode::Filmstrip.as_str().to_string(),
        platform: "filmstrip".to_string(),
        device: format!("{n} tiles"),
        width,
        height,
        renderer: rendered
            .first()
            .map(|(_, s)| s.renderer)
            .unwrap_or("software")
            .to_string(),
        capture_ms: start.elapsed().as_secs_f64() * 1000.0,
        audit: merged,
        tiles: tile_captures,
        image: sheet,
    };
    if let Some(path) = &opts.audit_json {
        captured.write_audit_json(path)?;
    }
    Ok(captured)
}

/// Execute a serialized render job (used by the CLI and the core bridge).
pub fn run_job(job: RenderJob) -> Result<Captured, String> {
    let mut opts = ShotOptions::new(job.mode);
    opts.out = job.out.as_ref().map(PathBuf::from);
    opts.audit_json = job.audit_json.as_ref().map(PathBuf::from);
    opts.scale_limit = job.scale_limit;
    opts.scrolls = job.scrolls.clone();
    if job.mode == CaptureMode::Filmstrip {
        if job.tiles.is_empty() {
            // A filmstrip with one implicit tile is still useful (labeled
            // device shot) but a real matrix should pass `tiles`.
            let app = job
                .app
                .ok_or_else(|| "filmstrip job needs `tiles` or an `app` scene".to_string())?;
            let host = EmulatorHost::new(job.config.clone())?;
            let label = format!("{} - {}", host.profile.name, host.device.name);
            let tile = FilmstripTile {
                label,
                host,
                app,
                patches: job.patches.clone(),
            };
            return capture_filmstrip(&[tile], &opts);
        }
        let mut tiles = Vec::with_capacity(job.tiles.len());
        for tile in job.tiles {
            let host = EmulatorHost::new(tile.config)?;
            let label = if tile.label.is_empty() {
                format!("{} - {}", host.profile.name, host.device.name)
            } else {
                tile.label
            };
            tiles.push(FilmstripTile {
                label,
                host,
                app: tile.app,
                patches: tile.patches,
            });
        }
        return capture_filmstrip(&tiles, &opts);
    }
    let app = job
        .app
        .ok_or_else(|| "render job is missing `app` scene".to_string())?;
    let host = EmulatorHost::new(job.config)?;
    capture_patched(&host, app, &job.patches, &opts)
}

/// Resolve a profile + device pair without building a host (CLI helpers).
pub fn device_name(platform: &str, device: Option<&str>) -> Result<(String, String), String> {
    let (p, d) = resolve(platform, device)?;
    Ok((p.id, d.id))
}

fn settle(engine: &mut UiEngine, dl: &mut DrawList) -> (u32, bool) {
    let mut frames = 0;
    loop {
        let _ = engine.tick_and_compile(1.0 / 60.0, dl);
        frames += 1;
        if !engine.is_dirty_or_animating() || frames >= 240 {
            break;
        }
    }
    (frames, !engine.is_dirty_or_animating())
}

fn render_overlay_part(
    host: &EmulatorHost,
    part: sofuu_ui::Node,
    size: [f32; 2],
    scale: f32,
) -> Option<RgbaImage> {
    let root = sofuu_ui::ir::Node::new(
        "emu_overlay_root",
        sofuu_ui::NodeKind::Container {
            axis: sofuu_ui::Axis::Column,
            align: sofuu_ui::Align::Stretch,
            gap: 0.0,
            fold_width: None,
            scrollable: false,
            style: None,
            children: vec![part],
        },
    )
    .with_size(
        sofuu_ui::SizeSpec::Px(size[0]),
        sofuu_ui::SizeSpec::Px(size[1]),
    );
    let scene = Scene::new("emu_overlay", host.theme, root);
    let mut engine = UiEngine::new(scene, size);
    let mut dl = DrawList::new();
    let _ = settle(&mut engine, &mut dl);
    /* Overlay parts are composited over the device frame, so the scene's
     * full-viewport clear quad must not paint the theme background. */
    if let Some(q) = dl.quads.first_mut() {
        if q.rect[0] == 0.0 && q.rect[1] == 0.0 && q.rect[2] >= size[0] && q.rect[3] >= size[1] {
            q.color_top = 0;
            q.color_bottom = 0;
        }
    }
    let (img, _) = rasterize(
        &dl,
        &engine.atlas,
        size,
        scale,
        crate::image::TRANSPARENT,
    );
    Some(img)
}

fn pick_scale(viewport: [f32; 2], preferred: f32, limit: Option<f32>) -> f32 {
    let mut scale = preferred.max(1.0);
    if let Some(limit) = limit {
        scale = scale.min(limit.max(0.5));
    }
    // Keep 6-tile contact sheets and store uploads bounded.
    let max_dim = 4096.0f32;
    let by_w = max_dim / viewport[0].max(1.0);
    let by_h = max_dim / viewport[1].max(1.0);
    scale.min(by_w).min(by_h).max(0.5)
}

fn mode_name(mode: ThemeMode) -> String {
    match mode {
        ThemeMode::Dark => "dark".to_string(),
        _ => "light".to_string(),
    }
}

fn default_out(host: &EmulatorHost, mode: CaptureMode) -> PathBuf {
    PathBuf::from(format!(
        ".sofu/screenshots/{}-{}-{}.png",
        host.profile.id,
        host.device.id,
        mode.as_str()
    ))
}

fn absolute(path: &Path) -> String {
    if path.is_absolute() {
        path.display().to_string()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path).display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    }
}

fn unpack(c: u32) -> [u8; 4] {
    sofuu_ui::draw::unpack_rgba8(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::ir::{Align, Axis, BoxStyle, Node, NodeKind, SizeSpec};

    fn app_scene() -> Scene {
        let mut kids = Vec::new();
        kids.push(
            Node::new(
                "heading",
                NodeKind::Text {
                    content: "Checkout".to_string(),
                    size: 24.0,
                    color: None,
                    mono: false,
                    wrap: false,
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(40.0)),
        );
        for i in 0..4 {
            kids.push(
                Node::new(
                    &format!("row_{i}"),
                    NodeKind::Surface {
                        style: BoxStyle {
                            bg: Some(0xFF222222),
                            radius: [10.0; 4],
                            ..BoxStyle::default()
                        },
                        interactive: true,
                        child: None,
                    },
                )
                .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(56.0)),
            );
        }
        kids.push(
            Node::new(
                "submit_btn",
                NodeKind::Button {
                    label: "Submit".to_string(),
                    tone: sofuu_ui::Tone::Info,
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(48.0)),
        );
        Scene::new(
            "test",
            ThemeMode::Light,
            Node::new(
                "root",
                NodeKind::Container {
                    axis: Axis::Column,
                    align: Align::Stretch,
                    gap: 12.0,
                    fold_width: None,
                    scrollable: false,
                    style: None,
                    children: kids,
                },
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
        )
    }

    #[test]
    fn clean_capture_writes_png_and_audit() {
        let dir = tempdir();
        let host = EmulatorHost::new(EmulatorConfig {
            platform: "ios".to_string(),
            theme: ThemeMode::Light,
            ..EmulatorConfig::default()
        })
        .unwrap();
        let mut opts = ShotOptions::new(CaptureMode::Clean);
        opts.out = Some(dir.join("clean.png"));
        opts.audit_json = Some(dir.join("clean.json"));
        opts.scale_limit = Some(1.0);
        let cap = capture(&host, app_scene(), &opts).expect("capture");
        assert_eq!(cap.mode, "clean");
        assert!(cap.image.width >= 300);
        let path = cap.capture_png.clone().unwrap();
        assert!(std::path::Path::new(&path).exists());
        assert!(dir.join("clean.json").exists());
        assert_eq!(cap.audit.platform, "ios");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn annotated_capture_draws_overlay_pixels() {
        let dir = tempdir();
        let host = EmulatorHost::new(EmulatorConfig {
            platform: "ios".to_string(),
            ..EmulatorConfig::default()
        })
        .unwrap();
        let clean = render_frame(&host, app_scene(), CaptureMode::Clean, Some(1.0)).unwrap();
        let annotated = render_frame(&host, app_scene(), CaptureMode::Annotated, Some(1.0)).unwrap();
        let mut img = annotated.image.clone();
        annotate_audit(
            &mut img,
            &annotated.audit,
            &host.audit_rules(false),
            host.app_origin(),
            host.app_size(false),
            annotated.scale,
        );
        let diff = img
            .pixels
            .iter()
            .zip(clean.image.pixels.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diff > 100, "annotation changed only {diff} bytes");
        let mut opts = ShotOptions::new(CaptureMode::Annotated);
        opts.out = Some(dir.join("annotated.png"));
        opts.scale_limit = Some(1.0);
        let cap = capture(&host, app_scene(), &opts).unwrap();
        assert!(cap.audit.nodes.iter().any(|n| n.interactive));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn filmstrip_stitches_labeled_tiles() {
        let dir = tempdir();
        let ios = EmulatorHost::new(EmulatorConfig {
            platform: "ios".to_string(),
            ..EmulatorConfig::default()
        })
        .unwrap();
        let android = EmulatorHost::new(EmulatorConfig {
            platform: "android".to_string(),
            ..EmulatorConfig::default()
        })
        .unwrap();
        let tiles = vec![
            FilmstripTile {
                label: "iOS Light".to_string(),
                host: ios,
                app: app_scene(),
                patches: Vec::new(),
            },
            FilmstripTile {
                label: "Android Dark".to_string(),
                host: android,
                app: app_scene(),
                patches: Vec::new(),
            },
        ];
        let mut opts = ShotOptions::new(CaptureMode::Filmstrip);
        opts.out = Some(dir.join("strip.png"));
        let cap = capture_filmstrip(&tiles, &opts).unwrap();
        assert_eq!(cap.tiles.len(), 2);
        assert!(cap.image.width > 600);
        assert!(cap.image.height > 600);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn capture_is_device_pixels_and_has_no_strip() {
        let host = EmulatorHost::new(EmulatorConfig {
            platform: "android".to_string(),
            device: Some("sofuu-phone-411".to_string()),
            theme: ThemeMode::Light,
            ..EmulatorConfig::default()
        })
        .unwrap();
        let g = host.geometry();
        let shot = render_frame(&host, app_scene(), CaptureMode::Clean, None).unwrap();
        assert_eq!(
            shot.image.width,
            (g.width * g.scale).round() as u32,
            "capture width must be device points x scale"
        );
        assert_eq!(
            shot.image.height,
            (g.height * g.scale).round() as u32,
            "capture height must be device points x scale"
        );
        // The control strip is native host chrome and never enters the scene.
        let wrapped = host.wrap_device_scene(app_scene(), false);
        let json = serde_json::to_string(&wrapped).unwrap();
        assert!(!json.contains("emu_strip"));
        assert!(!json.contains("emu_toolbar"));
        assert!(!json.contains("emu_inspector"));
    }

    #[test]
    fn job_roundtrips_json() {
        let job = RenderJob {
            app: Some(app_scene()),
            mode: CaptureMode::Filmstrip,
            config: EmulatorConfig::default(),
            out: None,
            audit_json: None,
            scale_limit: Some(1.0),
            patches: Vec::new(),
            scrolls: Vec::new(),
            fail_on_error: false,
            tiles: Vec::new(),
        };
        let json = job.to_json();
        let back = RenderJob::from_json(&json).unwrap();
        assert_eq!(back.mode, CaptureMode::Filmstrip);
        assert!(back.app.is_some());
    }

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "sofuu-emu-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
