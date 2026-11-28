//! `sofuu-emu` — zero-OS UI emulator shell, geometric audit and AI-agent
//! screenshot engine.
//!
//! The crate is deliberately standalone: it links `sofuu-ui` only, never a
//! windowing toolkit or a GPU backend. That lets `sofuu-core` shell out to the
//! companion `sofuu-emu` binary for headless captures without taking a static
//! dependency on the GPU stack.
//!
//! | module | responsibility |
//! | :--- | :--- |
//! | [`profiles`] | five platform profiles + 19 device presets |
//! | [`host`] | device chrome around an isolated app viewport |
//! | [`scenarios`] | backend-free scenario fixtures |
//! | [`audit`](sofuu_ui::audit) | geometry/contrast violations (`sofuu-ui`) |
//! | [`raster`] | deterministic software rasterizer for `DrawList` |
//! | [`annotate`] | numbered boxes, violation callouts, labels |
//! | [`screenshot`] | `clean` / `annotated` / `filmstrip` PNG + audit JSON |

pub mod annotate;
pub mod gpu;
pub mod host;
pub mod image;
pub mod profiles;
pub mod raster;
pub mod scenarios;
pub mod screenshot;

pub use host::{EmulatorConfig, EmulatorHost, ShellAction};
pub use image::RgbaImage;
pub use profiles::{
    profiles, resolve, ChromeStyle, DeviceGeometry, DevicePreset, NavBar, Orientation,
    PlatformProfile, PunchHole,
};
pub use raster::{rasterize, rasterize_into, RasterStats, Transform};
pub use scenarios::{Scenario, ScenarioSet};
pub use screenshot::{
    capture, capture_filmstrip, capture_patched, render_frame, render_frame_patched, run_job,
    CaptureMode, Captured, FilmstripTile, FrameShot, RenderJob, RenderTileJob, ScrollAction,
    ShotOptions,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// JSON manifest of every profile, device preset and audit default.
///
/// Used by `sofuu-emu profiles` and by MCP tool discovery so agents can pick a
/// valid `platform` + `device` pair without guessing.
pub fn profiles_json() -> String {
    #[derive(serde::Serialize)]
    struct Manifest {
        version: &'static str,
        profiles: Vec<PlatformProfile>,
    }
    serde_json::to_string_pretty(&Manifest {
        version: VERSION,
        profiles: profiles(),
    })
    .unwrap_or_else(|_| "{}".to_string())
}

/// Render a serialized [`RenderJob`] and return the serialized [`Captured`]
/// metadata. Convenience for FFI/dynamic bridges.
pub fn render_job_json(job_json: &str) -> Result<String, String> {
    let job = RenderJob::from_json(job_json)?;
    let captured = run_job(job)?;
    Ok(captured.to_json())
}
