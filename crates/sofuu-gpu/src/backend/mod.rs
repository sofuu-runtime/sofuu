//! Platform window/GPU backend abstraction for [`crate::run_window`].
//!
//! The shared 2-thread scheduler in [`crate::loop_core`] is completely independent of the
//! window system and graphics API. Each backend implements [`NativeWindowBackend`] once and
//! inherits the same adaptive 120 Hz / blocking-idle pacing, patch pumping and input routing.

use sofuu_ui::{DrawList, FrameUniforms, GlyphAtlas, InputEvent};

use crate::WindowConfig;

#[cfg(target_os = "macos")]
pub mod macos_metal;

#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "android")]
pub mod android;

pub mod gles;

// The Vulkan module is pure Rust (libvulkan is resolved at runtime), so unit tests can build it
// on any host even though only Linux/Windows/Android select it as a window backend.
#[cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "android",
    target_os = "freebsd",
    target_os = "openbsd",
    test
))]
pub mod vulkan;

/// Window + swapchain plus input-event pumping contract implemented by every OS/GPU backend.
pub trait NativeWindowBackend {
    /// Current logical size and DPI scale factor `(w_px, h_px, scale)`.
    fn metrics(&self) -> (f32, f32, f32);

    /// Block Thread 0 efficiently inside the OS event loop until input arrives OR
    /// `timeout_sec` elapses (or indefinitely when `wait_forever == true`).
    ///
    /// Must return early immediately when [`wake_main_loop`] is invoked from Thread 1+.
    /// Returns `false` once the native window has closed.
    fn poll_or_wait(
        &mut self,
        wait_forever: bool,
        timeout_sec: f64,
        out_events: &mut Vec<InputEvent>,
    ) -> bool;

    /// Upload the frame's [`DrawList`] slices and present the swapchain frame.
    fn render(&mut self, uniforms: &FrameUniforms, dl: &DrawList);

    /// Reserved top safe-area inset (macOS frameless traffic lights); `0.0` elsewhere.
    fn top_inset(&self) -> f32 {
        0.0
    }

    /// Synchronize the native title bar / chrome with the active [`sofuu_ui::ThemeMode`].
    fn set_theme_dark(&mut self, _dark: bool) {}

    /// Human-readable backend name (`"metal"`, `"gles"`, `"gl"`, `"vulkan"`).
    fn backend_name(&self) -> &'static str;

    /// True after a swapchain/surface loss that the renderer must recover from.
    fn context_lost(&self) -> bool {
        false
    }

    /// Rebuild swapchain/surface resources after [`Self::context_lost`].
    fn recover_context(&mut self) {}
}

/// Normalize the small platform key-code enum emitted by the Linux/Windows/Android shims into
/// the semantic names understood by `UiEngine` (`Backspace`, `Enter`, `Escape`, arrows, ...).
/// Codes `>= 1000` are raw platform key codes and are surfaced as `keycode:N`.
#[allow(dead_code)] // Used by the Linux/Windows/Android backends, not the macOS build.
pub(crate) fn key_name(code: u32) -> String {
    match code {
        1 => "Backspace".to_string(),
        2 => "Enter".to_string(),
        3 => "Escape".to_string(),
        4 => "ArrowLeft".to_string(),
        5 => "ArrowRight".to_string(),
        6 => "ArrowUp".to_string(),
        7 => "ArrowDown".to_string(),
        8 => "Home".to_string(),
        9 => "End".to_string(),
        10 => "Tab".to_string(),
        11 => "Delete".to_string(),
        n if n >= 1000 => format!("keycode:{}", n - 1000),
        n => format!("keycode:{n}"),
    }
}

/// Wake Thread 0 from its OS-level blocking wait. Safe to call from any thread.
pub fn wake_main_loop() {
    #[cfg(target_os = "macos")]
    {
        macos_metal::wake_main_loop();
    }
    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
    {
        linux::wake_main_loop();
    }
    #[cfg(target_os = "windows")]
    {
        windows::wake_main_loop();
    }
    #[cfg(target_os = "android")]
    {
        android::wake_main_loop();
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        // Unsupported target: `create_window` returns a structured error before this is reachable.
    }
}

/// Create the native window and GPU renderer for the current target.
pub fn create_window(
    config: &WindowConfig,
    atlas: &GlyphAtlas,
) -> Result<Box<dyn NativeWindowBackend>, String> {
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos_metal::MacosMetalBackend::new(config, atlas)?))
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
    {
        Ok(Box::new(linux::LinuxBackend::new(config, atlas)?))
    }

    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(windows::WindowsBackend::new(config, atlas)?))
    }

    #[cfg(target_os = "android")]
    {
        Ok(Box::new(android::AndroidBackend::new(config, atlas)?))
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        let _ = (config, atlas);
        Err(
            "E_UI_NO_DISPLAY: no native window backend is implemented for this target; \
             omit --ui to run headless"
                .to_string(),
        )
    }
}

/// Name of the compile-time backend family (for diagnostics and the `sofuu ui status` surface).
pub const fn compiled_backend_family() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "metal"
    }
    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
    {
        "wayland/x11 + egl"
    }
    #[cfg(target_os = "windows")]
    {
        "win32 + wgl"
    }
    #[cfg(target_os = "android")]
    {
        "android + egl"
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        "unsupported"
    }
}
