# `sofuu-gpu` — Native window + GPU backends

Direct, dependency-light windowing and 2D GPU compositing for the Sofuu UI. One shared
`UiEngine` loop (`src/loop_core.rs`) drives every platform through the
`NativeWindowBackend` trait (`src/backend/mod.rs`); the scene, springs and input semantics are
identical everywhere.

No `winit`, no `wgpu`, no Skia/ANGLE/WebView. The OS window and GPU API are reached through a
small C shim compiled per target plus dynamically resolved entry points, so the crate has no
hard link-time dependency on Wayland, X11, EGL, GLES, GL, Vulkan or Win32 window libraries.

## Backends

| Target | Window / context | Default renderer | Optional renderer |
| :--- | :--- | :--- | :--- |
| macOS | AppKit `NSWindow` + `CAMetalLayer` | Metal | — |
| Linux / BSD | Wayland `xdg_wm_base`, X11 fallback; EGL | GLES 3.0, then desktop GL 3.3 | Vulkan 1.1 via `SOFUU_UI_BACKEND=vulkan` |
| Windows | Win32 `HWND`; WGL | desktop GL 3.3 Core | Vulkan 1.1 via `SOFUU_UI_BACKEND=vulkan` |
| Android | host-attached `ANativeWindow`; EGL | GLES 3.0 | Vulkan 1.1 via `SOFUU_UI_BACKEND=vulkan` |

The GL/GLES renderer is one Rust implementation (`src/backend/gles.rs`) with dynamically loaded
entry points. It uses a shared 4-vertex corner quad, four instanced pipelines, streaming instance
VBOs, an `R8` glyph atlas, and a `std140` clip UBO capped at 256 `vec4` clip rectangles. Scenes
with more clips are split into ordered clip windows automatically.

Vulkan (`src/backend/vulkan.rs`) is a raw dynamic `libvulkan` / `vulkan-1.dll` loader. Instances
and clips are `std430` storage buffers; per-draw `base_instance` travels in a 32-byte push
constant, so no `shaderDrawParameters` feature is required. Precompiled SPIR-V modules are
committed under `src/shaders/spirv/`; builds never invoke `glslc`/`shaderc`.

If Vulkan initialization fails (missing loader or ICD, unsupported surface, no suitable queue),
the backend logs the reason and falls back to GL/GLES without aborting.

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `SOFUU_UI_BACKEND=vulkan` | Try the direct Vulkan renderer first (Linux/Windows/Android). Falls back to GL/GLES on any initialization failure. |
| `SOFUU_UI_SCALE=1.5` | Override the compositor/DPI scale on Linux/X11 (accepted range `0.5..=4.0`). |
| `SOFUU_UI_FULLSCREEN=1` | Kiosk/fullscreen mode: Wayland `xdg_toplevel.set_fullscreen`, X11 `_NET_WM_STATE_FULLSCREEN` plus a screen-sized fallback, Win32 borderless monitor fullscreen. Android is always fullscreen. |
| `SOFUU_GPU_SKIP_NATIVE=1` | Build-time escape hatch: skip the native C shims and compile only Rust. Useful for cross-`cargo check` on a host without a target C toolchain; the resulting library cannot open a window. |

`SOFUU_UI_DRM=1` direct DRM/KMS scanout is **not implemented**. Use `SOFUU_UI_FULLSCREEN=1`
under a Wayland/X11 kiosk compositor for robotics/edge dashboards.

### Known limitations

- **X11 smooth scrolling:** the fallback translates core button `4`–`7` wheel events; XInput2
  valuators are not queried, so two-finger touchpad scrolling depends on the X server's button
  emulation.
- **Android touch scrolling:** drags arrive as `PointerDown`/`PointerMove`/`PointerUp`; scroll
  containers consume the host's explicit `sofuu_ui_android_push_scroll` deltas.
- **Vulkan** is opt-in (`SOFUU_UI_BACKEND=vulkan`) and has not been exercised against a real ICD
  on this development host. Any initialization failure falls back to GL/GLES.

## Threading and pacing

- Thread 0 owns the window, EGL/WGL/Vulkan context and every GPU object.
- Threads 1+ push `ScenePatch` values through `UiHandle`; `push_patch` sends over `mpsc` and
  calls the platform wake function (`eventfd`/pipe write, `PostMessageW(WM_APP+1)`,
  `sofuu_macos_wake`) to interrupt the OS wait immediately.
- While springs, kinetic scroll or pending patches keep the scene dirty, Thread 0 polls with a
  `<= 1/120 s` deadline. Once settled it blocks indefinitely in `poll()` /
  `MsgWaitForMultipleObjectsEx` / `nextEventMatchingMask` — 0% CPU/GPU at idle.
- DPI changes arrive as `Resize` events and update `FrameUniforms.scale_factor`, so a window
  dragged between 100%/150%/200% monitors stays sharp without restarting.

## Building

The Rust crate is `cdylib` + `rlib`. `build.rs` compiles exactly one native shim per target:

- macOS: `native/macos_metal.m` (Objective-C ARC), AppKit/Metal/QuartzCore frameworks.
- Linux/BSD: `native/linux_wayland_egl.c`, links only `libdl`/`libm`; every display/GPU library
  is `dlopen`ed at runtime.
- Windows: `native/windows_win32_gl.c`, links `user32`, `gdi32`, `opengl32`; `dwmapi.dll` is
  loaded dynamically.
- Android: `native/android_egl.c`; the NDK toolchain is auto-detected from `ANDROID_NDK_HOME`,
  `ANDROID_NDK_ROOT`, `ANDROID_HOME/ndk`, `~/Library/Android/sdk/ndk` or `~/Android/Sdk/ndk`.

On Debian/Ubuntu the runtime packages are typically:

```sh
sudo apt install libwayland-client0 libwayland-egl1 libegl1 libgles2 libgl1 \
                 libxkbcommon0 libx11-6
```

No `-dev` packages are needed to build or run; the C shim declares the ABI it uses.

## Android host integration

Android cannot create an Activity on its own. The host attaches an `ANativeWindow` before
opening the UI:

```c
sofuu_ui_android_attach_surface(window, width, height, density_scale);
sofuu_ui_run_json(/* ... */);                       // Thread 0
sofuu_ui_android_push_motion(action, x, y);         // host input thread
sofuu_ui_android_push_scroll(x, y, dx, dy);
sofuu_ui_android_push_key(keycode, modifiers);  // modifiers = SOFUU_MOD_* bits
sofuu_ui_android_push_text("hello");
sofuu_ui_android_push_close();
```

Key events on every platform carry the `SOFUU_MOD_*` bitmask (`Shift = 1`, `Ctrl = 2`, `Alt = 4`,
`Meta = 8`). Linux derives it from `xkb_state` / the X11 state mask, Windows from
`GetKeyState`, Android from the `modifiers` argument above.

`sofuu_ui_android_surface_changed` re-attaches a new surface after rotation / `SurfaceView`
recreation. The `UiEngine` scene, springs and scroll offsets survive; only GPU surface resources
are rebuilt. With Vulkan selected, the `VkSurfaceKHR` is recreated against the new
`ANativeWindow`.

## Shaders

The Metal sources are the reference. GL/GLES and Vulkan keep the same geometry, SDF and
premultiplied-alpha math:

- `src/shaders/msl/ui_shaders.metal` — Metal reference kernels.
- `src/shaders/glsl/ui_shaders_330.glsl` — desktop OpenGL 3.3 Core (one source compiled four
  times with `SOFUU_PIPELINE_*` / `SOFUU_VERTEX|FRAGMENT` defines).
- `src/shaders/glsl/ui_shaders_es3.glsl` — OpenGL ES 3.0.
- `src/shaders/glsl/ui_shaders_450.glsl` + `src/shaders/spirv/*.spv` — Vulkan 1.1.
- `src/shaders/generate_spirv.sh` — regenerate the committed SPIR-V modules (requires
  `glslangValidator`; not used by normal builds).

Blending is premultiplied alpha on every API:
`glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)` and the
Metal equivalent `sourceRGB = One`, `destinationRGB = OneMinusSourceAlpha`. Framebuffers are
non-sRGB UNORM (`BGRA8Unorm` on Metal, `RGBA8`/`BGRA8` UNORM elsewhere) so theme colors match
bit-for-bit.

## Tests and verification

Run:

```sh
cargo test -p sofuu-gpu
cargo check -p sofuu-gpu --examples
```

Unit tests cover shader source/entry-stage generation, SPIR-V headers and entry-point stages,
push-constant/SSBO struct layouts, descriptor bindings, `base_instance` presence, and GL clip
window partitioning/remapping.

`tests/gl330_offscreen.rs` is a real GPU execution test: it creates a macOS CGL Core Profile
context, compiles all four GLSL 330 pipelines on the driver, renders into an `RGBA8` FBO and
checks solid fill, clipping, >256-clip window rebasing and premultiplied blending by reading
pixels back. macOS implements OpenGL on top of Metal and needs the `MTLCompiler` XPC service;
sandboxed/background agent sessions cannot reach it and the driver aborts the process, so the
test is opt-in and must be run from a normal GUI login session:

```sh
SOFUU_GPU_GL_OFFSCREEN=1 cargo test -p sofuu-gpu --test gl330_offscreen -- --nocapture
```

Cross-target checks used during development (with the native shims compiled):

```sh
cargo check -p sofuu-gpu --target x86_64-unknown-linux-gnu   # zig cc wrappers
cargo check -p sofuu-gpu --target x86_64-pc-windows-gnu      # MinGW
cargo check -p sofuu-gpu --target aarch64-linux-android      # NDK autodetect
```

### Verification matrix (2026-11-27)

| Area | Status |
| :--- | :--- |
| macOS Metal `cargo check` / `cargo test` | Verified on the development host (14 unit tests + 1 opt-in offscreen test). |
| GLSL 330 / ES3 syntax | Validated with `glslangValidator` (ES3 validated as a temporary `#version 310 es` because glslang's SPIR-V front end does not accept 300 es). |
| SPIR-V modules | All 8 committed modules re-generated byte-identically and validated for magic, stage and bindings. |
| Linux Wayland/X11 + EGL runtime | **Unverified** — cross-compiled and syntax-checked only; no Linux display/GPU on the development host. |
| Windows Win32/WGL runtime | **Unverified** — cross-compiled and syntax-checked only. |
| Android EGL/GLES runtime | **Unverified** — cross-compiled and syntax-checked only; no device/emulator run. |
| Vulkan runtime (all platforms) | **Unverified** — loader/WSI/SPIR-V paths compile and are unit-tested statically; no Vulkan ICD exercised on the development host. |
| GL 3.3 renderer runtime (offscreen) | **Harness added, not executed here** — `tests/gl330_offscreen.rs` compiles the pipelines and checks geometry/clips/blending via CGL on macOS. This sandbox has no WindowServer/`MTLCompiler` access, so the driver aborts when asked to compile GLSL; run with `SOFUU_GPU_GL_OFFSCREEN=1` from a GUI session. |
| Metal ↔ GL/GLES offscreen pixel diff < 1 LSB | **Not run** — the GL side now has a CGL harness, but a Metal offscreen reference is still required; the Metal window shim is owned by another in-flight refactor. |
| 0% idle CPU / < 50 ms first frame / 100–200% DPI | **Unverified at runtime** — design and code paths are in place, but require physical Linux/Windows/Android hardware. |

No runtime result in this table should be read as validated on hardware that was not available.
