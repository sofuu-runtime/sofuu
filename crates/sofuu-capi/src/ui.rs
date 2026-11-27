// sofuu-capi::ui — Embedded Native Host UI C-ABI (PLAN-EMBEDDED-NATIVE-HOST-C-ABI).
//
// `libsofuu` gains an additive `sofuu_ui_*` surface (Tier 2 amendment) so a
// native host can embed `sofuu-ui`'s 120Hz compositor inside its OWN window
// with zero WebView and zero duplicate UI code:
//
//   Mode A (always): the host owns the GPU device and pulls a zero-copy
//     `SofuuUiDrawList` on every display-link tick.
//   Mode B (feature `ui-surface`): Sofuu draws onto a host layer; the surface
//     backend lands with PLAN-CROSS-PLATFORM-GPU-WINDOW-BACKENDS, so those
//     exports return SOFUU_ERR_NOT_IMPL until then.
//
// Threading: the render core (UiEngine + DrawList) is behind one Mutex and is
// touched only by `sofuu_ui_poll_draw_list` (host UI thread). Input, scene
// patches and semantic events live in their own short-held queues, so the
// Runtime thread (Thread R, where QuickJS lives) never blocks the render
// thread and vice versa. The wake callback is copied out of its lock before it
// runs, so a host callback may safely re-enter this module.
//
// Tombstone safety: `sofuu_ui_attach_runtime` stores a `Weak<UiInner>` keyed by
// the QuickJS context pointer. If the host frees the session before the
// runtime (or vice versa), `Weak::upgrade` fails and the JS globals answer
// `-2` / `false` instead of touching freed memory.

#![allow(clippy::missing_safety_doc)]

use std::collections::{HashMap, VecDeque};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::AssertUnwindSafe;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use serde_json::{Map as JsonMap, Value as JsonValue};

use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSContext, JSValue, JSValueConst,
};
use sofuu_ffi::qjs;
use sofuu_ui::auto_compose::{AutoAgentEvent, AutoComposer, AutoTriggerStat, PendingApproval};
use sofuu_ui::components::table::TableRow;
use sofuu_ui::draw::{ClipRect, DrawCmd, DrawList, GlyphInstance, QuadInstance, RibbonInstance, StrokeInstance};
use sofuu_ui::hit_test::{InputEvent, UIEvent};
use sofuu_ui::ir::ScenePatch;
use sofuu_ui::theme::ThemeMode;
use sofuu_ui::ui as kit;
use sofuu_ui::UiEngine;

use crate::CapiRuntime;

// ── Return codes (mirror include/sofuu_embed.h) ───────────────────

const OK: c_int = 0;
const ERR_INVALID_ARG: c_int = -1;
const ERR_NOT_IMPL: c_int = -2;
const ERR_INTERNAL: c_int = -6;
const ERR_PANIC: c_int = -10;

/// Bounded semantic-event backlog. A host that never pumps (or a JS pump that
/// has stalled) must not grow without bound; the oldest events are dropped.
const MAX_EVENTS: usize = 4096;

// ── C ABI POD types (byte-identical to sofuu-ui::draw) ────────────
//
// The instance structs are aliases, not copies: the compiler then guarantees
// parity instead of trusting two duplicating definitions to stay in sync.

pub type SofuuUiClipRect = ClipRect;
pub type SofuuUiDrawCmd = DrawCmd;
pub type SofuuUiQuadInstance = QuadInstance;
pub type SofuuUiRibbonInstance = RibbonInstance;
pub type SofuuUiStrokeInstance = StrokeInstance;
pub type SofuuUiGlyphInstance = GlyphInstance;

/// 32-byte per-frame GPU uniform block (`include/sofuu_embed.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SofuuUiFrameUniforms {
    /// Logical viewport `[width, height]`.
    pub viewport_size: [f32; 2],
    /// Atlas dimensions `[width, height]` in pixels.
    pub atlas_size: [f32; 2],
    /// Display DPI scale factor (`1.0`, `2.0` Retina, etc.).
    pub scale_factor: f32,
    /// Elapsed animation time in seconds.
    pub time_sec: f32,
    pub _pad: [f32; 2],
}

impl Default for SofuuUiFrameUniforms {
    fn default() -> Self {
        Self {
            viewport_size: [0.0; 2],
            atlas_size: [0.0; 2],
            scale_factor: 1.0,
            time_sec: 0.0,
            _pad: [0.0; 2],
        }
    }
}

/// One frame of zero-copy draw data. Every pointer borrows the session's
/// double-buffered `DrawList`; none of them is owned by the host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SofuuUiDrawList {
    pub uniforms: SofuuUiFrameUniforms,
    pub quads: *const SofuuUiQuadInstance,
    pub quad_count: u32,
    pub ribbons: *const SofuuUiRibbonInstance,
    pub ribbon_count: u32,
    pub strokes: *const SofuuUiStrokeInstance,
    pub stroke_count: u32,
    pub glyphs: *const SofuuUiGlyphInstance,
    pub glyph_count: u32,
    pub cmds: *const SofuuUiDrawCmd,
    pub cmd_count: u32,
    pub clips: *const SofuuUiClipRect,
    pub clip_count: u32,
    pub recompiled: u8,
    pub needs_next_frame: u8,
    pub theme_is_dark: u8,
    pub _reserved: u8,
}

impl Default for SofuuUiDrawList {
    fn default() -> Self {
        Self {
            uniforms: SofuuUiFrameUniforms::default(),
            quads: ptr::null(),
            quad_count: 0,
            ribbons: ptr::null(),
            ribbon_count: 0,
            strokes: ptr::null(),
            stroke_count: 0,
            glyphs: ptr::null(),
            glyph_count: 0,
            cmds: ptr::null(),
            cmd_count: 0,
            clips: ptr::null(),
            clip_count: 0,
            recompiled: 0,
            needs_next_frame: 0,
            theme_is_dark: 0,
            _reserved: 0,
        }
    }
}

/// Mirrors `sofuu_ui::hit_test::InputEvent` without heap ownership escapes:
/// `text_utf8` is borrowed only for the duration of the dispatch call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SofuuUiInputEvent {
    pub kind: u32,
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub scale_factor: f32,
    pub button: u32,
    pub modifiers: u32,
    pub text_utf8: *const c_char,
}

pub const SOFUU_INPUT_POINTER_MOVE: u32 = 1;
pub const SOFUU_INPUT_POINTER_DOWN: u32 = 2;
pub const SOFUU_INPUT_POINTER_UP: u32 = 3;
pub const SOFUU_INPUT_SCROLL: u32 = 4;
pub const SOFUU_INPUT_KEY_DOWN: u32 = 5;
pub const SOFUU_INPUT_TEXT_INPUT: u32 = 6;
pub const SOFUU_INPUT_RESIZE: u32 = 7;
pub const SOFUU_INPUT_CLOSE_REQUESTED: u32 = 8;

/// Host wake hook (`sofuu_ui_wake_cb`).
pub type SofuuUiWakeCb = extern "C" fn(*mut c_void);

// ── Session internals ─────────────────────────────────────────────

/// Inbound work moved from a producer thread to the render thread. Keeping
/// patches and Auto-Composer telemetry in ONE queue preserves the order the
/// host/JS observed, which matters when a `replace_scene` is followed by a
/// telemetry event.
enum Inbound {
    Patch(ScenePatch),
    AgentEvent(AutoAgentEvent),
    Triggers(Vec<AutoTriggerStat>),
    Rows(Vec<TableRow>),
}

#[derive(Clone, Copy)]
struct WakeSlot {
    cb: SofuuUiWakeCb,
    // Raw host pointers are not Send; usize keeps the Mutex Sync while the
    // function pointer (Send + Copy) remains callable.
    opaque: usize,
}

struct RenderCore {
    engine: UiEngine,
    /// Two buffers: the host may still be encoding/reading the previous frame
    /// when the next `poll_draw_list` runs (display links are async).
    draw_lists: [DrawList; 2],
    active: usize,
    auto: Option<AutoComposer>,
    auto_title: String,
    scale_factor: f32,
    time_sec: f32,
    bottom_inset: f32,
}

struct UiInner {
    wake: Mutex<Option<WakeSlot>>,
    inbound: Mutex<VecDeque<Inbound>>,
    inputs: Mutex<VecDeque<InputEvent>>,
    events: Mutex<VecDeque<String>>,
    render: Mutex<RenderCore>,
    attached: AtomicBool,
}

impl UiInner {
    fn new(config: SessionConfig) -> Self {
        let scene = kit::app(config.title.clone()).theme(config.theme).build();
        let engine = UiEngine::new(scene, [config.width, config.height]);
        Self {
            wake: Mutex::new(None),
            inbound: Mutex::new(VecDeque::new()),
            inputs: Mutex::new(VecDeque::new()),
            events: Mutex::new(VecDeque::new()),
            render: Mutex::new(RenderCore {
                engine,
                draw_lists: [DrawList::new(), DrawList::new()],
                active: 0,
                auto: None,
                auto_title: config.auto_title,
                scale_factor: 1.0,
                time_sec: 0.0,
                bottom_inset: 0.0,
            }),
            attached: AtomicBool::new(false),
        }
    }

    /// Fire the host wake hook without holding any lock across the callback.
    fn wake(&self) {
        let slot = *lock(&self.wake);
        if let Some(slot) = slot {
            (slot.cb)(slot.opaque as *mut c_void);
        }
    }
}

/// The opaque handle C sees. The inner state is Arc-owned so the JS globals
/// (which only hold a Weak) can outlive a prematurely-freed session safely.
pub struct SofuuUiSession {
    inner: Arc<UiInner>,
}

// ── Small helpers ─────────────────────────────────────────────────

/// Poison-tolerant locking: after a contained Rust panic the session must stay
/// usable for the next frame, not cascade `PoisonError` into the host.
#[inline]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn guarded(f: impl FnOnce() -> c_int) -> c_int {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(rc) => rc,
        Err(_) => ERR_PANIC,
    }
}

unsafe fn json_arg<'a>(ptr: *const c_char, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    if len == 0 {
        CStr::from_ptr(ptr).to_str().ok()
    } else {
        std::str::from_utf8(slice::from_raw_parts(ptr as *const u8, len)).ok()
    }
}

unsafe fn text_arg(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}

fn enqueue(inner: &UiInner, item: Inbound) {
    {
        let mut q = lock(&inner.inbound);
        q.push_back(item);
    }
    inner.wake();
}

fn push_events(inner: &UiInner, events: Vec<UIEvent>) {
    if events.is_empty() {
        return;
    }
    let mut q = lock(&inner.events);
    for ev in events {
        if q.len() >= MAX_EVENTS {
            q.pop_front();
        }
        if let Ok(json) = serde_json::to_string(&ev) {
            q.push_back(json);
        }
    }
}

// ── Session configuration ─────────────────────────────────────────

struct SessionConfig {
    title: String,
    auto_title: String,
    theme: ThemeMode,
    width: f32,
    height: f32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            title: "Sofuu".to_string(),
            auto_title: "Sofuu Agent Runtime".to_string(),
            theme: ThemeMode::Light,
            width: 1024.0,
            height: 680.0,
        }
    }
}

impl SessionConfig {
    /// `config_json` is intentionally tiny and domain-agnostic: title, theme
    /// and initial logical viewport. Unknown keys are ignored so hosts can
    /// forward a shared config object.
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        let mut cfg = Self::default();
        let Some(raw) = raw.filter(|s| !s.trim().is_empty()) else {
            return Ok(cfg);
        };
        let value: JsonValue =
            serde_json::from_str(raw).map_err(|e| format!("config_json is not JSON: {e}"))?;
        let obj = value
            .as_object()
            .ok_or_else(|| "config_json must be a JSON object".to_string())?;

        if let Some(title) = obj.get("title") {
            let title = title
                .as_str()
                .ok_or_else(|| "config_json.title must be a string".to_string())?;
            cfg.title = title.to_string();
            cfg.auto_title = format!("Sofuu // {title}");
        }
        if let Some(auto_title) = obj.get("auto_title") {
            cfg.auto_title = auto_title
                .as_str()
                .ok_or_else(|| "config_json.auto_title must be a string".to_string())?
                .to_string();
        }
        if let Some(theme) = obj.get("theme") {
            cfg.theme = match theme.as_str() {
                Some("light") => ThemeMode::Light,
                Some("dark") => ThemeMode::Dark,
                _ => {
                    return Err(
                        "config_json.theme must be \"light\" or \"dark\"".to_string()
                    )
                }
            };
        }
        if let Some(width) = obj.get("width") {
            let w = width
                .as_f64()
                .ok_or_else(|| "config_json.width must be a number".to_string())?;
            if w.is_finite() && w > 0.0 {
                cfg.width = w as f32;
            }
        }
        if let Some(height) = obj.get("height") {
            let h = height
                .as_f64()
                .ok_or_else(|| "config_json.height must be a number".to_string())?;
            if h.is_finite() && h > 0.0 {
                cfg.height = h as f32;
            }
        }
        Ok(cfg)
    }
}

// ── Lifecycle & configuration ─────────────────────────────────────

/// Create a UI session. See `include/sofuu_embed.h` for the config shape.
///
/// # Safety
/// `config_json` is NUL-terminated (or NULL); `out_err_json` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_session_new(
    config_json: *const c_char,
    out_err_json: *mut *mut c_char,
) -> *mut SofuuUiSession {
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        session_new_impl(config_json, out_err_json)
    })) {
        Ok(ui) => ui,
        Err(_) => {
            if !out_err_json.is_null() {
                *out_err_json = crate::error_envelope("panic", "panic in sofuu_ui_session_new");
            }
            ptr::null_mut()
        }
    }
}

unsafe fn session_new_impl(
    config_json: *const c_char,
    out_err_json: *mut *mut c_char,
) -> *mut SofuuUiSession {
    if !out_err_json.is_null() {
        *out_err_json = ptr::null_mut();
    }
    let raw = if config_json.is_null() {
        None
    } else {
        match CStr::from_ptr(config_json).to_str() {
            Ok(s) => Some(s),
            Err(_) => {
                if !out_err_json.is_null() {
                    *out_err_json = crate::error_envelope(
                        "bad_config",
                        "config_json is not valid UTF-8",
                    );
                }
                return ptr::null_mut();
            }
        }
    };
    let config = match SessionConfig::parse(raw) {
        Ok(c) => c,
        Err(msg) => {
            if !out_err_json.is_null() {
                *out_err_json = crate::error_envelope("bad_config", &msg);
            }
            return ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(SofuuUiSession {
        inner: Arc::new(UiInner::new(config)),
    }))
}

/// Destroy a session. NULL-tolerant. After this returns, JS globals attached
/// to it answer detached (-2 / false) instead of dereferencing freed memory.
///
/// # Safety
/// `ui` must have come from `sofuu_ui_session_new` and must not be in use by
/// another thread at the moment it is freed (the usual handle contract).
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_session_free(ui: *mut SofuuUiSession) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ui.is_null() {
            return;
        }
        let session = Box::from_raw(ui);
        session.inner.attached.store(false, Ordering::Release);
        forget_session(&session.inner);
        drop(session);
    }));
}

fn forget_session(inner: &Arc<UiInner>) {
    let target = Arc::as_ptr(inner);
    let mut map = attached_map().lock().unwrap_or_else(|e| e.into_inner());
    map.retain(|_, weak| !ptr::eq(weak.as_ptr(), target));
}

/// Install/clear the host wake hook. Any thread. NULL `cb` clears it.
///
/// # Safety
/// `ui` must be a live session; `cb`/`opaque` must remain valid while set.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_session_set_wake_cb(
    ui: *mut SofuuUiSession,
    cb: Option<SofuuUiWakeCb>,
    opaque: *mut c_void,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let slot = cb.map(|cb| WakeSlot {
            cb,
            opaque: opaque as usize,
        });
        let inner = &(*ui).inner;
        *lock(&inner.wake) = slot;
        OK
    })
}

/// Reserve top safe-area clearance while keeping the background full-bleed.
///
/// # Safety
/// `ui` must be a live session.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_session_set_safe_insets(
    ui: *mut SofuuUiSession,
    top_px: f32,
    bottom_px: f32,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let top = if top_px.is_finite() { top_px.max(0.0) } else { 0.0 };
        let bottom = if bottom_px.is_finite() {
            bottom_px.max(0.0)
        } else {
            0.0
        };
        let inner = &(*ui).inner;
        let mut render = lock(&inner.render);
        render.engine.set_top_inset(top);
        render.bottom_inset = bottom;
        OK
    })
}

/// Borrow the R8 SDF atlas. The pointer is valid until session_free.
///
/// # Safety
/// `ui` must be a live session; each out pointer may be NULL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_session_get_atlas(
    ui: *mut SofuuUiSession,
    out_pixels_r8: *mut *const u8,
    out_width: *mut u32,
    out_height: *mut u32,
    out_px_range: *mut f32,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let inner = &(*ui).inner;
        let render = lock(&inner.render);
        let atlas = &render.engine.atlas;
        if !out_pixels_r8.is_null() {
            *out_pixels_r8 = atlas.pixels.as_ptr();
        }
        if !out_width.is_null() {
            *out_width = atlas.width;
        }
        if !out_height.is_null() {
            *out_height = atlas.height;
        }
        if !out_px_range.is_null() {
            *out_px_range = atlas.px_range;
        }
        OK
    })
}

// ── Runtime attachment & JS globals ───────────────────────────────

/// Per-QuickJS-context attachment table. JS native callbacks receive only the
/// context, so this is the bridge from `ctx` to the session it drives. Entries
/// are Weak: a freed session is a tombstone, never a dangling pointer.
fn attached_map() -> &'static Mutex<HashMap<usize, Weak<UiInner>>> {
    static MAP: OnceLock<Mutex<HashMap<usize, Weak<UiInner>>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn js_lookup(ctx: *mut JSContext) -> Option<Arc<UiInner>> {
    if ctx.is_null() {
        return None;
    }
    let map = attached_map().lock().unwrap_or_else(|e| e.into_inner());
    let inner = map.get(&(ctx as usize))?.upgrade()?;
    if inner.attached.load(Ordering::Acquire) {
        Some(inner)
    } else {
        None
    }
}

/// Whether this QuickJS context currently drives an embedded UI session.
///
/// The blocking C-ABI call/eval paths consult this to avoid waiting for
/// libuv to go idle: a mounted `sofuu.ui` view runs a 16 ms event pump, so
/// idle never arrives. Cheap (one short mutex); never called from a JS
/// callback while the same lock is held.
pub(crate) fn runtime_is_attached(ctx: *mut JSContext) -> bool {
    js_lookup(ctx).is_some()
}

/// Bind a session to a runtime's QuickJS context on the owning thread.
///
/// # Safety
/// `rt` must be a live runtime and this call must run on its owning thread.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_attach_runtime(
    rt: *mut CapiRuntime,
    ui: *mut SofuuUiSession,
) -> c_int {
    guarded(|| {
        if rt.is_null() || ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let ctx = (*rt).ffi_rt.engine_ctx() as *mut JSContext;
        if ctx.is_null() {
            return ERR_INTERNAL;
        }
        let session = &*ui;
        let inner = session.inner.clone();
        inner.attached.store(true, Ordering::Release);
        {
            let mut map = attached_map().lock().unwrap_or_else(|e| e.into_inner());
            map.insert(ctx as usize, Arc::downgrade(&inner));
        }
        // JS_SetPropertyStr replaces the loader-backed globals installed by
        // sofuu-core's mod_ui_register; `sofuu.ui.isAttached()` now sees THIS
        // session. The three functions are process-wide statics that resolve
        // the session through the context key, never through globals.
        register_global_fn(ctx, "__sofuu_ui_is_attached", js_ui_is_attached);
        register_global_fn(ctx, "__sofuu_ui_push_patch", js_ui_push_patch);
        register_global_fn(ctx, "__sofuu_ui_poll_event", js_ui_poll_event);
        // Zero-boilerplate Auto-Composer (§6): ui.js's autoComposer() calls
        // these five. Installing them here means an embedded pure-JS agent
        // gets the same dashboard a standalone `sofuu run --ui` session has,
        // without libsofuu_gpu.
        register_global_fn(ctx, "__sofuu_ui_auto_event", js_ui_auto_event);
        register_global_fn(ctx, "__sofuu_ui_auto_triggers", js_ui_auto_triggers);
        register_global_fn(ctx, "__sofuu_ui_auto_rows", js_ui_auto_rows);
        register_global_fn(ctx, "__sofuu_ui_auto_mount", js_ui_auto_mount);
        register_global_fn(ctx, "__sofuu_ui_auto_scene", js_ui_auto_scene);
        OK
    })
}

/// Unbind a runtime's UI session. NULL-tolerant; safe to call twice.
///
/// # Safety
/// `rt` must be a live runtime and this call must run on its owning thread.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_detach_runtime(rt: *mut CapiRuntime) -> c_int {
    guarded(|| {
        if rt.is_null() {
            return ERR_INVALID_ARG;
        }
        let ctx = (*rt).ffi_rt.engine_ctx() as *mut JSContext;
        if ctx.is_null() {
            return ERR_INTERNAL;
        }
        detach_ctx(ctx);
        OK
    })
}

fn detach_ctx(ctx: *mut JSContext) {
    if ctx.is_null() {
        return;
    }
    let previous = {
        let mut map = attached_map().lock().unwrap_or_else(|e| e.into_inner());
        map.remove(&(ctx as usize))
    };
    if let Some(inner) = previous.and_then(|weak| weak.upgrade()) {
        inner.attached.store(false, Ordering::Release);
    }
}

/// Called by `sofuu_rt_free` before the QuickJS context is destroyed. Public
/// within the crate only: the C host never needs this hook.
pub(crate) fn detach_ctx_for_runtime_free(ctx: *mut JSContext) {
    detach_ctx(ctx);
}

// ── JS host functions (installed by sofuu_ui_attach_runtime) ──────

unsafe extern "C" fn js_ui_is_attached(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    js_new_bool(ctx, js_lookup(ctx).is_some())
}

unsafe extern "C" fn js_ui_push_patch(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || argv.is_null() {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    }
    let Some(json) = js_to_string(ctx, *argv) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    };
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_NOT_IMPL); // detached
    };
    match serde_json::from_str::<ScenePatch>(&json) {
        Ok(patch) => {
            enqueue(&inner, Inbound::Patch(patch));
            qjs::sofuu_js_new_int32(ctx, OK)
        }
        Err(_) => qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG),
    }
}

unsafe extern "C" fn js_ui_poll_event(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_null(); // detached
    };
    let next = lock(&inner.events).pop_front();
    match next {
        Some(json) => js_new_string(ctx, &json),
        None => qjs::sofuu_js_null(),
    }
}

// ── JS Auto-Composer bridge ───────────────────────────────────────
//
// `src/js/ui.js`'s `sofuu.ui.autoComposer()` talks to these five globals
// (the standalone `libsofuu_gpu` build installs equivalents in
// `sofuu-core/src/ui_bridge.rs`). With an embedded session the composer
// lives inside this crate's RenderCore, so a pure-JS agent gets the same
// zero-boilerplate dashboard without loading libsofuu_gpu.

unsafe extern "C" fn js_ui_auto_event(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || argv.is_null() {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    }
    let Some(json) = js_to_string(ctx, *argv) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    };
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_NOT_IMPL);
    };
    match AutoAgentEvent::from_runtime_json(&json) {
        Some(ev) => {
            enqueue(&inner, Inbound::AgentEvent(ev));
            qjs::sofuu_js_new_int32(ctx, OK)
        }
        None => qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG),
    }
}

unsafe extern "C" fn js_ui_auto_triggers(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || argv.is_null() {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    }
    let Some(json) = js_to_string(ctx, *argv) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    };
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_NOT_IMPL);
    };
    match serde_json::from_str::<Vec<AutoTriggerStat>>(&json) {
        Ok(triggers) => {
            enqueue(&inner, Inbound::Triggers(triggers));
            qjs::sofuu_js_new_int32(ctx, OK)
        }
        Err(_) => qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG),
    }
}

unsafe extern "C" fn js_ui_auto_rows(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || argv.is_null() {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    }
    let Some(json) = js_to_string(ctx, *argv) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG);
    };
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_NOT_IMPL);
    };
    match serde_json::from_str::<Vec<TableRow>>(&json) {
        Ok(rows) => {
            enqueue(&inner, Inbound::Rows(rows));
            qjs::sofuu_js_new_int32(ctx, OK)
        }
        Err(_) => qjs::sofuu_js_new_int32(ctx, ERR_INVALID_ARG),
    }
}

unsafe extern "C" fn js_ui_auto_mount(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let cfg = if argc >= 1 && !argv.is_null() {
        match js_to_string(ctx, *argv) {
            Some(json) => serde_json::from_str::<JsonValue>(&json).unwrap_or(JsonValue::Null),
            None => JsonValue::Null,
        }
    } else {
        JsonValue::Null
    };
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_new_int32(ctx, ERR_NOT_IMPL);
    };
    let title = cfg
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Sofuu Agent Runtime")
        .to_string();
    let theme = if cfg.get("theme").and_then(|v| v.as_str()) == Some("dark") {
        ThemeMode::Dark
    } else {
        ThemeMode::Light
    };

    {
        let mut render = lock(&inner.render);
        let core = &mut *render;
        let composer = ensure_auto(core);
        composer.title = title;
        composer.theme = theme;
        let scene = composer.build_scene();
        core.engine.set_scene(scene);
    }
    inner.wake();
    qjs::sofuu_js_new_int32(ctx, OK)
}

unsafe extern "C" fn js_ui_auto_scene(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let Some(inner) = js_lookup(ctx) else {
        return qjs::sofuu_js_null(); // detached
    };
    let json = {
        let mut render = lock(&inner.render);
        let core = &mut *render;
        let composer = ensure_auto(core);
        serde_json::to_string(&composer.build_scene()).ok()
    };
    match json {
        Some(json) => js_new_string(ctx, &json),
        None => qjs::sofuu_js_null(),
    }
}

// ── Scene + telemetry ingestion ───────────────────────────────────

/// Queue a `ScenePatch` JSON object. Thread-safe (Runtime or Host thread).
///
/// # Safety
/// `ui` must be a live session. `patch_json` is NUL-terminated when `len == 0`.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_push_patch_json(
    ui: *mut SofuuUiSession,
    patch_json: *const c_char,
    len: usize,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let Some(raw) = json_arg(patch_json, len) else {
            return ERR_INVALID_ARG;
        };
        match serde_json::from_str::<ScenePatch>(raw) {
            Ok(patch) => {
                enqueue(&(*ui).inner, Inbound::Patch(patch));
                OK
            }
            Err(_) => ERR_INVALID_ARG,
        }
    })
}

/// Ingest one generic agent/runtime telemetry event. Thread-safe.
///
/// Accepts both the canonical snake_case [`AutoAgentEvent`] shape and the
/// camelCase `{runId,agentName,chain,t,kind,payload}` shape emitted by
/// `sofuu_rt_call_stream("agent.run", …)`, so a native host can pipe streamed
/// events straight into the Auto-Composer without a translation layer.
///
/// # Safety
/// `ui` must be a live session; `event_json` must be NUL-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_ingest_agent_event_json(
    ui: *mut SofuuUiSession,
    event_json: *const c_char,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            return ERR_INVALID_ARG;
        }
        let Some(raw) = text_arg(event_json) else {
            return ERR_INVALID_ARG;
        };
        match parse_agent_event(&raw) {
            Some(ev) => {
                enqueue(&(*ui).inner, Inbound::AgentEvent(ev));
                OK
            }
            None => ERR_INVALID_ARG,
        }
    })
}

/// Parse either shape and normalize to the canonical struct. Returns None only
/// when the input is not a JSON object at all.
fn parse_agent_event(raw: &str) -> Option<AutoAgentEvent> {
    let value: JsonValue = serde_json::from_str(raw).ok()?;
    if !value.is_object() {
        return None;
    }
    // Already canonical? Accept the exact serde shape first (this is what a
    // native host written against sofuu-ui sends).
    if let Ok(ev) = serde_json::from_value::<AutoAgentEvent>(value.clone()) {
        let looks_canonical = value.get("payload").is_none()
            && value.get("agentName").is_none()
            && value.get("runId").is_none();
        if looks_canonical {
            return Some(ev);
        }
    }
    Some(normalize_agent_event(value))
}

fn normalize_agent_event(value: JsonValue) -> AutoAgentEvent {
    let obj = value.as_object().cloned().unwrap_or_default();
    let payload = obj
        .get("payload")
        .and_then(|p| p.as_object())
        .cloned()
        .unwrap_or_default();

    let mut out = JsonMap::new();
    let mut put = |key: &str, v: JsonValue| {
        out.insert(key.to_string(), v);
    };

    let run_id = obj
        .get("run_id")
        .or_else(|| obj.get("runId"))
        .or_else(|| obj.get("turn_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    put("run_id", JsonValue::String(run_id));

    let agent_name = obj
        .get("agent_name")
        .or_else(|| obj.get("agentName"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            // `agent.run` onStep: `name` is the agent at the top level.
            obj.get("name").and_then(|v| v.as_str())
        })
        .unwrap_or("")
        .to_string();
    put("agent_name", JsonValue::String(agent_name));

    if let Some(profile) = obj.get("profile") {
        let profile = match profile {
            JsonValue::String(s) => Some(s.clone()),
            JsonValue::Object(o) => ["name", "model", "id"]
                .iter()
                .find_map(|k| o.get(*k).and_then(|v| v.as_str()))
                .map(str::to_string),
            _ => None,
        };
        if let Some(profile) = profile {
            put("profile", JsonValue::String(profile));
        }
    }

    if let Some(depth) = obj.get("depth").and_then(|v| v.as_u64()) {
        put("depth", JsonValue::from(depth));
    }
    if let Some(chain) = obj.get("chain").and_then(|v| v.as_array()) {
        put("chain", JsonValue::Array(chain.clone()));
    }

    let kind = obj
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("event")
        .to_string();
    put("kind", JsonValue::String(kind.clone()));

    let tool = obj
        .get("tool")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("name").and_then(|v| v.as_str()))
        .or_else(|| payload.get("tool").and_then(|v| v.as_str()))
        .map(str::to_string);
    if let Some(tool) = tool {
        put("tool", JsonValue::String(tool));
    }

    let summary = obj
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            ["task", "step", "answer", "error", "result", "text"]
                .iter()
                .find_map(|k| payload.get(*k).and_then(|v| v.as_str()))
                .map(str::to_string)
        });
    if let Some(summary) = summary {
        put("title", JsonValue::String(summary));
    }

    if let Some(detail) = obj.get("detail").and_then(|v| v.as_str()) {
        put("detail", JsonValue::String(detail.to_string()));
    } else if let Some(args) = payload.get("args") {
        let detail = args
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| args.to_string());
        put("detail", JsonValue::String(detail));
    }

    let tokens = obj
        .get("tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| payload.get("tokens").and_then(|v| v.as_u64()))
        .or_else(|| {
            let usage = payload.get("usage").or_else(|| obj.get("usage"))?;
            let sum = ["total_tokens", "totalTokens"]
                .iter()
                .find_map(|k| usage.get(*k).and_then(|v| v.as_u64()))
                .or_else(|| {
                    let input = ["input_tokens", "promptTokens", "prompt_tokens"]
                        .iter()
                        .find_map(|k| usage.get(*k).and_then(|v| v.as_u64()))
                        .unwrap_or(0);
                    let output = ["output_tokens", "completionTokens", "completion_tokens"]
                        .iter()
                        .find_map(|k| usage.get(*k).and_then(|v| v.as_u64()))
                        .unwrap_or(0);
                    Some(input + output)
                })?;
            Some(sum)
        });
    if let Some(tokens) = tokens {
        put("tokens", JsonValue::from(tokens));
    }

    let elapsed_ms = obj
        .get("elapsed_ms")
        .or_else(|| obj.get("elapsedMs"))
        .or_else(|| obj.get("duration_ms"))
        .and_then(|v| v.as_u64())
        .or_else(|| payload.get("duration").and_then(|v| v.as_u64()))
        .or_else(|| payload.get("duration_ms").and_then(|v| v.as_u64()));
    if let Some(ms) = elapsed_ms {
        put("elapsed_ms", JsonValue::from(ms));
    }

    if let Some(time_label) = obj
        .get("time_label")
        .or_else(|| obj.get("timeLabel"))
        .and_then(|v| v.as_str())
    {
        put("time_label", JsonValue::String(time_label.to_string()));
    }

    if let Some(tone) = obj.get("tone").cloned() {
        put("tone", tone);
    }

    // HITL: accept the canonical `approval` object or derive one from an
    // `approval_request` / `approval_resolved` payload. This is what makes the
    // native stream path show the same [Approve]/[Always]/[Deny] bar.
    let approval = obj
        .get("approval")
        .cloned()
        .or_else(|| build_approval(&kind, &payload));
    if let Some(approval) = approval {
        put("approval", approval);
    }

    serde_json::from_value::<AutoAgentEvent>(JsonValue::Object(out))
        .unwrap_or_else(|_| AutoAgentEvent {
            kind,
            ..AutoAgentEvent::default()
        })
}

fn build_approval(kind: &str, payload: &JsonMap<String, JsonValue>) -> Option<JsonValue> {
    if kind != "approval_request" && kind != "approval_resolved" {
        return None;
    }
    let id = payload.get("id").and_then(|v| v.as_str())?.to_string();
    let tool = payload
        .get("tool")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args_preview = payload
        .get("args")
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .unwrap_or_else(|| v.to_string())
        })
        .unwrap_or_default();
    let mode = payload
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let approval = PendingApproval {
        id,
        tool,
        args_preview,
        mode,
    };
    serde_json::to_value(approval).ok()
}

// ── Input + semantic event queue ──────────────────────────────────

/// Queue an OS input event. Thread-safe; wakes the host so a paused display
/// link resumes and applies it on the next frame.
///
/// # Safety
/// `ui` must be a live session; `evt` must point to a readable struct.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_dispatch_input(
    ui: *mut SofuuUiSession,
    evt: *const SofuuUiInputEvent,
) -> c_int {
    guarded(|| {
        if ui.is_null() || evt.is_null() {
            return ERR_INVALID_ARG;
        }
        let evt = &*evt;
        let text = if evt.text_utf8.is_null() {
            None
        } else {
            Some(CStr::from_ptr(evt.text_utf8).to_string_lossy().into_owned())
        };
        let input = match evt.kind {
            SOFUU_INPUT_POINTER_MOVE => InputEvent::PointerMove {
                x: evt.x,
                y: evt.y,
            },
            SOFUU_INPUT_POINTER_DOWN => InputEvent::PointerDown {
                x: evt.x,
                y: evt.y,
                button: evt.button.min(u8::MAX as u32) as u8,
            },
            SOFUU_INPUT_POINTER_UP => InputEvent::PointerUp {
                x: evt.x,
                y: evt.y,
                button: evt.button.min(u8::MAX as u32) as u8,
            },
            SOFUU_INPUT_SCROLL => InputEvent::Scroll {
                x: evt.x,
                y: evt.y,
                delta_x: evt.dx,
                delta_y: evt.dy,
            },
            SOFUU_INPUT_KEY_DOWN => InputEvent::KeyDown {
                key: text.unwrap_or_default(),
                modifiers: evt.modifiers,
            },
            SOFUU_INPUT_TEXT_INPUT => InputEvent::TextInput {
                text: text.unwrap_or_default(),
            },
            SOFUU_INPUT_RESIZE => InputEvent::Resize {
                width: evt.dx,
                height: evt.dy,
                scale_factor: evt.scale_factor,
            },
            SOFUU_INPUT_CLOSE_REQUESTED => InputEvent::CloseRequested,
            _ => return ERR_INVALID_ARG,
        };
        {
            let inner = &(*ui).inner;
            {
                let mut q = lock(&inner.inputs);
                q.push_back(input);
            }
            inner.wake();
        }
        OK
    })
}

/// Pop the next semantic UI event as malloc'd JSON. Returns 1 when written,
/// 0 when the queue is empty. Thread-safe.
///
/// # Safety
/// `ui` must be a live session; `out_event_json` must be writable.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_poll_event_json(
    ui: *mut SofuuUiSession,
    out_event_json: *mut *mut c_char,
) -> c_int {
    guarded(|| {
        if ui.is_null() || out_event_json.is_null() {
            return ERR_INVALID_ARG;
        }
        *out_event_json = ptr::null_mut();
        let inner = &(*ui).inner;
        let next = lock(&inner.events).pop_front();
        match next {
            Some(json) => {
                *out_event_json = crate::malloc_cstr(&json);
                if (*out_event_json).is_null() {
                    ERR_INTERNAL
                } else {
                    1
                }
            }
            None => OK,
        }
    })
}

// ── Mode A: frame evaluation ──────────────────────────────────────

/// Drain queued scene patches / telemetry into the render core. Called with
/// the render lock held; the inbound queue lock is already released.
fn apply_inbound(core: &mut RenderCore, items: Vec<Inbound>) {
    for item in items {
        match item {
            Inbound::Patch(patch) => {
                core.engine.apply_patch(patch);
            }
            Inbound::AgentEvent(ev) => {
                let composer = ensure_auto(core);
                for patch in composer.ingest_event(ev) {
                    core.engine.apply_patch(patch);
                }
            }
            Inbound::Triggers(triggers) => {
                let composer = ensure_auto(core);
                for patch in composer.update_triggers(&triggers) {
                    core.engine.apply_patch(patch);
                }
            }
            Inbound::Rows(rows) => {
                let composer = ensure_auto(core);
                for patch in composer.update_state_rows(rows) {
                    core.engine.apply_patch(patch);
                }
            }
        }
    }
}

/// Return the session's Auto-Composer, seeding its dashboard scene on first
/// use. Shared by native stream ingestion and the JS `__sofuu_ui_auto_*`
/// bridge so both paths render the identical topology.
fn ensure_auto(core: &mut RenderCore) -> &mut AutoComposer {
    if core.auto.is_none() {
        let composer = AutoComposer::new(core.auto_title.clone(), core.engine.scene.theme);
        core.engine.set_scene(composer.build_scene());
        core.auto = Some(composer);
    }
    core.auto.as_mut().expect("auto composer seeded above")
}

fn fill_draw_list(
    core: &RenderCore,
    dl: &DrawList,
    compiled: bool,
    needs_next_frame: bool,
    out: *mut SofuuUiDrawList,
) {
    let atlas = &core.engine.atlas;
    let viewport = core.engine.viewport();
    // SAFETY: the caller checked out is non-NULL.
    unsafe {
        let out = &mut *out;
        out.uniforms = SofuuUiFrameUniforms {
            viewport_size: viewport,
            atlas_size: [atlas.width as f32, atlas.height as f32],
            scale_factor: core.scale_factor,
            time_sec: core.time_sec,
            _pad: [0.0; 2],
        };
        out.quads = dl.quads.as_ptr();
        out.quad_count = dl.quads.len() as u32;
        out.ribbons = dl.ribbons.as_ptr();
        out.ribbon_count = dl.ribbons.len() as u32;
        out.strokes = dl.strokes.as_ptr();
        out.stroke_count = dl.strokes.len() as u32;
        out.glyphs = dl.glyphs.as_ptr();
        out.glyph_count = dl.glyphs.len() as u32;
        out.cmds = dl.cmds.as_ptr();
        out.cmd_count = dl.cmds.len() as u32;
        out.clips = dl.clips.as_ptr();
        out.clip_count = dl.clips.len() as u32;
        out.recompiled = compiled as u8;
        out.needs_next_frame = needs_next_frame as u8;
        out.theme_is_dark = matches!(core.engine.scene.theme, ThemeMode::Dark) as u8;
        out._reserved = 0;
    }
}

/// Mode A: render-thread frame evaluation.
///
/// # Safety
/// `ui` must be a live session; `out_list` must be writable. The returned
/// pointers stay valid until the next poll/free on this session.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_poll_draw_list(
    ui: *mut SofuuUiSession,
    dt_sec: f32,
    width_px: f32,
    height_px: f32,
    scale_factor: f32,
    out_list: *mut SofuuUiDrawList,
) -> c_int {
    guarded(|| {
        if ui.is_null() || out_list.is_null() {
            return ERR_INVALID_ARG;
        }
        let inner = &(*ui).inner;
        let mut render = lock(&inner.render);
        let core = &mut *render;

        // Viewport: fall back to the last good size on a zero/NaN resize.
        let mut width = width_px;
        let mut height = height_px;
        if !width.is_finite() || width <= 0.0 {
            width = core.engine.viewport()[0].max(1.0);
        }
        if !height.is_finite() || height <= 0.0 {
            height = core.engine.viewport()[1].max(1.0);
        }
        let scale = if scale_factor.is_finite() && scale_factor > 0.0 {
            scale_factor
        } else {
            1.0
        };
        core.scale_factor = scale;

        let viewport = core.engine.viewport();
        if (viewport[0] - width).abs() > 0.25 || (viewport[1] - height).abs() > 0.25 {
            let events = core.engine.handle_input(InputEvent::Resize {
                width,
                height,
                scale_factor: scale,
            });
            push_events(inner, events);
        }

        // Move the queue payload out under the queue lock, then apply it under
        // the render lock only: producers never wait on a compiling frame.
        let inbound: Vec<Inbound> = {
            let mut q = lock(&inner.inbound);
            q.drain(..).collect()
        };
        if !inbound.is_empty() {
            apply_inbound(core, inbound);
        }

        let inputs: Vec<InputEvent> = {
            let mut q = lock(&inner.inputs);
            q.drain(..).collect()
        };
        for input in inputs {
            let events = core.engine.handle_input(input);
            push_events(inner, events);
        }

        let dt = if dt_sec.is_finite() {
            dt_sec.clamp(0.0, 0.25)
        } else {
            0.0
        };
        core.time_sec += dt;

        // Alternate buffers only when this tick actually compiled. An idle
        // tick must keep returning the last compiled geometry, not the other
        // (stale) buffer.
        core.active ^= 1;
        let status = core
            .engine
            .tick_and_compile(dt, &mut core.draw_lists[core.active]);
        if !status.compiled {
            core.active ^= 1;
        }

        let pending = !lock(&inner.inbound).is_empty() || !lock(&inner.inputs).is_empty();
        let needs_next_frame = status.animating || pending;
        let dl = &core.draw_lists[core.active];
        fill_draw_list(core, dl, status.compiled, needs_next_frame, out_list);
        OK
    })
}

// ── Mode B: direct surface (optional backend) ─────────────────────

// The Direct Surface renderer is specified by
// PLAN-CROSS-PLATFORM-GPU-WINDOW-BACKENDS. Keeping the symbols present in
// every build (returning -2 until the backend lands) means a host can link
// against one stable ABI and feature-detect at runtime, while Mode A — the
// mandatory first path — never depends on OS windowing or sofuu-gpu.

/// # Safety
/// `ui`/`mtl_device`/`ca_metal_layer` must be valid for the host's backend.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_surface_attach_metal(
    ui: *mut SofuuUiSession,
    _mtl_device: *mut c_void,
    _ca_metal_layer: *mut c_void,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            ERR_INVALID_ARG
        } else {
            ERR_NOT_IMPL
        }
    })
}

/// # Safety
/// `ui`/`egl_display`/`egl_surface`/`egl_context` must be valid for EGL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_surface_attach_egl(
    ui: *mut SofuuUiSession,
    _egl_display: *mut c_void,
    _egl_surface: *mut c_void,
    _egl_context: *mut c_void,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            ERR_INVALID_ARG
        } else {
            ERR_NOT_IMPL
        }
    })
}

/// # Safety
/// `ui` must be a live session with an attached surface.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_surface_render(
    ui: *mut SofuuUiSession,
    _dt_sec: f32,
    _width_px: u32,
    _height_px: u32,
    _scale_factor: f32,
) -> c_int {
    guarded(|| {
        if ui.is_null() {
            ERR_INVALID_ARG
        } else {
            ERR_NOT_IMPL
        }
    })
}

/// # Safety
/// `ui` must be a live session.
#[no_mangle]
pub unsafe extern "C" fn sofuu_ui_surface_detach(ui: *mut SofuuUiSession) -> c_int {
    guarded(|| {
        if ui.is_null() {
            ERR_INVALID_ARG
        } else {
            ERR_NOT_IMPL
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_patch_json() -> String {
        let scene = kit::app("Embedded")
            .add(kit::button("go", "Go"))
            .build();
        serde_json::to_string(&ScenePatch::ReplaceScene(scene)).unwrap()
    }

    #[test]
    fn config_parses_and_rejects_bad_json() {
        assert!(SessionConfig::parse(None).is_ok());
        let cfg = SessionConfig::parse(Some(r#"{"title":"X","theme":"dark","width":320}"#)).unwrap();
        assert_eq!(cfg.title, "X");
        assert_eq!(cfg.theme, ThemeMode::Dark);
        assert_eq!(cfg.width, 320.0);
        assert!(SessionConfig::parse(Some("nope")).is_err());
        assert!(SessionConfig::parse(Some(r#"{"theme":"sepia"}"#)).is_err());
    }

    #[test]
    fn stream_event_normalization_picks_payload_fields() {
        let ev = parse_agent_event(
            r#"{"runId":"r1","name":"researcher","depth":1,"kind":"tool",
                "payload":{"name":"web.search","args":"{\"q\":\"x\"}","duration":42}}"#,
        )
        .unwrap();
        assert_eq!(ev.run_id, "r1");
        assert_eq!(ev.agent_name, "researcher");
        assert_eq!(ev.kind, "tool");
        assert_eq!(ev.tool.as_deref(), Some("web.search"));
        assert_eq!(ev.elapsed_ms, Some(42));

        let approval = parse_agent_event(
            r#"{"kind":"approval_request",
                "payload":{"id":"a1","tool":"bash","args":"rm -rf","mode":"async"}}"#,
        )
        .unwrap();
        assert_eq!(approval.approval.unwrap().id, "a1");
    }

    #[test]
    fn session_roundtrip_polls_draw_list_and_dispatches_click() {
        unsafe {
            let mut err: *mut c_char = ptr::null_mut();
            let ui = sofuu_ui_session_new(ptr::null(), &mut err);
            assert!(!ui.is_null());
            assert!(err.is_null());

            let json = scene_patch_json();
            assert_eq!(
                sofuu_ui_push_patch_json(ui, json.as_ptr() as *const c_char, json.len()),
                OK
            );

            let mut dl = SofuuUiDrawList::default();
            assert_eq!(
                sofuu_ui_poll_draw_list(ui, 1.0 / 60.0, 800.0, 600.0, 2.0, &mut dl),
                OK
            );
            assert_eq!(dl.uniforms.viewport_size, [800.0, 600.0]);
            assert_eq!(dl.uniforms.scale_factor, 2.0);
            assert_eq!(dl.recompiled, 1);
            assert!(dl.quad_count > 0 && dl.cmd_count > 0 && dl.clip_count > 0);
            assert!(!dl.quads.is_null() && !dl.cmds.is_null());

            // Button centre from the same deterministic scene/layout.
            let scene = kit::app("Embedded")
                .add(kit::button("go", "Go"))
                .build();
            let mut probe = UiEngine::new(scene, [800.0, 600.0]);
            let mut probe_dl = DrawList::new();
            probe.tick_and_compile(1.0 / 60.0, &mut probe_dl);
            let rect = probe.layout.get("go").expect("button laid out");
            let cx = rect.x + rect.w / 2.0;
            let cy = rect.y + rect.h / 2.0;

            let down = SofuuUiInputEvent {
                kind: SOFUU_INPUT_POINTER_DOWN,
                x: cx,
                y: cy,
                dx: 0.0,
                dy: 0.0,
                scale_factor: 1.0,
                button: 0,
                modifiers: 0,
                text_utf8: ptr::null(),
            };
            let up = SofuuUiInputEvent {
                kind: SOFUU_INPUT_POINTER_UP,
                ..down
            };
            assert_eq!(sofuu_ui_dispatch_input(ui, &down), OK);
            assert_eq!(sofuu_ui_dispatch_input(ui, &up), OK);

            let mut saw_click = false;
            let mut settled = false;
            for _ in 0..240 {
                assert_eq!(
                    sofuu_ui_poll_draw_list(ui, 1.0 / 60.0, 800.0, 600.0, 2.0, &mut dl),
                    OK
                );
                let mut out: *mut c_char = ptr::null_mut();
                while sofuu_ui_poll_event_json(ui, &mut out) > 0 {
                    let ev = CStr::from_ptr(out).to_string_lossy().into_owned();
                    crate::sofuu_free(out as *mut c_void);
                    if ev.contains("\"type\":\"click\"") && ev.contains("\"id\":\"go\"") {
                        saw_click = true;
                    }
                }
                if dl.needs_next_frame == 0 {
                    settled = true;
                    break;
                }
            }
            assert!(saw_click, "pointer down/up must emit a click event");
            assert!(settled, "springs must settle to needs_next_frame == 0");

            sofuu_ui_session_free(ui);
        }
    }

    #[test]
    fn atlas_is_borrowed_and_click_queues_are_bounded() {
        unsafe {
            let ui = sofuu_ui_session_new(ptr::null(), ptr::null_mut());
            assert!(!ui.is_null());
            let mut pixels: *const u8 = ptr::null();
            let mut w = 0u32;
            let mut h = 0u32;
            let mut px_range = 0.0f32;
            assert_eq!(
                sofuu_ui_session_get_atlas(ui, &mut pixels, &mut w, &mut h, &mut px_range),
                OK
            );
            assert!(!pixels.is_null());
            assert!(w > 0 && h > 0);
            assert!(px_range > 0.0);
            sofuu_ui_session_free(ui);
        }
    }

    fn rss_kib() -> Option<u64> {
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    }

    /// Phase C5 stress gate: 10k compiled frames must reuse the DrawList
    /// vectors (zero per-frame heap growth) and the process RSS must stay
    /// flat. Ignored by default because it is deliberately long; run it with:
    ///   cargo test -p sofuu-capi --release ui_10k_frame_rss_gate -- \
    ///       --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "10k-frame RSS gate — run explicitly (Phase C5)"]
    fn ui_10k_frame_rss_gate() {
        unsafe {
            let ui = sofuu_ui_session_new(ptr::null(), ptr::null_mut());
            assert!(!ui.is_null());

            // A scene big enough that a leak per frame would be obvious:
            // text, a button, a stat, and a table.
            let scene = kit::app("Stress")
                .header(kit::title("Stress"))
                .stats([kit::stat("s1", "Steps", 1.0)])
                .add(kit::button("go", "Go"))
                .add(kit::text("tick", "warmup-content-0000"))
                .build();
            let json = serde_json::to_string(&ScenePatch::ReplaceScene(scene)).unwrap();
            assert_eq!(
                sofuu_ui_push_patch_json(ui, json.as_ptr() as *const c_char, json.len()),
                OK
            );

            let mut dl = SofuuUiDrawList::default();
            // Warm BOTH double buffers so capacities are established before
            // the measurement window.
            for i in 0..8 {
                let patch = format!(
                    r#"{{"op":"set_text","id":"tick","content":"warmup-content-{i:04}"}}"#
                );
                assert_eq!(
                    sofuu_ui_push_patch_json(ui, patch.as_ptr() as *const c_char, patch.len()),
                    OK
                );
                assert_eq!(
                    sofuu_ui_poll_draw_list(ui, 1.0 / 60.0, 900.0, 700.0, 2.0, &mut dl),
                    OK
                );
            }

            let inner = &(*ui).inner;
            let before = {
                let render = lock(&inner.render);
                (
                    render.draw_lists[0].quads.capacity(),
                    render.draw_lists[1].quads.capacity(),
                    render.draw_lists[0].glyphs.capacity(),
                    render.draw_lists[1].glyphs.capacity(),
                    render.draw_lists[0].cmds.capacity(),
                    render.draw_lists[1].cmds.capacity(),
                )
            };
            let rss_before = rss_kib();

            for i in 0..10_000u32 {
                let patch = format!(
                    r#"{{"op":"set_text","id":"tick","content":"frame-{i:04}"}}"#
                );
                assert_eq!(
                    sofuu_ui_push_patch_json(ui, patch.as_ptr() as *const c_char, patch.len()),
                    OK
                );
                assert_eq!(
                    sofuu_ui_poll_draw_list(ui, 1.0 / 60.0, 900.0, 700.0, 2.0, &mut dl),
                    OK
                );
            }

            let after = {
                let render = lock(&inner.render);
                (
                    render.draw_lists[0].quads.capacity(),
                    render.draw_lists[1].quads.capacity(),
                    render.draw_lists[0].glyphs.capacity(),
                    render.draw_lists[1].glyphs.capacity(),
                    render.draw_lists[0].cmds.capacity(),
                    render.draw_lists[1].cmds.capacity(),
                )
            };
            assert_eq!(
                before, after,
                "10k frames must not grow DrawList capacities (zero steady-state allocs)"
            );

            if let (Some(before), Some(after)) = (rss_before, rss_kib()) {
                let grown = after.saturating_sub(before);
                eprintln!("RSS: {before} KiB -> {after} KiB (+{grown} KiB over 10k frames)");
                assert!(
                    grown < 8 * 1024,
                    "10k frames grew RSS by {grown} KiB (ceiling 8 MiB)"
                );
            } else {
                eprintln!("skipping RSS assertion: `ps -o rss=` unavailable");
            }

            sofuu_ui_session_free(ui);
        }
    }
}
