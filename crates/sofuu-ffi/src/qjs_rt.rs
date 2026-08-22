// sofuu-ffi — hermetic QuickJS sandbox runtime for the RLM loop (PLAN-RLM R0).
//
// `QjsSandbox` owns a standalone JSRuntime/JSContext pair with hard limits
// (memory cap, stack cap, wall-clock deadline via JS_SetInterruptHandler)
// and a seven-function native whitelist (`__hx_*`). Nothing else is visible
// inside: `JS_NewContext`'s default intrinsics carry no std/os/quickjs-libc
// helpers, so sandbox code cannot reach fetch/fs/process/host globals.
//
// Native callbacks must be `'static`, so they cannot borrow the sandbox
// state. Instead each function value carries the sandbox id as its
// func_data (JS_NewCFunctionData) and looks the state up in a global
// registry keyed by that id (below). The entry is removed on Drop, before
// the JS side is freed — after that no callback can touch freed state.
//
// All `unsafe` for the RLM sandbox lives here; sofuu-core drives this
// module through safe functions only.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::qjs::{self, JSContext, JSRuntime, JSValue, JSValueConst};

/* ── Per-sandbox host state ──────────────────────────────────────
 * Plain data. Owned by a Box in sofuu-core (rlm/sandbox.rs), which also
 * computes the chunk map (UTF-8-safe char-offset logic) — the __hx_*
 * callbacks only ever slice with the byte ranges stored here. All offsets
 * exposed to JS are CHARACTER offsets (Unicode scalar values). */

pub struct SandboxState {
    /// The full context. Never copied wholesale into JS memory — sandbox
    /// code reads it through `__hx_chunk` / `__hx_peek`.
    pub context: String,
    /// Total context length in characters (= JS-visible `len()`).
    pub char_len: usize,
    /// Chunk map as byte ranges into `context`; both ends char-aligned.
    pub chunks: Vec<(usize, usize)>,
    /// Trace events recorded by native callbacks during the current eval
    /// (chunk/peek/emit; episode-level events live in sofuu-core). `t_ms`
    /// is relative to `eval_started`.
    pub events: Vec<SandboxEvent>,
    /// Events refused past the cap (G2: an unbounded trace is a memory
    /// bomb — `for(;;) chunk(0)` would otherwise grow this Vec until the
    /// deadline bites). Counted, never stored.
    pub dropped_events: u32,
    /// Set at the start of every snippet eval; cleared when it returns.
    pub eval_started: Option<Instant>,
}

/// One trace record; sofuu-core maps these into `rlm::RlmEvent`.
pub struct SandboxEvent {
    pub kind: String,
    pub repr: String,
    pub t_ms: u64,
}

/// Per-sandbox trace cap (PLAN-RLM security batch G2): at most this many
/// sandbox-side events are kept; the rest increment `dropped_events`.
/// sofuu-core applies the same cap episode-side, then merges to 512 total.
pub const MAX_TRACE_EVENTS: usize = 512;

impl SandboxState {
    pub fn new() -> Self {
        Self {
            context: String::new(),
            char_len: 0,
            chunks: Vec::new(),
            events: Vec::new(),
            dropped_events: 0,
            eval_started: None,
        }
    }

    fn now_ms(&self) -> u64 {
        self.eval_started
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Chunk text by index (None when out of range).
    fn chunk_text(&self, i: usize) -> Option<&str> {
        let (a, b) = *self.chunks.get(i)?;
        self.context.get(a..b)
    }

    /// Slice the context by CHARACTER offsets, clamped to bounds. `start >=
    /// end` yields the empty string.
    pub fn peek_chars(&self, start: usize, end: usize) -> String {
        let a = start.min(self.char_len);
        let b = end.min(self.char_len);
        if a >= b {
            return String::new();
        }
        self.context.chars().skip(a).take(b - a).collect()
    }

    fn push_event(&mut self, kind: &str, repr: String) {
        if self.events.len() >= MAX_TRACE_EVENTS {
            self.dropped_events = self.dropped_events.saturating_add(1);
            return;
        }
        let t_ms = self.now_ms();
        self.events.push(SandboxEvent {
            kind: kind.to_string(),
            repr,
            t_ms,
        });
    }
}

impl Default for SandboxState {
    fn default() -> Self {
        Self::new()
    }
}

/* ── Sandbox registry ────────────────────────────────────────────
 * Maps sandbox id → state pointer. Stored as usize so the map stays Send
 * without `unsafe impl`; only the lookups below (inside this crate) ever
 * cast back to a pointer. Locking never spans a JS eval, so callbacks
 * (which run inside one) cannot deadlock. */

fn registry() -> &'static Mutex<HashMap<u64, usize>> {
    static REG: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register `state` under `id`. Safe: storing a pointer cannot hurt;
/// dereferencing it only happens via the callbacks in this module.
pub fn register_state(id: u64, state: *mut SandboxState) {
    if let Ok(mut reg) = registry().lock() {
        reg.insert(id, state as usize);
    }
}

/// Remove the registry entry for `id` (called from `QjsSandbox::drop`).
pub fn unregister_state(id: u64) {
    if let Ok(mut reg) = registry().lock() {
        reg.remove(&id);
    }
}

fn lookup_state(id: u64) -> Option<*mut SandboxState> {
    registry()
        .lock()
        .ok()
        .and_then(|reg| reg.get(&id).copied())
        .map(|p| p as *mut SandboxState)
}

// ── Interrupt handler (wall-clock deadline) ─────────────────────

struct InterruptState {
    deadline: Instant,
}

unsafe extern "C" fn interrupt_cb(_rt: *mut JSRuntime, opaque: *mut c_void) -> c_int {
    if opaque.is_null() {
        return 0;
    }
    // SAFETY: opaque is a live Box<InterruptState> owned by QjsSandbox;
    // the box outlives the runtime (freed after JS_FreeRuntime in Drop).
    let st = unsafe { &*(opaque as *const InterruptState) };
    if Instant::now() >= st.deadline {
        1
    } else {
        0
    }
}

// ── Eval outcome (pre-sentinel-interpretation) ──────────────────

pub enum RawOutcome {
    /// Last-expression value, stringified.
    Value(String),
    /// Exception text as raw bytes. The \0-prefixed suspension/final
    /// sentinels thrown by `__hx_llmBatch` / `__hx_final` survive intact
    /// here — a plain CStr read would truncate at the NUL.
    Exception(Vec<u8>),
}

// ── The sandbox ─────────────────────────────────────────────────

/// A hermetic QuickJS runtime+context, its deadline box, and the id its
/// native functions are registered under. `!Send` by construction (raw
/// pointers) — a sandbox lives and dies on one thread, like QuickJS itself.
pub struct QjsSandbox {
    rt: *mut JSRuntime,
    ctx: *mut JSContext,
    id: u64,
    interrupt: *mut InterruptState,
}

impl QjsSandbox {
    /// Create a sandbox with the given limits and register `state` for the
    /// native callbacks. `state` must be a live, heap-pinned `SandboxState`
    /// that outlives the returned value (sofuu-core guarantees this via
    /// struct field drop order). Returns None if QuickJS cannot be set up.
    pub fn new(
        id: u64,
        mem_limit_bytes: usize,
        stack_bytes: usize,
        deadline: Duration,
        state: *mut SandboxState,
    ) -> Option<Self> {
        // SAFETY: plain QuickJS lifecycle on a fresh runtime, single thread.
        unsafe {
            let rt = qjs::JS_NewRuntime();
            if rt.is_null() {
                return None;
            }
            qjs::JS_SetMemoryLimit(rt, mem_limit_bytes);
            qjs::JS_SetMaxStackSize(rt, stack_bytes);
            let interrupt = Box::into_raw(Box::new(InterruptState {
                deadline: Instant::now() + deadline,
            }));
            qjs::JS_SetInterruptHandler(rt, Some(interrupt_cb), interrupt as *mut c_void);
            let ctx = qjs::JS_NewContext(rt);
            if ctx.is_null() {
                // SAFETY: interrupt was boxed above; rt has no context.
                drop(Box::from_raw(interrupt));
                qjs::JS_FreeRuntime(rt);
                return None;
            }
            register_state(id, state);
            let sb = Self {
                rt,
                ctx,
                id,
                interrupt,
            };
            if !sb.install_natives() {
                // Drop unregisters + frees in the right order.
                drop(sb);
                return None;
            }
            Some(sb)
        }
    }

    /// Re-arm the interrupt deadline: the next eval gets `budget` of wall
    /// time from now. The construction-time deadline is the episode cap;
    /// sofuu-core calls this before every snippet with the per-eval slice
    /// (min of remaining episode wall and `eval_slice_ms`), so one aborted
    /// eval doesn't leave the handler permanently expired.
    pub fn reset_deadline(&self, budget: Duration) {
        // SAFETY: the interrupt box is owned by self and alive (freed only
        // in Drop); the sandbox is single-threaded, no concurrent read.
        unsafe { (*self.interrupt).deadline = Instant::now() + budget };
    }

    /// Register the eight-name `__hx_*` whitelist on the global object.
    /// Each function value carries this sandbox's id as func_data[0] (an
    /// int32 JSValue); JS_NewCFunctionData DUPs it, so we free our copy.
    fn install_natives(&self) -> bool {
        const NATIVES: &[(&str, qjs::JSCFunctionDataFn, c_int)] = &[
            ("__hx_len", cb_len, 0),
            ("__hx_count", cb_count, 0),
            ("__hx_chunk", cb_chunk, 1),
            ("__hx_peek", cb_peek, 2),
            ("__hx_llmBatch", cb_llm_batch, 1),
            ("__hx_tool", cb_tool, 1),
            ("__hx_final", cb_final, 1),
            ("__hx_emit", cb_emit, 2),
        ];
        // SAFETY: ctx is live; all values freed or transferred as noted.
        unsafe {
            let ctp = qjs::CtxPtr::new(self.ctx);
            let global = qjs::global_object(ctp);
            let id_val = qjs::new_int32(ctp, self.id as i32);
            let data = [id_val];
            for (name, func, length) in NATIVES {
                let cname = CString::new(*name).unwrap_or_default();
                let f = qjs::JS_NewCFunctionData(self.ctx, *func, *length, 0, 1, data.as_ptr());
                if qjs::is_exception(f) || qjs::sofuu_js_set_property_str(
                    self.ctx,
                    global,
                    cname.as_ptr(),
                    f,
                ) < 0
                {
                    // SetPropertyStr failed → we still own f.
                    if !qjs::is_exception(f) {
                        qjs::sofuu_js_free_value(self.ctx, f);
                    }
                    qjs::sofuu_js_free_value(self.ctx, id_val);
                    qjs::sofuu_js_free_value(self.ctx, global);
                    return false;
                }
                // Property now owns the function value — do NOT free f
                // (same contract as bridge::register_global_fn).
            }
            qjs::sofuu_js_free_value(self.ctx, id_val);
            qjs::sofuu_js_free_value(self.ctx, global);
            true
        }
    }

    /// Evaluate `src` in the sandbox global scope and stringify the result.
    /// Safe: the raw pointers stay inside; returned data is owned. A NUL in
    /// `src` truncates the source at the C-string boundary (snippet text
    /// never legitimately contains one).
    pub fn eval(&self, src: &str) -> RawOutcome {
        let c_src = CString::new(src).unwrap_or_default();
        // SAFETY: ctx live; c_src is NUL-terminated, len excludes the NUL.
        unsafe {
            let r = qjs::JS_Eval(
                self.ctx,
                c_src.as_ptr(),
                c_src.as_bytes().len(),
                c"<rlm>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            if qjs::is_exception(r) {
                let exc = qjs::JS_GetException(self.ctx);
                // Length-aware read: sentinel payloads start with \0.
                let mut len: usize = 0;
                let p = qjs::JS_ToCStringLen2(self.ctx, &mut len, exc, 0);
                let bytes = if p.is_null() {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(p as *const u8, len).to_vec()
                };
                if !p.is_null() {
                    qjs::sofuu_js_free_cstring(self.ctx, p);
                }
                qjs::sofuu_js_free_value(self.ctx, exc);
                RawOutcome::Exception(bytes)
            } else {
                let p = qjs::sofuu_js_to_cstring(self.ctx, r);
                let s = if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                if !p.is_null() {
                    qjs::sofuu_js_free_cstring(self.ctx, p);
                }
                qjs::sofuu_js_free_value(self.ctx, r);
                RawOutcome::Value(s)
            }
        }
    }
}

impl Drop for QjsSandbox {
    fn drop(&mut self) {
        // Unregister FIRST: once the entry is gone, no callback can ever
        // reach the state Box (which sofuu-core frees after dropping us).
        unregister_state(self.id);
        // SAFETY: ctx belongs to rt; both are ours; interrupt box is ours.
        unsafe {
            qjs::JS_FreeContext(self.ctx);
            qjs::JS_FreeRuntime(self.rt);
            drop(Box::from_raw(self.interrupt));
        }
    }
}

/* ── __hx_* native callbacks ─────────────────────────────────────
 * One shared shape (JSCFunctionDataFn); func_data[0] is the sandbox id
 * installed by install_natives. These never unwind: no unwrap/expect, no
 * panicking paths — a Rust panic across a QuickJS C frame would be UB. */

/// Read the sandbox id from func_data[0] via the official converter (no
/// tag-layout guessing).
unsafe fn id_of(ctx: *mut JSContext, func_data: *mut JSValue) -> Option<u64> {
    if func_data.is_null() {
        return None;
    }
    let mut id: i64 = -1;
    // SAFETY: ctx is live (its call invoked us); func_data[0] is the int32
    // value we installed — JS_ToInt64 reads it without side effects.
    if unsafe { qjs::JS_ToInt64(ctx, &mut id, *func_data) } != 0 || id < 0 {
        return None;
    }
    Some(id as u64)
}

/// Resolve the SandboxState for the calling sandbox, then run `f` on it.
unsafe fn with_state<T>(
    ctx: *mut JSContext,
    func_data: *mut JSValue,
    f: impl FnOnce(&mut SandboxState) -> T,
) -> Option<T> {
    let ptr = lookup_state(unsafe { id_of(ctx, func_data) }?)?;
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the pointer was registered while a Box pinned it in
    // sofuu-core; that Box outlives the JS context calling us (drop order:
    // unregister → free JS → free state). No Rust &mut to the state is
    // held while JS runs.
    Some(f(unsafe { &mut *ptr }))
}

/// Numeric argument helper: `argv[idx]` as i64 (0 when absent).
unsafe fn arg_i64(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
    idx: c_int,
) -> i64 {
    if argv.is_null() || idx >= argc {
        return 0;
    }
    let mut v: i64 = 0;
    // SAFETY: argv has argc entries; ctx is live.
    unsafe { qjs::JS_ToInt64(ctx, &mut v, *argv.add(idx as usize)) };
    v
}

/// String argument helper: `argv[idx]` stringified ("" when absent).
unsafe fn arg_string(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
    idx: c_int,
) -> String {
    if argv.is_null() || idx >= argc {
        return String::new();
    }
    // SAFETY: argv has argc entries; ctx is live; p freed after copy.
    unsafe {
        let p = qjs::sofuu_js_to_cstring(ctx, *argv.add(idx as usize));
        if p.is_null() {
            return String::new();
        }
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, p);
        s
    }
}

/// Throw a plain-string exception (NUL-safe) and return the exception
/// sentinel value the ABI expects from a native.
unsafe fn throw_str(ctx: *mut JSContext, msg: &[u8]) -> JSValue {
    // SAFETY: ctx live; JS_Throw consumes the string value.
    unsafe {
        let v = qjs::JS_NewStringLen(ctx, msg.as_ptr() as *const c_char, msg.len());
        qjs::JS_Throw(ctx, v)
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

unsafe extern "C" fn cb_len(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let n = unsafe { with_state(ctx, data, |s| s.char_len) }.unwrap_or(0);
    // SAFETY: ctx live; trivial value creation.
    unsafe { qjs::sofuu_js_new_int64(ctx, n as i64) }
}

unsafe extern "C" fn cb_count(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let n = unsafe { with_state(ctx, data, |s| s.chunks.len()) }.unwrap_or(0);
    // SAFETY: ctx live; trivial value creation.
    unsafe { qjs::sofuu_js_new_int64(ctx, n as i64) }
}

unsafe extern "C" fn cb_chunk(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let i = unsafe { arg_i64(ctx, argc, argv, 0) };
    let text = unsafe {
        with_state(ctx, data, |s| {
            if i < 0 || i as usize >= s.chunks.len() {
                return None;
            }
            let t = s.chunk_text(i as usize).unwrap_or("").to_string();
            s.push_event("chunk", format!("chunk({}) → {} chars", i, t.chars().count()));
            Some(t)
        })
    };
    match text {
        Some(Some(t)) => {
            // SAFETY: ctx live; t's bytes copied by QuickJS.
            unsafe { qjs::JS_NewStringLen(ctx, t.as_ptr() as *const c_char, t.len()) }
        }
        // Out-of-range chunk indexes surface to the snippet as an error it
        // can read — the episode relays it back to the model as feedback.
        _ => unsafe { throw_str(ctx, b"rlm: chunk index out of range") },
    }
}

unsafe extern "C" fn cb_peek(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let a = unsafe { arg_i64(ctx, argc, argv, 0) };
    let b = unsafe { arg_i64(ctx, argc, argv, 1) };
    let text = unsafe {
        with_state(ctx, data, |s| {
            let a = a.clamp(0, s.char_len as i64) as usize;
            let b = b.clamp(0, s.char_len as i64) as usize;
            let t = s.peek_chars(a, b);
            s.push_event("peek", format!("peek({},{}) → {} chars", a, b, t.chars().count()));
            t
        })
    }
    .unwrap_or_default();
    // SAFETY: ctx live; text's bytes copied by QuickJS.
    unsafe { qjs::JS_NewStringLen(ctx, text.as_ptr() as *const c_char, text.len()) }
}

/// Never returns normally: throws the `\0RLM_SUSPEND:` sentinel carrying
/// the JSON-encoded prompt batch. The host catches it, fulfills the
/// prompts, and re-runs the snippet with the answers cached.
unsafe extern "C" fn cb_llm_batch(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    _data: *mut JSValue,
) -> JSValue {
    let payload = unsafe { arg_string(ctx, argc, argv, 0) };
    let mut buf = b"\0RLM_SUSPEND:".to_vec();
    buf.extend_from_slice(payload.as_bytes());
    unsafe { throw_str(ctx, &buf) }
}

/// PLAN-AGENTS A4 full form: agent-tool suspension. Same shape as
/// `cb_llm_batch` but the payload is a JSON-encoded batch of
/// `[{name, args}]` tool calls — the host executes them through the
/// agent's normal tool path and re-runs the snippet with the results
/// cached (`__TOOL_CACHE`, injected exactly like `__LLM_CACHE`).
unsafe extern "C" fn cb_tool(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    _data: *mut JSValue,
) -> JSValue {
    let payload = unsafe { arg_string(ctx, argc, argv, 0) };
    let mut buf = b"\0RLM_TOOL_SUSPEND:".to_vec();
    buf.extend_from_slice(payload.as_bytes());
    unsafe { throw_str(ctx, &buf) }
}

/// Never returns normally: throws the `\0RLM_FINAL:` sentinel carrying the
/// final answer.
unsafe extern "C" fn cb_final(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    _data: *mut JSValue,
) -> JSValue {
    let payload = unsafe { arg_string(ctx, argc, argv, 0) };
    let mut buf = b"\0RLM_FINAL:".to_vec();
    buf.extend_from_slice(payload.as_bytes());
    unsafe { throw_str(ctx, &buf) }
}

unsafe extern "C" fn cb_emit(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let tag = unsafe { arg_string(ctx, argc, argv, 0) };
    let text = unsafe { arg_string(ctx, argc, argv, 1) };
    let kind = if tag.is_empty() { "emit" } else { tag.as_str() };
    unsafe { with_state(ctx, data, |s| s.push_event(kind, truncate_chars(&text, 4000))) };
    // SAFETY: ctx live; trivial value creation.
    unsafe { qjs::sofuu_js_new_int32(ctx, 0) }
}
