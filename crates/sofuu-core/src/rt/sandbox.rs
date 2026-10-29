// rt/sandbox.rs — Deterministic Agent Sandboxing & Fine-Grained Fuel Meters
// (PLAN-RUNTIME Phase 5; v2 = SANDBOX_UPGRADE_PLAN.md, 2026-10-29).
//
// The sandbox is a standalone QuickJS runtime+context. It provides:
// - Deterministic instruction fuel metering via QuickJS interrupt counters,
//   or *no* meter at all when the caller asks for `fuel: 'unlimited'`.
// - Wall-clock deadlines, or none when the caller asks for `timeoutMs: 0`.
// - Physical memory ceilings (JS_SetMemoryLimit) & call stack caps.
// - Hermetic execution: no host std/os/fs/network/process, unless the
//   developer explicitly bridges a host function into the realm.
// - Callable host bridges (sync and promise-returning) with JSON marshaling.
// - Console capture / inherit / ignore / host callback routing.
// - Optional prototype freezing (intrinsic lock-down) for untrusted code.
// - TypeScript auto-stripping (the same scanner plugins use) and optional
//   JSX → h() transpilation, both in RAM with no build step.
// - One-shot `sofuu.sandbox.run/eval` and stateful `sofuu.sandbox.create`.
//
// v2 principle (RULES.md §1 / plan invariant I2): the runtime never clamps a
// budget the developer states. Defaults stay bounded because an LLM-driven
// `code_eval` must not wedge the loop, but an explicit larger value is
// honoured. `clamp()` therefore enforces only *floors* (a 1 MB / 16 KB
// minimum, where a smaller value is meaningless), never ceilings.
//
// Re-entrancy note: this module is the one place where JS callbacks are
// invoked while another JS call is on the stack (a host bridge called from
// inside a sandbox eval must call back into the host context). The same
// pattern is used by sofuu-ffi's QjsSandbox native whitelist; borrows are
// kept short and every callback re-resolves its state by id instead of
// capturing Rust references across the JS frame.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sofuu_ffi::qjs::{
    self, JSClassDef, JSContext, JSRuntime, JSValue, JSValueConst, JSPromiseStateEnum,
};

static SANDBOX_CLASS_ID: AtomicU32 = AtomicU32::new(0);
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/* id → *mut SandboxSession. The trampolines installed inside a sandbox
 * resolve their session here; the pointer is registered while the session is
 * heap-pinned (Box) and removed before the Box is freed. */
fn session_registry() -> &'static Mutex<HashMap<u64, usize>> {
    static REG: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_session(id: u64, session: *mut SandboxSession) {
    if let Ok(mut reg) = session_registry().lock() {
        reg.insert(id, session as usize);
    }
}

fn unregister_session(id: u64) {
    if let Ok(mut reg) = session_registry().lock() {
        reg.remove(&id);
    }
}

fn lookup_session(id: u64) -> Option<*mut SandboxSession> {
    session_registry()
        .lock()
        .ok()
        .and_then(|reg| reg.get(&id).copied())
        .map(|p| p as *mut SandboxSession)
}

/* id → *mut ActiveRun. A run is "parked" here only while it waits on a host
 * promise; the entry owns the Box until the run finishes. */
fn active_runs() -> &'static Mutex<HashMap<u64, usize>> {
    static ACTIVE: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn insert_active_run(id: u64, run: *mut ActiveRun) {
    if let Ok(mut reg) = active_runs().lock() {
        reg.insert(id, run as usize);
    }
}

fn take_active_run(id: u64) -> Option<*mut ActiveRun> {
    active_runs()
        .lock()
        .ok()
        .and_then(|mut reg| reg.remove(&id))
        .map(|p| p as *mut ActiveRun)
}

/* ── Console routing ─────────────────────────────────────────────
 * `console: 'capture' | 'inherit' | 'ignore' | (level, ...args) => {}`.
 * `captureConsole: false` (the 0.3 API) still means 'ignore'. */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleMode {
    Capture,
    Inherit,
    Ignore,
    Callback,
}

/* ── TypeScript ingestion ────────────────────────────────────────
 * Auto = run the code as written first and only retry the stripped source
 * when the first attempt fails to *parse*. That cannot break valid JS (a
 * stripper bug only affects code that was already a SyntaxError) and costs
 * one extra compile on genuinely-TypeScript input. */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsMode {
    Auto,
    Always,
    Never,
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// None = unmetered (`fuel: 'unlimited'` / `0` / null).
    pub fuel_limit: Option<u64>,
    /// None = no wall-clock deadline (`timeoutMs: 0` / null).
    pub timeout: Option<Duration>,
    pub max_memory_bytes: usize,
    pub max_stack_bytes: usize,
    pub console: ConsoleMode,
    /// 0 = unbounded (bounded in practice by MAX_LOG_TOTAL_BYTES).
    pub max_logs: usize,
    pub allow_async: bool,
    pub throw_on_error: bool,
    pub freeze_prototypes: bool,
    pub ts_mode: TsMode,
    pub jsx: bool,
}

/* Defaults are the plan's "safe default for agent tools": a 2 s deadline and
 * a 5M instruction meter. They are defaults, not caps — see clamp(). */
pub const SANDBOX_DEFAULT_FUEL: u64 = 5_000_000;
pub const SANDBOX_DEFAULT_TIMEOUT_MS: u64 = 2_000;
pub const SANDBOX_DEFAULT_MEMORY_BYTES: usize = 64 * 1024 * 1024;
pub const SANDBOX_DEFAULT_STACK_BYTES: usize = 512 * 1024;
pub const SANDBOX_DEFAULT_MAX_LOGS: usize = 512;
pub const SANDBOX_MIN_MEMORY_BYTES: usize = 1024 * 1024;
pub const SANDBOX_MIN_STACK_BYTES: usize = 16 * 1024;
/// One captured console line cannot exceed this (a Rust Vec is outside the
/// JS memory limit, so an uncapped line would bypass it).
pub const SANDBOX_MAX_LOG_LINE_BYTES: usize = 64 * 1024;
/// Total captured console bytes per run; beyond this, lines are dropped.
pub const SANDBOX_MAX_LOG_TOTAL_BYTES: usize = 8 * 1024 * 1024;
/// Safety bound on microtask pumping (a pathological promise chain that
/// re-queues forever must not spin the host thread with no deadline).
const MAX_PUMP_ITERATIONS: u32 = 100_000;

impl SandboxConfig {
    /// Enforce *floors* only. A developer-specified `timeoutMs`, `fuel`,
    /// `maxMemoryMB` or `maxStackKB` is never reduced (invariant I2).
    pub fn clamp(&mut self) {
        self.max_memory_bytes = self.max_memory_bytes.max(SANDBOX_MIN_MEMORY_BYTES);
        self.max_stack_bytes = self.max_stack_bytes.max(SANDBOX_MIN_STACK_BYTES);
    }

    /// The four presets from the plan table. Explicit options override them.
    pub fn preset(name: &str) -> Option<Self> {
        let mut c = SandboxConfig::default();
        match name {
            "strict" => {
                c.fuel_limit = Some(5_000_000);
                c.timeout = Some(Duration::from_millis(2_000));
                c.max_memory_bytes = 32 * 1024 * 1024;
                c.freeze_prototypes = true;
                c.ts_mode = TsMode::Auto;
            }
            "standard" => {
                c.fuel_limit = Some(20_000_000);
                c.timeout = Some(Duration::from_millis(10_000));
                c.max_memory_bytes = 64 * 1024 * 1024;
                c.freeze_prototypes = false;
                c.ts_mode = TsMode::Auto;
            }
            "compute" => {
                c.fuel_limit = None;
                c.timeout = Some(Duration::from_millis(300_000));
                c.max_memory_bytes = 512 * 1024 * 1024;
                c.freeze_prototypes = false;
                c.ts_mode = TsMode::Auto;
            }
            "worker" => {
                c.fuel_limit = None;
                c.timeout = None;
                c.max_memory_bytes = 256 * 1024 * 1024;
                c.freeze_prototypes = false;
                c.ts_mode = TsMode::Auto;
            }
            _ => return None,
        }
        Some(c)
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            fuel_limit: Some(SANDBOX_DEFAULT_FUEL),
            timeout: Some(Duration::from_millis(SANDBOX_DEFAULT_TIMEOUT_MS)),
            max_memory_bytes: SANDBOX_DEFAULT_MEMORY_BYTES,
            max_stack_bytes: SANDBOX_DEFAULT_STACK_BYTES,
            console: ConsoleMode::Capture,
            max_logs: SANDBOX_DEFAULT_MAX_LOGS,
            allow_async: true,
            throw_on_error: false,
            freeze_prototypes: false,
            ts_mode: TsMode::Auto,
            jsx: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptReason {
    None,
    FuelExhausted,
    Timeout,
}

pub struct InterruptState {
    pub ctx: *mut JSContext,
    pub deadline: Option<Instant>,
    pub fuel_limit: Option<u64>,
    pub fuel_remaining: i64,
    pub fuel_consumed: u64,
    pub fuel_quanta: u64,
    pub reason: InterruptReason,
}

/* C-02 (2026-10-29): a sanitized alias for the machine-readable error codes
 * produced by the sandbox. `run()` reports them on the result object
 * (`code`) and `eval()` / `throwOnError: true` throw an Error carrying them,
 * so a host can distinguish a deadline from a fuel cut without string
 * matching the message. */
pub fn sandbox_error_code(error_type: &str) -> &'static str {
    match error_type {
        "timeout" => "E_TIMEOUT",
        "fuel" => "E_FUEL",
        "oom" => "E_OOM",
        "stack" => "E_STACK",
        "closed" => "E_SANDBOX_CLOSED",
        "busy" => "E_BUSY",
        "bridge" => "E_BRIDGE",
        "syntax" => "E_SYNTAX",
        "globals" => "E_GLOBALS",
        _ => "E_SANDBOX",
    }
}

/* C-02: defined in deps/quickjs/quickjs.c. Non-zero when the interrupt
 * handler is being called from the native regexp backtracking loop instead
 * of a bytecode meter batch. Declared here (rather than in the shared
 * sofuu-ffi qjs module) because the sandbox is its only caller. */
extern "C" {
    fn JS_GetInterruptPollSource(rt: *mut JSRuntime) -> c_int;
}

/* Must match LRE_INTERRUPT_POLL_STEPS in deps/quickjs/libregexp.h: the
 * engine charges one fuel unit per backtracking step, and a poll covers
 * that many steps. Kept as a literal so the Rust side needs no build-time
 * binding; the unit test below pins the two values together. */
const REGEXP_POLL_STEPS: i64 = 4096;

unsafe extern "C" fn sandbox_interrupt_cb(rt: *mut JSRuntime, opaque: *mut c_void) -> c_int {
    if opaque.is_null() {
        return 0;
    }
    let st = unsafe { &mut *(opaque as *mut InterruptState) };
    let engine_poll = !rt.is_null() && JS_GetInterruptPollSource(rt) != 0;

    if engine_poll {
        /* Native regexp backtracking. It is only charged to the fuel meter
         * when the run has no wall-clock budget: the meter counts work in
         * bytecode-instruction units, and with a deadline configured the
         * clock is the authoritative budget for native matching (charging
         * 1 unit per step there would let the default 5M budget preempt any
         * deadline, turning `timeoutMs: 100` into a fuel error). With no
         * deadline (Rust API `timeout: None`) the meter must still bound the
         * match, so one fuel unit per step is charged here. */
        if st.deadline.is_none() && st.fuel_limit.is_some() {
            st.fuel_consumed = st.fuel_consumed.saturating_add(REGEXP_POLL_STEPS as u64);
            st.fuel_remaining -= REGEXP_POLL_STEPS;
            if st.fuel_remaining <= 0 {
                st.reason = InterruptReason::FuelExhausted;
                return 1; // abort!
            }
        }
    } else {
        // Record fuel for the batch that just completed
        let batch = st.fuel_quanta;
        st.fuel_consumed = st.fuel_consumed.saturating_add(batch);

        // Fuel check
        if st.fuel_limit.is_some() {
            st.fuel_remaining -= batch as i64;
            if st.fuel_remaining <= 0 {
                st.reason = InterruptReason::FuelExhausted;
                return 1; // abort!
            }
            if !st.ctx.is_null() {
                let next_batch = (st.fuel_remaining.max(1) as u64).min(10_000).max(1);
                st.fuel_quanta = next_batch;
                qjs::JS_SetInterruptCounter(st.ctx, next_batch as c_int);
            }
        }
    }

    // Deadline check
    if let Some(deadline) = st.deadline {
        if Instant::now() >= deadline {
            st.reason = InterruptReason::Timeout;
            return 1; // abort!
        }
    }

    0
}

// ── Console ──────────────────────────────────────────────────────

pub struct SandboxContextState {
    pub logs: Vec<String>,
    pub total_bytes: usize,
    pub console_mode: ConsoleMode,
    pub max_logs: usize,
    pub host_ctx: *mut JSContext,
    /// Host-owned. Only valid when console_mode == Callback.
    pub console_cb: JSValue,
}

const CONSOLE_LEVELS: [&str; 5] = ["log", "info", "warn", "error", "debug"];

unsafe extern "C" fn sandbox_console_fn(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    magic: c_int,
    _data: *mut JSValue,
) -> JSValue {
    let opaque = qjs::JS_GetContextOpaque(ctx);
    if opaque.is_null() {
        return qjs::sofuu_js_undefined();
    }
    let st = unsafe { &mut *(opaque as *mut SandboxContextState) };
    let level_idx = (magic.clamp(0, 4)) as usize;

    let mut parts: Vec<String> = Vec::new();
    for i in 0..argc {
        let v = unsafe { *argv.add(i as usize) };
        let cstr = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
        if !cstr.is_null() {
            let mut s = unsafe { CStr::from_ptr(cstr) }.to_string_lossy().into_owned();
            if s.len() > SANDBOX_MAX_LOG_LINE_BYTES {
                s.truncate(SANDBOX_MAX_LOG_LINE_BYTES);
                s.push('…');
            }
            parts.push(s);
            unsafe { qjs::sofuu_js_free_cstring(ctx, cstr) };
        }
    }

    match st.console_mode {
        ConsoleMode::Ignore => {}
        ConsoleMode::Inherit => {
            let line = parts.join(" ");
            match level_idx {
                2 | 3 => eprintln!("{line}"),
                _ => println!("{line}"),
            }
        }
        ConsoleMode::Callback => {
            unsafe { call_console_callback(st, level_idx, &parts) };
        }
        ConsoleMode::Capture => {
            if st.max_logs == 0 || st.logs.len() < st.max_logs {
                let mut line = parts.join(" ");
                if st.total_bytes.saturating_add(line.len()) > SANDBOX_MAX_LOG_TOTAL_BYTES {
                    return qjs::sofuu_js_undefined();
                }
                st.total_bytes += line.len();
                if st.total_bytes > SANDBOX_MAX_LOG_TOTAL_BYTES {
                    line.push_str(" [log budget reached]");
                }
                st.logs.push(line);
            }
        }
    }
    qjs::sofuu_js_undefined()
}

unsafe fn call_console_callback(st: &SandboxContextState, level_idx: usize, parts: &[String]) {
    let host = st.host_ctx;
    if host.is_null() || qjs::is_undefined(st.console_cb) {
        return;
    }
    unsafe {
        /* argv = [level, ...args]. JS_Call does not consume the values, so
         * each one is freed after the call returns. */
        let mut argv: Vec<JSValue> = Vec::with_capacity(parts.len() + 1);
        let lvl = CString::new(CONSOLE_LEVELS[level_idx]).unwrap_or_default();
        argv.push(qjs::sofuu_js_new_string(host, lvl.as_ptr()));
        for p in parts {
            let c = CString::new(p.as_str()).unwrap_or_default();
            argv.push(qjs::sofuu_js_new_string(host, c.as_ptr()));
        }
        let ret = qjs::JS_Call(
            host,
            st.console_cb,
            qjs::sofuu_js_undefined(),
            argv.len() as c_int,
            argv.as_ptr(),
        );
        for v in argv {
            qjs::sofuu_js_free_value(host, v);
        }
        if qjs::is_exception(ret) {
            let _ = take_exception_message(host);
        } else {
            qjs::sofuu_js_free_value(host, ret);
        }
    }
}

unsafe fn install_console(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let console = qjs::sofuu_js_new_object(ctx);

    for (idx, name) in CONSOLE_LEVELS.iter().enumerate() {
        let cname = CString::new(*name).unwrap_or_default();
        let fn_val = qjs::JS_NewCFunctionData(ctx, sandbox_console_fn, 0, idx as c_int, 0, std::ptr::null());
        qjs::sofuu_js_set_property_str(ctx, console, cname.as_ptr(), fn_val);
    }
    qjs::sofuu_js_set_property_str(ctx, global, c"console".as_ptr(), console);
    qjs::sofuu_js_free_value(ctx, global);
}

/* Minimal `h()`/`Fragment` so `jsx: true` output is evaluable in the
 * sandbox: a component becomes a plain {type, props, children} tree. */
const JSX_HELPERS_JS: &str = r#"(function () {
  function h(type, props) {
    var children = [];
    for (var i = 2; i < arguments.length; i++) children.push(arguments[i]);
    return { type: type, props: props || {}, children: children };
  }
  globalThis.h = h;
  globalThis.Fragment = 'Fragment';
})();"#;

const FREEZE_PROTOTYPES_JS: &str = r#"(function () {
  var intrinsics = [Object, Array, Function, String, Number, Boolean, Promise, RegExp, Error];
  for (var i = 0; i < intrinsics.length; i++) {
    try { Object.freeze(intrinsics[i]); } catch (e) {}
    try { Object.freeze(intrinsics[i].prototype); } catch (e) {}
  }
})();"#;

/// Run a tiny internal snippet in a fresh sandbox realm. Used only for the
/// JSX helpers and the prototype freeze; failures are non-fatal (the run
/// still proceeds, and the feature simply does not apply).
unsafe fn eval_internal(ctx: *mut JSContext, src: &str) {
    if let Ok(c) = CString::new(src) {
        let v = qjs::JS_Eval(ctx, c.as_ptr(), src.len(), c"<sandbox-init>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
        if unsafe { qjs::is_exception(v) } {
            let _ = unsafe { take_exception_message(ctx) };
        }
        unsafe { qjs::sofuu_js_free_value(ctx, v) };
    }
}

// ── Small JS helpers ────────────────────────────────────────────

unsafe fn js_to_rust_string(ctx: *mut JSContext, v: JSValueConst) -> String {
    unsafe {
        let c = qjs::sofuu_js_to_cstring(ctx, v);
        if c.is_null() {
            return String::new();
        }
        let s = CStr::from_ptr(c).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, c);
        s
    }
}

unsafe fn take_exception_message(ctx: *mut JSContext) -> String {
    unsafe {
        let exc = qjs::sofuu_js_get_exception(ctx);
        let mut msg = String::new();
        if !qjs::is_undefined(exc) && !qjs::is_null(exc) {
            msg = js_to_rust_string(ctx, exc);
        }
        qjs::sofuu_js_free_value(ctx, exc);
        if msg.is_empty() {
            "Unknown sandbox exception".to_string()
        } else {
            msg
        }
    }
}

unsafe fn json_stringify(ctx: *mut JSContext, v: JSValueConst) -> Result<String, String> {
    unsafe {
        let j = qjs::JS_JSONStringify(
            ctx,
            v,
            qjs::sofuu_js_undefined(),
            qjs::sofuu_js_undefined(),
        );
        if qjs::is_exception(j) {
            let msg = take_exception_message(ctx);
            return Err(format!("value is not JSON-serializable: {msg}"));
        }
        if qjs::is_undefined(j) {
            qjs::sofuu_js_free_value(ctx, j);
            return Err("value is not JSON-serializable (undefined)".to_string());
        }
        let s = js_to_rust_string(ctx, j);
        qjs::sofuu_js_free_value(ctx, j);
        Ok(s)
    }
}

unsafe fn json_parse(ctx: *mut JSContext, s: &str) -> Result<JSValue, String> {
    let c = CString::new(s).map_err(|_| "JSON contains NUL".to_string())?;
    unsafe {
        let v = qjs::JS_ParseJSON(ctx, c.as_ptr(), s.len(), c"<bridge>".as_ptr());
        if qjs::is_exception(v) {
            let msg = take_exception_message(ctx);
            qjs::sofuu_js_free_value(ctx, v);
            return Err(msg);
        }
        Ok(v)
    }
}

/// Build a plain Error with `message` inside `ctx` (never throws).
unsafe fn make_error_object(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe {
        let err = qjs::JS_NewError(ctx);
        if qjs::is_exception(err) {
            return qjs::sofuu_js_undefined();
        }
        let c = CString::new(msg).unwrap_or_else(|_| CString::new("sandbox error").unwrap());
        qjs::sofuu_js_set_property_str(ctx, err, c"message".as_ptr(), qjs::sofuu_js_new_string(ctx, c.as_ptr()));
        err
    }
}

/// Throw a plain Error with `message`; returns the exception sentinel.
unsafe fn throw_error_msg(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe {
        let err = make_error_object(ctx, msg);
        if qjs::is_exception(err) {
            return err;
        }
        qjs::JS_Throw(ctx, err)
    }
}

unsafe fn inject_globals_json(ctx: *mut JSContext, json_str: &str) -> Result<(), String> {
    if json_str.trim().is_empty() || json_str == "{}" {
        return Ok(());
    }
    let parsed = unsafe { json_parse(ctx, json_str)? };
    if !unsafe { qjs::is_object(parsed) } {
        unsafe { qjs::sofuu_js_free_value(ctx, parsed) };
        return Ok(());
    }

    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let mut len: u32 = 0;
    let mut tab: *mut qjs::JSPropertyEnum = std::ptr::null_mut();
    if unsafe {
        qjs::JS_GetOwnPropertyNames(
            ctx,
            &mut tab,
            &mut len,
            parsed,
            (qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_SYMBOL_MASK) as c_int,
        )
    } == 0
    {
        for i in 0..len {
            let prop = unsafe { *tab.add(i as usize) };
            let key_cstr = unsafe { qjs::JS_AtomToCString(ctx, prop.atom) };
            if !key_cstr.is_null() {
                let val = unsafe { qjs::sofuu_js_get_property(ctx, parsed, prop.atom) };
                unsafe { qjs::sofuu_js_set_property_str(ctx, global, key_cstr, val) };
                unsafe { qjs::JS_FreeCString(ctx, key_cstr) };
            }
            unsafe { qjs::JS_FreeAtom(ctx, prop.atom) };
        }
        unsafe { qjs::js_free(ctx, tab as *mut c_void) };
    }
    unsafe {
        qjs::sofuu_js_free_value(ctx, parsed);
        qjs::sofuu_js_free_value(ctx, global);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SandboxRunResult {
    pub success: bool,
    pub result_json: Option<String>,
    pub error: Option<String>,
    pub error_type: Option<String>,
    pub interrupted: bool,
    pub fuel_used: u64,
    pub elapsed_ms: f64,
    pub logs: Vec<String>,
}

impl SandboxRunResult {
    fn empty() -> Self {
        Self {
            success: false,
            result_json: None,
            error: Some("Sandbox run did not complete".to_string()),
            error_type: Some("runtime".to_string()),
            interrupted: false,
            fuel_used: 0,
            elapsed_ms: 0.0,
            logs: Vec::new(),
        }
    }
}

/// The outcome of `run_ex`. `root: Some(v)` means the run parked on a host
/// promise that has not settled yet; `v` is the sandbox-owned promise value
/// the caller must keep (and free) after the run completes.
pub struct RunState {
    pub result: SandboxRunResult,
    pub root: Option<JSValue>,
}

enum PumpOutcome {
    Fulfilled(JSValue),
    Rejected(JSValue),
    /// A host bridge is unresolved; resume from its settle callback.
    Suspended,
    /// The root promise can never settle and no bridge is outstanding: the
    /// legacy behaviour is to serialize the promise object itself ("{}").
    Stuck,
    Timeout,
    Failed(String),
}

struct HostBridge {
    name: String,
    /// Owned by the host context (`SandboxSession::host_ctx`).
    host_fn: JSValue,
}

struct PendingBridge {
    resolve: JSValue,
    reject: JSValue,
}

pub struct SandboxSession {
    pub id: u64,
    rt: *mut JSRuntime,
    ctx: *mut JSContext,
    interrupt_state: *mut InterruptState,
    context_state: *mut SandboxContextState,
    config: SandboxConfig,
    closed: bool,
    /// Host side of the bridge (the JS context that called sandbox.run).
    host_ctx: *mut JSContext,
    host_rt: *mut JSRuntime,
    /// True only when host jobs must be pumped (bridges installed).
    pump_host: bool,
    bridges: Vec<HostBridge>,
    /// Host-owned console callback (console: fn).
    console_cb: Option<JSValue>,
    pending_bridges: HashMap<u64, PendingBridge>,
    next_bridge_token: u64,
    busy: bool,
    pending_close: bool,
}

impl SandboxSession {
    pub fn new(mut config: SandboxConfig) -> Result<Self, String> {
        config.clamp();
        let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        unsafe {
            let rt = qjs::JS_NewRuntime();
            if rt.is_null() {
                return Err("Failed to allocate QuickJS runtime for sandbox".to_string());
            }

            qjs::JS_SetMemoryLimit(rt, config.max_memory_bytes);
            qjs::JS_SetMaxStackSize(rt, config.max_stack_bytes);

            let interrupt_state = Box::into_raw(Box::new(InterruptState {
                ctx: std::ptr::null_mut(),
                deadline: None,
                fuel_limit: config.fuel_limit,
                fuel_remaining: config.fuel_limit.map(|f| f as i64).unwrap_or(i64::MAX),
                fuel_consumed: 0,
                fuel_quanta: 0,
                reason: InterruptReason::None,
            }));

            let ctx = qjs::JS_NewContext(rt);
            if ctx.is_null() {
                qjs::JS_FreeRuntime(rt);
                drop(Box::from_raw(interrupt_state));
                return Err("Failed to allocate QuickJS context for sandbox".to_string());
            }

            (*interrupt_state).ctx = ctx;

            let context_state = Box::into_raw(Box::new(SandboxContextState {
                logs: Vec::new(),
                total_bytes: 0,
                console_mode: config.console,
                max_logs: config.max_logs,
                host_ctx: std::ptr::null_mut(),
                console_cb: qjs::sofuu_js_undefined(),
            }));

            qjs::JS_SetContextOpaque(ctx, context_state as *mut c_void);

            install_console(ctx);
            if config.jsx {
                eval_internal(ctx, JSX_HELPERS_JS);
            }
            if config.freeze_prototypes {
                eval_internal(ctx, FREEZE_PROTOTYPES_JS);
            }

            Ok(Self {
                id,
                rt,
                ctx,
                interrupt_state,
                context_state,
                config,
                closed: false,
                host_ctx: std::ptr::null_mut(),
                host_rt: std::ptr::null_mut(),
                pump_host: false,
                bridges: Vec::new(),
                console_cb: None,
                pending_bridges: HashMap::new(),
                next_bridge_token: 1,
                busy: false,
                pending_close: false,
            })
        }
    }

    /// Register the session so sandbox-side trampolines can find it. Must be
    /// called after the session is heap-pinned (Box) — the registry stores
    /// the address.
    pub fn register(&self) {
        register_session(self.id, self as *const Self as *mut SandboxSession);
    }

    /// Create the sandbox-side function objects for host bridges and record
    /// the host context used to call back into them.
    pub fn install_host_services(
        &mut self,
        host_ctx: *mut JSContext,
        bridges: Vec<(String, JSValue)>,
        console_cb: Option<JSValue>,
    ) -> Result<(), String> {
        if host_ctx.is_null() {
            return Err("sandbox host context is null".to_string());
        }
        self.host_ctx = host_ctx;
        self.host_rt = unsafe { qjs::JS_GetRuntime(host_ctx) };
        self.console_cb = console_cb;

        unsafe {
            let cst = &mut *self.context_state;
            cst.host_ctx = host_ctx;
            cst.console_cb = console_cb.unwrap_or_else(|| qjs::sofuu_js_undefined());
        }

        if bridges.is_empty() {
            self.pump_host = false;
            return Ok(());
        }
        self.pump_host = true;

        /* `pending` keeps ownership of the host function values until each
         * one is either installed (moved into `self.bridges`) or freed. An
         * early return would otherwise drop JSValues without releasing their
         * host references. */
        let mut pending: std::collections::VecDeque<(String, JSValue)> = bridges.into();
        let mut index = 0usize;
        unsafe {
            let global = qjs::sofuu_js_get_global_object(self.ctx);
            while let Some((name, host_fn)) = pending.pop_front() {
                let data = [
                    qjs::sofuu_js_new_int64(self.ctx, self.id as i64),
                    qjs::sofuu_js_new_int32(self.ctx, index as i32),
                ];
                let f =
                    qjs::JS_NewCFunctionData(self.ctx, bridge_trampoline, 0, 0, 2, data.as_ptr());
                qjs::sofuu_js_free_value(self.ctx, data[0]);
                qjs::sofuu_js_free_value(self.ctx, data[1]);
                let installed = if qjs::is_exception(f) {
                    Err("Failed to allocate a sandbox bridge function".to_string())
                } else {
                    let cname = CString::new(name.clone()).unwrap_or_default();
                    let rc = qjs::sofuu_js_set_property_str(self.ctx, global, cname.as_ptr(), f);
                    if rc < 0 {
                        Err(format!("Failed to install sandbox bridge '{name}'"))
                    } else {
                        Ok(())
                    }
                };
                match installed {
                    Ok(()) => {
                        self.bridges.push(HostBridge { name, host_fn });
                        index += 1;
                    }
                    Err(e) => {
                        qjs::sofuu_js_free_value(host_ctx, host_fn);
                        for (_, rest) in pending.drain(..) {
                            qjs::sofuu_js_free_value(host_ctx, rest);
                        }
                        qjs::sofuu_js_free_value(self.ctx, global);
                        return Err(e);
                    }
                }
            }
            qjs::sofuu_js_free_value(self.ctx, global);
        }
        Ok(())
    }

    /// Arm the wall-clock deadline and the fuel meter for one run (or resume)
    /// and install `sandbox_interrupt_cb`. The runtime is private to this
    /// session, so no pre-existing interrupt handler can be clobbered.
    unsafe fn arm_interrupts(&mut self, run_config: &SandboxConfig, start: Instant, resume: bool) {
        let st = unsafe { &mut *self.interrupt_state };
        st.ctx = self.ctx;
        st.deadline = run_config.timeout.map(|d| start + d);
        st.fuel_limit = run_config.fuel_limit;
        if !resume {
            st.fuel_consumed = 0;
            st.fuel_remaining = run_config.fuel_limit.map(|f| f as i64).unwrap_or(i64::MAX);
            st.reason = InterruptReason::None;
        }

        /* With no fuel meter there is nothing to charge; poll rarely (only a
         * deadline needs polls, and a 2B-op batch is the coarse bound). */
        let counter: i32 = if run_config.fuel_limit.is_some() {
            st.fuel_remaining.max(1).min(10_000) as i32
        } else if run_config.timeout.is_some() {
            10_000
        } else {
            i32::MAX
        };
        st.fuel_quanta = counter as u64;
        unsafe { qjs::JS_SetInterruptCounter(self.ctx, counter) };

        unsafe {
            qjs::JS_SetInterruptHandler(
                self.rt,
                Some(sandbox_interrupt_cb),
                self.interrupt_state as *mut c_void,
            );
        }
    }

    /// Remove the handler and clear the deadline/fuel state so nothing can
    /// interrupt (or be mis-metered by) execution that is not a sandbox run.
    unsafe fn disarm_interrupts(&mut self) {
        if !self.rt.is_null() {
            unsafe { qjs::JS_SetInterruptHandler(self.rt, None, std::ptr::null_mut()) };
        }
        if !self.interrupt_state.is_null() {
            let st = unsafe { &mut *self.interrupt_state };
            st.deadline = None;
            st.fuel_limit = None;
            /* `reason` is deliberately kept: the caller reads it right after
             * this returns to classify the failure (fuel vs deadline vs a
             * plain throw). The next run resets it when it arms fresh. */
        }
    }

    /// Fold the in-flight counter batch into the accumulated total. Called
    /// before parking a run and at completion, so a resumed run continues
    /// from the same meter instead of restarting it.
    unsafe fn settle_fuel(&mut self) {
        if self.interrupt_state.is_null() || self.ctx.is_null() {
            return;
        }
        let remaining = unsafe { qjs::JS_GetInterruptCounter(self.ctx) }.max(0) as u64;
        let st = unsafe { &mut *self.interrupt_state };
        let executed = st.fuel_quanta.saturating_sub(remaining);
        st.fuel_consumed = st.fuel_consumed.saturating_add(executed);
        if st.fuel_limit.is_some() {
            st.fuel_remaining -= executed as i64;
        }
        st.fuel_quanta = 0;
    }

    unsafe fn fuel_total(&self) -> u64 {
        if self.interrupt_state.is_null() || self.ctx.is_null() {
            return 0;
        }
        let remaining = unsafe { qjs::JS_GetInterruptCounter(self.ctx) }.max(0) as u64;
        let st = unsafe { &*self.interrupt_state };
        st.fuel_consumed
            .saturating_add(st.fuel_quanta.saturating_sub(remaining))
    }

    fn reason(&self) -> InterruptReason {
        if self.interrupt_state.is_null() {
            InterruptReason::None
        } else {
            unsafe { (*self.interrupt_state).reason }
        }
    }

    fn logs(&self) -> Vec<String> {
        if self.context_state.is_null() {
            Vec::new()
        } else {
            unsafe { (*self.context_state).logs.clone() }
        }
    }

    /// Compile only. Separating compilation from execution is what makes the
    /// ingest retries safe: a candidate is retried only when it fails to
    /// *parse*, so sandbox code with side effects (a bridge call, a console
    /// line) is never executed twice just because it threw at runtime.
    unsafe fn compile_once(&mut self, src: &str) -> Result<JSValue, String> {
        let c = CString::new(src).map_err(|_| "Code string contains NUL byte".to_string())?;
        let r = unsafe {
            qjs::JS_Eval(
                self.ctx,
                c.as_ptr(),
                src.len(),
                c"<sandbox>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL | qjs::JS_EVAL_FLAG_COMPILE_ONLY,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            let msg = unsafe { take_exception_message(self.ctx) };
            unsafe { qjs::sofuu_js_free_value(self.ctx, r) };
            Err(msg)
        } else {
            Ok(r)
        }
    }

    /// Execute a compiled script function. `JS_EvalFunction` frees `fun` on
    /// every path (quickjs.c: js_closure + JS_CallFree), so it is not freed
    /// again here.
    unsafe fn execute_compiled(&mut self, fun: JSValue) -> Result<JSValue, String> {
        let r = unsafe { qjs::JS_EvalFunction(self.ctx, fun) };
        if unsafe { qjs::is_exception(r) } {
            let msg = unsafe { take_exception_message(self.ctx) };
            unsafe { qjs::sofuu_js_free_value(self.ctx, r) };
            Err(msg)
        } else {
            Ok(r)
        }
    }

    /// W2: source ingestion. Candidates are tried in order and a candidate is
    /// only skipped while the previous one failed to *parse* (a SyntaxError):
    /// a real runtime throw is final, so `throw new Error()` is never retried
    /// under a different wrapper and its message is never replaced.
    ///
    ///   1. the code as written
    ///   2. the TypeScript-stripped code      (ts: 'auto')
    ///   3. the code in an async IIFE         (top-level await)
    ///   4. the stripped code in an async IIFE
    ///
    /// The async wrapper means a top-level `await` works, but its value comes
    /// from an explicit `return` — the script-mode last-expression shorthand
    /// cannot survive a function body without a parser (docs/SANDBOX.md).
    unsafe fn eval_with_ingest(
        &mut self,
        src: &str,
        cfg: &SandboxConfig,
    ) -> Result<JSValue, String> {
        let stripped = if cfg.ts_mode == TsMode::Auto {
            let s = crate::ts::strip(src);
            if s == src { None } else { Some(s) }
        } else {
            None
        };

        let mut candidates: Vec<String> = vec![src.to_string()];
        if let Some(ref s) = stripped {
            candidates.push(s.clone());
        }
        if cfg.allow_async {
            candidates.push(async_wrap(src));
            if let Some(ref s) = stripped {
                candidates.push(async_wrap(s));
            }
        }

        let mut first_error: Option<String> = None;
        let mut compiled: Option<JSValue> = None;
        for candidate in candidates {
            match unsafe { self.compile_once(&candidate) } {
                Ok(fun) => {
                    compiled = Some(fun);
                    break;
                }
                Err(msg) => {
                    let parse_error = msg.contains("SyntaxError");
                    if first_error.is_none() {
                        first_error = Some(msg.clone());
                    }
                    if !parse_error {
                        return Err(msg);
                    }
                }
            }
        }
        match compiled {
            Some(fun) => unsafe { self.execute_compiled(fun) },
            None => Err(first_error.unwrap_or_else(|| "sandbox: no evaluable source".to_string())),
        }
    }

    /// Execute `code`. Returns `root: Some` only when the run parked on a
    /// host promise; the caller owns that value and must resume it.
    pub fn run_ex(
        &mut self,
        code: &str,
        run_config: &SandboxConfig,
        globals_json: Option<&str>,
        start: Instant,
    ) -> RunState {
        let mut cfg = run_config.clone();
        cfg.clamp();

        if self.closed {
            let mut r = SandboxRunResult::empty();
            r.error = Some("Sandbox session is closed".to_string());
            r.error_type = Some("closed".to_string());
            return RunState { result: r, root: None };
        }
        if self.busy {
            let mut r = SandboxRunResult::empty();
            r.error = Some("E_BUSY: a run is still parked on an async host bridge".to_string());
            r.error_type = Some("busy".to_string());
            return RunState { result: r, root: None };
        }

        /* Leftovers from an abandoned run would resolve into a dead realm. */
        unsafe { self.discard_pending_bridges() };

        unsafe {
            let cst = &mut *self.context_state;
            cst.logs.clear();
            cst.total_bytes = 0;
            cst.console_mode = cfg.console;
            cst.max_logs = cfg.max_logs;
            cst.host_ctx = self.host_ctx;
            cst.console_cb = self.console_cb.unwrap_or_else(|| qjs::sofuu_js_undefined());
        }

        if let Some(json) = globals_json {
            if let Err(e) = unsafe { inject_globals_json(self.ctx, json) } {
                let mut r = SandboxRunResult::empty();
                r.error = Some(e);
                r.error_type = Some("globals".to_string());
                r.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                return RunState { result: r, root: None };
            }
        }

        let src = prepare_source(code, &cfg);

        unsafe { self.arm_interrupts(&cfg, start, false) };
        let attempt = unsafe { self.eval_with_ingest(&src, &cfg) };

        let val = match attempt {
            Ok(v) => v,
            Err(msg) => {
                unsafe {
                    self.disarm_interrupts();
                    self.settle_fuel();
                }
                let result = self.failure_from_message(msg, &cfg, start);
                return RunState { result, root: None };
            }
        };

        let is_promise =
            unsafe { qjs::JS_PromiseState(self.ctx, val) } != JSPromiseStateEnum::Invalid;

        if !cfg.allow_async || !is_promise {
            let json = unsafe { serialize_val_to_json(self.ctx, val) };
            unsafe { qjs::sofuu_js_free_value(self.ctx, val) };
            unsafe { self.disarm_interrupts() };
            let result = self.success_result(json, start);
            return RunState { result, root: None };
        }

        let outcome = unsafe { pump(self, val, &cfg, start) };
        unsafe {
            self.disarm_interrupts();
            self.settle_fuel();
        }

        match outcome {
            PumpOutcome::Suspended => RunState {
                result: SandboxRunResult::empty(),
                root: Some(val),
            },
            PumpOutcome::Stuck => {
                let json = unsafe { serialize_val_to_json(self.ctx, val) };
                unsafe { qjs::sofuu_js_free_value(self.ctx, val) };
                let result = self.success_result(json, start);
                RunState { result, root: None }
            }
            other => {
                let undefined = unsafe { qjs::sofuu_js_undefined() };
                unsafe { qjs::sofuu_js_free_value(self.ctx, val) };
                let result = self.assemble(&cfg, start, other, undefined);
                RunState { result, root: None }
            }
        }
    }

    /// Synchronous entry point (Rust API, tests). A run that parks on an
    /// async host bridge cannot be resumed without a JS caller, so it is
    /// reported as a typed bridge error rather than leaked.
    pub fn run(
        &mut self,
        code: &str,
        run_config: &SandboxConfig,
        globals_json: Option<&str>,
    ) -> SandboxRunResult {
        let start = Instant::now();
        let mut state = self.run_ex(code, run_config, globals_json, start);
        if let Some(root) = state.root.take() {
            unsafe { qjs::sofuu_js_free_value(self.ctx, root) };
            unsafe { self.discard_pending_bridges() };
            let mut r = state.result;
            r.success = false;
            r.error = Some(
                "E_BRIDGE: sandbox run parked on an async host bridge without a host context"
                    .to_string(),
            );
            r.error_type = Some("bridge".to_string());
            r.interrupted = false;
            return r;
        }
        state.result
    }

    fn success_result(&self, result_json: Option<String>, start: Instant) -> SandboxRunResult {
        SandboxRunResult {
            success: true,
            result_json,
            error: None,
            error_type: None,
            interrupted: false,
            fuel_used: unsafe { self.fuel_total() },
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            logs: self.logs(),
        }
    }

    fn failure_from_message(
        &self,
        msg: String,
        cfg: &SandboxConfig,
        start: Instant,
    ) -> SandboxRunResult {
        let reason = self.reason();
        let (message, etype, interrupted) = match reason {
            InterruptReason::FuelExhausted => (
                format!(
                    "Execution terminated: instruction fuel exhausted ({} limit reached)",
                    cfg.fuel_limit.unwrap_or(0)
                ),
                "fuel".to_string(),
                true,
            ),
            InterruptReason::Timeout => (
                format!(
                    "E_TIMEOUT: sandbox execution exceeded its {} ms budget after {:.2} ms (interrupted)",
                    cfg.timeout.map(|d| d.as_millis()).unwrap_or(0),
                    start.elapsed().as_secs_f64() * 1000.0
                ),
                "timeout".to_string(),
                true,
            ),
            InterruptReason::None => {
                let etype = classify_error(&msg);
                (msg, etype, false)
            }
        };
        SandboxRunResult {
            success: false,
            result_json: None,
            error: Some(message),
            error_type: Some(etype),
            interrupted,
            fuel_used: unsafe { self.fuel_total() },
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            logs: self.logs(),
        }
    }

    fn assemble(
        &self,
        cfg: &SandboxConfig,
        start: Instant,
        outcome: PumpOutcome,
        root: JSValueConst,
    ) -> SandboxRunResult {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let fuel = unsafe { self.fuel_total() };
        let logs = self.logs();

        match outcome {
            PumpOutcome::Fulfilled(v) => {
                let json = unsafe { serialize_val_to_json(self.ctx, v) };
                unsafe { qjs::sofuu_js_free_value(self.ctx, v) };
                SandboxRunResult {
                    success: true,
                    result_json: json,
                    error: None,
                    error_type: None,
                    interrupted: false,
                    fuel_used: fuel,
                    elapsed_ms,
                    logs,
                }
            }
            PumpOutcome::Rejected(v) => {
                let msg = unsafe { js_to_rust_string(self.ctx, v) };
                unsafe { qjs::sofuu_js_free_value(self.ctx, v) };
                SandboxRunResult {
                    success: false,
                    result_json: None,
                    error: Some(msg.clone()),
                    error_type: Some(classify_error(&msg)),
                    interrupted: false,
                    fuel_used: fuel,
                    elapsed_ms,
                    logs,
                }
            }
            PumpOutcome::Timeout => SandboxRunResult {
                success: false,
                result_json: None,
                error: Some(format!(
                    "E_TIMEOUT: sandbox execution exceeded its {} ms budget after {:.2} ms (interrupted)",
                    cfg.timeout.map(|d| d.as_millis()).unwrap_or(0),
                    elapsed_ms
                )),
                error_type: Some("timeout".to_string()),
                interrupted: true,
                fuel_used: fuel,
                elapsed_ms,
                logs,
            },
            PumpOutcome::Failed(msg) => {
                let reason = self.reason();
                let (message, etype, interrupted) = match reason {
                    InterruptReason::FuelExhausted => (
                        format!(
                            "Execution terminated: instruction fuel exhausted ({} limit reached)",
                            cfg.fuel_limit.unwrap_or(0)
                        ),
                        "fuel".to_string(),
                        true,
                    ),
                    InterruptReason::Timeout => (
                        format!(
                            "E_TIMEOUT: sandbox execution exceeded its {} ms budget after {:.2} ms (interrupted)",
                            cfg.timeout.map(|d| d.as_millis()).unwrap_or(0),
                            elapsed_ms
                        ),
                        "timeout".to_string(),
                        true,
                    ),
                    InterruptReason::None => {
                        let etype = classify_error(&msg);
                        (msg, etype, false)
                    }
                };
                SandboxRunResult {
                    success: false,
                    result_json: None,
                    error: Some(message),
                    error_type: Some(etype),
                    interrupted,
                    fuel_used: fuel,
                    elapsed_ms,
                    logs,
                }
            }
            PumpOutcome::Stuck => SandboxRunResult {
                success: true,
                result_json: unsafe { serialize_val_to_json(self.ctx, root) },
                error: None,
                error_type: None,
                interrupted: false,
                fuel_used: fuel,
                elapsed_ms,
                logs,
            },
            PumpOutcome::Suspended => SandboxRunResult::empty(),
        }
    }

    unsafe fn discard_pending_bridges(&mut self) {
        if self.pending_bridges.is_empty() {
            return;
        }
        let pending: Vec<PendingBridge> = self.pending_bridges.drain().map(|(_, v)| v).collect();
        for p in pending {
            unsafe {
                qjs::sofuu_js_free_value(self.ctx, p.resolve);
                qjs::sofuu_js_free_value(self.ctx, p.reject);
            }
        }
    }

    pub fn destroy(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        unregister_session(self.id);
        unsafe {
            self.discard_pending_bridges();
            if !self.host_ctx.is_null() {
                if let Some(cb) = self.console_cb.take() {
                    qjs::sofuu_js_free_value(self.host_ctx, cb);
                }
                for b in self.bridges.drain(..) {
                    qjs::sofuu_js_free_value(self.host_ctx, b.host_fn);
                }
            } else {
                self.bridges.clear();
            }
            if !self.ctx.is_null() {
                qjs::JS_FreeContext(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
            if !self.rt.is_null() {
                qjs::JS_SetInterruptHandler(self.rt, None, std::ptr::null_mut());
                qjs::JS_FreeRuntime(self.rt);
                self.rt = std::ptr::null_mut();
            }
            if !self.interrupt_state.is_null() {
                drop(Box::from_raw(self.interrupt_state));
                self.interrupt_state = std::ptr::null_mut();
            }
            if !self.context_state.is_null() {
                drop(Box::from_raw(self.context_state));
                self.context_state = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for SandboxSession {
    fn drop(&mut self) {
        self.destroy();
    }
}

fn classify_error(msg: &str) -> String {
    if msg.contains("out of memory") {
        "oom".to_string()
    } else if msg.contains("stack overflow") {
        "stack".to_string()
    } else if msg.contains("SyntaxError") {
        "syntax".to_string()
    } else {
        "runtime".to_string()
    }
}

/// Wrap the source in an async IIFE. The prefix shares line 1 with the user
/// code so only the first line's columns shift; every later line number in a
/// stack trace stays accurate.
fn async_wrap(src: &str) -> String {
    format!("(async () => {{{src}\n}})()")
}

/// W2: JSX (optional) then TypeScript (Always). Auto stripping is handled by
/// the parse-failure retry in `eval_with_ingest`.
fn prepare_source(code: &str, cfg: &SandboxConfig) -> String {
    let mut src = code.to_string();
    if cfg.jsx {
        if let Ok(jsx) = crate::frontend::transpile::transpile_jsx_only(&src) {
            src = jsx;
        }
    }
    if cfg.ts_mode == TsMode::Always {
        src = crate::ts::strip(&src);
    }
    src
}

unsafe fn serialize_val_to_json(ctx: *mut JSContext, val: JSValue) -> Option<String> {
    if unsafe { qjs::is_undefined(val) } {
        return None;
    }
    unsafe {
        let json_val = qjs::JS_JSONStringify(
            ctx,
            val,
            qjs::sofuu_js_undefined(),
            qjs::sofuu_js_undefined(),
        );
        if qjs::is_exception(json_val) || qjs::is_undefined(json_val) {
            if qjs::is_exception(json_val) {
                let _ = take_exception_message(ctx);
            }
            qjs::sofuu_js_free_value(ctx, json_val);
            // Fallback to string repr
            let cstr = qjs::sofuu_js_to_cstring(ctx, val);
            if !cstr.is_null() {
                let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, cstr);
                Some(format!("\"{}\"", s.replace('"', "\\\"")))
            } else {
                None
            }
        } else {
            let cstr = qjs::sofuu_js_to_cstring(ctx, json_val);
            let out = if !cstr.is_null() {
                let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, cstr);
                Some(s)
            } else {
                None
            };
            qjs::sofuu_js_free_value(ctx, json_val);
            out
        }
    }
}

// ── Pumping: drive both job queues until the root promise settles ──

fn deadline_passed(cfg: &SandboxConfig, start: Instant) -> bool {
    match cfg.timeout {
        Some(d) => Instant::now() >= start + d,
        None => false,
    }
}

unsafe fn pump(
    session: &mut SandboxSession,
    root: JSValue,
    cfg: &SandboxConfig,
    start: Instant,
) -> PumpOutcome {
    let mut iterations: u32 = 0;
    loop {
        if deadline_passed(cfg, start) {
            return PumpOutcome::Timeout;
        }
        let mut progressed = false;

        /* Sandbox jobs first: this is the continuation of the code that
         * called the bridge. */
        let mut pctx: *mut JSContext = std::ptr::null_mut();
        loop {
            let r = unsafe { qjs::JS_ExecutePendingJob(session.rt, &mut pctx) };
            if r < 0 {
                if session.reason() != InterruptReason::None {
                    return PumpOutcome::Failed(String::new());
                }
                let msg = unsafe { take_exception_message(session.ctx) };
                return PumpOutcome::Failed(msg);
            }
            if r == 0 {
                break;
            }
            progressed = true;
            if deadline_passed(cfg, start) {
                return PumpOutcome::Timeout;
            }
        }

        /* Host microtasks carry the promise-returning bridge resolutions.
         * Only pumped when a bridge exists: a sandbox without bridges must
         * not run unrelated host jobs. */
        if session.pump_host && !session.host_rt.is_null() {
            let mut hpctx: *mut JSContext = std::ptr::null_mut();
            loop {
                let r = unsafe { qjs::JS_ExecutePendingJob(session.host_rt, &mut hpctx) };
                if r < 0 {
                    /* The engine's rejection tracker reports host-side
                     * failures; clear the pending exception so it cannot
                     * leak into the caller's next evaluation. */
                    let _ = unsafe { take_exception_message(session.host_ctx) };
                    break;
                }
                if r == 0 {
                    break;
                }
                progressed = true;
                if deadline_passed(cfg, start) {
                    return PumpOutcome::Timeout;
                }
            }
        }

        match unsafe { qjs::JS_PromiseState(session.ctx, root) } {
            JSPromiseStateEnum::Fulfilled => {
                return PumpOutcome::Fulfilled(unsafe { qjs::JS_PromiseResult(session.ctx, root) });
            }
            JSPromiseStateEnum::Rejected => {
                return PumpOutcome::Rejected(unsafe { qjs::JS_PromiseResult(session.ctx, root) });
            }
            _ => {}
        }

        if !session.pending_bridges.is_empty() {
            if !progressed {
                return PumpOutcome::Suspended;
            }
            iterations += 1;
            if iterations > MAX_PUMP_ITERATIONS {
                return PumpOutcome::Failed(
                    "sandbox microtask pump exceeded its iteration bound".to_string(),
                );
            }
            continue;
        }

        if !progressed {
            return PumpOutcome::Stuck;
        }
    }
}

// ── Callable host bridges ───────────────────────────────────────

/// Sandbox-side function for a host bridge. Finds its session by the id in
/// func_data[0] and the bridge by func_data[1], marshals the arguments as
/// JSON, calls the host function, and either returns the JSON result or a
/// promise that settles when the host promise does.
unsafe extern "C" fn bridge_trampoline(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let (sid, index) = match unsafe { read_bridge_ids(ctx, data) } {
        Some(v) => v,
        None => return unsafe { throw_error_msg(ctx, "sandbox bridge: corrupt function data") },
    };
    let Some(ptr) = lookup_session(sid) else {
        return unsafe { throw_error_msg(ctx, "sandbox bridge: session is closed") };
    };
    let session = unsafe { &mut *ptr };

    let host_ctx = session.host_ctx;
    if host_ctx.is_null() {
        return unsafe { throw_error_msg(ctx, "sandbox bridge: no host context") };
    }
    let Some(bridge) = session.bridges.get(index) else {
        return unsafe { throw_error_msg(ctx, "sandbox bridge: unknown bridge") };
    };
    let host_fn = bridge.host_fn;
    let name = bridge.name.clone();

    let args_json = match unsafe { args_to_json(ctx, argc, argv) } {
        Ok(s) => s,
        Err(e) => return unsafe { throw_error_msg(ctx, &format!("bridge '{name}': {e}")) },
    };

    let host_args = match unsafe { json_to_values(host_ctx, &args_json) } {
        Ok(v) => v,
        Err(e) => return unsafe { throw_error_msg(ctx, &format!("bridge '{name}': {e}")) },
    };

    let ret = unsafe {
        qjs::JS_Call(
            host_ctx,
            host_fn,
            qjs::sofuu_js_undefined(),
            host_args.len() as c_int,
            host_args.as_ptr(),
        )
    };
    for v in &host_args {
        unsafe { qjs::sofuu_js_free_value(host_ctx, *v) };
    }

    if unsafe { qjs::is_exception(ret) } {
        let msg = unsafe { take_exception_message(host_ctx) };
        return unsafe { throw_error_msg(ctx, &format!("bridge '{name}' threw: {msg}")) };
    }

    let host_state = unsafe { qjs::JS_PromiseState(host_ctx, ret) };
    if host_state == JSPromiseStateEnum::Invalid {
        let json = match unsafe { json_stringify(host_ctx, ret) } {
            Ok(s) => s,
            Err(e) => {
                unsafe { qjs::sofuu_js_free_value(host_ctx, ret) };
                return unsafe { throw_error_msg(ctx, &format!("bridge '{name}': {e}")) };
            }
        };
        unsafe { qjs::sofuu_js_free_value(host_ctx, ret) };
        return match unsafe { json_parse(ctx, &json) } {
            Ok(v) => v,
            Err(e) => unsafe { throw_error_msg(ctx, &format!("bridge '{name}': {e}")) },
        };
    }

    // Promise-returning host function: hand the sandbox a real promise.
    let mut funcs: [JSValue; 2] = [qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined()];
    let spanned = unsafe { qjs::JS_NewPromiseCapability(ctx, funcs.as_mut_ptr()) };
    if unsafe { qjs::is_exception(spanned) } {
        unsafe { qjs::sofuu_js_free_value(host_ctx, ret) };
        return spanned;
    }

    let token = session.next_bridge_token;
    session.next_bridge_token += 1;
    session.pending_bridges.insert(
        token,
        PendingBridge {
            resolve: funcs[0],
            reject: funcs[1],
        },
    );

    unsafe { attach_host_then(sid, token, host_ctx, ret) };
    unsafe { qjs::sofuu_js_free_value(host_ctx, ret) };
    spanned
}

unsafe fn read_bridge_ids(ctx: *mut JSContext, data: *mut JSValue) -> Option<(u64, usize)> {
    if data.is_null() {
        return None;
    }
    unsafe {
        let mut id: i64 = -1;
        if qjs::JS_ToInt64(ctx, &mut id, *data) != 0 || id < 0 {
            return None;
        }
        let mut idx: i32 = -1;
        if qjs::JS_ToInt32(ctx, &mut idx, *data.add(1)) != 0 || idx < 0 {
            return None;
        }
        Some((id as u64, idx as usize))
    }
}

unsafe fn args_to_json(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Result<String, String> {
    unsafe {
        let arr = qjs::JS_NewArray(ctx);
        for i in 0..argc {
            let v = qjs::sofuu_js_dup_value(ctx, *argv.add(i as usize));
            qjs::JS_SetPropertyUint32(ctx, arr, i as u32, v);
        }
        let json = qjs::JS_JSONStringify(
            ctx,
            arr,
            qjs::sofuu_js_undefined(),
            qjs::sofuu_js_undefined(),
        );
        qjs::sofuu_js_free_value(ctx, arr);
        if qjs::is_exception(json) {
            let msg = take_exception_message(ctx);
            return Err(format!("arguments are not JSON-serializable: {msg}"));
        }
        let out = if qjs::is_undefined(json) {
            "[]".to_string()
        } else {
            js_to_rust_string(ctx, json)
        };
        qjs::sofuu_js_free_value(ctx, json);
        Ok(out)
    }
}

unsafe fn json_to_values(ctx: *mut JSContext, json: &str) -> Result<Vec<JSValue>, String> {
    unsafe {
        let arr = json_parse(ctx, json)?;
        if !qjs::is_object(arr) {
            qjs::sofuu_js_free_value(ctx, arr);
            return Ok(Vec::new());
        }
        let len_val = qjs::sofuu_js_get_property_str(ctx, arr, c"length".as_ptr());
        let mut len: i32 = 0;
        qjs::JS_ToInt32(ctx, &mut len, len_val);
        qjs::sofuu_js_free_value(ctx, len_val);
        let mut out = Vec::new();
        for i in 0..len.max(0) {
            out.push(qjs::JS_GetPropertyUint32(ctx, arr, i as u32));
        }
        qjs::sofuu_js_free_value(ctx, arr);
        Ok(out)
    }
}

unsafe fn attach_host_then(
    sid: u64,
    token: u64,
    host_ctx: *mut JSContext,
    promise: JSValueConst,
) {
    unsafe {
        let data = [
            qjs::sofuu_js_new_int64(host_ctx, sid as i64),
            qjs::sofuu_js_new_int64(host_ctx, token as i64),
        ];
        let on_ok = qjs::JS_NewCFunctionData(host_ctx, bridge_settle_cb, 1, 0, 2, data.as_ptr());
        let on_err = qjs::JS_NewCFunctionData(host_ctx, bridge_settle_cb, 1, 1, 2, data.as_ptr());
        qjs::sofuu_js_free_value(host_ctx, data[0]);
        qjs::sofuu_js_free_value(host_ctx, data[1]);

        let then_fn = qjs::sofuu_js_get_property_str(host_ctx, promise, c"then".as_ptr());
        if !qjs::is_exception(then_fn) && qjs::JS_IsFunction(host_ctx, then_fn) != 0 {
            let args = [on_ok, on_err];
            let r = qjs::JS_Call(host_ctx, then_fn, promise, 2, args.as_ptr());
            if qjs::is_exception(r) {
                let _ = take_exception_message(host_ctx);
            } else {
                qjs::sofuu_js_free_value(host_ctx, r);
            }
        }
        if qjs::is_exception(then_fn) {
            let _ = take_exception_message(host_ctx);
        } else {
            qjs::sofuu_js_free_value(host_ctx, then_fn);
        }
        qjs::sofuu_js_free_value(host_ctx, on_ok);
        qjs::sofuu_js_free_value(host_ctx, on_err);
    }
}

/// Host-side resolution of a bridge promise. `magic` 0 = fulfilled value,
/// 1 = rejection reason. Resolves the sandbox promise, then drives the
/// parked run (or leaves the active pump to observe the new microtasks).
unsafe extern "C" fn bridge_settle_cb(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    /* data[0] = session id, data[1] = bridge token (both int64). */
    let (sid, token) = match unsafe { read_bridge_ids(ctx, data) } {
        Some(v) => (v.0, v.1 as u64),
        None => return qjs::sofuu_js_undefined(),
    };

    let Some(ptr) = lookup_session(sid) else {
        return qjs::sofuu_js_undefined();
    };
    let session = unsafe { &mut *ptr };
    let Some(pending) = session.pending_bridges.remove(&token) else {
        return qjs::sofuu_js_undefined();
    };

    unsafe {
        if magic == 1 {
            let msg = if argc > 0 {
                let v = *argv;
                js_to_rust_string(ctx, v)
            } else {
                "host bridge promise rejected".to_string()
            };
            let err = make_error_object(session.ctx, &msg);
            let r = qjs::JS_Call(
                session.ctx,
                pending.reject,
                qjs::sofuu_js_undefined(),
                1,
                &err,
            );
            qjs::sofuu_js_free_value(session.ctx, err);
            qjs::sofuu_js_free_value(session.ctx, pending.resolve);
            if qjs::is_exception(r) {
                let _ = take_exception_message(session.ctx);
            } else {
                qjs::sofuu_js_free_value(session.ctx, r);
            }
        } else if argc > 0 {
            let host_val = *argv;
            match json_stringify(ctx, host_val).and_then(|j| json_parse(session.ctx, &j)) {
                Ok(v) => {
                    let r = qjs::JS_Call(
                        session.ctx,
                        pending.resolve,
                        qjs::sofuu_js_undefined(),
                        1,
                        &v,
                    );
                    qjs::sofuu_js_free_value(session.ctx, v);
                    qjs::sofuu_js_free_value(session.ctx, pending.reject);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(session.ctx);
                    } else {
                        qjs::sofuu_js_free_value(session.ctx, r);
                    }
                }
                Err(e) => {
                    let err = make_error_object(session.ctx, &format!("bridge result: {e}"));
                    let r = qjs::JS_Call(
                        session.ctx,
                        pending.reject,
                        qjs::sofuu_js_undefined(),
                        1,
                        &err,
                    );
                    qjs::sofuu_js_free_value(session.ctx, err);
                    qjs::sofuu_js_free_value(session.ctx, pending.resolve);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(session.ctx);
                    } else {
                        qjs::sofuu_js_free_value(session.ctx, r);
                    }
                }
            }
        } else {
            let v = qjs::sofuu_js_undefined();
            let r = qjs::JS_Call(session.ctx, pending.resolve, qjs::sofuu_js_undefined(), 1, &v);
            qjs::sofuu_js_free_value(session.ctx, pending.reject);
            if qjs::is_exception(r) {
                let _ = take_exception_message(session.ctx);
            } else {
                qjs::sofuu_js_free_value(session.ctx, r);
            }
        }
    }

    /* If this session's run is parked, resume it now. If it is not parked,
     * a synchronous pump is already running and will observe the sandbox
     * jobs this callback just queued. */
    unsafe { drive_parked_run(sid) };
    qjs::sofuu_js_undefined()
}

// ── Parked runs ─────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Run,
    Eval,
}

struct ActiveRun {
    sid: u64,
    session: *mut SandboxSession,
    owns_session: bool,
    root: JSValue,
    out_resolve: JSValue,
    out_reject: JSValue,
    keepalive: JSValue,
    config: SandboxConfig,
    start: Instant,
    mode: RunMode,
}

unsafe fn drive_parked_run(sid: u64) {
    let Some(ptr) = take_active_run(sid) else {
        return;
    };
    let run = unsafe { Box::from_raw(ptr) };
    let session = unsafe { &mut *run.session };

    unsafe { session.arm_interrupts(&run.config, run.start, true) };
    let outcome = unsafe { pump(session, run.root, &run.config, run.start) };
    unsafe {
        session.disarm_interrupts();
        session.settle_fuel();
    }

    if matches!(outcome, PumpOutcome::Suspended) {
        insert_active_run(sid, Box::into_raw(run));
    } else {
        unsafe { finish_run(run, outcome) };
    }
}

unsafe fn finish_run(run: Box<ActiveRun>, outcome: PumpOutcome) {
    let session = unsafe { &mut *run.session };
    let result = session.assemble(&run.config, run.start, outcome, run.root);
    let host = session.host_ctx;

    if !host.is_null() {
        unsafe {
            if run.mode == RunMode::Eval {
                if result.success {
                    let v = value_from_result_json(host, &result.result_json);
                    let r = qjs::JS_Call(host, run.out_resolve, qjs::sofuu_js_undefined(), 1, &v);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(host);
                    } else {
                        qjs::sofuu_js_free_value(host, r);
                    }
                    qjs::sofuu_js_free_value(host, v);
                } else {
                    let err = make_sandbox_error(host, &result);
                    let r = qjs::JS_Call(host, run.out_reject, qjs::sofuu_js_undefined(), 1, &err);
                    qjs::sofuu_js_free_value(host, err);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(host);
                    } else {
                        qjs::sofuu_js_free_value(host, r);
                    }
                }
            } else {
                let obj = build_js_result_object(host, &result);
                if run.config.throw_on_error && !result.success {
                    let err = make_sandbox_error(host, &result);
                    let r = qjs::JS_Call(host, run.out_reject, qjs::sofuu_js_undefined(), 1, &err);
                    qjs::sofuu_js_free_value(host, err);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(host);
                    } else {
                        qjs::sofuu_js_free_value(host, r);
                    }
                    qjs::sofuu_js_free_value(host, obj);
                } else {
                    let r = qjs::JS_Call(host, run.out_resolve, qjs::sofuu_js_undefined(), 1, &obj);
                    if qjs::is_exception(r) {
                        let _ = take_exception_message(host);
                    } else {
                        qjs::sofuu_js_free_value(host, r);
                    }
                    qjs::sofuu_js_free_value(host, obj);
                }
            }
            qjs::sofuu_js_free_value(host, run.out_resolve);
            qjs::sofuu_js_free_value(host, run.out_reject);
            if !qjs::is_undefined(run.keepalive) {
                qjs::sofuu_js_free_value(host, run.keepalive);
            }
        }
    }

    unsafe {
        session.busy = false;
        session.discard_pending_bridges();
        qjs::sofuu_js_free_value(session.ctx, run.root);
    }

    let close = session.pending_close;
    if run.owns_session || close {
        unregister_session(run.sid);
        session.destroy();
        if run.owns_session {
            unsafe { drop(Box::from_raw(run.session)) };
        }
    }
}

unsafe fn value_from_result_json(ctx: *mut JSContext, json: &Option<String>) -> JSValue {
    match json {
        Some(s) => unsafe { json_parse(ctx, s).unwrap_or_else(|_| qjs::sofuu_js_undefined()) },
        None => qjs::sofuu_js_undefined(),
    }
}

/// Suspend a run that returned `root: Some`. Creates the host promise the JS
/// caller awaits and parks the session until the last bridge settles.
unsafe fn suspend_run(
    ctx: *mut JSContext,
    session: *mut SandboxSession,
    owns_session: bool,
    root: JSValue,
    cfg: &SandboxConfig,
    start: Instant,
    mode: RunMode,
    keepalive: JSValue,
) -> JSValue {
    let mut funcs: [JSValue; 2] = [qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined()];
    let promise = unsafe { qjs::JS_NewPromiseCapability(ctx, funcs.as_mut_ptr()) };
    if unsafe { qjs::is_exception(promise) } {
        unsafe {
            qjs::sofuu_js_free_value((*session).ctx, root);
            if !qjs::is_undefined(keepalive) {
                qjs::sofuu_js_free_value(ctx, keepalive);
            }
        }
        if owns_session {
            unregister_session(unsafe { (*session).id });
            unsafe {
                (*session).destroy();
                drop(Box::from_raw(session));
            }
        }
        return promise;
    }

    let sid = unsafe { (*session).id };
    unsafe { (*session).busy = true };
    let run = Box::new(ActiveRun {
        sid,
        session,
        owns_session,
        root,
        out_resolve: funcs[0],
        out_reject: funcs[1],
        keepalive,
        config: cfg.clone(),
        start,
        mode,
    });
    insert_active_run(sid, Box::into_raw(run));
    promise
}

// ── Option parsing ──────────────────────────────────────────────

struct ParsedOptions {
    config: SandboxConfig,
    globals_json: Option<String>,
    /// Host-owned function values (top-level `globals` callables).
    bridges: Vec<(String, JSValue)>,
    /// Host-owned console callback.
    console_cb: Option<JSValue>,
}

impl ParsedOptions {
    unsafe fn free(&mut self, ctx: *mut JSContext) {
        for (_, v) in self.bridges.drain(..) {
            unsafe { qjs::sofuu_js_free_value(ctx, v) };
        }
        if let Some(cb) = self.console_cb.take() {
            unsafe { qjs::sofuu_js_free_value(ctx, cb) };
        }
    }
}

fn opt_error(msg: &str) -> String {
    format!("sandbox options: {msg}")
}

unsafe fn parse_options(
    ctx: *mut JSContext,
    opts_val: JSValueConst,
    base: &SandboxConfig,
) -> Result<ParsedOptions, String> {
    use qjs::sofuu_js_get_property_str as get;
    use qjs::sofuu_js_free_value as free;

    let mut config = base.clone();
    let mut globals_json = None;
    let mut bridges: Vec<(String, JSValue)> = Vec::new();
    let mut console_cb: Option<JSValue> = None;

    if !unsafe { qjs::is_object(opts_val) } {
        return Ok(ParsedOptions {
            config,
            globals_json,
            bridges,
            console_cb,
        });
    }

    unsafe {
        // ── preset (lowest precedence) ──
        let preset_val = get(ctx, opts_val, c"preset".as_ptr());
        if !qjs::is_undefined(preset_val) {
            let name = js_to_rust_string(ctx, preset_val);
            match SandboxConfig::preset(&name) {
                Some(c) => config = c,
                None => {
                    free(ctx, preset_val);
                    return Err(opt_error(&format!(
                        "unknown preset '{name}' (expected strict|standard|compute|worker)"
                    )));
                }
            }
        }
        free(ctx, preset_val);

        // ── fuel ──
        let fuel_val = get(ctx, opts_val, c"fuel".as_ptr());
        if !qjs::is_undefined(fuel_val) {
            if qjs::is_null(fuel_val) {
                config.fuel_limit = None;
            } else if qjs::is_number(fuel_val) {
                let mut n: f64 = 0.0;
                qjs::JS_ToFloat64(ctx, &mut n, fuel_val);
                if !n.is_finite() {
                    free(ctx, fuel_val);
                    return Err(opt_error("fuel must be a finite number or 'unlimited'"));
                }
                config.fuel_limit = if n <= 0.0 { None } else { Some(n as u64) };
            } else {
                let s = js_to_rust_string(ctx, fuel_val);
                if s == "unlimited" || s == "none" {
                    config.fuel_limit = None;
                } else {
                    free(ctx, fuel_val);
                    return Err(opt_error("fuel must be a number or 'unlimited'"));
                }
            }
        }
        free(ctx, fuel_val);

        // ── timeoutMs ──
        let timeout_val = get(ctx, opts_val, c"timeoutMs".as_ptr());
        if !qjs::is_undefined(timeout_val) {
            if qjs::is_null(timeout_val) {
                config.timeout = None;
            } else if qjs::is_number(timeout_val) {
                let mut ms: f64 = 0.0;
                qjs::JS_ToFloat64(ctx, &mut ms, timeout_val);
                /* NaN/Infinity are not budgets; a string is refused below
                 * rather than coerced through ToNumber. */
                if !ms.is_finite() {
                    free(ctx, timeout_val);
                    return Err(opt_error("timeoutMs must be a finite number, 0 or null"));
                }
                config.timeout = if ms <= 0.0 {
                    None
                } else {
                    Some(Duration::from_millis(ms as u64))
                };
            } else {
                free(ctx, timeout_val);
                return Err(opt_error("timeoutMs must be a number, 0 or null"));
            }
        }
        free(ctx, timeout_val);

        // ── maxMemoryMB ──
        let mem_val = get(ctx, opts_val, c"maxMemoryMB".as_ptr());
        if !qjs::is_undefined(mem_val) {
            let mut mb: f64 = 0.0;
            if !qjs::is_number(mem_val)
                || qjs::JS_ToFloat64(ctx, &mut mb, mem_val) != 0
                || !mb.is_finite()
            {
                free(ctx, mem_val);
                return Err(opt_error("maxMemoryMB must be a finite number of megabytes"));
            }
            if mb > 0.0 {
                config.max_memory_bytes = (mb * 1024.0 * 1024.0) as usize;
            }
        }
        free(ctx, mem_val);

        // ── maxStackKB ──
        let stack_val = get(ctx, opts_val, c"maxStackKB".as_ptr());
        if !qjs::is_undefined(stack_val) {
            let mut kb: f64 = 0.0;
            if !qjs::is_number(stack_val)
                || qjs::JS_ToFloat64(ctx, &mut kb, stack_val) != 0
                || !kb.is_finite()
            {
                free(ctx, stack_val);
                return Err(opt_error("maxStackKB must be a finite number of kilobytes"));
            }
            if kb > 0.0 {
                config.max_stack_bytes = (kb * 1024.0) as usize;
            }
        }
        free(ctx, stack_val);

        // ── captureConsole (0.2 API) ──
        let cap_val = get(ctx, opts_val, c"captureConsole".as_ptr());
        if !qjs::is_undefined(cap_val) {
            config.console = if qjs::JS_ToBool(ctx, cap_val) != 0 {
                ConsoleMode::Capture
            } else {
                ConsoleMode::Ignore
            };
        }
        free(ctx, cap_val);

        // ── console (v2 API) ──
        let console_val = get(ctx, opts_val, c"console".as_ptr());
        if !qjs::is_undefined(console_val) {
            if qjs::JS_IsFunction(ctx, console_val) != 0 {
                config.console = ConsoleMode::Callback;
                console_cb = Some(qjs::sofuu_js_dup_value(ctx, console_val));
            } else {
                let mode = js_to_rust_string(ctx, console_val);
                match mode.as_str() {
                    "capture" => config.console = ConsoleMode::Capture,
                    "inherit" => config.console = ConsoleMode::Inherit,
                    "ignore" => config.console = ConsoleMode::Ignore,
                    other => {
                        free(ctx, console_val);
                        return Err(opt_error(&format!(
                            "unknown console mode '{other}' (expected capture|inherit|ignore|function)"
                        )));
                    }
                }
            }
        }
        free(ctx, console_val);

        // ── maxLogs ──
        let logs_val = get(ctx, opts_val, c"maxLogs".as_ptr());
        if !qjs::is_undefined(logs_val) {
            let mut n: i64 = 0;
            if !qjs::is_number(logs_val) || qjs::JS_ToInt64(ctx, &mut n, logs_val) != 0 {
                free(ctx, logs_val);
                return Err(opt_error("maxLogs must be an integer (0 = unbounded)"));
            }
            config.max_logs = if n <= 0 { 0 } else { n as usize };
        }
        free(ctx, logs_val);

        // ── freezePrototypes ──
        let freeze_val = get(ctx, opts_val, c"freezePrototypes".as_ptr());
        if !qjs::is_undefined(freeze_val) {
            config.freeze_prototypes = qjs::JS_ToBool(ctx, freeze_val) != 0;
        }
        free(ctx, freeze_val);

        // ── ts ──
        let ts_val = get(ctx, opts_val, c"ts".as_ptr());
        if !qjs::is_undefined(ts_val) {
            if qjs::is_bool(ts_val) {
                config.ts_mode = if qjs::JS_ToBool(ctx, ts_val) != 0 {
                    TsMode::Always
                } else {
                    TsMode::Never
                };
            } else {
                let s = js_to_rust_string(ctx, ts_val);
                match s.as_str() {
                    "auto" => config.ts_mode = TsMode::Auto,
                    "true" | "always" => config.ts_mode = TsMode::Always,
                    "false" | "never" | "off" => config.ts_mode = TsMode::Never,
                    other => {
                        free(ctx, ts_val);
                        return Err(opt_error(&format!(
                            "unknown ts mode '{other}' (expected auto|true|false)"
                        )));
                    }
                }
            }
        }
        free(ctx, ts_val);

        // ── jsx ──
        let jsx_val = get(ctx, opts_val, c"jsx".as_ptr());
        if !qjs::is_undefined(jsx_val) {
            config.jsx = qjs::JS_ToBool(ctx, jsx_val) != 0;
        }
        free(ctx, jsx_val);

        // ── allowAsync ──
        let async_val = get(ctx, opts_val, c"allowAsync".as_ptr());
        if !qjs::is_undefined(async_val) {
            config.allow_async = qjs::JS_ToBool(ctx, async_val) != 0;
        }
        free(ctx, async_val);

        // ── throwOnError ──
        let throw_val = get(ctx, opts_val, c"throwOnError".as_ptr());
        if !qjs::is_undefined(throw_val) {
            config.throw_on_error = qjs::JS_ToBool(ctx, throw_val) != 0;
        }
        free(ctx, throw_val);

        // ── globals: data → JSON, top-level functions → bridges ──
        let globals_val = get(ctx, opts_val, c"globals".as_ptr());
        if qjs::is_object(globals_val) {
            let mut len: u32 = 0;
            let mut tab: *mut qjs::JSPropertyEnum = std::ptr::null_mut();
            if qjs::JS_GetOwnPropertyNames(
                ctx,
                &mut tab,
                &mut len,
                globals_val,
                1, // JS_GPN_STRING_MASK
            ) == 0
            {
                for i in 0..len {
                    let prop = *tab.add(i as usize);
                    let key = qjs::JS_AtomToCString(ctx, prop.atom);
                    if !key.is_null() {
                        let v = qjs::sofuu_js_get_property(ctx, globals_val, prop.atom);
                        if qjs::JS_IsFunction(ctx, v) != 0 {
                            let name = CStr::from_ptr(key).to_string_lossy().into_owned();
                            bridges.push((name, v)); // ownership moves to the bridge
                        } else {
                            qjs::sofuu_js_free_value(ctx, v);
                        }
                        qjs::JS_FreeCString(ctx, key);
                    }
                    qjs::JS_FreeAtom(ctx, prop.atom);
                }
                qjs::js_free(ctx, tab as *mut c_void);
            }

            let json_val = qjs::JS_JSONStringify(
                ctx,
                globals_val,
                qjs::sofuu_js_undefined(),
                qjs::sofuu_js_undefined(),
            );
            if !qjs::is_exception(json_val) && !qjs::is_undefined(json_val) {
                let cstr = qjs::sofuu_js_to_cstring(ctx, json_val);
                if !cstr.is_null() {
                    globals_json = Some(CStr::from_ptr(cstr).to_string_lossy().into_owned());
                    qjs::sofuu_js_free_cstring(ctx, cstr);
                }
            } else if qjs::is_exception(json_val) {
                let _ = take_exception_message(ctx);
            }
            qjs::sofuu_js_free_value(ctx, json_val);
        }
        free(ctx, globals_val);
    }

    Ok(ParsedOptions {
        config,
        globals_json,
        bridges,
        console_cb,
    })
}

// ── Result objects ──────────────────────────────────────────────

unsafe fn build_js_result_object(ctx: *mut JSContext, res: &SandboxRunResult) -> JSValue {
    unsafe {
        let obj = qjs::sofuu_js_new_object(ctx);

        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"success".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, if res.success { 1 } else { 0 }),
        );

        if let Some(ref json_str) = res.result_json {
            let c_json = CString::new(json_str.as_str())
                .unwrap_or_else(|_| CString::new("null").unwrap());
            let val = qjs::JS_ParseJSON(ctx, c_json.as_ptr(), json_str.len(), c"<result>".as_ptr());
            qjs::sofuu_js_set_property_str(ctx, obj, c"result".as_ptr(), val);
        } else {
            qjs::sofuu_js_set_property_str(ctx, obj, c"result".as_ptr(), qjs::sofuu_js_undefined());
        }

        if let Some(ref err) = res.error {
            let c_err =
                CString::new(err.as_str()).unwrap_or_else(|_| CString::new("error").unwrap());
            qjs::sofuu_js_set_property_str(
                ctx,
                obj,
                c"error".as_ptr(),
                qjs::sofuu_js_new_string(ctx, c_err.as_ptr()),
            );
        }

        if let Some(ref etype) = res.error_type {
            let c_etype =
                CString::new(etype.as_str()).unwrap_or_else(|_| CString::new("unknown").unwrap());
            qjs::sofuu_js_set_property_str(
                ctx,
                obj,
                c"errorType".as_ptr(),
                qjs::sofuu_js_new_string(ctx, c_etype.as_ptr()),
            );

            let code = sandbox_error_code(etype);
            let c_code = CString::new(code).unwrap_or_else(|_| CString::new("E_SANDBOX").unwrap());
            qjs::sofuu_js_set_property_str(
                ctx,
                obj,
                c"code".as_ptr(),
                qjs::sofuu_js_new_string(ctx, c_code.as_ptr()),
            );
        }

        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"interrupted".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, if res.interrupted { 1 } else { 0 }),
        );

        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"fuelUsed".as_ptr(),
            qjs::sofuu_js_new_float64(ctx, res.fuel_used as f64),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            c"elapsedMs".as_ptr(),
            qjs::sofuu_js_new_float64(ctx, res.elapsed_ms),
        );

        let logs_arr = qjs::JS_NewArray(ctx);
        for (i, log_line) in res.logs.iter().enumerate() {
            let c_line =
                CString::new(log_line.as_str()).unwrap_or_else(|_| CString::new("").unwrap());
            let line_val = qjs::sofuu_js_new_string(ctx, c_line.as_ptr());
            qjs::JS_SetPropertyUint32(ctx, logs_arr, i as u32, line_val);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"logs".as_ptr(), logs_arr);

        obj
    }
}

/* C-02: raise a catchable JS error for a failed run. A deadline violation
 * becomes `SandboxTimeoutError` with `.code === 'E_TIMEOUT'` (the message
 * also carries the budget and the elapsed time); every other failure keeps
 * the existing InternalError shape so only the timeout path changes. */
unsafe fn make_sandbox_error(ctx: *mut JSContext, result: &SandboxRunResult) -> JSValue {
    let err_type = result.error_type.as_deref().unwrap_or("runtime");
    let err_msg = result
        .error
        .clone()
        .unwrap_or_else(|| "Sandbox error".to_string());
    let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("Sandbox error").unwrap());

    unsafe {
        if err_type != "timeout" {
            let code = sandbox_error_code(err_type);
            let e = make_error_object(ctx, result.error.as_deref().unwrap_or("Sandbox error"));
            let c = CString::new(code).unwrap_or_else(|_| CString::new("E_SANDBOX").unwrap());
            qjs::sofuu_js_set_property_str(
                ctx,
                e,
                c"code".as_ptr(),
                qjs::sofuu_js_new_string(ctx, c.as_ptr()),
            );
            return e;
        }

        let err = qjs::JS_NewError(ctx);
        if qjs::is_exception(err) {
            return make_error_object(ctx, result.error.as_deref().unwrap_or("Sandbox error"));
        }
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"name".as_ptr(),
            qjs::sofuu_js_new_string(ctx, c"SandboxTimeoutError".as_ptr()),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"message".as_ptr(),
            qjs::sofuu_js_new_string(ctx, c_msg.as_ptr()),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"code".as_ptr(),
            qjs::sofuu_js_new_string(ctx, c"E_TIMEOUT".as_ptr()),
        );
        err
    }
}

unsafe fn throw_sandbox_failure(ctx: *mut JSContext, result: &SandboxRunResult) -> JSValue {
    unsafe {
        let err = make_sandbox_error(ctx, result);
        if qjs::is_exception(err) {
            return err;
        }
        qjs::JS_Throw(ctx, err)
    }
}

// ── Native JS Functions: sofuu.sandbox.run & sofuu.sandbox.eval ──

unsafe fn read_code_arg(ctx: *mut JSContext, argc: c_int, argv: *const JSValueConst) -> Result<String, JSValue> {
    if argc < 1 {
        return Err(unsafe { throw_error_msg(ctx, "sofuu.sandbox.run requires a code string") });
    }
    let cstr = unsafe { qjs::sofuu_js_to_cstring(ctx, *argv) };
    if cstr.is_null() {
        return Err(unsafe { throw_error_msg(ctx, "Invalid code string") });
    }
    let code = unsafe { CStr::from_ptr(cstr) }.to_string_lossy().into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, cstr) };
    Ok(code)
}

unsafe extern "C" fn js_sandbox_run(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let code = match unsafe { read_code_arg(ctx, argc, argv) } {
        Ok(c) => c,
        Err(v) => return v,
    };

    let mut parsed = match unsafe {
        parse_options(
            ctx,
            if argc > 1 { *argv.add(1) } else { qjs::sofuu_js_undefined() },
            &SandboxConfig::default(),
        )
    } {
        Ok(p) => p,
        Err(e) => return unsafe { throw_error_msg(ctx, &e) },
    };

    let mut session = match SandboxSession::new(parsed.config.clone()) {
        Ok(s) => Box::new(s),
        Err(e) => {
            unsafe { parsed.free(ctx) };
            return unsafe { throw_error_msg(ctx, &e) };
        }
    };
    session.register();

    let bridges = std::mem::take(&mut parsed.bridges);
    let console_cb = parsed.console_cb.take();
    if let Err(e) = session.install_host_services(ctx, bridges, console_cb) {
        unregister_session(session.id);
        session.destroy();
        return unsafe { throw_error_msg(ctx, &e) };
    }

    let start = Instant::now();
    let state = session.run_ex(
        &code,
        &parsed.config,
        parsed.globals_json.as_deref(),
        start,
    );

    match state.root {
        None => {
            let obj = unsafe { build_js_result_object(ctx, &state.result) };
            let id = session.id;
            unregister_session(id);
            session.destroy();
            obj
        }
        Some(root) => {
            let raw = Box::into_raw(session);
            unsafe {
                suspend_run(
                    ctx,
                    raw,
                    true,
                    root,
                    &parsed.config,
                    start,
                    RunMode::Run,
                    qjs::sofuu_js_undefined(),
                )
            }
        }
    }
}

unsafe extern "C" fn js_sandbox_eval(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let code = match unsafe { read_code_arg(ctx, argc, argv) } {
        Ok(c) => c,
        Err(v) => return v,
    };

    let mut parsed = match unsafe {
        parse_options(
            ctx,
            if argc > 1 { *argv.add(1) } else { qjs::sofuu_js_undefined() },
            &SandboxConfig::default(),
        )
    } {
        Ok(p) => p,
        Err(e) => return unsafe { throw_error_msg(ctx, &e) },
    };
    parsed.config.throw_on_error = true;

    let mut session = match SandboxSession::new(parsed.config.clone()) {
        Ok(s) => Box::new(s),
        Err(e) => {
            unsafe { parsed.free(ctx) };
            return unsafe { throw_error_msg(ctx, &e) };
        }
    };
    session.register();

    let bridges = std::mem::take(&mut parsed.bridges);
    let console_cb = parsed.console_cb.take();
    if let Err(e) = session.install_host_services(ctx, bridges, console_cb) {
        unregister_session(session.id);
        session.destroy();
        return unsafe { throw_error_msg(ctx, &e) };
    }

    let start = Instant::now();
    let state = session.run_ex(
        &code,
        &parsed.config,
        parsed.globals_json.as_deref(),
        start,
    );

    match state.root {
        None => {
            if !state.result.success {
                let v = unsafe { throw_sandbox_failure(ctx, &state.result) };
                let id = session.id;
                unregister_session(id);
                session.destroy();
                return v;
            }
            let v = unsafe { value_from_result_json(ctx, &state.result.result_json) };
            let id = session.id;
            unregister_session(id);
            session.destroy();
            v
        }
        Some(root) => {
            let raw = Box::into_raw(session);
            unsafe {
                suspend_run(
                    ctx,
                    raw,
                    true,
                    root,
                    &parsed.config,
                    start,
                    RunMode::Eval,
                    qjs::sofuu_js_undefined(),
                )
            }
        }
    }
}

// ── Stateful Sandbox Class ───────────────────────────────────────

unsafe extern "C" fn sandbox_instance_finalizer(_rt: *mut JSRuntime, val: JSValue) {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = unsafe { qjs::JS_GetOpaque(val, class_id) };
    if !ptr.is_null() {
        let mut session = unsafe { Box::from_raw(ptr as *mut SandboxSession) };
        session.destroy();
    }
}

unsafe extern "C" fn js_sandbox_instance_run(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe { sandbox_instance_call(ctx, this_val, argc, argv, RunMode::Run) }
}

unsafe extern "C" fn js_sandbox_instance_eval(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe { sandbox_instance_call(ctx, this_val, argc, argv, RunMode::Eval) }
}

unsafe fn sandbox_instance_call(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    mode: RunMode,
) -> JSValue {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = unsafe { qjs::JS_GetOpaque(this_val, class_id) };
    if ptr.is_null() {
        return unsafe { throw_error_msg(ctx, "Invalid Sandbox instance") };
    }
    let session = unsafe { &mut *(ptr as *mut SandboxSession) };

    if session.busy {
        return unsafe {
            throw_error_msg(ctx, "E_BUSY: this sandbox session is still awaiting an async bridge")
        };
    }

    let code = match unsafe { read_code_arg(ctx, argc, argv) } {
        Ok(c) => c,
        Err(v) => return v,
    };

    let mut parsed = match unsafe {
        parse_options(
            ctx,
            if argc > 1 { *argv.add(1) } else { qjs::sofuu_js_undefined() },
            &session.config.clone(),
        )
    } {
        Ok(p) => p,
        Err(e) => return unsafe { throw_error_msg(ctx, &e) },
    };
    if mode == RunMode::Eval {
        parsed.config.throw_on_error = true;
    }

    /* A stateful session installs its bridges once (at create). Options that
     * change the host seam per run are refused rather than silently ignored,
     * because the function objects were baked into the realm already. */
    if !parsed.bridges.is_empty() || parsed.console_cb.is_some() {
        unsafe { parsed.free(ctx) };
        return unsafe {
            throw_error_msg(
                ctx,
                "sandbox options: globals functions and console callbacks are fixed at create()",
            )
        };
    }

    let start = Instant::now();
    let state = session.run_ex(
        &code,
        &parsed.config,
        parsed.globals_json.as_deref(),
        start,
    );

    match state.root {
        None => {
            if mode == RunMode::Eval {
                if !state.result.success {
                    return unsafe { throw_sandbox_failure(ctx, &state.result) };
                }
                return unsafe { value_from_result_json(ctx, &state.result.result_json) };
            }
            if parsed.config.throw_on_error && !state.result.success {
                return unsafe { throw_sandbox_failure(ctx, &state.result) };
            }
            unsafe { build_js_result_object(ctx, &state.result) }
        }
        Some(root) => unsafe {
            let keepalive = qjs::sofuu_js_dup_value(ctx, this_val);
            suspend_run(
                ctx,
                ptr as *mut SandboxSession,
                false,
                root,
                &parsed.config,
                start,
                mode,
                keepalive,
            )
        },
    }
}

unsafe extern "C" fn js_sandbox_instance_close(
    _ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = unsafe { qjs::JS_GetOpaque(this_val, class_id) };
    if !ptr.is_null() {
        let session = unsafe { &mut *(ptr as *mut SandboxSession) };
        if session.busy {
            /* A parked run still owns the realm; tear down at completion.
             * The JS object (and therefore the Box) stays alive until then,
             * and the finalizer's destroy() is idempotent. */
            session.pending_close = true;
        } else {
            session.destroy();
        }
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_sandbox_create(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut parsed = match unsafe {
        parse_options(
            ctx,
            if argc > 0 { *argv } else { qjs::sofuu_js_undefined() },
            &SandboxConfig::default(),
        )
    } {
        Ok(p) => p,
        Err(e) => return unsafe { throw_error_msg(ctx, &e) },
    };

    let mut session = match SandboxSession::new(parsed.config.clone()) {
        Ok(s) => Box::new(s),
        Err(e) => {
            unsafe { parsed.free(ctx) };
            return unsafe { throw_error_msg(ctx, &e) };
        }
    };
    session.register();

    let bridges = std::mem::take(&mut parsed.bridges);
    let console_cb = parsed.console_cb.take();
    if let Err(e) = session.install_host_services(ctx, bridges, console_cb) {
        unregister_session(session.id);
        session.destroy();
        return unsafe { throw_error_msg(ctx, &e) };
    }

    if let Some(ref json) = parsed.globals_json {
        if let Err(e) = unsafe { inject_globals_json(session.ctx, json) } {
            unregister_session(session.id);
            session.destroy();
            return unsafe { throw_error_msg(ctx, &e) };
        }
    }

    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let obj = unsafe { qjs::JS_NewObjectClass(ctx, class_id as c_int) };
    if unsafe { qjs::is_exception(obj) } {
        unregister_session(session.id);
        session.destroy();
        return obj;
    }

    unsafe {
        let run_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_run, c"run".as_ptr(), 2);
        qjs::sofuu_js_set_property_str(ctx, obj, c"run".as_ptr(), run_fn);

        let eval_fn =
            qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_eval, c"eval".as_ptr(), 2);
        qjs::sofuu_js_set_property_str(ctx, obj, c"eval".as_ptr(), eval_fn);

        let close_fn =
            qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_close, c"close".as_ptr(), 0);
        qjs::sofuu_js_set_property_str(ctx, obj, c"close".as_ptr(), close_fn);

        let ptr = Box::into_raw(session);
        qjs::JS_SetOpaque(obj, ptr as *mut c_void);
    }

    obj
}

// ── Module Registration ──────────────────────────────────────────

pub unsafe fn mod_sandbox_register(ctx: *mut JSContext) {
    let rt = unsafe { qjs::JS_GetRuntime(ctx) };

    // Initialize SandboxInstance class ID if not done
    let mut class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    if class_id == 0 {
        unsafe { qjs::JS_NewClassID(&mut class_id) };
        SANDBOX_CLASS_ID.store(class_id, Ordering::Relaxed);

        let class_def = JSClassDef {
            class_name: c"SandboxSession".as_ptr(),
            finalizer: Some(sandbox_instance_finalizer),
            gc_mark: std::ptr::null_mut(),
            call: std::ptr::null_mut(),
            exotic: std::ptr::null_mut(),
        };
        unsafe { qjs::JS_NewClass(rt, class_id, &class_def) };
    }

    unsafe {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
        if qjs::is_undefined(sofuu_obj) {
            sofuu_obj = qjs::sofuu_js_new_object(ctx);
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sofuu".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sofuu_obj),
            );
        }

        let sandbox_obj = qjs::sofuu_js_new_object(ctx);

        let run_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_run, c"run".as_ptr(), 2);
        qjs::sofuu_js_set_property_str(ctx, sandbox_obj, c"run".as_ptr(), run_fn);

        let eval_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_eval, c"eval".as_ptr(), 2);
        qjs::sofuu_js_set_property_str(ctx, sandbox_obj, c"eval".as_ptr(), eval_fn);

        let create_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_create, c"create".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, sandbox_obj, c"create".as_ptr(), create_fn);

        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"sandbox".as_ptr(), sandbox_obj);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
        qjs::sofuu_js_free_value(ctx, global);
    }
}

// ── Unit Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_defaults_are_bounded() {
        let cfg = SandboxConfig::default();
        assert_eq!(cfg.fuel_limit, Some(SANDBOX_DEFAULT_FUEL));
        assert_eq!(
            cfg.timeout,
            Some(Duration::from_millis(SANDBOX_DEFAULT_TIMEOUT_MS))
        );
        let mut session = SandboxSession::new(cfg.clone()).expect("session");
        let start = Instant::now();
        let res = session.run("while(true) {}", &cfg, None);
        assert!(!res.success, "an unbounded program must not succeed");
        assert!(res.interrupted, "the meter must interrupt it");
        assert!(matches!(
            res.error_type.as_deref(),
            Some("fuel") | Some("timeout")
        ));
        assert!(
            start.elapsed() < Duration::from_millis(SANDBOX_DEFAULT_TIMEOUT_MS + 1500),
            "bounded run took {:?}",
            start.elapsed()
        );
    }

    /* C-02: the Rust poll charge and the C poll interval must agree, or the
     * fuel meter would under/over-charge native regexp work. */
    #[test]
    fn test_regexp_poll_steps_matches_quickjs() {
        let header = include_str!("../../../../deps/quickjs/libregexp.h");
        assert!(
            header.contains("#define LRE_INTERRUPT_POLL_STEPS 4096"),
            "deps/quickjs/libregexp.h poll interval changed: update REGEXP_POLL_STEPS"
        );
        assert_eq!(REGEXP_POLL_STEPS, 4096);
    }

    /* W1 / I2: budgets the developer states are never capped. Only floors
     * remain (a 0-byte stack is not a budget, it is a crash). */
    #[test]
    fn test_sandbox_config_is_not_capped() {
        let mut cfg = SandboxConfig::default();
        cfg.fuel_limit = Some(u64::MAX);
        cfg.timeout = Some(Duration::from_secs(3600));
        cfg.max_memory_bytes = 8 * 1024 * 1024 * 1024;
        cfg.max_stack_bytes = 64 * 1024 * 1024;
        cfg.clamp();
        assert_eq!(cfg.fuel_limit, Some(u64::MAX));
        assert_eq!(cfg.timeout, Some(Duration::from_secs(3600)));
        assert_eq!(cfg.max_memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(cfg.max_stack_bytes, 64 * 1024 * 1024);

        let mut tiny = SandboxConfig::default();
        tiny.max_memory_bytes = 1;
        tiny.max_stack_bytes = 1;
        tiny.clamp();
        assert_eq!(tiny.max_memory_bytes, SANDBOX_MIN_MEMORY_BYTES);
        assert_eq!(tiny.max_stack_bytes, SANDBOX_MIN_STACK_BYTES);
    }

    #[test]
    fn test_sandbox_presets() {
        let strict = SandboxConfig::preset("strict").expect("strict");
        assert_eq!(strict.fuel_limit, Some(5_000_000));
        assert!(strict.freeze_prototypes);
        let compute = SandboxConfig::preset("compute").expect("compute");
        assert_eq!(compute.fuel_limit, None);
        assert_eq!(compute.timeout, Some(Duration::from_millis(300_000)));
        let worker = SandboxConfig::preset("worker").expect("worker");
        assert_eq!(worker.fuel_limit, None);
        assert_eq!(worker.timeout, None);
        assert!(SandboxConfig::preset("nonsense").is_none());
    }

    /* W1: `fuel: None` + no deadline really is unmetered: a loop that would
     * exhaust the default 5M meter completes instead of failing. */
    #[test]
    fn test_sandbox_unlimited_fuel_is_unmetered() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.fuel_limit = None;
        config.timeout = None;
        let res = session.run(
            "let s = 0; for (let i = 0; i < 20000000; i++) { s += 1; } s;",
            &config,
            None,
        );
        assert!(res.success, "unmetered run failed: {:?}", res.error);
        assert_eq!(res.result_json.as_deref(), Some("20000000"));
    }

    #[test]
    fn test_sandbox_basic_eval() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let res = session.run("1 + 2 * 3;", &SandboxConfig::default(), None);
        assert!(res.success);
        assert_eq!(res.result_json.as_deref(), Some("7"));
        assert!(res.fuel_used > 0);
    }

    #[test]
    fn test_sandbox_fuel_exhaustion() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.fuel_limit = Some(50_000);
        let res = session.run("while(true) {}", &config, None);
        assert!(!res.success);
        assert!(res.interrupted);
        assert_eq!(res.error_type.as_deref(), Some("fuel"));
        assert!(res.fuel_used >= 50_000);
    }

    #[test]
    fn test_sandbox_timeout() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.timeout = Some(Duration::from_millis(10));
        // Unmetered fuel: only the deadline can stop this loop (see the P5
        // matrix race the old version of this test was pinned against).
        config.fuel_limit = None;
        let start = Instant::now();
        let res = session.run("while(true) {}", &config, None);
        assert!(!res.success);
        assert!(res.interrupted);
        assert_eq!(res.error_type.as_deref(), Some("timeout"));
        // Promptness, not a benchmark: a busy runner can delay the poll.
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    /* C-02: catastrophic regex backtracking runs in one QuickJS native
     * frame. Before the libregexp.c poll it ignored `timeoutMs` entirely
     * (measured: 1.9 s for N=24, 47 s for N=28) and returned success: true.
     * The deadline must now abort it promptly, report E_TIMEOUT, and leave
     * the runtime usable for the next call. */
    #[test]
    fn test_sandbox_regex_catastrophic_backtracking_times_out() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.timeout = Some(Duration::from_millis(100));
        config.fuel_limit = None;

        let start = Instant::now();
        let res = session.run(
            r#"/(a+)+$/.test("a".repeat(26) + "b")"#,
            &config,
            None,
        );
        let took = start.elapsed();

        assert!(!res.success, "catastrophic regex must not succeed");
        assert!(res.interrupted, "catastrophic regex must be interrupted");
        assert_eq!(
            res.error_type.as_deref(),
            Some("timeout"),
            "expected a timeout, got {:?} ({:?})",
            res.error_type,
            res.error
        );
        let msg = res.error.clone().unwrap_or_default();
        assert!(msg.contains("E_TIMEOUT"), "message must carry E_TIMEOUT: {msg}");
        assert!(msg.contains("100 ms"), "message must name the budget: {msg}");
        /* ≲150 ms is the target; keep slack for a loaded runner while still
         * failing loudly if the poll regresses to whole-second granularity. */
        assert!(
            took < Duration::from_millis(400),
            "regex timeout took {took:?} (must be ~100 ms)"
        );

        /* Determinism: an interrupted evaluation must not wedge the session. */
        let ok = session.run(r#"/abc/.test("xabcx")"#, &SandboxConfig::default(), None);
        assert!(ok.success, "ordinary regex after a timeout: {:?}", ok.error);
        assert_eq!(ok.result_json.as_deref(), Some("true"));

        let benign = session.run(
            r#"/(a+)+$/.test("a".repeat(12) + "b")"#,
            &SandboxConfig::default(),
            None,
        );
        assert!(benign.success, "benign regex: {:?}", benign.error);
        assert_eq!(benign.result_json.as_deref(), Some("false"));
    }

    /* C-02 safety net: with no wall-clock budget at all (fuel-only sandbox)
     * regexp steps are charged to the same meter, so a catastrophic pattern
     * still terminates through the fuel path instead of hanging. */
    #[test]
    fn test_sandbox_regex_fuel_only_still_bounded() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.timeout = None;
        config.fuel_limit = Some(100_000);

        let start = Instant::now();
        let res = session.run(
            r#"/(a+)+$/.test("a".repeat(26) + "b")"#,
            &config,
            None,
        );
        assert!(!res.success, "fuel-only sandbox must not succeed");
        assert!(res.interrupted);
        assert_eq!(res.error_type.as_deref(), Some("fuel"), "{:?}", res.error);
        assert!(
            start.elapsed() < Duration::from_millis(1000),
            "fuel-only regex took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn test_sandbox_memory_limit() {
        let mut config = SandboxConfig::default();
        config.max_memory_bytes = 2 * 1024 * 1024; // 2 MB
        let mut session = SandboxSession::new(config.clone()).expect("session");
        let code = "const chunks = []; while(true) { chunks.push(new Uint8Array(256 * 1024)); }";
        let res = session.run(code, &config, None);
        assert!(!res.success);
        assert_eq!(res.error_type.as_deref(), Some("oom"));
    }

    #[test]
    fn test_sandbox_stack_limit() {
        let mut config = SandboxConfig::default();
        config.max_stack_bytes = 128 * 1024; // 128 KB
        let mut session = SandboxSession::new(config.clone()).expect("session");
        let code = "function rec(n) { return rec(n + 1); } rec(0);";
        let res = session.run(code, &config, None);
        assert!(!res.success);
        assert_eq!(res.error_type.as_deref(), Some("stack"));
    }

    #[test]
    fn test_sandbox_stateful_session() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let config = SandboxConfig::default();

        let r1 = session.run("let count = 100;", &config, None);
        assert!(r1.success);

        let r2 = session.run("count += 50;", &config, None);
        assert!(r2.success);

        let r3 = session.run("count * 2;", &config, None);
        assert!(r3.success);
        assert_eq!(r3.result_json.as_deref(), Some("300"));
    }

    #[test]
    fn test_sandbox_console_capture() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut config = SandboxConfig::default();
        config.console = ConsoleMode::Capture;

        let code = "console.log('alpha', 1); console.warn('beta'); 'ok';";
        let res = session.run(code, &config, None);
        assert!(res.success);
        assert_eq!(res.logs.len(), 2);
        assert!(res.logs[0].contains("alpha 1"));
        assert!(res.logs[1].contains("beta"));
    }

    #[test]
    fn test_sandbox_console_ignore_and_cap() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");

        let mut ignore = SandboxConfig::default();
        ignore.console = ConsoleMode::Ignore;
        let r = session.run("console.log('hidden'); 1;", &ignore, None);
        assert!(r.success);
        assert!(r.logs.is_empty(), "ignore mode captured {:?}", r.logs);

        let mut capped = SandboxConfig::default();
        capped.max_logs = 2;
        let r2 = session.run(
            "for (let i = 0; i < 10; i++) console.log(i); 'done';",
            &capped,
            None,
        );
        assert!(r2.success);
        assert_eq!(r2.logs.len(), 2);
    }

    /* W2: TypeScript types are stripped automatically — interfaces,
     * annotations and generics all evaluate without a build step. */
    #[test]
    fn test_sandbox_typescript_auto_strip() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let cfg = SandboxConfig::default();
        let res = session.run(
            "interface Point { x: number; y: number }\n\
             const p: Point = { x: 3, y: 4 };\n\
             function norm<T extends number>(v: T): number { return v * v; }\n\
             norm(p.x) + norm(p.y);",
            &cfg,
            None,
        );
        assert!(res.success, "TS code failed: {:?}", res.error);
        assert_eq!(res.result_json.as_deref(), Some("25"));
    }

    #[test]
    fn test_sandbox_typescript_can_be_disabled() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let mut cfg = SandboxConfig::default();
        cfg.ts_mode = TsMode::Never;
        let res = session.run("const x: number = 1; x;", &cfg, None);
        assert!(!res.success);
        assert_eq!(res.error_type.as_deref(), Some("syntax"));
    }

    /* W5: a frozen realm refuses cross-turn prototype pollution. */
    #[test]
    fn test_sandbox_freeze_prototypes() {
        let mut config = SandboxConfig::default();
        config.freeze_prototypes = true;
        let mut session = SandboxSession::new(config.clone()).expect("session");

        let r1 = session.run(
            "try { Object.prototype.polluted = true; } catch (e) {} \
             try { Array.prototype.pushed = 1; } catch (e) {} \
             ({ p: ({}).polluted === undefined, a: [].pushed === undefined });",
            &config,
            None,
        );
        assert!(r1.success, "{:?}", r1.error);
        assert_eq!(
            r1.result_json.as_deref(),
            Some("{\"p\":true,\"a\":true}"),
            "prototypes must be immutable"
        );

        // Ordinary code still works in a frozen realm.
        let r2 = session.run("const a = [3, 1, 2]; a.sort(); a.join(',');", &config, None);
        assert!(r2.success, "{:?}", r2.error);
        assert_eq!(r2.result_json.as_deref(), Some("\"1,2,3\""));
    }

    #[test]
    fn test_sandbox_jsx_transpiles_to_vdom() {
        let mut config = SandboxConfig::default();
        config.jsx = true;
        let mut session = SandboxSession::new(config.clone()).expect("session");
        let res = session.run(
            "const el = <div className=\"box\">Hello {1 + 1}</div>; el.type + '|' + el.props.className + '|' + el.children[1];",
            &config,
            None,
        );
        assert!(res.success, "JSX run failed: {:?}", res.error);
        assert_eq!(res.result_json.as_deref(), Some("\"div|box|2\""));
    }

    #[test]
    fn test_sandbox_globals_injection() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let config = SandboxConfig::default();
        let globals = r#"{"factor": 5, "label": "test"}"#;

        let res = session.run("label + ':' + (factor * 10);", &config, Some(globals));
        assert!(res.success);
        assert_eq!(res.result_json.as_deref(), Some("\"test:50\""));
    }

    #[test]
    fn test_sandbox_async_promise() {
        let mut session = SandboxSession::new(SandboxConfig::default()).expect("session");
        let config = SandboxConfig::default();

        let res1 = session.run("Promise.resolve(42);", &config, None);
        assert!(res1.success);
        assert_eq!(res1.result_json.as_deref(), Some("42"));

        let res2 = session.run("Promise.reject(new Error('fail'));", &config, None);
        assert!(!res2.success);
        assert!(res2.error.unwrap().contains("fail"));

        // Non-promise with allow_async = true (verifies JSPromiseStateEnum::Invalid)
        let res3 = session.run("100 + 200;", &config, None);
        assert!(res3.success);
        assert_eq!(res3.result_json.as_deref(), Some("300"));
    }
}
