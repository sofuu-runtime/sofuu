// rt/sandbox.rs — Deterministic Agent Sandboxing & Fine-Grained Fuel Meters (PLAN-RUNTIME Phase 5).
//
// Modern sandboxing substrate providing:
// - Deterministic instruction fuel metering via QuickJS interrupt counters.
// - Hard wall-clock timeout bounds via high-resolution monotonic timer interrupts.
// - Physical memory ceilings (JS_SetMemoryLimit) & call stack caps (JS_SetMaxStackSize).
// - Hermetic execution: complete isolation from host std/os/fs/network/process.
// - Console output interception (console.log/info/warn/error) into structured log arrays.
// - JSON-safe bidirectional global variable injection and expression result extraction.
// - Async promise resolution support.
// - Both one-shot `sofuu.sandbox.run(code, opts)` and stateful `sofuu.sandbox.create(opts)`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use sofuu_ffi::qjs::{self, JSClassDef, JSContext, JSRuntime, JSValue, JSValueConst, JSPromiseStateEnum};

static SANDBOX_CLASS_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub fuel_limit: Option<u64>,
    pub timeout: Option<Duration>,
    pub max_memory_bytes: usize,
    pub max_stack_bytes: usize,
    pub capture_console: bool,
    pub allow_async: bool,
    pub throw_on_error: bool,
}

/* S2 (2026-10-23): the public sandbox API used to default to NO fuel and NO
 * deadline, and trusted every caller-supplied value. `run`/`eval` are
 * synchronous, so `sofuu.sandbox.run('while(true){}')` stopped the whole
 * libuv loop with no way back (verified live: an external alarm was the only
 * way to kill it). Defaults are now bounded and every value is clamped at
 * the native boundary — the tool (tools.js code_eval) keeps its own tighter
 * limits, but a host or inline tool cannot accidentally lift them. */
pub const SANDBOX_DEFAULT_FUEL: u64 = 5_000_000;
pub const SANDBOX_DEFAULT_TIMEOUT_MS: u64 = 2_000;
pub const SANDBOX_DEFAULT_MEMORY_BYTES: usize = 64 * 1024 * 1024;
pub const SANDBOX_DEFAULT_STACK_BYTES: usize = 512 * 1024;
pub const SANDBOX_MAX_FUEL: u64 = 50_000_000;
pub const SANDBOX_MAX_TIMEOUT_MS: u64 = 10_000;
pub const SANDBOX_MIN_MEMORY_BYTES: usize = 1024 * 1024;
pub const SANDBOX_MAX_MEMORY_BYTES: usize = 512 * 1024 * 1024;
pub const SANDBOX_MIN_STACK_BYTES: usize = 16 * 1024;
pub const SANDBOX_MAX_STACK_BYTES: usize = 4 * 1024 * 1024;

impl SandboxConfig {
    /// Enforce the native ceilings. Called by the session constructor and on
    /// every run so no caller (JS options included) can widen the bounds.
    pub fn clamp(&mut self) {
        if let Some(f) = self.fuel_limit {
            self.fuel_limit = Some(f.min(SANDBOX_MAX_FUEL));
        }
        if let Some(t) = self.timeout {
            self.timeout = Some(t.min(Duration::from_millis(SANDBOX_MAX_TIMEOUT_MS)));
        }
        self.max_memory_bytes = self
            .max_memory_bytes
            .clamp(SANDBOX_MIN_MEMORY_BYTES, SANDBOX_MAX_MEMORY_BYTES);
        self.max_stack_bytes = self
            .max_stack_bytes
            .clamp(SANDBOX_MIN_STACK_BYTES, SANDBOX_MAX_STACK_BYTES);
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            fuel_limit: Some(SANDBOX_DEFAULT_FUEL),
            timeout: Some(Duration::from_millis(SANDBOX_DEFAULT_TIMEOUT_MS)),
            max_memory_bytes: SANDBOX_DEFAULT_MEMORY_BYTES,
            max_stack_bytes: SANDBOX_DEFAULT_STACK_BYTES,
            capture_console: true,
            allow_async: true,
            throw_on_error: false,
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

unsafe extern "C" fn sandbox_interrupt_cb(_rt: *mut JSRuntime, opaque: *mut c_void) -> c_int {
    if opaque.is_null() {
        return 0;
    }
    let st = unsafe { &mut *(opaque as *mut InterruptState) };

    // Record fuel for the batch that just completed
    let batch = st.fuel_quanta;
    st.fuel_consumed = st.fuel_consumed.saturating_add(batch);

    // Fuel check
    if let Some(_) = st.fuel_limit {
        st.fuel_remaining -= batch as i64;
        if st.fuel_remaining <= 0 {
            st.reason = InterruptReason::FuelExhausted;
            return 1; // abort!
        }
        if !st.ctx.is_null() {
            let next_batch = (st.fuel_remaining as u64).min(10_000).max(1);
            st.fuel_quanta = next_batch;
            qjs::JS_SetInterruptCounter(st.ctx, next_batch as c_int);
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

pub struct SandboxContextState {
    pub logs: Vec<String>,
    pub capture_console: bool,
}

unsafe extern "C" fn sandbox_console_fn(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let opaque = qjs::JS_GetContextOpaque(ctx);
    if !opaque.is_null() {
        let st = unsafe { &mut *(opaque as *mut SandboxContextState) };
        if st.capture_console && st.logs.len() < 512 {
            let mut line = String::new();
            for i in 0..argc {
                if i > 0 {
                    line.push(' ');
                }
                let v = unsafe { *argv.add(i as usize) };
                let cstr = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
                if !cstr.is_null() {
                    if let Ok(s) = unsafe { CStr::from_ptr(cstr) }.to_str() {
                        line.push_str(s);
                    }
                    unsafe { qjs::sofuu_js_free_cstring(ctx, cstr) };
                }
            }
            st.logs.push(line);
        }
    }
    qjs::sofuu_js_undefined()
}

unsafe fn install_console(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let console = qjs::sofuu_js_new_object(ctx);

    let methods = [
        c"log",
        c"info",
        c"warn",
        c"error",
    ];
    for name in methods {
        let fn_val = qjs::sofuu_js_new_cfunction(
            ctx,
            sandbox_console_fn,
            name.as_ptr(),
            0,
        );
        qjs::sofuu_js_set_property_str(ctx, console, name.as_ptr(), fn_val);
    }
    qjs::sofuu_js_set_property_str(ctx, global, c"console".as_ptr(), console);
    qjs::sofuu_js_free_value(ctx, global);
}

unsafe fn inject_globals_json(ctx: *mut JSContext, json_str: &str) -> Result<(), String> {
    let c_json = match CString::new(json_str) {
        Ok(c) => c,
        Err(_) => return Err("Globals JSON contains NUL byte".to_string()),
    };
    let parsed = qjs::JS_ParseJSON(
        ctx,
        c_json.as_ptr(),
        json_str.len(),
        c"<globals>".as_ptr(),
    );
    if qjs::is_exception(parsed) {
        qjs::sofuu_js_free_value(ctx, parsed);
        return Err("Failed to parse globals into sandbox context".to_string());
    }

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut len: u32 = 0;
    let mut tab: *mut qjs::JSPropertyEnum = std::ptr::null_mut();
    if qjs::JS_GetOwnPropertyNames(
        ctx,
        &mut tab,
        &mut len,
        parsed,
        (qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_SYMBOL_MASK) as c_int,
    ) == 0 {
        for i in 0..len {
            let prop = *tab.add(i as usize);
            let key_cstr = qjs::JS_AtomToCString(ctx, prop.atom);
            if !key_cstr.is_null() {
                let val = qjs::sofuu_js_get_property(ctx, parsed, prop.atom);
                qjs::sofuu_js_set_property_str(ctx, global, key_cstr, val);
                qjs::JS_FreeCString(ctx, key_cstr);
            }
            qjs::JS_FreeAtom(ctx, prop.atom);
        }
        qjs::js_free(ctx, tab as *mut c_void);
    }
    qjs::sofuu_js_free_value(ctx, parsed);
    qjs::sofuu_js_free_value(ctx, global);
    Ok(())
}

#[derive(Debug)]
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

pub struct SandboxSession {
    rt: *mut JSRuntime,
    ctx: *mut JSContext,
    interrupt_state: *mut InterruptState,
    context_state: *mut SandboxContextState,
    config: SandboxConfig,
    closed: bool,
}

impl SandboxSession {
    pub fn new(mut config: SandboxConfig) -> Result<Self, String> {
        config.clamp();
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
                fuel_remaining: config.fuel_limit.unwrap_or(u64::MAX) as i64,
                fuel_consumed: 0,
                fuel_quanta: 10_000,
                reason: InterruptReason::None,
            }));

            qjs::JS_SetInterruptHandler(rt, Some(sandbox_interrupt_cb), interrupt_state as *mut c_void);

            let ctx = qjs::JS_NewContext(rt);
            if ctx.is_null() {
                qjs::JS_FreeRuntime(rt);
                drop(Box::from_raw(interrupt_state));
                return Err("Failed to allocate QuickJS context for sandbox".to_string());
            }

            (*interrupt_state).ctx = ctx;

            let context_state = Box::into_raw(Box::new(SandboxContextState {
                logs: Vec::new(),
                capture_console: config.capture_console,
            }));

            qjs::JS_SetContextOpaque(ctx, context_state as *mut c_void);

            install_console(ctx);

            Ok(Self {
                rt,
                ctx,
                interrupt_state,
                context_state,
                config,
                closed: false,
            })
        }
    }

    pub fn run(
        &mut self,
        code: &str,
        run_config: &SandboxConfig,
        globals_json: Option<&str>,
    ) -> SandboxRunResult {
        if self.closed {
            return SandboxRunResult {
                success: false,
                result_json: None,
                error: Some("Sandbox session is closed".to_string()),
                error_type: Some("closed".to_string()),
                interrupted: false,
                fuel_used: 0,
                elapsed_ms: 0.0,
                logs: Vec::new(),
            };
        }

        let start = Instant::now();
        /* S2: per-run ceilings are enforced here too — a caller can pass a
         * different config to every run on a stateful session. */
        let run_config = {
            let mut c = run_config.clone();
            c.clamp();
            c
        };

        unsafe {
            // Re-arm interrupt state
            let st = &mut *self.interrupt_state;
            st.deadline = run_config.timeout.map(|d| Instant::now() + d);
            st.fuel_limit = run_config.fuel_limit;
            st.fuel_remaining = run_config.fuel_limit.unwrap_or(u64::MAX) as i64;
            st.fuel_consumed = 0;
            st.reason = InterruptReason::None;

            let initial_counter = run_config
                .fuel_limit
                .unwrap_or(10_000)
                .min(10_000)
                .max(1);
            st.fuel_quanta = initial_counter;
            qjs::JS_SetInterruptCounter(self.ctx, initial_counter as c_int);

            // Reset logs
            let cst = &mut *self.context_state;
            cst.logs.clear();
            cst.capture_console = run_config.capture_console;

            // Inject globals if provided
            if let Some(json) = globals_json {
                if let Err(e) = inject_globals_json(self.ctx, json) {
                    return SandboxRunResult {
                        success: false,
                        result_json: None,
                        error: Some(e),
                        error_type: Some("globals".to_string()),
                        interrupted: false,
                        fuel_used: 0,
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        logs: Vec::new(),
                    };
                }
            }

            let c_code = match CString::new(code) {
                Ok(c) => c,
                Err(_) => {
                    return SandboxRunResult {
                        success: false,
                        result_json: None,
                        error: Some("Code string contains NUL byte".to_string()),
                        error_type: Some("syntax".to_string()),
                        interrupted: false,
                        fuel_used: 0,
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        logs: Vec::new(),
                    };
                }
            };

            let mut eval_val = qjs::JS_Eval(
                self.ctx,
                c_code.as_ptr(),
                code.len(),
                c"<sandbox>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );

            // Handle returned Promise if async is enabled
            if run_config.allow_async && !qjs::is_exception(eval_val) {
                let state = qjs::JS_PromiseState(self.ctx, eval_val);
                if state == JSPromiseStateEnum::Pending {
                    let mut pctx = std::ptr::null_mut();
                    loop {
                        let err = qjs::JS_ExecutePendingJob(self.rt, &mut pctx);
                        if err < 0 {
                            break;
                        }
                        if err == 0 {
                            break;
                        }
                        if let Some(deadline) = st.deadline {
                            if Instant::now() >= deadline {
                                st.reason = InterruptReason::Timeout;
                                break;
                            }
                        }
                    }
                }
                let final_state = qjs::JS_PromiseState(self.ctx, eval_val);
                if final_state == JSPromiseStateEnum::Fulfilled {
                    let resolved = qjs::JS_PromiseResult(self.ctx, eval_val);
                    qjs::sofuu_js_free_value(self.ctx, eval_val);
                    eval_val = resolved;
                } else if final_state == JSPromiseStateEnum::Rejected {
                    let rejected = qjs::JS_PromiseResult(self.ctx, eval_val);
                    qjs::sofuu_js_free_value(self.ctx, eval_val);
                    eval_val = qjs::JS_Throw(self.ctx, rejected);
                }
            }

            // Calculate precise fuel used
            let remaining_in_counter = qjs::JS_GetInterruptCounter(self.ctx).max(0) as u64;
            let executed_in_current_batch = st.fuel_quanta.saturating_sub(remaining_in_counter);
            let total_fuel = st.fuel_consumed.saturating_add(executed_in_current_batch);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let captured_logs = cst.logs.clone();

            if qjs::is_exception(eval_val) {
                let exc = qjs::sofuu_js_get_exception(self.ctx);
                let (error_msg, error_type, interrupted) = match st.reason {
                    InterruptReason::FuelExhausted => (
                        format!(
                            "Execution terminated: instruction fuel exhausted ({} limit reached)",
                            run_config.fuel_limit.unwrap_or(total_fuel)
                        ),
                        "fuel".to_string(),
                        true,
                    ),
                    InterruptReason::Timeout => (
                        format!(
                            "Execution terminated: timeout exceeded ({} ms limit)",
                            run_config
                                .timeout
                                .map(|d| d.as_millis())
                                .unwrap_or(elapsed_ms as u128)
                        ),
                        "timeout".to_string(),
                        true,
                    ),
                    InterruptReason::None => {
                        let cstr = qjs::sofuu_js_to_cstring(self.ctx, exc);
                        let msg = if !cstr.is_null() {
                            let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                            qjs::sofuu_js_free_cstring(self.ctx, cstr);
                            s
                        } else {
                            "Unknown sandbox exception".to_string()
                        };

                        let etype = if msg.contains("out of memory") {
                            "oom".to_string()
                        } else if msg.contains("stack overflow") {
                            "stack".to_string()
                        } else if msg.contains("SyntaxError") {
                            "syntax".to_string()
                        } else {
                            "runtime".to_string()
                        };

                        (msg, etype, false)
                    }
                };

                qjs::sofuu_js_free_value(self.ctx, exc);
                qjs::sofuu_js_free_value(self.ctx, eval_val);

                SandboxRunResult {
                    success: false,
                    result_json: None,
                    error: Some(error_msg),
                    error_type: Some(error_type),
                    interrupted,
                    fuel_used: total_fuel,
                    elapsed_ms,
                    logs: captured_logs,
                }
            } else {
                // Success: serialize result to JSON string
                let json_repr = serialize_val_to_json(self.ctx, eval_val);
                qjs::sofuu_js_free_value(self.ctx, eval_val);

                SandboxRunResult {
                    success: true,
                    result_json: json_repr,
                    error: None,
                    error_type: None,
                    interrupted: false,
                    fuel_used: total_fuel,
                    elapsed_ms,
                    logs: captured_logs,
                }
            }
        }
    }

    pub fn destroy(&mut self) {
        if !self.closed {
            self.closed = true;
            unsafe {
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
}

impl Drop for SandboxSession {
    fn drop(&mut self) {
        self.destroy();
    }
}

unsafe fn serialize_val_to_json(ctx: *mut JSContext, val: JSValue) -> Option<String> {
    if qjs::is_undefined(val) {
        return None;
    }
    let json_val = qjs::JS_JSONStringify(ctx, val, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
    if qjs::is_exception(json_val) || qjs::is_undefined(json_val) {
        if !qjs::is_exception(json_val) {
            qjs::sofuu_js_free_value(ctx, json_val);
        }
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

// ── JS API Bridge ────────────────────────────────────────────────

fn parse_options(ctx: *mut JSContext, opts_val: JSValueConst) -> (SandboxConfig, Option<String>) {
    let mut config = SandboxConfig::default();
    let mut globals_json = None;

    if unsafe { qjs::is_object(opts_val) } {
        unsafe {
            // fuel
            let fuel_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"fuel".as_ptr());
            if !qjs::is_undefined(fuel_val) {
                let mut fuel_num: f64 = 0.0;
                if qjs::JS_ToFloat64(ctx, &mut fuel_num, fuel_val) == 0 && fuel_num > 0.0 {
                    config.fuel_limit = Some(fuel_num as u64);
                }
            }
            qjs::sofuu_js_free_value(ctx, fuel_val);

            // timeoutMs
            let timeout_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"timeoutMs".as_ptr());
            if !qjs::is_undefined(timeout_val) {
                let mut ms: f64 = 0.0;
                if qjs::JS_ToFloat64(ctx, &mut ms, timeout_val) == 0 && ms > 0.0 {
                    config.timeout = Some(Duration::from_millis(ms as u64));
                }
            }
            qjs::sofuu_js_free_value(ctx, timeout_val);

            // maxMemoryMB
            let mem_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"maxMemoryMB".as_ptr());
            if !qjs::is_undefined(mem_val) {
                let mut mb: f64 = 0.0;
                if qjs::JS_ToFloat64(ctx, &mut mb, mem_val) == 0 && mb > 0.0 {
                    config.max_memory_bytes = (mb * 1024.0 * 1024.0) as usize;
                }
            }
            qjs::sofuu_js_free_value(ctx, mem_val);

            // maxStackKB
            let stack_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"maxStackKB".as_ptr());
            if !qjs::is_undefined(stack_val) {
                let mut kb: f64 = 0.0;
                if qjs::JS_ToFloat64(ctx, &mut kb, stack_val) == 0 && kb > 0.0 {
                    config.max_stack_bytes = (kb * 1024.0) as usize;
                }
            }
            qjs::sofuu_js_free_value(ctx, stack_val);

            // captureConsole
            let cap_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"captureConsole".as_ptr());
            if !qjs::is_undefined(cap_val) {
                config.capture_console = qjs::JS_ToBool(ctx, cap_val) != 0;
            }
            qjs::sofuu_js_free_value(ctx, cap_val);

            // throwOnError
            let throw_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"throwOnError".as_ptr());
            if !qjs::is_undefined(throw_val) {
                config.throw_on_error = qjs::JS_ToBool(ctx, throw_val) != 0;
            }
            qjs::sofuu_js_free_value(ctx, throw_val);

            // globals
            let globals_val = qjs::sofuu_js_get_property_str(ctx, opts_val, c"globals".as_ptr());
            if qjs::is_object(globals_val) {
                let json_val = qjs::JS_JSONStringify(ctx, globals_val, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
                if !qjs::is_exception(json_val) && !qjs::is_undefined(json_val) {
                    let cstr = qjs::sofuu_js_to_cstring(ctx, json_val);
                    if !cstr.is_null() {
                        globals_json = Some(CStr::from_ptr(cstr).to_string_lossy().into_owned());
                        qjs::sofuu_js_free_cstring(ctx, cstr);
                    }
                    qjs::sofuu_js_free_value(ctx, json_val);
                }
            }
            qjs::sofuu_js_free_value(ctx, globals_val);
        }
    }

    (config, globals_json)
}

unsafe fn build_js_result_object(ctx: *mut JSContext, res: SandboxRunResult) -> JSValue {
    let obj = qjs::sofuu_js_new_object(ctx);

    qjs::sofuu_js_set_property_str(ctx, obj, c"success".as_ptr(), qjs::sofuu_js_new_bool(ctx, if res.success { 1 } else { 0 }));

    if let Some(json_str) = res.result_json {
        let c_json = CString::new(json_str.clone()).unwrap_or_else(|_| CString::new("null").unwrap());
        let val = qjs::JS_ParseJSON(ctx, c_json.as_ptr(), json_str.len(), c"<result>".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, obj, c"result".as_ptr(), val);
    } else {
        qjs::sofuu_js_set_property_str(ctx, obj, c"result".as_ptr(), qjs::sofuu_js_undefined());
    }

    if let Some(ref err) = res.error {
        let c_err = CString::new(err.clone()).unwrap_or_else(|_| CString::new("error").unwrap());
        qjs::sofuu_js_set_property_str(ctx, obj, c"error".as_ptr(), qjs::sofuu_js_new_string(ctx, c_err.as_ptr()));
    }

    if let Some(ref etype) = res.error_type {
        let c_etype = CString::new(etype.clone()).unwrap_or_else(|_| CString::new("unknown").unwrap());
        qjs::sofuu_js_set_property_str(ctx, obj, c"errorType".as_ptr(), qjs::sofuu_js_new_string(ctx, c_etype.as_ptr()));
    }

    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"interrupted".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, if res.interrupted { 1 } else { 0 }),
    );

    qjs::sofuu_js_set_property_str(ctx, obj, c"fuelUsed".as_ptr(), qjs::sofuu_js_new_float64(ctx, res.fuel_used as f64));
    qjs::sofuu_js_set_property_str(ctx, obj, c"elapsedMs".as_ptr(), qjs::sofuu_js_new_float64(ctx, res.elapsed_ms));

    // Captured logs array
    let logs_arr = qjs::JS_NewArray(ctx);
    for (i, log_line) in res.logs.iter().enumerate() {
        let c_line = CString::new(log_line.clone()).unwrap_or_else(|_| CString::new("").unwrap());
        let line_val = qjs::sofuu_js_new_string(ctx, c_line.as_ptr());
        qjs::JS_SetPropertyUint32(ctx, logs_arr, i as u32, line_val);
    }
    qjs::sofuu_js_set_property_str(ctx, obj, c"logs".as_ptr(), logs_arr);

    obj
}

// ── Native JS Functions: sofuu.sandbox.run & sofuu.sandbox.eval ──

unsafe extern "C" fn js_sandbox_run(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        let msg = c"sofuu.sandbox.run requires code string argument";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }

    let code_cstr = qjs::sofuu_js_to_cstring(ctx, *argv.add(0));
    if code_cstr.is_null() {
        let msg = c"Invalid code string passed to sofuu.sandbox.run";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let code = match CStr::from_ptr(code_cstr).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            qjs::sofuu_js_free_cstring(ctx, code_cstr);
            let msg = c"Invalid UTF-8 in code string";
            return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        }
    };
    qjs::sofuu_js_free_cstring(ctx, code_cstr);

    let (config, globals_json) = if argc > 1 {
        parse_options(ctx, *argv.add(1))
    } else {
        (SandboxConfig::default(), None)
    };

    let mut session = match SandboxSession::new(config.clone()) {
        Ok(s) => s,
        Err(e) => {
            let c_err = CString::new(e).unwrap_or_else(|_| CString::new("Sandbox init error").unwrap());
            return qjs::JS_ThrowInternalError(ctx, c_err.as_ptr());
        }
    };

    let result = session.run(&code, &config, globals_json.as_deref());
    session.destroy();

    if config.throw_on_error && !result.success {
        let err_msg = result.error.unwrap_or_else(|| "Sandbox error".to_string());
        let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("Sandbox error").unwrap());
        return qjs::JS_ThrowInternalError(ctx, c_msg.as_ptr());
    }

    build_js_result_object(ctx, result)
}

unsafe extern "C" fn js_sandbox_eval(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        let msg = c"sofuu.sandbox.eval requires code string argument";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }

    let code_cstr = qjs::sofuu_js_to_cstring(ctx, *argv.add(0));
    if code_cstr.is_null() {
        let msg = c"Invalid code string passed to sofuu.sandbox.eval";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let code = match CStr::from_ptr(code_cstr).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            qjs::sofuu_js_free_cstring(ctx, code_cstr);
            let msg = c"Invalid UTF-8 in code string";
            return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        }
    };
    qjs::sofuu_js_free_cstring(ctx, code_cstr);

    let (mut config, globals_json) = if argc > 1 {
        parse_options(ctx, *argv.add(1))
    } else {
        (SandboxConfig::default(), None)
    };
    config.throw_on_error = true;

    let mut session = match SandboxSession::new(config.clone()) {
        Ok(s) => s,
        Err(e) => {
            let c_err = CString::new(e).unwrap_or_else(|_| CString::new("Sandbox init error").unwrap());
            return qjs::JS_ThrowInternalError(ctx, c_err.as_ptr());
        }
    };

    let result = session.run(&code, &config, globals_json.as_deref());
    session.destroy();

    if !result.success {
        let err_msg = result.error.unwrap_or_else(|| "Sandbox evaluation failed".to_string());
        let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("Sandbox evaluation failed").unwrap());
        return qjs::JS_ThrowInternalError(ctx, c_msg.as_ptr());
    }

    if let Some(json_str) = result.result_json {
        let c_json = CString::new(json_str.clone()).unwrap_or_else(|_| CString::new("null").unwrap());
        qjs::JS_ParseJSON(ctx, c_json.as_ptr(), json_str.len(), c"<eval-result>".as_ptr())
    } else {
        qjs::sofuu_js_undefined()
    }
}

// ── Stateful Sandbox Class ───────────────────────────────────────

unsafe extern "C" fn sandbox_instance_finalizer(_rt: *mut JSRuntime, val: JSValue) {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = qjs::JS_GetOpaque(val, class_id);
    if !ptr.is_null() {
        let mut session = Box::from_raw(ptr as *mut SandboxSession);
        session.destroy();
    }
}

unsafe extern "C" fn js_sandbox_instance_run(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = qjs::JS_GetOpaque(this_val, class_id);
    if ptr.is_null() {
        let msg = c"Invalid Sandbox instance";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let session = &mut *(ptr as *mut SandboxSession);

    if argc < 1 {
        let msg = c"Sandbox.run requires code string";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }

    let code_cstr = qjs::sofuu_js_to_cstring(ctx, *argv.add(0));
    if code_cstr.is_null() {
        let msg = c"Invalid code string";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let code = match CStr::from_ptr(code_cstr).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            qjs::sofuu_js_free_cstring(ctx, code_cstr);
            let msg = c"Invalid UTF-8 in code string";
            return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        }
    };
    qjs::sofuu_js_free_cstring(ctx, code_cstr);

    let (run_config, globals_json) = if argc > 1 {
        parse_options(ctx, *argv.add(1))
    } else {
        (session.config.clone(), None)
    };

    let result = session.run(&code, &run_config, globals_json.as_deref());
    if run_config.throw_on_error && !result.success {
        let err_msg = result.error.unwrap_or_else(|| "Sandbox error".to_string());
        let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("Sandbox error").unwrap());
        return qjs::JS_ThrowInternalError(ctx, c_msg.as_ptr());
    }

    build_js_result_object(ctx, result)
}

unsafe extern "C" fn js_sandbox_instance_eval(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = qjs::JS_GetOpaque(this_val, class_id);
    if ptr.is_null() {
        let msg = c"Invalid Sandbox instance";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let session = &mut *(ptr as *mut SandboxSession);

    if argc < 1 {
        let msg = c"Sandbox.eval requires code string";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }

    let code_cstr = qjs::sofuu_js_to_cstring(ctx, *argv.add(0));
    if code_cstr.is_null() {
        let msg = c"Invalid code string";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }
    let code = match CStr::from_ptr(code_cstr).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            qjs::sofuu_js_free_cstring(ctx, code_cstr);
            let msg = c"Invalid UTF-8 in code string";
            return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        }
    };
    qjs::sofuu_js_free_cstring(ctx, code_cstr);

    let (mut run_config, globals_json) = if argc > 1 {
        parse_options(ctx, *argv.add(1))
    } else {
        (session.config.clone(), None)
    };
    run_config.throw_on_error = true;

    let result = session.run(&code, &run_config, globals_json.as_deref());
    if !result.success {
        let err_msg = result.error.unwrap_or_else(|| "Sandbox evaluation failed".to_string());
        let c_msg = CString::new(err_msg).unwrap_or_else(|_| CString::new("Sandbox evaluation failed").unwrap());
        return qjs::JS_ThrowInternalError(ctx, c_msg.as_ptr());
    }

    if let Some(json_str) = result.result_json {
        let c_json = CString::new(json_str.clone()).unwrap_or_else(|_| CString::new("null").unwrap());
        qjs::JS_ParseJSON(ctx, c_json.as_ptr(), json_str.len(), c"<eval-result>".as_ptr())
    } else {
        qjs::sofuu_js_undefined()
    }
}

unsafe extern "C" fn js_sandbox_instance_close(
    _ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let ptr = qjs::JS_GetOpaque(this_val, class_id);
    if !ptr.is_null() {
        let session = &mut *(ptr as *mut SandboxSession);
        session.destroy();
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_sandbox_create(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let (config, globals_json) = if argc > 0 {
        parse_options(ctx, *argv.add(0))
    } else {
        (SandboxConfig::default(), None)
    };

    let mut session = match SandboxSession::new(config) {
        Ok(s) => Box::new(s),
        Err(e) => {
            let c_err = CString::new(e).unwrap_or_else(|_| CString::new("Sandbox init error").unwrap());
            return qjs::JS_ThrowInternalError(ctx, c_err.as_ptr());
        }
    };

    if let Some(ref json) = globals_json {
        if let Err(e) = inject_globals_json(session.ctx, json) {
            session.destroy();
            let c_err = CString::new(e).unwrap_or_else(|_| CString::new("Globals inject error").unwrap());
            return qjs::JS_ThrowInternalError(ctx, c_err.as_ptr());
        }
    }

    let class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    let obj = qjs::JS_NewObjectClass(ctx, class_id as c_int);
    if qjs::is_exception(obj) {
        session.destroy();
        return obj;
    }

    let run_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_run, c"run".as_ptr(), 2);
    qjs::sofuu_js_set_property_str(ctx, obj, c"run".as_ptr(), run_fn);

    let eval_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_eval, c"eval".as_ptr(), 2);
    qjs::sofuu_js_set_property_str(ctx, obj, c"eval".as_ptr(), eval_fn);

    let close_fn = qjs::sofuu_js_new_cfunction(ctx, js_sandbox_instance_close, c"close".as_ptr(), 0);
    qjs::sofuu_js_set_property_str(ctx, obj, c"close".as_ptr(), close_fn);

    let ptr = Box::into_raw(session);
    qjs::JS_SetOpaque(obj, ptr as *mut c_void);

    obj
}

// ── Module Registration ──────────────────────────────────────────

pub unsafe fn mod_sandbox_register(ctx: *mut JSContext) {
    let rt = qjs::JS_GetRuntime(ctx);

    // Initialize SandboxInstance class ID if not done
    let mut class_id = SANDBOX_CLASS_ID.load(Ordering::Relaxed);
    if class_id == 0 {
        qjs::JS_NewClassID(&mut class_id);
        SANDBOX_CLASS_ID.store(class_id, Ordering::Relaxed);

        let class_def = JSClassDef {
            class_name: c"SandboxSession".as_ptr(),
            finalizer: Some(sandbox_instance_finalizer),
            gc_mark: std::ptr::null_mut(),
            call: std::ptr::null_mut(),
            exotic: std::ptr::null_mut(),
        };
        qjs::JS_NewClass(rt, class_id, &class_def);
    }

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu_obj));
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

    #[test]
    fn test_sandbox_config_is_clamped() {
        let mut cfg = SandboxConfig::default();
        cfg.fuel_limit = Some(u64::MAX);
        cfg.timeout = Some(Duration::from_secs(3600));
        cfg.max_memory_bytes = usize::MAX;
        cfg.max_stack_bytes = usize::MAX;
        cfg.clamp();
        assert_eq!(cfg.fuel_limit, Some(SANDBOX_MAX_FUEL));
        assert_eq!(
            cfg.timeout,
            Some(Duration::from_millis(SANDBOX_MAX_TIMEOUT_MS))
        );
        assert_eq!(cfg.max_memory_bytes, SANDBOX_MAX_MEMORY_BYTES);
        assert_eq!(cfg.max_stack_bytes, SANDBOX_MAX_STACK_BYTES);

        let mut tiny = SandboxConfig::default();
        tiny.max_memory_bytes = 1;
        tiny.max_stack_bytes = 1;
        tiny.clamp();
        assert_eq!(tiny.max_memory_bytes, SANDBOX_MIN_MEMORY_BYTES);
        assert_eq!(tiny.max_stack_bytes, SANDBOX_MIN_STACK_BYTES);
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
        // The default fuel budget (5M) can run out before a 10 ms deadline is
        // observed on a loaded runner — the interrupt then reports "fuel" and
        // the assertion below fails on a healthy engine (seen in the P5
        // matrix run). Raise fuel to the clamp ceiling so only the deadline
        // can stop this loop.
        config.fuel_limit = Some(SANDBOX_MAX_FUEL);
        let start = Instant::now();
        let res = session.run("while(true) {}", &config, None);
        assert!(!res.success);
        assert!(res.interrupted);
        assert_eq!(res.error_type.as_deref(), Some("timeout"));
        // Promptness, not a benchmark: a busy runner can delay the poll.
        assert!(start.elapsed() < Duration::from_secs(2));
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
        config.capture_console = true;

        let code = "console.log('alpha', 1); console.warn('beta'); 'ok';";
        let res = session.run(code, &config, None);
        assert!(res.success);
        assert_eq!(res.logs.len(), 2);
        assert!(res.logs[0].contains("alpha 1"));
        assert!(res.logs[1].contains("beta"));
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
