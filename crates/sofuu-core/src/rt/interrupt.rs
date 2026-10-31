// rt/interrupt.rs — the engine runtime's ONE interrupt handler, and the
// regexp backtracking budget it enforces (AUDIT-2026-10-31 P1-14).
//
// Two things need to interrupt a running program on the engine context:
//
//   * sofuu.host.armDeadline(ms) — a wall clock on the whole call. The state
//     machine is rt/deadline.rs; this module only calls its poll.
//   * regexp backtracking. /(a+)+$/.test('a'.repeat(28)+'b') took 22.4 s with
//     the thread blocked: QuickJS's libregexp has no budget of its own. The
//     vendored engine already polls the interrupt handler from the
//     backtracking loop (deps/quickjs/libregexp.c, S2/C-02), but nothing ever
//     installed a handler outside the sandbox, so the poll was a no-op.
//
// QuickJS has a single interrupt slot per runtime, so this module owns it for
// the engine context (installed once, at boot) and dispatches in order:
// deadline arms first, then the regexp window.
//
// The regexp window is a wall clock around each call into a regexp-taking
// prototype method. A short bootstrap script (REGEX_GUARD_JS) wraps
// RegExp.prototype.test/exec, the RegExp @@-methods, and the String methods
// that accept a RegExp; each call opens a window and closes it in a finally,
// so a benign 2 ms match costs one atomic store on entry and one on exit.
//
// Budget: SOFUU_REGEX_BUDGET_MS, default 1000 ms per call, 0 disables the
// guard (no wrappers installed, no window opened). On expiry the poll returns
// 1 and QuickJS raises its UNCATCHABLE InternalError("interrupted"); the host
// rewrites that message into E_REGEX_BUDGET in report_uncaught, because a
// catastrophic match must be loud rather than a spinner. The guard is
// one-shot per call: firing clears the window so the next regexp is judged on
// its own work.
//
// The budget is a product-level policy (a legitimate 200 MB scan can take
// longer than a second), which is why it is generous, overridable and
// disableable rather than tied to the sandbox's per-run meter.

use std::ffi::c_int;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use sofuu_ffi::qjs::{self, JSContext, JSRuntime, JSValue, JSValueConst};

use crate::rt::deadline;

/// Default per-call regexp budget: 1 s. The measured catastrophic cases are
/// 1.4 s (n=24), 5.6 s (n=26) and 22.4 s (n=28); a legitimate match over a
/// large document still fits comfortably, and SOFUU_REGEX_BUDGET_MS moves it.
const DEFAULT_BUDGET_MS: u64 = 1_000;
/// The bytecode poll interval. QuickJS's own default
/// (JS_INTERRUPT_COUNTER_INIT) is the same 10 000, so installing the handler
/// adds no extra polling.
const POLL_BATCH: c_int = 10_000;

/// Per-call budget in microseconds; 0 = guard disabled.
static BUDGET_US: AtomicU64 = AtomicU64::new(0);
/// Absolute end (us since the epoch) of the regexp call in flight; 0 = none.
static WINDOW_END_US: AtomicU64 = AtomicU64::new(0);
/// Wrapper nesting depth (String.prototype.match -> RegExp @@match -> exec).
static DEPTH: AtomicI32 = AtomicI32::new(0);
/// Set when the window fired, so the host can name the abort E_REGEX_BUDGET.
static FIRED_US: AtomicU64 = AtomicU64::new(0);
/// The budget that fired (ms).
static FIRED_MS: AtomicU64 = AtomicU64::new(0);
/// Whether the handler is on this runtime (install is idempotent).
static INSTALLED: AtomicI32 = AtomicI32::new(0);

/// The bootstrap that wraps the regexp entry points. Native funnel:
/// __sofuu_regex_enter() / __sofuu_regex_leave() (below).
const REGEX_GUARD_JS: &[u8] = b"(function () {\n\
  var enter = globalThis.__sofuu_regex_enter, leave = globalThis.__sofuu_regex_leave;\n\
  if (typeof enter !== 'function' || typeof leave !== 'function') return;\n\
  function wrap(owner, name, regexArg) {\n\
    var orig = owner[name];\n\
    if (typeof orig !== 'function' || orig.__sofuuRegexGuarded) return;\n\
    var guarded = function () {\n\
      if (regexArg && !(arguments.length > 0 && arguments[0] instanceof RegExp)) {\n\
        return orig.apply(this, arguments);\n\
      }\n\
      enter();\n\
      try { return orig.apply(this, arguments); } finally { leave(); }\n\
    };\n\
    try {\n\
      Object.defineProperty(guarded, 'name', { value: orig.name, configurable: true });\n\
      Object.defineProperty(guarded, 'length', { value: orig.length, configurable: true });\n\
      guarded.__sofuuRegexGuarded = 1;\n\
      Object.defineProperty(owner, name, {\n\
        value: guarded, writable: true, enumerable: false, configurable: true\n\
      });\n\
    } catch (e) { /* frozen prototype: leave that entry point unguarded */ }\n\
  }\n\
  wrap(RegExp.prototype, 'exec');\n\
  wrap(RegExp.prototype, 'test');\n\
  var strings = ['match', 'matchAll', 'replace', 'replaceAll', 'search', 'split'];\n\
  for (var i = 0; i < strings.length; i++) wrap(String.prototype, strings[i], true);\n\
  var symbols = ['match', 'matchAll', 'replace', 'search', 'split'];\n\
  for (var j = 0; j < symbols.length; j++) wrap(RegExp.prototype, Symbol[symbols[j]]);\n\
})();\n\0";

/// The QuickJS poll hook for the engine runtime: 1 aborts the running script
/// with the uncatchable 'InternalError: interrupted'.
unsafe extern "C" fn engine_interrupt_cb(
    _rt: *mut JSRuntime,
    _opaque: *mut std::ffi::c_void,
) -> c_int {
    if deadline::poll_expired() != 0 {
        return 1;
    }
    if regex_expired() {
        return 1;
    }
    0
}

/// Has the regexp call in flight overrun its budget? Records the firing and
/// clears the window (one-shot) so the next call starts clean — nothing can
/// run a finally after an uncatchable interrupt.
fn regex_expired() -> bool {
    let end = WINDOW_END_US.load(Ordering::Relaxed);
    if end == 0 {
        return false;
    }
    let now = deadline::now_us();
    if now < end {
        return false;
    }
    FIRED_US.store(now, Ordering::SeqCst);
    FIRED_MS.store(BUDGET_US.load(Ordering::Relaxed) / 1000, Ordering::SeqCst);
    WINDOW_END_US.store(0, Ordering::SeqCst);
    DEPTH.store(0, Ordering::SeqCst);
    true
}

/// Rewrite a top-level error message when the regexp budget aborted the call.
///
/// Same contract as deadline::classify_interrupted: only a message carrying
/// "interrupted" while THIS guard actually fired is rewritten, so an unrelated
/// InternalError is left alone.
pub fn classify_interrupted(msg: &str) -> Option<String> {
    if FIRED_US.load(Ordering::Relaxed) == 0 || !msg.to_ascii_lowercase().contains("interrupted") {
        return None;
    }
    Some(format!(
        "E_REGEX_BUDGET: regexp abandoned — backtracking exceeded the {} ms per-call budget. \
         The pattern is catastrophic on this input (nested quantifiers such as /(a+)+$/ \
         backtrack exponentially). A regexp cannot be cancelled, so the call was aborted; \
         raise or disable the budget with SOFUU_REGEX_BUDGET_MS (0 disables it)",
        FIRED_MS.load(Ordering::Relaxed)
    ))
}

/// Forget a previous firing (a fresh window must not inherit its blame).
fn clear_fired() {
    FIRED_US.store(0, Ordering::SeqCst);
}

/// __sofuu_regex_enter() — open the window on the outermost call.
unsafe extern "C" fn js_regex_enter(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let budget = BUDGET_US.load(Ordering::Relaxed);
    if budget != 0 {
        let depth = DEPTH.load(Ordering::SeqCst);
        if depth == 0 {
            clear_fired();
            WINDOW_END_US.store(deadline::now_us().saturating_add(budget), Ordering::SeqCst);
        }
        DEPTH.store(depth.saturating_add(1), Ordering::SeqCst);
    }
    qjs::sofuu_js_undefined()
}

/// __sofuu_regex_leave() — close the window when the outermost call returns.
unsafe extern "C" fn js_regex_leave(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let depth = DEPTH.load(Ordering::SeqCst);
    if depth <= 1 {
        DEPTH.store(0, Ordering::SeqCst);
        WINDOW_END_US.store(0, Ordering::SeqCst);
    } else {
        DEPTH.store(depth - 1, Ordering::SeqCst);
    }
    qjs::sofuu_js_undefined()
}

/// The configured budget in ms: SOFUU_REGEX_BUDGET_MS, else the default. A
/// value that does not parse keeps the default — a typo must not silently
/// disable a security guard.
fn budget_ms_from_env() -> u64 {
    match std::env::var("SOFUU_REGEX_BUDGET_MS") {
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() {
                return DEFAULT_BUDGET_MS;
            }
            t.parse::<u64>().unwrap_or(DEFAULT_BUDGET_MS)
        }
        Err(_) => DEFAULT_BUDGET_MS,
    }
}

/// Install the interrupt handler on this runtime and (unless disabled) wrap
/// the regexp entry points. Idempotent; safe to call from both engine boot and
/// armDeadline.
///
/// # Safety
/// CONTEXT must be a live engine context on this thread.
pub unsafe fn ensure_installed(ctx: *mut JSContext) {
    if ctx.is_null() {
        return;
    }
    let rt = unsafe { qjs::JS_GetRuntime(ctx) };
    if rt.is_null() {
        return;
    }
    if INSTALLED.swap(1, Ordering::SeqCst) != 0 {
        return;
    }
    unsafe { qjs::JS_SetInterruptHandler(rt, Some(engine_interrupt_cb), std::ptr::null_mut()) };
    unsafe { qjs::JS_SetInterruptCounter(ctx, POLL_BATCH) };

    let ms = budget_ms_from_env();
    if ms == 0 {
        /* SOFUU_REGEX_BUDGET_MS=0: the deadline arms still work, no regexp
         * window is ever opened and no prototype is touched. */
        BUDGET_US.store(0, Ordering::SeqCst);
        return;
    }
    BUDGET_US.store(ms.saturating_mul(1000), Ordering::SeqCst);

    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    unsafe {
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"__sofuu_regex_enter".as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, js_regex_enter, c"__sofuu_regex_enter".as_ptr(), 0),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"__sofuu_regex_leave".as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, js_regex_leave, c"__sofuu_regex_leave".as_ptr(), 0),
        );
    }
    unsafe { qjs::sofuu_js_free_value(ctx, global) };
    let r = unsafe {
        qjs::JS_Eval(
            ctx,
            REGEX_GUARD_JS.as_ptr() as *const c_char,
            REGEX_GUARD_JS.len() - 1,
            c"<regex-guard>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        )
    };
    if unsafe { qjs::is_exception(r) } {
        /* A guard that cannot install must not take the runtime down: report
         * it on stderr and carry on unguarded (fail loud, keep running). */
        unsafe { qjs::js_std_dump_error(ctx) };
        eprintln!("[sofuu] the regexp backtracking guard could not install (see above)");
    }
    unsafe { qjs::sofuu_js_free_value(ctx, r) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::CtxPtr;
    use std::ffi::{CStr, CString};
    use std::time::Instant;

    /// A bare runtime + context with the guard armed at a small budget —
    /// the M0 test pattern fs.rs uses.
    unsafe fn guarded_engine(ms: u64) -> (*mut JSRuntime, *mut JSContext) {
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        BUDGET_US.store(ms * 1000, Ordering::SeqCst);
        unsafe { qjs::JS_SetInterruptHandler(rt, Some(engine_interrupt_cb), std::ptr::null_mut()) };
        unsafe { qjs::JS_SetInterruptCounter(ctx, POLL_BATCH) };
        let g = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                g,
                c"__sofuu_regex_enter".as_ptr(),
                qjs::sofuu_js_new_cfunction(ctx, js_regex_enter, c"__sofuu_regex_enter".as_ptr(), 0),
            );
            qjs::sofuu_js_set_property_str(
                ctx,
                g,
                c"__sofuu_regex_leave".as_ptr(),
                qjs::sofuu_js_new_cfunction(ctx, js_regex_leave, c"__sofuu_regex_leave".as_ptr(), 0),
            );
            qjs::sofuu_js_free_value(ctx, g);
        }
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                REGEX_GUARD_JS.as_ptr() as *const c_char,
                REGEX_GUARD_JS.len() - 1,
                c"<regex-guard>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "the guard must install");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };
        (rt, ctx)
    }

    unsafe fn eval(ctx: *mut JSContext, src: &str) -> (bool, String) {
        let code = CString::new(src).unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                code.as_ptr(),
                code.to_bytes().len(),
                c"<regex-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            let exc = unsafe { qjs::JS_GetException(ctx) };
            let s = unsafe { qjs::JS_ToString(ctx, exc) };
            let p = unsafe { qjs::sofuu_js_to_cstring(ctx, s) };
            let msg = if p.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            };
            unsafe {
                qjs::sofuu_js_free_cstring(ctx, p);
                qjs::sofuu_js_free_value(ctx, s);
                qjs::sofuu_js_free_value(ctx, exc);
                qjs::sofuu_js_free_value(ctx, r);
            }
            return (false, msg);
        }
        let s = unsafe { qjs::JS_ToString(ctx, r) };
        let p = unsafe { qjs::sofuu_js_to_cstring(ctx, s) };
        let out = if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        unsafe {
            qjs::sofuu_js_free_cstring(ctx, p);
            qjs::sofuu_js_free_value(ctx, s);
            qjs::sofuu_js_free_value(ctx, r);
        }
        (true, out)
    }

    /// P1-14: the measured 22.4 s case (n=28) must abort near the budget
    /// instead of pinning the thread. n=30 is the same shape, bigger.
    #[test]
    fn catastrophic_backtracking_is_bounded() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context; torn down below.
        let (rt, ctx) = unsafe { guarded_engine(50) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        let start = Instant::now();
        let (ok, out) = unsafe { eval(ctx, r#"/(a+)+$/.test("a".repeat(30) + "b")"#) };
        let took = start.elapsed();
        assert!(!ok, "a catastrophic pattern must not complete: {out}");
        assert!(
            out.to_lowercase().contains("interrupted"),
            "the engine's interrupt is what must stop it, got: {out}"
        );
        assert!(
            took < std::time::Duration::from_millis(2000),
            "bounded match took {took:?} with a 50 ms budget"
        );
        /* The host can name the abort: the classification only fires for a
         * message carrying "interrupted" after THIS guard fired. */
        let named = classify_interrupted(&out).expect("the abort must be classified");
        assert!(named.starts_with("E_REGEX_BUDGET"), "{named}");
        assert!(named.contains("50 ms"), "{named}");

        // SAFETY: teardown in M0 order.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// The guard must not touch ordinary matching, and must not stay fired:
    /// a benign pattern after an abort still works.
    #[test]
    fn benign_regexps_are_unaffected() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context; torn down below.
        let (rt, ctx) = unsafe { guarded_engine(200) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        let (ok, out) = unsafe { eval(ctx, r#"/a+/.test("xaay")"#) };
        assert!(ok, "a benign pattern must run: {out}");
        assert_eq!(out, "true");
        // String methods that take a RegExp go through the same window.
        let (ok2, out2) = unsafe { eval(ctx, r#""x a.y z".split(/[ .]/).length"#) };
        assert!(ok2, "split(RegExp) must run: {out2}");
        assert_eq!(out2, "4");
        let (ok3, out3) = unsafe { eval(ctx, r##""a1b2".replace(/\d/g, "#")"##) };
        assert!(ok3, "replace(RegExp) must run: {out3}");
        assert_eq!(out3, "a#b#");
        // A non-RegExp argument must not even open a window.
        let (ok4, out4) = unsafe { eval(ctx, r#""a.b".split(".").length"#) };
        assert!(ok4, "split(string) must run: {out4}");
        assert_eq!(out4, "2");
        // Abort, then a benign match: the one-shot reset must leave it usable.
        let (aborted, _) = unsafe { eval(ctx, r#"/(a+)+$/.test("a".repeat(30) + "b")"#) };
        assert!(!aborted, "the catastrophic pattern must abort");
        let (ok5, out5) = unsafe { eval(ctx, r#"/b/.test("abc")"#) };
        assert!(ok5, "a regexp after an abort must still run: {out5}");
        assert_eq!(out5, "true");

        // SAFETY: teardown in M0 order.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }
}
