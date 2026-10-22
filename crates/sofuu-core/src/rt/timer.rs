// rt/timer.rs — setTimeout / setInterval / clearTimeout / clearInterval /
// sofuu.sleep(ms) (PLAN-RUST-MIGRATION M2).
//
// Port of the deleted `src/io/timer.c`, semantics verbatim. Each timer is a
// malloc'd uv_timer_t whose leading `data` field points back at a Box'd
// Timer. The callback: (1) fires the JS function (or resolves a sleep
// promise), (2) for one-shot timers stops + closes the handle — the close
// callback frees everything (Box drop + handle free).
//
// C symbol replaced: `mod_timer_register` (engine.c calls it unchanged).

use std::cell::RefCell;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
use std::ptr;

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvCheck, UvHandle, UvTimer};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{sofuu_flush_jobs, sofuu_flush_jobs_bounded, sofuu_promise_new, sofuu_promise_resolve, PromiseHandle};

extern "C" {
    /// mod_process.c — dispatch to process.on('uncaughtException'); stays C
    /// until M3. Does NOT take ownership of `exc`.
    fn process_dispatch_uncaught(ctx: *mut JSContext, exc: JSValue) -> c_int;
}

/// Per-timer context (replaces the C `sofuu_timer_t`, which EMBEDDED the
/// uv handle — here the handle is a separate malloc'd block so the struct
/// needs no C layout).
struct Timer {
    ctx: *mut JSContext,
    callback: JSValue,
    repeat: c_int, /* 1 = setInterval, 0 = setTimeout */
    closing: c_int, /* 1 = clearInterval called, deferred close pending */
    id: c_uint,
    handle: *mut UvTimer,
    async_context: JSValue,
}

const MAX_TIMERS: usize = 1024;

thread_local! {
    /// Registry of active timers (linear scan, exactly like the C array).
    static TIMERS: RefCell<Vec<*mut Timer>> = const { RefCell::new(Vec::new()) };
    /// Global ID counter (wraps, like C's g_next_id++).
    static NEXT_ID: RefCell<c_uint> = const { RefCell::new(1) };
}

fn timer_register(t: *mut Timer) -> bool {
    TIMERS.with(|ts| {
        let mut ts = ts.borrow_mut();
        if ts.len() < MAX_TIMERS {
            ts.push(t);
            true
        } else {
            false
        }
    })
}

fn timer_unregister(t: *mut Timer) {
    TIMERS.with(|ts| {
        let mut ts = ts.borrow_mut();
        if let Some(i) = ts.iter().position(|x| *x == t) {
            ts.swap_remove(i);
        }
    });
}

unsafe fn timer_find(id: c_uint) -> Option<*mut Timer> {
    TIMERS.with(|ts| ts.borrow().iter().copied().find(|t| unsafe { (**t).id } == id))
}

/// Close callback: free the Box'd Timer + the handle storage once libuv
/// releases the handle.
unsafe extern "C" fn on_timer_close(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    // SAFETY: data was set at init; the handle is fully closed here.
    let t = *(handle as *mut *mut Timer);
    if t.is_null() {
        return;
    }
    qjs::sofuu_js_free_value((*t).ctx, (*t).callback);
    qjs::sofuu_js_free_value((*t).ctx, (*t).async_context);
    libc::free(handle as *mut c_void);
    drop(Box::from_raw(t));
}

/// libuv timer callback.
unsafe extern "C" fn on_timer(handle: *mut UvTimer) {
    // SAFETY: data set at init.
    let t = *(handle as *mut *mut Timer);
    let ctx = (*t).ctx;

    /* If clearInterval was called before this tick, skip the callback */
    if (*t).closing != 0 {
        return;
    }

    /* Call the JS callback with no arguments */
    let rt = qjs::JS_GetRuntime(ctx);
    let prev_ctx = qjs::JS_GetAsyncContext(rt);
    qjs::JS_SetAsyncContext(rt, (*t).async_context);
    let ret = qjs::JS_Call(ctx, (*t).callback, qjs::sofuu_js_undefined(), 0, ptr::null());
    qjs::JS_SetAsyncContext(rt, prev_ctx);
    if qjs::is_exception(ret) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        /* process.on('uncaughtException') gets the first shot; fall back to
         * the standard dump when no handler is registered. */
        if process_dispatch_uncaught(ctx, exc) == 0 {
            let str_v = qjs::JS_ToString(ctx, exc);
            let s = qjs::sofuu_js_to_cstring(ctx, str_v);
            let msg = if s.is_null() {
                "?".to_string()
            } else {
                std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned()
            };
            eprintln!("[timer] Uncaught: {}", msg);
            if !s.is_null() {
                qjs::sofuu_js_free_cstring(ctx, s);
            }
            qjs::sofuu_js_free_value(ctx, str_v);
        }
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    sofuu_flush_jobs_bounded(ctx, 1024);

    /* Check if the callback called clearInterval on THIS timer */
    if (*t).closing != 0 {
        return; /* already queued for close inside the callback */
    }

    /* One-shot: schedule close — free happens in on_timer_close */
    if (*t).repeat == 0 {
        uv::uv_timer_stop(handle);
        timer_unregister(t);
        uv::uv_close(handle as *mut UvHandle, Some(on_timer_close));
    }
}

/// Create + arm a timer; mirrors sofuu_timer_create. Returns 0 when the
/// registry is full (the timer is destroyed on the failure path).
unsafe fn timer_create(ctx: *mut JSContext, cb: JSValueConst, delay_ms: u64, repeat: c_int) -> c_uint {
    let id = NEXT_ID.with(|n| {
        let mut n = n.borrow_mut();
        let id = *n;
        *n = n.wrapping_add(1);
        id
    });
    let rt = qjs::JS_GetRuntime(ctx);
    let async_ctx = qjs::JS_GetAsyncContext(rt);
    let t = Box::into_raw(Box::new(Timer {
        ctx,
        callback: qjs::sofuu_js_dup_value(ctx, cb),
        repeat,
        closing: 0,
        id,
        handle: ptr::null_mut(),
        async_context: async_ctx,
    }));
    let handle = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
    // SAFETY: uv handles start with the `data` field.
    *(handle as *mut *mut Timer) = t;
    (*t).handle = handle;

    uv::uv_timer_init(sofuu_loop_get(), handle);
    uv::uv_timer_start(handle, Some(on_timer), delay_ms, if repeat != 0 { delay_ms } else { 0 });
    /* F-2: armed per-ctx — a multi-engine teardown must close it. */
    unsafe { crate::rt::event_loop::track_handle(ctx, handle as *mut UvHandle, Some(on_timer_close)) };

    if !timer_register(t) {
        /* Registry full: stop and destroy instead of half-registering (an
         * unregistered interval could never be cleared and would fire
         * forever). on_timer_close frees t. */
        uv::uv_timer_stop(handle);
        uv::uv_close(handle as *mut UvHandle, Some(on_timer_close));
        return 0;
    }
    (*t).id
}

unsafe fn js_set_timer(ctx: *mut JSContext, argc: c_int, argv: *const JSValueConst, repeat: c_int) -> JSValue {
    // SAFETY: argv is valid for argc entries (checked below).
    let first = if argc >= 1 { unsafe { *argv } } else { qjs::sofuu_js_undefined() };
    if argc < 1 || qjs::JS_IsFunction(ctx, first) == 0 {
        let msg = if repeat != 0 {
            c"setInterval: first argument must be a function"
        } else {
            c"setTimeout: first argument must be a function"
        };
        // SAFETY: fixed message; variadic call with no varargs.
        return unsafe { qjs::JS_ThrowTypeError(ctx, msg.as_ptr()) };
    }

    let mut delay: c_uint = 0;
    if argc >= 2 {
        qjs::sofuu_js_to_uint32(ctx, &mut delay, *argv.add(1));
    }

    let id = timer_create(ctx, first, delay as u64, repeat);
    if id == 0 {
        let msg = c"too many timers (max %d)";
        return unsafe { qjs::JS_ThrowTypeError(ctx, msg.as_ptr(), MAX_TIMERS as c_int) };
    }
    // SAFETY: ctx valid; returns an immediate uint32 value.
    unsafe { qjs::sofuu_js_new_uint32(ctx, id) }
}

unsafe extern "C" fn js_settimeout(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    js_set_timer(ctx, argc, argv, 0)
}

unsafe extern "C" fn js_setinterval(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    js_set_timer(ctx, argc, argv, 1)
}

unsafe extern "C" fn js_clear_timer(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let mut id: c_uint = 0;
    qjs::sofuu_js_to_uint32(ctx, &mut id, *argv);
    if let Some(t) = timer_find(id) {
        if (*t).closing == 0 {
            (*t).closing = 1; /* mark: do not re-enter callback */
            uv::uv_timer_stop((*t).handle);
            timer_unregister(t);
            /* Safe close: libuv calls on_timer_close when fully done */
            uv::uv_close((*t).handle as *mut UvHandle, Some(on_timer_close));
        }
    }
    qjs::sofuu_js_undefined()
}

// ── sofuu.sleep(ms) — Promise-based delay ─────────────────────────────

struct SleepReq {
    promise: *mut PromiseHandle,
    handle: *mut UvTimer,
}

unsafe extern "C" fn on_sleep_close(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    /* Timer handle fully closed — safe to free the container */
    // SAFETY: data set at init.
    let req = *(handle as *mut *mut SleepReq);
    if !req.is_null() {
        if !(*req).promise.is_null() {
            let p = (*req).promise;
            let ctx = (*p).ctx;
            qjs::sofuu_js_free_value(ctx, (*p).resolve);
            qjs::sofuu_js_free_value(ctx, (*p).reject);
            qjs::sofuu_js_free_value(ctx, (*p).async_context);
            libc::free(p as *mut c_void);
        }
        libc::free(handle as *mut c_void);
        drop(Box::from_raw(req));
    }
}

unsafe extern "C" fn on_sleep_timer(handle: *mut UvTimer) {
    // SAFETY: data set at init.
    let req = *(handle as *mut *mut SleepReq);
    let p = (*req).promise;
    let ctx = (*p).ctx; /* save before resolve frees p */

    (*req).promise = ptr::null_mut();
    uv::uv_timer_stop(handle);
    sofuu_promise_resolve(p, qjs::sofuu_js_undefined()); /* frees p */
    sofuu_flush_jobs(ctx);

    /* Schedule handle close — frees req in on_sleep_close */
    uv::uv_close(handle as *mut UvHandle, Some(on_sleep_close));
}

unsafe extern "C" fn js_sleep(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut ms: c_uint = 0;
    if argc >= 1 {
        qjs::sofuu_js_to_uint32(ctx, &mut ms, *argv);
    }

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    if qjs::is_exception(promise) {
        return promise;
    }

    let req = Box::into_raw(Box::new(SleepReq {
        promise: out,
        handle: ptr::null_mut(),
    }));
    let handle = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
    *(handle as *mut *mut SleepReq) = req;
    (*req).handle = handle;

    uv::uv_timer_init(sofuu_loop_get(), handle);
    uv::uv_timer_start(handle, Some(on_sleep_timer), ms as u64, 0);
    /* F-2: armed per-ctx — a multi-engine teardown must close it. */
    unsafe {
        crate::rt::event_loop::track_handle(ctx, handle as *mut UvHandle, Some(on_sleep_close))
    };

    promise
}


unsafe extern "C" fn js_scheduler_yield(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let zero = qjs::sofuu_js_new_uint32(ctx, 0);
    let ret = js_sleep(ctx, this_val, 1, &zero);
    qjs::sofuu_js_free_value(ctx, zero);
    ret
}

unsafe extern "C" fn js_setimmediate(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    js_set_timer(ctx, argc, argv, 0)
}

unsafe extern "C" fn js_queue_microtask(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::JS_IsFunction(ctx, *argv) == 0 {
        let msg = c"queueMicrotask: first argument must be a function";
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    }

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    if !qjs::is_exception(promise) {
        let then_fn = qjs::sofuu_js_get_property_str(ctx, promise, c"then".as_ptr());
        if !qjs::is_undefined(then_fn) {
            let res = qjs::JS_Call(ctx, then_fn, promise, 1, argv);
            qjs::sofuu_js_free_value(ctx, res);
            qjs::sofuu_js_free_value(ctx, then_fn);
        }
        sofuu_promise_resolve(out, qjs::sofuu_js_undefined());
        qjs::sofuu_js_free_value(ctx, promise);
    }
    qjs::sofuu_js_undefined()
}

// ── Registration (C symbol replacement for mod_timer_register) ────────

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_timer_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"setTimeout".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_settimeout, c"setTimeout".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"setInterval".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_setinterval, c"setInterval".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"clearTimeout".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_clear_timer, c"clearTimeout".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"clearInterval".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_clear_timer, c"clearInterval".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"setImmediate".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_setimmediate, c"setImmediate".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"clearImmediate".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_clear_timer, c"clearImmediate".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"queueMicrotask".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_queue_microtask, c"queueMicrotask".as_ptr(), 1),
    );

    // globalThis.scheduler = { yield: js_scheduler_yield }
    let scheduler_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        scheduler_obj,
        c"yield".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_scheduler_yield, c"yield".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(ctx, global, c"scheduler".as_ptr(), scheduler_obj);

    /* sofuu.sleep(ms) and sofuu.yield() — bonus utilities */
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
    }
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu_obj,
        c"sleep".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_sleep, c"sleep".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu_obj,
        c"yield".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_scheduler_yield, c"yield".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::promise::sofuu_rt_install_rejection_tracker;
    use sofuu_ffi::qjs::{self, CtxPtr, JSContext, JSValueConst};

    /// M2 proof: the FULL timer surface (setTimeout / setInterval /
    /// clearInterval / sofuu.sleep) driven from JS through the M1 loop —
    /// everything created, fired, resolved and cleaned up by the Rust port.
    #[test]
    fn m2_timers_fire_and_clear_through_rust_loop() {
        // The process-global uv loop is shared — serialize loop-driving
        // tests (see rt/mod.rs TEST_LOOP_LOCK).
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        // SAFETY: tracker + loop, test-scoped like the M1 proof test.
        unsafe { sofuu_rt_install_rejection_tracker(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers setTimeout/setInterval/clearTimeout/clearInterval
        // + sofuu.sleep on a fresh context.
        unsafe { mod_timer_register(ctx) };

        let script = c"var fired = 0; \
setTimeout(function () { fired += 1; }, 1); \
var iv = setInterval(function () { fired += 10; clearInterval(iv); }, 5); \
var slept = false; \
sofuu.sleep(3).then(function () { slept = true; });";
        // SAFETY: script is a valid C string; global eval.
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<m2-timer>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "timer setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: all timers fire, the interval is cleared from its own
        // callback, and the sleep promise resolves.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        // SAFETY: global props are live; freed below.
        let global = unsafe { qjs::global_object(ctp) };
        let fired = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"fired".as_ptr()) };
        let mut fired_i: i32 = 0;
        assert!(unsafe { qjs::to_int32(ctp, fired, &mut fired_i) });
        unsafe { qjs::sofuu_js_free_value(ctx, fired) };
        assert_eq!(fired_i, 11, "setTimeout (1) + one interval tick (10), then cleared");

        let slept = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"slept".as_ptr()) };
        let mut slept_b: c_int = 0;
        // JS_ToBool via the bindings:
        assert!(unsafe { qjs::JS_ToBool(ctx, slept) } == 1, "sleep chain must resolve");
        let _ = slept_b;
        unsafe { qjs::sofuu_js_free_value(ctx, slept) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    #[test]
    fn test_scheduler_yield_immediate_and_microtasks() {
        let _loop_guard = crate::rt::test_loop_lock();
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { sofuu_rt_install_rejection_tracker(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        unsafe { mod_timer_register(ctx) };

        let script = c"var yieldPassed = false; \
var sofuuYieldPassed = false; \
var immediateFired = false; \
var microtaskFired = false; \
setImmediate(function() { immediateFired = true; }); \
queueMicrotask(function() { microtaskFired = true; }); \
(async function() { \
  await scheduler.yield(); \
  yieldPassed = true; \
  await sofuu.yield(); \
  sofuuYieldPassed = true; \
})();";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<scheduler-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "eval must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(10))) };

        let global = unsafe { qjs::global_object(ctp) };

        let yp = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"yieldPassed".as_ptr()) };
        assert_eq!(unsafe { qjs::JS_ToBool(ctx, yp) }, 1, "scheduler.yield() must resolve");
        unsafe { qjs::sofuu_js_free_value(ctx, yp) };

        let syp = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sofuuYieldPassed".as_ptr()) };
        assert_eq!(unsafe { qjs::JS_ToBool(ctx, syp) }, 1, "sofuu.yield() must resolve");
        unsafe { qjs::sofuu_js_free_value(ctx, syp) };

        let im = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"immediateFired".as_ptr()) };
        assert_eq!(unsafe { qjs::JS_ToBool(ctx, im) }, 1, "setImmediate must fire");
        unsafe { qjs::sofuu_js_free_value(ctx, im) };

        let mt = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"microtaskFired".as_ptr()) };
        assert_eq!(unsafe { qjs::JS_ToBool(ctx, mt) }, 1, "queueMicrotask must fire");
        unsafe { qjs::sofuu_js_free_value(ctx, mt) };

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }
}
