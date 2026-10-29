// rt/promise.rs — QuickJS Promise ↔ libuv async bridge + deferred
// unhandled-rejection tracker (M1).
//
// Port of the retired `src/io/promises.c`, plus the rejection tracker that
// used to live in engine.c — its semantics move here verbatim (plan §2 M1):
// QuickJS reports a rejection with is_handled=false the moment it happens —
// even when a handler is attached an instant later (e.g. `await r.json()`).
// So pending rejections are remembered here, dropped on the is_handled=true
// retraction, and printed only at the end of a microtask drain.
//
// C symbols replaced: sofuu_promise_new / sofuu_promise_resolve /
// sofuu_promise_reject / sofuu_promise_reject_str / sofuu_flush_jobs
// (declared in src/io/promises.h — the remaining C modules like fs.c,
// timer.c, client.c, mcp.c keep calling them unchanged). The tracker is
// installed by engine.c under SOFUU_RUST_CORE via
// sofuu_rt_install_rejection_tracker / sofuu_rt_report_pending_rejections.
//
// Memory contract (unchanged from promises.h): the handle is malloc'd in
// sofuu_promise_new and freed by sofuu_promise_resolve/reject; resolve dups
// `value` internally so the CALLER still owns it; reject TAKES OWNERSHIP of
// `error`.

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem::{size_of, zeroed};
use std::ptr;

use sofuu_ffi::qjs::{self, JSContext, JSRuntime, JSValue, JSValueConst};

extern "C" {
    /// mod_process.c — dispatch to process.on('uncaughtException') if
    /// registered. Returns 1 when a handler ran (runtime keeps running), 0
    /// when nothing handled it. Does NOT take ownership of `exc`. Stays C
    /// until M3.
    fn process_dispatch_uncaught(ctx: *mut JSContext, exc: JSValue) -> c_int;
}

/// The C `sofuu_promise_t` — { ctx, resolve, reject } (promises.h). Callers
/// store the pointer as libuv req->data and never touch it directly.
#[repr(C)]
pub struct PromiseHandle {
    pub ctx: *mut JSContext,
    pub resolve: JSValue,
    pub reject: JSValue,
    pub async_context: JSValue,
}

/// # Safety
/// `ctx` must be a live context; `out` a valid pointer, written on success.
#[no_mangle]
pub unsafe extern "C" fn sofuu_promise_new(ctx: *mut JSContext, out: *mut *mut PromiseHandle) -> JSValue {
    let mut resolvers: [JSValue; 2] = [zeroed(); 2];
    // SAFETY: resolvers is a 2-element JSValue array that JS_NewPromiseCapability fills.
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    if qjs::is_exception(promise) {
        return promise; /* capability allocation failed */
    }

    let p = libc::malloc(size_of::<PromiseHandle>()) as *mut PromiseHandle;
    if p.is_null() {
        // OOM — no NULL deref, no leaked resolver refs.
        // SAFETY: resolvers hold live refs from JS_NewPromiseCapability.
        qjs::sofuu_js_free_value(ctx, resolvers[0]);
        qjs::sofuu_js_free_value(ctx, resolvers[1]);
        return qjs::sofuu_js_exception();
    }
    let rt = qjs::JS_GetRuntime(ctx);
    let async_ctx = qjs::JS_GetAsyncContext(rt);
    (*p).ctx = ctx;
    (*p).resolve = resolvers[0];
    (*p).reject = resolvers[1];
    (*p).async_context = async_ctx;

    *out = p;
    promise
}

/// # Safety
/// `p` must be a live handle from sofuu_promise_new (or NULL for a no-op);
/// `value` a live value of p's context (ownership stays with the caller).
#[no_mangle]
pub unsafe extern "C" fn sofuu_promise_resolve(p: *mut PromiseHandle, value: JSValue) {
    if p.is_null() {
        return; /* already settled & freed — caller must null its handle */
    }
    let ctx = (*p).ctx;
    let rt = qjs::JS_GetRuntime(ctx);
    let prev_ctx = qjs::JS_GetAsyncContext(rt);
    qjs::JS_SetAsyncContext(rt, (*p).async_context);
    // SAFETY: p->resolve is a live resolver function; JS_UNDEFINED this_obj.
    let ret = qjs::JS_Call(ctx, (*p).resolve, qjs::sofuu_js_undefined(), 1, &value);
    qjs::JS_SetAsyncContext(rt, prev_ctx);
    qjs::sofuu_js_free_value(ctx, prev_ctx);
    if qjs::is_exception(ret) {
        /* Log but don't crash — the promise was already settled or handler threw */
        // SAFETY: exception is pending on ctx; all values live.
        let exc = qjs::sofuu_js_get_exception(ctx);
        let msg_v = qjs::JS_ToString(ctx, exc);
        let s = qjs::sofuu_js_to_cstring(ctx, msg_v);
        let msg = if s.is_null() {
            "?".to_string()
        } else {
            // SAFETY: s valid until freed below.
            CStr::from_ptr(s).to_string_lossy().into_owned()
        };
        eprintln!("[sofuu] Promise resolve error: {}", msg);
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
        qjs::sofuu_js_free_value(ctx, msg_v);
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, (*p).resolve);
    qjs::sofuu_js_free_value(ctx, (*p).reject);
    qjs::sofuu_js_free_value(ctx, (*p).async_context);
    libc::free(p as *mut c_void);
}

/// # Safety
/// `p` must be a live handle from sofuu_promise_new (or NULL — a no-op that
/// leaks `error`, same as C); `error` ownership transfers in (freed here).
#[no_mangle]
pub unsafe extern "C" fn sofuu_promise_reject(p: *mut PromiseHandle, error: JSValue) {
    if p.is_null() {
        return; /* already settled & freed — caller must null its handle */
    }
    let ctx = (*p).ctx;
    let rt = qjs::JS_GetRuntime(ctx);
    let prev_ctx = qjs::JS_GetAsyncContext(rt);
    qjs::JS_SetAsyncContext(rt, (*p).async_context);
    // SAFETY: p->reject is a live resolver function; JS_UNDEFINED this_obj.
    let ret = qjs::JS_Call(ctx, (*p).reject, qjs::sofuu_js_undefined(), 1, &error);
    qjs::JS_SetAsyncContext(rt, prev_ctx);
    qjs::sofuu_js_free_value(ctx, prev_ctx);
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, error);
    qjs::sofuu_js_free_value(ctx, (*p).resolve);
    qjs::sofuu_js_free_value(ctx, (*p).reject);
    qjs::sofuu_js_free_value(ctx, (*p).async_context);
    libc::free(p as *mut c_void);
}

/// Convenience: reject with a plain C string message.
///
/// # Safety
/// `p` live handle or NULL; `msg` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn sofuu_promise_reject_str(p: *mut PromiseHandle, msg: *const c_char) {
    if p.is_null() {
        return;
    }
    let ctx = (*p).ctx;
    // SAFETY: err is a live fresh Error; JS_SetPropertyStr transfers the
    // message ref (do NOT free it afterward).
    let err = qjs::JS_NewError(ctx);
    let mstr = qjs::sofuu_js_new_string(ctx, if msg.is_null() { c"".as_ptr() } else { msg });
    qjs::sofuu_js_set_property_str(ctx, err, c"message".as_ptr(), mstr);
    // Ownership of `err` transfers into the reject path.
    sofuu_promise_reject(p, err);
}

/// Pump all pending QuickJS microtasks/jobs — bounded by max_jobs to prevent event-loop starvation.
/// Returns true if all jobs in the queue were drained, false if execution was interrupted by the bound.
#[no_mangle]
pub unsafe extern "C" fn sofuu_flush_jobs_bounded(ctx: *mut JSContext, max_jobs: usize) -> bool {
    let rt = qjs::JS_GetRuntime(ctx);
    let mut ctx2: *mut JSContext = ptr::null_mut();
    let mut executed = 0usize;

    loop {
        if executed >= max_jobs {
            /* R-14: this bound is now hit routinely (every timer callback
             * drains at most MICROTASK_BATCH jobs), so pending rejections are
             * NOT reported here. A rejection deferred at job N can still be
             * handled at job N+K of the SAME drain; reporting at the bound
             * would call a handled rejection "unhandled" and set the C-01
             * exit flag for a script that never left one unhandled. The
             * report happens only when the queue is genuinely empty (below)
             * or at shutdown. */
            return false;
        }

        let err = qjs::JS_ExecutePendingJob(rt, &mut ctx2);
        if err < 0 {
            /* A job that threw. QuickJS converts a handler throw into a
             * rejected derived promise, so what reaches here is either an
             * allocation failure (the JS heap ceiling — the job queue can no
             * longer grow, and a continuation that cannot be enqueued is
             * dropped inside quickjs.c) or an exception escaping a
             * continuation. Both are reported ONCE, under their real name,
             * and remembered for the exit code (C-01); the heap case is also
             * flagged fatal so the loop stops (C-03). */
            let exc = qjs::sofuu_js_get_exception(ctx2);
            if is_oom_reason(ctx2, exc) {
                report_microtask_overflow(ctx2, executed);
            } else if process_dispatch_uncaught(ctx2, exc) == 0 {
                /* Not handled by JS — the same one-shot "UncaughtException"
                 * formatting the engine uses for a synchronous throw. */
                crate::rt::engine::report_uncaught(ctx2, exc);
            }
            qjs::sofuu_js_free_value(ctx2, exc);
            break;
        }
        if err == 0 {
            break;
        }
        executed += 1;
    }
    sofuu_rt_report_pending_rejections(ctx);
    true
}

/// Pump all pending QuickJS microtasks/jobs — call after each libuv callback.
///
/// # Safety
/// `ctx` must be the live runtime context (loop thread).
#[no_mangle]
pub unsafe extern "C" fn sofuu_flush_jobs(ctx: *mut JSContext) {
    sofuu_flush_jobs_bounded(ctx, usize::MAX);
}

// ── Deferred unhandled-rejection tracker (moved verbatim from engine.c) ──

struct PendingRej {
    promise: JSValue, /* dup'd */
    reason: JSValue,  /* dup'd */
    reported: bool,
}

// Tracker state — keyed to the loop thread. The runtime is single-threaded;
// every callback (tracker, flush, shutdown report) runs on the same thread
// the C statics this replaces ran on, so a TLS slot is exact. (A `static`
// would need Sync, and the Vec is !Send because JSValue carries raw
// pointers.)
thread_local! {
    static PENDING: RefCell<Vec<PendingRej>> = const { RefCell::new(Vec::new()) };
}

/// A pathological flood with no drain prints the oldest eagerly (C: 64).
const EAGER_CAP: usize = 64;

/// Read a live JS value as a Rust string (borrowed value; nothing is freed).
///
/// # Safety
/// `ctx` live; `val` a value of `ctx`.
unsafe fn value_str(ctx: *mut JSContext, val: JSValueConst) -> Option<String> {
    let s = qjs::sofuu_js_to_cstring(ctx, val);
    if s.is_null() {
        return None;
    }
    // SAFETY: s valid until freed below.
    let out = CStr::from_ptr(s).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, s);
    Some(out)
}

/// True when `reason` is QuickJS's out-of-memory sentinel rather than a real
/// script error.
///
/// QuickJS throws JS_NULL when it cannot even allocate the Error object
/// (`quickjs.c` `JS_ThrowError2`: "out of memory: throw JS_NULL to avoid
/// recursing"), and that is the only value the ENGINE produces with no
/// name/message/stack. A literal `throw null` is indistinguishable from it —
/// the substrate ceiling is the overwhelmingly likely cause on this path, and
/// the message says so. (`undefined` is NOT the sentinel: `Promise.reject()`
/// is a normal rejection with an undefined reason and must keep printing as
/// such.) A real `InternalError: out of memory` — where allocating the error
/// object itself succeeded — is matched too.
///
/// # Safety
/// `ctx` live; `reason` a value of `ctx` (borrowed).
pub unsafe fn is_oom_reason(ctx: *mut JSContext, reason: JSValueConst) -> bool {
    if qjs::is_null(reason) {
        return true;
    }
    if qjs::is_undefined(reason) {
        return false;
    }
    let name = qjs::sofuu_js_get_property_str(ctx, reason, c"name".as_ptr());
    let msg = qjs::sofuu_js_get_property_str(ctx, reason, c"message".as_ptr());
    let name_s = value_str(ctx, name).unwrap_or_default();
    let msg_s = value_str(ctx, msg).unwrap_or_default();
    qjs::sofuu_js_free_value(ctx, name);
    qjs::sofuu_js_free_value(ctx, msg);
    name_s == "InternalError" && msg_s.to_ascii_lowercase().contains("out of memory")
}

/// Report the C-03 substrate failure LOUDLY and once: a real, catchable
/// RangeError with a message, instead of QuickJS's bare `null` (no
/// name/message/stack) and a silent exit 0.
///
/// A registered `process.on('uncaughtException')` handler gets the same
/// first shot every other fatal path gets; the failure is flagged fatal
/// regardless, because the continuations QuickJS could not enqueue are gone
/// — the runtime must not pretend the turn completed.
///
/// # Safety
/// `ctx` live (loop thread).
pub unsafe fn report_microtask_overflow(ctx: *mut JSContext, executed: usize) {
    let limit_mib = crate::rt::engine::js_heap_limit() / (1024 * 1024);
    let msg = format!(
        "Maximum microtask queue depth exceeded: the JavaScript heap hit its {} MiB ceiling \
         after {} continuation(s). QuickJS drops a promise continuation it cannot enqueue, so \
         the runtime reports this instead of silently losing work — raise SOFUU_JS_HEAP_MB or \
         reduce the fan-out width.",
        limit_mib, executed
    );
    // SAFETY: err is an owned error object, freed below.
    let err = crate::rt::engine::new_range_error(ctx, &msg);
    if process_dispatch_uncaught(ctx, err) == 0 {
        eprintln!("\x1b[31m[sofuu] UncaughtException:\x1b[0m {}", msg);
    }
    crate::rt::engine::note_fatal(ctx);
    qjs::sofuu_js_free_value(ctx, err);
}

/// Retract every deferred rejection whose reason is `reason` (pointer
/// identity). A synchronous module throw rejects BOTH the module promise and
/// the async module function's internal promise with the same error object;
/// the engine reports the throw once as `UncaughtException`, so the tracker
/// must not print those copies again.
///
/// # Safety
/// `ctx` live; `reason` a value of `ctx` (borrowed).
/// Internal helper (engine.rs is its only caller): deliberately NOT
/// `#[no_mangle]` and not `extern "C"`. As a `sofuu_rt_`-prefixed dynamic
/// export it entered the FROZEN v2 ABI surface that scripts/check_abi.sh
/// guards, so `make abi-check` failed with an "ABI ADDITION without
/// acknowledgment" for a symbol no host was ever meant to call.
pub unsafe fn sofuu_rt_suppress_rejections(ctx: *mut JSContext, reason: JSValueConst) {
    let ptr = qjs::sofuu_js_value_get_ptr(reason);
    PENDING.with(|p| {
        let mut pending = p.borrow_mut();
        let mut i = 0;
        while i < pending.len() {
            if qjs::sofuu_js_value_get_ptr(pending[i].reason) == ptr {
                let old = pending.remove(i);
                qjs::sofuu_js_free_value(ctx, old.promise);
                qjs::sofuu_js_free_value(ctx, old.reason);
            } else {
                i += 1;
            }
        }
    });
}

/// # Safety
/// `ctx` live; `reason` a live value of `ctx`.
unsafe fn print_rejection(ctx: *mut JSContext, reason: JSValueConst) {
    // An OOM is not a script-level rejection: name the real cause (C-03).
    if is_oom_reason(ctx, reason) {
        report_microtask_overflow(ctx, 0);
        return;
    }
    // SAFETY: str_v is live; freed below.
    let str_v = qjs::JS_ToString(ctx, reason);
    let msg = qjs::sofuu_js_to_cstring(ctx, str_v);
    let msg_s = if msg.is_null() {
        "(unknown)".to_string()
    } else {
        // SAFETY: msg valid until freed below.
        CStr::from_ptr(msg).to_string_lossy().into_owned()
    };
    eprintln!(
        "{}",
        crate::rt::engine::paint(
            "31",
            &format!("[sofuu] UnhandledPromiseRejection: {}", msg_s)
        )
    );
    if !msg.is_null() {
        qjs::sofuu_js_free_cstring(ctx, msg);
    }
    let stack = qjs::sofuu_js_get_property_str(ctx, reason, c"stack".as_ptr());
    if !qjs::is_undefined(stack) && !qjs::is_null(stack) {
        // SAFETY: ss valid until freed below.
        let ss = qjs::sofuu_js_to_cstring(ctx, stack);
        if !ss.is_null() {
            let st = CStr::from_ptr(ss).to_string_lossy();
            eprintln!("{}", st);
            qjs::sofuu_js_free_cstring(ctx, ss);
        }
    }
    qjs::sofuu_js_free_value(ctx, stack);
    qjs::sofuu_js_free_value(ctx, str_v);
    // C-01: an unhandled rejection is a fatal script error for the process
    // status, even though the event loop keeps running to idle.
    crate::rt::engine::note_uncaught_error(ctx);
}

/// Host promise rejection tracker (JS_SetHostPromiseRejectionTracker).
///
/// # Safety
/// Callback invoked by QuickJS on the loop thread with live values.
unsafe extern "C" fn rejection_tracker(
    ctx: *mut JSContext,
    promise: JSValueConst,
    reason: JSValueConst,
    is_handled: c_int,
    _opaque: *mut c_void,
) {
    if is_handled != 0 {
        /* Handler attached after the fact → retract. Promises are objects;
         * identity by underlying pointer is exact. */
        let ptr = qjs::sofuu_js_value_get_ptr(promise);
        PENDING.with(|p| {
            let mut pending = p.borrow_mut();
            if let Some(i) = pending
                .iter()
                .position(|r| qjs::sofuu_js_value_get_ptr(r.promise) == ptr)
            {
                let old = pending.remove(i);
                qjs::sofuu_js_free_value(ctx, old.promise);
                qjs::sofuu_js_free_value(ctx, old.reason);
            }
        });
        return;
    }

    /* Defer: may be retracted by a later is_handled=true event. A
     * pathological flood with no drain prints the oldest eagerly. The
     * eviction is removed under the borrow and PRINTED outside it — printing
     * runs JS (ToString/stack, an uncaughtException handler) and re-entering
     * the promise machinery while PENDING is borrowed would panic. */
    let evicted = PENDING.with(|p| {
        let mut pending = p.borrow_mut();
        let evicted = if pending.len() >= EAGER_CAP {
            Some(pending.remove(0))
        } else {
            None
        };
        /* One entry per promise: QuickJS reports each rejection once, but a
         * duplicate push (e.g. a promise re-reported by an embedded host)
         * would print the same error twice. */
        let ptr = qjs::sofuu_js_value_get_ptr(promise);
        if !pending
            .iter()
            .any(|r| qjs::sofuu_js_value_get_ptr(r.promise) == ptr)
        {
            // SAFETY: dup'd refs owned by the pending entry (freed on report).
            pending.push(PendingRej {
                promise: qjs::sofuu_js_dup_value(ctx, promise),
                reason: qjs::sofuu_js_dup_value(ctx, reason),
                reported: false,
            });
        }
        evicted
    });
    if let Some(old) = evicted {
        print_rejection(ctx, old.reason);
        qjs::sofuu_js_free_value(ctx, old.promise);
        qjs::sofuu_js_free_value(ctx, old.reason);
    }
}

/// Install the deferred-rejection tracker on a runtime. Replaces the C
/// `JS_SetHostPromiseRejectionTracker(rt, rejection_tracker, NULL)` under
/// SOFUU_RUST_CORE (engine.c).
///
/// # Safety
/// `rt` must be the live engine runtime.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_install_rejection_tracker(rt: *mut JSRuntime) {
    qjs::JS_SetHostPromiseRejectionTracker(rt, Some(rejection_tracker), ptr::null_mut());
}

/// Print-and-clear every rejection that was never handled. Called after
/// each completed microtask drain (end of sofuu_flush_jobs) and at shutdown
/// (engine_destroy under SOFUU_RUST_CORE).
///
/// The queue is drained under the PENDING borrow and PRINTED outside it:
/// printing runs JS (ToString / `stack` / a user uncaughtException handler),
/// which can re-enter the promise machinery and would otherwise panic on a
/// second `borrow_mut` of the same RefCell.
///
/// # Safety
/// `ctx` must be the live runtime context.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_report_pending_rejections(ctx: *mut JSContext) {
    let drained: Vec<JSValue> = PENDING.with(|p| {
        let mut pending = p.borrow_mut();
        let mut reasons = Vec::new();
        for r in pending.drain(..) {
            if !r.reported {
                reasons.push(r.reason);
            } else {
                qjs::sofuu_js_free_value(ctx, r.reason);
            }
            qjs::sofuu_js_free_value(ctx, r.promise);
        }
        reasons
    });
    for reason in drained {
        print_rejection(ctx, reason);
        qjs::sofuu_js_free_value(ctx, reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::{self, CtxPtr, JSCFunction, JSValueConst};
    use sofuu_ffi::uv::{self, UvTimer};

    /// libuv handles start with `data` — the only field we touch.
    #[repr(C)]
    struct TimerHandle {
        data: *mut c_void,
    }

    /// M1 proof (plan §2 M1): a promise created in Rust, resolved from a
    /// uv timer callback, must fire a JS `.then` through the interleaved
    /// loop — the whole promise↔loop bridge working end to end.
    #[test]
    fn m1_promise_resolved_from_uv_timer_fires_js_then() {
        // The process-global uv loop is shared — serialize loop-driving
        // tests (see rt/mod.rs TEST_LOOP_LOCK).
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 test pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        // SAFETY: fresh valid context on this thread.
        let ctp = unsafe { CtxPtr::new(ctx) };

        // SAFETY: installs the M1 tracker so rejections defer, not dump.
        unsafe { sofuu_rt_install_rejection_tracker(rt) };
        // SAFETY: boots the global loop (test-only process lifetime).
        unsafe { crate::rt::event_loop::sofuu_loop_init() };

        // Register makePromise() — creates a Rust promise + a 1ms uv timer.
        // SAFETY: make_promise_fn is 'static; property takes the func ref.
        let func = unsafe { qjs::new_cfunction(ctp, "makePromise", make_promise_fn, 0) };
        let global = unsafe { qjs::global_object(ctp) };
        unsafe { qjs::sofuu_js_set_property_str(ctx, global, c"makePromise".as_ptr(), func) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        let decl = c"var resolvedValue = -1;";
        // SAFETY: decl is a valid C string; global eval (byte length from
        // the CStr itself — never hand-count).
        let r1 = unsafe {
            qjs::JS_Eval(
                ctx,
                decl.as_ptr(),
                decl.to_bytes().len(),
                c"<m1-decl>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r1) }, "decl must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r1) };

        let src = c"makePromise().then(function (v) { resolvedValue = v; });";
        let r2 = unsafe {
            qjs::JS_Eval(
                ctx,
                src.as_ptr(),
                src.to_bytes().len(),
                c"<m1-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r2) }, "makePromise().then must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r2) };

        // Drain: timer fires → Rust resolve → JS .then runs.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        // Read the resolved value back through the bindings.
        // SAFETY: global object + property are live; freed below.
        let global = unsafe { qjs::global_object(ctp) };
        let v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"resolvedValue".as_ptr()) };
        let mut out: i32 = -1;
        assert!(
            unsafe { qjs::to_int32(ctp, v, &mut out) },
            "resolvedValue must be a number"
        );
        assert_eq!(out, 42, "JS .then must see the Rust-resolved value");
        unsafe { qjs::sofuu_js_free_value(ctx, v) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // SAFETY: closes the (already fired) timer handle before teardown.
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        // SAFETY: teardown of the context/runtime we created — AFTER the
        // loop is closed and no values remain rooted.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// The native `makePromise()`: creates a waitable promise and arms a
    /// 1ms uv timer whose callback resolves it with 42.
    unsafe extern "C" fn make_promise_fn(
        ctx: *mut JSContext,
        _this: JSValueConst,
        _argc: c_int,
        _argv: *const JSValueConst,
    ) -> JSValue {
        // SAFETY: ctx valid; out written by sofuu_promise_new.
        let mut out: *mut PromiseHandle = ptr::null_mut();
        let prom = sofuu_promise_new(ctx, &mut out);

        // SAFETY: timer storage must be the FULL uv_timer_t size — only the
        // leading `data` field is touched, but uv_timer_init writes the rest.
        let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut TimerHandle;
        if timer.is_null() {
            return qjs::sofuu_js_exception();
        }
        // SAFETY: global loop is initialized; timer is valid storage.
        let lp = crate::rt::event_loop::sofuu_loop_get();
        let rc = uv::uv_timer_init(lp, timer as *mut UvTimer);
        if rc == 0 {
            (*timer).data = out as *mut c_void;
            uv::uv_timer_start(timer as *mut UvTimer, Some(timer_done_cb), 1, 0);
        } else {
            libc::free(timer as *mut c_void);
        }
        prom
    }

    /// Timer callback: resolve the promise stored in handle->data with 42,
    /// then close the handle — the storage is released by the close
    /// callback, never before (a freed handle still linked into the loop's
    /// handle queue corrupts the walk/closing path).
    unsafe extern "C" fn timer_done_cb(timer: *mut UvTimer) {
        // SAFETY: uv handle layout — data is the first field.
        let th = timer as *mut TimerHandle;
        let p = (*th).data as *mut PromiseHandle;
        let ctx = (*p).ctx;
        // SAFETY: v is created here; sofuu_promise_resolve dups it and the
        // caller still owns the original → free after.
        let v = qjs::sofuu_js_new_int32(ctx, 42);
        sofuu_promise_resolve(p, v);
        qjs::sofuu_js_free_value(ctx, v);
        // SAFETY: handle is live; close callback frees the storage.
        uv::uv_close(timer as *mut sofuu_ffi::uv::UvHandle, Some(timer_close_cb));
    }

    unsafe extern "C" fn timer_close_cb(timer: *mut sofuu_ffi::uv::UvHandle) {
        // SAFETY: the close callback runs when the handle is off the queue.
        libc::free(timer as *mut c_void);
    }
}
