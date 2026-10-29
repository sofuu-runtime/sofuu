// rt/deadline.rs — a JS-callable interrupt deadline for synchronous code.
//
//   sofuu.host.armDeadline(ms)   absolute deadline = now + ms, process-global
//   sofuu.host.disarmDeadline()  drop this arm; the last one clears it
//
// Why: a plugin/program evaluated by 'sofuu run' / chat / serve / eval runs on
// the engine context, where nothing armed a QuickJS interrupt handler — a
// pure-CPU plugin could never be stopped. sandbox.rs has its own meter, but
// only for sandbox runs (private JSRuntime); this module arms the ENGINE
// context, which is the one every user program shares.
//
// Contract:
//   * One absolute deadline in a process-global atomic (us since the epoch),
//     read by the interrupt handler on every poll — no allocation and no lock
//     on the handler path.
//   * Nested arms keep the EARLIEST deadline of all active arms plus a count.
//     disarmDeadline() drops the newest arm and only at zero clears the
//     deadline, so one call's disarm can never cancel another call's earlier
//     one.
//   * When the deadline fires the handler returns 1 (QuickJS raises the
//     ordinary, catchable InternalError 'interrupted') and the deadline STAYS
//     armed: a caller that swallows the error cannot make progress.
//   * The sandbox runtime is private, so no pre-existing handler can be
//     clobbered. The engine's previous poll COUNTER is saved and restored on
//     the final disarm, and the handler is cleared there.

use std::ffi::{c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sofuu_ffi::qjs::{self, JSContext, JSRuntime, JSValue, JSValueConst};

/// QuickJS runs the handler after this many bytecode ops (the sandbox uses the
/// same batch, so a deadline is felt at the same granularity in both).
const POLL_BATCH: c_int = 10_000;
/// Nested arms tracked explicitly; deeper nesting folds into the nearest.
const MAX_ARMS: usize = 16;
/// The 'no deadline' sentinel: no arm is 0 us since the epoch.
const DISARMED: u64 = 0;
/// A day is longer than any plausible synchronous plugin call.
const MAX_ARM_MS: f64 = 86_400_000.0;

/// Earliest active deadline in us since the Unix epoch; DISARMED = none.
/// Written under BOOK by arm/disarm only; read (relaxed) by the handler.
static EARLIEST_US: AtomicU64 = AtomicU64::new(DISARMED);
/// Number of active arms. Only 0->1 installs the handler, 1->0 removes it.
static ARMS: AtomicI32 = AtomicI32::new(0);

struct Book {
    deadlines: [u64; MAX_ARMS],
    /// The budget (ms) each arm asked for — reported back as "after <ms> ms".
    budget_ms: [u64; MAX_ARMS],
    n: usize,
    /// The engine context's poll counter before the first arm (restored on
    /// the final disarm, so nothing that predates us is left re-tuned).
    prev_counter: c_int,
}

static BOOK: Mutex<Book> = Mutex::new(Book {
    deadlines: [0; MAX_ARMS],
    budget_ms: [0; MAX_ARMS],
    n: 0,
    prev_counter: 0,
});

/// Budget (ms) of the arm that owns EARLIEST_US, published by publish().
static EARLIEST_MS: AtomicU64 = AtomicU64::new(0);
/// Epoch-us of the last time the handler actually aborted (0 = never / reset).
/// QuickJS turns the abort into an UNCATCHEABLE InternalError, so this is the
/// only record that the error a host is holding came from OUR deadline.
static FIRED_US: AtomicU64 = AtomicU64::new(0);
/// The budget (ms) of the arm that fired — copied from EARLIEST_MS on firing.
static FIRED_MS: AtomicU64 = AtomicU64::new(0);

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Recompute and publish the earliest active deadline. Caller holds the lock.
fn publish(book: &Book) {
    let mut min = DISARMED;
    let mut ms = 0;
    let mut i = 0;
    while i < book.n {
        let d = book.deadlines[i];
        if d != DISARMED && (min == DISARMED || d < min) {
            min = d;
            ms = book.budget_ms[i];
        }
        i += 1;
    }
    EARLIEST_US.store(min, Ordering::SeqCst);
    EARLIEST_MS.store(ms, Ordering::SeqCst);
}

/// The QuickJS poll hook. 1 aborts the running script with the ordinary
/// 'InternalError: interrupted'; the deadline is deliberately left in place.
unsafe extern "C" fn deadline_interrupt_cb(_rt: *mut JSRuntime, _opaque: *mut c_void) -> c_int {
    let deadline = EARLIEST_US.load(Ordering::Relaxed);
    if deadline == DISARMED {
        return 0;
    }
    if now_us() >= deadline {
        /* Record the firing BEFORE returning: the exception QuickJS raises is
         * uncatchable, so nothing else can tell the host this was a deadline. */
        FIRED_US.store(now_us(), Ordering::SeqCst);
        FIRED_MS.store(EARLIEST_MS.load(Ordering::Relaxed), Ordering::SeqCst);
        return 1;
    }
    0
}

/// Rewrite a top-level error message when OUR deadline aborted the call.
///
/// QuickJS raises the interrupt as an InternalError marked UNCATCHEABLE
/// (deps/quickjs/quickjs.c __js_poll_interrupts -> JS_SetUncatchableError), so
/// no JS — not even a finally — can turn it into E_TIMEOUT. The host does it
/// here instead, and ONLY for a message carrying "interrupted" while the
/// handler has actually fired: unrelated InternalErrors are left alone.
pub fn classify_interrupted(msg: &str) -> Option<String> {
    if FIRED_US.load(Ordering::Relaxed) == 0 || !msg.to_ascii_lowercase().contains("interrupted") {
        return None;
    }
    Some(format!(
        "E_TIMEOUT: plugin deadline exceeded — the call was interrupted after {} ms \
         (a synchronous call cannot be cancelled; it was aborted)",
        FIRED_MS.load(Ordering::Relaxed)
    ))
}

/// armDeadline(ms) -> undefined. Throws a TypeError for a non-positive ms.
unsafe extern "C" fn js_arm_deadline(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut ms: f64 = 0.0;
    if argc < 1 || unsafe { qjs::JS_ToFloat64(ctx, &mut ms, *argv) } != 0 || !(ms > 0.0) {
        return unsafe {
            qjs::JS_ThrowTypeError(ctx, c"armDeadline(ms): a positive number is required".as_ptr())
        };
    }
    let deadline = now_us().saturating_add((ms.min(MAX_ARM_MS) as u64).saturating_mul(1000));

    let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    /* A fired arm can never be disarmed: QuickJS marks the interrupt error
     * UNCATCHEABLE (deps/quickjs/quickjs.c __js_poll_interrupts), so no
     * try/finally runs after it. Recovery for the NEXT turn is therefore the
     * host's job — a new arm supersedes a stale fired one instead of
     * inheriting its expired minimum forever (a bricked chat/serve). */
    {
        let d = EARLIEST_US.load(Ordering::Relaxed);
        if d != DISARMED && now_us() >= d {
            book.n = 0;
            ARMS.store(0, Ordering::SeqCst);
            FIRED_US.store(0, Ordering::SeqCst);
        }
    }
    let budget = ms.min(MAX_ARM_MS) as u64;
    if book.n < MAX_ARMS {
        let n = book.n;
        book.deadlines[n] = deadline;
        book.budget_ms[n] = budget;
        book.n = n + 1;
    } else {
        /* Full: keep the earliest by overwriting the loosest arm, so a deep
         * nest cannot drop the deadline that matters most. */
        let mut loosest = 0;
        let mut i = 1;
        while i < MAX_ARMS {
            if book.deadlines[i] > book.deadlines[loosest] {
                loosest = i;
            }
            i += 1;
        }
        if deadline < book.deadlines[loosest] {
            book.deadlines[loosest] = deadline;
            book.budget_ms[loosest] = budget;
        }
    }
    if ARMS.fetch_add(1, Ordering::SeqCst) == 0 {
        /* First arm: remember the engine's poll counter, install the hook and
         * make it run often enough for a deadline to be felt. */
        book.prev_counter = unsafe { qjs::JS_GetInterruptCounter(ctx) };
        let rt = unsafe { qjs::JS_GetRuntime(ctx) };
        if !rt.is_null() {
            unsafe { qjs::JS_SetInterruptHandler(rt, Some(deadline_interrupt_cb), std::ptr::null_mut()) };
        }
        unsafe { qjs::JS_SetInterruptCounter(ctx, POLL_BATCH) };
    }
    publish(&book);
    unsafe { qjs::sofuu_js_undefined() }
}

/// disarmDeadline() -> undefined. Drops the newest arm; at zero it clears the
/// deadline, removes the hook and restores the pre-arm poll counter.
unsafe extern "C" fn js_disarm_deadline(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    if book.n > 0 {
        let n = book.n - 1;
        book.n = n;
        book.deadlines[n] = DISARMED;
        book.budget_ms[n] = 0;
    }
    publish(&book);
    if ARMS.fetch_sub(1, Ordering::SeqCst) <= 1 {
        /* Last arm gone: nothing may interrupt this runtime any more. */
        FIRED_US.store(0, Ordering::SeqCst);
        let rt = unsafe { qjs::JS_GetRuntime(ctx) };
        if !rt.is_null() {
            unsafe { qjs::JS_SetInterruptHandler(rt, None, std::ptr::null_mut()) };
        }
        if book.prev_counter > 0 {
            unsafe { qjs::JS_SetInterruptCounter(ctx, book.prev_counter) };
        }
    }
    unsafe { qjs::sofuu_js_undefined() }
}

/// Register sofuu.host.armDeadline / sofuu.host.disarmDeadline, creating the
/// sofuu.host namespace when it does not exist yet.
///
/// # Safety
/// ctx must be the live engine context (called once at boot).
pub unsafe fn mod_deadline_register(ctx: *mut JSContext) {
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let mut sofuu_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr()) };
    if unsafe { qjs::is_undefined(sofuu_obj) } {
        sofuu_obj = unsafe { qjs::sofuu_js_new_object(ctx) };
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sofuu".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sofuu_obj),
            )
        };
    }

    let mut host = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"host".as_ptr()) };
    if !unsafe { qjs::is_object(host) } {
        unsafe { qjs::sofuu_js_free_value(ctx, host) };
        host = unsafe { qjs::sofuu_js_new_object(ctx) };
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                sofuu_obj,
                c"host".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, host),
            )
        };
    }

    let arm = unsafe { qjs::sofuu_js_new_cfunction(ctx, js_arm_deadline, c"armDeadline".as_ptr(), 1) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, host, c"armDeadline".as_ptr(), arm) };
    let dis = unsafe { qjs::sofuu_js_new_cfunction(ctx, js_disarm_deadline, c"disarmDeadline".as_ptr(), 0) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, host, c"disarmDeadline".as_ptr(), dis) };

    unsafe { qjs::sofuu_js_free_value(ctx, host) };
    unsafe { qjs::sofuu_js_free_value(ctx, sofuu_obj) };
    unsafe { qjs::sofuu_js_free_value(ctx, global) };
}
