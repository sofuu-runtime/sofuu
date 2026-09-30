// rt/loop.rs — global libuv event loop + QuickJS microtask interleaving (M1).
//
// Port of the retired `src/io/loop.c`, semantics verbatim: after every libuv
// I/O callback we pump QuickJS pending jobs (Promise .then chains,
// async/await continuations), by running the loop in UV_RUN_ONCE mode in a
// tight loop until BOTH the libuv loop and the JS job queue are drained.
//
// C symbols replaced: sofuu_loop_init / sofuu_loop_get / sofuu_loop_run /
// sofuu_loop_close (declared in src/io/loop.h — still included by the C
// callers; only the implementation moved here).
//
// Storage note: the loop is allocated ONCE per process and never freed.
// engine_destroy() runs sofuu_loop_close() BEFORE mod_process_cleanup(),
// which still touches uv handles (uv_timer_stop / uv_close / uv_read_stop)
// that hold a pointer to the loop — so the storage must outlive the close,
// exactly like the old C `static uv_loop_t g_loop`.

use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use sofuu_ffi::qjs::JSContext;
use sofuu_ffi::uv::{self, UvHandle, UvLoop, UV_RUN_DEFAULT, UV_RUN_NOWAIT, UV_RUN_ONCE};

use crate::rt::promise::sofuu_flush_jobs;

/// The global libuv loop (`static uv_loop_t g_loop` in the retired loop.c).
/// Sized via a C shim — uv_loop_t is opaque to Rust.
static G_LOOP: AtomicPtr<UvLoop> = AtomicPtr::new(ptr::null_mut());
/// How many engines currently hold the live loop (H1 multi-instance). The
/// loop is process-global by design, so `sofuu_loop_init` is idempotent —
/// a second engine in the same process reuses the existing loop and just
/// takes a reference — and `sofuu_loop_close` only tears down when the LAST
/// holder releases it. `refs == 0` means "not live"; the storage is still
/// retained (never freed) per the module docs, and the next init allocates
/// a fresh loop exactly like the original single-instance behavior.
static G_LOOP_REFS: AtomicUsize = AtomicUsize::new(0);

extern "C" {
    /// `sizeof(uv_loop_t)` (src/ffi_shim.c).
    fn sofuu_uv_loop_size() -> usize;
    /// mod_process.c — dispatch any SIGINT/SIGTERM that fired since the last
    /// check (process.on handlers or the exit(130/143) fallback). Stays C
    /// until M3; the retired loop.c called it after every UV_RUN_ONCE.
    fn process_dispatch_pending_signals();
}

// ── F-2 (AUDIT-2026-09-01-CLI): per-ctx armed-handle registry ──────────
//
// The loop is process-global; `sofuu_loop_close` deliberately tears it down
// only when the LAST engine releases it. That leaves a gap: destroying one
// of several live engines does NOT close that engine's handles, and its
// still-armed timers/pipes/polls carry `data` pointers into the Rust boxes
// and JSContext that engine_destroy is about to free. The next engine's
// uv_run would fire them into freed memory (use-after-free).
//
// Every module that arms a uv handle for a specific engine registers it
// here (track_handle); the handle's close callback unregisters it
// (untrack_handle). At engine teardown, sofuu_loop_shutdown_engine closes
// exactly the handles that belong to the dying ctx and drains the close
// callbacks out of the shared loop — while every other engine's handles
// stay live. Attribution is by JSContext, so two engines on ONE thread are
// attributed correctly too (capi multi-instance runs them that way).

/// `uv_close_cb` shape — the same signature the uv modules already use.
pub type UvCloseCb = unsafe extern "C" fn(*mut UvHandle);

struct ArmedHandle {
    /// The engine context the handle belongs to (attribution key).
    ctx: *mut JSContext,
    /// The armed uv handle (uv_timer_t / uv_pipe_t / uv_poll_t / …).
    handle: *mut UvHandle,
    /// The close callback the owning module wants (frees its storage/box).
    close: Option<UvCloseCb>,
}

// SAFETY: the raw pointers are only ever dereferenced on the loop thread
// (track at arm time, untrack in close callbacks, shutdown in the teardown
// path — all loop-thread), and every mutation of the registry is behind the
// ARMED mutex. The registry never "sends" a handle anywhere; it is a
// bookkeeping list for loop-thread consumers.
unsafe impl Send for ArmedHandle {}

static ARMED: Mutex<Vec<ArmedHandle>> = Mutex::new(Vec::new());

/// Register an armed uv handle against `ctx` so a later per-engine teardown
/// can close it without touching other engines' handles. Re-registering the
/// same handle pointer is a no-op (dedupe by pointer). Call right after the
/// handle is armed (init/start); the handle's close callback must call
/// untrack_handle.
///
/// # Safety
/// `handle` must be a live uv handle armed on the global loop; `ctx` must be
/// the engine context that owns it. Loop thread.
pub unsafe fn track_handle(ctx: *mut JSContext, handle: *mut UvHandle, close: Option<UvCloseCb>) {
    if handle.is_null() {
        return;
    }
    if let Ok(mut armed) = ARMED.lock() {
        if armed.iter().any(|e| e.handle == handle) {
            return;
        }
        armed.push(ArmedHandle { ctx, handle, close });
    }
}

/// Drop the registration for `handle` (called from its close callback — the
/// handle is fully closed by then). Unknown pointers are ignored.
pub fn untrack_handle(handle: *mut UvHandle) {
    if let Ok(mut armed) = ARMED.lock() {
        armed.retain(|e| e.handle != handle);
    }
}

/// Close and drain every armed handle that belongs to `ctx`, leaving all
/// other engines' handles live. Single-engine teardown is unaffected: the
/// last-engine `sofuu_loop_close` walk closes whatever is left (and
/// untracks it).
///
/// Ordering matters: close callbacks fire during the drain while `ctx` is
/// still live, so cbs that free JSValues with ctx stay valid. The ARMED lock
/// is released before uv_run (close cbs may call track/untrack).
///
/// # Safety
/// `ctx` must be the dying engine's context; must run on the loop thread
/// before JS_FreeContext/JS_FreeRuntime. No new handles for `ctx` may be
/// armed afterwards.
#[no_mangle]
pub unsafe extern "C" fn sofuu_loop_shutdown_engine(ctx: *mut JSContext) {
    if loop_ptr().is_null() {
        return;
    }
    /* Extract this engine's entries; drop the lock before uv_run — close
     * callbacks may re-enter track/untrack. */
    let mine: Vec<ArmedHandle> = match ARMED.lock() {
        Ok(mut armed) => {
            let (mine, rest): (Vec<_>, Vec<_>) = armed.drain(..).partition(|e| e.ctx == ctx);
            *armed = rest;
            mine
        }
        Err(_) => return, /* poisoned registry — last-engine walk still cleans up */
    };
    if mine.is_empty() {
        return;
    }
    for e in &mine {
        if uv::uv_is_closing(e.handle) == 0 {
            uv::uv_close(e.handle, e.close);
        }
    }
    /* Drain the shared loop so the close callbacks fire NOW (while ctx is
     * still live) instead of during another engine's later uv_run. NOWAIT
     * never blocks on the surviving engines' long timers; a handful of
     * iterations is plenty — libuv delivers pending close callbacks on the
     * very next closing phase, and none of our close cbs re-arm handles. */
    let lp = loop_ptr();
    for _ in 0..8 {
        if uv::uv_loop_alive(lp) == 0 {
            break;
        }
        uv::uv_run(lp, UV_RUN_NOWAIT);
    }
}

fn loop_ptr() -> *mut UvLoop {
    G_LOOP.load(Ordering::Relaxed)
}

/// # Safety
/// Called once at runtime boot (engine_create). Must run on the loop thread.
/// Idempotent: a second engine in the same process reuses the existing loop
/// (H1 multi-instance — each engine just takes a reference). After a full
/// close (refs dropped to 0) the next init allocates a fresh loop, exactly
/// like the original single-instance behavior.
#[no_mangle]
pub unsafe extern "C" fn sofuu_loop_init() {
    if G_LOOP_REFS.load(Ordering::Relaxed) > 0 {
        // Already live — another engine holds it; just take a reference.
        G_LOOP_REFS.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let lp = libc::malloc(sofuu_uv_loop_size()) as *mut UvLoop;
    if lp.is_null() {
        eprintln!("[sofuu] sofuu_loop_init: out of memory allocating uv loop");
        return;
    }
    // SAFETY: lp is a fresh, correctly-sized allocation for a uv_loop_t.
    let rc = uv::uv_loop_init(lp);
    if rc != 0 {
        eprintln!("[sofuu] sofuu_loop_init: uv_loop_init failed (rc={})", rc);
        libc::free(lp as *mut c_void);
        return;
    }
    G_LOOP.store(lp, Ordering::Relaxed);
    G_LOOP_REFS.store(1, Ordering::Relaxed);
}

/// # Safety
/// The loop must have been initialized (engine boot).
#[no_mangle]
pub unsafe extern "C" fn sofuu_loop_get() -> *mut UvLoop {
    loop_ptr()
}

/// Run until both are true:
///   1. No pending libuv handles/requests (UV_RUN_ONCE returns 0)
///   2. No pending QuickJS jobs
///
/// # Safety
/// `ctx` must be the live runtime context (loop thread).
#[no_mangle]
pub unsafe extern "C" fn sofuu_loop_run(ctx: *mut JSContext) {
    sofuu_loop_run_bounded(ctx, None)
}

/// [`sofuu_loop_run`] with an optional wall-clock bound.
///
/// The unbounded loop can never give up on its own: it exits only when libuv
/// reports no pending work AND no handle is still alive. A test that leaves a
/// single handle open — or a transfer that never completes — therefore spins
/// forever. That is exactly what happened on the Linux CI runners, where
/// `cargo test` sat in one http_client test until the 35-minute job timeout
/// killed it, while the same test passes in 2.8s locally.
///
/// `max` bounds the whole run. It is intentionally None for the product path
/// (`sofuu_loop_run`): a long-lived server SHOULD keep serving, and the
/// network layer has its own request deadlines. The bound exists for the
/// test helper, where "hang forever" must be "fail with a message".
///
/// # Safety
/// `ctx` must be the live runtime context (loop thread).
pub(crate) unsafe fn sofuu_loop_run_bounded(
    ctx: *mut JSContext,
    max: Option<std::time::Duration>,
) {
    let started = max.map(|_| std::time::Instant::now());
    if loop_ptr().is_null() {
        return;
    }
    loop {
        /* First: pump any immediately-available JS microtasks */
        sofuu_flush_jobs(ctx);

        /* Then: run one round of libuv I/O (blocking with timeout) */
        let has_pending = uv::uv_run(loop_ptr(), UV_RUN_ONCE);

        /* Dispatch any SIGINT/SIGTERM that fired (process.on handlers or
         * the exit(130/143) fallback) — previously dead code. */
        process_dispatch_pending_signals();

        /* After libuv fires callbacks, flush JS microtasks again */
        sofuu_flush_jobs(ctx);

        if has_pending == 0 && uv::uv_loop_alive(loop_ptr()) == 0 {
            break;
        }

        /* Test-only escape hatch: a bound means a hang is reported as a
         * failure instead of stalling the runner forever. */
        if let (Some(t0), Some(limit)) = (started, max) {
            if t0.elapsed() > limit {
                eprintln!(
                    "[sofuu] sofuu_loop_run_bounded: giving up after {:?} \
                     (handles still alive — a test is leaking one, or a \
                     transfer never completed)",
                    limit
                );
                break;
            }
        }
    }

    /* Final flush after the loop fully drains */
    sofuu_flush_jobs(ctx);
}

/// Walk callback: closes any handle that is still active.
unsafe extern "C" fn walk_close_cb(handle: *mut UvHandle, _arg: *mut c_void) {
    // SAFETY: handle is a live handle visited by uv_walk.
    if uv::uv_is_closing(handle) == 0 {
        uv::uv_close(handle, None);
    }
    /* Last-engine walk: whatever it closes is gone — drop its registration
     * so the registry never holds stale pointers across a full close. */
    untrack_handle(handle);
}

/// Drain all remaining handles before closing the loop — prevents the GC
/// assertion by ensuring no I/O callbacks fire after JS_FreeContext().
///
/// # Safety
/// Must run on the loop thread; no uv work may be started afterwards on the
/// returned loop (the storage intentionally stays allocated — see module
/// docs: mod_process_cleanup still touches handles that reference it).
///
/// H1 multi-instance: the loop is shared by every engine in the process, so
/// a teardown only walks+closes it when the LAST engine releases its
/// reference. Destroying one of several live engines leaves the loop (and
/// the other engines' handles) intact.
#[no_mangle]
pub unsafe extern "C" fn sofuu_loop_close() {
    let lp = loop_ptr();
    if lp.is_null() || G_LOOP_REFS.load(Ordering::Relaxed) == 0 {
        return;
    }
    // Release one reference; bail unless this was the last holder.
    let prev = G_LOOP_REFS.fetch_sub(1, Ordering::Relaxed);
    if prev > 1 {
        return;
    }
    /* Walk all active handles and close them */
    uv::uv_walk(lp, Some(walk_close_cb), ptr::null_mut());

    /* Let the queued close callbacks fire — NOWAIT, never ONCE/DEFAULT.
     *
     * uv_close finalizes a handle on a later loop turn, so the loop has to
     * run once more after the walk. But the walk runs while a curl socket may
     * still be mid-transfer, and uv_run in ONCE/DEFAULT mode BLOCKS waiting
     * for I/O on it. That is the stall: on the Linux CI runners
     * sofuu-core --lib wedged forever in engine teardown inside an
     * http_client test, with the assertion output never even flushed.
     * UV_RUN_NOWAIT drains the close callbacks that are already queued and
     * returns immediately, so teardown cannot be held hostage by an in-flight
     * transfer. This is a product path: a host destroying a SofuuRuntime
     * while a request is in flight previously hung on shutdown. */
    uv::uv_run(lp, UV_RUN_NOWAIT);

    /* UV_EBUSY here means a handle is still registered, i.e. something was
     * never closed. Historically this return code was discarded (the retired
     * C loop.c did the same), which is how a leaked handle stayed invisible
     * until a later teardown wedged. Log it so a leak is diagnosable instead
     * of silent; the loop storage is intentionally left allocated either way
     * (see this function's docs). */
    let rc = uv::uv_loop_close(lp);
    if rc != 0 {
        eprintln!(
            "[sofuu] sofuu_loop_close: uv_loop_close returned {rc} \
             (UV_EBUSY) — a handle was still active at teardown"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::timer::mod_timer_register;
    use sofuu_ffi::qjs::{self, CtxPtr};

    /// F-2 (AUDIT-2026-09-01-CLI): destroying ONE of several engines must
    /// close only ITS armed handles. Engine A arms a 50ms timer, then is
    /// torn down exactly like engine_destroy's handle phase
    /// (shutdown_engine → JS_FreeContext/JS_FreeRuntime); engine B's 200ms
    /// timer then runs on the shared loop. Before the per-ctx registry,
    /// A's timer stayed armed and fired into the freed context during B's
    /// run (UAF — deterministic under sanitizers, and locked here by the
    /// white-box registry asserts regardless).
    #[test]
    fn shutdown_engine_closes_only_that_ctx_armed_handles() {
        // The process-global uv loop is shared — serialize loop-driving
        // tests (see rt/mod.rs TEST_LOOP_LOCK).
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            sofuu_loop_init();

            /* Engine A: fresh runtime/context + timer module + 50ms timer. */
            let rt_a = qjs::JS_NewRuntime();
            let ctx_a = qjs::JS_NewContext(rt_a);
            mod_timer_register(ctx_a);
            let src_a = c"globalThis.__a_fired = 0; \
setTimeout(function () { globalThis.__a_fired = 1; }, 50);";
            let r = qjs::JS_Eval(
                ctx_a,
                src_a.as_ptr(),
                src_a.to_bytes().len(),
                c"<f2-engine-a>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "engine A script must not throw");
            qjs::sofuu_js_free_value(ctx_a, r);

            /* Engine B: same setup, 200ms timer. */
            let rt_b = qjs::JS_NewRuntime();
            let ctx_b = qjs::JS_NewContext(rt_b);
            let ctp = CtxPtr::new(ctx_b);
            mod_timer_register(ctx_b);
            let src_b = c"globalThis.__b_fired = 0; \
setTimeout(function () { globalThis.__b_fired = 1; }, 200);";
            let r = qjs::JS_Eval(
                ctx_b,
                src_b.as_ptr(),
                src_b.to_bytes().len(),
                c"<f2-engine-b>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "engine B script must not throw");
            qjs::sofuu_js_free_value(ctx_b, r);

            /* Both timers are tracked against their own context. */
            assert_eq!(
                ARMED.lock().unwrap().len(),
                2,
                "A's + B's timers must be tracked"
            );

            /* Teardown A exactly like engine_destroy's handle phase. */
            sofuu_loop_shutdown_engine(ctx_a);
            {
                let armed = ARMED.lock().unwrap();
                assert_eq!(armed.len(), 1, "shutdown must drain only A's entry");
                assert_eq!(armed[0].ctx, ctx_b, "the surviving entry must be B's");
            }
            /* The shared loop must still be alive: B's timer survived. */
            assert!(
                uv::uv_loop_alive(loop_ptr()) != 0,
                "engine B's timer must survive engine A's teardown"
            );

            /* A's JS heap is gone. Before the fix, A's still-armed 50ms
             * timer fired into this freed context during B's run below. */
            qjs::JS_FreeContext(ctx_a);
            qjs::JS_FreeRuntime(rt_a);

            /* Drive the loop like engine B would: B's 200ms timer fires,
             * then the loop drains and run() returns. */
            sofuu_loop_run(ctx_b);

            let rd = qjs::JS_Eval(
                ctx_b,
                c"__b_fired".as_ptr(),
                c"__b_fired".to_bytes().len(),
                c"<f2-read>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(rd), "reading __b_fired must not throw");
            let mut b_i: i32 = 0;
            assert!(qjs::to_int32(ctp, rd, &mut b_i), "__b_fired must be an int");
            qjs::sofuu_js_free_value(ctx_b, rd);
            assert_eq!(b_i, 1, "engine B's timer must fire on the shared loop");
            assert!(
                ARMED.lock().unwrap().is_empty(),
                "B's timer must be untracked once closed"
            );

            /* M0/M1 discipline: loop teardown BEFORE context teardown. */
            sofuu_loop_close();
            qjs::JS_FreeContext(ctx_b);
            qjs::JS_FreeRuntime(rt_b);
        }
    }

    /// No-op timer callback: the test never lets the loop reach the deadline.
    unsafe extern "C" fn noop_timer_cb(_t: *mut uv::UvTimer) {}

    /// Teardown closes a live handle and leaves the loop empty.
    ///
    /// Covers the HANDLE half of `sofuu_loop_close`: an armed uv_timer keeps
    /// the loop non-idle, the walk closes it, and the loop must come back
    /// empty. That invariant is worth pinning on its own.
    ///
    /// What this deliberately does NOT claim: it is not the regression test
    /// for the Linux CI hang. That hang is a pending REQUEST, not a handle —
    /// `uv_walk` visits handles only, so it cannot cancel a request, and
    /// `uv_run(UV_RUN_DEFAULT)` then waits for a completion that never comes.
    /// A timer is closed by the walk, so this test passes in 0.00s even with
    /// `UV_RUN_DEFAULT` restored (verified). Reproducing the real condition
    /// needs a request that stays pending forever; a FIFO `uv_fs_open` does
    /// that, but it also blocks a libuv threadpool thread and prevents the
    /// test process from exiting, so it is not usable as a gate. The
    /// UV_RUN_NOWAIT change is therefore justified by libuv's documented
    /// semantics — NOWAIT never blocks on I/O — not by a test that fails
    /// without it.
    #[test]
    fn loop_close_closes_a_live_handle_without_blocking() {
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            sofuu_loop_init();

            /* Arm a genuinely live libuv handle: a 60s one-shot timer keeps
             * the loop non-idle for its whole duration, so a close that waits
             * for idleness would block for a minute. */
            /* A uv_timer_t must be heap-allocated at its real size — a bare
             * `*mut UvTimer` local is only 8 bytes and uv_timer_init writes
             * past it (that segfaulted the first draft of this test). */
            let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut uv::UvTimer;
            assert!(!timer.is_null(), "timer allocation must succeed");
            uv::uv_timer_init(loop_ptr(), timer);
            uv::uv_timer_start(timer, Some(noop_timer_cb), 60_000, 0);
            assert!(
                uv::uv_loop_alive(loop_ptr()) != 0,
                "an armed timer must make the loop live, else this proves nothing"
            );

            /* Must return promptly and drain the handle. */
            sofuu_loop_close();

            assert_eq!(
                uv::uv_loop_alive(loop_ptr()),
                0,
                "teardown must leave no live handles behind"
            );

            /* Re-open the shared loop for the next test on this thread. */
            sofuu_loop_init();
            sofuu_loop_close();
        }
    }
}
