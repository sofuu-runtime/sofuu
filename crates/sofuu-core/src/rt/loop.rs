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
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use sofuu_ffi::qjs::JSContext;
use sofuu_ffi::uv::{self, UvHandle, UvLoop, UV_RUN_DEFAULT, UV_RUN_ONCE};

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

    /* Run the loop briefly to let close callbacks fire */
    uv::uv_run(lp, UV_RUN_DEFAULT);

    uv::uv_loop_close(lp); /* rc ignored — same as the retired C code */
}
