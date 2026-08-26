// rt/host_poke.rs — thread-safe host→engine wakeup (PLAN-DESKTOP A).
//
// The engine evaluates one turn as a single blocking eval on its own thread.
// An embedded host (the desktop app) still needs to reach INTO that busy
// engine to cancel a turn or resolve a pending tool-approval prompt. libuv's
// `uv_async_t` is the one handle designed for this: `uv_async_send` is the
// only uv call that is safe from another thread, and it wakes a loop blocked
// in `uv_run` even when the handle is unref'd.
//
// Shape:
//   - one unref'd `uv_async_t` on the shared global loop, created at engine
//     boot on the loop thread (unref'd ⇒ it never keeps the loop alive, so
//     `sofuu_loop_run`'s drain semantics are unchanged for every other host);
//   - `host_poke_send(json)` may be called from ANY thread: it pushes the
//     message into a mutex'd queue and sends;
//   - the async callback (loop thread) drains the queue and delivers each
//     message to the JS global `__host_poke(json)`. All state stays in JS —
//     cancel maps to `sofuu.agent.cancel`, approvals resolve a stored JS
//     promise — so there is no Rust-side JS-value registry.
//
// Messages sent before the JS side defines `__host_poke` are re-queued and
// delivered on the next poke (nothing is dropped during boot).

use std::collections::VecDeque;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

use sofuu_ffi::qjs::{self, JSContext};
use sofuu_ffi::uv::{self, UvAsync, UvHandle};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::sofuu_flush_jobs;

/// Messages queued by host threads, drained by the loop thread.
static QUEUE: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
/// The live async handle; null until `host_poke_init` runs.
static HANDLE: AtomicPtr<UvAsync> = AtomicPtr::new(ptr::null_mut());

/// Per-handle context: the leading `data` field of the uv handle points at
/// this (same pattern as rt/timer.rs).
struct PokeData {
    ctx: *mut JSContext,
}

/// Queue one JSON message and wake the engine loop.
///
/// Thread-safe by design — this is the whole point of the module. Safe to
/// call before `host_poke_init` (the message waits in the queue) and from
/// any thread while a turn's blocking eval is running.
pub fn host_poke_send(json: &str) {
    if let Ok(mut q) = QUEUE.lock() {
        q.push_back(json.to_string());
    }
    let h = HANDLE.load(Ordering::Acquire);
    if h.is_null() {
        return; // not initialized yet — the message stays queued
    }
    // SAFETY: the handle was initialized by host_poke_init and is only
    // closed at engine teardown on the loop thread; uv_async_send is the
    // one libuv call documented safe from any thread.
    unsafe { uv::uv_async_send(h) };
}

/// Deliver queued messages to the JS global `__host_poke(json)`.
unsafe extern "C" fn on_poke(handle: *mut UvAsync) {
    // SAFETY: data was set at init; the callback runs on the loop thread.
    let d = *(handle as *mut *mut PokeData);
    if d.is_null() {
        return;
    }
    let ctx = (*d).ctx;
    loop {
        let msg = match QUEUE.lock() {
            Ok(mut q) => q.pop_front(),
            Err(_) => return,
        };
        let Some(msg) = msg else { return };

        // SAFETY: ctx is the live engine context (loop thread).
        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let f = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"__host_poke".as_ptr()) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        if unsafe { qjs::JS_IsFunction(ctx, f) } == 0 {
            // No handler yet (early boot) — put the message back and wait
            // for the next send; pokes are never dropped.
            unsafe { qjs::sofuu_js_free_value(ctx, f) };
            if let Ok(mut q) = QUEUE.lock() {
                q.push_front(msg);
            }
            return;
        }

        let cmsg = CString::new(msg).unwrap_or_default();
        let arg = unsafe { qjs::sofuu_js_new_string(ctx, cmsg.as_ptr()) };
        let args = [arg];
        let ret = unsafe {
            qjs::JS_Call(ctx, f, qjs::sofuu_js_undefined(), 1, args.as_ptr() as *const _)
        };
        if unsafe { qjs::is_exception(ret) } {
            // A broken __host_poke must not take the loop down — dump and
            // continue draining (mirrors the timer callback's discipline).
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        unsafe { qjs::sofuu_js_free_value(ctx, ret) };
        unsafe { qjs::sofuu_js_free_value(ctx, arg) };
        unsafe { qjs::sofuu_js_free_value(ctx, f) };
        // Let the handler's promise resolutions run before the next message.
        unsafe { sofuu_flush_jobs(ctx) };
    }
}

/// Close callback: release the PokeData + handle storage once libuv is done.
unsafe extern "C" fn on_poke_close(handle: *mut UvHandle) {
    // SAFETY: data was set at init; the handle is fully closed here.
    let d = *(handle as *mut *mut PokeData);
    if !d.is_null() {
        drop(unsafe { Box::from_raw(d) });
    }
    unsafe { libc::free(handle as *mut c_void) };
}

/// Create the poke handle on the shared loop.
///
/// # Safety
/// Must run on the loop thread after `sofuu_loop_init`, with a live `ctx`.
/// Idempotent: the FIRST engine to initialize owns the poke (P0-7: exactly
/// one runtime per process in every embedded host, so this is the only one).
#[no_mangle]
pub unsafe extern "C" fn sofuu_host_poke_init(ctx: *mut JSContext) {
    if !HANDLE.load(Ordering::Relaxed).is_null() {
        return;
    }
    let lp = unsafe { sofuu_loop_get() };
    if lp.is_null() {
        return;
    }
    let handle = unsafe { libc::malloc(uv::sofuu_uv_async_size()) } as *mut UvAsync;
    if handle.is_null() {
        return;
    }
    let d = Box::into_raw(Box::new(PokeData { ctx }));
    // SAFETY: handle is a fresh allocation sized for a uv_async_t.
    unsafe { *(handle as *mut *mut PokeData) = d };
    let rc = unsafe { uv::uv_async_init(lp, handle, Some(on_poke)) };
    if rc != 0 {
        unsafe {
            drop(Box::from_raw(d));
            libc::free(handle as *mut c_void);
        }
        return;
    }
    // SAFETY: initialized above; unref so the poke never keeps the loop
    // alive on its own (drain semantics for every other host stay intact).
    unsafe { uv::uv_unref(handle as *mut UvHandle) };
    HANDLE.store(handle, Ordering::Release);
}

/// Close the poke handle before the loop tears down.
///
/// # Safety
/// Loop thread only; call before `sofuu_loop_close` (the close walk there
/// skips handles already closing, and the run that follows fires our close
/// callback, freeing the storage).
#[no_mangle]
pub unsafe extern "C" fn sofuu_host_poke_shutdown() {
    let h = HANDLE.swap(ptr::null_mut(), Ordering::AcqRel);
    if h.is_null() {
        return;
    }
    // SAFETY: h is the live handle; we are on the loop thread.
    unsafe { uv::uv_close(h as *mut UvHandle, Some(on_poke_close)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::promise::sofuu_rt_install_rejection_tracker;

    /// Proof: a poke sent from ANOTHER thread wakes the loop while the main
    /// thread is blocked draining it, and the message reaches `__host_poke`.
    #[test]
    fn host_poke_wakes_blocked_loop_cross_thread() {
        // The process-global uv loop is shared — serialize loop-driving
        // tests (see rt/mod.rs TEST_LOOP_LOCK).
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        unsafe { sofuu_rt_install_rejection_tracker(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        unsafe { crate::rt::timer::mod_timer_register(ctx) };
        unsafe { sofuu_host_poke_init(ctx) };

        // JS side: wait on a promise that only a poke can resolve. A long
        // keep-alive timer keeps the loop BLOCKED in uv_run (exactly like a
        // real turn's live I/O handles); the poke handler clears it so the
        // loop drains the moment the poke lands.
        let setup = c"globalThis.__got = null; \
globalThis.__keepalive = setTimeout(function () {}, 30000); \
globalThis.__host_poke = function (json) { \
  globalThis.__got = json; \
  clearTimeout(globalThis.__keepalive); \
  if (globalThis.__wake) { globalThis.__wake(json); globalThis.__wake = null; } \
}; \
globalThis.__wait = new Promise(function (r) { globalThis.__wake = r; });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                setup.as_ptr(),
                setup.to_bytes().len(),
                c"<poke-setup>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // The waiter: an eval that only completes once the poke resolves it.
        let waiter = c"globalThis.__wait.then(function (v) { globalThis.__done = v; });";

        // Another thread pokes while the loop below is blocked in uv_run.
        let sender = std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            host_poke_send(r#"{"type":"cancel"}"#);
        });

        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                waiter.as_ptr(),
                waiter.to_bytes().len(),
                c"<poke-waiter>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "waiter must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        let started = std::time::Instant::now();
        // Blocks until the cross-thread poke resolves the promise (the
        // keep-alive timer alone would hold it for 30s — the poke must
        // arrive first and clear it).
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };
        sender.join().unwrap();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the poke must wake the loop promptly, not wait out the timer"
        );

        // SAFETY: ctx live on this thread; globals freed below.
        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let done = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"__done".as_ptr()) };
        let s = unsafe { qjs::sofuu_js_to_cstring(ctx, done) };
        assert!(!s.is_null(), "the poke must resolve the waiter");
        let got = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().into_owned();
        unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        assert_eq!(got, r#"{"type":"cancel"}"#);
        unsafe { qjs::sofuu_js_free_value(ctx, done) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // SAFETY: teardown in the engine order (poke → loop → ctx → rt).
        unsafe { sofuu_host_poke_shutdown() };
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Messages sent before `__host_poke` exists are re-queued, not dropped.
    #[test]
    fn host_poke_requeues_when_handler_missing() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        unsafe { sofuu_rt_install_rejection_tracker(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        unsafe { crate::rt::timer::mod_timer_register(ctx) };
        unsafe { sofuu_host_poke_init(ctx) };

        // Send while no __host_poke exists; the send itself wakes the loop.
        host_poke_send(r#"{"type":"early"}"#);
        // A timer gives the loop something real to drain so loop_run returns.
        let setup = c"globalThis.__seen = []; \
setTimeout(function () { globalThis.__ticked = true; }, 20);";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                setup.as_ptr(),
                setup.to_bytes().len(),
                c"<poke-early>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) });
        unsafe { qjs::sofuu_js_free_value(ctx, r) };
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        // Now install the handler and poke again — BOTH messages must arrive
        // in order (the early one was re-queued).
        let install = c"globalThis.__host_poke = function (json) { globalThis.__seen.push(json); };";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                install.as_ptr(),
                install.to_bytes().len(),
                c"<poke-install>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) });
        unsafe { qjs::sofuu_js_free_value(ctx, r) };
        host_poke_send(r#"{"type":"late"}"#);
        let drain = c"setTimeout(function () {}, 1);";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                drain.as_ptr(),
                drain.to_bytes().len(),
                c"<poke-drain>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) });
        unsafe { qjs::sofuu_js_free_value(ctx, r) };
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let seen = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"__seen".as_ptr()) };
        let json = unsafe { qjs::JS_JSONStringify(ctx, seen, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined()) };
        let s = unsafe { qjs::sofuu_js_to_cstring(ctx, json) };
        let got = unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().into_owned();
        unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        unsafe { qjs::sofuu_js_free_value(ctx, json) };
        unsafe { qjs::sofuu_js_free_value(ctx, seen) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        assert_eq!(got, r#"["{\"type\":\"early\"}","{\"type\":\"late\"}"]"#);

        unsafe { sofuu_host_poke_shutdown() };
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }
}
