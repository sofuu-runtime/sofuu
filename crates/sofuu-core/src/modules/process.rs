// modules/process.rs — process object + TTY readline + chat bridges
// (PLAN-RUST-MIGRATION M3).
//
// Port of the deleted `src/modules/mod_process.c` (1,920 lines), semantics
// verbatim: the process object (argv/env/exit/cwd/chdir/on/off/stdin/
// stdout/stderr), SIGINT/SIGTERM flag-setting + loop-thread dispatch, the
// async TTY readline (raw mode, live-region box renderer, slash-command
// completion, history with ~/.sofuu/chat_history persistence, ESC state
// machine, selector overlay), the stdin pipe, and the __tui_* / __selector_*
// / __prompt / __readline / __ttyRaw / __ttyNormal chat bridges.
//
// C symbols replaced (all callers unchanged): mod_process_register,
// mod_process_set_args, mod_process_cleanup, process_dispatch_pending_signals,
// process_dispatch_uncaught. The TUI itself stays C (src/io/tui.c) until M10
// — this module calls it over FFI like the C code did.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::os::raw::c_long;
use std::ptr;
use std::sync::atomic::{AtomicI32, Ordering};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvHandle, UvPipe, UvSignal, UvStream, UvTimer, UvTty};

use crate::embed_config;
use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject_str, sofuu_promise_resolve,
    PromiseHandle,
};

// ── tui.c FFI (stays C until M10) ────────────────────────────────────
extern "C" {
    fn tui_init();
    fn tui_enter();
    fn tui_exit();
    fn tui_active() -> c_int;
    fn tui_relayout();
    fn tui_set_header(line: *const c_char);
    fn tui_log(line: *const c_char);
    fn tui_log_last(line: *const c_char);
    fn tui_scroll(delta: c_int);
    fn tui_conv_top() -> c_int;
    fn tui_conv_bottom() -> c_int;
    fn tui_box_top() -> c_int;
    fn tui_phase_row() -> c_int;
    fn tui_input_row() -> c_int;
    fn tui_footer_row() -> c_int;
    fn tui_footer2_row() -> c_int;
    fn tui_width() -> c_int;
    fn tui_height() -> c_int;
    fn tui_disp_width(s: *const c_char, len: usize) -> c_int;
    fn tui_truncate_cells(s: *const c_char, max_cells: c_int) -> usize;
    fn tui_render_conversation_range(bottom: c_int);
    fn tui_render_conversation();
    fn tui_place_cursor(col: usize);
}

// Windows has no `environ` symbol (the CRT keeps the block internal);
// the process.env builder below uses std::env::vars() there instead.
#[cfg(not(target_os = "windows"))]
extern "C" {
    /// The C environment block (like `extern char **environ`).
    static mut environ: *mut *mut c_char;
}

// ── module state (single-threaded runtime — C-stattic parity) ────────

/* Input buffer cap. Was 8 KiB (the C readline's g_line_buf), which silently
 * truncated a large pasted prompt at half its size. One megabyte holds any
 * realistic prompt/commit-diff/code paste with room to spare; the Vec grows
 * on demand, so the raise costs nothing until it is used. */
const LINE_BUF_CAP: usize = 1 << 20;
const HIST_MAX: usize = 64;
const HIST_LEN: usize = 512;
const RL_QUEUE_MAX: usize = 64;
const MAX_SUGG: usize = 8;
/* Inline (non-TUI) input box: max rows drawn for one line buffer. A big
 * multi-line paste can hold thousands of newlines; the box shows the LAST
 * few rows (where the cursor is) plus a head count, instead of scrolling
 * the whole conversation off-screen. */
const MAX_BOX_ROWS: usize = 6;

thread_local! {
    static ARGC: Cell<c_int> = const { Cell::new(0) };
    static ARGV: RefCell<Vec<CString>> = const { RefCell::new(Vec::new()) };

    /// g_ctx — set in mod_process_register.
    static CTX: Cell<*mut JSContext> = const { Cell::new(ptr::null_mut()) };

    // Async TTY readline state.
    static TTY_STDIN: Cell<*mut UvTty> = const { Cell::new(ptr::null_mut()) };
    static TTY_OPEN: Cell<c_int> = const { Cell::new(0) };
    static STDIN_IS_TTY: Cell<c_int> = const { Cell::new(0) };
    static TTY_CTX: Cell<*mut JSContext> = const { Cell::new(ptr::null_mut()) };
    static READLINE_PROMISE: Cell<*mut PromiseHandle> = const { Cell::new(ptr::null_mut()) };
    static LINE_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LINE_TRUNCATED: Cell<c_int> = const { Cell::new(0) };
    static RL_PROMPT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };

    // stdin pipe (process.stdin) state.
    static STDIN_PIPE: Cell<*mut UvPipe> = const { Cell::new(ptr::null_mut()) };
    static STDIN_PIPE_OPEN: Cell<c_int> = const { Cell::new(0) };
    static STDIN_CTX: Cell<*mut JSContext> = const { Cell::new(ptr::null_mut()) };
    static STDIN_ON_DATA: Cell<JSValue> = const { Cell::new(zero_jsvalue()) };
    static STDIN_ON_END: Cell<JSValue> = const { Cell::new(zero_jsvalue()) };
    static STDIN_ON_ERROR: Cell<JSValue> = const { Cell::new(zero_jsvalue()) };
    /// F-2: the engine that stored the .on() handlers (first writer wins —
    /// STDIN_CTX is only stamped when the pipe itself opens, but "end"/
    /// "error" handlers can be registered without it). mod_process_cleanup
    /// frees the handler values with the OWNER's ctx; freeing them with a
    /// different engine's ctx is UB.
    static STDIN_OWNER: Cell<*mut JSContext> = const { Cell::new(ptr::null_mut()) };

    // readline pre-buffer queue (ring of 63 usable slots, like C).
    static RL_QUEUE: RefCell<VecDeque<Vec<u8>>> = const { RefCell::new(VecDeque::new()) };

    // Live-region UI state.
    static UI_LINES: Cell<c_int> = const { Cell::new(0) };
    static STATUS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static HINTS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static METRIC: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };          /* right-aligned (RAM) */
    static METRIC_LEFT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };     /* left-aligned (ctx) */

    // History (RAM) + persistence flag.
    static HISTORY: RefCell<VecDeque<Vec<u8>>> = const { RefCell::new(VecDeque::new()) };
    static HIST_POS: Cell<c_long> = const { Cell::new(-1) };
    static CHAT_HIST_LOADED: Cell<c_int> = const { Cell::new(0) };

    // Slash-command completion: highlighted suggestion index (-1 = none).
    static COMPLETE_SEL: Cell<c_int> = const { Cell::new(-1) };

    // ESC sequence state machine + bare-ESC disambiguation timer.
    static ESC_STATE: Cell<c_int> = const { Cell::new(0) };
    static ESC_TIMER: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
    static ESC_TIMER_INIT: Cell<c_int> = const { Cell::new(0) };



    // SIGWINCH watcher (TUI resize).
    static WINCH: Cell<*mut UvSignal> = const { Cell::new(ptr::null_mut()) };
    static WINCH_STARTED: Cell<c_int> = const { Cell::new(0) };

    // Selector (picker overlay).
    static SELECTOR_ACTIVE: Cell<c_int> = const { Cell::new(0) };
    static SELECTOR_PROMISE: Cell<*mut PromiseHandle> = const { Cell::new(ptr::null_mut()) };
    static SEL_TOP: Cell<c_int> = const { Cell::new(-1) };
    static SEL_ROWS: Cell<c_int> = const { Cell::new(0) };

    // Focus: window (conversation scroll) vs input field. Scroll keys act on
    // the window when focus is window; otherwise they do input history.
    static FOCUS_WINDOW: Cell<c_int> = const { Cell::new(0) };
    static MOUSE_ENABLED: Cell<c_int> = const { Cell::new(0) };
    static MOUSE_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };

    // App-managed text selection (the TUI owns the alt screen + mouse, so
    // the terminal's native click-drag never runs). Rows are SCREEN rows;
    // tui.rs maps them to buffer text at copy time. -1 = no selection.
    static SEL_ANCHOR: Cell<c_int> = const { Cell::new(-1) };
    static SEL_END: Cell<c_int> = const { Cell::new(-1) };
    // Transient "Copied N lines" notice: drawn on the phase row (h-6, a
    // blank gap row) and cleared by a uv timer + on every selection change.
    static SEL_NOTICE_MS: Cell<i64> = const { Cell::new(0) };
}

/// A zeroed JSValue for Cell initializers (never read before it is set to a
/// real value — undefined/null are immediates, so a zero tag is fine).
const fn zero_jsvalue() -> JSValue {
    JSValue {
        u: sofuu_ffi::qjs::JSValueUnion { int32: 0 },
        tag: 0,
    }
}

/// g_argc / g_argv — C symbol replacement.
///
/// # Safety
/// `argv` must be the argv array from main (lives for the process lifetime).
#[no_mangle]
pub unsafe extern "C" fn mod_process_set_args(argc: c_int, argv: *mut *mut c_char) {
    ARGC.with(|a| a.set(argc));
    ARGV.with(|v| {
        let mut v = v.borrow_mut();
        v.clear();
        if !argv.is_null() {
            for i in 0..argc {
                // SAFETY: argv[i] is a live NUL-terminated string from main.
                let p = *argv.add(i as usize);
                if p.is_null() {
                    break;
                }
                let s = CStr::from_ptr(p).to_bytes();
                v.push(CString::new(s).unwrap_or_default());
            }
        }
    });
}

// ── process callbacks (hidden props on the process object) ───────────

/// Retrieve a signal callback stored on the process object.
unsafe fn get_process_cb(ctx: *mut JSContext, propname: &str) -> JSValue {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let process = qjs::sofuu_js_get_property_str(ctx, global, c"process".as_ptr());
    let cb = qjs::sofuu_js_get_property_str(ctx, process, CString::new(propname).unwrap_or_default().as_ptr());
    qjs::sofuu_js_free_value(ctx, process);
    qjs::sofuu_js_free_value(ctx, global);
    cb
}

/// Store a signal callback on the process object (GC-tracked there).
unsafe fn set_process_cb(ctx: *mut JSContext, propname: &str, fn_: JSValue) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let process = qjs::sofuu_js_get_property_str(ctx, global, c"process".as_ptr());
    qjs::sofuu_js_set_property_str(ctx, process, CString::new(propname).unwrap_or_default().as_ptr(), fn_);
    qjs::sofuu_js_free_value(ctx, process);
    qjs::sofuu_js_free_value(ctx, global);
}

unsafe fn dispatch_signal_cb(ctx: *mut JSContext, propname: &str, name: &str) {
    let cb = get_process_cb(ctx, propname);
    if qjs::JS_IsFunction(ctx, cb) == 0 {
        qjs::sofuu_js_free_value(ctx, cb);
        return;
    }
    let arg = qjs::sofuu_js_new_string(ctx, CString::new(name).unwrap_or_default().as_ptr());
    let ret = qjs::JS_Call(ctx, cb, qjs::sofuu_js_undefined(), 1, &arg);
    qjs::sofuu_js_free_value(ctx, arg);
    qjs::sofuu_js_free_value(ctx, cb);
    if qjs::is_exception(ret) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        let msg_v = qjs::JS_ToString(ctx, exc);
        let s = qjs::sofuu_js_to_cstring(ctx, msg_v);
        let msg = if s.is_null() {
            "?".to_string()
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        };
        eprintln!("[sofuu] Error in {} handler: {}", name, msg);
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
        qjs::sofuu_js_free_value(ctx, msg_v);
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    sofuu_flush_jobs(ctx);
}

/// Dispatch an uncaught exception (top-level eval failure or a job-drain
/// exception) to process.on('uncaughtException') if one is registered.
/// Returns 1 when a handler ran; 0 when nothing handled it.
/// C symbol replacement (loop/promise/timer callers unchanged).
///
/// # Safety
/// `ctx` live; `exc` a live exception value (ownership stays with the caller).
#[no_mangle]
pub unsafe extern "C" fn process_dispatch_uncaught(ctx: *mut JSContext, exc: JSValue) -> c_int {
    let cb = get_process_cb(ctx, "__uncaught_handler");
    let handled = qjs::JS_IsFunction(ctx, cb);
    if handled == 0 {
        qjs::sofuu_js_free_value(ctx, cb);
        return 0;
    }
    let ret = qjs::JS_Call(ctx, cb, qjs::sofuu_js_undefined(), 1, &exc);
    if qjs::is_exception(ret) {
        let e2 = qjs::sofuu_js_get_exception(ctx);
        let msg_v = qjs::JS_ToString(ctx, e2);
        let s = qjs::sofuu_js_to_cstring(ctx, msg_v);
        let msg = if s.is_null() {
            "?".to_string()
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        };
        eprintln!("[sofuu] Error in uncaughtException handler: {}", msg);
        if !s.is_null() {
            qjs::sofuu_js_free_cstring(ctx, s);
        }
        qjs::sofuu_js_free_value(ctx, msg_v);
        qjs::sofuu_js_free_value(ctx, e2);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, cb);
    sofuu_flush_jobs(ctx);
    1
}

/// Invoke a no-arg UI hook global (e.g. __on_esc / __on_ctrl_c registered
/// by the chat driver). No-op when the hook doesn't exist. Does NOT flush
/// jobs — we may be inside a libuv read callback; the loop pumps.
unsafe fn call_ui_hook(ctx: *mut JSContext, name: &str) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fn_ = qjs::sofuu_js_get_property_str(ctx, global, CString::new(name).unwrap_or_default().as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    if qjs::JS_IsFunction(ctx, fn_) == 0 {
        qjs::sofuu_js_free_value(ctx, fn_);
        return;
    }
    let ret = qjs::JS_Call(ctx, fn_, qjs::sofuu_js_undefined(), 0, ptr::null());
    if qjs::is_exception(ret) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, ret);
}

// ── raw POSIX signal handlers (flags only, like C) ───────────────────

static GOT_SIGINT: AtomicI32 = AtomicI32::new(0);
static GOT_SIGTERM: AtomicI32 = AtomicI32::new(0);

extern "C" fn posix_sigint_handler(_sig: c_int) {
    GOT_SIGINT.store(1, Ordering::SeqCst);
}
extern "C" fn posix_sigterm_handler(_sig: c_int) {
    GOT_SIGTERM.store(1, Ordering::SeqCst);
}

/* Windows has no POSIX sigaction; the CLI is console-app driven, so
   signal callbacks are POSIX-only (dispatch still runs, flags stay 0). */
#[cfg(not(target_os = "windows"))]
unsafe fn install_sigaction(signum: c_int, handler: extern "C" fn(c_int)) {
    let mut sa: libc::sigaction = std::mem::zeroed();
    sa.sa_sigaction = handler as usize;
    libc::sigemptyset(&mut sa.sa_mask);
    sa.sa_flags = libc::SA_RESTART;
    libc::sigaction(signum, &sa, ptr::null_mut());
}

#[cfg(target_os = "windows")]
unsafe fn install_sigaction(_signum: c_int, _handler: extern "C" fn(c_int)) {}

/// Dispatch any SIGINT/SIGTERM that fired since the last check. C symbol
/// replacement (sofuu_loop_run calls it after every UV_RUN_ONCE).
#[no_mangle]
pub unsafe extern "C" fn process_dispatch_pending_signals() {
    let ctx = CTX.with(|c| c.get());
    if ctx.is_null() {
        return;
    }

    if GOT_SIGINT.swap(0, Ordering::SeqCst) != 0 {
        let cb = get_process_cb(ctx, "__sigint_handler");
        if qjs::JS_IsFunction(ctx, cb) != 0 {
            dispatch_signal_cb(ctx, "__sigint_handler", "SIGINT");
        } else {
            qjs::sofuu_js_free_value(ctx, cb);
            if embed_config::is_embedded() {
                /* Host owns the process — never libc::exit (H2.1). */
                embed_config::log("warn", "SIGINT received (no handler); ignoring in embedded mode");
            } else {
                eprintln!();
                libc::exit(130);
            }
        }
        qjs::sofuu_js_free_value(ctx, cb);
    }
    if GOT_SIGTERM.swap(0, Ordering::SeqCst) != 0 {
        let cb = get_process_cb(ctx, "__sigterm_handler");
        if qjs::JS_IsFunction(ctx, cb) != 0 {
            dispatch_signal_cb(ctx, "__sigterm_handler", "SIGTERM");
        } else {
            qjs::sofuu_js_free_value(ctx, cb);
            if embed_config::is_embedded() {
                /* Host owns the process — never libc::exit (H2.1). */
                embed_config::log("warn", "SIGTERM received (no handler); ignoring in embedded mode");
            } else {
                libc::exit(143);
            }
        }
        qjs::sofuu_js_free_value(ctx, cb);
    }
}

// ── selector helpers (shared with the readline) ──────────────────────

/// Forward a selector key event to the driver's __selector_key(name, ch).
unsafe fn selector_key(ctx: *mut JSContext, name: &str, ch: Option<&str>) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fn_ = qjs::sofuu_js_get_property_str(ctx, global, c"__selector_key".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    if qjs::JS_IsFunction(ctx, fn_) == 0 {
        qjs::sofuu_js_free_value(ctx, fn_);
        return;
    }
    let name_c = CString::new(name).unwrap_or_default();
    let ch_c = ch.map(|c| CString::new(c).unwrap_or_default());
    let mut args: [JSValueConst; 2] = [
        // SAFETY: both strings live for the call.
        qjs::sofuu_js_new_string(ctx, name_c.as_ptr()),
        ch_c.as_ref().map_or(qjs::sofuu_js_undefined(), |c| qjs::sofuu_js_new_string(ctx, c.as_ptr())),
    ];
    let ret = qjs::JS_Call(ctx, fn_, qjs::sofuu_js_undefined(), 2, args.as_mut_ptr());
    if qjs::is_exception(ret) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, args[0]);
    qjs::sofuu_js_free_value(ctx, args[1]);
    qjs::sofuu_js_free_value(ctx, fn_);
}

/// Resolve a pending selector as cancelled (Ctrl-C / Ctrl-D / EOF paths).
/// No re-render: those paths exit the chat right after.
unsafe fn selector_resolve_cancel(ctx: *mut JSContext) {
    let p = SELECTOR_PROMISE.with(|s| s.replace(ptr::null_mut()));
    if p.is_null() {
        return;
    }
    SELECTOR_ACTIVE.with(|a| a.set(0));
    SEL_TOP.with(|t| t.set(-1));
    SEL_ROWS.with(|r| r.set(0));
    if TTY_OPEN.with(|o| o.get()) != 0 {
        // SAFETY: stdin_rl_stream is live while tty is open.
        uv::uv_unref(stdin_rl_stream() as *mut UvHandle);
    }
    let r = qjs::sofuu_js_new_string(ctx, c"cancel".as_ptr());
    sofuu_promise_resolve(p, r);
    qjs::sofuu_js_free_value(ctx, r);
}

// ── process.stdin — libuv pipe on fd 0 ───────────────────────────────

unsafe extern "C" fn stdin_alloc_cb(_h: *mut UvHandle, suggested: usize, buf: *mut uv::UvBuf) {
    // SAFETY: libuv-provided buffer slot.
    (*buf).base = libc::malloc(suggested) as *mut c_char;
    (*buf).len = (*buf).base.is_null().then_some(0).unwrap_or(suggested);
}

/// Callbacks below are only reached while the pipe exists.
unsafe extern "C" fn stdin_read_cb(stream: *mut UvStream, nread: isize, buf: *const uv::UvBuf) {
    let ctx = STDIN_CTX.with(|c| c.get());
    if ctx.is_null() {
        if !(*buf).base.is_null() {
            libc::free((*buf).base as *mut c_void);
        }
        return;
    }
    if nread > 0 {
        let data = STDIN_ON_DATA.with(|d| d.get());
        if qjs::JS_IsFunction(ctx, data) != 0 {
            let chunk = qjs::JS_NewStringLen(ctx, (*buf).base, nread as usize);
            let ret = qjs::JS_Call(ctx, data, qjs::sofuu_js_undefined(), 1, &chunk);
            qjs::sofuu_js_free_value(ctx, chunk);
            qjs::sofuu_js_free_value(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    } else if nread == uv::UV_EOF as isize {
        uv::uv_read_stop(stream);
        let end = STDIN_ON_END.with(|e| e.get());
        if qjs::JS_IsFunction(ctx, end) != 0 {
            let ret = qjs::JS_Call(ctx, end, qjs::sofuu_js_undefined(), 0, ptr::null());
            qjs::sofuu_js_free_value(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    } else if nread < 0 && nread != uv::UV_EOF as isize {
        let err = STDIN_ON_ERROR.with(|e| e.get());
        if qjs::JS_IsFunction(ctx, err) != 0 {
            let errv = qjs::sofuu_js_new_string(ctx, uv::uv_strerror(nread as c_int));
            let ret = qjs::JS_Call(ctx, err, qjs::sofuu_js_undefined(), 1, &errv);
            qjs::sofuu_js_free_value(ctx, errv);
            qjs::sofuu_js_free_value(ctx, ret);
            sofuu_flush_jobs(ctx);
        }
    }
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

/// Open the stdin pipe once and start reading (same as C).
unsafe fn ensure_stdin_open(ctx: *mut JSContext) {
    if STDIN_PIPE_OPEN.with(|o| o.get()) != 0 {
        return;
    }
    STDIN_CTX.with(|c| c.set(ctx));
    let pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    STDIN_PIPE.with(|p| p.set(pipe));
    uv::uv_pipe_init(sofuu_loop_get(), pipe, 0);
    uv::uv_pipe_open(pipe, 0);
    STDIN_PIPE_OPEN.with(|o| o.set(1));
    /* The pipe IS reffed — it keeps the loop alive until EOF/close. */
    /* F-2: armed per-ctx — a multi-engine teardown must close it
     * (mod_process_cleanup closes it with None, same orphan cb). */
    crate::rt::event_loop::track_handle(ctx, pipe as *mut UvHandle, None);
}

unsafe extern "C" fn js_stdin_on(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"stdin.on requires (event, callback)".as_ptr());
    }

    let event_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if event_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let event = CStr::from_ptr(event_ptr).to_string_lossy().into_owned();

    if qjs::JS_IsFunction(ctx, *argv.add(1)) == 0 {
        qjs::sofuu_js_free_cstring(ctx, event_ptr);
        return qjs::JS_ThrowTypeError(ctx, c"stdin.on: handler must be a function".as_ptr());
    }

    let fn_ = qjs::sofuu_js_dup_value(ctx, *argv.add(1));

    /* F-2: first engine to store a handler owns the STDIN_ON_* values. */
    if STDIN_OWNER.with(|o| o.get()).is_null() {
        STDIN_OWNER.with(|o| o.set(ctx));
    }

    match event.as_str() {
        "data" => {
            STDIN_ON_DATA.with(|d| {
                qjs::sofuu_js_free_value(ctx, d.replace(fn_));
            });
            ensure_stdin_open(ctx);
            let pipe = STDIN_PIPE.with(|p| p.get());
            uv::uv_read_start(pipe as *mut UvStream, Some(stdin_alloc_cb), Some(stdin_read_cb));
        }
        "end" => {
            STDIN_ON_END.with(|e| {
                qjs::sofuu_js_free_value(ctx, e.replace(fn_));
            });
        }
        "error" => {
            STDIN_ON_ERROR.with(|e| {
                qjs::sofuu_js_free_value(ctx, e.replace(fn_));
            });
        }
        _ => {
            qjs::sofuu_js_free_value(ctx, fn_);
        }
    }

    qjs::sofuu_js_free_cstring(ctx, event_ptr);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_stdin_resume(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    ensure_stdin_open(ctx);
    let pipe = STDIN_PIPE.with(|p| p.get());
    uv::uv_read_start(pipe as *mut UvStream, Some(stdin_alloc_cb), Some(stdin_read_cb));
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_stdin_pause(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    if STDIN_PIPE_OPEN.with(|o| o.get()) != 0 {
        let pipe = STDIN_PIPE.with(|p| p.get());
        uv::uv_read_stop(pipe as *mut UvStream);
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_stdin_set_encoding(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    /* Always UTF-8 — no-op */
    qjs::sofuu_js_undefined()
}

// ── process.stdout / process.stderr ──────────────────────────────────

unsafe extern "C" fn js_stdout_write(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 0);
    }
    let s = qjs::sofuu_js_to_cstring(ctx, *argv);
    if !s.is_null() {
        // SAFETY: s is a live C string from the JS value.
        let text = CStr::from_ptr(s).to_bytes();
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text);
        let _ = out.flush();
        qjs::sofuu_js_free_cstring(ctx, s);
    }
    qjs::sofuu_js_new_bool(ctx, 1)
}

unsafe extern "C" fn js_stderr_write(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 0);
    }
    let s = qjs::sofuu_js_to_cstring(ctx, *argv);
    if !s.is_null() {
        let text = CStr::from_ptr(s).to_bytes();
        use std::io::Write;
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(text);
        let _ = err.flush();
        qjs::sofuu_js_free_cstring(ctx, s);
    }
    qjs::sofuu_js_new_bool(ctx, 1)
}

// ── readline queue / history / width helpers ─────────────────────────

fn rl_queue_push(line: &[u8]) {
    RL_QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if q.len() >= RL_QUEUE_MAX - 1 {
            return; /* queue full — drop */
        }
        q.push_back(line.to_vec());
    });
}

fn rl_queue_pop() -> Option<Vec<u8>> {
    RL_QUEUE.with(|q| q.borrow_mut().pop_front())
}

fn rl_queue_empty() -> bool {
    RL_QUEUE.with(|q| q.borrow().is_empty())
}

fn tty_width() -> c_int {
    #[cfg(not(target_os = "windows"))]
    {
        // SAFETY: ioctl on stdout with a zeroed winsize.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as c_int;
            }
        }
    }
    80
}

fn tty_newlines_in(s: &[u8]) -> c_int {
    s.iter().filter(|&&b| b == b'\n').count() as c_int
}

// ── chat history persistence (~/.sofuu/chat_history) ──────────────────

fn chat_history_path() -> std::path::PathBuf {
    /* Honors the embedded config_root (PLAN-HEADLESS H2.4). */
    std::path::PathBuf::from(crate::embed_config::config_dir("chat_history"))
}

fn chat_history_load() {
    if CHAT_HIST_LOADED.with(|l| l.get()) != 0 {
        return;
    }
    CHAT_HIST_LOADED.with(|l| l.set(1));

    let path = chat_history_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    HISTORY.with(|h| {
        let mut h = h.borrow_mut();
        for raw in text.lines() {
            let line = raw.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                continue;
            }
            if h.len() >= HIST_MAX {
                h.pop_front();
            }
            let mut bytes = line.as_bytes().to_vec();
            bytes.truncate(HIST_LEN - 1);
            h.push_back(bytes);
        }
    });
}

/// Append one entry to ~/.sofuu/chat_history (best-effort: a missing
/// ~/.sofuu dir or unwritable file must never break the chat).
///
/// proc-11: the file is append-only forever, so it grows without bound
/// (a heavy CLI user measured years of entries). When it passes the cap,
/// rewrite it keeping the newest CHAT_HISTORY_KEEP lines.
const CHAT_HISTORY_MAX: u64 = 4 * 1024 * 1024; /* 4MB */
const CHAT_HISTORY_KEEP: usize = 2000;

/// proc-11 helper (pure, unit-tested): from a whole chat-history file,
/// keep the newest `keep` lines as a rewrite payload. Lines are
/// newline-delimited; the trailing newline is normalized.
fn chat_history_trimmed(text: &str, keep: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(keep);
    let mut out = String::with_capacity(text.len() / 2);
    for l in &lines[start..] {
        out.push_str(l);
        out.push('\n');
    }
    out
}

fn chat_history_append(line: &[u8]) {
    let dir = chat_history_path().parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
    let _ = std::fs::create_dir_all(&dir);
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(chat_history_path()) {
        let _ = f.write_all(line);
        let _ = f.write_all(b"\n");
        /* proc-11: enforce the cap after appending. A rewrite races only
         * with the single-threaded history writer (this fn), and a failed
         * trim is best-effort — the append already succeeded. */
        if let Ok(meta) = f.metadata() {
            if meta.len() > CHAT_HISTORY_MAX {
                drop(f);
                if let Ok(text) = std::fs::read_to_string(chat_history_path()) {
                    let trimmed = chat_history_trimmed(&text, CHAT_HISTORY_KEEP);
                    let _ = std::fs::write(chat_history_path(), trimmed);
                }
            }
        }
    }
}

/// UTF-8 display width in terminal CELLS — mirrors tty_disp_width() in the
/// C file (continuation bytes 0, ANSI sequences invisible).
/// Byte index at which `bytes` should be cut so it occupies at most
/// `keep` DISPLAY CELLS.
///
/// One shared implementation for the readline's truncation sites. They all
/// measured with tui_disp_width (cell-accurate) but used to cut one cell
/// per char, so a wide CJK/emoji row was allowed ~2x its real width and
/// overran the input box / suggestion box borders.
///
/// Returns a byte index that is always a char boundary and never exceeds
/// `bytes.len()` — callers slice with it directly.
fn cut_to_cells(bytes: &[u8], keep: usize) -> usize {
    let mut cells = 0usize;
    let mut cut = 0usize;
    for k in 0..bytes.len() {
        if (bytes[k] & 0xC0) != 0x80 {
            let ln = utf8_len(bytes[k]);
            let c = std::str::from_utf8(&bytes[k..(k + ln).min(bytes.len())])
                .ok()
                .and_then(|c| c.chars().next())
                .unwrap_or('\u{fffd}');
            cells += crate::rt::tui::char_cells_pub(c);
        }
        if cells > keep {
            break;
        }
        cut = k + 1;
    }
    cut
}

/// UTF-8 sequence length implied by a lead byte. Paired with the
/// cell-aware truncation below (mirrors truncate_cells in rt/tui.rs).
fn utf8_len(c: u8) -> usize {
    if c >= 0xF0 {
        4
    } else if c >= 0xE0 {
        3
    } else if c >= 0xC0 {
        2
    } else {
        1
    }
}

fn tty_disp_width(s: &[u8]) -> usize {
    let mut w = 0usize;
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if c == 0x1b {
            i += 1;
            if i < s.len() && s[i] == b'[' {
                while i + 1 < s.len() {
                    i += 1;
                    let x = s[i];
                    if (0x40..=0x7E).contains(&x) {
                        break; /* final byte */
                    }
                }
            }
            i += 1;
            continue;
        }
        if (c & 0xC0) != 0x80 {
            w += 1;
        }
        i += 1;
    }
    w
}

// ── slash-command completion (__chat_complete, registered by chat.rs) ──

/// Fallback: filter the hardcoded command list by the prefix (no JS needed).
fn fallback_complete(prefix: &[u8], out_count: &mut usize) -> Option<CString> {
    *out_count = 0;
    let prefix_str = String::from_utf8_lossy(prefix);
    let mut out = String::new();
    for cmd in FALLBACK_COMMANDS {
        if cmd.starts_with(prefix_str.as_ref()) {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(cmd);
        }
    }
    *out_count = out.lines().filter(|l| !l.is_empty()).count();
    if out.is_empty() {
        None
    } else {
        Some(CString::new(out).unwrap_or_default())
    }
}

/// Returns "\n"-joined matches and their count (0 = none). The command list
/// lives in exactly one place (chat.rs) — but a hardcoded fallback here
/// guarantees suggestions ALWAYS work even if the JS bridge is missing or
/// throws (the TUI must never silently lose the / menu).
/* P3 (AUDIT-2026-09-07): pruned /models, /baseurl, /apikey — they were
 * deliberately removed from ALL_COMMANDS (dispatch answers "unknown"; a
 * test pins it), so the offline fallback must not suggest them either. */
const FALLBACK_COMMANDS: &[&str] = &[
    "/help", "/version", "/model", "/provider",
    "/effort", "/compact", "/clear", "/brain", "/rlm", "/ctx", "/maxout",
    "/tools", "/sessions", "/context", "/work", "/done", "/note", "/notify",
    "/sync", "/exit",
];

unsafe fn tty_complete_matches(
    ctx: *mut JSContext,
    prefix: &[u8],
    out_count: &mut usize,
) -> Option<CString> {
    *out_count = 0;
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fn_ = qjs::sofuu_js_get_property_str(ctx, global, c"__chat_complete".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    if qjs::JS_IsFunction(ctx, fn_) == 0 {
        qjs::sofuu_js_free_value(ctx, fn_);
        return fallback_complete(prefix, out_count);
    }

    let pv = qjs::JS_NewStringLen(ctx, prefix.as_ptr() as *const c_char, prefix.len());
    let ret = qjs::JS_Call(ctx, fn_, qjs::sofuu_js_undefined(), 1, &pv);
    qjs::sofuu_js_free_value(ctx, pv);
    qjs::sofuu_js_free_value(ctx, fn_);
    if qjs::is_exception(ret) {
        qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, ret);
        return fallback_complete(prefix, out_count);
    }

    let s_ptr = qjs::sofuu_js_to_cstring(ctx, ret);
    qjs::sofuu_js_free_value(ctx, ret);
    if s_ptr.is_null() {
        return None;
    }
    let bytes = CStr::from_ptr(s_ptr).to_bytes();
    /* Copy BEFORE freeing: `bytes` borrows the QuickJS C string, which is
     * freed below — using it afterwards reads dangling (zeroed) memory and
     * the newline count collapsed to 1. */
    let owned = bytes.to_vec();
    qjs::sofuu_js_free_cstring(ctx, s_ptr);
    if owned.is_empty() {
        return None;
    }
    let out = CString::new(owned.clone()).unwrap_or_default();

    let mut count = 0usize;
    let mut p = owned.as_slice();
    while !p.is_empty() {
        let nl = p.iter().position(|&b| b == b'\n');
        let l = nl.unwrap_or(p.len());
        if l > 0 {
            count += 1;
        }
        match nl {
            None => break,
            Some(i) => p = &p[i + 1..],
        }
    }
    *out_count = count;
    Some(out)
}

/// The Nth newline-separated suggestion line (0-based) as a byte slice, or
/// None when out of range. Used by the arrow-key completion navigator.
fn nth_match(matches: &[u8], idx: usize) -> Option<&[u8]> {
    let mut p = matches;
    let mut cur = 0usize;
    while !p.is_empty() {
        let nl = p.iter().position(|&b| b == b'\n');
        let l = nl.unwrap_or(p.len());
        if l > 0 {
            if cur == idx {
                return Some(&p[..l]);
            }
            cur += 1;
        }
        match nl {
            None => break,
            Some(i) => p = &p[i + 1..],
        }
    }
    None
}

/// Look up a command's description/options via the `__chat_command_info`
/// JS bridge (registered by chat.rs). Returns (desc, opts) or None when the
/// bridge is unavailable. Used to render the help line in the completion menu.
unsafe fn command_info_lookup(ctx: *mut JSContext, cmd: &str) -> Option<(String, String)> {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fn_ = qjs::sofuu_js_get_property_str(ctx, global, c"__chat_command_info".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    if qjs::JS_IsFunction(ctx, fn_) == 0 {
        qjs::sofuu_js_free_value(ctx, fn_);
        return None;
    }
    let cmd_c = CString::new(cmd).unwrap_or_default();
    let arg = qjs::sofuu_js_new_string(ctx, cmd_c.as_ptr());
    let ret = qjs::JS_Call(ctx, fn_, qjs::sofuu_js_undefined(), 1, &arg);
    qjs::sofuu_js_free_value(ctx, arg);
    qjs::sofuu_js_free_value(ctx, fn_);
    if qjs::is_exception(ret) {
        qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, ret);
        return None;
    }
    let s_ptr = qjs::sofuu_js_to_cstring(ctx, ret);
    qjs::sofuu_js_free_value(ctx, ret);
    if s_ptr.is_null() {
        return None;
    }
    let json = CStr::from_ptr(s_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, s_ptr);
    /* Parse {"desc": "...", "opts": "..."} without serde — this module has
     * no serde dependency; do a minimal JSON parse via QuickJS. */
    let parsed = qjs::JS_ParseJSON(
        ctx,
        CString::new(json.clone()).unwrap_or_default().as_ptr(),
        json.len(),
        c"<cmd_info>".as_ptr(),
    );
    if qjs::is_exception(parsed) {
        qjs::sofuu_js_get_exception(ctx);
        return None;
    }
    let d = qjs::sofuu_js_get_property_str(ctx, parsed, c"desc".as_ptr());
    let o = qjs::sofuu_js_get_property_str(ctx, parsed, c"opts".as_ptr());
    let desc = {
        let p = qjs::sofuu_js_to_cstring(ctx, d);
        let s = if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() };
        if !p.is_null() { qjs::sofuu_js_free_cstring(ctx, p); }
        s
    };
    let opts = {
        let p = qjs::sofuu_js_to_cstring(ctx, o);
        let s = if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() };
        if !p.is_null() { qjs::sofuu_js_free_cstring(ctx, p); }
        s
    };
    qjs::sofuu_js_free_value(ctx, d);
    qjs::sofuu_js_free_value(ctx, o);
    qjs::sofuu_js_free_value(ctx, parsed);
    if desc.is_empty() && opts.is_empty() {
        None
    } else {
        Some((desc, opts))
    }
}

// ── the live-region renderer (inline, non-TUI) ───────────────────────

/// Draw one row of the input box (inline mode), pixel-perfect port of the C
/// fputs/fwrite sequence.
#[allow(clippy::too_many_arguments)]
unsafe fn render_box_row(
    row: usize,
    pos: usize,
    _input_rows: c_int,
    inner: c_int,
    out: &mut impl std::io::Write,
) -> usize {
    // row content: g_line_buf[pos..row_end]
    let line = LINE_BUF.with(|b| b.borrow().clone());
    let mut row_end = pos;
    while row_end < line.len() && line[row_end] != b'\n' {
        row_end += 1;
    }
    let seg = &line[pos..row_end];

    let _ = write!(out, "│ ");
    let mut prow = 0usize;
    if row == 0 {
        let prompt = RL_PROMPT.with(|p| p.borrow().clone());
        if !prompt.is_empty() {
            let _ = write!(out, "{}", String::from_utf8_lossy(&prompt));
            prow = tty_disp_width(&prompt);
        }
    }
    let avail = if inner as usize > prow {
        inner as usize - prow
    } else {
        0
    };
    let row_cells = tty_disp_width(seg);
    let mut used;
    if row_cells > avail && avail > 0 {
        let keep = if avail >= 1 { avail - 1 } else { 0 };
        let cut = cut_to_cells(seg, keep);
        let _ = write!(out, "…");
        let _ = out.write_all(&seg[cut..]);
        used = inner as usize;
    } else {
        let _ = out.write_all(seg);
        used = prow + row_cells;
    }
    while used < inner as usize {
        let _ = write!(out, " ");
        used += 1;
    }
    let _ = write!(out, "│\n");
    row_end + if row_end < line.len() { 1 } else { 0 }
}

/// Legacy inline renderer (non-TUI tty).
unsafe fn tty_render_line_ui(ctx: *mut JSContext, autocomplete: c_int) {
    use std::io::Write;

    /* Slash-command matching — shared by both renderers. */
    let line = LINE_BUF.with(|b| b.borrow().clone());
    let is_slash = !line.is_empty()
        && line[0] == b'/'
        && !line.contains(&b' ')
        && !line.is_empty();
    let (matches, n) = if is_slash {
        let mut n = 0usize;
        let m = tty_complete_matches(ctx, &line, &mut n);
        (m, n)
    } else {
        (None, 0)
    };

    let mut completed = 0;
    let matches_bytes: Option<Vec<u8>> = matches.as_ref().map(|m| m.to_bytes().to_vec());
    if autocomplete != 0 && n == 1 {
        if let Some(mb) = &matches_bytes {
            let mlen = mb.iter().position(|&b| b == b'\n').unwrap_or(mb.len());
            if mlen > line.len() {
                // SAFETY-free replacement of the C memcpy into g_line_buf.
                let mut new_line = line.clone();
                new_line.clear();
                new_line.extend_from_slice(&mb[..mlen]);
                new_line.push(b' ');
                LINE_BUF.with(|b| *b.borrow_mut() = new_line);
                completed = 1;
            }
        }
    }
    let already_complete = n == 1
        && matches_bytes.is_some()
        && matches_bytes
            .as_ref()
            .map(|mb| mb.iter().position(|&b| b == b'\n').unwrap_or(mb.len()) == line.len() && mb[..line.len()] == line[..])
            .unwrap_or(false);

    if unsafe { tui_active() } != 0 {
        tty_render_tui(ctx, matches.as_ref().map(|m| m.to_bytes()), n, completed, already_complete);
        return;
    }

    let mut W = tty_width();
    if W < 60 {
        W = 60;
    }
    if W > 200 {
        W = 200;
    }
    let inner = W - 4; /* "│ " ... " │" */
    let mut out = std::io::stdout().lock();

    /* 1. Erase the previous block from wherever the cursor is. */
    let ui_lines = UI_LINES.with(|l| l.get());
    if ui_lines > 0 {
        let _ = write!(out, "\x1b[{}A", ui_lines);
        let _ = write!(out, "\x1b[J");
        UI_LINES.with(|l| l.set(0));
    } else {
        let _ = write!(out, "\r\x1b[K");
    }

    /* 2. Draw suggestions above the box (dim), capped + scrolling window. */
    let mut shown = 0usize;
    if n > 0 && matches.is_some() && completed == 0 && !already_complete {
        let nb = matches_bytes.as_deref().unwrap_or(&[]);
        let sel = COMPLETE_SEL.with(|s| s.get());
        let mut start = 0usize;
        if sel >= 0 && (sel as usize) >= MAX_SUGG {
            start = (sel as usize) - MAX_SUGG + 1;
        }
        let mut m: &[u8] = nb;
        let mut mi = 0usize;
        loop {
            if m.is_empty() {
                break;
            }
            let nl = m.iter().position(|&b| b == b'\n');
            let l = nl.unwrap_or(m.len());
            if l > 0 {
                if mi >= start && shown < MAX_SUGG {
                    if mi == sel as usize {
                        let _ = write!(out, "  \x1b[7m{}\x1b[0m\n", String::from_utf8_lossy(&m[..l]));
                    } else {
                        let _ = write!(out, "  \x1b[2m{}\x1b[0m\n", String::from_utf8_lossy(&m[..l]));
                    }
                    shown += 1;
                }
            }
            mi += 1;
            match nl {
                None => break,
                Some(i) => m = &m[i + 1..],
            }
        }
        if n as usize > start + shown {
            let _ = write!(out, "  \x1b[2m… {} more\x1b[0m\n", n - (start + shown));
        }
    }

    /* 3. The box (accent borders): top border, input rows, bottom border.
     * A multi-line buffer (Alt/Shift+Enter, pasted text) grows the box one
     * row per newline — clamped to MAX_BOX_ROWS with a head count, so a
     * large paste cannot scroll the whole conversation away. */
    let total_rows = 1 + tty_newlines_in(&line) as usize;
    let hidden_rows = if total_rows > MAX_BOX_ROWS { total_rows - MAX_BOX_ROWS } else { 0 };
    let input_rows = (total_rows - hidden_rows) as c_int;
    if hidden_rows > 0 {
        let head = if hidden_rows > 1 { "rows" } else { "row" };
        let _ = write!(out, "\x1b[2m… {hidden_rows} more {head} (↑ recall · Enter submits all)\x1b[0m\n");
    }
    let _ = write!(out, "\x1b[2;35m╭");
    for _ in 0..(W - 2) {
        let _ = write!(out, "─");
    }
    let _ = write!(out, "╮\x1b[0m\n");

    /* Render only the TAIL rows: skip past `hidden_rows` newlines first. */
    let mut pos = 0usize;
    if hidden_rows > 0 {
        let mut skipped = 0usize;
        while pos < line.len() && skipped < hidden_rows {
            if line[pos] == b'\n' {
                skipped += 1;
            }
            pos += 1;
        }
    }
    for row in 0..input_rows {
        pos = render_box_row(row as usize, pos, input_rows, inner, &mut out);
    }

    let _ = write!(out, "\x1b[2;35m╰");
    for _ in 0..(W - 2) {
        let _ = write!(out, "─");
    }
    let _ = write!(out, "╯\x1b[0m\n");

    /* 4. Dim status footer. */
    let status = STATUS.with(|s| s.borrow().clone());
    if !status.is_empty() {
        let _ = write!(out, "\x1b[2m{}\x1b[0m\n", String::from_utf8_lossy(&status));
    } else {
        let _ = write!(out, "\n");
    }

    /* UI_ROWS includes the top+bottom borders and the hidden-rows notice
     * (1 line) so the erase math on the next render stays exact. */
    UI_LINES.with(|l| {
        l.set((shown + 2 + input_rows as usize + 1 + if hidden_rows > 0 { 1 } else { 0 }) as c_int)
    });
    let _ = out.flush();
}

// ── TUI renderer: absolute row positioning inside the alt screen ────

unsafe fn tty_render_tui(
    ctx: *mut JSContext,
    matches: Option<&[u8]>,
    n: usize,
    completed: c_int,
    already_complete: bool,
) {
    use std::io::Write;
    let W = tui_width();
    let gutter = crate::rt::tui::GUTTER as c_int;
    let inner = W - 2 - gutter; // 1 left │ + GUTTER + text + 1 right │  ≡ W-4 when GUTTER=2
    let scrolled = crate::rt::tui::tui_scroll_pos();
    let mut out = std::io::stdout().lock();

    /* While a picker overlay owns rows 1..H-3: footer only. */
    let mut used = 0usize;
    if SELECTOR_ACTIVE.with(|a| a.get()) == 0 {
        let box_top = tui_box_top();
        let conv_bottom = tui_conv_bottom();
        let mut shown = 0usize;
        let mut info_shown = 0usize;
        let clear_lo = (conv_bottom - MAX_SUGG as c_int).min(box_top - 1);
        for r in clear_lo..=conv_bottom {
            let _ = write!(out, "\x1b[{};1H\x1b[K", r);
        }
        for r in (conv_bottom + 1)..box_top {
            let _ = write!(out, "\x1b[{};1H\x1b[K", r);
        }
        /* The typed line may be an EXACT command even when it's also a
         * prefix of others (e.g. "/model" vs "/models"). When no list is
         * active (or nothing is highlighted) this shows the command's info.
         * When a multi-match list IS active with a highlight, the
         * highlighted command's info (drawn below) takes priority.
         * Drawn AFTER the list so it never gets overwritten. */
        let line_bytes = LINE_BUF.with(|b| b.borrow().clone());
        let line_str = String::from_utf8_lossy(&line_bytes);
        let list_active = n > 0 && matches.is_some() && completed == 0 && !already_complete
            && COMPLETE_SEL.with(|s| s.get()) >= 0;
        if n > 0 && matches.is_some() && completed == 0 && !already_complete {
            let mut row = tui_box_top() - 1;
            let nb = matches.unwrap_or(&[]);
            let sel = COMPLETE_SEL.with(|s| s.get());
            /* Scroll window: keep the highlighted item visible. The list
             * draws bottom-up (row starts at box_top-1 and decreases), so
             * the window is [start..start+MAX_SUGG) over the matches. */
            let mut start = 0usize;
            if sel >= 0 && (sel as usize) >= MAX_SUGG {
                start = (sel as usize) - MAX_SUGG + 1;
            }
            let mut m: &[u8] = nb;
            let mut mi = 0usize;
            let mut sel_line: Option<Vec<u8>> = None;
            while !m.is_empty() && shown < MAX_SUGG {
                let nl = m.iter().position(|&b| b == b'\n');
                let l = nl.unwrap_or(m.len());
                if l > 0 {
                    if mi >= start {
                        if mi == sel as usize {
                            /* highlighted suggestion — reverse video */
                            let _ = write!(out, "\x1b[{};1H\x1b[K  \x1b[7m{}\x1b[0m", row, String::from_utf8_lossy(&m[..l]));
                            sel_line = Some(m[..l].to_vec());
                        } else {
                            let _ = write!(out, "\x1b[{};1H\x1b[K  \x1b[2m{}\x1b[0m", row, String::from_utf8_lossy(&m[..l]));
                        }
                        shown += 1;
                        row -= 1;
                    }
                    mi += 1;
                }
                match nl {
                    None => break,
                    Some(i) => m = &m[i + 1..],
                }
            }
            /* The highlighted command's description + options render on the
             * row directly ABOVE the suggestion list. */
            let info_row = tui_box_top() - 1 - shown as c_int;
            if let Some(sline) = sel_line {
                let cmd_str = String::from_utf8_lossy(&sline);
                if let Some((desc, opts)) = command_info_lookup(ctx, &cmd_str) {
                    let mut info = desc.to_string();
                    if !opts.is_empty() {
                        info.push_str(" — ");
                        info.push_str(&opts);
                    }
                    let _ = write!(out, "\x1b[{};1H\x1b[K  \x1b[36m{}\x1b[0m", info_row, info);
                    info_shown = 1;
                }
            }
            /* "+N more" hint above the window when there are hidden items. */
            if n as usize > start + shown {
                let _ = write!(out, "\x1b[{};1H\x1b[K  \x1b[2m… {} more\x1b[0m", info_row - 1, n - (start + shown));
            }
        }
        /* Exact-command info (drawn after the list so it wins any overlap):
         * shows for "/rlm", "/model" (even though it also prefixes
         * "/models"), etc. Suppressed while a list highlight is active. */
        if !list_active {
            if let Some((desc, opts)) = command_info_lookup(ctx, &line_str) {
                let mut info = desc.to_string();
                if !opts.is_empty() {
                    info.push_str(" — ");
                    info.push_str(&opts);
                }
                let info_row = if n > 0 && matches.is_some() && !already_complete {
                    tui_box_top() - 1 - shown as c_int
                } else {
                    tui_box_top() - 1
                };
                let _ = write!(out, "\x1b[{};1H\x1b[K  \x1b[36m{}\x1b[0m", info_row, info);
                info_shown = 1;
            }
        }

        /* 2. Conversation viewport, shrunk by the suggestion rows PLUS the
         * info line (which sits one row above the list — inside the
         * conversation redraw range otherwise). */
        tui_render_conversation_range(tui_conv_bottom() - (shown as c_int) - (info_shown as c_int));

        /* 3. The input box at fixed rows (T, I, I+1) + footer (F). */
        let T = tui_box_top();
        let I = tui_input_row();

        let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[2;35m╭", T);
        for _ in 0..(W - 2) {
            let _ = write!(out, "─");
        }
        let _ = write!(out, "╮\x1b[0m");

        let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[2;35m│\x1b[0m  ", I);
        let mut prow = 0usize;
        let prompt = RL_PROMPT.with(|p| p.borrow().clone());
        if !prompt.is_empty() {
            let _ = write!(out, "{}", String::from_utf8_lossy(&prompt));
            prow = tui_disp_width(prompt.as_ptr() as *const c_char, prompt.len()) as usize;
        }
        /* single input row: show the LAST segment of a multiline buffer */
        let line = LINE_BUF.with(|b| b.borrow().clone());
        let mut seg = 0usize;
        let mut newlines = 0usize;
        for (k, &b) in line.iter().enumerate() {
            if b == b'\n' {
                seg = k + 1;
                newlines += 1;
            }
        }
        /* multi-line buffer (pasted text / Alt+Enter): prefix a dim marker so
         * the row reads as "more above" instead of a lone orphan line. */
        if newlines > 0 {
            let marker = if newlines > 1 {
                format!("\x1b[2m[+{} lines]\x1b[0m", newlines)
            } else {
                "\x1b[2m[+1 line]\x1b[0m".to_string()
            };
            let _ = write!(out, "{}", marker);
            /* "[+N lines]" cells = 2 + digits + 7; "[+1 line]" = 9. */
            prow += if newlines > 1 { 9 + newlines.to_string().len() } else { 9 };
        }
        let row_len = line.len() - seg;
        let avail = if inner > prow as c_int { (inner - prow as c_int) as usize } else { 0 };
        let row_cells = tui_disp_width(line[seg..].as_ptr() as *const c_char, row_len) as usize;
        if row_cells > avail && avail > 0 {
            let keep = if avail >= 1 { avail - 1 } else { 0 };
            let cut = cut_to_cells(&line[seg..], keep);
            let _ = write!(out, "…");
            let _ = out.write_all(&line[seg + cut..]);
            used = inner as usize;
        } else {
            let _ = out.write_all(&line[seg..]);
            used = prow + row_cells;
        }
        /* Pad to the right border with a SEPARATE counter — `used` must
         * keep the text-end column for cursor placement below. */
        let mut pad = used;
        while pad < inner as usize {
            let _ = write!(out, " ");
            pad += 1;
        }
        let _ = write!(out, "\x1b[2;35m│\x1b[0m");
        let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[2;35m╰", I + 1);
        for _ in 0..(W - 2) {
            let _ = write!(out, "─");
        }
        let _ = write!(out, "╯\x1b[0m");
    }

    /* 4. Footer row F (H-2): left status, right dim hints. */
    let F = tui_footer_row();
    let _ = write!(out, "\x1b[{};1H\x1b[K", F);
    let hints = HINTS.with(|h| h.borrow().clone());
    let status = STATUS.with(|s| s.borrow().clone());
    let hints_w = if !hints.is_empty() {
        tui_disp_width(hints.as_ptr() as *const c_char, hints.len()) as usize
    } else {
        0
    };
    if !status.is_empty() {
        let avail = W - gutter - 1 - hints_w as c_int;
        if avail < 1 { /* no room */ } else {
        /* Bounded truncation: STATUS is a non-NUL-terminated Vec<u8>; the
         * strlen-walking C-ABI variant used to read past its allocation
         * and slice out of range (the 2026-09-09 abort). */
        let bl = crate::rt::tui::truncate_cells(&status, avail as usize);
        let _ = write!(out, "\x1b[{};{}H", F, 1 + gutter);
        let _ = out.write_all(&status[..bl]);
        }
    }
    if hints_w > 0 {
        let mut hcol = W - hints_w as c_int + 1;
        if hcol < 1 {
            hcol = 1;
        }
        let _ = write!(out, "\x1b[{};{}H\x1b[2m{}\x1b[0m", F, hcol, String::from_utf8_lossy(&hints));
    }
    if scrolled > 0 {
        let ind = format!(" ▲ {} scrolled · PgDn to live ", scrolled);
        let iw = tui_disp_width(ind.as_ptr() as *const c_char, ind.len()) as c_int;
        let avail = W - gutter - 1 - hints_w as c_int - 1;
        if avail > iw {
            let icol = W - hints_w as c_int - iw - 1;
            let _ = write!(out, "\x1b[{};{}H\x1b[2m{}\x1b[0m", F, icol.max(1+gutter), ind);
        }
    }

    /* 5. Footer row 2 (H-1): left context metric + right RAM metric — GUTTER-aligned with H-2. */
    let metric = METRIC.with(|m| m.borrow().clone());
    let _ = write!(out, "\x1b[{};1H\x1b[K", tui_footer2_row());
    let metric_l = METRIC_LEFT.with(|m| m.borrow().clone());
    if !metric_l.is_empty() {
        let avail_l = W - gutter - 1 - if !metric.is_empty() { tui_disp_width(metric.as_ptr() as *const c_char, metric.len()) as c_int + 1 } else { 0 };
        /* METRIC_LEFT is likewise non-NUL-terminated — same bounded call
         * as STATUS above (2026-09-09 abort was exactly this slice at
         * 141 past an 80-byte buffer on a wide terminal). */
        let bl_l = if avail_l > 0 { crate::rt::tui::truncate_cells(&metric_l, avail_l as usize) } else { 0 };
        if bl_l > 0 {
            let _ = write!(out, "\x1b[{};{}H\x1b[2m{}\x1b[0m", tui_footer2_row(), 1+gutter, String::from_utf8_lossy(&metric_l[..bl_l]));
        }
    }
    if !metric.is_empty() {
        let mw = tui_disp_width(metric.as_ptr() as *const c_char, metric.len()) as c_int;
        let mut mcol = W - mw + 1;
        if mcol < 1 {
            mcol = 1;
        }
        let _ = write!(out, "\x1b[{};{}H\x1b[2m{}\x1b[0m", tui_footer2_row(), mcol, String::from_utf8_lossy(&metric));
    }

    /* 6. Cursor: visible just AFTER the last typed char.
     * Row: "│· ·❯·<text>│" with GUTTER spaces. The cursor must be one cell past
     * the text (on the padding/closing-border side), not on the last glyph. */
    if READLINE_PROMISE.with(|p| p.get()).is_null() == false {
        let _ = write!(out, "\x1b[?25h");
        let cursor_col = 1 + crate::rt::tui::GUTTER + used;
        tui_place_cursor(cursor_col);
    } else {
        let _ = write!(out, "\x1b[?25l");
    }
    let _ = out.flush();
}

// ── JS: __chat_status / __chat_is_tty / TUI bridges ──────────────────

unsafe extern "C" fn js_chat_status(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut args = [ctx; 1];
    let _ = &mut args;
    if argc >= 1 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !s.is_null() {
            STATUS.with(|st| *st.borrow_mut() = CStr::from_ptr(s).to_bytes().to_vec());
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    if argc >= 2 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv.add(1));
        if !s.is_null() {
            HINTS.with(|h| *h.borrow_mut() = CStr::from_ptr(s).to_bytes().to_vec());
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    if argc >= 3 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv.add(2));
        if !s.is_null() {
            METRIC.with(|m| *m.borrow_mut() = CStr::from_ptr(s).to_bytes().to_vec());
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    if argc >= 4 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv.add(3));
        if !s.is_null() {
            METRIC_LEFT.with(|m| *m.borrow_mut() = CStr::from_ptr(s).to_bytes().to_vec());
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    /* Only re-render on a real terminal — piped output must stay clean. */
    if TTY_OPEN.with(|o| o.get()) != 0 && STDIN_IS_TTY.with(|t| t.get()) != 0 {
        tty_render_line_ui(ctx, 0);
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_chat_is_tty(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    // SAFETY: isatty(0) — trivial.
    qjs::sofuu_js_new_bool(ctx_unused(), if libc::isatty(0) == 1 { 1 } else { 0 })
}

// ctx-unused helper: js_chat_is_tty needs a ctx only for the bool; pass the
// registered one (functions above the registration have ctx available).
unsafe fn ctx_unused() -> *mut JSContext {
    CTX.with(|c| c.get())
}

unsafe extern "C" fn on_winch(_handle: *mut UvSignal, _signum: c_int) {
    if tui_active() == 0 {
        return;
    }
    tui_relayout();
    let tty_ctx = TTY_CTX.with(|c| c.get());
    if TTY_OPEN.with(|o| o.get()) != 0 && !tty_ctx.is_null() {
        tty_render_line_ui(tty_ctx, 0);
    }
    /* A resize wipes the overlay zone — ask the driver to re-draw it. */
    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 && !tty_ctx.is_null() {
        selector_key(tty_ctx, "redraw", None);
    }
}

/* SIGWINCH is POSIX-only; Windows console resize arrives differently. */
#[cfg(not(target_os = "windows"))]
unsafe fn ensure_winch_watch() {
    if WINCH_STARTED.with(|w| w.get()) != 0 {
        return;
    }
    WINCH_STARTED.with(|w| w.set(1));
    let winch = libc::malloc(uv::sofuu_uv_signal_size()) as *mut UvSignal;
    WINCH.with(|w| w.set(winch));
    uv::uv_signal_init(sofuu_loop_get(), winch);
    uv::uv_signal_start(winch, Some(on_winch), libc::SIGWINCH);
    uv::uv_unref(winch as *mut UvHandle); /* never keep the loop alive for this */
}

#[cfg(target_os = "windows")]
unsafe fn ensure_winch_watch() {}

extern "C" {
    fn tui_clear_gap();
    fn tui_discard_last();
}

unsafe extern "C" fn js_tui_on(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    /* SIGWINCH watcher is a signal source — gated in embedded mode (H2.2). */
    if !embed_config::is_embedded() || embed_config::signals_enabled() {
        ensure_winch_watch();
    }
    tui_enter();
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_off(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    tui_exit();
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_log(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !s.is_null() {
            tui_log(s);
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_log_last(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !s.is_null() {
            tui_log_last(s);
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_scroll(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut d: c_int = 0;
    if argc >= 1 {
        qjs::JS_ToInt32(ctx, &mut d, *argv);
    }
    tui_scroll(d);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_set_header(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc >= 1 {
        let s = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !s.is_null() {
            tui_set_header(s);
            qjs::sofuu_js_free_cstring(ctx, s);
        }
    }
    qjs::sofuu_js_undefined()
}

// ── selector bridges ─────────────────────────────────────────────────

unsafe fn stdin_rl_stream() -> *mut UvStream {
    if STDIN_IS_TTY.with(|t| t.get()) != 0 {
        TTY_STDIN.with(|t| t.get()) as *mut UvStream
    } else {
        STDIN_RL_PIPE.with(|p| p.get()) as *mut UvStream
    }
}

thread_local! {
    static STDIN_RL_PIPE: Cell<*mut UvPipe> = const { Cell::new(ptr::null_mut()) };
}

unsafe extern "C" fn js_selector_open(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    if !SELECTOR_PROMISE.with(|p| p.get()).is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"selector already open".as_ptr());
    }

    ensure_tty_open(ctx);
    uv::uv_read_start(stdin_rl_stream(), Some(tty_read_alloc_cb), Some(tty_read_cb));
    uv::uv_ref(stdin_rl_stream() as *mut UvHandle);

    SELECTOR_ACTIVE.with(|a| a.set(1));
    SEL_TOP.with(|t| t.set(-1));
    SEL_ROWS.with(|r| r.set(0));
    if tui_active() != 0 {
        tui_discard_last();
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "\x1b[?25l");
        let _ = out.flush();
    }
    let mut out_p: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out_p);
    SELECTOR_PROMISE.with(|p| p.set(out_p));
    promise
}

unsafe extern "C" fn js_selector_draw(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    use std::io::Write;
    if SELECTOR_ACTIVE.with(|a| a.get()) == 0 || tui_active() == 0 || argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let s_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if s_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let s = CStr::from_ptr(s_ptr).to_bytes();
    let mut out = std::io::stdout().lock();

    let W = tui_width();
    let H = tui_height();
    let zone_bottom = H - 3;
    let cap = zone_bottom; /* rows 1..H-3 are available */

    let mut lines = 1usize;
    for &b in s {
        if b == b'\n' {
            lines += 1;
        }
    }
    if !s.is_empty() && s[s.len() - 1] == b'\n' {
        lines -= 1;
    }
    if lines > cap as usize {
        lines = cap as usize;
    }

    let top = zone_bottom - lines as c_int + 1;
    let sel_top = SEL_TOP.with(|t| t.get());
    let sel_rows = SEL_ROWS.with(|r| r.get());
    if sel_top >= 0 {
        let last_old = sel_top + sel_rows - 1;
        let new_bottom = top + lines as c_int - 1;
        // Clear the entire previous picker rectangle first, then the
        // new one fully overwrites. This handles filter narrowing,
        // expanding, and H/W changes without stranded rows.
        for r in sel_top..=last_old {
            let _ = write!(out, "\x1b[{};1H\x1b[K", r);
        }
        // Bottom-tail when H shrank: old zone_bottom may be below new one,
        // but that tail was part of the old rect already (cleared above).
        // Keep top-gap handling via the full clear — no partial gap needed.
        let _ = new_bottom;
    }
    let mut p: &[u8] = s;
    for i in 0..lines {
        let nl = p.iter().position(|&b| b == b'\n');
        let len = nl.unwrap_or(p.len());
        let tmp = &p[..len];
        let tcs = CString::new(tmp).unwrap_or_default();
        let bl = tui_truncate_cells(tcs.as_ptr(), W);
        let _ = write!(out, "\x1b[{};1H\x1b[K", top + i as c_int);
        let _ = out.write_all(&tmp[..bl.min(tmp.len())]);
        p = match nl {
            None => &p[p.len()..],
            Some(i) => &p[i + 1..],
        };
    }
    SEL_TOP.with(|t| t.set(top));
    SEL_ROWS.with(|r| r.set(lines as c_int));
    let _ = out.flush();
    qjs::sofuu_js_free_cstring(ctx, s_ptr);
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_selector_close(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut res: Option<Vec<u8>> = None;
    if argc >= 1 {
        let rp = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !rp.is_null() {
            res = Some(CStr::from_ptr(rp).to_bytes().to_vec());
            qjs::sofuu_js_free_cstring(ctx, rp);
        }
    }

    // Capture previous picker rect before clearing state, so we can erase
    // the entire overlay zone (not just 1..h-8). Gap rows h-7,h-6 live
    // outside the conversation viewport and would otherwise ghost.
    let prev_top = SEL_TOP.with(|t| t.get());
    let prev_rows = SEL_ROWS.with(|r| r.get());
    SELECTOR_ACTIVE.with(|a| a.set(0));
    if tui_active() != 0 {
        if prev_top >= 0 && prev_rows > 0 {
            use std::io::Write;
            let mut out2 = std::io::stdout().lock();
            for r in prev_top..prev_top + prev_rows {
                let _ = write!(out2, "\x1b[{};1H\x1b[K", r);
            }
            let _ = out2.flush();
            tui_clear_gap();
        }
        tui_render_conversation();
        tty_render_line_ui(ctx, 0);
    }
    SEL_TOP.with(|t| t.set(-1));
    SEL_ROWS.with(|r| r.set(0));
    if TTY_OPEN.with(|o| o.get()) != 0 {
        uv::uv_unref(stdin_rl_stream() as *mut UvHandle); /* open() ref'd */
    }
    let p = SELECTOR_PROMISE.with(|s| s.replace(ptr::null_mut()));
    if !p.is_null() {
        let r = match &res {
            Some(r) if !r.is_empty() => {
                let c = CString::new(r.clone()).unwrap_or_default();
                qjs::sofuu_js_new_string(ctx, c.as_ptr())
            }
            _ => qjs::sofuu_js_new_string(ctx, c"cancel".as_ptr()),
        };
        sofuu_promise_resolve(p, r);
        qjs::sofuu_js_free_value(ctx, r);
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tui_width_js(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    // SAFETY: ctx from the registered context — see js_tui_height_js.
    qjs::sofuu_js_new_int32(ctx_unused(), tui_width())
}

unsafe extern "C" fn js_tui_height_js(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    qjs::sofuu_js_new_int32(ctx_unused(), tui_height())
}

// ── history navigation ───────────────────────────────────────────────

unsafe fn tty_history_push() {
    let line = LINE_BUF.with(|b| b.borrow().clone());
    if line.is_empty() {
        return;
    }
    HISTORY.with(|h| {
        let mut h = h.borrow_mut();
        let last = h.back().cloned().unwrap_or_default();
        if line == last {
            return;
        }
        if h.len() >= HIST_MAX {
            h.pop_front();
        }
        let mut bytes = line;
        bytes.truncate(HIST_LEN - 1);
        h.push_back(bytes);
    });
    HIST_POS.with(|p| p.set(-1));
}

unsafe fn tty_history_nav(ctx: *mut JSContext, dir: c_int) {
    let count = HISTORY.with(|h| h.borrow().len()) as c_int;
    if count == 0 {
        return;
    }
    let pos = HIST_POS.with(|p| p.get()) as c_int;
    let target: c_int = if pos == -1 {
        /* Fresh line: Up → newest entry; Down → stay fresh. */
        if dir < 0 {
            count - 1
        } else {
            -1
        }
    } else {
        let mut t = pos + dir; /* c_int + c_int */
        if t >= count {
            t = -1; /* Down past newest → fresh */
        }
        if t < -1 {
            t = 0;
        }
        t
    };
    HIST_POS.with(|p| p.set(target as c_long));
    LINE_BUF.with(|b| {
        let mut b = b.borrow_mut();
        b.clear();
        if target != -1 {
            if let Some(entry) = HISTORY.with(|h| h.borrow().get(target as usize).cloned()) {
                b.extend_from_slice(&entry);
            }
        }
    });
    tty_render_line_ui(ctx, 0);
}

// ── ESC-sequence state machine ───────────────────────────────────────

unsafe extern "C" fn esc_timeout_cb(_t: *mut UvTimer) {
    let state = ESC_STATE.with(|s| s.get());
    if state == 0 {
        return;
    }
    let was_bare = state == 1; /* only ESC seen so far */
    ESC_STATE.with(|s| s.set(0));
    if !was_bare {
        return; /* partial sequence: drop it */
    }
    let ctx = TTY_CTX.with(|c| c.get());
    if ctx.is_null() {
        return;
    }
    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
        selector_key(ctx, "esc", None);
    } else if !READLINE_PROMISE.with(|p| p.get()).is_null() {
        if LINE_BUF.with(|b| b.borrow().len()) > 0 {
            LINE_BUF.with(|b| b.borrow_mut().clear());
            tty_render_line_ui(ctx, 0);
        }
    } else {
        call_ui_hook(ctx, "__on_esc");
    }
}

unsafe fn esc_timer_kick() {
    let init = ESC_TIMER_INIT.with(|i| i.get());
    if init == 0 {
        let t = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
        ESC_TIMER.with(|e| e.set(t));
        uv::uv_timer_init(sofuu_loop_get(), t);
        uv::uv_unref(t as *mut UvHandle);
        ESC_TIMER_INIT.with(|i| i.set(1));
    }
    let t = ESC_TIMER.with(|e| e.get());
    uv::uv_timer_start(t, Some(esc_timeout_cb), 250, 0);
    /* F-2: armed per-ctx (the registered engine) — a multi-engine teardown
     * must close it; orphan None, static storage like the cleanup path. */
    let owner = CTX.with(|c| c.get());
    if !owner.is_null() {
        crate::rt::event_loop::track_handle(owner, t as *mut UvHandle, None);
    }
}

unsafe fn esc_timer_cancel() {
    if ESC_TIMER_INIT.with(|i| i.get()) != 0 {
        uv::uv_timer_stop(ESC_TIMER.with(|e| e.get()));
    }
}

// ── bracketed-paste body accumulation ─────────────────────────────────

/// Append one literal byte to the line buffer while a bracketed paste is
/// open. CR normalizes to LF (multi-line editor convention), the buffer cap
/// is enforced with the same truncation flag as typed input, and the live
/// region is NOT redrawn per byte — a large paste can be tens of thousands
/// of bytes and the renderer would turn the paste into an O(n) redraw
/// storm; one redraw happens when ESC[201~ closes the paste.
unsafe fn paste_push(c: u8) {
    let b = if c == b'\r' { b'\n' } else { c };
    let len = LINE_BUF.with(|l| l.borrow().len());
    if len < LINE_BUF_CAP - 1 {
        LINE_BUF.with(|l| l.borrow_mut().push(b));
    } else {
        LINE_TRUNCATED.with(|t| t.set(1));
    }
}

// ── app-managed text selection (mouse drag + Ctrl-K copy) ────────────
// The TUI owns the alternate screen with mouse reporting on, so the
// terminal's native click-drag selection never happens — selection must be
// app-managed: left-drag over conversation rows highlights them, Ctrl-K
// yanks the selected rows to the system clipboard (pbcopy on macOS, OSC 52
// otherwise). Row math mirrors render_rows: viewport height, G_SCROLL
// clamp, buffer index = start + (row - top).

/// Repaint the conversation when the selection changed (screen rows are
/// 1-based conversation rows; anything else clears it).
unsafe fn sel_repaint(ctx: *mut JSContext) {
    if ctx.is_null() {
        return;
    }
    if tui_active() != 0 {
        crate::rt::tui::tui_render_conversation();
    }
    tty_render_line_ui(ctx, 0);
}

/// Current selection as (anchor, end) screen rows; (-1, -1) = none.
/// Read by rt::tui's renderer so the highlight survives every repaint.
pub(crate) unsafe fn sel_range() -> (c_int, c_int) {
    (SEL_ANCHOR.with(|s| s.get()), SEL_END.with(|s| s.get()))
}

/// Clamp a raw SGR row (already offset by 1 for the GUTTER) to the
/// conversation viewport; returns -1 when outside it.
unsafe fn sel_clamp_row(y: c_int) -> c_int {
    let top = tui_conv_top();
    let bottom = tui_conv_bottom();
    if y < top {
        return -1;
    }
    if y > bottom {
        return -1;
    }
    y
}

/// Copy the selected rows (inclusive) to the system clipboard. SGR mouse
/// coordinates are 1-based and the render adds a GUTTER, so the first
/// selectable column is 1 + GUTTER; select the FULL row regardless of the
/// click column — line-level selection is unambiguous on wrapped rows.
pub(crate) unsafe fn sel_copy(ctx: *mut JSContext) -> bool {
    if ctx.is_null() {
        return false;
    }
    let a = SEL_ANCHOR.with(|s| s.get());
    let b = SEL_END.with(|s| s.get());
    if a < 0 || b < 0 {
        return false;
    }
    let (r0, r1) = if a <= b { (a, b) } else { (b, a) };
    let p = crate::rt::tui::tui_selected_rows(r0, r1);
    if p.is_null() {
        return false;
    }
    /* CStr::from_ptr needs a NUL-terminated buffer — tui_selected_rows
     * always NUL-terminates (CString::new). Empty result frees cleanly. */
    let bytes = std::ffi::CStr::from_ptr(p).to_bytes().to_vec();
    libc::free(p as *mut c_void);
    if bytes.is_empty() {
        return false;
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    /* Rows may embed SGR color codes (styled output) — strip them so the
     * clipboard gets clean text. Plain rows pass through untouched. */
    let plain = strip_ansi(&text);
    /* SOFUU_SEL_DEBUG=<path>: dump the exact bytes that go to the
     * clipboard — the e2e harness forensics hook (never set normally). */
    if let Ok(p) = std::env::var("SOFUU_SEL_DEBUG") {
        let _ = std::fs::write(&p, plain.as_bytes());
        /* P3 (AUDIT-2026-09-07): this dump is clipboard exfiltration when
         * the env var is set — announce it so it can never fire without a
         * visible trace. The dump file itself stays byte-identical (the
         * e2e harness compares it against the clipboard). */
        eprintln!("[sofuu] SOFUU_SEL_DEBUG: clipboard copy dumped to {p}");
    }
    let n_lines = (r1 - r0 + 1) as usize;
    let ok = clipboard_write(&plain);
    if ok {
        sel_show_notice(ctx, n_lines);
    }
    ok
}

/// Show a transient "Copied N lines" notice on the phase row (h-6, blank
/// gap). A uv timer clears it; any repaint of that row replaces it too.
unsafe fn sel_show_notice(ctx: *mut JSContext, n_lines: usize) {
    if ctx.is_null() || tui_active() == 0 {
        return;
    }
    let msg = format!("Copied {n_lines} line{} to clipboard", if n_lines == 1 { "" } else { "s" });
    let row = tui_phase_row();
    {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "\x1b[{};1H\x1b[K\x1b[2m  {}\x1b[0m", row, msg);
        let _ = out.flush();
    }
    /* remember + arm the clearing timer */
    SEL_NOTICE_MS.with(|s| s.set(now_ms()));
    notice_timer_kick();
}

/// Remove ANSI/CSI escape sequences from a byte string (SGR colors etc.).
pub(crate) fn strip_ansi(s: &str) -> String {
    let b = s.as_bytes();
    let n = b.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if b[i] == 0x1b {
            // ESC [ … final byte in 0x40..=0x7E; also cover 2-char escapes.
            i += 1;
            if i < n && b[i] == b'[' {
                i += 1;
                while i < n && !(0x40..=0x7E).contains(&b[i]) {
                    i += 1;
                }
                i += 1; /* consume the final byte */
            } else if i < n {
                i += 1;
            }
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Monotonic milliseconds (uv loop time — the cached coarse clock).
unsafe fn now_ms() -> i64 {
    uv::uv_now(sofuu_loop_get())
}

/// Write text to the system clipboard. macOS: pbcopy via popen (fonts-
/// free, always present). Other hosts: OSC 52 — terminals that understand
/// it (iTerm2, WezTerm, kitty, most Linux terms) write the clipboard
/// themselves; tmux forwards it if set-clipboard is on.
pub(crate) unsafe fn clipboard_write(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        let mode = CString::new("w").unwrap();
        let cmd = CString::new("pbcopy").unwrap();
        let f = libc::popen(cmd.as_ptr(), mode.as_ptr());
        if !f.is_null() {
            let mut off: usize = 0;
            let bytes = text.as_bytes();
            while off < bytes.len() {
                let n = libc::fwrite(bytes.as_ptr().add(off) as *const c_void, 1,
                                     bytes.len() - off, f);
                if n == 0 {
                    break;
                }
                off += n;
            }
            let _ = libc::fflush(f);
            libc::pclose(f);
            return true;
        }
        return false;
    }
    #[cfg(not(target_os = "macos"))]
    {
        /* OSC 52: ESC ] 52 ; c ; <base64> BEL. 100k payload cap — terminal
         * emulators commonly reject larger writes, and a copy is exactly
         * the kind of paste that then lands somewhere unwanted. */
        const CAP: usize = 100_000;
        let bytes: &[u8] = if text.len() > CAP { &text.as_bytes()[..CAP] } else { text.as_bytes() };
        let mut out = String::with_capacity(bytes.len() / 3 * 4 + 32);
        out.push_str("\x1b]52;c;");
        out.push_str(&b64_encode(bytes));
        out.push('\x07');
        use std::io::Write;
        let mut o = std::io::stdout().lock();
        let _ = o.write_all(out.as_bytes());
        let _ = o.flush();
        true
    }
}

/// Standard base64 (RFC 4648) encode — small, allocation-simple, no
/// external crates. OSC 52 needs it; kept next to its only caller.
pub(crate) fn b64_encode(data: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | data[i + 2] as u32;
        out.push(TBL[(n >> 18) as usize & 63] as char);
        out.push(TBL[(n >> 12) as usize & 63] as char);
        out.push(TBL[(n >> 6) as usize & 63] as char);
        out.push(TBL[n as usize & 63] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let n = (data[i] as u32) << 16;
        out.push(TBL[(n >> 18) as usize & 63] as char);
        out.push(TBL[(n >> 12) as usize & 63] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
        out.push(TBL[(n >> 18) as usize & 63] as char);
        out.push(TBL[(n >> 12) as usize & 63] as char);
        out.push(TBL[(n >> 6) as usize & 63] as char);
        out.push('=');
    }
    out
}

// ── copy-notice timer (uv one-shot) ───────────────────────────────────

unsafe extern "C" fn sel_notice_timer_cb(_t: *mut UvTimer) {
    /* Only clear when no NEWER notice arrived since this timer was armed
     * (a rapid second copy re-arms the timer; the first fire must not wipe
     * the newer message). */
    let armed = SEL_NOTICE_MS.with(|s| s.replace(0));
    if armed == 0 {
        return;
    }
    let _ = armed;
    if tui_active() == 0 {
        return;
    }
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = write!(out, "\x1b[{};1H\x1b[K", tui_phase_row());
    let _ = out.flush();
}

thread_local! {
    static SEL_NOTICE_TIMER: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
}

unsafe fn notice_timer_kick() {
    let mut t = SEL_NOTICE_TIMER.with(|t| t.get());
    if t.is_null() {
        t = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
        uv::uv_timer_init(sofuu_loop_get(), t);
        uv::uv_unref(t as *mut UvHandle);
        SEL_NOTICE_TIMER.with(|c| c.set(t));
    }
    uv::uv_timer_stop(t);
    uv::uv_timer_start(t, Some(sel_notice_timer_cb), 1800, 0);
    /* F-2: armed per-ctx (the registered engine) — a multi-engine teardown
     * must close it; orphan None, static storage like the cleanup path. */
    let owner = CTX.with(|c| c.get());
    if !owner.is_null() {
        crate::rt::event_loop::track_handle(owner, t as *mut UvHandle, None);
    }
}

// ── the TTY read callback (raw keystrokes → line buffer) ─────────────

/// Is the current line an in-progress slash command with suggestions?
unsafe fn completion_active(ctx: *mut JSContext) -> bool {
    let line = LINE_BUF.with(|b| b.borrow().clone());
    if line.is_empty() || line[0] != b'/' || line.contains(&b' ') {
        return false;
    }
    let mut n = 0usize;
    let m = tty_complete_matches(ctx, &line, &mut n);
    /* A single exact match (e.g. "/rlm") still counts as active so its
     * description/options render and Tab can accept it. */
    let active = m.is_some() && n >= 1;
    if !active {
        COMPLETE_SEL.with(|s| s.set(-1));
    }
    active
}

/// Move the completion highlight (dir = -1 up / +1 down). Clamps to the
/// visible suggestion count. The highlight is applied on the next render.
unsafe fn completion_nav(ctx: *mut JSContext, dir: c_int) {
    if !completion_active(ctx) {
        return;
    }
    let line = LINE_BUF.with(|b| b.borrow().clone());
    let mut n = 0usize;
    let _ = tty_complete_matches(ctx, &line, &mut n);
    let n = n as c_int;
    if n <= 0 {
        return;
    }
    COMPLETE_SEL.with(|s| {
        let cur = s.get();
        let next = if cur < 0 {
            /* fresh: Up → last suggestion, Down → first */
            if dir < 0 { n - 1 } else { 0 }
        } else {
            let mut t = cur + dir;
            if t < 0 { t = n - 1; }
            if t >= n { t = 0; }
            t
        };
        s.set(next);
    });
    tty_render_line_ui(ctx, 0);
}

/// Accept the highlighted completion: replace the line with the selected
/// suggestion and SUBMIT it immediately — the driver's slash dispatch then
/// runs the command (config commands open their picker panel, instant
/// commands like /compact /clear execute right away).
unsafe fn completion_accept(ctx: *mut JSContext) {
    let sel = COMPLETE_SEL.with(|s| s.get());
    if sel < 0 {
        return;
    }
    let line = LINE_BUF.with(|b| b.borrow().clone());
    let mut n = 0usize;
    let m = tty_complete_matches(ctx, &line, &mut n);
    let Some(m) = m else { return };
    let m = m.to_bytes();
    let Some(choice) = nth_match(m, sel as usize) else { return };
    let mut new_line = line.clone();
    new_line.clear();
    new_line.extend_from_slice(choice);
    LINE_BUF.with(|b| *b.borrow_mut() = new_line);
    COMPLETE_SEL.with(|s| s.set(-1));
    tty_render_line_ui(ctx, 0);
    rl_submit(ctx);
}

/// Deliver the current line to the waiting readline promise (or queue it),
/// exactly like the Enter path — this is what makes a command RUN.
unsafe fn rl_submit(ctx: *mut JSContext) {
    use std::io::Write;
    let stream = stdin_rl_stream();
    if !stream.is_null() {
        uv::uv_unref(stream as *mut UvHandle);
    }
    let line = LINE_BUF.with(|b| b.borrow().clone());
    if !line.is_empty() {
        chat_history_append(&line);
    }
    let p = READLINE_PROMISE.with(|r| r.replace(ptr::null_mut()));
    if !p.is_null() {
        if tui_active() != 0 {
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "\x1b[?25l");
            let _ = out.flush();
        }
        /* proc-10: JS_NewStringLen — CString::new fails on an interior NUL
         * (a pasted `\0` byte truncated the delivered line to ""). */
        let jline = qjs::JS_NewStringLen(ctx, line.as_ptr() as *const c_char, line.len());
        sofuu_promise_resolve(p, jline);
        qjs::sofuu_js_free_value(ctx, jline);
    } else {
        rl_queue_push(&line);
    }
    LINE_BUF.with(|b| b.borrow_mut().clear());
    COMPLETE_SEL.with(|s| s.set(-1));
    if STDIN_IS_TTY.with(|t| t.get()) != 0 {
        tty_render_line_ui(ctx, 0);
    }
}

/// tty_esc_step — ESC-sequence state machine, ported verbatim.
unsafe fn tty_esc_step(c: u8, ctx: *mut JSContext) -> c_int {
    let state = ESC_STATE.with(|s| s.get());
    match state {
        1 => {
            /* after bare ESC */
            if c == b'[' {
                ESC_STATE.with(|s| s.set(2));
                return 2;
            }
            if c == b'\r' || c == b'\n' {
                /* Alt+Enter → newline (ptys may deliver CR as LF) */
                if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                    return 0; /* no multi-line in pickers */
                }
                if LINE_BUF.with(|b| b.borrow().len()) < LINE_BUF_CAP - 1 {
                    LINE_BUF.with(|b| b.borrow_mut().push(b'\n'));
                    tty_render_line_ui(ctx, 0);
                }
                return 0;
            }
            if c == 0x1b {
                ESC_STATE.with(|s| s.set(1));
                return 1; /* double ESC: still pending */
            }
            /* bare ESC → cancel picker | clear the line | stop streaming */
            if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                selector_key(ctx, "esc", None);
            } else if !READLINE_PROMISE.with(|p| p.get()).is_null() {
                if LINE_BUF.with(|b| b.borrow().len()) > 0 {
                    LINE_BUF.with(|b| b.borrow_mut().clear());
                    tty_render_line_ui(ctx, 0);
                }
            } else {
                call_ui_hook(ctx, "__on_esc");
            }
            0
        }
        2 => {
            /* ESC [ */
            match c {
                b'A' => {
                    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                        selector_key(ctx, "up", None);
                    } else if FOCUS_WINDOW.with(|f| f.get()) != 0 || crate::rt::tui::tui_scroll_pos() > 0 {
                        tui_scroll((tui_conv_bottom() - tui_conv_top() + 1) / 2);
                        tty_render_line_ui(ctx, 0);
                    } else if completion_active(ctx) {
                        completion_nav(ctx, -1);
                    } else {
                        tty_history_nav(ctx, -1);
                    }
                    0
                }
                b'B' => {
                    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                        selector_key(ctx, "down", None);
                    } else if FOCUS_WINDOW.with(|f| f.get()) != 0 || crate::rt::tui::tui_scroll_pos() > 0 {
                        tui_scroll(-((tui_conv_bottom() - tui_conv_top() + 1) / 2));
                        if crate::rt::tui::tui_scroll_pos() == 0 {
                            FOCUS_WINDOW.with(|f| f.set(0));
                        }
                        tty_render_line_ui(ctx, 0);
                    } else if completion_active(ctx) {
                        completion_nav(ctx, 1);
                    } else {
                        tty_history_nav(ctx, 1);
                    }
                    0
                }
                b'C' => {
                    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                        selector_key(ctx, "right", None);
                    }
                    0
                }
                b'D' => {
                    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                        selector_key(ctx, "left", None);
                    }
                    0
                }
                b'Z' => {
                    if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                        selector_key(ctx, "shift-tab", None);
                    }
                    0
                }
                b'1' => 3, /* Shift+Enter? */
                b'5' => 7, /* PageUp? */
                b'6' => 8, /* PageDown? */
                b'<' => {
                    MOUSE_BUF.with(|b| b.borrow_mut().clear());
                    9
                }, /* SGR mouse ESC[< */
                b'M' => {
                    // SGR mouse reports are longer (ESC[<...), we handle them
                    // in state 9/10. Legacy X10 is ESC[M + 3 bytes — treat as
                    // mouse: 32=click, 64=wheel up, 65=wheel down. We only
                    // need scroll direction.
                    // For simplicity, swallow the 3-byte payload and scroll.
                    // Mark that we need 3 more bytes; use a tiny counter via
                    // ESC_STATE value 11.
                    ESC_STATE.with(|s| s.set(11));
                    return 11;
                }
                b'2' => 20, /* bracketed paste: ESC[200~ opens, ESC[201~ closes */
                _ => 0,   /* other: drop */
            }
        }
        /* ── Bracketed paste (DECSET 2004) ──────────────────────────────
         * A paste is wrapped by the terminal in ESC[200~ … ESC[201~ (we
         * request 2004 when the TTY opens). While the body is open EVERY
         * byte — including CR/LF — is appended to the line buffer as literal
         * text: newlines inside a paste never reach the Enter-submit path,
         * so a large multi-line paste lands as ONE prompt that waits for a
         * real Enter.
         * Idle side: ESC [ '2'(20) '0'(21) '0'(30) '~' → body opens | '1'(31)
         * '~' → stray end marker, dropped. Body side: 23 →ESC(24) '['(25)
         * '2'(26) '0'(27) '1'(28) '~' → closes. Any body ESC-sequence that
         * is NOT the end marker is pushed back as literal text (pasted
         * terminal output can legitimately contain ANSI colors). */
        20 => {
            if c == b'0' {
                21
            } else {
                0
            }
        }
        21 => {
            if c == b'0' {
                30
            } else if c == b'1' {
                31
            } else {
                0
            }
        }
        30 => {
            if c == b'~' {
                /* ESC[200~ — paste body opens. */
                23
            } else {
                0
            }
        }
        31 => {
            /* ESC[201~ with no open paste — inert. */
            0
        }
        23 => {
            if c == 0x03 {
                /* Ctrl-C inside a paste: abort paste mode (a paste whose end
                 * marker never arrives would otherwise eat every keystroke);
                 * a second Ctrl-C takes the normal quit path. The already
                 * pasted text stays in the buffer. */
                0
            } else if c == 0x1b {
                24
            } else {
                paste_push(c);
                23
            }
        }
        24 => {
            if c == b'[' {
                25
            } else {
                paste_push(0x1b);
                paste_push(c);
                23
            }
        }
        25 => {
            if c == b'2' {
                26
            } else {
                paste_push(0x1b);
                paste_push(b'[');
                paste_push(c);
                23
            }
        }
        26 => {
            /* End-marker candidate ESC[2…: the next char splits — '0' walks
             * toward ESC[201~ (state 27), anything else is literal body text. */
            if c == b'0' {
                27
            } else if c == b'1' {
                /* ESC[21~ is not a real marker; treat as literal text. */
                paste_push(0x1b);
                paste_push(b'[');
                paste_push(b'2');
                paste_push(b'1');
                paste_push(c);
                23
            } else {
                paste_push(0x1b);
                paste_push(b'[');
                paste_push(b'2');
                paste_push(c);
                23
            }
        }
        27 => {
            /* saw ESC[20 — '1' → state 28 (end candidate ESC[201…); a '0'
             * here means ESC[200~ INSIDE the paste (a nested open marker,
             * which terminals never send): keep it as literal text. */
            if c == b'1' {
                28
            } else {
                paste_push(0x1b);
                paste_push(b'[');
                paste_push(b'2');
                paste_push(b'0');
                paste_push(c);
                23
            }
        }
        28 => {
            if c == b'~' {
                /* ESC[201~ — paste closes: one redraw shows the tail. */
                if !ctx.is_null() {
                    tty_render_line_ui(ctx, 0);
                }
                0
            } else {
                paste_push(0x1b);
                paste_push(b'[');
                paste_push(b'2');
                paste_push(b'0');
                paste_push(b'1');
                paste_push(c);
                23
            }
        }
        9 => {
            // SGR mouse: ESC [ < btn ; x ; y M/m . Accumulate in MOUSE_BUF.
            MOUSE_BUF.with(|b| b.borrow_mut().push(c));
            if c == b'M' || c == b'm' {
                let pressed = c == b'M';
                let buf = MOUSE_BUF.with(|b| b.borrow().clone());
                MOUSE_BUF.with(|b| b.borrow_mut().clear());
                ESC_STATE.with(|s| s.set(0));
                let s = String::from_utf8_lossy(&buf);
                let s = s.trim_end_matches(|ch| ch == 'M' || ch == 'm');
                // s like "64;80;10" or "0;80;10"
                let parts: Vec<&str> = s.split(';').collect();
                if parts.len() >= 3 {
                    let btn: i32 = parts[0].parse().unwrap_or(0);
                    let y: i32 = parts[2].parse().unwrap_or(0);
                    if !pressed {
                        // mouse up — ends a drag; the selection is already final
                    } else if btn == 64 {
                        // wheel up: selection rows map to buffer positions
                        // through G_SCROLL — scrolling would silently move
                        // them, so drop the selection on scroll.
                        SEL_ANCHOR.with(|a| a.set(-1));
                        SEL_END.with(|e| e.set(-1));
                        FOCUS_WINDOW.with(|f| f.set(1));
                        tui_scroll(3);
                        sel_repaint(ctx);
                    } else if btn == 65 {
                        SEL_ANCHOR.with(|a| a.set(-1));
                        SEL_END.with(|e| e.set(-1));
                        tui_scroll(-3);
                        if crate::rt::tui::tui_scroll_pos() == 0 { FOCUS_WINDOW.with(|f| f.set(0)); }
                        sel_repaint(ctx);
                    } else if btn == 0 {
                        // left press — focus by row AND seed a selection:
                        // drag then extends SEL_END (btn 32).
                        if y >= tui_conv_top() && y <= tui_conv_bottom() {
                            FOCUS_WINDOW.with(|f| f.set(1));
                            SEL_ANCHOR.with(|a| a.set(y));
                            SEL_END.with(|e| e.set(y));
                            sel_repaint(ctx);
                        } else {
                            // press outside the conversation clears any
                            // selection and keeps the old focus logic
                            SEL_ANCHOR.with(|a| a.set(-1));
                            SEL_END.with(|e| e.set(-1));
                            if y >= tui_box_top() && y <= tui_box_top() + 2 {
                                FOCUS_WINDOW.with(|f| f.set(0));
                            }
                            sel_repaint(ctx);
                        }
                    } else if btn == 32 {
                        // left drag — extend the selection if one is open
                        let a = SEL_ANCHOR.with(|s| s.get());
                        if a >= 0 {
                            let row = sel_clamp_row(y);
                            if row >= 0 {
                                SEL_END.with(|e| e.set(row));
                                sel_repaint(ctx);
                            }
                        }
                    }
                }
                return 0;
            }
            return 9;
        }
        11 => {
            let btn = c;
            if btn == 64 || btn == 96 {
                FOCUS_WINDOW.with(|f| f.set(1));
                tui_scroll(3);
                tty_render_line_ui(ctx, 0);
            } else if btn == 65 || btn == 97 {
                tui_scroll(-3);
                if crate::rt::tui::tui_scroll_pos() == 0 { FOCUS_WINDOW.with(|f| f.set(0)); }
                tty_render_line_ui(ctx, 0);
            }
            ESC_STATE.with(|s| s.set(12));
            return 12;
        }
        12 | 13 => {
            if ESC_STATE.with(|s| s.get()) == 13 {
                let y = c as i32 - 32;
                if y >= tui_conv_top() && y <= tui_conv_bottom() {
                    FOCUS_WINDOW.with(|f| f.set(1));
                    tty_render_line_ui(ctx, 0);
                } else if y >= tui_box_top() && y <= tui_box_top() + 2 {
                    FOCUS_WINDOW.with(|f| f.set(0));
                    tty_render_line_ui(ctx, 0);
                }
                ESC_STATE.with(|s| s.set(0));
            } else {
                ESC_STATE.with(|s| s.set(13));
            }
            return ESC_STATE.with(|s| s.get());
        }
        3 => {
            if c == b'3' {
                4
            } else {
                0
            }
        }
        4 => {
            if c == b';' {
                5
            } else {
                0
            }
        }
        5 => {
            if c == b'2' {
                6
            } else {
                0
            }
        }
        6 => {
            if c == b'u' {
                /* Shift+Enter → newline */
                if LINE_BUF.with(|b| b.borrow().len()) < LINE_BUF_CAP - 1 {
                    LINE_BUF.with(|b| b.borrow_mut().push(b'\n'));
                    tty_render_line_ui(ctx, 0);
                }
            }
            0
        }
        7 => {
            if c == b'~' {
                if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                    selector_key(ctx, "pageup", None);
                } else {
                    FOCUS_WINDOW.with(|f| f.set(1));
                    tui_scroll((tui_conv_bottom() - tui_conv_top() + 1) / 2);
                    tty_render_line_ui(ctx, 0);
                }
            }
            0
        }
        8 => {
            if c == b'~' {
                if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
                    selector_key(ctx, "pagedown", None);
                } else {
                    tui_scroll(-((tui_conv_bottom() - tui_conv_top() + 1) / 2));
                    if crate::rt::tui::tui_scroll_pos() == 0 {
                        FOCUS_WINDOW.with(|f| f.set(0));
                    }
                    tty_render_line_ui(ctx, 0);
                }
            }
            0
        }
        _ => 0,
    }
}

unsafe extern "C" fn tty_read_alloc_cb(
    _h: *mut UvHandle,
    suggested: usize,
    buf: *mut uv::UvBuf,
) {
    (*buf).base = libc::malloc(suggested) as *mut c_char;
    (*buf).len = if (*buf).base.is_null() { 0 } else { suggested };
}

/// Deliver EOF-style cancellation to a pending readline (shared by the
/// EOF / Ctrl-C / Ctrl-D paths).
unsafe fn rl_resolve_null(stream: *mut UvStream) {
    let p = READLINE_PROMISE.with(|r| r.replace(ptr::null_mut()));
    if p.is_null() {
        return;
    }
    uv::uv_read_stop(stream);
    uv::uv_unref(stream as *mut UvHandle);
    if tui_active() != 0 {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "\x1b[?25l");
        let _ = out.flush();
    }
    // SAFETY: JS_NULL is an immediate value; the resolve dups as needed.
    sofuu_promise_resolve(p, qjs::sofuu_js_null());
}

unsafe extern "C" fn tty_read_cb(stream: *mut UvStream, nread: isize, buf: *const uv::UvBuf) {
    let ctx = TTY_CTX.with(|c| c.get());

    if nread == uv::UV_EOF as isize || nread == 0 {
        if !(*buf).base.is_null() {
            libc::free((*buf).base as *mut c_void);
        }
        /* EOF while a picker is open: unwind it, the exit path follows. */
        selector_resolve_cancel(ctx);
        /* Ctrl-D: erase the live region so the box doesn't linger. */
        let ui_lines = UI_LINES.with(|l| l.get());
        if STDIN_IS_TTY.with(|t| t.get()) != 0 && ui_lines > 0 {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "\x1b[{}A", ui_lines - 2);
            let _ = write!(out, "\x1b[J");
            let _ = out.flush();
            UI_LINES.with(|l| l.set(0));
        }
        /* Deliver EOF to pending promise */
        rl_resolve_null(stream);
        return;
    }

    if nread < 0 {
        if !(*buf).base.is_null() {
            libc::free((*buf).base as *mut c_void);
        }
        return;
    }

    /* Accumulate bytes, deliver each complete line */
    for i in 0..nread {
        let c = (*buf).base.add(i as usize).read() as u8;

        /* ── Escape sequences (raw mode). */
        if ESC_STATE.with(|s| s.get()) > 0 {
            ESC_STATE.with(|s| s.set(tty_esc_step(c, ctx)));
            if ESC_STATE.with(|s| s.get()) != 1 {
                esc_timer_cancel(); /* sequence continues or done */
            }
            continue;
        }
        if c == 0x1b && STDIN_IS_TTY.with(|t| t.get()) != 0 {
            ESC_STATE.with(|s| s.set(1));
            esc_timer_kick(); /* no next byte within 60ms → bare Esc */
            continue;
        }

        if SELECTOR_ACTIVE.with(|a| a.get()) != 0 {
            /* Selector mode: the picker owns the keyboard. */
            if c == 0x03 || c == 0x04 {
                /* Ctrl-C / Ctrl-D: quit the chat */
                call_ui_hook(ctx, "__on_ctrl_c");
                selector_resolve_cancel(ctx);
            } else if c == b'\t' {
                selector_key(ctx, "tab", None);
            } else if c == b'\n' || c == b'\r' {
                selector_key(ctx, "enter", None);
            } else if c == 0x7f || c == 0x08 {
                selector_key(ctx, "backspace", None);
            } else if (0x20..0x7f).contains(&c) {
                selector_key(ctx, "char", Some(&(c as char).to_string()));
            }
            continue;
        }

        if c == 0x03 {
            /* Ctrl-C: quit the chat. */
            call_ui_hook(ctx, "__on_ctrl_c");
            rl_resolve_null(stream);
            continue;
        }
        if c == 0x0b {
            /* Ctrl-K: copy the mouse-selected conversation rows to the
             * clipboard (app-managed selection — see sel_copy). */
            if SEL_ANCHOR.with(|s| s.get()) >= 0 {
                if !sel_copy(ctx) {
                    /* nothing selected/copy failed — still clear the paint
                     * so the stale highlight doesn't linger. */
                    SEL_ANCHOR.with(|s| s.set(-1));
                    SEL_END.with(|e| e.set(-1));
                    sel_repaint(ctx);
                }
            }
            continue;
        }
        if c == 0x04 && STDIN_IS_TTY.with(|t| t.get()) != 0 {
            /* Ctrl-D: on an empty line this is EOF; with text it's ignored. */
            if LINE_BUF.with(|b| b.borrow().len()) == 0 {
                let ui_lines = UI_LINES.with(|l| l.get());
                if ui_lines > 0 {
                    use std::io::Write;
                    let mut out = std::io::stdout().lock();
                    let _ = write!(out, "\x1b[{}A", ui_lines - 2);
                    let _ = write!(out, "\x1b[J");
                    let _ = out.flush();
                    UI_LINES.with(|l| l.set(0));
                }
                rl_resolve_null(stream);
                return;
            }
            continue;
        }
        if c == b'\t' {
            /* TAB: slash-command completion inside the chat. If a highlight
             * is active, accept it; otherwise render (or cycle) matches.
             * On a non-slash line TAB is otherwise free, so it cycles the
             * permission mode (full → edit → plan → full), like other CLIs
             * bind Shift+Tab. The JS hook reuses the /mode path (Rust echoes
             * + persists), so the inline box must be erased BEFORE the echo
             * lands and redrawn after; the TUI logs into its own conversation
             * region and needs no erase. */
            let line = LINE_BUF.with(|b| b.borrow().clone());
            if !line.is_empty() && line[0] == b'/' {
                if COMPLETE_SEL.with(|s| s.get()) >= 0 {
                    completion_accept(ctx);
                } else {
                    tty_render_line_ui(ctx, 1);
                }
            } else {
                if unsafe { tui_active() } == 0 {
                    use std::io::Write;
                    let ui_lines = UI_LINES.with(|l| l.get());
                    if ui_lines > 0 {
                        let mut out = std::io::stdout().lock();
                        let _ = write!(out, "\x1b[{}A\x1b[J", ui_lines);
                        let _ = out.flush();
                        UI_LINES.with(|l| l.set(0));
                    }
                }
                call_ui_hook(ctx, "__on_tab");
                tty_render_line_ui(ctx, 0);
            }
            continue;
        }
        if c == 0x7f || c == 0x08 {
            /* Backspace: drop the last char and redraw. */
            COMPLETE_SEL.with(|s| s.set(-1));
            if LINE_BUF.with(|b| b.borrow().len()) > 0 {
                LINE_BUF.with(|b| b.borrow_mut().pop());
                if STDIN_IS_TTY.with(|t| t.get()) != 0 {
                    tty_render_line_ui(ctx, 0);
                }
            }
            continue;
        }
        if c == b'\n' || c == b'\r' {
            /* Enter. When a completion highlight is active, accept it
             * (fill the highlighted command) instead of submitting the
             * raw "/…" prefix. */
            if COMPLETE_SEL.with(|s| s.get()) >= 0 {
                completion_accept(ctx);
                continue;
            }
            /* Enter. */
            use std::io::Write;
            if STDIN_IS_TTY.with(|t| t.get()) != 0 {
                if tui_active() != 0 {
                    /* Redraw with the submitted line still visible — do NOT
                     * clear the buffer here (delivery uses it below). */
                    tty_render_line_ui(ctx, 0);
                } else {
                    let ui_lines = UI_LINES.with(|l| l.get());
                    if ui_lines > 0 {
                        let mut up = ui_lines - 2;
                        if up < 1 {
                            up = 1;
                        }
                        let mut out = std::io::stdout().lock();
                        let _ = write!(out, "\x1b[{}A", up);
                        let _ = write!(out, "\x1b[J");
                        let _ = out.flush();
                        UI_LINES.with(|l| l.set(0));
                    }
                }
                tty_history_push();
                let line = LINE_BUF.with(|b| b.borrow().clone());
                if !line.is_empty() {
                    chat_history_append(&line);
                }
            }
            /* Trim trailing \r */
            LINE_BUF.with(|b| {
                let mut b = b.borrow_mut();
                if let Some(&b'\r') = b.last() {
                    b.pop();
                }
            });
            let line = LINE_BUF.with(|b| b.borrow().clone());

            let p = READLINE_PROMISE.with(|r| r.replace(ptr::null_mut()));
            if !p.is_null() {
                /* A promise is waiting — deliver immediately. The read STAYS
                 * armed (unref'd): Esc/Ctrl-C must still reach us during the
                 * stream, and keystrokes accumulate as type-ahead. */
                uv::uv_unref(stream as *mut UvHandle);
                if tui_active() != 0 {
                    let mut out = std::io::stdout().lock();
                    let _ = write!(out, "\x1b[?25l");
                    let _ = out.flush();
                }
                /* proc-10: JS_NewStringLen — interior NUL must not truncate
                 * the delivered line (same fix as the readline-promise path). */
                let jline = qjs::JS_NewStringLen(ctx, line.as_ptr() as *const c_char, line.len());
                sofuu_promise_resolve(p, jline);
                qjs::sofuu_js_free_value(ctx, jline);
                /* NOTE: do NOT call sofuu_flush_jobs here — inside libuv cb */
            } else {
                /* No promise yet — queue the line */
                rl_queue_push(&line);
            }
            LINE_BUF.with(|b| b.borrow_mut().clear());
            COMPLETE_SEL.with(|s| s.set(-1));
            /* Instantly redraw the input box EMPTY — the submitted text must
             * vanish from the field the moment Enter is pressed (the driver
             * re-renders again when it re-arms the next prompt). */
            if STDIN_IS_TTY.with(|t| t.get()) != 0 {
                tty_render_line_ui(ctx, 0);
            }
            if LINE_TRUNCATED.with(|t| t.get()) != 0 {
                eprintln!("[sofuu] input truncated at {} bytes", LINE_BUF_CAP - 1);
                LINE_TRUNCATED.with(|t| t.set(0));
            }
        } else if LINE_BUF.with(|b| b.borrow().len()) < LINE_BUF_CAP - 1 {
            COMPLETE_SEL.with(|s| s.set(-1));
            LINE_BUF.with(|b| b.borrow_mut().push(c));
            /* TTY: redraw the live region after EVERY keystroke. */
            if STDIN_IS_TTY.with(|t| t.get()) != 0 {
                tty_render_line_ui(ctx, 0);
            }
        } else {
            LINE_TRUNCATED.with(|t| t.set(1)); /* chars past LINE_BUF_CAP-1 (1 MiB) dropped — the old "8191" comment predated the cap raise */
        }
    }

    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

// ── stdin handle: uv_tty_t for real TTYs, uv_pipe_t for piped stdin ──

/// Open the stdin handle once — keeps the loop alive between turns.
unsafe fn ensure_tty_open(ctx: *mut JSContext) {
    if TTY_OPEN.with(|o| o.get()) != 0 {
        return;
    }
    TTY_CTX.with(|c| c.set(ctx));
    let loop_ = sofuu_loop_get();

    let is_tty = libc::isatty(0);
    STDIN_IS_TTY.with(|t| t.set(is_tty));

    if is_tty == 1 {
        let tty = libc::malloc(uv::sofuu_uv_tty_size()) as *mut UvTty;
        TTY_STDIN.with(|t| t.set(tty));
        uv::uv_tty_init(loop_, tty, 0, /*readable=*/1);
        /* RAW mode: the readline owns echo + editing. */
        uv::uv_tty_set_mode(tty, uv::UV_TTY_MODE_RAW);
        #[cfg(not(target_os = "windows"))]
        {
            /* Belt & braces: make sure ECHO is off in raw mode too. */
            let mut tio: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut tio) == 0 {
                tio.c_lflag &= !(libc::ECHO | libc::ECHONL);
                libc::tcsetattr(0, libc::TCSANOW, &tio);
            }
        }
        uv::uv_unref(tty as *mut UvHandle);
        // Enable mouse: clicks set focus (window vs input), wheel scrolls
        if MOUSE_ENABLED.with(|m| m.get()) == 0 {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(b"\x1b[?1000h\x1b[?1002h\x1b[?1006h");
            // Bracketed paste (DECSET 2004): the terminal wraps pasted text in
            // ESC[200~ … ESC[201~, so a multi-line paste can be buffered as ONE
            // inert unit instead of its embedded CR/LF bytes hitting the Enter
            // path and auto-submitting line after line.
            let _ = out.write_all(b"\x1b[?2004h");
            let _ = out.flush();
            MOUSE_ENABLED.with(|m| m.set(1));
        }
    } else {
        /* stdin is a pipe/file — use uv_pipe_t */
        let pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
        STDIN_RL_PIPE.with(|p| p.set(pipe));
        uv::uv_pipe_init(loop_, pipe, 0);
        uv::uv_pipe_open(pipe, 0);
        uv::uv_unref(pipe as *mut UvHandle);
    }
    TTY_OPEN.with(|o| o.set(1));
    /* F-2: the readline stdin handle is armed per-ctx — a multi-engine
     * teardown must close it (mod_process_cleanup closes with None, same
     * orphan cb; TTY_CTX is the attribution cell). */
    let h = if STDIN_IS_TTY.with(|t| t.get()) != 0 {
        TTY_STDIN.with(|t| t.get()) as *mut UvHandle
    } else {
        STDIN_RL_PIPE.with(|p| p.get()) as *mut UvHandle
    };
    if !h.is_null() {
        crate::rt::event_loop::track_handle(ctx, h as *mut UvHandle, None);
    }
}

unsafe extern "C" fn js_tty_raw(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    if STDIN_IS_TTY.with(|t| t.get()) != 0 && TTY_OPEN.with(|o| o.get()) != 0 {
        /* IO_RAW: no echo, no canonical processing. */
        uv::uv_tty_set_mode(TTY_STDIN.with(|t| t.get()), uv::UV_TTY_MODE_RAW);
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_tty_normal(
    _ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    if STDIN_IS_TTY.with(|t| t.get()) != 0 && TTY_OPEN.with(|o| o.get()) != 0 {
        uv::uv_tty_set_mode(TTY_STDIN.with(|t| t.get()), uv::UV_TTY_MODE_NORMAL);
        if MOUSE_ENABLED.with(|m| m.get()) != 0 {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(b"\x1b[?2004l\x1b[?1006l\x1b[?1002l\x1b[?1000l");
            let _ = out.flush();
            MOUSE_ENABLED.with(|m| m.set(0));
        }
    }
    qjs::sofuu_js_undefined()
}

// ── JS: __readline(prompt_str) → Promise<string|null> ────────────────

unsafe extern "C" fn js_io_readline(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    ensure_tty_open(ctx); /* sets g_stdin_is_tty — must run BEFORE the prompt */

    /* Remember the prompt. On a TTY it is rendered INSIDE the input box by
     * the live-region renderer; piped input just echoes it like before. */
    RL_PROMPT.with(|p| p.borrow_mut().clear());
    if argc >= 1 {
        let label = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !label.is_null() {
            let bytes = CStr::from_ptr(label).to_bytes();
            if STDIN_IS_TTY.with(|t| t.get()) != 0 {
                RL_PROMPT.with(|p| {
                    let mut p = p.borrow_mut();
                    p.clear();
                    p.extend_from_slice(bytes);
                });
            } else {
                use std::io::Write;
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(bytes);
                let _ = out.flush();
            }
            qjs::sofuu_js_free_cstring(ctx, label);
        }
    }

    /* Keep whatever the user already typed ahead (the read stays armed
     * between turns). */
    HIST_POS.with(|p| p.set(-1));

    /* Load persisted chat history the first time the readline arms. */
    chat_history_load();

    let mut prom: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut prom);

    /* Fast path: a line was already buffered (fast piped / pre-typed input) */
    if !rl_queue_empty() {
        if let Some(line) = rl_queue_pop() {
            /* proc-10: same interior-NUL truncation as the callback paths. */
            let jline = qjs::JS_NewStringLen(ctx, line.as_ptr() as *const c_char, line.len());
            sofuu_promise_resolve(prom, jline);
            qjs::sofuu_js_free_value(ctx, jline);
        } else {
            sofuu_promise_resolve(prom, qjs::sofuu_js_null());
        }
        return promise;
    }

    /* Slow path: arm the read and draw the empty input box. A second
     * __readline while one is pending would overwrite the first handle and
     * leave its JS promise pending forever — reject it instead. */
    if STDIN_IS_TTY.with(|t| t.get()) != 0 {
        tty_render_line_ui(ctx, 0);
    }
    let old = READLINE_PROMISE.with(|r| r.replace(prom));
    if !old.is_null() {
        sofuu_promise_reject_str(old, c"readline superseded by a new readline call".as_ptr());
    }
    uv::uv_ref(stdin_rl_stream() as *mut UvHandle);
    uv::uv_read_start(stdin_rl_stream(), Some(tty_read_alloc_cb), Some(tty_read_cb));

    promise
}

// ── sofuu.io.prompt(label) — legacy synchronous fallback (non-TTY only) ─

unsafe extern "C" fn js_io_prompt(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    /* Stop libuv TTY reading if open */
    if TTY_OPEN.with(|o| o.get()) != 0 {
        uv::uv_read_stop(stdin_rl_stream());
    }

    /* Save terminal and switch to canonical + echo (POSIX termios; the
     * Windows console path skips the mode dance). */
    let is_tty = libc::isatty(0);
    #[cfg(not(target_os = "windows"))]
    let mut saved: libc::termios = std::mem::zeroed();
    #[cfg(not(target_os = "windows"))]
    if is_tty == 1 {
        libc::tcgetattr(0, &mut saved);
        let mut cooked = saved;
        cooked.c_lflag |= libc::ICANON | libc::ECHO | libc::ECHOE | libc::ECHOK;
        cooked.c_lflag &= !libc::ECHONL;
        cooked.c_iflag |= libc::ICRNL;
        libc::tcsetattr(0, libc::TCSANOW, &cooked);
    }
    #[cfg(target_os = "windows")]
    let _ = is_tty;

    if argc >= 1 {
        let label = qjs::sofuu_js_to_cstring(ctx, *argv);
        if !label.is_null() {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(CStr::from_ptr(label).to_bytes());
            let _ = out.flush();
            qjs::sofuu_js_free_cstring(ctx, label);
        }
    }

    /* Read one line (canonical mode returns per line) — like fgets. */
    let mut buf = [0u8; 4096];
    /* count is size_t on POSIX, c_uint on Windows — `as _` targets the
       parameter type on each platform */
    let n = libc::read(0, buf.as_mut_ptr() as *mut c_void, (buf.len() - 1) as _);
    let result = if n <= 0 {
        qjs::sofuu_js_null()
    } else {
        let mut len = n as usize;
        if buf[len - 1] == b'\n' {
            len -= 1;
        }
        let c = CString::new(&buf[..len]).unwrap_or_default();
        qjs::sofuu_js_new_string(ctx, c.as_ptr())
    };

    #[cfg(not(target_os = "windows"))]
    if is_tty == 1 {
        libc::tcsetattr(0, libc::TCSANOW, &saved);
    }
    if TTY_OPEN.with(|o| o.get()) != 0 {
        uv::uv_read_start(stdin_rl_stream(), Some(tty_read_alloc_cb), Some(tty_read_cb));
    }

    result
}

// ── process.exit / cwd / chdir / on / off ────────────────────────────

unsafe extern "C" fn js_process_exit(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let mut code: c_int = 0;
    if argc >= 1 {
        qjs::JS_ToInt32(ctx, &mut code, *argv);
    }

    let gctx = CTX.with(|c| c.get());
    if !gctx.is_null() {
        let exit_cb = get_process_cb(gctx, "__exit_handler");
        if qjs::JS_IsFunction(gctx, exit_cb) != 0 {
            dispatch_signal_cb(gctx, "__exit_handler", "exit");
        }
        qjs::sofuu_js_free_value(gctx, exit_cb);
    }

    /* Embedded mode: never libc::exit — the host owns the process. Throw
     * a catchable ExitError instead (PLAN-HEADLESS H2.1). */
    if embed_config::is_embedded() {
        let err = qjs::JS_NewError(ctx);
        let msg = CString::new(format!("process.exit({})", code)).unwrap_or_default();
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"name".as_ptr(),
            qjs::sofuu_js_new_string(ctx, c"ExitError".as_ptr()),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"message".as_ptr(),
            qjs::sofuu_js_new_string(ctx, msg.as_ptr()),
        );
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"exitCode".as_ptr(),
            qjs::sofuu_js_new_int32(ctx, code),
        );
        qjs::JS_Throw(ctx, err);
        return qjs::sofuu_js_exception();
    }

    libc::exit(code);
}

unsafe extern "C" fn js_process_cwd(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let mut buf = [0u8; 4096];
    let mut size = buf.len();
    // SAFETY: buf is writable; size in/out.
    if uv::uv_cwd(buf.as_mut_ptr() as *mut c_char, &mut size) != 0 {
        return qjs::JS_ThrowTypeError(ctx, c"cwd() failed".as_ptr());
    }
    let c = CString::new(&buf[..size]).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, c.as_ptr())
}

unsafe extern "C" fn js_process_chdir(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"chdir requires 1 arg".as_ptr());
    }
    let dir = qjs::sofuu_js_to_cstring(ctx, *argv);
    if dir.is_null() {
        return qjs::sofuu_js_exception();
    }
    let r = uv::uv_chdir(dir);
    qjs::sofuu_js_free_cstring(ctx, dir);
    if r != 0 {
        return qjs::JS_ThrowTypeError(ctx, c"chdir failed: %s".as_ptr(), uv::uv_strerror(r));
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_process_on(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"process.on requires (event, fn)".as_ptr());
    }
    if qjs::JS_IsFunction(ctx, *argv.add(1)) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"process.on: handler must be a function".as_ptr());
    }

    let event_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if event_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let event = CStr::from_ptr(event_ptr).to_string_lossy().into_owned();

    match event.as_str() {
        "SIGINT" => {
            set_process_cb(ctx, "__sigint_handler", qjs::sofuu_js_dup_value(ctx, *argv.add(1)));
            /* Embedded hosts keep their own signal handling unless they
             * opted in via enable_signals (PLAN-HEADLESS H2.2). */
            if !embed_config::is_embedded() || embed_config::signals_enabled() {
                install_sigaction(libc::SIGINT, posix_sigint_handler);
            }
        }
        "SIGTERM" => {
            set_process_cb(ctx, "__sigterm_handler", qjs::sofuu_js_dup_value(ctx, *argv.add(1)));
            if !embed_config::is_embedded() || embed_config::signals_enabled() {
                install_sigaction(libc::SIGTERM, posix_sigterm_handler);
            }
        }
        "exit" => {
            set_process_cb(ctx, "__exit_handler", qjs::sofuu_js_dup_value(ctx, *argv.add(1)));
        }
        "uncaughtException" => {
            set_process_cb(ctx, "__uncaught_handler", qjs::sofuu_js_dup_value(ctx, *argv.add(1)));
        }
        "unhandledRejection" => {
            /* Acknowledged — the M1 rejection tracker handles this. */
        }
        _ => { /* silently ignore unknown events */ }
    }

    qjs::sofuu_js_free_cstring(ctx, event_ptr);
    qjs::sofuu_js_dup_value(ctx, _this) /* chainable */
}

unsafe extern "C" fn js_process_off(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let event_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if event_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let event = CStr::from_ptr(event_ptr).to_string_lossy().into_owned();

    match event.as_str() {
        "SIGINT" => set_process_cb(ctx, "__sigint_handler", qjs::sofuu_js_undefined()),
        "SIGTERM" => set_process_cb(ctx, "__sigterm_handler", qjs::sofuu_js_undefined()),
        "exit" => set_process_cb(ctx, "__exit_handler", qjs::sofuu_js_undefined()),
        "uncaughtException" => set_process_cb(ctx, "__uncaught_handler", qjs::sofuu_js_undefined()),
        _ => {}
    }

    qjs::sofuu_js_free_cstring(ctx, event_ptr);
    qjs::sofuu_js_undefined()
}

// ── registration (C symbol replacement for mod_process_register) ─────

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_process_register(ctx: *mut JSContext) {
    CTX.with(|c| c.set(ctx));

    let global = qjs::sofuu_js_get_global_object(ctx);
    let process = qjs::sofuu_js_new_object(ctx);

    /* -- Metadata -- */
    let ver_c = CString::new(env!("CARGO_PKG_VERSION")).unwrap_or_default();
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"version".as_ptr(),
        qjs::sofuu_js_new_string(ctx, ver_c.as_ptr()),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"runtime".as_ptr(),
        qjs::sofuu_js_new_string(ctx, c"sofuu".as_ptr()),
    );

    let platform = if cfg!(target_os = "macos") {
        c"darwin"
    } else if cfg!(target_os = "windows") {
        c"win32"
    } else {
        c"linux"
    };
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"platform".as_ptr(),
        qjs::sofuu_js_new_string(ctx, platform.as_ptr()),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"pid".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, libc::getpid() as i32),
    );

    /* -- process.argv -- */
    let argv_arr = qjs::JS_NewArray(ctx);
    ARGV.with(|v| {
        let v = v.borrow();
        for (i, a) in v.iter().enumerate() {
            qjs::JS_SetPropertyUint32(
                ctx,
                argv_arr,
                i as u32,
                qjs::sofuu_js_new_string(ctx, a.as_ptr()),
            );
        }
    });
    qjs::sofuu_js_set_property_str(ctx, process, c"argv".as_ptr(), argv_arr);

    /* -- process.env -- */
    let env_obj = qjs::sofuu_js_new_object(ctx);
    #[cfg(not(target_os = "windows"))]
    {
        // SAFETY: environ is the process environment block.
        if !environ.is_null() {
            let mut i = 0;
            loop {
                let entry = *environ.add(i);
                if entry.is_null() {
                    break;
                }
                let entry_c = CStr::from_ptr(entry);
                if let Some(eq) = entry_c.to_bytes().iter().position(|&b| b == b'=') {
                    let key = CString::new(&entry_c.to_bytes()[..eq]).unwrap_or_default();
                    let val = CStr::from_ptr(entry.add(eq + 1));
                    qjs::sofuu_js_set_property_str(
                        ctx,
                        env_obj,
                        key.as_ptr(),
                        qjs::sofuu_js_new_string(ctx, val.as_ptr()),
                    );
                }
                i += 1;
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        for (k, v) in std::env::vars() {
            let key = CString::new(k).unwrap_or_default();
            let val = CString::new(v).unwrap_or_default();
            qjs::sofuu_js_set_property_str(
                ctx,
                env_obj,
                key.as_ptr(),
                qjs::sofuu_js_new_string(ctx, val.as_ptr()),
            );
        }
    }
    qjs::sofuu_js_set_property_str(ctx, process, c"env".as_ptr(), env_obj);

    /* -- Functions -- */
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"exit".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_process_exit, c"exit".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"cwd".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_process_cwd, c"cwd".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"chdir".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_process_chdir, c"chdir".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"on".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_process_on, c"on".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        process,
        c"off".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_process_off, c"off".as_ptr(), 1),
    );

    /* -- process.stdin -- */
    STDIN_ON_DATA.with(|d| d.set(qjs::sofuu_js_undefined()));
    STDIN_ON_END.with(|e| e.set(qjs::sofuu_js_undefined()));
    STDIN_ON_ERROR.with(|e| e.set(qjs::sofuu_js_undefined()));

    let stdin_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"on".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stdin_on, c"on".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"resume".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stdin_resume, c"resume".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"pause".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stdin_pause, c"pause".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"setEncoding".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stdin_set_encoding, c"setEncoding".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"fd".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdin_obj,
        c"isTTY".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, libc::isatty(0)),
    );
    qjs::sofuu_js_set_property_str(ctx, process, c"stdin".as_ptr(), stdin_obj);

    /* -- process.stdout -- */
    let stdout_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        stdout_obj,
        c"write".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stdout_write, c"write".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdout_obj,
        c"fd".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stdout_obj,
        c"isTTY".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, libc::isatty(1)),
    );
    qjs::sofuu_js_set_property_str(ctx, process, c"stdout".as_ptr(), stdout_obj);

    /* -- process.stderr -- */
    let stderr_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        stderr_obj,
        c"write".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_stderr_write, c"write".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stderr_obj,
        c"fd".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        stderr_obj,
        c"isTTY".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, libc::isatty(2)),
    );
    qjs::sofuu_js_set_property_str(ctx, process, c"stderr".as_ptr(), stderr_obj);

    /* Attach to global */
    qjs::sofuu_js_set_property_str(ctx, global, c"process".as_ptr(), process);

    /* Register prompt functions as globals */
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__prompt".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_io_prompt, c"__prompt".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__readline".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_io_readline, c"__readline".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__chat_status".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_chat_status, c"__chat_status".as_ptr(), 3),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__chat_is_tty".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_chat_is_tty, c"__chat_is_tty".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_on".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_on, c"__tui_on".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_off".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_off, c"__tui_off".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_log".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_log, c"__tui_log".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_log_last".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_log_last, c"__tui_log_last".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_scroll".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_scroll, c"__tui_scroll".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_set_header".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_set_header, c"__tui_set_header".as_ptr(), 1),
    );
    tui_init();

    /* TTY mode helpers */
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__ttyRaw".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tty_raw, c"__ttyRaw".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__ttyNormal".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tty_normal, c"__ttyNormal".as_ptr(), 0),
    );

    /* Interactive selector (picker overlay) + size queries */
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__selector_open".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_selector_open, c"__selector_open".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__selector_draw".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_selector_draw, c"__selector_draw".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__selector_close".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_selector_close, c"__selector_close".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_width".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_width_js, c"__tui_width".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"__tui_height".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_tui_height_js, c"__tui_height".as_ptr(), 0),
    );

    qjs::sofuu_js_free_value(ctx, global);

    /*
     * NOTE: SIGINT/SIGTERM use POSIX sigaction flag-setters; the JS
     * callbacks are dispatched from sofuu_loop_run after each I/O poll via
     * process_dispatch_pending_signals() (safe JS-thread context).
     * SIGWINCH is delivered through a uv_signal_t watcher — see on_winch.
     */
}

/// mod_process_cleanup — call BEFORE JS_FreeContext. C symbol replacement.
/// Releases C-static JSValue references held by stdin's event handlers.
/// Signal callbacks are stored on the process JS object and freed
/// automatically when JS_FreeContext destroys the global scope.
///
/// # Safety
/// `ctx` must be the live engine context (teardown path).
#[no_mangle]
pub unsafe extern "C" fn mod_process_cleanup(ctx: *mut JSContext) {
    /* F-2 (AUDIT-2026-09-01-CLI): several engines can share one thread
     * (capi multi-instance). Every stdin/TUI static below belongs to the
     * engine that armed it — a non-owner destroy must leave the owner's
     * machinery alone: freeing its JSValues with the WRONG ctx is UB, and
     * stopping its timers/rejecting its promises breaks the live TUI. A
     * null cell means "nobody owns it" — the single-engine case, where
     * everything is mine (identical to the pre-F-2 behavior). */
    let on_owner = STDIN_OWNER.with(|o| o.get());
    let tty_owner = TTY_CTX.with(|c| c.get());
    let pipe_owner = STDIN_CTX.with(|c| c.get());
    let mine_on = on_owner.is_null() || on_owner == ctx;
    let mine_tty = tty_owner.is_null() || tty_owner == ctx;
    let mine_pipe = pipe_owner.is_null() || pipe_owner == ctx;

    if mine_on {
        STDIN_ON_DATA.with(|d| qjs::sofuu_js_free_value(ctx, d.replace(qjs::sofuu_js_undefined())));
        STDIN_ON_END.with(|e| qjs::sofuu_js_free_value(ctx, e.replace(qjs::sofuu_js_undefined())));
        STDIN_ON_ERROR.with(|e| qjs::sofuu_js_free_value(ctx, e.replace(qjs::sofuu_js_undefined())));
        STDIN_OWNER.with(|o| o.set(ptr::null_mut()));
    }
    CTX.with(|c| c.set(ptr::null_mut()));
    if mine_tty && ESC_TIMER_INIT.with(|i| i.get()) != 0 {
        let t = ESC_TIMER.with(|e| e.get());
        if uv::uv_is_closing(t as *const UvHandle) == 0 {
            uv::uv_timer_stop(t);
            /* C-host direct call (no loop teardown): close it here so a
             * later uv_walk never visits a freed-storage handle. */
            uv::uv_close(t as *mut UvHandle, None);
            crate::rt::event_loop::untrack_handle(t as *mut UvHandle);
        }
        /* Reset the slot either way: shutdown_engine/loop_close may have
         * already closed it, and a stale INIT flag would make the next
         * engine's esc_timer_kick re-start a closed handle. */
        ESC_TIMER.with(|e| e.set(ptr::null_mut()));
        ESC_TIMER_INIT.with(|i| i.set(0));
    }

    /* Clean up async readline state */
    if mine_tty && TTY_OPEN.with(|o| o.get()) != 0 {
        uv::uv_read_stop(stdin_rl_stream());
        if STDIN_IS_TTY.with(|t| t.get()) != 0 {
            // Restore the terminal fully: cooked mode + bracketed-paste/mouse
            // modes off, or the host shell inherits stray DECSET states.
            if MOUSE_ENABLED.with(|m| m.get()) != 0 {
                use std::io::Write;
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(b"\x1b[?2004l\x1b[?1006l\x1b[?1002l\x1b[?1000l");
                let _ = out.flush();
                MOUSE_ENABLED.with(|m| m.set(0));
            }
            uv::uv_tty_reset_mode();
            let tty = TTY_STDIN.with(|t| t.get());
            if uv::uv_is_closing(tty as *const UvHandle) == 0 {
                uv::uv_close(tty as *mut UvHandle, None);
                crate::rt::event_loop::untrack_handle(tty as *mut UvHandle);
            }
        } else {
            let pipe = STDIN_RL_PIPE.with(|p| p.get());
            if uv::uv_is_closing(pipe as *const UvHandle) == 0 {
                uv::uv_close(pipe as *mut UvHandle, None);
                crate::rt::event_loop::untrack_handle(pipe as *mut UvHandle);
            }
        }
        TTY_OPEN.with(|o| o.set(0));
    }
    /* P3 (AUDIT-2026-09-07): the process-API stdin pipe was never torn
     * down — a script that opened stdin (stdin.on("data")/resume) leaked
     * the malloc'd uv_pipe_t and left fd 0 in a started-read state at
     * engine teardown. Same late-close pattern as the readline branch
     * above: the loop storage outlives this call (see rt/loop.rs), and
     * uv_is_closing guards a handle the loop walk already closed.
     * F-2: only the engine that opened it (STDIN_CTX) may touch it. */
    if mine_pipe && STDIN_PIPE_OPEN.with(|o| o.get()) != 0 {
        let pipe = STDIN_PIPE.with(|p| p.get());
        if !pipe.is_null() && uv::uv_is_closing(pipe as *const UvHandle) == 0 {
            uv::uv_read_stop(pipe as *mut UvStream);
            uv::uv_close(pipe as *mut UvHandle, None);
            crate::rt::event_loop::untrack_handle(pipe as *mut UvHandle);
        }
        STDIN_PIPE.with(|p| p.set(ptr::null_mut()));
        STDIN_PIPE_OPEN.with(|o| o.set(0));
        STDIN_CTX.with(|c| c.set(ptr::null_mut()));
    }
    /* The selection-notice timer (1800 ms) was missed when ESC_TIMER got
     * its stop — it must not fire into a torn-down TUI. F-2: TUI-owner
     * scoped, same stale-pointer reset as ESC_TIMER. */
    if mine_tty {
        let sel_t = SEL_NOTICE_TIMER.with(|t| t.get());
        if !sel_t.is_null() {
            if uv::uv_is_closing(sel_t as *const UvHandle) == 0 {
                uv::uv_timer_stop(sel_t);
                uv::uv_close(sel_t as *mut UvHandle, None);
                crate::rt::event_loop::untrack_handle(sel_t as *mut UvHandle);
            }
            SEL_NOTICE_TIMER.with(|t| t.set(ptr::null_mut()));
        }
        let rl = READLINE_PROMISE.with(|r| r.replace(ptr::null_mut()));
        if !rl.is_null() {
            /* A pending readline should not leak its promise handle at teardown. */
            sofuu_promise_reject_str(rl, c"stdin closed".as_ptr());
        }
        let sel = SELECTOR_PROMISE.with(|s| s.replace(ptr::null_mut()));
        if !sel.is_null() {
            sofuu_promise_reject_str(sel, c"stdin closed".as_ptr());
        }
        SELECTOR_ACTIVE.with(|a| a.set(0));
        TTY_CTX.with(|c| c.set(ptr::null_mut()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /* proc-11: the history file cap must keep the NEWEST lines when it
     * rewrites, not the oldest. */
    #[test]
    fn chat_history_trimmed_keeps_newest_lines() {
        let text = (0..10).map(|i| format!("line-{i}")).collect::<Vec<_>>().join("\n");
        let out = chat_history_trimmed(&text, 4);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["line-6", "line-7", "line-8", "line-9"]);
    }

    /// The readline truncates a row that overruns its box. It MEASURES in
    /// display cells but used to CUT one cell per char, so a wide CJK/emoji
    /// row got ~2x the budget and spilled past the border. cut_to_cells
    /// must agree with tui_disp_width on the same bytes.
    #[test]
    fn cut_to_cells_never_exceeds_a_cell_budget() {
        let samples = [
            "plain ascii text",
            "\u{7d20}\u{98a8}\u{7d20}\u{98a8}",   // CJK: 2 cells each
            "\u{1f389}\u{1f389}\u{1f389}",             // emoji: 2 cells each
            "mix \u{7d20}\u{1f389} ab",                 // interleaved
            "e\u{0301}combining",                        // zero-width
            "",
        ];
        for text in samples {
            let b = text.as_bytes();
            for keep in 0..=20usize {
                let cut = cut_to_cells(b, keep);
                assert!(cut <= b.len(), "{text:?} keep={keep} cut={cut} len={}", b.len());
                assert!(
                    text.is_char_boundary(cut),
                    "{text:?} keep={keep} cut={cut} splits a char"
                );
                let shown = crate::rt::tui::truncate_cells(&b[..cut], keep);
                assert!(
                    shown <= b.len(),
                    "{text:?} keep={keep} produced {shown} bytes for a {keep}-cell budget"
                );
            }
        }
    }

    /// The specific failure: a row of wide glyphs must be cut at roughly
    /// half its character count, not all of it.
    #[test]
    fn cut_to_cells_halves_wide_glyph_rows() {
        let cjk = "\u{7d20}".repeat(30).into_bytes();
        let cut = cut_to_cells(&cjk, 10);
        // 10 cells of 2-cell glyphs = 5 chars = 15 bytes.
        assert_eq!(cut, 15, "expected 5 CJK chars, cut at {cut}");
        assert!(cut < cjk.len(), "row must actually be truncated");

        let emoji = "\u{1f389}".repeat(30).into_bytes();
        let cut_e = cut_to_cells(&emoji, 10);
        assert_eq!(cut_e, 20, "expected 5 emoji (4 bytes each), cut at {cut_e}");
    }

    #[test]
    fn chat_history_trimmed_handles_short_and_empty() {
        assert_eq!(chat_history_trimmed("a\nb\nc", 10), "a\nb\nc\n");
        assert_eq!(chat_history_trimmed("", 5), "");
        assert_eq!(chat_history_trimmed("only", 0), "");
    }
}
