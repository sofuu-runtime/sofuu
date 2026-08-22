// rt/process_spawn.rs — sofuu.spawn / sofuu.exec (PLAN-RUST-MIGRATION M2).
//
// Port of the deleted `src/io/subprocess.c`, semantics verbatim. The C code
// embedded the uv handles inside the request structs; here each handle is a
// separate malloc'd block whose leading `data` field points back at the
// Box'd request. The close-callback fence is preserved: the LAST of the
// (pipes + process) close callbacks frees the request AND the handle
// storages, so neither the GC finalizer nor libuv can touch freed memory.
//
// C symbol replaced: `mod_subprocess_register` (engine.c calls it unchanged).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use sofuu_ffi::qjs::{self, JSCFunctionListEntry, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{
    self, UvBuf, UvHandle, UvPipe, UvProcess, UvProcessOptions, UvStdioContainer, UvStdioData,
    UvStream, UvWriteReq,
};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject, sofuu_promise_resolve,
    PromiseHandle,
};

// ── sofuu.spawn — callback-based child process ───────────────────────

/// Replaces the C `subprocess_req_t` (which embedded `uv_process_t`, three
/// `uv_pipe_t`, options and stdio — here those are separate allocations
/// referenced by pointer; only handle->data and pipe->data point back).
struct SpawnReq {
    ctx: *mut JSContext,
    process: *mut UvProcess,
    stdin_pipe: *mut UvPipe,
    stdout_pipe: *mut UvPipe,
    stderr_pipe: *mut UvPipe,

    on_stdout: JSValue,
    on_stderr: JSValue,
    on_exit: JSValue,

    /* Dup'd ref to the JS Subprocess object: the close-callback fence frees
     * the request before the object is necessarily GC'd, so the fence must
     * be able to null the object's opaque (post-exit write()/kill() would
     * otherwise dereference freed memory). Released in spawn_free. */
    self_obj: JSValue,

    command: CString,
    args: Vec<CString>,
    args_ptr: Vec<*mut c_char>,
    cwd: Option<CString>,
    opts: UvProcessOptions,
    stdio: [UvStdioContainer; 3],

    /* Lifecycle: `done` is set in on_process_exit; `closing` counts the 4
     * handles (3 pipes + process) being closed — the LAST close callback
     * frees the request + handle storages. */
    done: c_int,
    closing: c_int,
}

unsafe fn spawn_free(req: *mut SpawnReq) {
    // The JS object may still be referenced by user code — null its opaque
    // so a post-exit write()/kill() sees the NULL guard instead of this
    // freed request, then drop our keeping reference.
    if !qjs::is_undefined((*req).self_obj) {
        qjs::JS_SetOpaque((*req).self_obj, ptr::null_mut());
        qjs::sofuu_js_free_value((*req).ctx, (*req).self_obj);
        (*req).self_obj = qjs::sofuu_js_undefined();
    }
    // All 4 handles are closed by now — release their storages.
    // (On the uv_spawn failure path `process` was already freed directly and
    // nulled — free(NULL) is a no-op there.)
    libc::free((*req).process as *mut c_void);
    libc::free((*req).stdin_pipe as *mut c_void);
    libc::free((*req).stdout_pipe as *mut c_void);
    libc::free((*req).stderr_pipe as *mut c_void);
    drop(Box::from_raw(req)); /* drops command/args/cwd */
}

/// The last of the 4 handle close callbacks owns the req free.
unsafe extern "C" fn spawn_close_cb(handle: *mut UvHandle) {
    // SAFETY: data set at init; handle is fully closed here.
    let req = *(handle as *mut *mut SpawnReq);
    if req.is_null() {
        return;
    }
    if (*req).closing > 0 {
        (*req).closing -= 1;
        if (*req).closing == 0 {
            spawn_free(req);
        }
    }
}

unsafe extern "C" fn alloc_buffer(
    _handle: *mut UvHandle,
    suggested_size: usize,
    buf: *mut UvBuf,
) {
    // SAFETY: buf is libuv-provided storage for this allocation.
    (*buf).base = libc::malloc(suggested_size) as *mut c_char;
    /* On OOM libuv delivers nread > 0 with a NULL base — len must be 0 so
     * the read callbacks never build a slice from NULL. */
    (*buf).len = if (*buf).base.is_null() { 0 } else { suggested_size };
}

unsafe extern "C" fn on_stdout_read(
    pipe: *mut UvStream,
    nread: isize,
    buf: *const UvBuf,
) {
    // SAFETY: pipe->data set at init.
    let req = *(pipe as *mut *mut SpawnReq);
    if nread > 0 {
        if qjs::JS_IsFunction((*req).ctx, (*req).on_stdout) != 0 {
            let str_v = qjs::JS_NewStringLen((*req).ctx, (*buf).base, nread as usize);
            let ret = qjs::JS_Call((*req).ctx, (*req).on_stdout, qjs::sofuu_js_undefined(), 1, &str_v);
            qjs::sofuu_js_free_value((*req).ctx, str_v);
            qjs::sofuu_js_free_value((*req).ctx, ret);
            sofuu_flush_jobs((*req).ctx);
        }
    } else if nread < 0 {
        uv::uv_read_stop(pipe);
    }
    // SAFETY: buf.base was malloc'd by alloc_buffer.
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn on_stderr_read(
    pipe: *mut UvStream,
    nread: isize,
    buf: *const UvBuf,
) {
    let req = *(pipe as *mut *mut SpawnReq);
    if nread > 0 {
        if qjs::JS_IsFunction((*req).ctx, (*req).on_stderr) != 0 {
            let str_v = qjs::JS_NewStringLen((*req).ctx, (*buf).base, nread as usize);
            let ret = qjs::JS_Call((*req).ctx, (*req).on_stderr, qjs::sofuu_js_undefined(), 1, &str_v);
            qjs::sofuu_js_free_value((*req).ctx, str_v);
            qjs::sofuu_js_free_value((*req).ctx, ret);
            sofuu_flush_jobs((*req).ctx);
        }
    } else if nread < 0 {
        uv::uv_read_stop(pipe);
    }
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn on_process_exit(
    process: *mut UvProcess,
    exit_status: i64,
    _term_signal: c_int,
) {
    // SAFETY: process->data set at init.
    let req = *(process as *mut *mut SpawnReq);
    if req.is_null() || (*req).done != 0 {
        return;
    }
    (*req).done = 1;

    uv::uv_read_stop((*req).stdout_pipe as *mut UvStream);
    uv::uv_read_stop((*req).stderr_pipe as *mut UvStream);

    if qjs::JS_IsFunction((*req).ctx, (*req).on_exit) != 0 {
        let val = qjs::sofuu_js_new_int32((*req).ctx, exit_status as i32);
        let ret = qjs::JS_Call((*req).ctx, (*req).on_exit, qjs::sofuu_js_undefined(), 1, &val);
        qjs::sofuu_js_free_value((*req).ctx, val);
        qjs::sofuu_js_free_value((*req).ctx, ret);
        sofuu_flush_jobs((*req).ctx);
    }

    /* Callbacks done — the JS refs are released here (the finalizer may
     * have replaced them with undefined first, which frees as a no-op). */
    qjs::sofuu_js_free_value((*req).ctx, (*req).on_stdout);
    qjs::sofuu_js_free_value((*req).ctx, (*req).on_stderr);
    qjs::sofuu_js_free_value((*req).ctx, (*req).on_exit);
    (*req).on_stdout = qjs::sofuu_js_undefined();
    (*req).on_stderr = qjs::sofuu_js_undefined();
    (*req).on_exit = qjs::sofuu_js_undefined();

    /* Free the request only when ALL FOUR handles finished closing. */
    (*req).closing = 4;
    uv::uv_close((*req).stdin_pipe as *mut UvHandle, Some(spawn_close_cb));
    uv::uv_close((*req).stdout_pipe as *mut UvHandle, Some(spawn_close_cb));
    uv::uv_close((*req).stderr_pipe as *mut UvHandle, Some(spawn_close_cb));
    uv::uv_close(process as *mut UvHandle, Some(spawn_close_cb));
}

static SUBPROCESS_CLASS_ID: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn js_subprocess_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let req = qjs::JS_GetOpaque(val, SUBPROCESS_CLASS_ID.load(Ordering::Relaxed));
    if req.is_null() {
        return;
    }
    let req = req as *mut SpawnReq;
    /* NEVER free req here — see the C comment: the close-callback fence in
     * on_process_exit owns the memory. Only detach the JS callback refs
     * (freeing the old values — on_process_exit frees the undefined
     * replacements as a no-op, so without this they would leak). */
    if (*req).done == 0 {
        qjs::sofuu_js_free_value((*req).ctx, (*req).on_stdout);
        qjs::sofuu_js_free_value((*req).ctx, (*req).on_stderr);
        qjs::sofuu_js_free_value((*req).ctx, (*req).on_exit);
        (*req).on_stdout = qjs::sofuu_js_undefined();
        (*req).on_stderr = qjs::sofuu_js_undefined();
        (*req).on_exit = qjs::sofuu_js_undefined();
    }
}

unsafe extern "C" fn js_subprocess_kill(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let req = qjs::JS_GetOpaque(this_val, SUBPROCESS_CLASS_ID.load(Ordering::Relaxed));
    if req.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"process already exited".as_ptr());
    }
    let req = req as *mut SpawnReq;
    uv::uv_process_kill((*req).process, 15); /* SIGTERM */
    qjs::sofuu_js_undefined()
}

#[repr(C)]
struct WriteReq {
    req: UvWriteReq,
    buf: UvBuf,
}

unsafe extern "C" fn on_write_done(wreq: *mut UvWriteReq, _status: c_int) {
    // SAFETY: wreq is the leading field of a malloc'd WriteReq.
    let wr = wreq as *mut WriteReq;
    libc::free((*wr).buf.base as *mut c_void);
    libc::free(wr as *mut c_void);
}

unsafe extern "C" fn js_subprocess_write(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"write expects 1 arg".as_ptr());
    }
    let req = qjs::JS_GetOpaque(this_val, SUBPROCESS_CLASS_ID.load(Ordering::Relaxed));
    if req.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"process already exited".as_ptr());
    }
    let req = req as *mut SpawnReq;

    let mut len: usize = 0;
    let str_v = qjs::JS_ToCStringLen2(ctx, &mut len, *argv, 0);
    if str_v.is_null() {
        return qjs::sofuu_js_exception();
    }

    let wr = libc::malloc(uv::sofuu_uv_write_size()) as *mut WriteReq;
    if wr.is_null() {
        qjs::sofuu_js_free_cstring(ctx, str_v);
        return qjs::JS_ThrowTypeError(ctx, c"out of memory".as_ptr());
    }
    (*wr).buf.base = libc::malloc(len) as *mut c_char;
    if (*wr).buf.base.is_null() {
        libc::free(wr as *mut c_void);
        qjs::sofuu_js_free_cstring(ctx, str_v);
        return qjs::JS_ThrowTypeError(ctx, c"out of memory".as_ptr());
    }
    (*wr).buf.len = len;
    // SAFETY: copies len bytes from the JS string into the new buffer.
    std::ptr::copy_nonoverlapping(str_v as *const u8, (*wr).buf.base as *mut u8, len);
    qjs::sofuu_js_free_cstring(ctx, str_v);

    let rc = uv::uv_write(
        wr as *mut UvWriteReq,
        (*req).stdin_pipe as *mut UvStream,
        &(*wr).buf,
        1,
        Some(on_write_done),
    );
    if rc < 0 {
        /* Synchronous failure: libuv will NOT call on_write_done — free the
         * request and its buffer here or they leak. */
        libc::free((*wr).buf.base as *mut c_void);
        libc::free(wr as *mut c_void);
        return qjs::JS_ThrowTypeError(ctx, c"write failed: %s".as_ptr(), uv::uv_strerror(rc));
    }
    qjs::sofuu_js_undefined()
}

thread_local! {
    // JS_CFUNC_DEF(name, length, func): magic=0, u.func = { length, generic, func }.
    static SUBPROCESS_PROTO_FUNCS: [JSCFunctionListEntry; 2] = [
        JSCFunctionListEntry {
            name: c"kill".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc {
                length: 0,
                cproto: 0, /* JS_CFUNC_generic */
                _pad: [0; 6],
                cfunc: js_subprocess_kill,
            },
        },
        JSCFunctionListEntry {
            name: c"write".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc {
                length: 1,
                cproto: 0, /* JS_CFUNC_generic */
                _pad: [0; 6],
                cfunc: js_subprocess_write,
            },
        },
    ];
}

unsafe extern "C" fn js_sofuu_spawn(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || !qjs::is_object(*argv) {
        return qjs::JS_ThrowTypeError(ctx, c"spawn requires options object".as_ptr());
    }

    let opts = *argv;
    let cmd_val = qjs::sofuu_js_get_property_str(ctx, opts, c"command".as_ptr());
    if qjs::is_undefined(cmd_val) {
        return qjs::JS_ThrowTypeError(ctx, c"options.command is required".as_ptr());
    }

    let cmd_ptr = qjs::sofuu_js_to_cstring(ctx, cmd_val);
    if cmd_ptr.is_null() {
        qjs::sofuu_js_free_value(ctx, cmd_val);
        return qjs::JS_ThrowTypeError(ctx, c"command must be a string".as_ptr());
    }
    let command = CString::new(CStr::from_ptr(cmd_ptr).to_bytes()).unwrap_or_default();
    qjs::sofuu_js_free_cstring(ctx, cmd_ptr);
    qjs::sofuu_js_free_value(ctx, cmd_val);

    // Allocate + size the request; failures below just drop it.
    let mut req = Box::new(SpawnReq {
        ctx,
        process: ptr::null_mut(),
        stdin_pipe: ptr::null_mut(),
        stdout_pipe: ptr::null_mut(),
        stderr_pipe: ptr::null_mut(),
        on_stdout: qjs::sofuu_js_undefined(),
        on_stderr: qjs::sofuu_js_undefined(),
        on_exit: qjs::sofuu_js_undefined(),
        self_obj: qjs::sofuu_js_undefined(),
        command,
        args: Vec::new(),
        args_ptr: Vec::new(),
        cwd: None,
        opts: UvProcessOptions {
            exit_cb: Some(on_process_exit),
            file: ptr::null(),
            args: ptr::null_mut(),
            env: ptr::null_mut(),
            cwd: ptr::null(),
            flags: 0,
            stdio_count: 0,
            stdio: ptr::null_mut(),
            uid: 0,
            gid: 0,
        },
        stdio: [
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
        ],
        done: 0,
        closing: 0,
    });

    let mut args_val = qjs::sofuu_js_get_property_str(ctx, opts, c"args".as_ptr());
    if qjs::JS_IsArray(ctx, args_val) != 0 {
        let len_val = qjs::sofuu_js_get_property_str(ctx, args_val, c"length".as_ptr());
        let mut len: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut len, len_val);
        qjs::sofuu_js_free_value(ctx, len_val);

        req.args.push(req.command.clone());
        for i in 0..len {
            let el = qjs::JS_GetPropertyUint32(ctx, args_val, i);
            let el_ptr = qjs::sofuu_js_to_cstring(ctx, el);
            if el_ptr.is_null() {
                /* Non-string arg — clean up and reject. */
                qjs::sofuu_js_free_value(ctx, el);
                qjs::sofuu_js_free_value(ctx, args_val);
                drop(req);
                return qjs::JS_ThrowTypeError(ctx, c"spawn args must be strings".as_ptr());
            }
            req.args.push(CString::new(CStr::from_ptr(el_ptr).to_bytes()).unwrap_or_default());
            qjs::sofuu_js_free_cstring(ctx, el_ptr);
            qjs::sofuu_js_free_value(ctx, el);
        }
    } else {
        req.args.push(req.command.clone());
    }
    qjs::sofuu_js_free_value(ctx, args_val);

    /* Callbacks: read AFTER args (C peeks JS props in the same order) */
    let _ = &mut args_val; // (args_val already freed — C frees it at the same point)
    req.on_stdout = qjs::sofuu_js_get_property_str(ctx, opts, c"onStdout".as_ptr());
    req.on_stderr = qjs::sofuu_js_get_property_str(ctx, opts, c"onStderr".as_ptr());
    req.on_exit = qjs::sofuu_js_get_property_str(ctx, opts, c"onExit".as_ptr());

    // ── uv side ─────────────────────────────────────────────────────
    let loop_ = sofuu_loop_get();
    // SAFETY: pipe/process storages malloc'd at their true size.
    let stdin_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stdout_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stderr_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let process = libc::malloc(uv::sofuu_uv_process_size()) as *mut UvProcess;
    req.stdin_pipe = stdin_pipe;
    req.stdout_pipe = stdout_pipe;
    req.stderr_pipe = stderr_pipe;
    req.process = process;

    uv::uv_pipe_init(loop_, stdin_pipe, 0);
    uv::uv_pipe_init(loop_, stdout_pipe, 0);
    uv::uv_pipe_init(loop_, stderr_pipe, 0);
    *(stdin_pipe as *mut *mut SpawnReq) = &mut *req;
    *(stdout_pipe as *mut *mut SpawnReq) = &mut *req;
    *(stderr_pipe as *mut *mut SpawnReq) = &mut *req;

    req.args_ptr = req.args.iter().map(|a| a.as_ptr() as *mut c_char).collect();
    /* execvpe/posix_spawn walk argv until NULL — without the terminator
     * they read past the Vec (EFAULT, heap-layout dependent). */
    req.args_ptr.push(ptr::null_mut());
    req.opts.file = req.command.as_ptr();
    req.opts.args = req.args_ptr.as_mut_ptr();
    req.opts.env = ptr::null_mut();
    req.opts.flags = 0;
    req.opts.uid = 0;
    req.opts.gid = 0;

    req.stdio[0] = UvStdioContainer {
        flags: uv::UV_CREATE_PIPE | uv::UV_READABLE_PIPE,
        data: UvStdioData { stream: stdin_pipe as *mut c_void },
    };
    req.stdio[1] = UvStdioContainer {
        flags: uv::UV_CREATE_PIPE | uv::UV_WRITABLE_PIPE,
        data: UvStdioData { stream: stdout_pipe as *mut c_void },
    };
    req.stdio[2] = UvStdioContainer {
        flags: uv::UV_CREATE_PIPE | uv::UV_WRITABLE_PIPE,
        data: UvStdioData { stream: stderr_pipe as *mut c_void },
    };
    req.opts.stdio_count = 3;
    req.opts.stdio = req.stdio.as_mut_ptr();

    let cwd_val = qjs::sofuu_js_get_property_str(ctx, opts, c"cwd".as_ptr());
    if !qjs::is_undefined(cwd_val) {
        let cwd_ptr = qjs::sofuu_js_to_cstring(ctx, cwd_val);
        if cwd_ptr.is_null() {
            qjs::sofuu_js_free_value(ctx, cwd_val);
            drop(req);
            return qjs::JS_ThrowTypeError(ctx, c"cwd must be a string".as_ptr());
        }
        req.cwd = Some(CString::new(CStr::from_ptr(cwd_ptr).to_bytes()).unwrap_or_default());
        qjs::sofuu_js_free_cstring(ctx, cwd_ptr);
    }
    qjs::sofuu_js_free_value(ctx, cwd_val);
    req.opts.cwd = req.cwd.as_ref().map_or(ptr::null(), |c| c.as_ptr());

    let req_ptr = Box::into_raw(req);
    // (re-pointer the pipes now that the Box is stable)
    *(stdin_pipe as *mut *mut SpawnReq) = req_ptr;
    *(stdout_pipe as *mut *mut SpawnReq) = req_ptr;
    *(stderr_pipe as *mut *mut SpawnReq) = req_ptr;

    let r = uv::uv_spawn(loop_, process, &(*req_ptr).opts);
    if r != 0 {
        /* Cleanup on failure: release the JS callback refs and close all
         * four handles with the spawn_close_cb fence — the LAST close
         * frees the req Box AND all four handle storages. The process
         * handle MUST go through uv_close too: uv_spawn already queued it
         * in loop->handle_queue (the dequeue-on-error path is #if 0'd in
         * the vendored libuv), so freeing its storage directly would
         * leave a dangling entry that uv_walk visits later. */
        qjs::sofuu_js_free_value(ctx, (*req_ptr).on_stdout);
        qjs::sofuu_js_free_value(ctx, (*req_ptr).on_stderr);
        qjs::sofuu_js_free_value(ctx, (*req_ptr).on_exit);
        (*req_ptr).on_stdout = qjs::sofuu_js_undefined();
        (*req_ptr).on_stderr = qjs::sofuu_js_undefined();
        (*req_ptr).on_exit = qjs::sofuu_js_undefined();
        (*req_ptr).closing = 4;
        /* The process handle's data must point at the req BEFORE its close
         * callback runs (spawn_close_cb reads it) — the success path sets
         * it after uv_spawn, which we never reach here. */
        *(process as *mut *mut SpawnReq) = req_ptr;
        uv::uv_close(stdin_pipe as *mut UvHandle, Some(spawn_close_cb));
        uv::uv_close(stdout_pipe as *mut UvHandle, Some(spawn_close_cb));
        uv::uv_close(stderr_pipe as *mut UvHandle, Some(spawn_close_cb));
        uv::uv_close(process as *mut UvHandle, Some(spawn_close_cb));
        return qjs::JS_ThrowTypeError(ctx, c"spawn failed: %s".as_ptr(), uv::uv_strerror(r));
    }

    (*req_ptr).process = process;
    /* The process handle's data field must point back at the req too —
     * on_process_exit reads it (same as C's req->process.data = req). */
    *(process as *mut *mut SpawnReq) = req_ptr;
    let _ = uv::uv_read_start(stdout_pipe as *mut UvStream, Some(alloc_buffer), Some(on_stdout_read));
    let _ = uv::uv_read_start(stderr_pipe as *mut UvStream, Some(alloc_buffer), Some(on_stderr_read));

    let obj = qjs::JS_NewObjectClass(ctx, SUBPROCESS_CLASS_ID.load(Ordering::Relaxed) as c_int);
    qjs::JS_SetOpaque(obj, req_ptr as *mut c_void);
    /* Keep the object alive until the fence frees the request: spawn_free
     * needs a valid JSValue to null the opaque through. */
    (*req_ptr).self_obj = qjs::sofuu_js_dup_value(ctx, obj);

    /* The argv/cwd buffers must stay valid until exit_cb — they live in
     * the Box, which the close-callback fence frees. */
    obj
}

// ── sofuu.exec(cmd[, args[, opts]]) → Promise<{code, stdout, stderr}> ──
// Captures stdout/stderr; resolves on exit. Rejects if the child cannot
// be spawned (no exit callback runs in that case).

struct ExecReq {
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,
    process: *mut UvProcess,
    stdout_pipe: *mut UvPipe,
    stderr_pipe: *mut UvPipe,
    opts: UvProcessOptions,
    stdio: [UvStdioContainer; 3],
    argv: Vec<CString>,
    argv_ptr: Vec<*mut c_char>,
    command: CString,
    cwd: Option<CString>,
    out_buf: Vec<u8>,
    err_buf: Vec<u8>,
    done: c_int,
    closing: c_int,
}

const EXEC_CAP: usize = 64 * 1024 * 1024; /* 64MB hard cap, like C */

fn exec_capture(buf: &mut Vec<u8>, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    if buf.len() + data.len() + 1 > EXEC_CAP {
        return;
    }
    buf.extend_from_slice(data);
}

unsafe fn exec_free(req: *mut ExecReq) {
    // Both pipes (and the process) — the fence frees the storages with the Box.
    libc::free((*req).process as *mut c_void);
    libc::free((*req).stdout_pipe as *mut c_void);
    libc::free((*req).stderr_pipe as *mut c_void);
    drop(Box::from_raw(req));
}

unsafe extern "C" fn exec_close_cb(handle: *mut UvHandle) {
    let req = *(handle as *mut *mut ExecReq);
    if req.is_null() {
        return;
    }
    if (*req).closing > 0 {
        (*req).closing -= 1;
        if (*req).closing == 0 {
            exec_free(req);
        }
    }
}

unsafe extern "C" fn exec_alloc_buffer(
    _handle: *mut UvHandle,
    suggested_size: usize,
    buf: *mut UvBuf,
) {
    (*buf).base = libc::malloc(suggested_size) as *mut c_char;
    /* On OOM libuv delivers nread > 0 with a NULL base — len must be 0 so
     * the read callbacks never build a slice from NULL. */
    (*buf).len = if (*buf).base.is_null() { 0 } else { suggested_size };
}

unsafe extern "C" fn exec_on_stdout(pipe: *mut UvStream, nread: isize, buf: *const UvBuf) {
    let req = *(pipe as *mut *mut ExecReq);
    if nread > 0 {
        let slice = std::slice::from_raw_parts((*buf).base as *const u8, nread as usize);
        exec_capture(&mut (*req).out_buf, slice);
    } else if nread < 0 {
        uv::uv_read_stop(pipe);
    }
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn exec_on_stderr(pipe: *mut UvStream, nread: isize, buf: *const UvBuf) {
    let req = *(pipe as *mut *mut ExecReq);
    if nread > 0 {
        let slice = std::slice::from_raw_parts((*buf).base as *const u8, nread as usize);
        exec_capture(&mut (*req).err_buf, slice);
    } else if nread < 0 {
        uv::uv_read_stop(pipe);
    }
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn exec_on_exit(
    process: *mut UvProcess,
    exit_status: i64,
    term_signal: c_int,
) {
    let req = *(process as *mut *mut ExecReq);
    if req.is_null() || (*req).done != 0 {
        return;
    }
    (*req).done = 1;

    uv::uv_read_stop((*req).stdout_pipe as *mut UvStream);
    uv::uv_read_stop((*req).stderr_pipe as *mut UvStream);

    let ctx = (*req).ctx;
    let result = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        result,
        c"code".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, exit_status as i32),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        result,
        c"signal".as_ptr(),
        qjs::sofuu_js_new_int32(ctx, term_signal),
    );
    let out = if (*req).out_buf.is_empty() {
        qjs::sofuu_js_new_string(ctx, c"".as_ptr())
    } else {
        qjs::JS_NewStringLen(ctx, (*req).out_buf.as_ptr() as *const c_char, (*req).out_buf.len())
    };
    qjs::sofuu_js_set_property_str(ctx, result, c"stdout".as_ptr(), out);
    let err = if (*req).err_buf.is_empty() {
        qjs::sofuu_js_new_string(ctx, c"".as_ptr())
    } else {
        qjs::JS_NewStringLen(ctx, (*req).err_buf.as_ptr() as *const c_char, (*req).err_buf.len())
    };
    qjs::sofuu_js_set_property_str(ctx, result, c"stderr".as_ptr(), err);
    sofuu_promise_resolve((*req).promise, result);
    qjs::sofuu_js_free_value(ctx, result);

    (*req).closing = 3;
    uv::uv_close((*req).stdout_pipe as *mut UvHandle, Some(exec_close_cb));
    uv::uv_close((*req).stderr_pipe as *mut UvHandle, Some(exec_close_cb));
    uv::uv_close(process as *mut UvHandle, Some(exec_close_cb));
}

unsafe extern "C" fn js_sofuu_exec(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::sofuu_js_is_string(*argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"exec expects a command string".as_ptr());
    }

    let cmd_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if cmd_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let command = CString::new(CStr::from_ptr(cmd_ptr).to_bytes()).unwrap_or_default();
    qjs::sofuu_js_free_cstring(ctx, cmd_ptr);

    let mut req = Box::new(ExecReq {
        ctx,
        promise: ptr::null_mut(),
        process: ptr::null_mut(),
        stdout_pipe: ptr::null_mut(),
        stderr_pipe: ptr::null_mut(),
        opts: UvProcessOptions {
            exit_cb: Some(exec_on_exit),
            file: ptr::null(),
            args: ptr::null_mut(),
            env: ptr::null_mut(),
            cwd: ptr::null(),
            flags: 0,
            stdio_count: 0,
            stdio: ptr::null_mut(),
            uid: 0,
            gid: 0,
        },
        stdio: [
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
        ],
        argv: Vec::new(),
        argv_ptr: Vec::new(),
        command,
        cwd: None,
        out_buf: Vec::new(),
        err_buf: Vec::new(),
        done: 0,
        closing: 0,
    });

    req.argv.push(req.command.clone());

    /* argv: [command, args..., NULL] */
    if argc > 1 && qjs::JS_IsArray(ctx, *argv.add(1)) != 0 {
        let lv = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"length".as_ptr());
        let mut arg_count: u32 = 0;
        qjs::sofuu_js_to_uint32(ctx, &mut arg_count, lv);
        qjs::sofuu_js_free_value(ctx, lv);

        for i in 0..arg_count {
            let el = qjs::JS_GetPropertyUint32(ctx, *argv.add(1), i);
            if qjs::sofuu_js_is_string(el) == 0 {
                qjs::sofuu_js_free_value(ctx, el);
                drop(req);
                return qjs::JS_ThrowTypeError(ctx, c"exec args must be strings".as_ptr());
            }
            let s = qjs::sofuu_js_to_cstring(ctx, el);
            req.argv.push(CString::new(CStr::from_ptr(s).to_bytes()).unwrap_or_default());
            qjs::sofuu_js_free_cstring(ctx, s);
            qjs::sofuu_js_free_value(ctx, el);
        }
    }

    /* opts: { cwd } */
    if argc > 2 && qjs::is_object(*argv.add(2)) {
        let cwd_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(2), c"cwd".as_ptr());
        if qjs::sofuu_js_is_string(cwd_val) != 0 {
            let cs = qjs::sofuu_js_to_cstring(ctx, cwd_val);
            req.cwd = Some(CString::new(CStr::from_ptr(cs).to_bytes()).unwrap_or_default());
            qjs::sofuu_js_free_cstring(ctx, cs);
        }
        qjs::sofuu_js_free_value(ctx, cwd_val);
    }

    // ── uv side ─────────────────────────────────────────────────────
    let loop_ = sofuu_loop_get();
    let stdout_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stderr_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let process = libc::malloc(uv::sofuu_uv_process_size()) as *mut UvProcess;
    req.stdout_pipe = stdout_pipe;
    req.stderr_pipe = stderr_pipe;
    req.process = process;

    uv::uv_pipe_init(loop_, stdout_pipe, 0);
    uv::uv_pipe_init(loop_, stderr_pipe, 0);

    req.argv_ptr = req.argv.iter().map(|a| a.as_ptr() as *mut c_char).collect();
    /* Same NULL terminator requirement as sofuu.spawn (execvpe walks argv). */
    req.argv_ptr.push(ptr::null_mut());
    req.opts.file = req.command.as_ptr();
    req.opts.args = req.argv_ptr.as_mut_ptr();
    req.opts.cwd = req.cwd.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    req.opts.stdio_count = 3;
    req.opts.stdio = req.stdio.as_mut_ptr();
    /* stdio[0] (child stdin) is ignored → /dev/null; we only capture
     * stdout/stderr. */
    req.stdio[0] = UvStdioContainer { flags: uv::UV_IGNORE, data: UvStdioData { stream: ptr::null_mut() } };
    req.stdio[1] = UvStdioContainer {
        flags: uv::UV_CREATE_PIPE | uv::UV_WRITABLE_PIPE,
        data: UvStdioData { stream: stdout_pipe as *mut c_void },
    };
    req.stdio[2] = UvStdioContainer {
        flags: uv::UV_CREATE_PIPE | uv::UV_WRITABLE_PIPE,
        data: UvStdioData { stream: stderr_pipe as *mut c_void },
    };

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    req.promise = out;

    let req_ptr = Box::into_raw(req);
    *(stdout_pipe as *mut *mut ExecReq) = req_ptr;
    *(stderr_pipe as *mut *mut ExecReq) = req_ptr;

    let r = uv::uv_spawn(loop_, process, &(*req_ptr).opts);
    if r != 0 {
        sofuu_promise_reject((*req_ptr).promise, qjs::sofuu_js_new_string(ctx, uv::uv_strerror(r)));
        /* All three handles go through uv_close (the process handle too —
         * uv_spawn queued it even though it failed; see js_sofuu_spawn). */
        (*req_ptr).closing = 3;
        /* exec_close_cb reads handle->data — set it for the process handle
         * (the success path does this after uv_spawn). */
        *(process as *mut *mut ExecReq) = req_ptr;
        uv::uv_close(stdout_pipe as *mut UvHandle, Some(exec_close_cb));
        uv::uv_close(stderr_pipe as *mut UvHandle, Some(exec_close_cb));
        uv::uv_close(process as *mut UvHandle, Some(exec_close_cb));
        return promise;
    }

    (*req_ptr).process = process;
    /* process handle data → req (exec_on_exit reads it). */
    *(process as *mut *mut ExecReq) = req_ptr;
    uv::uv_read_start(stdout_pipe as *mut UvStream, Some(exec_alloc_buffer), Some(exec_on_stdout));
    uv::uv_read_start(stderr_pipe as *mut UvStream, Some(exec_alloc_buffer), Some(exec_on_stderr));

    promise
}

// ── Registration (C symbol replacement for mod_subprocess_register) ──

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_subprocess_register(ctx: *mut JSContext) {
    let mut class_id: qjs::JSClassID = 0;
    qjs::JS_NewClassID(&mut class_id);
    SUBPROCESS_CLASS_ID.store(class_id, Ordering::Relaxed);

    let class_def = qjs::JSClassDef {
        class_name: c"Subprocess".as_ptr(),
        finalizer: Some(js_subprocess_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &class_def);

    let proto = qjs::sofuu_js_new_object(ctx);
    SUBPROCESS_PROTO_FUNCS.with(|f| qjs::JS_SetPropertyFunctionList(ctx, proto, f.as_ptr(), 2));
    qjs::JS_SetClassProto(ctx, class_id, proto);

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
    }

    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu_obj,
        c"spawn".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_sofuu_spawn, c"spawn".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu_obj,
        c"exec".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_sofuu_exec, c"exec".as_ptr(), 3),
    );
    qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::{self, CtxPtr};

    /// Audit P0-2 regression: a failed uv_spawn must throw a TypeError and
    /// tear down cleanly — the old code freed the process storage directly
    /// AND again in spawn_free via the pipe-close fence (double free).
    /// Also covers P0-6: after a normal exit, write()/kill() on the kept
    /// JS object must throw "process already exited" instead of touching
    /// the freed request.
    #[test]
    fn spawn_failure_and_post_exit_use_are_clean() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var phase1 = 'no-throw', exited = false, after = 'none'; \
try { sofuu.spawn({ command: 'sofuu_no_such_binary_zzz', args: [] }); } \
catch (e) { phase1 = (e && e.message) ? e.message : String(e); } \
var proc = sofuu.spawn({ command: '/bin/sleep', args: ['0.05'], \
onExit: function () { exited = true; } });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-audit>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "spawn setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: the failed spawn's pipes close (fence → spawn_free — this
        // is where the old code double-freed), then sleep exits.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let phase1 = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"phase1".as_ptr()) };
        let phase1_s = unsafe { qjs::sofuu_js_to_cstring(ctx, phase1) };
        let msg = if phase1_s.is_null() {
            String::new()
        } else {
            // SAFETY: phase1_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(phase1_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, phase1_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, phase1) };
        assert!(
            msg.starts_with("spawn failed"),
            "failed spawn must throw a TypeError naming the failure, got: {msg}"
        );

        let exited = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"exited".as_ptr()) };
        assert!(unsafe { qjs::JS_ToBool(ctx, exited) } == 1, "onExit must have fired");
        unsafe { qjs::sofuu_js_free_value(ctx, exited) };

        // Post-exit write(): the fence already freed the request and nulled
        // the opaque — this must be a clean TypeError, not a UAF.
        let post = c"try { proc.write('x'); after = 'no-throw'; } \
catch (e) { after = (e && e.message) ? e.message : 'caught'; }";
        let r2 = unsafe {
            qjs::JS_Eval(
                ctx,
                post.as_ptr(),
                post.to_bytes().len(),
                c"<spawn-audit-2>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r2) });
        unsafe { qjs::sofuu_js_free_value(ctx, r2) };
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let after = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"after".as_ptr()) };
        let after_s = unsafe { qjs::sofuu_js_to_cstring(ctx, after) };
        let after_msg = if after_s.is_null() {
            String::new()
        } else {
            // SAFETY: after_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(after_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, after_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, after) };
        assert_eq!(after_msg, "process already exited", "post-exit write must throw cleanly");

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }
}

