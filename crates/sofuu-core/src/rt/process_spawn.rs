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
    UvStream, UvTimer, UvWriteReq,
};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject, sofuu_promise_reject_str,
    sofuu_promise_resolve, PromiseHandle,
};

// ── P1-10 (AUDIT-2026-09-07): the child-environment allowlist ────────
//
// Children (sofuu.spawn, sofuu.exec, MCP servers) no longer inherit the
// FULL parent environment. The host process carries provider keys in env
// (rt/ai.rs reads OPENAI_API_KEY et al), and a spawned child — a
// third-party MCP server above all — could exfiltrate them with one
// `console.log(process.env)`. Every uv_spawn site builds the child env
// through build_child_env: the parent environment filtered to a
// functional allowlist (paths, locale, terminal identity — no
// credentials), optionally extended by the JS caller's `env` option
// ({ NAME: "value" } sets/replaces, { NAME: null } removes).
const CHILD_ENV_UNIX: &[&str] = &[
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SHELL", "USER", "LOGNAME", "LANG", "LANGUAGE",
    "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_TIME", "LC_NUMERIC", "LC_MESSAGES", "TERM",
    "COLORTERM", "TZ", "PWD", "SSH_AUTH_SOCK", "DISPLAY", "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
];

const CHILD_ENV_WINDOWS: &[&str] = &[
    "PATH", "SystemRoot", "SystemDrive", "windir", "ComSpec", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "ProgramData",
    "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles",
    "CommonProgramFiles(x86)", "USERNAME", "COMPUTERNAME", "OS", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
];

pub(crate) const CHILD_ENV_ERR: &CStr = c"env values must be strings or null";

fn child_env_name_eq(a: &str, b: &str) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

fn child_env_del(entries: &mut Vec<CString>, name: &str) {
    entries.retain(|e| match e.to_string_lossy().split_once('=') {
        Some((n, _)) => !child_env_name_eq(n, name),
        None => true,
    });
}

fn child_env_set(entries: &mut Vec<CString>, name: &str, value: &str) {
    child_env_del(entries, name);
    if let Ok(c) = CString::new(format!("{}={}", name, value)) {
        entries.push(c);
    }
}

/// Build the child environment for the three spawn sites. Returns the
/// "NAME=VALUE" C strings; the caller builds the NULL-terminated pointer
/// array and keeps the Vec alive across uv_spawn (libuv consumes env
/// synchronously — posix_spawn/execve run inside the call).
pub(crate) fn build_child_env(
    ctx: *mut JSContext,
    opts: JSValueConst,
) -> Result<Vec<CString>, &'static CStr> {
    let allow: &[&str] = if cfg!(windows) { CHILD_ENV_WINDOWS } else { CHILD_ENV_UNIX };
    let mut entries: Vec<CString> = Vec::new();

    for (name_os, val_os) in std::env::vars_os() {
        let name = name_os.to_string_lossy();
        if !allow.iter().any(|a| child_env_name_eq(a, &name)) {
            continue;
        }
        if let Ok(c) = CString::new(format!("{}={}", name, val_os.to_string_lossy())) {
            entries.push(c);
        }
    }

    /* Caller overrides: opts.env — { NAME: "value" | null }. */
    // SAFETY: ctx is live for the whole call (same contract as the js_*
    // entry points); every JSValue produced is freed before return.
    unsafe {
        if qjs::is_object(opts) {
            let env_val = qjs::sofuu_js_get_property_str(ctx, opts, c"env".as_ptr());
            if qjs::is_object(env_val) {
                let mut props: *mut qjs::JSPropertyEnum = ptr::null_mut();
                let mut nprops: u32 = 0;
                if qjs::JS_GetOwnPropertyNames(
                    ctx,
                    &mut props,
                    &mut nprops,
                    env_val,
                    qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_ENUM_ONLY,
                ) != 0
                {
                    qjs::sofuu_js_free_value(ctx, env_val);
                    return Err(CHILD_ENV_ERR);
                }
                let mut err: bool = false;
                for p in 0..nprops as usize {
                    // SAFETY: props array of nprops entries, malloc'd by quickjs.
                    let atom = (*props.add(p)).atom;
                    let k = qjs::JS_AtomToCString(ctx, atom);
                    let v = qjs::sofuu_js_get_property(ctx, env_val, atom);
                    qjs::JS_FreeAtom(ctx, atom);
                    if !k.is_null() {
                        let name = CStr::from_ptr(k).to_string_lossy().into_owned();
                        qjs::sofuu_js_free_cstring(ctx, k);
                        if qjs::sofuu_js_is_string(v) != 0 {
                            let vs = qjs::sofuu_js_to_cstring(ctx, v);
                            if !vs.is_null() {
                                // SAFETY: vs is a live CString from QuickJS.
                                let val = CStr::from_ptr(vs).to_string_lossy();
                                child_env_set(&mut entries, &name, &val);
                                qjs::sofuu_js_free_cstring(ctx, vs);
                            }
                        } else if !qjs::is_null(v) && !qjs::is_undefined(v) {
                            err = true;
                        }
                    }
                    qjs::sofuu_js_free_value(ctx, v);
                    if err {
                        break;
                    }
                }
                qjs::js_free(ctx, props as *mut c_void);
                if err {
                    qjs::sofuu_js_free_value(ctx, env_val);
                    return Err(CHILD_ENV_ERR);
                }
            }
            qjs::sofuu_js_free_value(ctx, env_val);
        }
    }

    Ok(entries)
}

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
    /* P1-10: the filtered child environment — the CStrings own the memory
     * the opts.env pointer array points into; both must outlive uv_spawn. */
    env: Vec<CString>,
    env_ptr: Vec<*mut c_char>,
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
    crate::rt::event_loop::untrack_handle(handle);
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
            if qjs::is_exception(str_v) {
                /* P1-11: a failed build (OOM or a >2^30−1-char chunk; invalid
                 * UTF-8 does NOT take this path — QuickJS maps it to U+FFFD)
                 * leaves an exception-tagged value + a pending exception.
                 * Handing the tag to JS_Call is UB, and there is no channel
                 * to deliver the chunk while allocation is failing — clear
                 * the exception and drop it. */
                let exc = qjs::sofuu_js_get_exception((*req).ctx);
                qjs::sofuu_js_free_value((*req).ctx, exc);
            } else {
                let ret = qjs::JS_Call((*req).ctx, (*req).on_stdout, qjs::sofuu_js_undefined(), 1, &str_v);
                qjs::sofuu_js_free_value((*req).ctx, ret);
                sofuu_flush_jobs((*req).ctx);
            }
            qjs::sofuu_js_free_value((*req).ctx, str_v);
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
            if qjs::is_exception(str_v) {
                /* P1-11: same contract as on_stdout_read. */
                let exc = qjs::sofuu_js_get_exception((*req).ctx);
                qjs::sofuu_js_free_value((*req).ctx, exc);
            } else {
                let ret = qjs::JS_Call((*req).ctx, (*req).on_stderr, qjs::sofuu_js_undefined(), 1, &str_v);
                qjs::sofuu_js_free_value((*req).ctx, ret);
                sofuu_flush_jobs((*req).ctx);
            }
            qjs::sofuu_js_free_value((*req).ctx, str_v);
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
    term_signal: c_int,
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
        /* proc-2: onExit receives (code, signal) — term_signal was parsed
         * but never delivered, so children killed by a signal reported
         * signal 0 forever. */
        let vals = [
            qjs::sofuu_js_new_int32((*req).ctx, exit_status as i32),
            qjs::sofuu_js_new_int32((*req).ctx, term_signal),
        ];
        let ret = qjs::JS_Call((*req).ctx, (*req).on_exit, qjs::sofuu_js_undefined(), 2, vals.as_ptr());
        qjs::sofuu_js_free_value((*req).ctx, vals[0]);
        qjs::sofuu_js_free_value((*req).ctx, vals[1]);
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

/// Write payload — owned by the Box, recovered from `uv_write_t.data`
/// (libuv reserves req->data for user state and `uv__req_init` never
/// touches offset 0 on Unix; embedding a uv_write_t mirror inside the req
/// region corrupts it — uv_write_t is larger than the mirror and
/// `uv__req_init` stores `type` at offset 8, which the mirror reserved for
/// buf.base. Same M5 pattern as mcp.rs `write_str_dup`.)
struct WriteReq {
    data: *mut c_char,
}

unsafe extern "C" fn on_write_done(req: *mut UvWriteReq, _status: c_int) {
    // SAFETY: req->data (first field of uv_req_t) was set at write time.
    let wr = *(req as *mut *mut WriteReq);
    libc::free((*wr).data as *mut c_void);
    libc::free(req as *mut c_void);
    drop(Box::from_raw(wr));
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

    let data = libc::malloc(len) as *mut c_char;
    if data.is_null() {
        qjs::sofuu_js_free_cstring(ctx, str_v);
        return qjs::JS_ThrowTypeError(ctx, c"out of memory".as_ptr());
    }
    // SAFETY: copies len bytes from the JS string into the new buffer.
    std::ptr::copy_nonoverlapping(str_v as *const u8, data as *mut u8, len);
    qjs::sofuu_js_free_cstring(ctx, str_v);

    /* P0-1 (AUDIT-2026-09-07): the write req is a full-size uv_write_t with
     * the payload Box hung off req->data (offset 0 — uv__req_init never
     * touches it); the payload itself is a separate malloc freed in
     * on_write_done. */
    let wr = Box::into_raw(Box::new(WriteReq { data }));
    let wreq = libc::malloc(uv::sofuu_uv_write_size()) as *mut UvWriteReq;
    if wreq.is_null() {
        libc::free(data as *mut c_void);
        drop(Box::from_raw(wr));
        return qjs::JS_ThrowTypeError(ctx, c"out of memory".as_ptr());
    }
    *(wreq as *mut *mut WriteReq) = wr;
    let b = UvBuf { base: data, len };

    let rc = uv::uv_write(
        wreq,
        (*req).stdin_pipe as *mut UvStream,
        &b,
        1,
        Some(on_write_done),
    );
    if rc < 0 {
        /* Synchronous failure: libuv will NOT call on_write_done — free the
         * req and its payload here or they leak. */
        libc::free(data as *mut c_void);
        libc::free(wreq as *mut c_void);
        drop(Box::from_raw(wr));
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

    /* P1-10 (AUDIT-2026-09-07): build the filtered child env FIRST so a bad
     * `env` option rejects before any uv resource is allocated. */
    let child_env = match build_child_env(ctx, opts) {
        Ok(e) => e,
        Err(msg) => {
            return qjs::JS_ThrowTypeError(ctx, c"spawn: %s".as_ptr(), msg.as_ptr());
        }
    };

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
        env: child_env,
        env_ptr: Vec::new(),
        opts: UvProcessOptions {
            exit_cb: Some(on_process_exit),
            file: ptr::null(),
            args: ptr::null_mut(),
            env: ptr::null_mut(),
            cwd: ptr::null(),
            flags: 0,
            stdio_count: 0,
            stdio: ptr::null_mut(),
            #[cfg(unix)]
            uid: 0,
            #[cfg(unix)]
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
    /* P1-10: the filtered child env (never the full parent environment). */
    req.env_ptr = req.env.iter().map(|e| e.as_ptr() as *mut c_char).collect();
    req.env_ptr.push(ptr::null_mut());
    req.opts.env = req.env_ptr.as_mut_ptr();
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
    /* F-2: all four handles are armed per-ctx — a multi-engine teardown
     * must close them. Their lifetime is fence-based (spawn_close_cb's
     * counter frees everything), so they track with a None close cb: the
     * shutdown just requests the close, parity with the last-engine walk. */
    unsafe {
        crate::rt::event_loop::track_handle(ctx, process as *mut UvHandle, None);
        crate::rt::event_loop::track_handle(ctx, stdin_pipe as *mut UvHandle, None);
        crate::rt::event_loop::track_handle(ctx, stdout_pipe as *mut UvHandle, None);
        crate::rt::event_loop::track_handle(ctx, stderr_pipe as *mut UvHandle, None);
    }
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
    /* P1-10: filtered child env; keeps the memory opts.env points into. */
    env: Vec<CString>,
    env_ptr: Vec<*mut c_char>,
    out_buf: Vec<u8>,
    err_buf: Vec<u8>,
    /* proc-1: exec() drops chunks past EXEC_CAP (memory safety) — these
     * flags make the loss visible as stdoutTruncated/stderrTruncated. */
    out_trunc: c_int,
    err_trunc: c_int,
    /* proc-3: exec(opts).timeout (ms) — a uv_timer that SIGTERMs the child
     * so the promise settles; NULL when no timeout was requested. */
    timer: *mut UvTimer,
    done: c_int,
    closing: c_int,
}

const EXEC_CAP: usize = 64 * 1024 * 1024; /* 64MB hard cap, like C */

fn exec_capture(buf: &mut Vec<u8>, trunc: &mut c_int, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    if buf.len() + data.len() + 1 > EXEC_CAP {
        /* proc-1: the drop is now recorded — exec_on_exit surfaces it as
         * stdoutTruncated/stderrTruncated instead of silently shortening
         * the captured output. */
        *trunc = 1;
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
    crate::rt::event_loop::untrack_handle(handle);
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
        exec_capture(&mut (*req).out_buf, &mut (*req).out_trunc, slice);
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
        exec_capture(&mut (*req).err_buf, &mut (*req).err_trunc, slice);
    } else if nread < 0 {
        uv::uv_read_stop(pipe);
    }
    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

/* proc-3: exec(opts).timeout — fired once when the child outlives the
 * deadline; SIGTERM gives the child a real chance to clean up, and
 * exec_on_exit then settles the promise with signal 15. */
unsafe extern "C" fn exec_timer_cb(timer: *mut UvTimer) {
    let req = *(timer as *mut *mut ExecReq);
    if req.is_null() || (*req).done != 0 {
        uv::uv_timer_stop(timer);
        return;
    }
    uv::uv_process_kill((*req).process, libc::SIGTERM);
}

/* The timer handle has its own close callback — it only frees the handle
 * storage and never touches the closing counter (which tracks the two
 * pipes + the process). */
unsafe extern "C" fn exec_timer_close_cb(timer: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(timer);
    libc::free(timer as *mut c_void);
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
    /* proc-1: report capture-cap drops instead of silently shortening. */
    qjs::sofuu_js_set_property_str(
        ctx,
        result,
        c"stdoutTruncated".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, (*req).out_trunc),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        result,
        c"stderrTruncated".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, (*req).err_trunc),
    );
    let out = if (*req).out_buf.is_empty() {
        qjs::sofuu_js_new_string(ctx, c"".as_ptr())
    } else {
        qjs::JS_NewStringLen(ctx, (*req).out_buf.as_ptr() as *const c_char, (*req).out_buf.len())
    };
    let err = if (*req).err_buf.is_empty() {
        qjs::sofuu_js_new_string(ctx, c"".as_ptr())
    } else {
        qjs::JS_NewStringLen(ctx, (*req).err_buf.as_ptr() as *const c_char, (*req).err_buf.len())
    };
    if qjs::is_exception(out) || qjs::is_exception(err) {
        /* P1-11: a failed string build (OOM, or output beyond the 2^30−1
         * char cap; invalid UTF-8 does NOT take this path — QuickJS maps it
         * to U+FFFD) leaves an exception-tagged value + a pending exception.
         * Passing the tag into JS_SetPropertyStr/promise resolution is UB —
         * clear the exception and reject the exec promise instead. */
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, out);
        qjs::sofuu_js_free_value(ctx, err);
        sofuu_promise_reject_str(
            (*req).promise,
            c"subprocess output could not be converted to a string (out of memory)".as_ptr(),
        );
    } else {
        qjs::sofuu_js_set_property_str(ctx, result, c"stdout".as_ptr(), out);
        qjs::sofuu_js_set_property_str(ctx, result, c"stderr".as_ptr(), err);
        sofuu_promise_resolve((*req).promise, result);
    }
    qjs::sofuu_js_free_value(ctx, result);

    /* proc-3: the timeout timer is no longer needed once the child exited
     * (the close cb frees only the handle — closing stays 3). */
    if !(*req).timer.is_null() {
        uv::uv_timer_stop((*req).timer);
        uv::uv_close((*req).timer as *mut UvHandle, Some(exec_timer_close_cb));
        (*req).timer = ptr::null_mut();
    }

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

    /* P1-10: filtered child env; exec's opts (argv[2]) may carry `env`. */
    let child_env = if argc > 2 && {
        // SAFETY: argc > 2 checked — argv[2] is in range.
        qjs::is_object(*argv.add(2))
    } {
        // SAFETY: argv[2] in range and live for the call.
        match build_child_env(ctx, *argv.add(2)) {
            Ok(e) => e,
            Err(msg) => {
                return qjs::JS_ThrowTypeError(ctx, c"exec: %s".as_ptr(), msg.as_ptr());
            }
        }
    } else {
        // SAFETY: no opts object — still apply the allowlist.
        build_child_env(ctx, qjs::sofuu_js_undefined()).unwrap_or_default()
    };

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
            #[cfg(unix)]
            uid: 0,
            #[cfg(unix)]
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
        env: child_env,
        env_ptr: Vec::new(),
        out_buf: Vec::new(),
        err_buf: Vec::new(),
        out_trunc: 0,
        err_trunc: 0,
        timer: ptr::null_mut(),
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

    /* opts: { cwd, timeout (ms, proc-3) } */
    let mut timeout_ms: i64 = 0;
    if argc > 2 && qjs::is_object(*argv.add(2)) {
        let cwd_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(2), c"cwd".as_ptr());
        if qjs::sofuu_js_is_string(cwd_val) != 0 {
            let cs = qjs::sofuu_js_to_cstring(ctx, cwd_val);
            req.cwd = Some(CString::new(CStr::from_ptr(cs).to_bytes()).unwrap_or_default());
            qjs::sofuu_js_free_cstring(ctx, cs);
        }
        qjs::sofuu_js_free_value(ctx, cwd_val);

        let tmo_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(2), c"timeout".as_ptr());
        if !qjs::is_undefined(tmo_val) && !qjs::is_null(tmo_val) {
            let mut t: i64 = 0;
            if qjs::JS_ToInt64(ctx, &mut t, tmo_val) == 0 && t > 0 {
                timeout_ms = t;
            }
        }
        qjs::sofuu_js_free_value(ctx, tmo_val);
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
    /* P1-10: the filtered child env (never the full parent environment). */
    req.env_ptr = req.env.iter().map(|e| e.as_ptr() as *mut c_char).collect();
    req.env_ptr.push(ptr::null_mut());
    req.opts.env = req.env_ptr.as_mut_ptr();
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

    /* proc-3: arm the timeout timer AFTER into_raw — the callback reads
     * handle->data. Fires once; exec_on_exit stops+closes it. */
    if timeout_ms > 0 {
        let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
        uv::uv_timer_init(loop_, timer);
        *(timer as *mut *mut ExecReq) = req_ptr;
        (*req_ptr).timer = timer;
        uv::uv_timer_start(timer, Some(exec_timer_cb), timeout_ms as u64, 0);
    }

    let r = uv::uv_spawn(loop_, process, &(*req_ptr).opts);
    if r != 0 {
        sofuu_promise_reject((*req_ptr).promise, qjs::sofuu_js_new_string(ctx, uv::uv_strerror(r)));
        /* All three handles go through uv_close (the process handle too —
         * uv_spawn queued it even though it failed; see js_sofuu_spawn). */
        (*req_ptr).closing = 3;
        /* exec_close_cb reads handle->data — set it for the process handle
         * (the success path does this after uv_spawn). */
        *(process as *mut *mut ExecReq) = req_ptr;
        /* proc-3: release the timeout timer too, or the Box would outlive
         * its last closer (leak) with the timer armed. */
        if !(*req_ptr).timer.is_null() {
            uv::uv_timer_stop((*req_ptr).timer);
            uv::uv_close(
                (*req_ptr).timer as *mut UvHandle,
                Some(exec_timer_close_cb),
            );
            (*req_ptr).timer = ptr::null_mut();
        }
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

    /* F-2: all four handles (3 fence-freed + the timeout timer) are armed
     * per-ctx — a multi-engine teardown must close them. The fence handles
     * track with None (parity with the last-engine walk); the timer's own
     * close cb frees its storage. */
    unsafe {
        crate::rt::event_loop::track_handle(ctx, process as *mut UvHandle, None);
        crate::rt::event_loop::track_handle(ctx, stdout_pipe as *mut UvHandle, None);
        crate::rt::event_loop::track_handle(ctx, stderr_pipe as *mut UvHandle, None);
        if !(*req_ptr).timer.is_null() {
            crate::rt::event_loop::track_handle(
                ctx,
                (*req_ptr).timer as *mut UvHandle,
                Some(exec_timer_close_cb),
            );
        }
    }

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
        let _loop_guard = crate::rt::test_loop_lock();
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

    /// Audit P0-1 regression: Subprocess.write() must deliver the payload
    /// through a full-size uv_write_t whose payload Box hangs off req->data.
    /// The old code embedded a uv_write_t mirror inside the req allocation —
    /// uv__req_init's `type` store landed on buf.base, libuv wrote the
    /// payload through the resulting wild pointer and on_write_done freed
    /// the clobbered value (crash/heap corruption on essentially every
    /// write). head -n 2 echoes exactly two lines back, proving both queued
    /// writes arrived intact; with the old layout this test crashes or
    /// aborts in malloc before reaching the assertions.
    #[test]
    fn subprocess_write_delivers_payload() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var chunks = '', exited = false, code = -1; \
var p = sofuu.spawn({ command: '/usr/bin/head', args: ['-n', '2'], \
onStdout: function (s) { chunks += s; }, \
onExit: function (c) { exited = true; code = c; } }); \
p.write('hello-p0-1\\n'); \
p.write('second-line\\n');";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-p0-1>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "spawn+write must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: both writes flush, head echoes two lines and exits, the
        // close fence tears the request down (frees payload + req + Box).
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let chunks_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"chunks".as_ptr()) };
        let chunks_s = unsafe { qjs::sofuu_js_to_cstring(ctx, chunks_v) };
        let got = if chunks_s.is_null() {
            String::new()
        } else {
            // SAFETY: chunks_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(chunks_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, chunks_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, chunks_v) };
        assert_eq!(
            got, "hello-p0-1\nsecond-line\n",
            "both writes must reach the child byte-intact"
        );

        let exited_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"exited".as_ptr()) };
        assert!(unsafe { qjs::JS_ToBool(ctx, exited_v) } == 1, "onExit must have fired");
        unsafe { qjs::sofuu_js_free_value(ctx, exited_v) };

        let code_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"code".as_ptr()) };
        let mut code: i32 = -1;
        // SAFETY: ctx live; code_v is a JS number (head exit status).
        unsafe { qjs::JS_ToInt32(ctx, &mut code, code_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, code_v) };
        assert_eq!(code, 0, "head must exit cleanly after consuming both lines");

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Audit P1-10 regression: children receive the FILTERED child env —
    /// never the full parent environment. A secret planted in the parent
    /// env must NOT reach the child, an allowlisted var (PATH) must, and an
    /// explicit `env` override must.
    #[test]
    #[cfg(unix)]
    fn spawn_env_is_filtered() {
        // SAFETY: no other test reads this name; the process env write is
        // the same pattern the chat.rs config tests already use.
        std::env::set_var("SOFUU_P1_10_SECRET", "hunter2-secret-value");
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var envOut = '', exited = false, code = -1; \
sofuu.spawn({ command: '/usr/bin/env', args: [], \
env: { SOFUU_P1_10_PASSTHROUGH: 'yes' }, \
onStdout: function (s) { envOut += s; }, \
onExit: function (c) { exited = true; code = c; } });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-p1-10>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "spawn with env option must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: env prints, child exits, close fence tears the req down.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let out_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"envOut".as_ptr()) };
        let out_s = unsafe { qjs::sofuu_js_to_cstring(ctx, out_v) };
        let got = if out_s.is_null() {
            String::new()
        } else {
            // SAFETY: out_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(out_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, out_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, out_v) };

        let exited_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"exited".as_ptr()) };
        assert!(unsafe { qjs::JS_ToBool(ctx, exited_v) } == 1, "onExit must have fired");
        unsafe { qjs::sofuu_js_free_value(ctx, exited_v) };

        let code_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"code".as_ptr()) };
        let mut code: i32 = -1;
        // SAFETY: ctx live; code_v is a JS number (env exit status).
        unsafe { qjs::JS_ToInt32(ctx, &mut code, code_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, code_v) };
        assert_eq!(code, 0, "env must exit cleanly");

        assert!(
            got.contains("SOFUU_P1_10_PASSTHROUGH=yes"),
            "the explicit env override must reach the child; got: {got}"
        );
        assert!(got.contains("PATH="), "allowlisted PATH must reach the child");
        assert!(
            !got.contains("SOFUU_P1_10_SECRET"),
            "a non-allowlisted parent var must NOT reach the child"
        );
        assert!(
            !got.contains("hunter2-secret-value"),
            "the planted secret value must NOT reach the child"
        );

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Audit P1-11 regression: in exec_on_exit a failed JS_NewStringLen
    /// over the captured output returns an exception-tagged JSValue that
    /// used to flow straight into JS_SetPropertyStr/promise resolution —
    /// UB. Deterministic trigger: the child's output is buffered in a
    /// Rust Vec (libc heap, invisible to the JS limit); capping the JS
    /// heap (8 MB) AFTER issuing the exec makes the 32 MB stdout string
    /// build OOM while the tiny handlers still run. (The audit's original
    /// trigger claim — binary/invalid-UTF-8 output — is wrong: QuickJS
    /// maps invalid bytes to U+FFFD; only OOM and the 2^30−1 char cap
    /// throw.) With the fix the exec promise rejects cleanly and the
    /// runtime stays usable; the positive control proves a small exec
    /// under the same limit still resolves.
    #[test]
    fn exec_rejects_when_output_string_build_ooms() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let tmp = std::env::temp_dir().join(format!("sofuu_p111x_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let big = tmp.join("big.txt");
        let small = tmp.join("small.txt");
        std::fs::write(&big, vec![b'a'; 32 * 1024 * 1024]).unwrap();
        std::fs::write(&small, b"ok").unwrap();

        let script = CString::new(format!(
            "var rejected = 'no-reject', resolved = false, posVal = '', posErr = ''; \
sofuu.exec('/bin/cat', ['{}/big.txt']).then(function (r) {{ resolved = true; }}, \
function (e) {{ rejected = (e && e.message) ? e.message : String(e); }}); \
sofuu.exec('/bin/cat', ['{}/small.txt']).then(function (r) {{ posVal = r.stdout; }}, \
function (e) {{ posErr = (e && e.message) ? e.message : String(e); }});",
            tmp.to_string_lossy(),
            tmp.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-p111>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "exec setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // P1-11 fault injection: cap the JS heap between issuing the execs
        // and draining them. exec buffers the child's bytes in a Rust Vec
        // (libc heap), so the capture itself succeeds; only the final
        // 32 MB string build in exec_on_exit fails. libc/libuv buffers are
        // not JS allocations and are unaffected by the limit.
        // SAFETY: rt is live on this thread.
        unsafe { qjs::JS_SetMemoryLimit(rt, 8 * 1024 * 1024) };

        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let rejected = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"rejected".as_ptr()) };
        let rejected_s = unsafe { qjs::sofuu_js_to_cstring(ctx, rejected) };
        let rejected_msg = if rejected_s.is_null() {
            String::new()
        } else {
            // SAFETY: rejected_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(rejected_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, rejected_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, rejected) };

        let resolved_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"resolved".as_ptr()) };
        let resolved_b = unsafe { qjs::JS_ToBool(ctx, resolved_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, resolved_v) };

        let pos_val = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"posVal".as_ptr()) };
        let pos_s = unsafe { qjs::sofuu_js_to_cstring(ctx, pos_val) };
        let pos_val_s = if pos_s.is_null() {
            String::new()
        } else {
            // SAFETY: pos_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(pos_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, pos_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, pos_val) };
        let pos_err = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"posErr".as_ptr()) };
        let pos_err_s = unsafe { qjs::sofuu_js_to_cstring(ctx, pos_err) };
        let pos_err_msg = if pos_err_s.is_null() {
            String::new()
        } else {
            // SAFETY: pos_err_s is a live CString from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(pos_err_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, pos_err_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, pos_err) };

        assert!(
            rejected_msg.contains("out of memory"),
            "OOM string build must reject with the out-of-memory error, got: {rejected_msg}"
        );
        assert!(resolved_b == 0, "the failed exec must NOT resolve");
        assert_eq!(pos_val_s, "ok", "small exec under the same limit must still resolve");
        assert_eq!(pos_err_msg, "", "positive control must not reject: {pos_err_msg}");

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Audit proc-1 regression: exec() dropped every chunk past the 64MB
    /// capture cap SILENTLY — the caller saw a short stdout with no hint
    /// that anything was lost. The cap stays (memory safety), but the
    /// result must now carry stdoutTruncated/stderrTruncated. 70MB of
    /// zeroes through the pipe flips the flag while staying under the
    /// 2^30-1 character string cap; the positive control (small output)
    /// must report stdoutTruncated === false.
    #[test]
    fn exec_flags_truncated_capture() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var res = null, err = ''; \
sofuu.exec('/usr/bin/head', ['-c', '70000000', '/dev/zero']).then(function (r) { res = r; }, \
function (e) { err = (e && e.message) ? e.message : String(e); }); \
var pres = null; \
sofuu.exec('/bin/echo', ['small-pos']).then(function (r) { pres = r; });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-proc-1>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "exec setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: head streams 70MB, both children exit, promises settle.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let err_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"err".as_ptr()) };
        let err_s = unsafe { qjs::sofuu_js_to_cstring(ctx, err_v) };
        let err = if err_s.is_null() {
            String::new()
        } else {
            // SAFETY: err_s is a live C string from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(err_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, err_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, err_v) };
        assert_eq!(err, "", "the big-output exec must resolve, not reject");

        let res_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"res".as_ptr()) };
        let trunc_v = unsafe { qjs::sofuu_js_get_property_str(ctx, res_v, c"stdoutTruncated".as_ptr()) };
        let trunc_b = unsafe { qjs::JS_ToBool(ctx, trunc_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, trunc_v) };
        let code_v = unsafe { qjs::sofuu_js_get_property_str(ctx, res_v, c"code".as_ptr()) };
        let mut code: i32 = -1;
        // SAFETY: ctx live; code_v is a JS number (head exit status).
        unsafe { qjs::JS_ToInt32(ctx, &mut code, code_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, code_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, res_v) };
        assert_eq!(code, 0, "head must exit cleanly after streaming 70MB");
        assert!(
            trunc_b == 1,
            "70MB through a 64MB capture cap must set stdoutTruncated"
        );

        let pres_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"pres".as_ptr()) };
        let ptrunc_v = unsafe { qjs::sofuu_js_get_property_str(ctx, pres_v, c"stdoutTruncated".as_ptr()) };
        let ptrunc_b = unsafe { qjs::JS_ToBool(ctx, ptrunc_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, ptrunc_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, pres_v) };
        assert!(
            ptrunc_b == 0,
            "small output must report stdoutTruncated === false (positive control)"
        );

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Audit proc-2 regression: the Subprocess exit event dropped
    /// term_signal — on_exit was invoked with (exit_status) only, so a
    /// SIGTERM/SIGKILL death was indistinguishable from a clean exit 0.
    /// The callback now receives (exit_status, term_signal); one-argument
    /// call sites keep working (additive contract).
    #[test]
    #[cfg(unix)]
    fn subprocess_on_exit_reports_signal() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var code = -1, sig = -1, exited = false; \
var p = sofuu.spawn({ command: '/bin/sh', args: ['-c', 'kill -TERM $$'], \
onExit: function (c, s) { exited = true; code = c; sig = s; } });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-proc-2>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "spawn must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: sh TERMs itself, the close fence tears the req down.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let exited_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"exited".as_ptr()) };
        assert!(unsafe { qjs::JS_ToBool(ctx, exited_v) } == 1, "onExit must have fired");
        unsafe { qjs::sofuu_js_free_value(ctx, exited_v) };

        let sig_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sig".as_ptr()) };
        let mut sig: i32 = -1;
        // SAFETY: ctx live; sig_v is a JS number (or undefined pre-fix → 0).
        unsafe { qjs::JS_ToInt32(ctx, &mut sig, sig_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, sig_v) };
        assert_eq!(sig, 15, "a SIGTERM death must reach on_exit as term_signal=15");

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Audit proc-3 regression: exec() had no timeout — a child that never
    /// exited held the promise (and the event loop) forever. opts.timeout
    /// (ms) now arms a uv_timer that SIGTERMs the child so the promise
    /// settles with signal=15; the loop must drain right after.
    #[test]
    #[cfg(unix)]
    fn exec_timeout_kills_child() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers the Subprocess class + sofuu.spawn/exec.
        unsafe { mod_subprocess_register(ctx) };

        let script = c"var res = null, err = ''; \
sofuu.exec('/bin/sleep', ['30'], { timeout: 150 }).then(function (r) { res = r; }, \
function (e) { err = (e && e.message) ? e.message : String(e); });";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<spawn-proc-3>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "exec with timeout must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: the timer fires at 150ms, SIGTERM kills sleep, the exec
        // promise settles and the loop drains (pre-fix: blocked 30s here,
        // then resolved with signal=0 → the assertion below fails).
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let global = unsafe { qjs::global_object(ctp) };
        let err_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"err".as_ptr()) };
        let err_s = unsafe { qjs::sofuu_js_to_cstring(ctx, err_v) };
        let err = if err_s.is_null() {
            String::new()
        } else {
            // SAFETY: err_s is a live C string from QuickJS.
            unsafe { std::ffi::CStr::from_ptr(err_s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, err_s) };
        unsafe { qjs::sofuu_js_free_value(ctx, err_v) };
        assert_eq!(err, "", "the timed-out exec must still resolve");

        let res_v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"res".as_ptr()) };
        let sig_v = unsafe { qjs::sofuu_js_get_property_str(ctx, res_v, c"signal".as_ptr()) };
        let mut sig: i32 = -1;
        // SAFETY: ctx live; sig_v is a JS number (SIGTERM from the timer).
        unsafe { qjs::JS_ToInt32(ctx, &mut sig, sig_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, sig_v) };
        unsafe { qjs::sofuu_js_free_value(ctx, res_v) };
        assert_eq!(sig, 15, "the timeout must kill the child with SIGTERM");
        // The loop drained well under a second of wall clock — but the
        // real proof is that it drained at all (assertions ran).

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0/M1 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }
}

