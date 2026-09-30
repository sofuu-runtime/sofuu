// rt/mcp.rs — Sofuu MCP (Model Context Protocol) module (PLAN-RUST-MIGRATION M6).
//
// Port of the deleted `src/mcp/mcp.c` (1,245 lines), semantics verbatim:
//   sofuu.mcp.connect(command) → Promise<MCPClient> (spawns a child, JSON-RPC
//   over stdin/stdout pipes; initialize handshake)
//   client.call/listTools/listResources/disconnect
//   sofuu.mcp.serve(opts) → MCPServer (stdio transport) with
//   server.tool(name, opts, handler) + server.start()
//
// THE M6 MANDATE: the client RESPONSE routing uses the Rust
// `mcp::jsonrpc::parse` (safe serde) — the retired C `json_get_field`
// string scan is GONE. The old C-scan path was kept only because the first
// Rust attempt crashed (refcount/GC inside the uv read callback); here the
// discipline is: every JSValue created in a read-callback path is freed
// within the same function, promises are the M1 Rust handles, and no jobs
// are flushed from inside a libuv callback except at the exact points the C
// code flushed (mcp_process_line's tail + initialize completion).
//
// C symbols replaced: `mod_mcp_register` (engine.c calls it unchanged).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicI32, Ordering};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvHandle, UvPipe, UvProcess, UvStream, UvWriteReq};

use crate::mcp::jsonrpc::Inbound;
use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject, sofuu_promise_resolve,
    PromiseHandle,
};

const MAX_PENDING: usize = 64;
const MCP_READ_BUF: usize = 65536;
const MAX_TOOLS: usize = 64;

/// g_next_id (C static).
static NEXT_ID: AtomicI32 = AtomicI32::new(1);

fn next_id() -> i32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── pending call tracking ───────────────────────────────────────────

#[derive(Clone, Copy)]
struct PendingCall {
    id: i32,
    promise: *mut PromiseHandle,
    ctx: *mut JSContext,
}

// ── MCP Client ──────────────────────────────────────────────────────

static MCP_CLIENT_CLASS_ID: AtomicU32 = AtomicU32::new(0);

struct McpClient {
    process: *mut UvProcess,
    stdin_pipe: *mut UvPipe,
    stdout_pipe: *mut UvPipe,
    stderr_pipe: *mut UvPipe,

    ctx: *mut JSContext,
    self_val: JSValue, /* the JS MCPClient object */

    read_buf: [u8; MCP_READ_BUF],
    read_len: usize,

    pending: [PendingCall; MAX_PENDING],
    pending_count: usize,

    initialized: c_int,   /* whether initialize handshake done */
    initialize_id: i32,   /* req id of the initialize call */
    connect_promise: *mut PromiseHandle,

    /* Lifecycle: `done` is set in mcp_on_exit; `closing` counts the 4
     * handles (stdin/stdout/stderr pipes + process) being closed — the LAST
     * close callback frees the client + the handle storages. */
    done: c_int,
    closing: c_int,
}

/// P0-2 regression witness: counts completed client frees (test builds only).
/// The disconnect bug's failure mode is heap-garbage-dependent (wild deref OR
/// a stalled fence that leaks silently), so the JS surface cannot observe it
/// — the fence completion is asserted through this counter instead.
#[cfg(test)]
static TEST_MCP_CLIENT_FREES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// P1-7 witness: the server Box must be freed exactly once per server — and
/// NEVER while a tool promise is still pending. The old closure held the
/// server as a raw int64 pointer, so the finalizer could run (Box freed)
/// while the settle callback still read the stale pointer = UAF.
#[cfg(test)]
static TEST_MCP_SERVER_FREES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

unsafe fn mcp_client_free(client: *mut McpClient) {
    #[cfg(test)]
    TEST_MCP_CLIENT_FREES.fetch_add(1, Ordering::Relaxed);
    libc::free((*client).process as *mut c_void);
    libc::free((*client).stdin_pipe as *mut c_void);
    libc::free((*client).stdout_pipe as *mut c_void);
    libc::free((*client).stderr_pipe as *mut c_void);
    drop(Box::from_raw(client));
}

/// P0-2 (AUDIT-2026-09-07): every handle closed through this callback must
/// carry the client pointer in handle->data — mcp_on_exit closes all four
/// (stdin/stdout/stderr pipes + process), and stdin/stderr never had ->data
/// written before, so the callback read stale malloc garbage and dereferenced
/// it. The connect path now stamps all four.
unsafe extern "C" fn mcp_client_close_cb(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    // SAFETY: all four client handles get ->data = client at connect.
    let client = *(handle as *mut *mut McpClient);
    if client.is_null() {
        return;
    }
    if (*client).closing > 0 {
        (*client).closing -= 1;
        if (*client).closing == 0 {
            mcp_client_free(client);
        }
    }
}

unsafe extern "C" fn mcp_client_finalizer(_rt: *mut qjs::JSRuntime, _val: JSValue) {
    /* NEVER free the client here: while the child runs, the uv handles and
     * their callbacks still point at it; once it exited, the close-callback
     * fence already freed it. */
}

// ── write plumbing (M5 pattern: payload Box in req->data) ───────────

struct WriteReq {
    data: *mut c_char,
}

unsafe extern "C" fn on_mcp_write_done(req: *mut UvWriteReq, _status: c_int) {
    // SAFETY: req->data set at send.
    let wr = *(req as *mut *mut WriteReq);
    libc::free((*wr).data as *mut c_void);
    libc::free(req as *mut c_void);
    drop(Box::from_raw(wr));
}

/// Returns libuv's rc: 0 when the write was queued, <0 on a synchronous
/// failure (EBADF/EPIPE — libuv will NOT call on_write_done in that case,
/// so the caller owns any promise that was waiting on this write; P1-9).
unsafe fn write_str_dup(stream: *mut UvStream, msg: &[u8]) -> c_int {
    let copy = libc::malloc(msg.len()) as *mut c_char;
    std::ptr::copy_nonoverlapping(msg.as_ptr(), copy as *mut u8, msg.len());
    let wr = Box::into_raw(Box::new(WriteReq { data: copy }));
    let req = libc::malloc(uv::sofuu_uv_write_size()) as *mut UvWriteReq;
    *(req as *mut *mut WriteReq) = wr;
    let buf = uv::UvBuf {
        base: copy,
        len: msg.len(),
    };
    let rc = uv::uv_write(req, stream, &buf, 1, Some(on_mcp_write_done));
    if rc < 0 {
        /* Synchronous failure (EPIPE/EBADF after child death): libuv will
         * NOT call on_write_done — free here or the req + payload leak. */
        libc::free(copy as *mut c_void);
        libc::free(req as *mut c_void);
        drop(Box::from_raw(wr));
    }
    rc
}

/// Frees a bare pipe storage once libuv is fully done with it. uv_close is
/// asynchronous — libuv's closing-handles pass still reads the handle (and
/// its close_cb slot) on later loop iterations, so the storage must not be
/// freed directly after uv_close.
unsafe extern "C" fn free_pipe_cb(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    libc::free(handle as *mut c_void);
}

/// Returns the uv_write rc: 0 = queued, <0 = synchronous failure (libuv will
/// NOT call on_write_done). A caller with a promise riding on the write MUST
/// settle it when rc < 0 — P1-9: a swallowed failure left pending call()
/// promises pending forever and the MAX_PENDING slot never reclaimed.
unsafe fn mcp_client_send(client: *mut McpClient, msg: &[u8]) -> c_int {
    write_str_dup((*client).stdin_pipe as *mut UvStream, msg)
}

// ── client response routing (pure Rust parse — M6 mandate) ──────────

unsafe fn mcp_process_line(client: *mut McpClient, line: &[u8]) {
    let ctx = (*client).ctx;
    let Ok(input) = std::str::from_utf8(line) else {
        return;
    };
    let parsed = crate::mcp::jsonrpc::parse(input);
    let Inbound::Response { id, result, error } = parsed else {
        return; /* requests/notifications/garbage are not for us */
    };
    let msg_id = id as i32;

    /* Check if this is a response to a pending call */
    for i in 0..(*client).pending_count {
        if (*client).pending[i].id != msg_id {
            continue;
        }

        let promise = (*client).pending[i].promise;
        let was_init_id = msg_id == (*client).initialize_id;

        if let Some(result_v) = result {
            /* Parse result into a JS value */
            let result_json = serde_json::to_string(&result_v).unwrap_or_else(|_| "null".into());
            let rc = cstr_json(&result_json);
            let mut jresult = qjs::JS_ParseJSON(ctx, rc.as_ptr(), rc.as_bytes().len(), c"<mcp>".as_ptr());
            if qjs::is_exception(jresult) {
                qjs::sofuu_js_get_exception(ctx);
                jresult = qjs::sofuu_js_new_string(ctx, rc.as_ptr());
            }
            sofuu_promise_resolve(promise, jresult);
            qjs::sofuu_js_free_value(ctx, jresult);
        } else if let Some(error_v) = error {
            let msg_text = error_v.message.as_str();
            let mc = cstr_text(msg_text);
            let jerr = qjs::sofuu_js_new_string(ctx, mc.as_ptr());
            sofuu_promise_reject(promise, jerr); /* reject consumes jerr */
        } else {
            sofuu_promise_resolve(promise, qjs::sofuu_js_null());
        }

        /* Remove from pending */
        (*client).pending[i] = (*client).pending[(*client).pending_count - 1];
        (*client).pending_count -= 1;

        sofuu_flush_jobs(ctx);

        /*
         * Resolve the connect promise only if this response matches the
         * exact request ID we sent for "initialize".
         */
        if (*client).initialized == 0 && !(*client).connect_promise.is_null() && was_init_id {
            (*client).initialized = 1;
            /* Send initialized notification */
            let notif = jsonrpc_notify_cstr("notifications/initialized", None);
            mcp_client_send(client, notif.as_bytes());
            drop(notif);

            /* Resolve the connect promise with the client object itself */
            let self_js = qjs::sofuu_js_dup_value(ctx, (*client).self_val);
            sofuu_promise_resolve((*client).connect_promise, self_js);
            qjs::sofuu_js_free_value(ctx, self_js);
            sofuu_flush_jobs(ctx);
            (*client).connect_promise = ptr::null_mut();
        }
        return;
    }
}

/* P3 (AUDIT-2026-09-07): `CString::new(..).unwrap_or_default()` handed back
 * an EMPTY string whenever the payload carried an interior NUL — a server
 * error message containing \u0000 became an empty rejection, and a JSON
 * frame would have been silently dropped. These builders escape instead:
 * JSON paths use the \u0000 escape (a raw NUL byte is invalid inside JSON
 * text anyway), text paths a readable \0. */
fn cstr_json_bytes(b: &[u8]) -> CString {
    if !b.contains(&0) {
        return CString::new(b.to_vec()).unwrap_or_default();
    }
    let mut out = Vec::with_capacity(b.len() + 16);
    for &c in b {
        if c == 0 {
            out.extend_from_slice(b"\\u0000");
        } else {
            out.push(c);
        }
    }
    CString::new(out).unwrap_or_default()
}

fn cstr_json(s: &str) -> CString {
    cstr_json_bytes(s.as_bytes())
}

/// Text-facing variant (error messages shown to JS): NUL becomes a
/// readable \0 escape instead of emptying the message.
fn cstr_text(s: &str) -> CString {
    if !s.contains('\0') {
        return CString::new(s).unwrap_or_default();
    }
    CString::new(s.replace('\0', "\\0")).unwrap_or_default()
}

/// Thin adapters over the Rust jsonrpc builders — same strings the C
/// adapters produced (newline-terminated).
fn jsonrpc_request_cstr(id: i32, method: &str, params_json: Option<&str>) -> CString {
    let s = crate::mcp::jsonrpc::request(id as u32, method, params_json);
    cstr_json(&s)
}

fn jsonrpc_notify_cstr(method: &str, params_json: Option<&str>) -> CString {
    let s = crate::mcp::jsonrpc::notify(method, params_json);
    cstr_json(&s)
}

fn jsonrpc_response_cstr(id: i32, result_json: &str) -> CString {
    let s = crate::mcp::jsonrpc::response(id as u32, Some(result_json));
    cstr_json(&s)
}

fn jsonrpc_error_cstr(id: i32, code: i32, message: &str) -> CString {
    let s = crate::mcp::jsonrpc::error(id as u32, code, message);
    cstr_json(&s)
}

/// Called by libuv when data arrives from child stdout.
unsafe extern "C" fn mcp_on_read(stream: *mut UvStream, nread: isize, buf: *const uv::UvBuf) {
    // SAFETY: stdout pipe's data set at connect.
    let client = *(stream as *mut *mut McpClient);

    if nread > 0 {
        let chunk = std::slice::from_raw_parts((*buf).base as *const u8, nread as usize);
        /* Append to line buffer */
        let c = &mut *client;
        let space = MCP_READ_BUF - c.read_len - 1;
        if space == 0 {
            // Over-long line (>64KB, no newline): drop the buffer and fail
            // the connection instead of wedging forever with copy=0.
            // A hostile/broken MCP server must not pin pending promises.
            c.read_len = 0;
            c.read_buf[0] = 0;
            uv::uv_read_stop(stream);
            if !c.connect_promise.is_null() {
                let err = qjs::sofuu_js_new_string(
                    c.ctx,
                    c"MCP server sent an over-long line (>64KB)".as_ptr(),
                );
                sofuu_promise_reject(c.connect_promise, err);
                c.connect_promise = ptr::null_mut();
            }
            if !buf.is_null() && !(*buf).base.is_null() {
                libc::free((*buf).base as *mut c_void);
            }
            return;
        }
        let copy = chunk.len().min(space);
        c.read_buf[c.read_len..c.read_len + copy].copy_from_slice(&chunk[..copy]);
        c.read_len += copy;
        c.read_buf[c.read_len] = 0;

        /* Process complete newline-delimited JSON lines */
        let mut head = 0usize;
        while let Some(nl) = c.read_buf[head..c.read_len]
            .iter()
            .position(|&b| b == b'\n')
        {
            let nl = head + nl;
            let mut end = nl;
            if end > head && c.read_buf[end - 1] == b'\r' {
                end -= 1;
            }
            if end > head {
                mcp_process_line(client, &c.read_buf[head..end]);
            }
            head = nl + 1;
        }

        /* Slide remaining partial line to front */
        let remaining = c.read_len - head;
        c.read_buf.copy_within(head..c.read_len, 0);
        c.read_len = remaining;
        c.read_buf[remaining] = 0;

        /* NOTE: no sofuu_flush_jobs here at all — we are inside a libuv
         * read callback; running JS re-enters the loop mid-dispatch and
         * races the write side. The loop pumps jobs after uv_run. */
    } else if nread < 0 && nread != uv::UV_EOF as isize {
        /* Connection broken */
        if !(*client).connect_promise.is_null() {
            let ctx = (*client).ctx;
            let err = qjs::sofuu_js_new_string(ctx, c"MCP server disconnected".as_ptr());
            sofuu_promise_reject((*client).connect_promise, err); /* consumes err */
            sofuu_flush_jobs(ctx);
            (*client).connect_promise = ptr::null_mut();
        }
    }

    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn mcp_on_alloc(_h: *mut UvHandle, suggested: usize, buf: *mut uv::UvBuf) {
    (*buf).base = libc::malloc(suggested) as *mut c_char;
    (*buf).len = suggested;
}

unsafe extern "C" fn mcp_on_exit(_proc: *mut UvProcess, _exit_status: i64, _term_signal: c_int) {
    // SAFETY: process handle's data set after spawn.
    let client = *(_proc as *mut *mut McpClient);
    if client.is_null() || (*client).done != 0 {
        return;
    }
    (*client).done = 1;
    let ctx = (*client).ctx;

    /* Reject all pending call promises so callers don't hang forever */
    for i in 0..(*client).pending_count {
        if (*client).pending[i].promise.is_null() {
            continue;
        }
        let err = qjs::sofuu_js_new_string(ctx, c"MCP server process exited".as_ptr());
        sofuu_promise_reject((*client).pending[i].promise, err); /* consumes err */
    }
    (*client).pending_count = 0;

    /* Reject the connect promise if still pending */
    if !(*client).connect_promise.is_null() {
        let err = qjs::sofuu_js_new_string(ctx, c"MCP server exited before initialize".as_ptr());
        sofuu_promise_reject((*client).connect_promise, err); /* consumes err */
        (*client).connect_promise = ptr::null_mut();
    }

    sofuu_flush_jobs(ctx);

    /* Stop reads; close stdin/stdout/stderr/process — the last close callback
     * frees the client. stderr was uv_pipe_init'd at connect even though it
     * is UV_IGNORE'd for the child, so it must be closed too (else its
     * storage leaks and the handle stays in the loop). */
    uv::uv_read_stop((*client).stdout_pipe as *mut UvStream);
    (*client).closing = 4;
    uv::uv_close((*client).stdin_pipe as *mut UvHandle, Some(mcp_client_close_cb));
    uv::uv_close((*client).stdout_pipe as *mut UvHandle, Some(mcp_client_close_cb));
    uv::uv_close((*client).stderr_pipe as *mut UvHandle, Some(mcp_client_close_cb));
    uv::uv_close(_proc as *mut UvHandle, Some(mcp_client_close_cb));
}

// ── JS: client.call / listTools / listResources / disconnect ────────

unsafe extern "C" fn js_mcp_call(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let client = qjs::JS_GetOpaque(this_val, MCP_CLIENT_CLASS_ID.load(Ordering::Relaxed)) as *mut McpClient;
    if client.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid MCPClient".as_ptr());
    }
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"call: method required".as_ptr());
    }

    let method_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if method_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let method_js = CStr::from_ptr(method_ptr).to_bytes().to_vec();
    qjs::sofuu_js_free_cstring(ctx, method_ptr);
    let mut method = method_js; /* may be replaced by the static tools/call */

    /* Serialize params to JSON */
    let mut params_json: Option<CString> = None;
    if argc > 1 && !qjs::is_undefined(*argv.add(1)) {
        let json_str = qjs::JS_JSONStringify(ctx, *argv.add(1), qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
        if !qjs::is_exception(json_str) {
            let s = qjs::sofuu_js_to_cstring(ctx, json_str);
            if !s.is_null() {
                let bytes = CStr::from_ptr(s).to_bytes();
                params_json = Some(cstr_json_bytes(bytes));
                qjs::sofuu_js_free_cstring(ctx, s);
            }
            qjs::sofuu_js_free_value(ctx, json_str);
        }
    }

    /* Bare tool name → wrap into tools/call. */
    let method_is_bare = !method.contains(&b'/');
    let wrapped_params: Option<CString>;
    if method_is_bare {
        let name_str = String::from_utf8_lossy(&method);
        /* proc-9: the old hand-rolled escaper covered only `"` and `\` —
         * a newline or tab inside a tool name produced invalid JSON that
         * the server would drop or misparse. json_escape handles every
         * control character plus U+2028/2029. */
        let esc = crate::rt::ai::json_escape(Some(&name_str));
        let pj = params_json.as_ref().map(|c| c.to_string_lossy().into_owned());
        let wrap = format!(
            "{{\"name\":\"{}\",\"arguments\":{}}}",
            esc,
            pj.as_deref().unwrap_or("{}")
        );
        wrapped_params = Some(cstr_json(&wrap));
        params_json = wrapped_params;
        method = b"tools/call".to_vec();
    }

    let id = next_id();
    let msg = jsonrpc_request_cstr(
        id,
        std::str::from_utf8(&method).unwrap_or("tools/call"),
        params_json.as_ref().map(|c| c.to_string_lossy().into_owned()).as_deref(),
    );

    /* Register pending promise */
    if (*client).pending_count >= MAX_PENDING {
        return qjs::JS_ThrowRangeError(ctx, c"Too many pending MCP calls".as_ptr());
    }

    let mut promise_out: *mut PromiseHandle = ptr::null_mut();
    let ret = sofuu_promise_new(ctx, &mut promise_out);

    (*client).pending[(*client).pending_count].id = id;
    (*client).pending[(*client).pending_count].promise = promise_out;
    (*client).pending[(*client).pending_count].ctx = ctx;
    (*client).pending_count += 1;

    let msg_bytes = msg.as_bytes();
    let msg_owned = msg_bytes.to_vec();
    let rc = mcp_client_send(client, &msg_owned);
    if rc < 0 {
        /* P1-9: the write failed synchronously (EPIPE/EBADF — dead child).
         * No response can ever arrive, so settle the promise NOW and
         * reclaim the pending slot, or `call()` promises pile up until
         * MAX_PENDING wedges the client. */
        let p = (*client).pending[(*client).pending_count - 1].promise;
        (*client).pending_count -= 1;
        let err = qjs::sofuu_js_new_string(ctx, c"MCP write failed (server unreachable)".as_ptr());
        sofuu_promise_reject(p, err); /* consumes err */
        sofuu_flush_jobs(ctx);
    }

    ret
}

unsafe extern "C" fn js_mcp_list_tools(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let method = qjs::sofuu_js_new_string(ctx, c"tools/list".as_ptr());
    let result = js_mcp_call(ctx, this_val, 1, &method);
    qjs::sofuu_js_free_value(ctx, method);
    result
}

unsafe extern "C" fn js_mcp_list_resources(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let method = qjs::sofuu_js_new_string(ctx, c"resources/list".as_ptr());
    let result = js_mcp_call(ctx, this_val, 1, &method);
    qjs::sofuu_js_free_value(ctx, method);
    result
}

unsafe extern "C" fn js_mcp_disconnect(
    _ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let client = qjs::JS_GetOpaque(this_val, MCP_CLIENT_CLASS_ID.load(Ordering::Relaxed)) as *mut McpClient;
    if !client.is_null() && !(*client).process.is_null() {
        uv::uv_process_kill((*client).process, libc::SIGTERM);
    }
    qjs::sofuu_js_undefined()
}

thread_local! {
    // JS_CFUNC_DEF(name, length, func): magic=0, u.func = { length, generic, func }.
    static MCP_CLIENT_PROTO: [qjs::JSCFunctionListEntry; 4] = [
        qjs::JSCFunctionListEntry {
            name: c"call".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_mcp_call },
        },
        qjs::JSCFunctionListEntry {
            name: c"listTools".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_mcp_list_tools },
        },
        qjs::JSCFunctionListEntry {
            name: c"listResources".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_mcp_list_resources },
        },
        qjs::JSCFunctionListEntry {
            name: c"disconnect".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_mcp_disconnect },
        },
    ];
}

// ── shell-lite tokenizer (quotes + backslash escapes, max 63 args) ──

fn tokenize_command(cmd: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = cmd.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && out.len() < 63 {
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let mut tok: Vec<u8> = Vec::new();
        while i < bytes.len() && !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            let c = bytes[i];
            match c {
                b'\'' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'\'' {
                        tok.push(bytes[i]);
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 1;
                        }
                        tok.push(bytes[i]);
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                b'\\' if i + 1 < bytes.len() => {
                    i += 1;
                    tok.push(bytes[i]);
                    i += 1;
                }
                _ => {
                    tok.push(c);
                    i += 1;
                }
            }
        }
        out.push(String::from_utf8_lossy(&tok).into_owned());
    }
    out
}

// ── JS: sofuu.mcp.connect(command) → Promise<MCPClient> ─────────────

unsafe extern "C" fn js_mcp_connect(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"mcp.connect: command required".as_ptr());
    }

    let cmd_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if cmd_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let cmd = CStr::from_ptr(cmd_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, cmd_ptr);

    let args_arr = tokenize_command(&cmd);
    if args_arr.is_empty() {
        return qjs::JS_ThrowTypeError(ctx, c"mcp.connect: empty command".as_ptr());
    }
    let args_count = args_arr.len();

    /* Allocate client */
    let mut client = Box::new(McpClient {
        process: ptr::null_mut(),
        stdin_pipe: ptr::null_mut(),
        stdout_pipe: ptr::null_mut(),
        stderr_pipe: ptr::null_mut(),
        ctx,
        self_val: qjs::sofuu_js_undefined(),
        read_buf: [0u8; MCP_READ_BUF],
        read_len: 0,
        pending: [PendingCall {
            id: 0,
            promise: ptr::null_mut(),
            ctx: ptr::null_mut(),
        }; MAX_PENDING],
        pending_count: 0,
        initialized: 0,
        initialize_id: 0,
        connect_promise: ptr::null_mut(),
        done: 0,
        closing: 0,
    });

    /* Create JS object for the client (the opaque is the Box'd client — the
     * failure path detaches it before dropping the last ref). */
    client.self_val = qjs::JS_NewObjectClass(ctx, MCP_CLIENT_CLASS_ID.load(Ordering::Relaxed) as c_int);
    qjs::JS_SetOpaque(client.self_val, &mut *client as *mut McpClient as *mut c_void);

    /* Setup pipes */
    let loop_ = sofuu_loop_get();
    let stdin_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stdout_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stderr_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let process = libc::malloc(uv::sofuu_uv_process_size()) as *mut UvProcess;
    client.stdin_pipe = stdin_pipe;
    client.stdout_pipe = stdout_pipe;
    client.stderr_pipe = stderr_pipe;
    client.process = process;

    uv::uv_pipe_init(loop_, stdin_pipe, 0);
    uv::uv_pipe_init(loop_, stdout_pipe, 0);
    uv::uv_pipe_init(loop_, stderr_pipe, 0);

    let mut stdio: [uv::UvStdioContainer; 3] = [
        uv::UvStdioContainer { flags: 0, data: uv::UvStdioData { stream: ptr::null_mut() } },
        uv::UvStdioContainer { flags: 0, data: uv::UvStdioData { stream: ptr::null_mut() } },
        uv::UvStdioContainer { flags: 0, data: uv::UvStdioData { stream: ptr::null_mut() } },
    ];
    stdio[0].flags = uv::UV_CREATE_PIPE | uv::UV_READABLE_PIPE;
    stdio[0].data.stream = stdin_pipe as *mut c_void;
    stdio[1].flags = uv::UV_CREATE_PIPE | uv::UV_WRITABLE_PIPE;
    stdio[1].data.stream = stdout_pipe as *mut c_void;
    stdio[2].flags = uv::UV_IGNORE;

    // P3: args derive from `cmd`, which arrived through a C string — no
    // interior NUL can reach this CString::new, so it cannot silently
    // empty an argument.
    let argv_c: Vec<CString> = args_arr
        .iter()
        .map(|a| CString::new(a.as_str()).unwrap_or_default())
        .collect();
    let mut argv_ptr: Vec<*mut c_char> = argv_c.iter().map(|c| c.as_ptr() as *mut c_char).collect();
    argv_ptr.push(ptr::null_mut());

    /* P1-10 (AUDIT-2026-09-07): MCP servers are third-party code — the
     * child env must be the functional allowlist, NEVER the full parent
     * environment (provider keys live in env; one `console.log(process.env)`
     * from a hostile server would exfiltrate them). */
    let env_c = crate::rt::process_spawn::build_child_env(ctx, qjs::sofuu_js_undefined())
        .unwrap_or_default();
    let mut env_ptr: Vec<*mut c_char> = env_c.iter().map(|e| e.as_ptr() as *mut c_char).collect();
    env_ptr.push(ptr::null_mut());

    let opts = uv::UvProcessOptions {
        exit_cb: Some(mcp_on_exit),
        file: argv_c[0].as_ptr(),
        args: argv_ptr.as_mut_ptr(),
        env: env_ptr.as_mut_ptr(),
        cwd: ptr::null(),
        flags: 0,
        stdio_count: 3,
        stdio: stdio.as_mut_ptr(),
        #[cfg(unix)]
        uid: 0,
        #[cfg(unix)]
        gid: 0,
    };

    let client_ptr = Box::into_raw(client);
    let r = uv::uv_spawn(loop_, process, &opts);
    let _ = args_count;

    if r < 0 {
        /* uv_spawn never fired exit_cb and no pipe reads started, so the
         * only owner of the client is the JS object. Detach the opaque
         * BEFORE dropping the last ref: otherwise the GC finalizer runs
         * free(client) and this explicit free() double-frees. */
        qjs::JS_SetOpaque((*client_ptr).self_val, ptr::null_mut());
        qjs::sofuu_js_free_value(ctx, (*client_ptr).self_val);
        /* The pipe storages are freed from the close callbacks — freeing
         * them here would race libuv's closing-handles pass (UAF). The
         * process handle also goes through uv_close: uv_spawn queued it
         * in loop->handle_queue even though it failed (the vendored
         * libuv's error-dequeue is #if 0'd), so a direct free would leave
         * a dangling entry for uv_walk. */
        uv::uv_close(stdin_pipe as *mut UvHandle, Some(free_pipe_cb));
        uv::uv_close(stdout_pipe as *mut UvHandle, Some(free_pipe_cb));
        uv::uv_close(stderr_pipe as *mut UvHandle, Some(free_pipe_cb));
        uv::uv_close(process as *mut UvHandle, Some(free_pipe_cb));
        drop(Box::from_raw(client_ptr));
        return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), uv::uv_strerror(r));
    }

    /* CRITICAL: proc->data must point back at the client so mcp_on_exit
     * fires (bug fixed in C; kept here). */
    *(process as *mut *mut McpClient) = client_ptr;

    /* P0-2 (AUDIT-2026-09-07): ALL FOUR handles must carry ->data. The
     * exit path closes stdin/stdout/stderr/process with
     * mcp_client_close_cb, which reads handle->data; stdin/stderr were
     * left as stale malloc garbage (uv__handle_init never zeroes it and
     * the pipes are plain malloc, not calloc), so every disconnect
     * dereferenced a wild pointer. The spawn-fail path above is unaffected
     * — its free_pipe_cb never reads ->data. */
    *(stdin_pipe as *mut *mut McpClient) = client_ptr;
    *(stdout_pipe as *mut *mut McpClient) = client_ptr;
    *(stderr_pipe as *mut *mut McpClient) = client_ptr;
    let _rs_rc = uv::uv_read_start(stdout_pipe as *mut UvStream, Some(mcp_on_alloc), Some(mcp_on_read));

    /* F-2: all four handles are armed per-ctx — a multi-engine teardown must
     * close them (with free_pipe_cb, which only frees storage: the client
     * box is fence-freed via the closing counter, parity with the
     * last-engine walk that closes with None). */
    unsafe {
        crate::rt::event_loop::track_handle(ctx, process as *mut UvHandle, Some(free_pipe_cb));
        crate::rt::event_loop::track_handle(ctx, stdin_pipe as *mut UvHandle, Some(free_pipe_cb));
        crate::rt::event_loop::track_handle(ctx, stdout_pipe as *mut UvHandle, Some(free_pipe_cb));
        crate::rt::event_loop::track_handle(ctx, stderr_pipe as *mut UvHandle, Some(free_pipe_cb));
    }

    /* Create the connect promise */
    let mut cp_out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut cp_out);
    (*client_ptr).connect_promise = cp_out;

    /* Send MCP initialize request */
    let init_id = next_id();
    let init_msg = jsonrpc_request_cstr(
        init_id,
        "initialize",
        Some("{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"sofuu\",\"version\":\"0.1.0\"}}"),
    );

    /* Register initialize as a pending call and remember its id */
    (*client_ptr).initialize_id = init_id;
    (*client_ptr).pending[0].id = init_id;
    (*client_ptr).pending[0].promise = ptr::null_mut(); /* handled via connect_promise */
    (*client_ptr).pending[0].ctx = ctx;
    (*client_ptr).pending_count = 1;

    let rc = mcp_client_send(client_ptr, init_msg.as_bytes());
    if rc < 0 {
        /* P1-9: the initialize write failed synchronously (EPIPE/EBADF).
         * No initialize response can ever arrive, so settle the connect
         * promise NOW — otherwise a dead-at-birth server leaves callers
         * awaiting `connect()` forever. pending[0] is the initialize
         * registration (null promise — it rides on connect_promise);
         * mcp_on_exit later sees count 0 / null handle and no-ops. */
        if !(*client_ptr).connect_promise.is_null() {
            let err = qjs::sofuu_js_new_string(
                ctx,
                c"MCP write failed (server unreachable)".as_ptr(),
            );
            sofuu_promise_reject((*client_ptr).connect_promise, err); /* consumes err */
            (*client_ptr).connect_promise = ptr::null_mut();
        }
        (*client_ptr).pending_count = 0;
        sofuu_flush_jobs(ctx);
    }

    promise
}

// ── MCP Server (stdio transport) ────────────────────────────────────

static MCP_SERVER_CLASS_ID: AtomicU32 = AtomicU32::new(0);

struct McpTool {
    name: Vec<u8>,        /* up to 127 bytes (C: char[128]) */
    description: CString, /* heap — unbounded */
    schema_json: CString, /* JSON string for input schema */
    handler: JSValue,     /* JS async function */
}

struct McpServer {
    ctx: *mut JSContext,
    self_val: JSValue,
    tools: Vec<McpTool>,
    tool_count: usize,

    stdin_pipe: *mut UvPipe,
    stdout_pipe: *mut UvPipe,

    read_buf: [u8; MCP_READ_BUF],
    read_len: usize,
}

unsafe extern "C" fn mcp_server_finalizer(rt: *mut qjs::JSRuntime, val: JSValue) {
    let srv = qjs::JS_GetOpaque(val, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed)) as *mut McpServer;
    if srv.is_null() {
        return;
    }
    #[cfg(test)]
    TEST_MCP_SERVER_FREES.fetch_add(1, Ordering::Relaxed);
    for tool in (*srv).tools.iter() {
        /* handlers were dup'd with JS_DupValue — free them using the runtime */
        qjs::sofuu_js_free_value_rt(rt, tool.handler);
    }
    drop(Box::from_raw(srv));
}

unsafe fn mcp_server_write_str(srv: *mut McpServer, msg: &[u8]) {
    write_str_dup((*srv).stdout_pipe as *mut UvStream, msg);
}

/// Recover the server from a settle-callback's data[1] slot. The old code
/// stored the raw `*mut McpServer` as an int64 — but the finalizer frees the
/// Box unconditionally, so a GC'd server between `.then` attach and settle
/// made every later read of data[1] a use-after-free (P1-7). data[1] is now
/// the server's own JSValue (GC-marked in C-function data, quickjs.c:1485),
/// keeping the object — and its Box — alive while the reaction lives; a
/// non-object slot recovers to null and the response is reported lost.
///
/// # Safety
/// `data` must point at the C-function's 2-entry data array (or be null).
unsafe fn mcp_tool_data_server(data: *mut JSValue) -> *mut McpServer {
    if data.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: data points at the C-function data array.
    let v = *data.add(1);
    if !qjs::is_object(v) {
        return ptr::null_mut();
    }
    let srv = qjs::JS_GetOpaque(v, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed)) as *mut McpServer;
    if srv.is_null() {
        return ptr::null_mut();
    }
    srv
}

/// Shared tail for both settle callbacks: write through the server's stdout
/// pipe while the server object is reachable, else say where it went.
unsafe fn mcp_tool_write_response(srv: *mut McpServer, resp: &[u8]) {
    if !srv.is_null() {
        mcp_server_write_str(srv, resp);
    } else {
        eprintln!("[sofuu mcp] async tool response lost (no server)");
    }
}

/// Callback for resolving async tool handlers. data[0] = req_id (int32),
/// data[1] = the server's JSValue — captured at JS_NewCFunctionData time so
/// the response is written through the server's own stdout pipe.
unsafe extern "C" fn mcp_tool_then_cb(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let mut req_id: i32 = -1;
    qjs::JS_ToInt32(ctx, &mut req_id, *data);
    let srv = mcp_tool_data_server(data);

    let json_val = qjs::JS_JSONStringify(ctx, *argv, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
    let rs = if qjs::is_exception(json_val) {
        None
    } else {
        let s = qjs::sofuu_js_to_cstring(ctx, json_val);
        if s.is_null() {
            None
        } else {
            let out = Some(CStr::from_ptr(s).to_string_lossy().into_owned());
            qjs::sofuu_js_free_cstring(ctx, s);
            out
        }
    };

    /* Heap-growable response buffer — large async results must never be
     * truncated on a 4KB stack array. */
    let rs_safe = rs.as_deref().unwrap_or("\"\"");
    let rb = format!("{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}", rs_safe);

    let resp = jsonrpc_response_cstr(req_id, &rb);
    mcp_tool_write_response(srv, resp.as_bytes());

    if !qjs::is_exception(json_val) {
        qjs::sofuu_js_free_value(ctx, json_val);
    }
    qjs::sofuu_js_undefined()
}

/// P1-8: rejection twin of mcp_tool_then_cb. Without it a rejected handler
/// promise answered nothing — the client's pending slot (MAX_PENDING=64)
/// only frees when a response line arrives (mcp_process_line), so repeated
/// handler rejections permanently exhausted tool calls.
unsafe extern "C" fn mcp_tool_reject_cb(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let mut req_id: i32 = -1;
    qjs::JS_ToInt32(ctx, &mut req_id, *data);
    let srv = mcp_tool_data_server(data);
    let resp = jsonrpc_error_cstr(req_id, -32603, "Tool handler rejected");
    mcp_tool_write_response(srv, resp.as_bytes());
    qjs::sofuu_js_undefined()
}

/// Route an incoming RPC request to a registered tool.
unsafe fn mcp_server_handle_request(srv: *mut McpServer, line: &[u8]) {
    let ctx = (*srv).ctx;
    let Ok(input) = std::str::from_utf8(line) else {
        return;
    };

    /* Parse inbound with the Rust jsonrpc::parse (safe serde). */
    let parsed = crate::mcp::jsonrpc::parse(input);
    let (req_id, method): (i32, Option<String>) = match parsed {
        Inbound::Request { id, method, .. } => (id as i32, Some(method)),
        _ => (0, None), /* notifications (and garbage) get no reply */
    };

    /* JSON-RPC notifications carry no id and MUST NOT get a response. */
    let Some(method) = method else {
        return;
    };

    /* --- initialize --- */
    if method == "initialize" {
        let resp = jsonrpc_response_cstr(
            req_id,
            "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"sofuu-mcp\",\"version\":\"0.1.0\"}}",
        );
        mcp_server_write_str(srv, resp.as_bytes());

    /* --- tools/list --- */
    } else if method == "tools/list" {
        let mut tools_json = String::from("[");
        for (i, tool) in (*srv).tools.iter().enumerate() {
            if i > 0 {
                tools_json.push(',');
            }
            let n = String::from_utf8_lossy(&tool.name);
            let d = tool.description.to_string_lossy();
            let s = tool.schema_json.to_string_lossy();
            tools_json.push_str(&format!(
                "{{\"name\":\"{}\",\"description\":\"{}\",\"inputSchema\":{}}}",
                json_escape(&n),
                json_escape(&d),
                s
            ));
        }
        tools_json.push(']');
        let result = format!("{{\"tools\":{}}}", tools_json);
        let resp = jsonrpc_response_cstr(req_id, &result);
        mcp_server_write_str(srv, resp.as_bytes());

    /* --- tools/call --- */
    } else if method == "tools/call" {
        /* Extract params from the parsed request. */
        let mut toolname: Option<String> = None;
        let mut args_str: Option<CString> = None;
        let params = crate::mcp::jsonrpc::parse(input);
        if let Inbound::Request { params: Some(p), .. } = params {
            if let Some(name) = p.get("name").and_then(|n| n.as_str()) {
                toolname = Some(name.to_string());
            }
            if let Some(args) = p.get("arguments") {
                let s = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                args_str = Some(cstr_json(&s));
            }
        }

        /* Find the tool */
        let mut tool: Option<usize> = None;
        if let Some(name) = &toolname {
            for (i, t) in (*srv).tools.iter().enumerate() {
                if String::from_utf8_lossy(&t.name) == name.as_str() {
                    tool = Some(i);
                    break;
                }
            }
        }

        match tool {
            None => {
                let err = jsonrpc_error_cstr(req_id, -32601, "Tool not found");
                mcp_server_write_str(srv, err.as_bytes());
            }
            Some(ti) => {
                /* Parse args into JS object */
                let js_args = if let Some(ac) = &args_str {
                    let parsed = qjs::JS_ParseJSON(ctx, ac.as_ptr(), ac.as_bytes().len(), c"<mcp-args>".as_ptr());
                    if qjs::is_exception(parsed) {
                        qjs::sofuu_js_new_object(ctx)
                    } else {
                        parsed
                    }
                } else {
                    qjs::sofuu_js_new_object(ctx)
                };

                /* Call the JS handler and handle both sync and async results */
                let srv_ref = &*srv;
                let handler = srv_ref.tools[ti].handler;
                let ret = qjs::JS_Call(ctx, handler, qjs::sofuu_js_undefined(), 1, &js_args);
                qjs::sofuu_js_free_value(ctx, js_args);

                if qjs::is_exception(ret) {
                    qjs::js_std_dump_error(ctx);
                    let err = jsonrpc_error_cstr(req_id, -32603, "Tool handler threw");
                    mcp_server_write_str(srv, err.as_bytes());
                } else {
                    /* Check if the return value is a Promise by testing for
                     * a callable .then property. */
                    let then_fn = qjs::sofuu_js_get_property_str(ctx, ret, c"then".as_ptr());
                    let is_promise = qjs::JS_IsFunction(ctx, then_fn) != 0;
                    qjs::sofuu_js_free_value(ctx, then_fn);

                    if is_promise {
                        /* P1-7: capture req_id + the server's JSValue (NOT
                         * a raw pointer — the old int64 encoding let the
                         * finalizer free the Box while a tool promise was
                         * pending = UAF). QuickJS GC-marks C-function data
                         * (quickjs.c:1485), so this dup pins the object
                         * until the promise settles; the callbacks recover
                         * it via mcp_tool_data_server. */
                        let js_id = qjs::sofuu_js_new_int32(ctx, req_id);
                        let js_srv = qjs::sofuu_js_dup_value(ctx, (*srv).self_val);
                        let mut data_arr: [JSValue; 2] = [js_id, js_srv];

                        /* .then(on_ok, on_err) — P1-8: the rejection handler
                         * answers the request with a JSON-RPC error so the
                         * caller's pending slot frees even when the tool
                         * handler rejects. */
                        let then_ok =
                            qjs::JS_NewCFunctionData(ctx, mcp_tool_then_cb, 1, 0, 2, data_arr.as_mut_ptr());
                        let then_err =
                            qjs::JS_NewCFunctionData(ctx, mcp_tool_reject_cb, 1, 0, 2, data_arr.as_mut_ptr());

                        /* Handle resolution + rejection */
                        let handlers: [JSValueConst; 2] = [then_ok, then_err];
                        let ret2 = qjs::JS_Call(ctx, then_fn, ret, 2, handlers.as_ptr());
                        if qjs::is_exception(ret2) {
                            qjs::js_std_dump_error(ctx);
                            let err = jsonrpc_error_cstr(req_id, -32603, "Internal error in promise chain");
                            mcp_server_write_str(srv, err.as_bytes());
                        }

                        qjs::sofuu_js_free_value(ctx, then_err);
                        qjs::sofuu_js_free_value(ctx, then_ok);
                        qjs::sofuu_js_free_value(ctx, ret2);
                        qjs::sofuu_js_free_value(ctx, data_arr[1]);
                        qjs::sofuu_js_free_value(ctx, data_arr[0]);

                        /* Pump microtasks once to settle synchronous promises */
                        sofuu_flush_jobs(ctx);

                        /* If still unresolved, the async response arrives
                         * when the promise settles (mcp_tool_then_cb). */
                    } else {
                        /* Sync path (or settled promise after flush) */
                        let json_val = qjs::JS_JSONStringify(ctx, ret, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
                        let result_str = if qjs::is_exception(json_val) {
                            None
                        } else {
                            let s = qjs::sofuu_js_to_cstring(ctx, json_val);
                            if s.is_null() {
                                None
                            } else {
                                let out = Some(CStr::from_ptr(s).to_string_lossy().into_owned());
                                qjs::sofuu_js_free_cstring(ctx, s);
                                out
                            }
                        };

                        /* Heap-growable — large sync results must not truncate. */
                        let rs_safe = result_str.as_deref().unwrap_or("\"\"");
                        let result_buf = format!("{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}", rs_safe);
                        let resp = jsonrpc_response_cstr(req_id, &result_buf);
                        mcp_server_write_str(srv, resp.as_bytes());

                        if !qjs::is_exception(json_val) {
                            qjs::sofuu_js_free_value(ctx, json_val);
                        }
                    }

                    qjs::sofuu_js_free_value(ctx, ret);
                }
            }
        }

    /* --- unknown method --- */
    } else {
        let err = jsonrpc_error_cstr(req_id, -32601, "Method not found");
        mcp_server_write_str(srv, err.as_bytes());
    }

    /* NOTE: no sofuu_flush_jobs here — inside a libuv read callback; the
     * loop pumps jobs after uv_run. */
}

/// Escape `"` and `\` inside a JSON string body (the tools/list appender —
/// C only escaped quotes for names; keep the same shape).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

unsafe extern "C" fn mcp_server_on_read(stream: *mut UvStream, nread: isize, buf: *const uv::UvBuf) {
    // SAFETY: stdin pipe's data set at start().
    let srv = *(stream as *mut *mut McpServer);

    if nread > 0 {
        let chunk = std::slice::from_raw_parts((*buf).base as *const u8, nread as usize);
        let s = &mut *srv;
        let space = MCP_READ_BUF - s.read_len - 1;
        let copy = chunk.len().min(space);
        s.read_buf[s.read_len..s.read_len + copy].copy_from_slice(&chunk[..copy]);
        s.read_len += copy;
        s.read_buf[s.read_len] = 0;

        let mut head = 0usize;
        while let Some(nl) = s.read_buf[head..s.read_len]
            .iter()
            .position(|&b| b == b'\n')
        {
            let nl = head + nl;
            let mut end = nl;
            if end > head && s.read_buf[end - 1] == b'\r' {
                end -= 1;
            }
            if end > head {
                mcp_server_handle_request(srv, &s.read_buf[head..end]);
            }
            head = nl + 1;
        }

        let remaining = s.read_len - head;
        s.read_buf.copy_within(head..s.read_len, 0);
        s.read_len = remaining;
        s.read_buf[remaining] = 0;
    }

    if !(*buf).base.is_null() {
        libc::free((*buf).base as *mut c_void);
    }
}

unsafe extern "C" fn mcp_server_on_alloc(_h: *mut UvHandle, sz: usize, buf: *mut uv::UvBuf) {
    (*buf).base = libc::malloc(sz) as *mut c_char;
    (*buf).len = sz;
}

// ── JS: server.tool(name, { description, schema }, handler) ─────────

unsafe extern "C" fn js_server_tool(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let srv = qjs::JS_GetOpaque(this_val, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed)) as *mut McpServer;
    if srv.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid MCPServer".as_ptr());
    }
    if argc < 3 {
        return qjs::JS_ThrowTypeError(ctx, c"tool: name, opts, handler required".as_ptr());
    }
    if (*srv).tool_count >= MAX_TOOLS {
        return qjs::JS_ThrowRangeError(ctx, c"Too many tools".as_ptr());
    }

    let name_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if name_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let mut name = CStr::from_ptr(name_ptr).to_bytes().to_vec();
    name.truncate(127);
    qjs::sofuu_js_free_cstring(ctx, name_ptr);

    let mut description: CString = CString::new("").unwrap_or_default();
    let mut schema_json: CString = CString::new("{}").unwrap_or_default();

    /* opts: { description, schema } */
    if qjs::is_object(*argv.add(1)) {
        let desc = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"description".as_ptr());
        if !qjs::is_undefined(desc) {
            let d = qjs::sofuu_js_to_cstring(ctx, desc);
            if !d.is_null() {
                description = CString::new(CStr::from_ptr(d).to_bytes()).unwrap_or_default();
                qjs::sofuu_js_free_cstring(ctx, d);
            }
        }
        qjs::sofuu_js_free_value(ctx, desc);

        let schema = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"schema".as_ptr());
        if !qjs::is_undefined(schema) {
            let json = qjs::JS_JSONStringify(ctx, schema, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
            if !qjs::is_exception(json) {
                let s = qjs::sofuu_js_to_cstring(ctx, json);
                if !s.is_null() {
                    schema_json = CString::new(CStr::from_ptr(s).to_bytes()).unwrap_or_default();
                    qjs::sofuu_js_free_cstring(ctx, s);
                }
                qjs::sofuu_js_free_value(ctx, json);
            }
        }
        qjs::sofuu_js_free_value(ctx, schema);
    }

    if qjs::JS_IsFunction(ctx, *argv.add(2)) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"tool: handler must be a function".as_ptr());
    }
    let handler = qjs::sofuu_js_dup_value(ctx, *argv.add(2));
    (*srv).tools.push(McpTool {
        name,
        description,
        schema_json,
        handler,
    });
    (*srv).tool_count += 1;

    qjs::sofuu_js_undefined()
}

// ── JS: server.start() — begins listening on stdio ──────────────────

unsafe extern "C" fn js_server_start(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let srv = qjs::JS_GetOpaque(this_val, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed)) as *mut McpServer;
    if srv.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid MCPServer".as_ptr());
    }

    let loop_ = sofuu_loop_get();
    let stdin_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    let stdout_pipe = libc::malloc(uv::sofuu_uv_pipe_size()) as *mut UvPipe;
    (*srv).stdin_pipe = stdin_pipe;
    (*srv).stdout_pipe = stdout_pipe;
    uv::uv_pipe_init(loop_, stdin_pipe, 0);
    uv::uv_pipe_init(loop_, stdout_pipe, 0);

    uv::uv_pipe_open(stdin_pipe, 0); /* fd 0 = stdin  */
    uv::uv_pipe_open(stdout_pipe, 1); /* fd 1 = stdout */

    *(stdin_pipe as *mut *mut McpServer) = srv;
    uv::uv_read_start(stdin_pipe as *mut UvStream, Some(mcp_server_on_alloc), Some(mcp_server_on_read));

    /* F-2: both stdio pipes are armed per-ctx (data = srv, freed by
     * free_pipe_cb) — a multi-engine teardown must close them. */
    unsafe {
        crate::rt::event_loop::track_handle((*srv).ctx, stdin_pipe as *mut UvHandle, Some(free_pipe_cb));
        crate::rt::event_loop::track_handle((*srv).ctx, stdout_pipe as *mut UvHandle, Some(free_pipe_cb));
    }

    eprintln!(
        "[sofuu mcp] server started ({} tools registered)",
        (*srv).tool_count
    );
    let _ = ctx;
    qjs::sofuu_js_undefined()
}

thread_local! {
    static MCP_SERVER_PROTO: [qjs::JSCFunctionListEntry; 2] = [
        qjs::JSCFunctionListEntry {
            name: c"tool".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 3, cproto: 0, _pad: [0; 6], cfunc: js_server_tool },
        },
        qjs::JSCFunctionListEntry {
            name: c"start".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_server_start },
        },
    ];
}

// ── JS: sofuu.mcp.serve(opts?) → MCPServer ──────────────────────────

unsafe extern "C" fn js_mcp_serve(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let srv = Box::new(McpServer {
        ctx,
        self_val: qjs::sofuu_js_undefined(),
        tools: Vec::new(),
        tool_count: 0,
        stdin_pipe: ptr::null_mut(),
        stdout_pipe: ptr::null_mut(),
        read_buf: [0u8; MCP_READ_BUF],
        read_len: 0,
    });

    let obj = qjs::JS_NewObjectClass(ctx, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed) as c_int);
    let srv_ptr = Box::into_raw(srv);
    qjs::JS_SetOpaque(obj, srv_ptr as *mut c_void);
    (*srv_ptr).self_val = obj;

    obj
}

thread_local! {
    static MCP_FUNCS: [qjs::JSCFunctionListEntry; 2] = [
        qjs::JSCFunctionListEntry {
            name: c"connect".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_mcp_connect },
        },
        qjs::JSCFunctionListEntry {
            name: c"serve".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_mcp_serve },
        },
    ];
}

// ── registration (C symbol replacement for mod_mcp_register) ────────

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_mcp_register(ctx: *mut JSContext) {
    /* Register MCPClient class */
    let mut class_id: qjs::JSClassID = 0;
    qjs::JS_NewClassID(&mut class_id);
    MCP_CLIENT_CLASS_ID.store(class_id, Ordering::Relaxed);
    let client_class = qjs::JSClassDef {
        class_name: c"MCPClient".as_ptr(),
        finalizer: Some(mcp_client_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    let _nc1 = qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &client_class);
    let client_proto = qjs::sofuu_js_new_object(ctx);
    MCP_CLIENT_PROTO.with(|f| qjs::JS_SetPropertyFunctionList(ctx, client_proto, f.as_ptr(), 4));
    qjs::JS_SetClassProto(ctx, class_id, client_proto);

    /* Register MCPServer class — JS_NewClassID only ALLOCATES when
     * *pclass_id == 0 (quickjs.c:3396-3412), so reset the local like the
     * C code's `static JSClassID id = 0` did per class. */
    class_id = 0;
    qjs::JS_NewClassID(&mut class_id);
    MCP_SERVER_CLASS_ID.store(class_id, Ordering::Relaxed);
    let server_class = qjs::JSClassDef {
        class_name: c"MCPServer".as_ptr(),
        finalizer: Some(mcp_server_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    let _nc2 = qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &server_class);
    let server_proto = qjs::sofuu_js_new_object(ctx);
    MCP_SERVER_PROTO.with(|f| qjs::JS_SetPropertyFunctionList(ctx, server_proto, f.as_ptr(), 2));
    qjs::JS_SetClassProto(ctx, class_id, server_proto);

    /* Attach mcp object to sofuu global */
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    if qjs::is_undefined(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }

    let mcp_obj = qjs::sofuu_js_new_object(ctx);
    MCP_FUNCS.with(|f| qjs::JS_SetPropertyFunctionList(ctx, mcp_obj, f.as_ptr(), 2));
    qjs::sofuu_js_set_property_str(ctx, sofuu, c"mcp".as_ptr(), mcp_obj);

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}

// ── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// P0-2 regression (AUDIT-2026-09-07): a real MCP disconnect must run
    /// the 4-handle close fence to completion. Before the fix, stdin/stderr
    /// carried stale malloc garbage in ->data: mcp_client_close_cb either
    /// dereferenced it (wild read/write, sometimes abort) or the null check
    /// short-circuited and the fence stalled at closing=2 — the client and
    /// all four handle storages leaked. Either way the witness counter
    /// never reaches one free per disconnect, deterministically.
    #[test]
    fn mcp_disconnect_runs_the_close_fence() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        let base = TEST_MCP_CLIENT_FREES.load(Ordering::Relaxed);
        // SAFETY: standalone runtime + context (M0 pattern, as in the
        // P0-1 subprocess_write_delivers_payload test).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.mcp on the global.
        unsafe { mod_mcp_register(ctx) };

        // /usr/bin/true exits immediately: exit_cb → mcp_on_exit → close×4.
        // (Not /bin/true — macOS ships `true` in /usr/bin; the ENOENT there
        // is correct spawn behavior, verified live.) The connect promise
        // rejects ("exited before initialize"); the catch keeps it from
        // surfacing as an unhandled rejection.
        let script = c"sofuu.mcp.connect('/usr/bin/true').catch(function () {});";
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<mcp-p0-2>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            // The bare test runtime has no std helpers, so js_std_dump_error
            // is silent — pull the exception message out directly.
            let exc = unsafe { qjs::JS_GetException(ctx) };
            let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, exc) };
            let detail = if msg.is_null() {
                String::from("<unknown>")
            } else {
                // SAFETY: msg is a live C string from QuickJS.
                unsafe { std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned() }
            };
            unsafe {
                if !msg.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, msg);
                }
                qjs::sofuu_js_free_value(ctx, exc);
            }
            panic!("connect must not throw: {detail}");
        }
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: exit + all four close callbacks run, fence reaches zero.
        unsafe { crate::rt::event_loop::sofuu_loop_run(ctx) };

        let frees = TEST_MCP_CLIENT_FREES.load(Ordering::Relaxed) - base;
        assert_eq!(
            frees, 1,
            "the 4-handle close fence must free the client exactly once \
             (0 = fence stalled — a handle's ->data is not stamped)"
        );

        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Eval a script, panicking with the exception text on failure (the bare
    /// test runtime has no std helpers, so js_std_dump_error is silent).
    /// Returns the result value — caller frees.
    unsafe fn eval_or_panic(ctx: *mut JSContext, tag: &CStr, script: &CStr) -> JSValue {
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                tag.as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            let exc = unsafe { qjs::JS_GetException(ctx) };
            let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, exc) };
            let detail = if msg.is_null() {
                String::from("<unknown>")
            } else {
                // SAFETY: msg is a live C string from QuickJS.
                unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
            };
            unsafe {
                if !msg.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, msg);
                }
                qjs::sofuu_js_free_value(ctx, exc);
            }
            panic!("{} must not throw: {detail}", tag.to_string_lossy());
        }
        r
    }

    /// Read one newline-terminated response line from the capture pipe. The
    /// uv_write that produced it must already have been dispatched by the
    /// caller running the loop; the fd is blocking, so data (or EOF) waits.
    unsafe fn read_response_line(fd: c_int) -> String {
        let mut buf = [0u8; 4096];
        let mut out: Vec<u8> = Vec::new();
        loop {
            let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
            assert!(n >= 0, "read from capture pipe failed");
            if n == 0 {
                panic!("capture pipe closed before a response line arrived");
            }
            out.extend_from_slice(&buf[..n as usize]);
            if out.contains(&b'\n') {
                break;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Shared P1-7/P1-8 harness: standalone runtime with sofuu.mcp registered
    /// and the server's stdout wired to a capture pipe (uv_pipe_open ADOPTS
    /// the write end — libuv closes it at handle close, so the test owns only
    /// the read end). Returns (rt, ctx, server pointer, stdout pipe, read fd,
    /// base free count). The test never calls start(), which would adopt
    /// real stdio.
    unsafe fn p1_server_harness(
        script: &CStr,
    ) -> (
        *mut qjs::JSRuntime,
        *mut JSContext,
        *mut McpServer,
        *mut UvPipe,
        c_int,
        usize,
    ) {
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.mcp on the global.
        unsafe { mod_mcp_register(ctx) };

        let mut fds = [0 as c_int; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe");
        let resp_rd = fds[0];
        let stdout_pipe = unsafe { libc::malloc(uv::sofuu_uv_pipe_size()) } as *mut UvPipe;
        unsafe {
            uv::uv_pipe_init(crate::rt::event_loop::sofuu_loop_get(), stdout_pipe, 0);
            uv::uv_pipe_open(stdout_pipe, fds[1] as u64);
        }

        // SAFETY: script registers globalThis.srv (+ tools); may not throw.
        unsafe {
            let r = eval_or_panic(ctx, c"<mcp-p1>", script);
            qjs::sofuu_js_free_value(ctx, r);
        }

        // Recover the server pointer from the live JS object and wire stdout.
        let srv = unsafe {
            let v = eval_or_panic(ctx, c"<mcp-p1-srv>", c"globalThis.srv");
            assert!(qjs::is_object(v), "globalThis.srv must be the server object");
            let p = qjs::JS_GetOpaque(v, MCP_SERVER_CLASS_ID.load(Ordering::Relaxed)) as *mut McpServer;
            qjs::sofuu_js_free_value(ctx, v);
            assert!(!p.is_null(), "server opaque missing");
            (*p).stdout_pipe = stdout_pipe;
            p
        };
        let base = TEST_MCP_SERVER_FREES.load(Ordering::Relaxed);
        (rt, ctx, srv, stdout_pipe, resp_rd, base)
    }

    /// Shared teardown: close the capture handle (fires free_pipe_cb — frees
    /// the storage; libuv closes the adopted write end), then the read end,
    /// then the loop + runtime.
    unsafe fn p1_server_teardown(
        rt: *mut qjs::JSRuntime,
        ctx: *mut JSContext,
        stdout_pipe: *mut UvPipe,
        resp_rd: c_int,
    ) {
        unsafe {
            uv::uv_close(stdout_pipe as *mut UvHandle, Some(free_pipe_cb));
            uv::uv_run(crate::rt::event_loop::sofuu_loop_get(), uv::UV_RUN_ONCE);
            crate::rt::event_loop::sofuu_loop_close();
            libc::close(resp_rd);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    /// P1-7 regression (AUDIT-2026-09-07): dropping the JS server object
    /// while an async tool call is pending must NOT free the server Box —
    /// the settle callbacks capture a GC-marked reference to the server
    /// (C-function data), not a raw pointer. The old int64 capture let the
    /// finalizer run between `.then` attach and settle, making every later
    /// read of data[1] a use-after-free; the counter pins that precondition
    /// deterministically (free while pending = delta 1, not 0).
    #[test]
    fn mcp_pending_tool_promise_keeps_server_alive_until_settle() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        // SAFETY: standalone harness per the P0-2 pattern.
        let (rt, ctx, srv, stdout_pipe, resp_rd, base) =
            unsafe { p1_server_harness(c"globalThis.srv = sofuu.mcp.serve(); globalThis.resolveIt = null; srv.tool('slow', {}, function () { return new Promise(function (res) { globalThis.resolveIt = res; }); });") };

        // Fire an async tools/call: handle_request attaches both settle
        // handlers (their C-function data holds a reference to the server
        // object) and flushes jobs once — the promise is still pending.
        let call = c"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"slow\",\"arguments\":{}}}";
        unsafe { mcp_server_handle_request(srv, call.to_bytes()) };

        // THE P1-7 INVARIANT: drop the only JS reference and collect — the
        // pending reaction pins the server, so the finalizer must not run.
        unsafe {
            let r = eval_or_panic(ctx, c"<mcp-p1-7-drop>", c"globalThis.srv = null;");
            qjs::sofuu_js_free_value(ctx, r);
            qjs::JS_RunGC(rt);
        }
        let delta_pending = TEST_MCP_SERVER_FREES.load(Ordering::Relaxed) - base;
        assert_eq!(
            delta_pending, 0,
            "server Box was freed while a tool promise was still pending — \
             the settle-callback capture is not GC-reachable (P1-7 UAF precondition)"
        );

        // Settle: the reaction job runs, mcp_tool_then_cb stringifies the
        // result and writes the response through the capture pipe.
        unsafe {
            let r = eval_or_panic(ctx, c"<mcp-p1-7-settle>", c"globalThis.resolveIt('done');");
            qjs::sofuu_js_free_value(ctx, r);
            sofuu_flush_jobs(ctx);
            // uv_write is asynchronous — dispatch it before reading.
            uv::uv_run(crate::rt::event_loop::sofuu_loop_get(), uv::UV_RUN_ONCE);
        }
        let line = unsafe { read_response_line(resp_rd) };
        assert!(
            line.contains("\"id\":7") && line.contains("done"),
            "async tool response must arrive after settle, got: {line}"
        );

        // After settle the reaction detaches, the captured reference dies,
        // and the server is freed exactly once.
        unsafe {
            qjs::JS_RunGC(rt);
            sofuu_flush_jobs(ctx);
        }
        let delta_total = TEST_MCP_SERVER_FREES.load(Ordering::Relaxed) - base;
        assert_eq!(delta_total, 1, "server must be freed exactly once, after settle");

        unsafe { p1_server_teardown(rt, ctx, stdout_pipe, resp_rd) };
    }

    /// P1-8 regression (AUDIT-2026-09-07): a rejected async tool handler
    /// must still produce a JSON-RPC error line — before the fix `.then`
    /// carried only a fulfill handler, so rejections answered nothing and
    /// the client's pending slot (MAX_PENDING=64) leaked per rejection.
    #[test]
    fn mcp_rejected_tool_handler_still_answers_the_request() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        // SAFETY: standalone harness per the P0-2 pattern.
        let (rt, ctx, srv, stdout_pipe, resp_rd, _base) =
            unsafe { p1_server_harness(c"globalThis.srv = sofuu.mcp.serve(); srv.tool('boom', {}, function () { return Promise.reject(new Error('boom')); });") };

        // The already-rejected promise settles its reaction during
        // handle_request's internal flush; dispatch the queued uv_write.
        let call = c"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{\"name\":\"boom\",\"arguments\":{}}}";
        unsafe { mcp_server_handle_request(srv, call.to_bytes()) };
        unsafe { uv::uv_run(crate::rt::event_loop::sofuu_loop_get(), uv::UV_RUN_ONCE) };

        let line = unsafe { read_response_line(resp_rd) };
        assert!(
            line.contains("\"id\":9") && line.contains("\"error\"") && line.contains("-32603"),
            "rejected handler must answer with a JSON-RPC error, got: {line}"
        );

        unsafe { p1_server_teardown(rt, ctx, stdout_pipe, resp_rd) };
    }

    /// P1-9 regression (AUDIT-2026-09-07): a synchronous uv_write failure
    /// (EPIPE/EBADF after child death) must settle the pending `call()`
    /// promise and reclaim the MAX_PENDING slot — before the fix the rc was
    /// swallowed, libuv never fires on_write_done on a sync failure, and the
    /// promise hung forever while slots piled up until the client wedged.
    /// The failure vector is deterministic without any process: a freshly
    /// uv_pipe_init'd pipe was never opened, so its fd is −1 and
    /// uv__check_before_write returns UV_EBADF before uv__req_init (deps/
    /// libuv/src/unix/stream.c) — on_write_done will never run.
    #[test]
    fn mcp_sync_write_failure_settles_the_call_promise() {
        let _loop_guard = crate::rt::TEST_LOOP_LOCK.lock().unwrap();
        // SAFETY: standalone harness per the P0-2 pattern.
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.mcp + both class IDs on the runtime.
        unsafe { mod_mcp_register(ctx) };

        // A Rust-constructed client with NO spawn: stdin is a pipe that was
        // init'd but never opened (fd −1). js_mcp_call recovers it through
        // JS_GetOpaque exactly as it would a connect()ed client.
        let stdin_pipe = unsafe { libc::malloc(uv::sofuu_uv_pipe_size()) } as *mut UvPipe;
        unsafe { uv::uv_pipe_init(crate::rt::event_loop::sofuu_loop_get(), stdin_pipe, 0) };
        let mut client = unsafe {
            Box::new(McpClient {
                process: ptr::null_mut(),
                stdin_pipe,
                stdout_pipe: ptr::null_mut(),
                stderr_pipe: ptr::null_mut(),
                ctx,
                self_val: qjs::sofuu_js_undefined(),
                read_buf: [0u8; MCP_READ_BUF],
                read_len: 0,
                pending: [PendingCall { id: 0, promise: ptr::null_mut(), ctx: ptr::null_mut() }; MAX_PENDING],
                pending_count: 0,
                initialized: 0,
                initialize_id: 0,
                connect_promise: ptr::null_mut(),
                done: 0,
                closing: 0,
            })
        };
        let client_ptr = Box::into_raw(client);
        // Same wiring as js_mcp_connect: the single ref lives in self_val.
        unsafe {
            (*client_ptr).self_val =
                qjs::JS_NewObjectClass(ctx, MCP_CLIENT_CLASS_ID.load(Ordering::Relaxed) as c_int);
            qjs::JS_SetOpaque((*client_ptr).self_val, client_ptr as *mut c_void);
        }

        // Fire the call: argv[0] = method string; this_val = the client object.
        let method_val = unsafe { qjs::sofuu_js_new_string(ctx, c"tools/list".as_ptr()) };
        let argv_arr: [qjs::JSValueConst; 1] = [method_val];
        let ret = unsafe { js_mcp_call(ctx, (*client_ptr).self_val, 1, argv_arr.as_ptr()) };
        unsafe { qjs::sofuu_js_free_value(ctx, method_val) };
        if unsafe { qjs::is_exception(ret) } {
            let exc = unsafe { qjs::sofuu_js_get_exception(ctx) };
            let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, exc) };
            let detail = if msg.is_null() { String::from("<unknown>") } else {
                unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() }
            };
            unsafe {
                if !msg.is_null() { qjs::sofuu_js_free_cstring(ctx, msg); }
                qjs::sofuu_js_free_value(ctx, exc);
            }
            panic!("call must not throw: {detail}");
        }
        assert!(unsafe { qjs::is_object(ret) }, "call must return the promise");

        // Park the promise on the global (SetPropertyStr steals the ref),
        // attach a catch, and drain the reaction the fix's flush queued.
        unsafe {
            let global = qjs::sofuu_js_get_global_object(ctx);
            qjs::sofuu_js_set_property_str(ctx, global, c"p9".as_ptr(), ret);
            qjs::sofuu_js_free_value(ctx, global);
            let r = eval_or_panic(ctx, c"<mcp-p1-9-catch>",
                c"globalThis.p9msg = null; globalThis.p9.catch(function (e) { globalThis.p9msg = String(e); });");
            qjs::sofuu_js_free_value(ctx, r);
            sofuu_flush_jobs(ctx);
            let r = eval_or_panic(ctx, c"<mcp-p1-9-read>", c"String(globalThis.p9msg)");
            assert!(qjs::sofuu_js_is_string(r) != 0, "catch handler must have settled the promise");
            let s = qjs::sofuu_js_to_cstring(ctx, r);
            let msg = CStr::from_ptr(s).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, s);
            qjs::sofuu_js_free_value(ctx, r);
            assert_eq!(
                msg, "MCP write failed (server unreachable)",
                "the pending promise must settle with the write-failure reason"
            );
        }

        // THE P1-9 INVARIANT: the slot was reclaimed synchronously.
        assert_eq!(
            unsafe { (*client_ptr).pending_count },
            0,
            "failed write must reclaim its pending slot (rc swallowed = leak)"
        );

        // Teardown: never-spawned client — free the pipe through the close
        // callback (free_pipe_cb frees the storage; it never reads ->data),
        // detach the opaque, drop the Box manually (finalizer is a no-op).
        unsafe {
            uv::uv_close(stdin_pipe as *mut UvHandle, Some(free_pipe_cb));
            uv::uv_run(crate::rt::event_loop::sofuu_loop_get(), uv::UV_RUN_ONCE);
            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_SetOpaque((*client_ptr).self_val, ptr::null_mut());
            qjs::sofuu_js_free_value(ctx, (*client_ptr).self_val);
            drop(Box::from_raw(client_ptr));
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }
}
