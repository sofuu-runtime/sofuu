// rt/http_client.rs — the Fetch API over libcurl (PLAN-RUST-MIGRATION M4).
//
// Port of the deleted `src/http/client.c` (1,032 lines), semantics verbatim:
// fetch(url, {method, headers, body}) → Promise<Response> with
// Response.{status,ok,statusText,url,headers.get,text,json,arrayBuffer} and
// streaming Response.body (async iterator over curl write-callback chunks).
// One CURLM multi handle is bridged to the libuv loop via
// CURLMOPT_SOCKETFUNCTION/TIMERFUNCTION + uv_poll sockets + a uv_check_t
// that fulfils stream waiters OUTSIDE curl callbacks (never run JS inside
// curl_multi_socket_action — CURLM_RECURSIVE_API_CALL).
//
// C symbols replaced: `mod_http_client_register` (engine.c calls it
// unchanged) + the global `fetch`/`sofuu.fetch`/`__fetch_stream_*` surface.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
use std::ptr;

use sofuu_ffi::curl::{self, Curl, CurlM, CurlSlist};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvCheck, UvHandle, UvPoll, UvTimer};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject, sofuu_promise_reject_str,
    sofuu_promise_resolve, PromiseHandle,
};

const STREAM_CAP: usize = 256 * 1024 * 1024; /* 256MB hard cap, like C */
const BODY_CAP: usize = 256 * 1024 * 1024; /* 256MB hard cap on the buffered body */
const MAX_REDIRECTS: c_int = 5; /* manual hop cap (net-4); replaces FOLLOWLOCATION */
const HEADER_CAP: usize = 256 * 1024; /* accumulated header-block cap, per response stage */

/// Test-only gate that restores the pre-net-7 per-byte chunk fill so the
/// bulk-delivery test can A/B the two paths in one process. Never set in
/// production builds (cfg(test) + default false).
#[cfg(test)]
static NET7_LEGACY_FILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Test-only A/B instrumentation: how many chunk fills ran, on which branch,
/// and how much time the fill itself took (thread-local; printed per leg).
#[cfg(test)]
thread_local! {
    static N7_FILL_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static N7_FILL_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static N7_LEGACY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// ── global curl-multi / libuv state ──────────────────────────────────

thread_local! {
    static CURL_HANDLE: Cell<*mut CurlM> = const { Cell::new(ptr::null_mut()) };
    static TIMEOUT_TIMER: Cell<*mut UvTimer> = const { Cell::new(ptr::null_mut()) };
    static GLOBAL_INIT_DONE: Cell<c_int> = const { Cell::new(0) };

    /// Live responses — so the uv_check_t can fulfil waiters. (C used an
    /// intrusive linked list; Vec iteration order is functionally equal.)
    static RESPONSES: RefCell<Vec<*mut ResponseData>> = const { RefCell::new(Vec::new()) };

    static STREAM_CHECK: Cell<*mut UvCheck> = const { Cell::new(ptr::null_mut()) };
    static STREAM_CHECK_INIT: Cell<c_int> = const { Cell::new(0) };
}

fn curl_multi() -> *mut CurlM {
    CURL_HANDLE.with(|c| c.get())
}

// ── fetch_req_t — per-request state ──────────────────────────────────

struct FetchReq {
    /* Magic check: curl callbacks can fire on stale userdata (e.g. a
     * socket event after the transfer's DONE freed the req). Derefing
     * garbage was the model-picker SIGBUS. */
    magic: usize,
    ctx: *mut JSContext,
    promise: *mut PromiseHandle,

    /* Response body */
    body: Vec<u8>,

    /* Raw response headers (concatenated) */
    headers_raw: Vec<u8>,

    /* The streaming response (created on the first header block; NULL until
     * then or if the transfer fails before any header arrives). */
    resp: *mut ResponseData,
    resp_cap: usize,

    /* libcurl handle + slist */
    easy: *mut Curl,
    req_headers: *mut CurlSlist,

    /* Manual redirect state (net-4/net-5/P1-3): FOLLOWLOCATION is off, so
     * check_multi_info_global re-issues 3xx hops itself. */
    hops: c_int,
    /* 1 once opensocket_cb refused a link-local/metadata address (net-4). */
    refused: c_int,
    /* Parsed request headers, kept so each redirect hop can rebuild a fresh
     * slist (credentials get stripped on origin change — net-5). CurlSlist
     * is opaque in the FFI, so the list cannot be walked/rebuilt in place. */
    headers_vec: Vec<(String, String)>,
    /* CUSTOMREQUEST value if the caller set one (method-conversion rules). */
    method: Option<String>,
    /* js-9 (AUDIT-2026-09-07): per-request body cap from options.maxBodyBytes
     * (0 = the BODY_CAP default). write_body_cb aborts the transfer when the
     * buffered body would cross it; cap_hit makes the DONE error path word
     * the rejection after the cap instead of a generic curl write error. */
    max_body: usize,
    cap_hit: c_int,
}

const REQ_MAGIC: usize = 0x5FF51_0002;

/// Validate the userdata passed to a curl callback. On failure: log once
/// and return false — the callback then aborts the transfer (return 0)
/// instead of dereferencing stale memory.
unsafe fn valid_req(req: *mut FetchReq) -> bool {
    if req.is_null() || (*req).magic != REQ_MAGIC {
        eprintln!("[http] stale curl callback userdata {req:p}");
        return false;
    }
    true
}

/// Response JS class state — owned by the QuickJS finalizer (Box::into_raw
/// at build_response; freed in response_finalizer).
struct ResponseData {
    body: Vec<u8>,
    headers_raw: Vec<u8>,
    url: CString,
    status_text: CString,

    /* Streaming: enqueued chunks + end flags. */
    chunks: VecDeque<Vec<u8>>,
    done: c_int,
    errored: c_int,
    /* P1-5: curl's error text captured when the transfer failed mid-flight —
     * consumers reject with this instead of a clean end-of-stream. */
    err_msg: Option<String>,
    /* 1 once the finalizer ran while the transfer was still writing into
     * this ResponseData — the free is deferred to the transfer's DONE. */
    finalized: usize,
    pending_next: *mut PromiseHandle, /* one awaiting consumer at a time */
    pending_resolve: JSValue,         /* the resolver for the stream */
    /* net-8: a LIST — a single slot made the second mid-stream text()/
     * json()/arrayBuffer() waiter pre-resolve and read a partial body. */
    pending_settle: Vec<*mut PromiseHandle>,
    ctx: *mut JSContext,              /* the context that owns the response */
}

// ── Response finalizer ───────────────────────────────────────────────

unsafe extern "C" fn response_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let r = qjs::JS_GetOpaque(val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return;
    }
    let r = r as *mut ResponseData;
    /* If the transfer is still writing into this ResponseData (streaming:
     * req.resp → it), the free must be deferred to the transfer's DONE —
     * the curl callbacks fault on freed memory (the app-closing crash).
     * Keep the response in the RESPONSES list and its waiters pending: the
     * DONE pump settles them and the orphan is reclaimed at the next fetch. */
    if (*r).done == 0 {
        (*r).finalized = 1;
        return;
    }

    /* Transfer finished — the response is dead from here on. */
    RESPONSES.with(|rs| {
        let mut rs = rs.borrow_mut();
        if let Some(i) = rs.iter().position(|x| *x == r) {
            rs.swap_remove(i);
        }
    });
    /* Still-pending settle waiters would leak — reject them all (net-8). */
    for p in (*r).pending_settle.drain(..) {
        sofuu_promise_reject_str(p, c"response finalized".as_ptr());
    }
    if !(*r).pending_next.is_null() {
        /* A pending stream waiter would leak too — reject it. */
        let p = (*r).pending_next;
        sofuu_promise_reject_str(p, c"response finalized".as_ptr());
        (*r).pending_next = ptr::null_mut();
    }
    // pending_resolve holds a dup'd JSValue resolver (see js_fetch_stream_wait).
    // Free it here or one JSValue ref leaks per GC'd streaming Response.
    // JSValue is a tagged 64-bit value; JS_FREE value releases the ref.
    if !qjs::is_undefined((*r).pending_resolve) {
        qjs::sofuu_js_free_value((*r).ctx, (*r).pending_resolve);
        (*r).pending_resolve = qjs::sofuu_js_undefined();
    }
    drop(Box::from_raw(r));
}

static RESPONSE_CLASS_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/* ResponseData whose JS object was collected before the transfer finished.
 * The transfer's DONE settles its waiters; the body remains readable only
 * inside that eval_string, so reclaiming at the NEXT fetch (when every
 * prior microtask has long flushed) is the only safe free point. */
thread_local! {
    static ORPHAN_RESP: std::cell::Cell<*mut ResponseData> = const { std::cell::Cell::new(ptr::null_mut()) };
}

/// Reclaim the previous orphaned response (safe: called at fetch start,
/// far from any curl callback and after all earlier microtasks ran).
unsafe fn reclaim_orphan_resp() {
    let orphan = ORPHAN_RESP.with(|o| o.replace(ptr::null_mut()));
    if !orphan.is_null() {
        RESPONSES.with(|rs| {
            let mut rs = rs.borrow_mut();
            if let Some(i) = rs.iter().position(|x| *x == orphan) {
                rs.swap_remove(i);
            }
        });
        /* net-9: waiters the DONE pump never reached would leak their
         * handles and hang their awaits forever — reject before freeing,
         * mirroring response_finalizer's cleanup (net-8: all of them). */
        for p in (*orphan).pending_settle.drain(..) {
            sofuu_promise_reject_str(p, c"response finalized".as_ptr());
        }
        if !(*orphan).pending_next.is_null() {
            sofuu_promise_reject_str(
                (*orphan).pending_next,
                c"response finalized".as_ptr(),
            );
            (*orphan).pending_next = ptr::null_mut();
        }
        if !qjs::is_undefined((*orphan).pending_resolve) {
            qjs::sofuu_js_free_value((*orphan).ctx, (*orphan).pending_resolve);
            (*orphan).pending_resolve = qjs::sofuu_js_undefined();
        }
        drop(Box::from_raw(orphan));
    }
}

// ── Response.text() / json() / arrayBuffer() ─────────────────────────

/// The transport-error text carried by an errored ResponseData (P1-5) —
/// the curl message captured at the completion ERROR branch.
unsafe fn resp_err_msg(r: *mut ResponseData) -> String {
    (*r).err_msg
        .clone()
        .unwrap_or_else(|| "fetch response incomplete: connection error".to_string())
}

/// A promise rejected with the response's transport error (P1-5: an errored
/// transfer must surface as a rejection everywhere — stream next()/wait(),
/// pending read fulfillment, settle waiters, and direct body reads — never
/// as a clean end-of-stream or a truncated-body resolve).
unsafe fn rejected_body_promise(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    let msg = resp_err_msg(r);
    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let err = new_fetch_error(ctx, &msg);
    let ret = qjs::JS_Call(ctx, resolvers[1], qjs::sofuu_js_undefined(), 1, &err);
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, err);
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

unsafe fn response_text_impl(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: never resolve with a truncated body. */
        return rejected_body_promise(ctx, r);
    }
    let data = (*r).body.clone();
    /* net-1: JS_NewStringLen is length-based, so pass the raw bytes —
     * a CString detour truncated the body at its first NUL byte. */
    let str_v = qjs::JS_NewStringLen(ctx, data.as_ptr() as *const c_char, data.len());

    /* Wrap in a pre-resolved promise (spec says .text() returns Promise) */
    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 1, &str_v);
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, str_v);
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

/// JS_NewCFunctionData callback: (ctx, this, argc, argv, magic, data).
unsafe extern "C" fn response_text_then(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let mut r: *mut ResponseData = ptr::null_mut();
    if !data.is_null() {
        let mut p: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut p, *data);
        r = p as *mut ResponseData;
    }
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    response_text_impl(ctx, r)
}

unsafe extern "C" fn response_text(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let r = qjs::JS_GetOpaque(this_val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    let r = r as *mut ResponseData;
    if (*r).done != 0 {
        return response_text_impl(ctx, r);
    }

    /* Wait for the transfer, then read the body. */
    let wait = response_wait_done(ctx, r);
    let data_arr: [JSValue; 1] = [qjs::sofuu_js_new_int64(ctx, r as i64)];
    let cb = qjs::JS_NewCFunctionData(ctx, response_text_then, 0, 0, 1, data_arr.as_ptr());
    let then_fn = qjs::sofuu_js_get_property_str(ctx, wait, c"then".as_ptr());
    let chained = qjs::JS_Call(ctx, then_fn, wait, 1, &cb);
    qjs::sofuu_js_free_value(ctx, then_fn);
    qjs::sofuu_js_free_value(ctx, cb);
    qjs::sofuu_js_free_value(ctx, data_arr[0]);
    qjs::sofuu_js_free_value(ctx, wait);
    chained
}

unsafe fn response_json_impl(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: never resolve with a truncated body. */
        return rejected_body_promise(ctx, r);
    }
    /* net-1: parse the raw bytes — a CString detour dropped everything
     * past the first NUL, silently parsing only the surviving prefix.
     * Empty body still parses as literal "null". */
    let src: &[u8] = if (*r).body.is_empty() { b"null" } else { &(*r).body };
    let parsed = qjs::JS_ParseJSON(
        ctx,
        src.as_ptr() as *const c_char,
        src.len(),
        c"<fetch response>".as_ptr(),
    );

    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    if qjs::is_exception(parsed) {
        /* Reject with the parse error */
        let exc = qjs::sofuu_js_get_exception(ctx);
        let ret = qjs::JS_Call(ctx, resolvers[1], qjs::sofuu_js_undefined(), 1, &exc);
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, exc);
    } else {
        let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 1, &parsed);
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, parsed);
    }
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

unsafe extern "C" fn response_json_then(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let mut r: *mut ResponseData = ptr::null_mut();
    if !data.is_null() {
        let mut p: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut p, *data);
        r = p as *mut ResponseData;
    }
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    response_json_impl(ctx, r)
}

unsafe extern "C" fn response_json(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let r = qjs::JS_GetOpaque(this_val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    let r = r as *mut ResponseData;
    if (*r).done != 0 {
        return response_json_impl(ctx, r);
    }

    let wait = response_wait_done(ctx, r);
    let data_arr: [JSValue; 1] = [qjs::sofuu_js_new_int64(ctx, r as i64)];
    let cb = qjs::JS_NewCFunctionData(ctx, response_json_then, 0, 0, 1, data_arr.as_ptr());
    let then_fn = qjs::sofuu_js_get_property_str(ctx, wait, c"then".as_ptr());
    let chained = qjs::JS_Call(ctx, then_fn, wait, 1, &cb);
    qjs::sofuu_js_free_value(ctx, then_fn);
    qjs::sofuu_js_free_value(ctx, cb);
    qjs::sofuu_js_free_value(ctx, data_arr[0]);
    qjs::sofuu_js_free_value(ctx, wait);
    chained
}

unsafe fn response_ab_impl(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: never resolve with a truncated body. */
        return rejected_body_promise(ctx, r);
    }
    let len = (*r).body.len();
    /* js_malloc takes the JSContext (quickjs.h:393), not the runtime —
     * passing JS_GetRuntime(ctx) here corrupted the malloc-state read. */
    let buf = qjs::js_malloc(ctx, if len > 0 { len } else { 1 });
    let buf = buf as *mut u8;
    if len > 0 {
        std::ptr::copy_nonoverlapping((*r).body.as_ptr(), buf, len);
    }
    let ab = qjs::JS_NewArrayBuffer(ctx, buf as *mut c_void, len, None, ptr::null_mut(), 0);

    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 1, &ab);
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, ab);
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

unsafe extern "C" fn response_ab_then(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    let mut r: *mut ResponseData = ptr::null_mut();
    if !data.is_null() {
        let mut p: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut p, *data);
        r = p as *mut ResponseData;
    }
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    response_ab_impl(ctx, r)
}

unsafe extern "C" fn response_array_buffer(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let r = qjs::JS_GetOpaque(this_val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    let r = r as *mut ResponseData;
    if (*r).done != 0 {
        return response_ab_impl(ctx, r);
    }

    let wait = response_wait_done(ctx, r);
    let data_arr: [JSValue; 1] = [qjs::sofuu_js_new_int64(ctx, r as i64)];
    let cb = qjs::JS_NewCFunctionData(ctx, response_ab_then, 0, 0, 1, data_arr.as_ptr());
    let then_fn = qjs::sofuu_js_get_property_str(ctx, wait, c"then".as_ptr());
    let chained = qjs::JS_Call(ctx, then_fn, wait, 1, &cb);
    qjs::sofuu_js_free_value(ctx, then_fn);
    qjs::sofuu_js_free_value(ctx, cb);
    qjs::sofuu_js_free_value(ctx, data_arr[0]);
    qjs::sofuu_js_free_value(ctx, wait);
    chained
}

/*
 * headers.get(name) — minimal but correct case-insensitive lookup.
 * The function is bound (JS_NewCFunctionData) to the live ResponseData
 * pointer, so every call sees the headers that have arrived so far —
 * including the ones that came in AFTER the Response object was built.
 */
unsafe extern "C" fn headers_get(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_null();
    }
    let mut r: *mut ResponseData = ptr::null_mut();
    if !data.is_null() {
        let mut p: i64 = 0;
        qjs::JS_ToInt64(ctx, &mut p, *data);
        r = p as *mut ResponseData;
    }
    if r.is_null() || !is_live_response(r) {
        return qjs::sofuu_js_null();
    }
    let raw: &[u8] = &(*r).headers_raw;

    let name_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if name_ptr.is_null() {
        return qjs::sofuu_js_null();
    }
    let name = CStr::from_ptr(name_ptr).to_bytes();

    /* Case-insensitive scan through raw headers.
     * P2-34 (AUDIT-2026-09-01): duplicate header names return the LAST
     * occurrence (the JS Headers/Response spec — get() is last-wins), and
     * obs-fold continuation lines (a line starting with SP/HTAB, RFC 7230
     * deprecated but still emitted by some servers) join the previous
     * header's value with a single space. */
    let mut result = qjs::sofuu_js_null();
    let mut p: &[u8] = raw;
    /* Pending match: (value bytes) of the most recent matching header,
     * possibly still being extended by obs-fold lines. */
    let mut pending: Option<Vec<u8>> = None;
    /* True while subsequent fold lines extend `pending`. */
    let mut extending = false;

    while !p.is_empty() {
        /* Find end of this header line */
        let eol = p
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|i| i)
            .unwrap_or(p.len());

        /* Obs-fold continuation: a line starting with SP/HTAB extends the
         * previous header line's value. */
        if p[0] == b' ' || p[0] == b'\t' {
            if extending {
                if let Some(v) = pending.as_mut() {
                    let mut vstart = 0usize;
                    while vstart < eol && (p[vstart] == b' ' || p[vstart] == b'\t') {
                        vstart += 1;
                    }
                    let mut vend = eol;
                    while vend > vstart && (p[vend - 1] == b'\r' || p[vend - 1] == b' ') {
                        vend -= 1;
                    }
                    if vend > vstart {
                        v.push(b' ');
                        v.extend_from_slice(&p[vstart..vend]);
                    }
                }
            }
            /* Non-extending folds (after a non-matching header) are ignored. */
            if eol < p.len() && p[eol] == b'\r' {
                p = &p[(eol + 2).min(p.len())..];
            } else {
                p = &p[(eol + 1).min(p.len())..];
            }
            continue;
        }

        /* A fresh header line closes the obs-fold window (the previous
         * match is final). The match itself SURVIVES non-matching lines —
         * clearing it here made every lookup miss whenever the wanted
         * header wasn't the LAST header on the wire (P2-34 regression,
         * caught by the rewritten fetch_test against python http.server). */
        extending = false;

        /* Find the colon */
        if let Some(colon) = p[..eol].iter().position(|&b| b == b':') {
            let key_len = colon;
            if key_len == name.len() && p[..key_len].eq_ignore_ascii_case(name) {
                /* Found it — extract value; keep scanning so a LATER
                 * duplicate can win. */
                let mut vstart = colon + 1;
                while vstart < p.len() && p[vstart] == b' ' {
                    vstart += 1;
                }
                let mut vlen = eol.saturating_sub(vstart);
                while vlen > 0 && (p[vstart + vlen - 1] == b'\r' || p[vstart + vlen - 1] == b' ') {
                    vlen -= 1;
                }
                pending = Some(p[vstart..vstart + vlen].to_vec());
                extending = true;
            }
        }

        if eol < p.len() && p[eol] == b'\r' {
            p = &p[(eol + 2).min(p.len())..];
        } else {
            p = &p[(eol + 1).min(p.len())..];
        }
    }

    if let Some(v) = pending {
        let c = CString::new(v).unwrap_or_default();
        result = qjs::JS_NewStringLen(ctx, c.as_ptr(), c.as_bytes().len());
    }

    qjs::sofuu_js_free_cstring(ctx, name_ptr);
    result
}

// ── Streaming helpers ────────────────────────────────────────────────

/// Drain one chunk into the consumer-facing Uint8Array.
unsafe fn response_next_chunk(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    let Some(chunk) = (*r).chunks.pop_front() else {
        return qjs::sofuu_js_undefined();
    };

    /* net-7: one JS_NewArrayBufferCopy + a single Uint8Array view — the old
     * per-byte JS_SetPropertyUint32 loop crossed the FFI once per byte,
     * which dominated large-stream delivery. The constructor call is kept
     * (not bypassed) because the P1-4 GC probe depends on running inside
     * it; `new Uint8Array(ab)` copies again once, still O(n) native. */
    #[cfg(test)]
    let t_fill = std::time::Instant::now();
    #[cfg(test)]
    N7_FILL_CALLS.with(|c| c.set(c.get() + 1));
    #[cfg(test)]
    if NET7_LEGACY_FILL.load(std::sync::atomic::Ordering::Relaxed) {
        /* A/B leg: pre-fix per-byte fill (used only by the bulk test). */
        let global = qjs::sofuu_js_get_global_object(ctx);
        let ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Uint8Array".as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let lenv = qjs::sofuu_js_new_int32(ctx, chunk.len() as i32);
        let arr = qjs::JS_CallConstructor(ctx, ctor, 1, &lenv);
        qjs::sofuu_js_free_value(ctx, lenv);
        qjs::sofuu_js_free_value(ctx, ctor);
        if !qjs::is_exception(arr) {
            for (i, &b) in chunk.iter().enumerate() {
                qjs::JS_SetPropertyUint32(ctx, arr, i as u32, qjs::sofuu_js_new_int32(ctx, b as i32));
            }
        }
        N7_LEGACY_CALLS.with(|c| c.set(c.get() + 1));
        N7_FILL_NS.with(|c| c.set(c.get() + t_fill.elapsed().as_nanos() as u64));
        return arr;
    }
    let global = qjs::sofuu_js_get_global_object(ctx);
    let ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Uint8Array".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let ab = qjs::JS_NewArrayBufferCopy(ctx, chunk.as_ptr(), chunk.len());
    if qjs::is_exception(ab) {
        qjs::sofuu_js_free_value(ctx, ctor);
        #[cfg(test)]
        N7_FILL_NS.with(|c| c.set(c.get() + t_fill.elapsed().as_nanos() as u64));
        return ab;
    }
    let arr = qjs::JS_CallConstructor(ctx, ctor, 1, &ab);
    qjs::sofuu_js_free_value(ctx, ab);
    qjs::sofuu_js_free_value(ctx, ctor);
    #[cfg(test)]
    N7_FILL_NS.with(|c| c.set(c.get() + t_fill.elapsed().as_nanos() as u64));
    arr
}

/// The JS stream implementation (one global instance, driven per-response).
const BODY_ITER_FACTORY: &str = "(function() { \
    return { \
      [Symbol.asyncIterator]: function() { return this; }, \
      next: function() { \
        var r = globalThis.__fetch_stream_get(this); \
        if (r === null) return Promise.resolve({ done: true }); \
        var res = globalThis.__fetch_stream_next(r); \
        if (res) return Promise.resolve(res); \
        return globalThis.__fetch_stream_wait(r); \
      } \
    }; \
  })";

/// Response.body — the async iterator object.
unsafe extern "C" fn response_body_getter(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let r = qjs::JS_GetOpaque(this_val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response".as_ptr());
    }
    /* P3 (AUDIT-2026-09-07): the factory is compiled once at boot and
     * cached as a global — re-evaluating the same source on every body
     * access was pure waste. Fall back to an inline eval if the boot-time
     * global is missing. */
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut factory =
        qjs::sofuu_js_get_property_str(ctx, global, c"__fetch_body_iter_factory".as_ptr());
    qjs::sofuu_js_free_value(ctx, global);
    let mut owned = false;
    if qjs::is_undefined(factory) || qjs::is_null(factory) {
        qjs::sofuu_js_free_value(ctx, factory);
        owned = true;
        let factory_c = CString::new(BODY_ITER_FACTORY).unwrap();
        factory = qjs::JS_Eval(
            ctx,
            factory_c.as_ptr(),
            factory_c.as_bytes().len(),
            c"<body-iter>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
    }
    if qjs::is_exception(factory) {
        return qjs::sofuu_js_exception();
    }
    let iter = qjs::JS_Call(ctx, factory, qjs::sofuu_js_undefined(), 0, ptr::null());
    if owned {
        qjs::sofuu_js_free_value(ctx, factory);
    }
    if qjs::is_exception(iter) {
        return qjs::sofuu_js_exception();
    }
    /* Store the response pointer on the iterator for the bridges. */
    qjs::sofuu_js_set_property_str(ctx, iter, c"__resp".as_ptr(), qjs::sofuu_js_new_int64(ctx, r as i64));
    iter
}

/// Validate that `p` is a live ResponseData pointer from the RESPONSES list
/// (bridges receive raw int64s from JS, so we must not deref arbitrary values).
fn is_live_response(r: *const ResponseData) -> bool {
    RESPONSES.with(|rs| rs.borrow().iter().any(|x| *x as *const ResponseData == r))
}

/// __fetch_stream_get(iter) → the response pointer (int64) or JS_NULL.
unsafe extern "C" fn js_fetch_stream_get(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || !qjs::is_object(*argv) {
        return qjs::sofuu_js_null();
    }
    let pv = qjs::sofuu_js_get_property_str(ctx, *argv, c"__resp".as_ptr());
    let mut p: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut p, pv);
    qjs::sofuu_js_free_value(ctx, pv);
    if p != 0 {
        let r = p as *mut ResponseData;
        if !is_live_response(r) {
            return qjs::sofuu_js_null();
        }
        qjs::sofuu_js_new_int64(ctx, p)
    } else {
        qjs::sofuu_js_null()
    }
}

/// __fetch_stream_next(respPtr) → {value,done} or undefined if none ready.
unsafe extern "C" fn js_fetch_stream_next(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let mut p: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut p, *argv);
    let r = p as *mut ResponseData;
    if r.is_null() || !is_live_response(r) {
        return qjs::JS_ThrowTypeError(ctx, c"Invalid Response stream handle".as_ptr());
    }

    let chunk = response_next_chunk(ctx, r);
    if !qjs::is_undefined(chunk) {
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, obj, c"value".as_ptr(), chunk); /* consumes */
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 0));
        return obj;
    }
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: a transport error surfaces as a REJECTED next() — the
         * factory's Promise.resolve preserves the rejection, so for-await
         * consumers see the failure instead of a clean end-of-stream. */
        return rejected_body_promise(ctx, r);
    }
    if (*r).done != 0 {
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 1));
        qjs::sofuu_js_set_property_str(ctx, obj, c"value".as_ptr(), qjs::sofuu_js_undefined());
        return obj;
    }
    qjs::sofuu_js_undefined() /* no chunk yet — the consumer must wait */
}

/// __fetch_stream_wait(respPtr) → Promise<{value,done}>.
unsafe extern "C" fn js_fetch_stream_wait(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let mut p: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut p, *argv);
    let r = p as *mut ResponseData;

    /* Helper: resolved promise with one argument value (or none). */
    macro_rules! resolved {
        ($val:expr) => {{
            let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
            let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
            let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 1, &$val);
            qjs::sofuu_js_free_value(ctx, ret);
            qjs::sofuu_js_free_value(ctx, $val);
            qjs::sofuu_js_free_value(ctx, resolvers[0]);
            qjs::sofuu_js_free_value(ctx, resolvers[1]);
            promise
        }};
    }

    if r.is_null() || !is_live_response(r) {
        let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
        let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
        let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 0, ptr::null());
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, resolvers[0]);
        qjs::sofuu_js_free_value(ctx, resolvers[1]);
        return promise;
    }
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: the transfer failed — waiters reject, never see {done:true}. */
        return rejected_body_promise(ctx, r);
    }
    /* If a chunk arrived between next() and wait(), return it directly. */
    let chunk = response_next_chunk(ctx, r);
    if !qjs::is_undefined(chunk) {
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, obj, c"value".as_ptr(), chunk);
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 0));
        return resolved!(obj);
    }
    if (*r).done != 0 {
        let obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 1));
        return resolved!(obj);
    }
    /* No chunk, not done — create a pending promise to resolve on arrival.
     * Only ONE consumer may wait at a time (single pending_next slot); a
     * second waiter would silently overwrite the first and strand it. */
    if !(*r).pending_next.is_null() {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"Response stream already has a pending reader — wait for it before calling next() again".as_ptr(),
        );
    }
    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let pnext = Box::into_raw(Box::new(PromiseHandle {
        ctx,
        resolve: qjs::sofuu_js_dup_value(ctx, resolvers[0]),
        reject: qjs::sofuu_js_dup_value(ctx, resolvers[1]),
    }));
    (*r).pending_next = pnext;
    (*r).pending_resolve = qjs::sofuu_js_dup_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

/// Resolve a pending stream waiter with the next chunk (or done).
/// Called from the write callback and the completion path (loop thread).
unsafe fn response_fulfill_pending(ctx: *mut JSContext, r: *mut ResponseData) {
    if (*r).pending_next.is_null() {
        return;
    }
    let p = (*r).pending_next;
    let resolve = (*r).pending_resolve;
    (*r).pending_next = ptr::null_mut();
    (*r).pending_resolve = qjs::sofuu_js_undefined();

    if (*r).errored != 0 {
        /* P1-5: the transfer failed mid-body — reject the pending read
         * instead of handing it a clean {done:true} end-of-stream. r is
         * read only before new_fetch_error runs any JS (P1-4 rule). */
        let err = new_fetch_error(ctx, &resp_err_msg(r));
        let p_box = Box::from_raw(p);
        let ret = qjs::JS_Call(ctx, p_box.reject, qjs::sofuu_js_undefined(), 1, &err);
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, err);
        qjs::sofuu_js_free_value(ctx, resolve);
        qjs::sofuu_js_free_value(ctx, p_box.resolve);
        qjs::sofuu_js_free_value(ctx, p_box.reject);
        return;
    }

    let chunk = response_next_chunk(ctx, r);
    let obj = qjs::sofuu_js_new_object(ctx);
    if !qjs::is_undefined(chunk) {
        qjs::sofuu_js_set_property_str(ctx, obj, c"value".as_ptr(), chunk); /* consumes */
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 0));
    } else {
        qjs::sofuu_js_set_property_str(ctx, obj, c"done".as_ptr(), qjs::sofuu_js_new_bool(ctx, 1));
    }
    let ret = qjs::JS_Call(ctx, resolve, qjs::sofuu_js_undefined(), 1, &obj);
    qjs::sofuu_js_free_value(ctx, ret);
    qjs::sofuu_js_free_value(ctx, obj);
    qjs::sofuu_js_free_value(ctx, resolve);
    let p_box = Box::from_raw(p);
    qjs::sofuu_js_free_value(ctx, p_box.resolve);
    qjs::sofuu_js_free_value(ctx, p_box.reject);
}

/// Build a Response JS object from a completed fetch_req_t. With streaming
/// (take_body=0) the response is built on the first header line while the
/// body is still arriving; the response_data_t is stored back into
/// req->resp so the write callback and completion path enqueue/finalize on
/// the SAME struct the JS object holds.
unsafe fn build_response(
    ctx: *mut JSContext,
    req: &mut FetchReq,
    status: c_long,
    final_url: *const c_char,
    take_body: bool,
) -> JSValue {
    let data = Box::new(ResponseData {
        body: if take_body { std::mem::take(&mut req.body) } else { Vec::new() },
        headers_raw: std::mem::take(&mut req.headers_raw),
        url: CString::new(if final_url.is_null() { b"".to_vec() } else { CStr::from_ptr(final_url).to_bytes().to_vec() })
            .unwrap_or_default(),
        status_text: CString::new(match status {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "",
        })
        .unwrap_or_default(),
        chunks: VecDeque::new(),
        done: 0,
        errored: 0,
        err_msg: None,
        finalized: 0,
        pending_next: ptr::null_mut(),
        pending_resolve: qjs::sofuu_js_undefined(),
        pending_settle: Vec::new(),
        ctx,
    });
    let data_ptr = Box::into_raw(data);
    req.resp = data_ptr;
    /* Register for the stream check callback. */
    RESPONSES.with(|rs| rs.borrow_mut().push(data_ptr));

    /* Create the JS object */
    let obj = qjs::JS_NewObjectClass(ctx, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed) as c_int);
    qjs::JS_SetOpaque(obj, data_ptr as *mut c_void);

    /* Properties */
    qjs::sofuu_js_set_property_str(ctx, obj, c"status".as_ptr(), qjs::sofuu_js_new_int32(ctx, status as i32));
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"ok".as_ptr(),
        qjs::sofuu_js_new_bool(ctx, if (200..300).contains(&status) { 1 } else { 0 }),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"statusText".as_ptr(),
        qjs::sofuu_js_new_string(ctx, (*data_ptr).status_text.as_ptr()),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"url".as_ptr(),
        qjs::sofuu_js_new_string(ctx, (*data_ptr).url.as_ptr()),
    );

    /* Methods */
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"text".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, response_text, c"text".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"json".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, response_json, c"json".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"arrayBuffer".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, response_array_buffer, c"arrayBuffer".as_ptr(), 0),
    );
    /* Response.body — a real accessor returning the async iterator. */
    let body_getter = qjs::sofuu_js_new_cfunction(ctx, response_body_getter, c"body".as_ptr(), 0);
    let body_atom = qjs::JS_NewAtom(ctx, c"body".as_ptr());
    qjs::JS_DefineProperty(
        ctx,
        obj,
        body_atom,
        qjs::sofuu_js_undefined(),
        body_getter,
        qjs::sofuu_js_undefined(),
        qjs::JS_PROP_HAS_GET | qjs::JS_PROP_HAS_CONFIGURABLE,
    );
    qjs::JS_FreeAtom(ctx, body_atom);
    qjs::sofuu_js_free_value(ctx, body_getter);

    /* headers object with .get() method — a getter function that resolves
     * headers from the LIVE ResponseData each call, so headers arriving
     * after the first callback are always visible. */
    let headers_obj = qjs::sofuu_js_new_object(ctx);
    let hdr_data: [JSValue; 1] = [qjs::sofuu_js_new_int64(ctx, data_ptr as i64)];
    let get_fn = qjs::JS_NewCFunctionData(
        ctx,
        headers_get as qjs::JSCFunctionDataFn,
        1,
        0,
        1,
        hdr_data.as_ptr(),
    );
    /* NOTE: set_property_str STEALS the reference — do NOT free get_fn
     * afterwards (a double-free corrupts the heap and crashes later). */
    qjs::sofuu_js_set_property_str(ctx, headers_obj, c"get".as_ptr(), get_fn);
    qjs::sofuu_js_free_value(ctx, hdr_data[0]);
    qjs::sofuu_js_set_property_str(ctx, obj, c"headers".as_ptr(), headers_obj);

    obj
}

// ── curl-multi ↔ libuv bridge ────────────────────────────────────────

struct CurlContext {
    sockfd: c_int,
    poll_handle: *mut UvPoll,
}

unsafe extern "C" fn curl_close_cb(handle: *mut UvHandle) {
    // SAFETY: data set at creation; handle fully closed here.
    let ctx = *(handle as *mut *mut CurlContext);
    libc::free(handle as *mut c_void);
    drop(Box::from_raw(ctx));
}

/// Mirror C exactly: uv_close the poll handle; the close callback (which
/// reads handle->data) frees BOTH the poll storage and the Box'd context.
unsafe fn destroy_curl_context(context: *mut CurlContext) {
    uv::uv_close((*context).poll_handle as *mut UvHandle, Some(curl_close_cb));
}

unsafe fn create_curl_context(sockfd: c_int) -> *mut CurlContext {
    let context = Box::into_raw(Box::new(CurlContext {
        sockfd,
        poll_handle: ptr::null_mut(),
    }));
    let poll = libc::malloc(uv::sofuu_uv_poll_size()) as *mut UvPoll;
    (*context).poll_handle = poll;
    uv::uv_poll_init_socket(sofuu_loop_get(), poll, sockfd);
    *(poll as *mut *mut CurlContext) = context;
    context
}

// ── Response body + header write callbacks ───────────────────────────

unsafe extern "C" fn write_body_cb(
    ptr: *mut c_void,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let n = size * nmemb;
    let req = userdata as *mut FetchReq;
    if !valid_req(req) {
        return 0;
    }
    /* Keep the full body for text()/json()/arrayBuffer() — capped so a
     * misbehaving endpoint cannot exhaust memory. js-9: the caller may
     * lower the cap per request (maxBodyBytes); crossing it aborts the
     * transfer and the DONE error path names the cap. */
    let body_cap = if (*req).max_body > 0 { (*req).max_body } else { BODY_CAP };
    if (*req).body.len() + n > body_cap {
        (*req).cap_hit = 1;
        return 0; /* hard cap; aborts the transfer */
    }
    (*req).body.extend_from_slice(std::slice::from_raw_parts(ptr as *const u8, n));

    /* Streaming: enqueue the chunk ONLY — never run JS here (we are inside
     * curl_multi_socket_action; the uv_check_t fulfils waiters in a safe
     * loop context after this callback returns). */
    if !(*req).resp.is_null() {
        /* Chunks queue unconditionally once the response exists. A reader
         * gate ("queue only once .body is iterated") was tried and reverted:
         * bytes that arrive before the first read would be lost — on
         * localhost that is every byte (the transfer completes inside one
         * loop tick, before the consumer's first microtask). The duplicate
         * buffering this costs is bounded by STREAM_CAP and freed with the
         * response; correctness of late-attached consumers outranks it. */
        if (*req).resp_cap + n > STREAM_CAP {
            return 0; /* hard cap */
        }
        (*req).resp_cap += n;
        response_enqueue((*req).resp, std::slice::from_raw_parts(ptr as *const u8, n));
        fetch_stream_pump();
    }
    n
}

unsafe extern "C" fn write_header_cb(
    ptr: *mut c_void,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let n = size * nmemb;
    let req = userdata as *mut FetchReq;
    if !valid_req(req) {
        return 0;
    }
    let block = std::slice::from_raw_parts(ptr as *const u8, n);
    let is_header_line = !block.starts_with(b"HTTP/")
        && !block.starts_with(b":")
        && block.iter().any(|&b| b == b':');

    if is_header_line {
        /* net-4 hardening: cap the accumulated header block so a hostile
         * endpoint cannot balloon memory with endless headers. Returning 0
         * aborts the transfer (surfaces as a write error). */
        let dst_len = if (*req).resp.is_null() {
            (*req).headers_raw.len()
        } else {
            (*(*req).resp).headers_raw.len()
        };
        if dst_len.saturating_add(n) > HEADER_CAP {
            return 0;
        }
        if (*req).resp.is_null() {
            /* Response not built yet — accumulate in the req. */
            (*req).headers_raw.extend_from_slice(block);
        } else {
            /* Response exists — append straight into its live buffer so
             * headers.get() (which reads the ResponseData) sees them. */
            (*(*req).resp).headers_raw.extend_from_slice(block);
        }
    }

    /* The first NON-status callback = the response headers: build the
     * Response now (status known) and resolve the fetch promise, so
     * consumers can start reading res.body while the transfer is still in
     * flight. build_response moves the accumulated headers into the
     * ResponseData; later header lines append directly to it.
     *
     * P1-3: NOT on a 3xx — the fetch promise must resolve only on the FINAL
     * response (r.status must never be the first hop's 3xx). Redirect-hop
     * headers keep accumulating in req.headers_raw so the completion path
     * can read Location and re-issue the transfer. */
    if (*req).resp.is_null() && is_header_line {
        let mut status: c_long = 0;
        curl::curl_easy_getinfo((*req).easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
        if !(300..400).contains(&status) {
            let mut final_url: *mut c_char = ptr::null_mut();
            curl::curl_easy_getinfo((*req).easy, curl::CURLINFO_EFFECTIVE_URL, &mut final_url);

            let ctx = (*req).ctx;
            let mut req_ref = &mut *req;
            let response = build_response(ctx, &mut req_ref, status, final_url, false);
            (*(*req).resp).done = 0;
            sofuu_promise_resolve((*req).promise, response);
            qjs::sofuu_js_free_value(ctx, response);
            /* NOTE: no sofuu_flush_jobs here — we are inside
             * curl_multi_socket_action; running JS microtasks would re-enter
             * curl (CURLM_RECURSIVE_API_CALL). The loop pumps after uv_run. */
        }
    }
    n
}

/// Enqueue a chunk (called from the curl write callback, loop thread).
unsafe fn response_enqueue(r: *mut ResponseData, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    // P3 (AUDIT-2026-09-07): the old ResponseChunk wrapper heap-allocated a
    // Box per chunk just to carry a dead `len` field (suppressed with
    // `let _ = c.len;`) before the Vec was moved out — plain push_back.
    (*r).chunks.push_back(data.to_vec());
}

/// Kick the check handle so pending waiters are fulfilled after the current
/// uv_run iteration (safe JS context). Safe to call from curl callbacks.
unsafe fn fetch_stream_pump() {
    let init = STREAM_CHECK_INIT.with(|i| i.get());
    if init == 0 {
        let check = libc::malloc(uv::sofuu_uv_check_size()) as *mut UvCheck;
        STREAM_CHECK.with(|s| s.set(check));
        uv::uv_check_init(sofuu_loop_get(), check);
        STREAM_CHECK_INIT.with(|i| i.set(1));
    }
    let check = STREAM_CHECK.with(|s| s.get());
    uv::uv_check_start(check, Some(stream_check_cb));
}

unsafe extern "C" fn stream_check_cb(_h: *mut UvCheck) {
    let check = STREAM_CHECK.with(|s| s.get());
    uv::uv_check_stop(check);
    /* One pass: fulfill any response with queued chunks or done.
     * P1-4: the worklist is snapshotted under the borrow and the borrow is
     * DROPPED before any JS runs — response_fulfill_pending's
     * JS_CallConstructor allocates and can collect a dead Response, whose
     * finalizer borrow_mut()s this same cell (BorrowMutError abort).
     * Candidates are re-validated with is_live_response before their JS
     * runs; after its constructor call response_fulfill_pending never
     * dereferences r again, so a response collected mid-fulfillment is
     * harmless. Per-response order (fulfill, then settle) is preserved. */
    let mut work: Vec<(
        *mut JSContext,
        *mut ResponseData,
        Vec<*mut PromiseHandle>,
        Option<String>,
    )> = Vec::new();
    RESPONSES.with(|rs| {
        let rs = rs.borrow();
        for &r in rs.iter() {
            let mut settle_err: Option<String> = None;
            let mut settles: Vec<*mut PromiseHandle> = Vec::new();
            if (*r).done != 0 {
                settles = std::mem::take(&mut (*r).pending_settle);
                /* P1-5: snapshot the error message under the borrow — the
                 * settle path below never dereferences r again (P1-4 rule). */
                if (*r).errored != 0 {
                    settle_err = Some(resp_err_msg(r));
                }
            }
            let needs_fulfill =
                !(*r).pending_next.is_null() && (!(*r).chunks.is_empty() || (*r).done != 0);
            if !settles.is_empty() || needs_fulfill {
                work.push(((*r).ctx, r, settles, settle_err));
            }
        }
    });
    for (ctx, r, settles, settle_err) in work {
        if is_live_response(r) {
            response_fulfill_pending(ctx, r);
        }
        for settle in settles {
            match &settle_err {
                /* P1-5: an errored transfer rejects its text()/json()/
                 * arrayBuffer() waiters instead of resolving them. reject
                 * consumes the error (promise.rs memory contract). */
                Some(msg) => sofuu_promise_reject(settle, new_fetch_error(ctx, msg)),
                None => sofuu_promise_resolve(settle, qjs::sofuu_js_undefined()),
            }
        }
    }
}

/// Return a promise that resolves (undefined) once the body transfer is
/// complete — text()/json()/arrayBuffer() await this before reading.
unsafe fn response_wait_done(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    if (*r).done != 0 && (*r).errored != 0 {
        /* P1-5: an errored transfer's waiters reject — resolving here made
         * the chained text()/json()/arrayBuffer() read a truncated body. */
        return rejected_body_promise(ctx, r);
    }
    if (*r).done != 0 {
        /* Done AND clean (the errored case returned above): pre-resolve so
         * the chained text() reads the completed body synchronously. */
        let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
        let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
        let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 0, ptr::null());
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, resolvers[0]);
        qjs::sofuu_js_free_value(ctx, resolvers[1]);
        return promise;
    }
    /* net-8: NOT done — always arm. The old `|| !pending_settle.is_null()`
     * pre-resolved every waiter after the first, so the second mid-stream
     * text()/json()/arrayBuffer() read a partial body. A Vec holds any
     * number of concurrent waiters; the DONE pump settles them all. */
    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let p = Box::into_raw(Box::new(PromiseHandle {
        ctx,
        resolve: qjs::sofuu_js_dup_value(ctx, resolvers[0]),
        reject: qjs::sofuu_js_dup_value(ctx, resolvers[1]),
    }));
    (*r).pending_settle.push(p);
    qjs::sofuu_js_free_value(ctx, resolvers[0]);
    qjs::sofuu_js_free_value(ctx, resolvers[1]);
    promise
}

// ── Multi-handle completion ──────────────────────────────────────────

unsafe fn check_multi_info_global() {
    let mut pending: c_int = 0;
    loop {
        let message = curl::curl_multi_info_read(curl_multi(), &mut pending);
        if message.is_null() {
            break;
        }
        let msg = *message;
        if msg.msg != curl::CURLMSG_DONE {
            continue;
        }

        let easy = msg.easy_handle;
        let return_code = msg.result;

        let mut req_ptr: *mut c_void = ptr::null_mut();
        curl::curl_easy_getinfo(easy, curl::CURLINFO_PRIVATE, &mut req_ptr);
        let req = req_ptr as *mut FetchReq;

        /* A redirect hop re-arms the SAME easy handle — the transfer must
         * stay registered with the multi and req must stay alive. */
        let mut keep_transfer = false;

        if !req.is_null() {
            let ctx = (*req).ctx;

            if return_code == curl::CURLE_OK {
                let mut status: c_long = 0;
                let mut final_url: *mut c_char = ptr::null_mut();
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
                curl::curl_easy_getinfo(easy, curl::CURLINFO_EFFECTIVE_URL, &mut final_url);

                /* P1-3 + net-4: FOLLOWLOCATION is off, so a 3xx here is a
                 * HOP, not the response — re-issue the next request and keep
                 * the promise pending. It resolves only on the final (non-
                 * 3xx) response, so r.status is never the first hop's 3xx. */
                /* Settled in an arm below → the fall-through finalize must
                 * NOT touch the promise again: reject frees the handle
                 * (promise.rs memory contract), so a second settle would be
                 * a use-after-free. */
                let mut rejected = false;
                if (*req).resp.is_null() && (300..400).contains(&status) {
                    let cur_url = if final_url.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(final_url).to_string_lossy().into_owned()
                    };
                    match extract_location(&(*req).headers_raw) {
                        Some(loc) if (*req).hops >= MAX_REDIRECTS => {
                            sofuu_promise_reject(
                                (*req).promise,
                                new_fetch_error(ctx, "fetch: too many redirects"),
                            );
                            rejected = true;
                        }
                        Some(loc) => match resolve_location(&cur_url, &loc) {
                            Some(next) => match redirect_reissue(req, status, &cur_url, &next) {
                                Ok(()) => keep_transfer = true,
                                Err(reason) => {
                                    sofuu_promise_reject((*req).promise, new_fetch_error(ctx, reason));
                                    rejected = true;
                                }
                            },
                            None => {
                                sofuu_promise_reject(
                                    (*req).promise,
                                    new_fetch_error(ctx, "fetch: invalid redirect Location"),
                                );
                                rejected = true;
                            }
                        },
                        /* No Location (304 and friends) — this IS the final
                         * response: fall through with the accumulated
                         * headers/body. */
                        None => {}
                    }
                }

                if !keep_transfer && !rejected {
                    if !(*req).resp.is_null() {
                        /* Streaming: the response already resolved on the first
                         * header line; transfer the buffered body + mark done.
                         * Pending waiters are fulfilled by the uv_check_t. */
                        let resp = (*req).resp;
                        (*resp).body = std::mem::take(&mut (*req).body);
                        (*resp).done = 1;
                        if (*resp).finalized != 0 {
                            /* The JS object was collected mid-transfer; the pump
                             * will settle its waiters, and the response is
                             * reclaimed at the next fetch — after which every
                             * reader microtask has run and nothing references it
                             * (freeing now would hit the async parse). */
                            reclaim_orphan_resp();
                            ORPHAN_RESP.with(|o| o.set(resp));
                            (*req).resp = ptr::null_mut();
                        }
                        fetch_stream_pump();
                    } else {
                        /* No header block (e.g. empty body) — build + resolve. */
                        let mut req_ref = &mut *req;
                        let response = build_response(ctx, &mut req_ref, status, final_url, true);
                        sofuu_promise_resolve((*req).promise, response);
                        qjs::sofuu_js_free_value(ctx, response);
                    }
                }
            } else {
                /* net-4: when opensocket refused a link-local/metadata
                 * target, say so — the generic connect error hides the why. */
                let msg_c = if (*req).refused != 0 {
                    CString::new("fetch blocked: link-local/metadata address refused").unwrap()
                } else if (*req).cap_hit != 0 {
                    /* js-9: write_body_cb aborted this transfer on the
                     * caller's maxBodyBytes — name the cap instead of
                     * surfacing curl's generic write error. */
                    CString::new(format!(
                        "fetch: response body exceeds maxBodyBytes ({} bytes cap)",
                        (*req).max_body
                    ))
                    .unwrap_or_default()
                } else {
                    CString::new(CStr::from_ptr(curl::curl_easy_strerror(return_code)).to_bytes())
                        .unwrap_or_default()
                };
                let err = qjs::JS_NewError(ctx);
                qjs::sofuu_js_set_property_str(ctx, err, c"message".as_ptr(), qjs::sofuu_js_new_string(ctx, msg_c.as_ptr()));
                if !(*req).resp.is_null() {
                    /* Transfer error: end the stream with done. */
                    qjs::sofuu_js_free_value(ctx, err); /* not consumed below */
                    let resp = (*req).resp;
                    (*resp).done = 1;
                    (*resp).errored = 1;
                    /* P1-5: capture curl's error text — every consume surface
                     * rejects with this instead of ending cleanly. */
                    (*resp).err_msg = Some(msg_c.to_string_lossy().into_owned());
                    if (*resp).finalized != 0 {
                        /* P1-2: same park as the DONE path — the JS object was
                         * collected mid-transfer, so the error path must not
                         * drop this ~body-sized buffer unreclaimable. The pump
                         * settles its waiters; the orphan is reclaimed at the
                         * next fetch, after every reader microtask has run. */
                        reclaim_orphan_resp();
                        ORPHAN_RESP.with(|o| o.set(resp));
                        (*req).resp = ptr::null_mut();
                    }
                    fetch_stream_pump();
                } else {
                    sofuu_promise_reject((*req).promise, err);
                }
            }

            if !keep_transfer {
                if !(*req).req_headers.is_null() {
                    curl::curl_slist_free_all((*req).req_headers);
                }
                drop(Box::from_raw(req));

                sofuu_flush_jobs(ctx);
            }
        }

        if !keep_transfer {
            curl::curl_multi_remove_handle(curl_multi(), easy);
            curl::curl_easy_cleanup(easy);
        }
    }
}

unsafe extern "C" fn curl_perform_global(req: *mut UvPoll, _status: c_int, events: c_int) {
    // SAFETY: the poll handle's data points at its CurlContext (set at
    // creation) — exactly C's `req->data` contract.
    let context = *(req as *mut *mut CurlContext);
    let sockfd = (*context).sockfd;
    let mut flags: c_int = 0;
    if events & uv::UV_READABLE != 0 {
        flags |= curl::CURL_CSELECT_IN;
    }
    if events & uv::UV_WRITABLE != 0 {
        flags |= curl::CURL_CSELECT_OUT;
    }
    let mut running_handles: c_int = 0;
    curl::curl_multi_socket_action(curl_multi(), sockfd, flags, &mut running_handles);
    check_multi_info_global();
}

unsafe extern "C" fn on_timeout(_t: *mut UvTimer) {
    let mut running_handles: c_int = 0;
    curl::curl_multi_socket_action(curl_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running_handles);
    check_multi_info_global();
}

unsafe extern "C" fn handle_socket(
    _easy: *mut Curl,
    s: c_int,
    action: c_int,
    _userp: *mut c_void,
    socketp: *mut c_void,
) -> c_int {
    if action == curl::CURL_POLL_IN || action == curl::CURL_POLL_OUT || action == curl::CURL_POLL_INOUT {
        let curl_context: *mut CurlContext = if !socketp.is_null() {
            socketp as *mut CurlContext
        } else {
            let ctx = create_curl_context(s);
            curl::curl_multi_assign(curl_multi(), s, ctx as *mut c_void);
            ctx
        };
        let mut events: c_int = 0;
        if action != curl::CURL_POLL_IN {
            events |= uv::UV_WRITABLE;
        }
        if action != curl::CURL_POLL_OUT {
            events |= uv::UV_READABLE;
        }
        uv::uv_poll_start((*curl_context).poll_handle, events, Some(curl_perform_global));
    } else {
        if !socketp.is_null() {
            let ctx = socketp as *mut CurlContext;
            uv::uv_poll_stop((*ctx).poll_handle);
            destroy_curl_context(ctx); /* ctx freed by curl_close_cb */
            curl::curl_multi_assign(curl_multi(), s, ptr::null_mut());
        }
    }
    0
}

unsafe extern "C" fn start_timeout(_multi: *mut CurlM, timeout_ms: c_long, _userp: *mut c_void) -> c_int {
    let timer = TIMEOUT_TIMER.with(|t| t.get());
    if timeout_ms < 0 {
        uv::uv_timer_stop(timer);
    } else {
        let ms = if timeout_ms == 0 { 1 } else { timeout_ms as u64 };
        uv::uv_timer_start(timer, Some(on_timeout), ms, 0);
    }
    0
}

// ── SSRF guards + manual redirect machinery (net-4 / net-5 / net-6 / P1-3) ──

/// Entry-level URL guard: the cloud-metadata IP is never a legitimate fetch
/// target. Some(reason) = reject the fetch with that message.
fn ssrf_url_reject(url: &str) -> Option<&'static str> {
    if url.to_ascii_lowercase().contains("169.254.169.254") {
        Some("fetch blocked: cloud metadata IP")
    } else {
        None
    }
}

/// fetch only speaks http(s) (web.js's own guard mirrors this).
fn scheme_ok(url: &str) -> bool {
    let low = url.to_ascii_lowercase();
    low.starts_with("http://") || low.starts_with("https://")
}

/// (scheme, authority, path+query without fragment) for an http(s) URL.
/// None when the URL has no scheme, a non-http scheme, or an empty
/// authority (e.g. `http:///x`).
fn url_parts(url: &str) -> Option<(&str, &str, &str)> {
    let (s, rest) = url.split_once("://")?;
    let sl = s.to_ascii_lowercase();
    if sl != "http" && sl != "https" {
        return None;
    }
    let no_frag = rest.split('#').next().unwrap_or(rest);
    match no_frag.find('/') {
        Some(i) => Some((s, &no_frag[..i], &no_frag[i..])),
        None => Some((s, no_frag, "/")),
    }
}

/// scheme://authority (lowercased, scheme included so an https→http
/// downgrade also counts as an origin change). None when the URL does not
/// parse to a non-empty authority. Over-matching (case differences) errs
/// safe: credentials get stripped one hop early.
fn origin_of(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let low = scheme.to_ascii_lowercase();
    if low != "http" && low != "https" {
        return None;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() {
        return None;
    }
    Some(format!("{}://{}", low, authority.to_ascii_lowercase()))
}

/// RFC 3986 §5.2.4 remove_dot_segments over a URL path.
fn remove_dot_segments(path: &str) -> String {
    let mut segs: Vec<&str> = Vec::new();
    let mut trailing = false;
    for seg in path.split('/').skip(1) {
        match seg {
            "" | "." => trailing = true,
            ".." => {
                segs.pop();
                trailing = true;
            }
            s => {
                segs.push(s);
                trailing = false;
            }
        }
    }
    let mut out = String::new();
    for s in &segs {
        out.push('/');
        out.push_str(s);
    }
    if trailing && !out.ends_with('/') {
        out.push('/');
    }
    if out.is_empty() {
        out.push('/');
    }
    out
}

/// Resolve a Location value against the current URL (RFC 3986 §5, relative
/// reference rules). Absolute non-http(s) targets → None (net-4: no ftp://
/// or file:// redirect targets). Dot segments are normalized in the result.
fn resolve_location(cur: &str, loc_raw: &str) -> Option<String> {
    let loc = loc_raw.trim();
    let loc = loc.split('#').next().unwrap_or("");
    if loc.is_empty() {
        return None;
    }
    /* Absolute URI? ':' before any '/'. Non-http(s) absolute targets are
     * refused — anything with a scheme that is not http(s) never resolves. */
    let is_absolute = matches!(
        loc.char_indices().find(|&(_, c)| c == ':' || c == '/'),
        Some((_, ':'))
    );
    if is_absolute && !(loc.starts_with("http://") || loc.starts_with("https://")) {
        return None;
    }
    let (scheme, auth, cur_path) = url_parts(cur)?;
    let path_only = cur_path.split('?').next().unwrap_or("/");
    let merged: String = if is_absolute {
        loc.to_string()
    } else if let Some(rest) = loc.strip_prefix("//") {
        format!("{}://{}", scheme, rest)
    } else if loc.starts_with('/') {
        format!("{}://{}{}", scheme, auth, loc)
    } else if loc.starts_with('?') {
        format!("{}://{}{}{}", scheme, auth, path_only, loc)
    } else {
        let dir = match path_only.rfind('/') {
            Some(i) => &path_only[..=i],
            None => "/",
        };
        format!("{}://{}{}{}", scheme, auth, dir, loc)
    };
    /* Normalize dot segments in the final path (and only the path). */
    let (s2, a2, p2) = url_parts(&merged)?;
    if a2.is_empty() {
        return None;
    }
    let (path, query) = match p2.find('?') {
        Some(i) => (&p2[..i], &p2[i..]),
        None => (p2, ""),
    };
    Some(format!("{}://{}{}{}", s2, a2, remove_dot_segments(path), query))
}

/// Pull the Location header value out of a raw header block
/// (case-insensitive name match, first match wins, CR/LF trimmed).
fn extract_location(headers: &[u8]) -> Option<String> {
    for line in headers.split(|&b| b == b'\n') {
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if line.len() >= 9 && line[..9].eq_ignore_ascii_case(b"location:") {
            return Some(String::from_utf8_lossy(&line[9..]).trim().to_string());
        }
    }
    None
}

/// True when the address libcurl is about to connect to is link-local — the
/// metadata class: 169.254/16, IPv4-mapped ::ffff:169.254/16, and
/// fe80::/10. Wire-level guard: fires for EVERY connection (initial or
/// redirect hop) on the resolved sockaddr, so alternate IPv4 encodings and
/// DNS names that resolve into the range are covered with no URL parsing.
/// sockaddr layout (probe-verified): sockaddr_in.sin_addr at addr+4,
/// sockaddr_in6.sin6_addr at addr+8, both network byte order.
unsafe fn sockaddr_is_linklocal(sa: *const curl::CurlSockaddr) -> bool {
    let sa = &*sa;
    match sa.family {
        libc::AF_INET => sa.addrlen >= 6 && sa.addr[4] == 169 && sa.addr[5] == 254,
        libc::AF_INET6 => {
            /* fe80::/10 */
            if sa.addrlen >= 10 && sa.addr[8] == 0xfe && (sa.addr[9] & 0xc0) == 0x80 {
                return true;
            }
            /* IPv4-mapped ::ffff:a.b.c.d */
            sa.addrlen >= 24
                && sa.addr[8..18].iter().all(|&b| b == 0)
                && sa.addr[18] == 0xff
                && sa.addr[19] == 0xff
                && sa.addr[20] == 169
                && sa.addr[21] == 254
        }
        _ => false,
    }
}

/// CURLOPT_OPENSOCKETFUNCTION — replaces socket(2) for every connection
/// libcurl makes (libcurl sets the fd non-blocking itself). Refusing
/// link-local targets: returning CURL_SOCKET_BAD aborts the connection with
/// CURLE_COULDNT_CONNECT, the documented IP-block-listing mechanism.
unsafe extern "C" fn opensocket_cb(
    clientp: *mut c_void,
    _purpose: c_int,
    address: *mut curl::CurlSockaddr,
) -> c_int {
    if address.is_null() {
        return curl::CURL_SOCKET_BAD;
    }
    if sockaddr_is_linklocal(address) {
        let req = clientp as *mut FetchReq;
        if !req.is_null() && (*req).magic == REQ_MAGIC {
            (*req).refused = 1;
        }
        return curl::CURL_SOCKET_BAD;
    }
    libc::socket((*address).family, (*address).socktype, (*address).protocol)
}

fn new_fetch_error(ctx: *mut JSContext, msg: &str) -> JSValue {
    unsafe {
        let err = qjs::JS_NewError(ctx);
        let mc = CString::new(msg).unwrap_or_default();
        qjs::sofuu_js_set_property_str(
            ctx,
            err,
            c"message".as_ptr(),
            qjs::sofuu_js_new_string(ctx, mc.as_ptr()),
        );
        err
    }
}

/// CURLOPT_HTTPGET = CURLOPTTYPE_LONG(0) + 80 (SDK curl.h). Setting it
/// resets the method to GET and clears POST data — used by the redirect
/// method-conversion rules.
const CURLOPT_HTTPGET: c_int = 80;

/// Re-issue the transfer at the next redirect hop. On Err(reason) the
/// caller rejects the fetch promise with that message.
unsafe fn redirect_reissue(
    req: *mut FetchReq,
    status: c_long,
    cur_url: &str,
    next_url: &str,
) -> Result<(), &'static str> {
    let easy = (*req).easy;
    if !scheme_ok(next_url) {
        return Err("fetch blocked: redirect target must be http(s)");
    }
    if let Some(reason) = ssrf_url_reject(next_url) {
        return Err(reason);
    }
    /* Method conversion (fetch spec): 303 → GET unless HEAD; 301/302 → GET
     * when the original was POST. 307/308 keep the method and body. */
    let m = (*req)
        .method
        .as_deref()
        .unwrap_or("GET")
        .to_ascii_uppercase();
    let to_get = match status {
        303 => m != "HEAD",
        301 | 302 => m == "POST",
        _ => false,
    };
    if to_get {
        (*req).method = None;
        curl::curl_easy_setopt(easy, curl::CURLOPT_CUSTOMREQUEST, ptr::null::<c_char>());
        curl::curl_easy_setopt(easy, CURLOPT_HTTPGET, 1 as c_long);
    }
    /* net-5: never forward Authorization/Cookie to a different origin. The
     * slist is opaque in the FFI, so rebuild it from the parsed headers —
     * and set HTTPHEADER unconditionally (even a NULL pointer must replace
     * the stale one, which points at a list that was just freed). */
    if origin_of(cur_url) != origin_of(next_url) {
        if !(*req).req_headers.is_null() {
            curl::curl_slist_free_all((*req).req_headers);
            (*req).req_headers = ptr::null_mut();
        }
        for (k, v) in &(*req).headers_vec {
            if k.eq_ignore_ascii_case("authorization") || k.eq_ignore_ascii_case("cookie") {
                continue;
            }
            let hc = CString::new(format!("{}: {}", k, v)).unwrap_or_default();
            /* P3 (AUDIT-2026-09-07): append returns NULL on OOM and the old
             * code assigned it straight over the live list — the rebuilt
             * header set was lost (and the partial list leaked), so the
             * redirect hop went out headerless. Free the partial list and
             * abort the hop instead. */
            let prev = (*req).req_headers;
            (*req).req_headers = curl::curl_slist_append((*req).req_headers, hc.as_ptr());
            if (*req).req_headers.is_null() {
                if !prev.is_null() {
                    curl::curl_slist_free_all(prev);
                }
                return Err("fetch: out of memory building redirect headers");
            }
        }
        curl::curl_easy_setopt(easy, curl::CURLOPT_HTTPHEADER, (*req).req_headers);
    }
    /* Fresh per-hop state: the new response must not inherit this hop's
     * accumulated headers/body (nor a cap hit — the cap aborts the
     * transfer, so this only guards a future reissue path). */
    (*req).body.clear();
    (*req).headers_raw.clear();
    (*req).resp_cap = 0;
    (*req).cap_hit = 0;
    (*req).hops += 1;

    let next_c = match CString::new(next_url) {
        Ok(c) => c,
        Err(_) => return Err("fetch: invalid redirect Location"),
    };
    curl::curl_multi_remove_handle(curl_multi(), easy);
    curl::curl_easy_setopt(easy, curl::CURLOPT_URL, next_c.as_ptr());
    /* net-12: an unchecked add failure left the transfer outside the multi
     * with nothing driving it — the promise pends forever and the FetchReq
     * leaks. On Err the caller rejects the promise and tears the request
     * down (slist + Box + easy, check_multi_info_global). */
    if curl::curl_multi_add_handle(curl_multi(), easy) != curl::CURLM_OK {
        return Err("fetch: failed to re-issue redirect transfer");
    }
    /* Re-adding does not restart socket activity by itself — kick the multi
     * so libcurl re-arms its timer/socket set (same reason the initial add
     * in js_sofuu_fetch kicks). Runs from a uv callback, never inside a
     * libcurl callback, so no CURLM_RECURSIVE_API_CALL. */
    let mut running: c_int = 0;
    curl::curl_multi_socket_action(curl_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running);
    Ok(())
}

// ── JS: fetch(url, options?) → Promise<Response> ─────────────────────

unsafe extern "C" fn js_sofuu_fetch(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"fetch requires a URL".as_ptr());
    }

    /* P2-33 (AUDIT-2026-09-01): a streamed Response whose JS object was
     * GC'd mid-transfer parks itself in ORPHAN_RESP at DONE, with the
     * design comment promising "reclaiming at the NEXT fetch is the only
     * safe free point" — but this entry never called the reclaim, so a
     * process that orphans exactly one response leaked its body/headers
     * forever. Safe here: between fetches no curl callback can run and
     * every reader microtask of the previous fetch has already drained. */
    reclaim_orphan_resp();

    let url_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if url_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    // SSRF guard: block cloud metadata IP outright (169.254.169.254 is
    // never a legitimate fetch target; localhost stays allowed for local
    // dev/servers/tests). Full private-IP blocking would break
    // fetch('http://127.0.0.1:…') local servers, so only the metadata
    // endpoint is denied here — the wire-level opensocket guard plus the
    // manual per-hop redirect re-issue below handle the rest (net-4).
    {
        let url_str = CStr::from_ptr(url_ptr).to_string_lossy();
        if ssrf_url_reject(&url_str).is_some() {
            qjs::sofuu_js_free_cstring(ctx, url_ptr);
            return qjs::JS_ThrowTypeError(ctx, c"fetch blocked: cloud metadata IP".as_ptr());
        }
        if !scheme_ok(&url_str) {
            qjs::sofuu_js_free_cstring(ctx, url_ptr);
            return qjs::JS_ThrowTypeError(ctx, c"fetch blocked: http(s) only".as_ptr());
        }
    }
    /* curl only delivers response headers to CURLOPT_HEADERFUNCTION when
     * CURLOPT_HEADER is set (curl >= 7.85 splits HTTP/2 pseudo-headers). */
    /* CURLOPT_HTTP_VERSION = 2 → CURL_HTTP_VERSION_1_1 (curl.h). */
    const CURLOPT_HTTP_VERSION: c_int = 84;
    const CURL_HTTP_VERSION_1_1: c_long = 2;
    const CURLOPT_ACCEPT_ENCODING: c_int = 10102;

    let mut req = Box::new(FetchReq {
        magic: REQ_MAGIC,
        ctx,
        promise: ptr::null_mut(),
        body: Vec::new(),
        headers_raw: Vec::new(),
        resp: ptr::null_mut(),
        resp_cap: 0,
        easy: curl::curl_easy_init(),
        req_headers: ptr::null_mut(),
        hops: 0,
        refused: 0,
        headers_vec: Vec::new(),
        method: None,
        max_body: 0,
        cap_hit: 0,
    });
    if req.easy.is_null() {
        // curl_easy_init OOM — fail loudly instead of NULL-deref below.
        return qjs::JS_ThrowTypeError(ctx, c"fetch: out of memory (curl init failed)".as_ptr());
    }

    curl::curl_easy_setopt(req.easy, curl::CURLOPT_URL, url_ptr);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_WRITEFUNCTION, write_body_cb as *const c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut FetchReq as *mut c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_HEADERFUNCTION, write_header_cb as *const c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_HEADERDATA, &mut *req as *mut FetchReq as *mut c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_PRIVATE, &mut *req as *mut FetchReq as *mut c_void);
    /* net-4: redirects are re-issued manually by check_multi_info_global
     * (per-hop scheme/SSRF guards, credential stripping on origin change,
     * promise resolves only on the final response) — FOLLOWLOCATION must
     * stay OFF and libcurl's own redirect knobs are not set. */
    /* P3 (AUDIT-2026-09-07): the old CURLOPT_TIMEOUT_MS=120s default was a
     * TOTAL-TRANSFER cap — it killed legitimate long streams (SSE, big
     * downloads) mid-flight even while actively transferring. Stall
     * detection instead: 30s to connect, then abort only if the transfer
     * crawls below 1 byte/s for 60s. An explicit timeoutMs option still
     * installs a total cap (js-8, below). */
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_CONNECTTIMEOUT_MS as c_int, 30_000 as c_long);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_LOW_SPEED_LIMIT as c_int, 1 as c_long);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_LOW_SPEED_TIME as c_int, 60 as c_long);
    /* NOSIGNAL: never let libcurl raise SIGPIPE/SIGALRM out of DNS
     * timeouts — the process drives curl from multiple threads. */
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_NOSIGNAL as c_int, 1 as c_long);
    /* net-4 (wire level): every connection — initial or redirect hop — goes
     * through opensocket_cb, which refuses link-local/metadata targets
     * (169.254/16, fe80::/10, IPv4-mapped) on the RESOLVED sockaddr, so
     * alternate IPv4 encodings and DNS names are covered too. */
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_OPENSOCKETFUNCTION, opensocket_cb as *const c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_OPENSOCKETDATA, &mut *req as *mut FetchReq as *mut c_void);
    /* NOTE: CURLOPT_HEADER stays OFF — with it on, curl ALSO feeds response
     * headers to the body callback and text()/json() get "HTTP/1.1 200 OK"
     * prepended. The header callback captures real headers separately. */
    curl::curl_easy_setopt(req.easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1); /* avoid HTTP/2 pseudo-header splitting */
    curl::curl_easy_setopt(req.easy, CURLOPT_ACCEPT_ENCODING, ptr::null::<c_char>()); /* enable compressed transfers */

    qjs::sofuu_js_free_cstring(ctx, url_ptr);

    /* Parse options: { method, headers, body } */
    if argc > 1 && qjs::is_object(*argv.add(1)) {
        /* method — net-6: must be a valid RFC 7230 token; a CR/LF (or any
         * other non-token byte) would inject a header into the request
         * line via CURLOPT_CUSTOMREQUEST. Only `easy` exists at this
         * point, so the error path is just its cleanup. */
        let method_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"method".as_ptr());
        if !qjs::is_undefined(method_val) {
            let method = qjs::sofuu_js_to_cstring(ctx, method_val);
            if !method.is_null() {
                let mb = CStr::from_ptr(method).to_bytes();
                let is_token = !mb.is_empty()
                    && mb.iter().all(|&c| {
                        matches!(c,
                            b'!' | b'#' | b'$' | b'%' | b'&' | b'\'' | b'*' | b'+' | b'-' | b'.'
                            | b'^' | b'_' | b'`' | b'|' | b'~'
                            | b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z')
                    });
                if !is_token {
                    qjs::sofuu_js_free_cstring(ctx, method);
                    qjs::sofuu_js_free_value(ctx, method_val);
                    curl::curl_easy_cleanup(req.easy);
                    return qjs::JS_ThrowTypeError(ctx, c"fetch: invalid HTTP method".as_ptr());
                }
                (*req).method = Some(String::from_utf8_lossy(mb).into_owned());
                curl::curl_easy_setopt(req.easy, curl::CURLOPT_CUSTOMREQUEST, method);
                qjs::sofuu_js_free_cstring(ctx, method);
            }
        }
        qjs::sofuu_js_free_value(ctx, method_val);

        /* js-8 (AUDIT-2026-09-07): per-request deadline. The JS layer races
         * a timeout promise against the fetch, but the LOSING transfer used
         * to keep running; handing curl the caller's deadline via
         * CURLOPT_TIMEOUT_MS kills the transfer when the race rejects.
         * Clamped to 1..600000 ms; the same easy handle is re-armed across
         * redirect hops (redirect_reissue), so the deadline carries over.
         * Without the option only the stall detector above applies (P3:
         * no more blanket 120s total-transfer cap). */
        let timeout_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"timeoutMs".as_ptr());
        if qjs::is_number(timeout_val) {
            let mut tms: f64 = 0.0;
            qjs::JS_ToFloat64(ctx, &mut tms, timeout_val);
            if tms > 0.0 && tms <= 600_000.0 {
                curl::curl_easy_setopt(req.easy, curl::CURLOPT_TIMEOUT_MS as c_int, tms as c_long);
            }
        }
        qjs::sofuu_js_free_value(ctx, timeout_val);

        /* js-9: per-request body cap (0/absent = the BODY_CAP default). */
        let maxbody_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"maxBodyBytes".as_ptr());
        if qjs::is_number(maxbody_val) {
            let mut mb: f64 = 0.0;
            qjs::JS_ToFloat64(ctx, &mut mb, maxbody_val);
            if mb > 0.0 {
                (*req).max_body = if (mb as usize) > BODY_CAP { BODY_CAP } else { mb as usize };
            }
        }
        qjs::sofuu_js_free_value(ctx, maxbody_val);

        /* headers */
        let mut slist_failed = false;
        let headers_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"headers".as_ptr());
        if qjs::is_object(headers_val) {
            let mut ptab: *mut qjs::JSPropertyEnum = ptr::null_mut();
            let mut plen: u32 = 0;
            if qjs::JS_GetOwnPropertyNames(
                ctx,
                &mut ptab,
                &mut plen,
                headers_val,
                qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_ENUM_ONLY,
            ) == 0
            {
                for i in 0..plen {
                    let key = qjs::JS_AtomToCString(ctx, (*ptab.add(i as usize)).atom);
                    let val = qjs::sofuu_js_get_property(ctx, headers_val, (*ptab.add(i as usize)).atom);
                    let vs = qjs::sofuu_js_to_cstring(ctx, val);
                    if !key.is_null() && !vs.is_null() {
                        /* A CR/LF inside the name or value would inject
                         * extra headers via the curl slist — skip the header
                         * entirely rather than emit a hostile one (same rule
                         * as the AI path's API-key guard). */
                        let k = CStr::from_ptr(key).to_string_lossy();
                        let v = CStr::from_ptr(vs).to_string_lossy();
                        if k.contains(['\r', '\n']) || v.contains(['\r', '\n']) {
                            qjs::sofuu_js_free_cstring(ctx, key);
                            qjs::sofuu_js_free_cstring(ctx, vs);
                            qjs::sofuu_js_free_value(ctx, val);
                            qjs::JS_FreeAtom(ctx, (*ptab.add(i as usize)).atom);
                            continue;
                        }
                        let hbuf = format!("{}: {}", k, v);
                        let hc = CString::new(hbuf).unwrap_or_default();
                        /* P3 (AUDIT-2026-09-07): append returns NULL on OOM;
                         * the old code carried an empty if and kept going —
                         * every header (auth, content-type) silently dropped
                         * and the partial list leaked. Save the head, free
                         * the partial list, fail the fetch cleanly. */
                        let prev = req.req_headers;
                        req.req_headers = curl::curl_slist_append(req.req_headers, hc.as_ptr());
                        if req.req_headers.is_null() {
                            if !prev.is_null() {
                                curl::curl_slist_free_all(prev);
                            }
                            slist_failed = true;
                            qjs::sofuu_js_free_cstring(ctx, key);
                            qjs::sofuu_js_free_cstring(ctx, vs);
                            qjs::sofuu_js_free_value(ctx, val);
                            qjs::JS_FreeAtom(ctx, (*ptab.add(i as usize)).atom);
                            break;
                        }
                        /* Keep the parsed pair for per-hop slist rebuilds
                         * (net-5 credential stripping on origin change). */
                        req.headers_vec.push((k.into_owned(), v.into_owned()));
                    }
                    if !key.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, key);
                    }
                    if !vs.is_null() {
                        qjs::sofuu_js_free_cstring(ctx, vs);
                    }
                    qjs::sofuu_js_free_value(ctx, val);
                    qjs::JS_FreeAtom(ctx, (*ptab.add(i as usize)).atom);
                }
                qjs::js_free(ctx, ptab as *mut c_void);
            }
        }
        qjs::sofuu_js_free_value(ctx, headers_val);
        if slist_failed {
            /* P3 (AUDIT-2026-09-07): the header build failed — raise a clean
             * JS error instead of issuing a headerless request. Mirrors the
             * promise-alloc teardown below: the partial slist was freed in
             * the loop, so only the easy handle is live here. `req` is
             * still an owned Box at this point, so its Vec fields drop. */
            curl::curl_easy_cleanup(req.easy);
            return qjs::JS_ThrowInternalError(
                ctx,
                c"fetch: out of memory building request headers".as_ptr(),
            );
        }
        if !req.req_headers.is_null() {
            curl::curl_easy_setopt(req.easy, curl::CURLOPT_HTTPHEADER, req.req_headers);
        }

        /* body */
        let body_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"body".as_ptr());
        if !qjs::is_undefined(body_val) && !qjs::is_null(body_val) {
            let body = qjs::sofuu_js_to_cstring(ctx, body_val);
            if !body.is_null() {
                curl::curl_easy_setopt(req.easy, curl::CURLOPT_COPYPOSTFIELDS, body);
                qjs::sofuu_js_free_cstring(ctx, body);
            }
        }
        qjs::sofuu_js_free_value(ctx, body_val);
    }

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    if qjs::is_exception(promise) || out.is_null() {
        // Promise alloc failed (OOM) — tear down instead of adding a
        // handle with a NULL promise (which would leak the Response).
        if !req.req_headers.is_null() {
            curl::curl_slist_free_all(req.req_headers);
        }
        curl::curl_easy_cleanup(req.easy);
        return promise;
    }
    req.promise = out;

    let req_ptr = Box::into_raw(req);
    /* net-12: if the add fails the transfer is never driven — reject the
     * promise and tear the FetchReq down instead of leaking it. The easy
     * handle was never added, so remove/cleanup is all that's needed. The
     * (rejected) promise still goes back to the caller. */
    if curl::curl_multi_add_handle(curl_multi(), (*req_ptr).easy) != curl::CURLM_OK {
        let handle = (*req_ptr).promise;
        if !(*req_ptr).req_headers.is_null() {
            curl::curl_slist_free_all((*req_ptr).req_headers);
        }
        curl::curl_easy_cleanup((*req_ptr).easy);
        drop(Box::from_raw(req_ptr));
        sofuu_promise_reject_str(handle, c"fetch: failed to start transfer".as_ptr());
        sofuu_flush_jobs(ctx);
        return promise;
    }

    let mut running_handles: c_int = 0;
    curl::curl_multi_socket_action(curl_multi(), curl::CURL_SOCKET_TIMEOUT, 0, &mut running_handles);

    promise
}

// ── registration (C symbol replacement for mod_http_client_register) ──

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_http_client_register(ctx: *mut JSContext) {
    /* Register Response class */
    let mut class_id: qjs::JSClassID = 0;
    qjs::JS_NewClassID(&mut class_id);
    RESPONSE_CLASS_ID.store(class_id, std::sync::atomic::Ordering::Relaxed);
    let class_def = qjs::JSClassDef {
        class_name: c"Response".as_ptr(),
        finalizer: Some(response_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &class_def);

    let init = GLOBAL_INIT_DONE.with(|g| g.get());
    if init == 0 {
        curl::curl_global_init(curl::CURL_GLOBAL_ALL);
        let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
        TIMEOUT_TIMER.with(|t| t.set(timer));
        uv::uv_timer_init(sofuu_loop_get(), timer);

        let multi = curl::curl_multi_init();
        CURL_HANDLE.with(|c| c.set(multi));
        curl::curl_multi_setopt(multi, CURLMOPT_SOCKETFUNCTION, handle_socket as *const c_void);
        curl::curl_multi_setopt(multi, CURLMOPT_TIMERFUNCTION, start_timeout as *const c_void);

        GLOBAL_INIT_DONE.with(|g| g.set(1));
    }

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    if qjs::is_undefined(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }

    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu,
        c"fetch".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_sofuu_fetch, c"fetch".as_ptr(), 1),
    );

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);

    /* Also install as global `fetch` (mirrors browser/Deno/Bun behavior). */
    let global2 = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        global2,
        c"fetch".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_sofuu_fetch, c"fetch".as_ptr(), 1),
    );

    /* Response.body streaming bridges. */
    qjs::sofuu_js_set_property_str(
        ctx,
        global2,
        c"__fetch_stream_get".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_fetch_stream_get, c"__fetch_stream_get".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global2,
        c"__fetch_stream_next".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_fetch_stream_next, c"__fetch_stream_next".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        global2,
        c"__fetch_stream_wait".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_fetch_stream_wait, c"__fetch_stream_wait".as_ptr(), 1),
    );
    /* P3 (AUDIT-2026-09-07): compile the stream-iterator factory ONCE at
     * boot instead of re-evaluating the same JS source on every
     * Response.body access. response_body_getter falls back to an inline
     * eval if this global is missing. */
    let factory_src = CString::new(BODY_ITER_FACTORY).unwrap();
    let factory = qjs::JS_Eval(
        ctx,
        factory_src.as_ptr(),
        factory_src.as_bytes().len(),
        c"<body-iter>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    if !qjs::is_exception(factory) {
        qjs::sofuu_js_set_property_str(
            ctx,
            global2,
            c"__fetch_body_iter_factory".as_ptr(),
            factory,
        );
    }
    qjs::sofuu_js_free_value(ctx, global2);
}

// CURLMOPT_* — FUNCTIONPOINT(20000) + enum index (multi.h: SOCKETFUNCTION=1,
// TIMERFUNCTION=4). Verified against the system SDK headers after a wrong
// guess silently disabled the socket/timer callbacks.
const CURLMOPT_SOCKETFUNCTION: c_int = 20001;
const CURLMOPT_TIMERFUNCTION: c_int = 20004;

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::{self, CtxPtr};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    // ── pure-function unit tests (no runtime) ────────────────────────

    #[test]
    fn resolve_location_cases() {
        assert_eq!(resolve_location("http://a.com/x/y", "/z").unwrap(), "http://a.com/z");
        assert_eq!(resolve_location("http://a.com/x/y?q", "g").unwrap(), "http://a.com/x/g");
        assert_eq!(resolve_location("http://a.com/x/y", "../up").unwrap(), "http://a.com/up");
        assert_eq!(resolve_location("http://a.com/x/y", "//b.com/p").unwrap(), "http://b.com/p");
        assert_eq!(resolve_location("http://a.com/x/y", "?n=1").unwrap(), "http://a.com/x/y?n=1");
        assert_eq!(resolve_location("http://a.com/x/y", "https://b.io/p").unwrap(), "https://b.io/p");
        assert_eq!(resolve_location("http://a.com/a/b/../c", "d").unwrap(), "http://a.com/a/d");
        assert_eq!(
            resolve_location("http://a.com/x/y", "  http://B.io/p#frag  ").unwrap(),
            "http://B.io/p"
        );
        /* net-4: non-http absolute targets never resolve. */
        assert!(resolve_location("http://a.com/x", "ftp://b.io/p").is_none());
        assert!(resolve_location("http://a.com/x", "file:///etc/passwd").is_none());
        assert!(resolve_location("http://a.com/x", "").is_none());
    }

    #[test]
    fn sockaddr_linklocal_classification() {
        let mut v4 = curl::CurlSockaddr {
            family: libc::AF_INET,
            socktype: 1,
            protocol: 6,
            addrlen: 16,
            addr: [0u8; 128],
        };
        v4.addr[4] = 169;
        v4.addr[5] = 254;
        v4.addr[6] = 169;
        v4.addr[7] = 254;
        assert!(unsafe { sockaddr_is_linklocal(&v4) });
        let mut pub4 = v4.clone_addr_like();
        pub4.addr[4] = 8;
        pub4.addr[5] = 8;
        assert!(!unsafe { sockaddr_is_linklocal(&pub4) });

        let mut v6 = curl::CurlSockaddr {
            family: libc::AF_INET6,
            socktype: 1,
            protocol: 6,
            addrlen: 28,
            addr: [0u8; 128],
        };
        /* fe80::1 */
        v6.addr[8] = 0xfe;
        v6.addr[9] = 0x80;
        v6.addr[15] = 1;
        assert!(unsafe { sockaddr_is_linklocal(&v6) });
        /* fec0:: (site-local) is NOT fe80::/10 */
        let mut site = v6.clone_addr_like();
        site.addr[9] = 0xc0;
        assert!(!unsafe { sockaddr_is_linklocal(&site) });
        /* ::ffff:169.254.169.254 */
        let mut mapped = v6.clone_addr_like();
        mapped.addr[8..18].fill(0);
        mapped.addr[18] = 0xff;
        mapped.addr[19] = 0xff;
        mapped.addr[20] = 169;
        mapped.addr[21] = 254;
        assert!(unsafe { sockaddr_is_linklocal(&mapped) });
        /* ::ffff:8.8.8.8 */
        let mut mapped_pub = mapped.clone_addr_like();
        mapped_pub.addr[20] = 8;
        mapped_pub.addr[21] = 8;
        assert!(!unsafe { sockaddr_is_linklocal(&mapped_pub) });
    }

    // Tiny helper so the classification test can clone-and-tweak structs
    // (CurlSockaddr is Copy-able via its fields; this avoids derive noise).
    impl CurlSockaddrExt for curl::CurlSockaddr {
        fn clone_addr_like(&self) -> curl::CurlSockaddr {
            curl::CurlSockaddr {
                family: self.family,
                socktype: self.socktype,
                protocol: self.protocol,
                addrlen: self.addrlen,
                addr: self.addr,
            }
        }
    }
    trait CurlSockaddrExt {
        fn clone_addr_like(&self) -> curl::CurlSockaddr;
    }

    // ── live event-loop harness ──────────────────────────────────────

    struct TestServer {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    /// Scripted HTTP server: hands out `responses[i]` to the i-th connection
    /// and records every raw request. Responses must use Connection: close.
    fn spawn_server(responses: Vec<String>) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let reqs = requests.clone();
        std::thread::spawn(move || {
            for (i, stream) in listener.incoming().enumerate() {
                if i >= responses.len() {
                    break;
                }
                let Ok(mut stream) = stream else { break };
                let resp = responses[i].clone();
                let reqs = reqs.clone();
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        match stream.read(&mut chunk) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    reqs.lock().unwrap().push(String::from_utf8_lossy(&buf).into_owned());
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                });
            }
        });
        TestServer { url: format!("http://127.0.0.1:{}", port), requests }
    }

    fn resp_302(location: &str) -> String {
        format!(
            "HTTP/1.1 302 Found\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            location
        )
    }

    fn resp_200(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    unsafe fn read_global(ctx: *mut qjs::JSContext, name: &CStr) -> String {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let v = qjs::sofuu_js_get_property_str(ctx, global, name.as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        qjs::sofuu_js_free_value(ctx, v);
        let s = if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        if !p.is_null() {
            qjs::sofuu_js_free_cstring(ctx, p);
        }
        s
    }

    /// Boot a fresh runtime + loop, register the fetch surface, eval the
    /// script, run the loop to completion, then hand the still-live ctx to
    /// `check` for global assertions. Err = the eval raised (script bugs /
    /// unexpected sync throws — intentional sync throws are caught by the
    /// scripts themselves into globals).
    unsafe fn with_engine(script: &str, check: impl FnOnce(*mut qjs::JSContext)) -> Result<(), String> {
        let _loop_guard = crate::rt::test_loop_lock();
        let rt = qjs::JS_NewRuntime();
        let ctx = qjs::JS_NewContext(rt);
        let _ctx_guard = CtxPtr::new(ctx);
        crate::rt::event_loop::sofuu_loop_init();
        mod_http_client_register(ctx);

        let mut eval_err: Option<String> = None;
        let script_c = CString::new(script).unwrap();
        let r = qjs::JS_Eval(
            ctx,
            script_c.as_ptr(),
            script_c.as_bytes().len(),
            c"<fetch-test>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
        if qjs::is_exception(r) {
            qjs::js_std_dump_error(ctx);
            eval_err = Some("script eval threw".into());
        } else {
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run(ctx);
        }

        if eval_err.is_none() {
            check(ctx);
        }
        crate::rt::event_loop::sofuu_loop_close();
        qjs::JS_FreeContext(ctx);
        qjs::JS_FreeRuntime(rt);
        eval_err.map_or(Ok(()), Err)
    }

    // A sync-throw (entry guards, net-6) AND a promise rejection must both
    // land in globalThis.err; only an unexpected resolve says RESOLVED.
    // Takes a THUNK so a fetch() that throws synchronously is still caught.
    const GUARD_SNIPPET: &str = "\
function go(f) {
  try {
    f().then(function (r) { globalThis.err = 'RESOLVED ' + r.status; },
             function (e) { globalThis.err = String(e && e.message || e); });
  } catch (e) {
    globalThis.err = String(e && e.message || e);
  }
}";

    // ── net-6: method token validation ───────────────────────────────

    #[test]
    fn method_crlf_injection_rejected() {
        unsafe { with_engine(
            &format!(
                "{snippet}
                 globalThis.err='';
                 go(function () {{ return fetch('http://127.0.0.1:9/x', {{method: 'GET\\r\\nX-Inj: 1'}}); }});",
                snippet = GUARD_SNIPPET
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.contains("invalid HTTP method"), "got: {err}");
            },
        )
        .unwrap();
        }
    }

    // ── net-1: NUL-containing bodies survive text()/json() ───────────
    // Negative control: against the pre-fix CString detour the text() test
    // observed codes == "" (unwrap_or_default emptied the body) and the
    // json() test RESOLVED with the silently-truncated prefix {"a":1}.

    #[test]
    fn nul_body_text_round_trips_fully() {
        let b = spawn_server(vec![resp_200("AB\u{0}CD")]);
        unsafe { with_engine(
            &format!(
                "globalThis.err='';
                 fetch('{url}/x').then(function (r) {{ return r.text(); }})
                  .then(function (t) {{
                    globalThis.codes = Array.from(t).map(function (c) {{ return c.charCodeAt(0); }}).join(',');
                  }}, function (e) {{ globalThis.err = String(e && e.message || e); }});",
                url = b.url
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.is_empty(), "unexpected error: {err}");
                let codes = unsafe { read_global(ctx, c"codes") };
                assert_eq!(codes, "65,66,0,67,68", "NUL must survive the round-trip");
            },
        )
        .unwrap();
        }
    }

    #[test]
    fn nul_body_json_rejects_instead_of_silent_prefix() {
        let b = spawn_server(vec![resp_200("{\"a\":1}\u{0}JUNK")]);
        unsafe { with_engine(
            &format!(
                "globalThis.out='';
                 fetch('{url}/x').then(function (r) {{ return r.json(); }})
                  .then(function (j) {{ globalThis.out = 'RESOLVED ' + JSON.stringify(j); }},
                        function (e) {{ globalThis.out = 'REJECTED ' + String(e && e.message || e); }});",
                url = b.url
            ),
            |ctx| {
                let out = unsafe { read_global(ctx, c"out") };
                assert!(
                    out.starts_with("REJECTED"),
                    "a NUL-corrupted wire body must reject, got: {out}"
                );
            },
        )
        .unwrap();
        }
    }

    #[test]
    fn json_plain_body_still_parses() {
        let b = spawn_server(vec![resp_200("{\"a\":1}")]);
        unsafe { with_engine(
            &format!(
                "globalThis.out='';
                 fetch('{url}/x').then(function (r) {{ return r.json(); }})
                  .then(function (j) {{ globalThis.out = 'RESOLVED ' + JSON.stringify(j); }},
                        function (e) {{ globalThis.out = 'REJECTED ' + String(e && e.message || e); }});",
                url = b.url
            ),
            |ctx| {
                let out = unsafe { read_global(ctx, c"out") };
                assert_eq!(out, "RESOLVED {\"a\":1}", "plain JSON must parse unchanged");
            },
        )
        .unwrap();
        }
    }

    // ── net-4 wire level: link-local refused via opensocket ──────────

    #[test]
    fn linklocal_v6_refused_at_wire_level() {
        unsafe { with_engine(
            &format!(
                "{snippet}
                 globalThis.err='';
                 go(function () {{ return fetch('http://[fe80::1]:1/x'); }});",
                snippet = GUARD_SNIPPET
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.contains("blocked"), "expected the wire-level refusal, got: {err}");
            },
        )
        .unwrap();
        }
    }

    // ── entry guard: literal metadata IP (kept from P1-15) ───────────

    #[test]
    fn metadata_ip_entry_guard() {
        unsafe { with_engine(
            &format!(
                "{snippet}
                 globalThis.err='';
                 go(function () {{ return fetch('http://169.254.169.254/latest/meta-data/'); }});",
                snippet = GUARD_SNIPPET
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.contains("metadata"), "got: {err}");
            },
        )
        .unwrap();
        }
    }

    // ── net-5 + P1-3: cross-origin credential strip, final status ────

    #[test]
    fn redirect_chain_strips_credentials_cross_origin() {
        let b = spawn_server(vec![resp_200("hello")]);
        let a = spawn_server(vec![
            /* hop 1: same-origin relative redirect (credentials kept) */
            resp_302("/r2"),
            /* hop 2: cross-origin redirect (credentials stripped) */
            resp_302(&format!("{}/r3", b.url)),
        ]);

        unsafe { with_engine(
            &format!(
                "globalThis.err=''; globalThis.st=''; globalThis.body=''; globalThis.furl='';
                 fetch('{a}/r1', {{headers: {{Authorization: 'Bearer s3cr3t', Cookie: 'k=v', 'X-Tag': 'keepme'}}}})
                   .then(function (r) {{
                     globalThis.st = String(r.status); globalThis.furl = String(r.url || '');
                     return r.text();
                   }})
                   .then(function (t) {{ globalThis.body = t; }})
                   .catch(function (e) {{ globalThis.err = String(e && e.message || e); }});",
                a = a.url
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                let st = unsafe { read_global(ctx, c"st") };
                let body = unsafe { read_global(ctx, c"body") };
                let furl = unsafe { read_global(ctx, c"furl") };
                assert!(err.is_empty(), "unexpected error: {err}");
                /* P1-3: the promise carries the FINAL response's status. */
                assert_eq!(st, "200", "status must be the final response's");
                assert_eq!(body, "hello");
                assert!(furl.starts_with(&b.url) && furl.ends_with("/r3"), "final url: {furl}");

                let a_reqs = a.requests.lock().unwrap();
                assert_eq!(a_reqs.len(), 2, "server A saw {} requests", a_reqs.len());
                for r in a_reqs.iter() {
                    assert!(
                        r.contains("Authorization: Bearer s3cr3t"),
                        "same-origin hop must keep credentials: {r}"
                    );
                    assert!(r.contains("Cookie: k=v"), "same-origin hop must keep cookies: {r}");
                }
                let b_reqs = b.requests.lock().unwrap();
                assert_eq!(b_reqs.len(), 1);
                assert!(
                    b_reqs[0].contains("X-Tag: keepme"),
                    "non-credential headers must survive: {}",
                    b_reqs[0]
                );
                assert!(
                    !b_reqs[0].contains("Authorization"),
                    "cross-origin hop must strip Authorization: {}",
                    b_reqs[0]
                );
                assert!(
                    !b_reqs[0].contains("Cookie"),
                    "cross-origin hop must strip Cookie: {}",
                    b_reqs[0]
                );
            },
        )
        .unwrap();
        }
    }

    // ── P1-3: 3xx without Location finalizes with its own status ─────

    #[test]
    fn redirect_without_location_finalizes() {
        let s = spawn_server(vec![
            "HTTP/1.1 304 Not Modified\r\nETag: \"x\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_string(),
        ]);
        unsafe { with_engine(
            &format!(
                "globalThis.err=''; globalThis.st='';
                 fetch('{u}/x')
                   .then(function (r) {{ globalThis.st = String(r.status); }})
                   .catch(function (e) {{ globalThis.err = String(e && e.message || e); }});",
                u = s.url
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                let st = unsafe { read_global(ctx, c"st") };
                assert!(err.is_empty(), "unexpected error: {err}");
                assert_eq!(st, "304");
            },
        )
        .unwrap();
        }
    }

    // ── net-4: non-http redirect target refused ──────────────────────

    #[test]
    fn redirect_to_ftp_refused() {
        let s = spawn_server(vec![resp_302("ftp://127.0.0.1/x")]);
        unsafe { with_engine(
            &format!(
                "globalThis.err=''; globalThis.st='';
                 fetch('{u}/x')
                   .then(function (r) {{ globalThis.st = String(r.status); }})
                   .catch(function (e) {{ globalThis.err = String(e && e.message || e); }});",
                u = s.url
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                let st = unsafe { read_global(ctx, c"st") };
                assert!(st.is_empty(), "must not resolve, got status {st}");
                assert!(err.contains("redirect"), "got: {err}");
            },
        )
        .unwrap();
        }
    }

    // ── P1-2: the transfer-ERROR path must park a Response whose JS object
    // was GC'd mid-transfer, exactly like the DONE path. Before the fix it
    // only set done=1/errored=1 and left the finalized ResponseData in
    // RESPONSES unreclaimed — the body buffer leaked for the process life.

    unsafe extern "C" fn js_test_gc(
        ctx: *mut qjs::JSContext,
        _this: qjs::JSValueConst,
        _argc: c_int,
        _argv: *const qjs::JSValueConst,
    ) -> qjs::JSValue {
        qjs::JS_RunGC(qjs::JS_GetRuntime(ctx));
        qjs::sofuu_js_undefined()
    }

    #[test]
    fn transfer_error_parks_gc_d_mid_transfer_response() {
        use std::sync::atomic::{AtomicBool, Ordering as AOrd};

        // Three-connection choreography:
        //   conn 0 (/x):      headers with Content-Length 100, send 7 bytes,
        //                     then HOLD the connection open — the transfer
        //                     is mid-flight while the fetch promise is
        //                     already resolved.
        //   conn 1 (/signal): arms the kill: abort conn 0 → curl ends it
        //                     with CURLE_PARTIAL_FILE (the ERROR path).
        //   conn 2 (/y):      plain 200; js_sofuu_fetch runs
        //                     reclaim_orphan_resp at entry.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut held_x: Option<std::net::TcpStream> = None;
            for (i, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { break };
                if i >= 3 { break; }
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") { break; }
                        }
                        Err(_) => break,
                    }
                }
                match i {
                    0 => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial");
                        let _ = stream.flush();
                        held_x = Some(stream); /* mid-transfer: do NOT close */
                    }
                    1 => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
                        let _ = stream.flush();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        /* EOF with 93 bytes unsent → curl error on /x. */
                        if let Some(mut sx) = held_x.take() {
                            let _ = sx.shutdown(std::net::Shutdown::Both);
                        }
                    }
                    _ => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
                        let _ = stream.flush();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                    }
                }
            }
        });

        let script = format!(
            "globalThis.state = '';
             fetch('{u}/x').then(function (r) {{
               // Fetch already resolved (header block) but the transfer is
               // still open. Drop the ONLY reference and force collection:
               // the finalizer runs with done==0 → finalized=1 (deferred).
               globalThis.hold = r;
               globalThis.hold = null;
               __run_gc();
               globalThis.state += 'dropped;';
               return fetch('{u}/signal');
             }}).then(function () {{
               globalThis.state += 'signalled;';
               // Entry runs reclaim_orphan_resp — with the fix the parked
               // orphan is freed here, far from any curl callback.
               return fetch('{u}/y');
             }}).then(function () {{
               globalThis.state += 'reclaimed;';
             }}).catch(function (e) {{
               globalThis.state += 'err:' + (e && e.message || e);
             }});",
            u = format!("http://127.0.0.1:{}", port)
        );

        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            crate::rt::event_loop::sofuu_loop_init();
            sofuu_ffi::bridge::register_global_fn(
                ctx, "__run_gc", js_test_gc as sofuu_ffi::bridge::JSCFunction);
            mod_http_client_register(ctx);

            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(ctx, script_c.as_ptr(), script_c.as_bytes().len(),
                                 c"<p1-2>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
            assert!(!qjs::is_exception(r), "script eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run(ctx);

            let state = read_global(ctx, c"state");
            assert_eq!(
                state, "dropped;signalled;reclaimed;",
                "choreography must complete without a fetch error"
            );
            // Determinism: collect the /signal and /y Response objects —
            // their transfers completed (done==1), so their finalizers
            // unregister them; the only possible survivor is the /x orphan.
            qjs::JS_RunGC(rt);
            // The /x error may be drained one socket-action pass AFTER
            // /signal's DONE, i.e. after fetch('/y') already ran its entry
            // reclaim — the park can postdate it. Reclaiming here (loop
            // drained, far from any curl callback) mirrors production's
            // next-fetch reclaim. Pre-fix nothing is parked and the
            // finalized ResponseData still sits in RESPONSES → len 1.
            reclaim_orphan_resp();
            let leaked = RESPONSES.with(|rs| rs.borrow().len());
            assert_eq!(leaked, 0, "finalized mid-transfer response must be reclaimed");

            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // ── net-9: reclaim_orphan_resp must settle armed waiters before free ──
    // The orphan park (done==1, finalized==1) is the post-collection state
    // of a mid-transfer response whose JS object is already gone — so this
    // test drives the struct directly, no JS Response wrapper. Both waiter
    // slots are armed exactly as response_wait_done / js_fetch_stream_wait
    // arm them. Pre-fix: reclaim freed the Box and dropped the handles —
    // both awaits hung forever (out1/out2 stay '').
    #[test]
    fn orphan_reclaim_rejects_armed_waiters() {
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);

            // A parked-orphan-shaped ResponseData (done + finalized), and
            // registered in RESPONSES like build_response registers it.
            let r = Box::into_raw(Box::new(ResponseData {
                body: Vec::new(),
                headers_raw: Vec::new(),
                url: CString::new("http://127.0.0.1/x").unwrap(),
                status_text: CString::new("OK").unwrap(),
                chunks: VecDeque::new(),
                done: 1,
                errored: 0,
                err_msg: None,
                finalized: 1,
                pending_next: ptr::null_mut(),
                pending_resolve: qjs::sofuu_js_undefined(),
                pending_settle: Vec::new(),
                ctx,
            }));
            RESPONSES.with(|rs| rs.borrow_mut().push(r));
            // Park it as the DONE path does for a finalized response.
            ORPHAN_RESP.with(|o| o.set(r));

            // Arm the settle waiter (response_wait_done's pending branch).
            let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
            let promise_settle = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
            (*r).pending_settle.push(Box::into_raw(Box::new(PromiseHandle {
                ctx,
                resolve: qjs::sofuu_js_dup_value(ctx, resolvers[0]),
                reject: qjs::sofuu_js_dup_value(ctx, resolvers[1]),
            })));
            qjs::sofuu_js_free_value(ctx, resolvers[0]);
            qjs::sofuu_js_free_value(ctx, resolvers[1]);

            // Arm the stream waiter (js_fetch_stream_wait's pending branch,
            // including the dup'd pending_resolve the fulfill path frees).
            let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
            let promise_next = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
            (*r).pending_next = Box::into_raw(Box::new(PromiseHandle {
                ctx,
                resolve: qjs::sofuu_js_dup_value(ctx, resolvers[0]),
                reject: qjs::sofuu_js_dup_value(ctx, resolvers[1]),
            }));
            (*r).pending_resolve = qjs::sofuu_js_dup_value(ctx, resolvers[0]);
            qjs::sofuu_js_free_value(ctx, resolvers[0]);
            qjs::sofuu_js_free_value(ctx, resolvers[1]);

            // Publish the promises (setPropertyStr transfers the ref) and
            // attach rejection observers from JS.
            let global = qjs::sofuu_js_get_global_object(ctx);
            qjs::sofuu_js_set_property_str(ctx, global, c"p1".as_ptr(), promise_settle);
            qjs::sofuu_js_set_property_str(ctx, global, c"p2".as_ptr(), promise_next);
            qjs::sofuu_js_free_value(ctx, global);
            let script = c"globalThis.out1=''; globalThis.out2='';
               globalThis.p1.then(function (v) { globalThis.out1 = 'RESOLVED ' + String(v); },
                                  function (e) { globalThis.out1 = 'REJECTED ' + (e && e.message || e); });
               globalThis.p2.then(function (v) { globalThis.out2 = 'RESOLVED ' + String(v); },
                                  function (e) { globalThis.out2 = 'REJECTED ' + (e && e.message || e); });";
            let sres = qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<net-9>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(sres), "observer eval threw");
            qjs::sofuu_js_free_value(ctx, sres);

            // The next fetch's entry reclaim. With the fix both waiters are
            // rejected before the Box is freed; pre-fix they never fire.
            reclaim_orphan_resp();
            crate::rt::promise::sofuu_flush_jobs(ctx);

            let out1 = read_global(ctx, c"out1");
            let out2 = read_global(ctx, c"out2");
            assert!(
                out1.starts_with("REJECTED") && out1.contains("finalized"),
                "settle waiter must reject at reclaim, got: {out1:?}"
            );
            assert!(
                out2.starts_with("REJECTED") && out2.contains("finalized"),
                "stream waiter must reject at reclaim, got: {out2:?}"
            );
            let left = RESPONSES.with(|rs| rs.borrow().len());
            assert_eq!(left, 0, "orphan must leave RESPONSES");

            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // ── net-12: curl_multi_add_handle must be checked ───────────────────

    // Site 2 (js_sofuu_fetch): a failed multi add previously left the
    // FetchReq outside the multi — the promise pended forever and the req
    // leaked. The test forces the add to fail by swapping the thread-local
    // multi for a null handle (CURLM_BAD_HANDLE) after registration but
    // before the fetch eval, and proves the promise REJECTS instead.
    // Pre-fix: out stays '' (pends) and the req leaks.
    #[test]
    fn fetch_rejects_when_multi_add_fails() {
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            crate::rt::event_loop::sofuu_loop_init();
            mod_http_client_register(ctx);

            // Force curl_multi_add_handle to fail: no multi registered.
            let saved = CURL_HANDLE.with(|c| c.replace(ptr::null_mut()));

            let script = c"globalThis.err='';
               try { fetch('http://127.0.0.1:9/x').then(
                 function (r) { globalThis.err = 'RESOLVED ' + r.status; },
                 function (e) { globalThis.err = 'REJECTED ' + (e && e.message || e); }); }
               catch (e) { globalThis.err = 'THREW ' + (e && e.message || e); }";
            let r = qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<net-12>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            assert!(!qjs::is_exception(r), "eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run(ctx);

            CURL_HANDLE.with(|c| c.set(saved));

            let err = read_global(ctx, c"err");
            assert!(
                err.starts_with("REJECTED") && err.contains("start transfer"),
                "failed multi add must reject the promise, got: {err:?}"
            );

            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // Site 1 (redirect_reissue): same failure mode one hop in — the caller
    // (check_multi_info_global) must reject with the reissue reason and tear
    // down. Drives redirect_reissue directly against a null multi.
    #[test]
    fn redirect_reissue_errors_when_multi_add_fails() {
        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            let easy = curl::curl_easy_init();
            assert!(!easy.is_null());
            let req = Box::into_raw(Box::new(FetchReq {
                magic: REQ_MAGIC,
                ctx,
                promise: ptr::null_mut(),
                body: Vec::new(),
                headers_raw: Vec::new(),
                resp: ptr::null_mut(),
                resp_cap: 0,
                easy,
                req_headers: ptr::null_mut(),
                hops: 0,
                refused: 0,
                headers_vec: Vec::new(),
                method: None,
                max_body: 0,
                cap_hit: 0,
            }));

            // Force the add to fail: no multi registered at all.
            let saved = CURL_HANDLE.with(|c| c.replace(ptr::null_mut()));
            let res = redirect_reissue(req, 302, "http://127.0.0.1:1/a", "http://127.0.0.1:2/b");
            CURL_HANDLE.with(|c| c.set(saved));

            assert_eq!(
                res,
                Err("fetch: failed to re-issue redirect transfer"),
                "failed multi add must surface as Err"
            );

            drop(Box::from_raw(req));
            curl::curl_easy_cleanup(easy);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // ── net-7: chunk delivery must not cross the FFI per byte ───────────
    // response_next_chunk used to build Uint8Array(len) and fill it with one
    // JS_SetPropertyUint32 per byte — O(n) generic property writes on the
    // stream hot path. Now: one JS_NewArrayBufferCopy + one Uint8Array view.

    // Byte-exactness of the new delivery: all 256 byte values × 4 rounds
    // through the body iterator — NUL must be present, and 0xFF must arrive
    // as 255 (a signed view would deliver −1 and skew the sum).
    #[test]
    fn stream_chunk_bytes_exact_all_values() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut body = Vec::new();
        for _ in 0..4 {
            for b in 0..=255u8 {
                body.push(b);
            }
        }
        let want_sum: u64 = body.iter().map(|&b| b as u64).sum();
        let want_len = body.len();
        let mut raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(&body);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let _ = stream.write_all(&raw);
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Both);
                break; /* exactly one connection */
            }
        });

        unsafe { with_engine(
            &format!(
                "globalThis.err=''; globalThis.count=0; globalThis.sum=0; globalThis.sawNul=0;
                 fetch('{url}/x').then(function (r) {{
                   globalThis.hold = r; /* keep the Response alive while streaming */
                   var it = r.body[Symbol.asyncIterator]();
                   function pump() {{
                     return it.next().then(function (res) {{
                       if (res.done) return;
                       for (var i = 0; i < res.value.length; i++) {{
                         globalThis.count++;
                         globalThis.sum += res.value[i];
                         if (res.value[i] === 0) globalThis.sawNul = 1;
                       }}
                       return pump();
                     }});
                   }}
                   return pump();
                 }}).then(function () {{ }},
                       function (e) {{ globalThis.err = String(e && e.message || e); }});",
                url = format!("http://127.0.0.1:{}", port)
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.is_empty(), "unexpected error: {err}");
                let count = unsafe { read_global(ctx, c"count") };
                assert_eq!(count, "1024", "byte count through the view");
                let saw_nul = unsafe { read_global(ctx, c"sawNul") };
                assert_eq!(saw_nul, "1", "NUL byte must survive the view");
                let sum = unsafe { read_global(ctx, c"sum") };
                assert_eq!(
                    sum,
                    want_sum.to_string(),
                    "byte sum must match (0xFF as 255, not −1)"
                );
                let _ = want_len;
            },
        )
        .unwrap();
        }
    }

    // Hot-path shape: bulk streaming must be fast now that the per-byte
    // property-write loop is gone. A/B measured IN ONE PROCESS — the same
    // payload streams twice, once with the legacy per-byte fill leg and once
    // with the fixed single-ArrayBuffer path — and the fixed leg must win by
    // ≥1.8×. Reverting the fix makes both legs identical (ratio ≈ 1.0), so
    // the negative control is structural: the test fails if the fix regresses.
    #[test]
    fn stream_bulk_delivers_fast_ab() {
        let n: usize = 1 << 24; /* 16 MiB per leg — per-byte crossings (~45ns)
                                 * scale linearly with bytes, transfer overhead
                                 * is shared, so the legs separate cleanly. */
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body: Vec<u8> = (0..n).map(|i| ((i.wrapping_mul(31)).wrapping_add(7) & 0xFF) as u8).collect();
        let mut raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(&body);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let _ = stream.write_all(&raw);
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        });

        let script = |url: String| format!(
            "globalThis.err=''; globalThis.count=0; globalThis.chunks=0;
             fetch('{url}/x').then(function (r) {{
               globalThis.hold = r; /* keep the Response alive while streaming */
               var it = r.body[Symbol.asyncIterator]();
               function pump() {{
                 return it.next().then(function (res) {{
                   if (res.done) return;
                   /* O(chunks) consumer — a per-byte JS loop here would
                    * mask the native per-byte FFI cost net-7 fixes. */
                   globalThis.chunks++;
                   globalThis.count += res.value.length;
                   return pump();
                 }});
               }}
               return pump();
             }}).then(function () {{ }},
                   function (e) {{ globalThis.err = String(e && e.message || e); }});"
        );

        let mut legs: Vec<(&str, f64, u64)> = Vec::new();
        for (label, legacy) in [("legacy", true), ("fixed", false)] {
            /* Each leg on its own thread: the uv-bound engine state (curl
             * multi, timeout timer, stream check) is created once per thread
             * on the first event loop, so a second with_engine boot in the
             * same thread would stall waiters on the closed loop 1. */
            let url = format!("http://127.0.0.1:{}", port);
            let (tx, rx) = std::sync::mpsc::channel::<(f64, u64, u64, u64)>();
            let script = script(url);
            std::thread::spawn(move || {
                NET7_LEGACY_FILL.store(legacy, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                unsafe { with_engine(
                    &script,
                    |ctx| {
                        let err = unsafe { read_global(ctx, c"err") };
                        assert!(err.is_empty(), "unexpected error in {label} leg: {err}");
                        let count = unsafe { read_global(ctx, c"count") };
                        assert_eq!(count, format!("{n}"), "{label} leg: all bytes delivered");
                        let chunks = unsafe { read_global(ctx, c"chunks") };
                        let fills = N7_FILL_CALLS.with(|c| c.get());
                        let legacy_fills = N7_LEGACY_CALLS.with(|c| c.get());
                        let fill_ns = N7_FILL_NS.with(|c| c.get());
                        eprintln!(
                            "net-7 leg {label}: chunks={chunks} fills={fills} legacy_fills={legacy_fills} fill_ms={:.1}",
                            fill_ns as f64 / 1e6
                        );
                        tx.send((t0.elapsed().as_secs_f64(), fills, legacy_fills, fill_ns)).unwrap();
                    },
                )
                .unwrap();
                }
            }).join().unwrap();
            let (wall, _fills, _legacy_fills, fill_ns) = rx.recv().unwrap();
            legs.push((label, wall, fill_ns));
        }
        NET7_LEGACY_FILL.store(false, std::sync::atomic::Ordering::Relaxed);

        let lookup = |want: &str| {
            legs.iter()
                .find(|(l, _, _)| *l == want)
                .map(|(_, t, f)| (*t, *f))
                .unwrap()
        };
        let (fixed, fixed_fill_ns) = lookup("fixed");
        let (legacy, legacy_fill_ns) = lookup("legacy");
        let ratio = legacy / fixed;
        let fill_ratio = legacy_fill_ns as f64 / (fixed_fill_ns.max(1) as f64);
        eprintln!(
            "net-7 A/B: fixed {fixed:.3}s, legacy {legacy:.3}s, ratio {ratio:.2}; fill ratio {fill_ratio:.1}"
        );
        /* The structural signal is the native fill-time ratio measured inside
         * response_next_chunk itself: transfer overhead and leg-position warm-up
         * cancel out (proven: reverting the fix leaves wall-ratio ~6× via
         * position asymmetry but drops fill ratio to 1.0). */
        assert!(
            fill_ratio >= 4.0,
            "per-byte fill must be ≥4× slower inside response_next_chunk than the \
             single-ArrayBuffer path (fixed {fixed:.3}s fill {fixed_fill_ns}ns, legacy \
             {legacy:.3}s fill {legacy_fill_ns}ns, fill ratio {fill_ratio:.1})"
        );
    }

    // ── net-8: a second body reader must see the FULL body ───────────────
    // The audited shape: TWO text() readers issued BACK-TO-BACK MID-STREAM.
    // response_wait_done's single pending_settle slot pre-resolved every
    // waiter after the first, so the second text() read the still-partial
    // body. Fix: a Vec of waiters; every pre-done reader arms, the DONE
    // pump settles them all. (A post-done second reader was never broken.)
    #[test]
    fn second_body_reader_gets_full_body() {
        let n: usize = 1 << 20; /* 1 MiB total */
        let first: usize = 64 << 10; /* 64 KiB written before the hold */
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        /* Printable ASCII only (32..126): one byte per char, so JS string
         * .length equals the byte count — no UTF-16 pair collapsing. */
        let body: Vec<u8> = (0..n).map(|i| (32 + (i.wrapping_mul(31) % 95)) as u8).collect();
        let mut raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(&body[..first]);
        let rest = body[first..].to_vec();
        let server = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let _ = stream.write_all(&raw);
                let _ = stream.flush();
                std::thread::sleep(std::time::Duration::from_millis(150));
                let _ = stream.write_all(&rest);
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Both);
                break; /* exactly one connection */
            }
        });

        unsafe { with_engine(
            &format!(
                "globalThis.err=''; globalThis.first=0; globalThis.t1=0; globalThis.t2=0;
                 fetch('{url}/x').then(function (r) {{
                   globalThis.hold = r; /* keep the Response alive across readers */
                   return r.body[Symbol.asyncIterator]().next();
                 }}).then(function (res) {{
                   globalThis.first = res.value.length;
                   /* Both readers issued while the transfer is STILL in
                    * flight — both must arm and both must see the whole
                    * body once done. */
                   var p1 = globalThis.hold.text();
                   var p2 = globalThis.hold.text();
                   return Promise.all([p1, p2]);
                 }}).then(function (ts) {{
                   globalThis.t1 = ts[0].length;
                   globalThis.t2 = ts[1].length;
                 }}, function (e) {{ globalThis.err = String(e && e.message || e); }});",
                url = format!("http://127.0.0.1:{}", port)
            ),
            |ctx| {
                let err = unsafe { read_global(ctx, c"err") };
                assert!(err.is_empty(), "unexpected error: {err}");
                let first = unsafe { read_global(ctx, c"first") };
                let first: usize = first.parse().unwrap_or(0);
                assert!(first > 0 && first < n, "probe must consume a partial chunk (got {first})");
                let t1 = unsafe { read_global(ctx, c"t1") };
                assert_eq!(t1, format!("{n}"), "first mid-stream text() must see the full body");
                let t2 = unsafe { read_global(ctx, c"t2") };
                assert_eq!(t2, format!("{n}"), "second mid-stream text() must see the full body");
            },
        )
        .unwrap();
        }
        server.join().unwrap();
    }

    // ── P1-4: stream_check_cb must not hold the RESPONSES borrow across JS ──
    // response_fulfill_pending → response_next_chunk runs JS_CallConstructor
    // (Uint8Array). Under the OLD inline-borrow body, a GC triggered inside
    // that constructor collects a dead Response whose finalizer — the
    // transfer is already done (done==1) — borrow_mut()s the very cell the
    // callback still holds → BorrowMutError abort of the runtime.

    #[test]
    fn stream_check_gc_during_constructor_no_borrow_abort() {
        // One connection: headers flush first, then a ~150ms hold — the
        // client resolves the fetch and arms pending_next inside that window
        // — then body + EOF in one pass. Content-Length is satisfied by that
        // single write, so curl completes the transfer in the same socket
        // pass: by the time the uv_check fulfill runs, done==1 is set and
        // the finalizer takes the borrow_mut path.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = Vec::new();
                let mut piece = [0u8; 4096];
                loop {
                    match stream.read(&mut piece) {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&piece[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") { break; }
                        }
                        Err(_) => break,
                    }
                }
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n");
                let _ = stream.flush();
                std::thread::sleep(std::time::Duration::from_millis(150));
                let _ = stream.write_all(b"chunked");
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Both);
                break; /* exactly one connection */
            }
        });

        let script = format!(
            "globalThis.state = '';
             var RealU8 = Uint8Array;
             var trapped = 0;
             globalThis.Uint8Array = function (len) {{
               if (!trapped) {{
                 trapped = 1;
                 // The ONLY reference to the Response dies here, inside the
                 // Uint8Array constructor that response_next_chunk runs
                 // while (pre-fix) stream_check_cb still holds the borrow.
                 globalThis.hold = null;
                 __run_gc();
                 globalThis.state += 'gc-in-ctor;';
               }}
               return new RealU8(len);
             }};
             fetch('{u}/x').then(function (r) {{
               globalThis.hold = r;
               return r.body[Symbol.asyncIterator]().next();
             }}).then(function (res) {{
               globalThis.state += 'chunk:' + (res && res.value ? res.value.length : -1)
                 + ';done:' + (res && res.done ? 1 : 0) + ';';
             }}).catch(function (e) {{
               globalThis.state += 'err:' + (e && e.message || e) + ';';
             }});",
            u = format!("http://127.0.0.1:{}", port)
        );

        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            crate::rt::event_loop::sofuu_loop_init();
            sofuu_ffi::bridge::register_global_fn(
                ctx, "__run_gc", js_test_gc as sofuu_ffi::bridge::JSCFunction);
            mod_http_client_register(ctx);

            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(ctx, script_c.as_ptr(), script_c.as_bytes().len(),
                                 c"<p1-4>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
            assert!(!qjs::is_exception(r), "script eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run(ctx);

            let state = read_global(ctx, c"state");
            // The chunk fulfillment is the FIRST Uint8Array construction in
            // this script, so 'gc-in-ctor' proves the constructor — and the
            // GC it triggered — ran inside stream_check_cb's fulfill pass.
            assert!(
                state.contains("gc-in-ctor;"),
                "GC must run inside the Uint8Array constructor during fulfill; got: {state}"
            );
            assert_eq!(
                state, "gc-in-ctor;chunk:7;done:0;",
                "stream must still deliver the chunk after the mid-fulfill GC"
            );
            // The finalizer ran with done==1 → it removed the response from
            // RESPONSES itself. Nothing may linger (and nothing may have been
            // parked — done was already 1 at finalizer time).
            let left = RESPONSES.with(|rs| rs.borrow().len());
            assert_eq!(left, 0, "finalized response must leave RESPONSES empty");

            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    // ── P1-5: a mid-body transport error must REJECT stream and body
    // consumers — never surface as a clean end-of-stream or a truncated
    // body resolve (silent truncation of provider JSON/SSE bodies).

    #[test]
    fn transfer_error_rejects_stream_and_body_consumers() {
        // Same three-connection choreography as
        // transfer_error_parks_gc_d_mid_transfer_response, minus the GC drop:
        //   conn 0 (/x):      headers + Content-Length 100, 7 bytes, HOLD.
        //   conn 1 (/signal): 200 then shutdown BOTH on conn 0 → curl ends /x
        //                     with a mid-body error (the completion ERROR path).
        //   conn 2 (/y):      plain 200 control.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut held_x: Option<std::net::TcpStream> = None;
            for (i, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { break };
                if i >= 3 {
                    break;
                }
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                match i {
                    0 => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial");
                        let _ = stream.flush();
                        held_x = Some(stream);
                    }
                    1 => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
                        let _ = stream.flush();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        /* EOF with 93 bytes unsent → curl error on /x. */
                        if let Some(mut sx) = held_x.take() {
                            let _ = sx.shutdown(std::net::Shutdown::Both);
                        }
                    }
                    _ => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
                        let _ = stream.flush();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                    }
                }
            }
        });

        let script = format!(
            "globalThis.state = '';
             globalThis.errmsg = '';
             (async function () {{
               try {{
                 const r = await fetch('{u}/x');
                 globalThis.hold = r;
                 globalThis.state += 'got-resp;';
                 fetch('{u}/signal');
                 const textP = r.text();  // settle waiter armed while in flight
                 try {{
                   for await (const chunk of r.body) {{ globalThis.state += 'chunk;'; }}
                   globalThis.state += 'clean-end;';
                 }} catch (e) {{
                   globalThis.errmsg = String(e && e.message || e);
                   globalThis.state += 'stream-err;';
                 }}
                 try {{ await textP; globalThis.state += 'text-resolved;'; }}
                 catch (e) {{ globalThis.state += 'text-err;'; }}
                 try {{ await r.arrayBuffer(); globalThis.state += 'ab-resolved;'; }}
                 catch (e) {{ globalThis.state += 'ab-err;'; }}
               }} catch (e) {{
                 globalThis.state += 'outer-err:' + (e && e.message || e) + ';';
               }}
               try {{
                 const r2 = await fetch('{u}/y');
                 globalThis.state += 'ctrl:' + (await r2.text()) + ';';
               }} catch (e) {{
                 globalThis.state += 'ctrl-err:' + (e && e.message || e) + ';';
               }}
             }})();",
            u = format!("http://127.0.0.1:{}", port)
        );

        let _loop_guard = crate::rt::test_loop_lock();
        unsafe {
            let rt = qjs::JS_NewRuntime();
            let ctx = qjs::JS_NewContext(rt);
            let _ctx_guard = CtxPtr::new(ctx);
            crate::rt::event_loop::sofuu_loop_init();
            mod_http_client_register(ctx);

            let script_c = CString::new(script).unwrap();
            let r = qjs::JS_Eval(ctx, script_c.as_ptr(), script_c.as_bytes().len(),
                                 c"<p1-5>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL);
            assert!(!qjs::is_exception(r), "script eval threw");
            qjs::sofuu_js_free_value(ctx, r);
            crate::rt::event_loop::sofuu_loop_run(ctx);

            let state = read_global(ctx, c"state");
            let errmsg = read_global(ctx, c"errmsg");
            assert!(
                state.contains("stream-err"),
                "for-await over an errored body must reject; got: {state}"
            );
            assert!(
                !state.contains("clean-end"),
                "stream must not end cleanly: {state}"
            );
            assert!(
                !errmsg.is_empty(),
                "rejection must carry a message; got: {state}"
            );
            assert!(
                state.contains("text-err"),
                "in-flight text() must reject: {state}"
            );
            assert!(
                !state.contains("text-resolved"),
                "text() must not resolve truncated: {state}"
            );
            assert!(
                state.contains("ab-err"),
                "post-error arrayBuffer() must reject: {state}"
            );
            assert!(
                state.contains("ctrl:ok"),
                "healthy control fetch must still resolve: {state}"
            );

            crate::rt::event_loop::sofuu_loop_close();
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }
}
