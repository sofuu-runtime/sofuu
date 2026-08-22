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
const FETCH_TIMEOUT_MS: c_long = 120_000; /* sane default; stalls no longer hang forever */

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
}

struct ResponseChunk {
    len: usize,
    data: Vec<u8>,
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
    pending_next: *mut PromiseHandle, /* one awaiting consumer at a time */
    pending_resolve: JSValue,         /* the resolver for the stream */
    pending_settle: *mut PromiseHandle, /* text()/json()/arrayBuffer() waiter */
    ctx: *mut JSContext,              /* the context that owns the response */
}

// ── Response finalizer ───────────────────────────────────────────────

unsafe extern "C" fn response_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let r = qjs::JS_GetOpaque(val, RESPONSE_CLASS_ID.load(std::sync::atomic::Ordering::Relaxed));
    if r.is_null() {
        return;
    }
    let r = r as *mut ResponseData;
    /* Unlink from the global list. */
    RESPONSES.with(|rs| {
        let mut rs = rs.borrow_mut();
        if let Some(i) = rs.iter().position(|x| *x == r) {
            rs.swap_remove(i);
        }
    });
    /* A still-pending settle waiter would leak — reject it. */
    if !(*r).pending_settle.is_null() {
        sofuu_promise_reject_str(
            (*r).pending_settle,
            c"response finalized".as_ptr(),
        );
        (*r).pending_settle = ptr::null_mut();
    }
    if !(*r).pending_next.is_null() {
        /* A pending stream waiter would leak too — reject it. */
        let p = (*r).pending_next;
        sofuu_promise_reject_str(p, c"response finalized".as_ptr());
        (*r).pending_next = ptr::null_mut();
    }
    drop(Box::from_raw(r));
}

static RESPONSE_CLASS_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// ── Response.text() / json() / arrayBuffer() ─────────────────────────

unsafe fn response_text_impl(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    let data = (*r).body.clone();
    let len = data.len();
    let c = CString::new(data).unwrap_or_default();
    let str_v = qjs::JS_NewStringLen(ctx, c.as_ptr(), len);

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
    let src = if (*r).body.is_empty() {
        CString::new("null").unwrap()
    } else {
        CString::new((*r).body.clone()).unwrap_or_default()
    };
    let parsed = qjs::JS_ParseJSON(ctx, src.as_ptr(), src.as_bytes().len(), c"<fetch response>".as_ptr());

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
    let len = (*r).body.len();
    let buf = qjs::js_malloc(qjs::JS_GetRuntime(ctx), if len > 0 { len } else { 1 });
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

    /* Case-insensitive scan through raw headers */
    let mut result = qjs::sofuu_js_null();
    let mut p: &[u8] = raw;

    while !p.is_empty() {
        /* Find end of this header line */
        let eol = p
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|i| i)
            .unwrap_or(p.len());

        /* Find the colon */
        if let Some(colon) = p[..eol].iter().position(|&b| b == b':') {
            let key_len = colon;
            if key_len == name.len() && p[..key_len].eq_ignore_ascii_case(name) {
                /* Found it — extract value */
                let mut vstart = colon + 1;
                while vstart < p.len() && p[vstart] == b' ' {
                    vstart += 1;
                }
                let mut vlen = eol - vstart;
                /* Trim trailing \r / space */
                while vlen > 0 && (p[vstart + vlen - 1] == b'\r' || p[vstart + vlen - 1] == b' ') {
                    vlen -= 1;
                }
                let c = CString::new(&p[vstart..vstart + vlen]).unwrap_or_default();
                result = qjs::JS_NewStringLen(ctx, c.as_ptr(), vlen);
                break;
            }
        }

        if eol < p.len() && p[eol] == b'\r' {
            p = &p[(eol + 2).min(p.len())..];
        } else {
            p = &p[(eol + 1).min(p.len())..];
        }
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

    /* Build Uint8Array(len) and fill via indexed properties. */
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
    let factory_c = CString::new(BODY_ITER_FACTORY).unwrap();
    let factory = qjs::JS_Eval(
        ctx,
        factory_c.as_ptr(),
        factory_c.as_bytes().len(),
        c"<body-iter>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    if qjs::is_exception(factory) {
        return qjs::sofuu_js_exception();
    }
    let iter = qjs::JS_Call(ctx, factory, qjs::sofuu_js_undefined(), 0, ptr::null());
    qjs::sofuu_js_free_value(ctx, factory);
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
        pending_next: ptr::null_mut(),
        pending_resolve: qjs::sofuu_js_undefined(),
        pending_settle: ptr::null_mut(),
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
    /* Keep the full body for text()/json()/arrayBuffer() — capped so a
     * misbehaving endpoint cannot exhaust memory. */
    if (*req).body.len() + n > BODY_CAP {
        return 0; /* hard cap; aborts the transfer */
    }
    (*req).body.extend_from_slice(std::slice::from_raw_parts(ptr as *const u8, n));

    /* Streaming: enqueue the chunk ONLY — never run JS here (we are inside
     * curl_multi_socket_action; the uv_check_t fulfils waiters in a safe
     * loop context after this callback returns). */
    if !(*req).resp.is_null() {
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

    let block = std::slice::from_raw_parts(ptr as *const u8, n);
    let is_header_line = !block.starts_with(b"HTTP/")
        && !block.starts_with(b":")
        && block.iter().any(|&b| b == b':');

    if is_header_line {
        let block = std::slice::from_raw_parts(ptr as *const u8, n);
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
     * ResponseData; later header lines append directly to it. */
    if (*req).resp.is_null() && is_header_line {
        let mut status: c_long = 0;
        curl::curl_easy_getinfo((*req).easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
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
    n
}

/// Enqueue a chunk (called from the curl write callback, loop thread).
unsafe fn response_enqueue(r: *mut ResponseData, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let c = Box::new(ResponseChunk {
        len: data.len(),
        data: data.to_vec(),
    });
    (*r).chunks.push_back(c.data);
    // NOTE: `len` mirrors the C chunk struct; kept for parity documentation.
    let _ = c.len;
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
    /* One pass: fulfill any response with queued chunks or done. */
    RESPONSES.with(|rs| {
        let rs = rs.borrow();
        for &r in rs.iter() {
            if !(*r).pending_next.is_null() && (!(*r).chunks.is_empty() || (*r).done != 0) {
                response_fulfill_pending((*r).ctx, r);
            }
            if !(*r).pending_settle.is_null() && (*r).done != 0 {
                let p = (*r).pending_settle;
                (*r).pending_settle = ptr::null_mut();
                sofuu_promise_resolve(p, qjs::sofuu_js_undefined());
            }
        }
    });
}

/// Return a promise that resolves (undefined) once the body transfer is
/// complete — text()/json()/arrayBuffer() await this before reading.
unsafe fn response_wait_done(ctx: *mut JSContext, r: *mut ResponseData) -> JSValue {
    if (*r).done != 0 || !(*r).pending_settle.is_null() {
        let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
        let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
        let ret = qjs::JS_Call(ctx, resolvers[0], qjs::sofuu_js_undefined(), 0, ptr::null());
        qjs::sofuu_js_free_value(ctx, ret);
        qjs::sofuu_js_free_value(ctx, resolvers[0]);
        qjs::sofuu_js_free_value(ctx, resolvers[1]);
        return promise;
    }
    let mut resolvers: [JSValue; 2] = [std::mem::zeroed(); 2];
    let promise = qjs::JS_NewPromiseCapability(ctx, resolvers.as_mut_ptr());
    let p = Box::into_raw(Box::new(PromiseHandle {
        ctx,
        resolve: qjs::sofuu_js_dup_value(ctx, resolvers[0]),
        reject: qjs::sofuu_js_dup_value(ctx, resolvers[1]),
    }));
    (*r).pending_settle = p;
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

        if !req.is_null() {
            let ctx = (*req).ctx;

            if return_code == curl::CURLE_OK {
                let mut status: c_long = 0;
                let mut final_url: *mut c_char = ptr::null_mut();
                curl::curl_easy_getinfo(easy, curl::CURLINFO_RESPONSE_CODE, &mut status as *mut c_long);
                curl::curl_easy_getinfo(easy, curl::CURLINFO_EFFECTIVE_URL, &mut final_url);

                if !(*req).resp.is_null() {
                    /* Streaming: the response already resolved on the first
                     * header line; transfer the buffered body + mark done.
                     * Pending waiters are fulfilled by the uv_check_t. */
                    let resp = (*req).resp;
                    (*resp).body = std::mem::take(&mut (*req).body);
                    (*resp).done = 1;
                    fetch_stream_pump();
                } else {
                    /* No header block (e.g. empty body) — build + resolve. */
                    let mut req_ref = &mut *req;
                    let response = build_response(ctx, &mut req_ref, status, final_url, true);
                    sofuu_promise_resolve((*req).promise, response);
                    qjs::sofuu_js_free_value(ctx, response);
                }
            } else {
                let err = qjs::JS_NewError(ctx);
                let msg_c = CString::new(CStr::from_ptr(curl::curl_easy_strerror(return_code)).to_bytes())
                    .unwrap_or_default();
                qjs::sofuu_js_set_property_str(ctx, err, c"message".as_ptr(), qjs::sofuu_js_new_string(ctx, msg_c.as_ptr()));
                if !(*req).resp.is_null() {
                    /* Transfer error: end the stream with done. */
                    (*(*req).resp).done = 1;
                    (*(*req).resp).errored = 1;
                    fetch_stream_pump();
                } else {
                    sofuu_promise_reject((*req).promise, err);
                }
            }

            if !(*req).req_headers.is_null() {
                curl::curl_slist_free_all((*req).req_headers);
            }
            drop(Box::from_raw(req));

            sofuu_flush_jobs(ctx);
        }

        curl::curl_multi_remove_handle(curl_multi(), easy);
        curl::curl_easy_cleanup(easy);
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

    let url_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if url_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    /* curl only delivers response headers to CURLOPT_HEADERFUNCTION when
     * CURLOPT_HEADER is set (curl >= 7.85 splits HTTP/2 pseudo-headers). */
    /* CURLOPT_HTTP_VERSION = 2 → CURL_HTTP_VERSION_1_1 (curl.h). */
    const CURLOPT_HTTP_VERSION: c_int = 84;
    const CURL_HTTP_VERSION_1_1: c_long = 2;
    const CURLOPT_ACCEPT_ENCODING: c_int = 10102;

    let mut req = Box::new(FetchReq {
        ctx,
        promise: ptr::null_mut(),
        body: Vec::new(),
        headers_raw: Vec::new(),
        resp: ptr::null_mut(),
        resp_cap: 0,
        easy: curl::curl_easy_init(),
        req_headers: ptr::null_mut(),
    });

    curl::curl_easy_setopt(req.easy, curl::CURLOPT_URL, url_ptr);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_WRITEFUNCTION, write_body_cb as *const c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_WRITEDATA, &mut *req as *mut FetchReq as *mut c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_HEADERFUNCTION, write_header_cb as *const c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_HEADERDATA, &mut *req as *mut FetchReq as *mut c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_PRIVATE, &mut *req as *mut FetchReq as *mut c_void);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_FOLLOWLOCATION as c_int, 1 as c_long);
    curl::curl_easy_setopt(req.easy, curl::CURLOPT_TIMEOUT_MS as c_int, FETCH_TIMEOUT_MS as c_long); /* no infinite wait */
    /* NOTE: CURLOPT_HEADER stays OFF — with it on, curl ALSO feeds response
     * headers to the body callback and text()/json() get "HTTP/1.1 200 OK"
     * prepended. The header callback captures real headers separately. */
    curl::curl_easy_setopt(req.easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1); /* avoid HTTP/2 pseudo-header splitting */
    curl::curl_easy_setopt(req.easy, CURLOPT_ACCEPT_ENCODING, ptr::null::<c_char>()); /* enable compressed transfers */

    qjs::sofuu_js_free_cstring(ctx, url_ptr);

    /* Parse options: { method, headers, body } */
    if argc > 1 && qjs::is_object(*argv.add(1)) {
        /* method */
        let method_val = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"method".as_ptr());
        if !qjs::is_undefined(method_val) {
            let method = qjs::sofuu_js_to_cstring(ctx, method_val);
            if !method.is_null() {
                curl::curl_easy_setopt(req.easy, curl::CURLOPT_CUSTOMREQUEST, method);
                qjs::sofuu_js_free_cstring(ctx, method);
            }
        }
        qjs::sofuu_js_free_value(ctx, method_val);

        /* headers */
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
                        req.req_headers = curl::curl_slist_append(req.req_headers, hc.as_ptr());
                        if req.req_headers.is_null() {
                            /* OOM — curl_slist_append returns NULL */
                        }
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
    req.promise = out;

    let req_ptr = Box::into_raw(req);
    curl::curl_multi_add_handle(curl_multi(), (*req_ptr).easy);

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
    qjs::sofuu_js_free_value(ctx, global2);
}

// CURLMOPT_* — FUNCTIONPOINT(20000) + enum index (multi.h: SOCKETFUNCTION=1,
// TIMERFUNCTION=4). Verified against the system SDK headers after a wrong
// guess silently disabled the socket/timer callbacks.
const CURLMOPT_SOCKETFUNCTION: c_int = 20001;
const CURLMOPT_TIMERFUNCTION: c_int = 20004;
