// rt/http_server.rs — Sofuu HTTP/1.1 server (PLAN-RUST-MIGRATION M5).
//
// Port of the deleted `src/http/server.c` (459 lines), semantics verbatim:
//   Sofuu.createServer(handler) → server with .listen(port [, host] [, cb])
//   sofuu.serve(port, handler)  → legacy
//   req: { method, url, body }  —  res: { writeHead, write, end, send }
// Requests parse through the vendored http-parser (stays C until M11); the
// response is a single buffered write with Content-Length, or chunked for
// streaming. The refcount/closing dance is preserved exactly: connection
// ref (1) + live res object ref (1); only ONE path may uv_close (a racing
// read-EOF would otherwise double-close and double-free).
//
// C symbols replaced: `mod_http_server_register` + the global
// `createServer` / `sofuu.createServer` / `sofuu.serve` surface.

use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use sofuu_ffi::http_parser::{self, HttpParser, HttpParserSettings};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvHandle, UvStream, UvTcp, UvTimer, UvWriteReq};

use crate::rt::event_loop::sofuu_loop_get;

const MAX_REQ_URL: usize = 8 * 1024;
const MAX_REQ_BODY: usize = 16 * 1024 * 1024;
const MAX_REQ_HEADERS: usize = 32 * 1024; /* total name+value bytes */
const MAX_REQ_HEADER_COUNT: usize = 128;
/* net-2: out-of-band response headers from writeHead, bounded like the
 * request side so a hostile handler can't balloon the header block. */
const MAX_RESP_HEADERS: usize = 64;
const MAX_RESP_HEADER_BYTES: usize = 16 * 1024;
/* AUDIT-2026-10-31 P1-11: req.on() listeners are owned JS references held
 * across the handler frame, so the count is bounded like every other
 * per-connection resource in this file. */
const MAX_REQ_LISTENERS: usize = 16;

/* net-3 (AUDIT-2026-09-07): connection deadlines. Headers carry an ABSOLUTE
 * deadline from accept — deliberately NOT reset by received bytes, so
 * byte-dribbling (slowloris) cannot extend it. Once headers complete the
 * handle switches to an idle deadline that any body byte or response write
 * resets. Tests shorten both via TimeoutGuard (tests run single-threaded
 * under TEST_LOOP_LOCK). */
static HEADERS_TIMEOUT_MS: AtomicU64 = AtomicU64::new(30_000);
static IDLE_TIMEOUT_MS: AtomicU64 = AtomicU64::new(75_000);

static RES_CLASS_ID: AtomicU32 = AtomicU32::new(0);
static SRV_CLASS_ID: AtomicU32 = AtomicU32::new(0);

// ── structs ─────────────────────────────────────────────────────────

struct Server {
    tcp: *mut UvTcp,
    ctx: *mut JSContext,
    handler: JSValue,
    listening: c_int, /* tcp handle was uv_tcp_init'd */
    /* Root for the JS server object while it is listening. Without it a
     * server created inside a function (module pattern, IIFE, any helper)
     * is collected as soon as the last JS reference goes away and its
     * finalizer closes the listener — the socket never accepts, silently.
     * Node keeps a live server alive the same way. */
    anchor: JSValue,
}

struct Client {
    tcp: *mut UvTcp,
    srv: *mut Server,
    parser: HttpParser,
    /* request accumulation */
    url: Vec<u8>,
    body: Vec<u8>,
    /* R-32: the route this connection served (the url as parsed, kept after
     * `url` is reset) plus its accept time, so a disconnect log can name
     * WHICH route died and how long it lived. */
    route: Vec<u8>,
    started: std::time::Instant,
    disconnect_logged: c_int,
    /* header accumulation (http-parser fragments names/values across reads) */
    hdr_field: Vec<u8>,
    hdr_value: Vec<u8>,
    headers: Vec<(Vec<u8>, Vec<u8>)>,
    hdr_bytes: usize,
    /* net-3: per-connection deadline timer (malloc'd uv_timer_t). Armed at
     * accept with the ABSOLUTE headers deadline; re-armed with the idle
     * deadline once headers complete / on body or response activity. */
    hdr_timer: *mut UvTimer,
    /* response state */
    status: c_int,
    ctype: RefCell<Vec<u8>>,
    /* net-2: writeHead headers beyond Content-Type (name, value) —
     * spliced into whichever header block the response emits. */
    out_headers: Vec<(Vec<u8>, Vec<u8>)>,
    headers_set: c_int,
    streaming: c_int, /* chunked response started */
    finished: c_int,  /* response ended — further res calls are no-ops */
    refcount: c_int,  /* connection (1) + live res JS object (1) */
    closing: c_int,   /* uv_close already requested — no double close */
    peer_closed: c_int, /* R2 (T2.4): the client went away (read EOF/error) */
    /* JS objects */
    req_val: Cell<JSValue>,
    res_val: Cell<JSValue>,
    /* AUDIT-2026-10-31 P1-11: req.on('data'|'end'|'close') listeners. The
     * request body is already buffered when the handler runs, so the events
     * are dispatched after the handler's SYNCHRONOUS frame (every listener it
     * registers is in place by then) and each reference is freed as it fires.
     * `handler_active` marks that frame; `ev_dispatching` keeps a listener
     * that ends the response from re-entering the dispatcher. */
    on_data: Vec<JSValue>,
    on_end: Vec<JSValue>,
    on_close: Vec<JSValue>,
    ev_data_sent: c_int,
    ev_end_sent: c_int,
    ev_close_sent: c_int,
    ev_dispatching: c_int,
    handler_active: c_int,
}

/// Zeroed JSValue for the initial cells (see rt/promise.rs convention).
const fn zero_jsvalue() -> JSValue {
    JSValue {
        u: sofuu_ffi::qjs::JSValueUnion { int32: 0 },
        tag: 0,
    }
}

impl Client {
    fn new(srv: *mut Server) -> Self {
        Client {
            tcp: ptr::null_mut(),
            srv,
            parser: HttpParser {
                bits0: 0,
                nread: 0,
                content_length: 0,
                http_major: 0,
                http_minor: 0,
                bits1: 0,
                data: ptr::null_mut(),
            },
            url: Vec::new(),
            body: Vec::new(),
            route: Vec::new(),
            started: std::time::Instant::now(),
            disconnect_logged: 0,
            hdr_field: Vec::new(),
            hdr_value: Vec::new(),
            headers: Vec::new(),
            hdr_bytes: 0,
            hdr_timer: ptr::null_mut(),
            status: 0,
            ctype: RefCell::new(Vec::new()),
            out_headers: Vec::new(),
            headers_set: 0,
            streaming: 0,
            finished: 0,
            refcount: 1,
            peer_closed: 0,
            closing: 0,
            req_val: Cell::new(zero_jsvalue()),
            res_val: Cell::new(zero_jsvalue()),
            on_data: Vec::new(),
            on_end: Vec::new(),
            on_close: Vec::new(),
            ev_data_sent: 0,
            ev_end_sent: 0,
            ev_close_sent: 0,
            ev_dispatching: 0,
            handler_active: 0,
        }
    }
}

/// Release one reference; free the client (and its buffers) at zero.
unsafe fn client_release(c: *mut Client) {
    if c.is_null() {
        return;
    }
    (*c).refcount -= 1;
    if (*c).refcount <= 0 {
        /* net-3: the deadline timer dies with the client. client_release is
         * the single zero-ref path, so no close path can leak or outlive it. */
        if !(*c).hdr_timer.is_null() {
            uv::uv_timer_stop((*c).hdr_timer);
            if uv::uv_is_closing((*c).hdr_timer as *const UvHandle) == 0 {
                uv::uv_close((*c).hdr_timer as *mut UvHandle, Some(timer_free_cb));
            } else {
                /* Teardown walk closed it with a None callback — we still
                 * own the malloc'd handle block. */
                libc::free((*c).hdr_timer as *mut c_void);
            }
            (*c).hdr_timer = ptr::null_mut();
        }
        libc::free((*c).tcp as *mut c_void);
        drop(Box::from_raw(c));
    }
}

unsafe extern "C" fn client_free(h: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(h);
    // SAFETY: the tcp handle's data points at the client (set at accept).
    let c = *(h as *mut *mut Client);
    client_release(c);
}

/// net-3: uv_close callback for the deadline timer — the handle block is
/// malloc'd storage we own.
unsafe extern "C" fn timer_free_cb(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    libc::free(handle as *mut c_void);
}

/// The route this connection served, shaped for a log line: the parsed url
/// with any query string dropped (it can carry tokens, ?token=…), `-` before
/// a url is known.
unsafe fn route_for_log(c: *mut Client) -> String {
    if (*c).route.is_empty() {
        return "-".to_string();
    }
    let raw = String::from_utf8_lossy(&(*c).route).into_owned();
    match raw.split_once('?') {
        Some((path, _)) => path.to_string(),
        None => raw,
    }
}

/// R-32: one INFO-level line per abandoned connection. Before this, a client
/// vanishing mid-response (or an SSE generation being cut) left nothing at
/// all in the log — the failure was indistinguishable from the server
/// hanging. The line carries what an operator needs: the ROUTE, the elapsed
/// milliseconds since accept, and whether a streaming generation was still
/// in flight (`res.closed()` true ⇒ the JS generation loop is cancelling).
unsafe fn log_client_gone(c: *mut Client, kind: &str) {
    if c.is_null() || (*c).disconnect_logged != 0 {
        return; /* one line per connection, whichever path notices first */
    }
    (*c).disconnect_logged = 1;
    let route = route_for_log(c);
    let elapsed_ms = (*c).started.elapsed().as_millis();
    let generation_cancelled = (*c).streaming != 0 && (*c).finished == 0;
    eprintln!(
        "[sofuu/http] info: client {kind} route={route} elapsed_ms={elapsed_ms} generation_cancelled={generation_cancelled}"
    );
}

/// net-3: (re)arm the per-connection deadline. Called on the loop thread
/// only (accept, parser callbacks, response writes).
unsafe fn client_arm_timer(c: *mut Client, ms: u64) {
    if ms == 0 {
        /* "No deadline" must also CLEAR a previously armed one: the accept
         * path arms HEADERS_TIMEOUT_MS, and the parser callbacks re-arm
         * with IDLE_TIMEOUT_MS once headers are in. Returning early left
         * the headers timer ticking — serve's idle-free long prefill would
         * have been killed by the (new) 30s accept deadline mid-completion. */
        if !(*c).hdr_timer.is_null() {
            uv::uv_timer_stop((*c).hdr_timer);
        }
        return;
    }
    if (*c).hdr_timer.is_null() {
        return; /* OOM at accept — run without a deadline, as before */
    }
    uv::uv_timer_start((*c).hdr_timer, Some(on_deadline_timeout), ms, 0);
}

/// net-3: a connection overstayed its deadline — silent or dribbling
/// headers, a stalled body, or no request/response activity. Close it; the
/// peer sees EOF instead of a held-open fd. The closing flag keeps this
/// safe against a racing on_read close (single-owner uv_close rule).
unsafe extern "C" fn on_deadline_timeout(_t: *mut UvTimer) {
    // SAFETY: the timer region's user slot holds the Client (set at accept,
    // mirroring the tcp handle slot).
    let c = *(_t as *mut *mut Client);
    if (*c).closing == 0 {
        /* R-32: the server gave up on this connection — log it like any
         * other abandonment (route, elapsed, generation cut). */
        log_client_gone(c, "deadline expired");
        /* AUDIT-2026-10-31 P1-11: drop req.on() references this connection
         * will never dispatch. */
        client_free_listeners(c);
        (*c).closing = 1;
        uv::uv_close((*c).tcp as *mut UvHandle, Some(client_free));
    }
}

fn status_text(status: c_int) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        // An unlisted code still needs A reason; "Unknown" no longer lies
        // about a 4xx/5xx the way a blanket "OK" did. (P2, audit 2026.)
        _ => "Unknown",
    }
}

// ── write request: heap-allocates the full response buffer ─────────

/// Write payload — owned by the Box, recovered from `uv_write_t.data`
/// (libuv reserves req->data for user state; the uv_write_t region itself
/// is a separate malloc at its true size — mirroring the C struct inside
/// the uv region corrupts it, because uv_write_t is larger than the mirror).
struct WriteReq {
    data: *mut c_char,
    client: *mut Client,
    close_after: c_int, /* close the connection once this write flushes */
}

unsafe extern "C" fn on_write_done(req: *mut UvWriteReq, status: c_int) {
    // SAFETY: req->data (first field of uv_req_t) was set at queue_write;
    // libuv reserves it for the user.
    let wr = *(req as *mut *mut WriteReq);
    let c = (*wr).client;
    let close_after = (*wr).close_after;
    libc::free((*wr).data as *mut c_void);
    libc::free(req as *mut c_void);
    drop(Box::from_raw(wr));
    if status < 0 {
        /* R-32: a failed write IS the abort surfacing (EPIPE/ECONNRESET) —
         * log it before the close fence runs. */
        log_client_gone(c, "write failed");
    }
    if close_after != 0 && (*c).closing == 0 {
        /* Only ONE path may own uv_close: a racing on_read(EOF/abort)
         * would otherwise double-close the handle and double-free. */
        /* AUDIT-2026-10-31 P1-11: a response that never dispatched its
         * listeners (failed write) releases them here. */
        client_free_listeners(c);
        (*c).closing = 1;
        uv::uv_close((*c).tcp as *mut UvHandle, Some(client_free));
    }
}

/// Queue a heap buffer for writing; takes ownership of `buf`.
unsafe fn queue_write(c: *mut Client, buf: *mut c_char, len: usize, close_after: c_int) {
    /* net-3: response activity resets the idle deadline (streaming writes
     * keep the connection provably alive). */
    if !(*c).hdr_timer.is_null() {
        client_arm_timer(c, IDLE_TIMEOUT_MS.load(Ordering::Relaxed));
    }
    let wr = Box::into_raw(Box::new(WriteReq {
        data: buf,
        client: c,
        close_after,
    }));
    let wreq = libc::malloc(uv::sofuu_uv_write_size()) as *mut UvWriteReq;
    if wreq.is_null() {
        // OOM: drop ownership cleanly instead of leaking buf/Box.
        libc::free(buf as *mut c_void);
        drop(Box::from_raw(wr));
        return;
    }
    *(wreq as *mut *mut WriteReq) = wr;
    let b = uv::UvBuf { base: buf, len };
    let rc = uv::uv_write(wreq, (*c).tcp as *mut UvStream, &b, 1, Some(on_write_done));
    if rc < 0 {
        // libuv never fires on_write_done on sync failure — free here.
        libc::free(buf as *mut c_void);
        libc::free(wreq as *mut c_void);
        drop(Box::from_raw(wr));
    }
}

/// net-2: the extra writeHead() header lines ("Name: value\r\n" each),
/// empty when none were set. Names/values were CR/LF-guarded at writeHead.
unsafe fn out_header_lines(c: *mut Client) -> String {
    let mut s = String::new();
    for (n, v) in (*c).out_headers.iter() {
        s.push_str(&String::from_utf8_lossy(n));
        s.push_str(": ");
        s.push_str(&String::from_utf8_lossy(v));
        s.push_str("\r\n");
    }
    s
}

/// Buffered single-shot response with Content-Length, then close.
unsafe fn send_response(c: *mut Client, status: c_int, ct: &str, body: &[u8]) {
    let hdr = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
        status,
        status_text(status),
        ct,
        body.len(),
        out_header_lines(c)
    );
    let hl = hdr.len();
    let buf = libc::malloc(hl + body.len()) as *mut c_char;
    std::ptr::copy_nonoverlapping(hdr.as_ptr(), buf as *mut u8, hl);
    if !body.is_empty() {
        std::ptr::copy_nonoverlapping(body.as_ptr(), buf.add(hl) as *mut u8, body.len());
    }
    queue_write(c, buf, hl + body.len(), 1);
}

/// Begin a chunked (Transfer-Encoding: chunked) streaming response.
unsafe fn res_start_chunked(c: *mut Client) {
    let st = if (*c).status > 0 { (*c).status } else { 200 };
    let ct = if !(*c).ctype.borrow().is_empty() {
        String::from_utf8_lossy(&(*c).ctype.borrow()).into_owned()
    } else {
        "text/plain".to_string()
    };
    let hdr = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n{}\r\n",
        st,
        status_text(st),
        ct,
        out_header_lines(c)
    );
    let hl = hdr.len();
    let buf = libc::malloc(hl) as *mut c_char;
    std::ptr::copy_nonoverlapping(hdr.as_ptr(), buf as *mut u8, hl);
    queue_write(c, buf, hl, 0);
    (*c).streaming = 1;
}

// ── res JS methods ──────────────────────────────────────────────────

unsafe extern "C" fn js_res_write_head(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let c = qjs::JS_GetOpaque(this_val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() || (*c).finished != 0 {
        return qjs::sofuu_js_undefined();
    }
    if argc >= 1 {
        qjs::JS_ToInt32(ctx, &mut (*c).status, *argv);
    }
    if argc >= 2 && qjs::is_object(*argv.add(1)) {
        let mut ctype: Option<Vec<u8>> = None;
        let mut extras: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut extra_bytes: usize = 0;
        /* net-2: enumerate ALL own enumerable string properties, not just
         * Content-Type — every other header was silently dropped from the
         * wire. Same guard class as the fetch path: a CR/LF in name or
         * value skips the header entirely (no injected/split headers),
         * and the count/byte caps mirror the request-side limits. */
        let hobj = *argv.add(1);
        let mut ptab: *mut qjs::JSPropertyEnum = ptr::null_mut();
        let mut plen: u32 = 0;
        if qjs::JS_GetOwnPropertyNames(
            ctx,
            &mut ptab,
            &mut plen,
            hobj,
            qjs::JS_GPN_STRING_MASK | qjs::JS_GPN_ENUM_ONLY,
        ) == 0
        {
            for i in 0..plen {
                let atom = (*ptab.add(i as usize)).atom;
                let key = qjs::JS_AtomToCString(ctx, atom);
                let val = qjs::sofuu_js_get_property(ctx, hobj, atom);
                let vs = qjs::sofuu_js_to_cstring(ctx, val);
                if !key.is_null() && !vs.is_null() {
                    let k = CStr::from_ptr(key).to_string_lossy();
                    let v = CStr::from_ptr(vs).to_string_lossy();
                    let is_ct = k.eq_ignore_ascii_case("content-type");
                    if is_ct {
                        /* Content-Type keeps the old single-value path
                         * (stripped + stored in ctype below). */
                        if ctype.is_none() {
                            let mut bytes = v.as_bytes().to_vec();
                            bytes.truncate(255);
                            ctype = Some(bytes);
                        }
                    } else if extras.len() < MAX_RESP_HEADERS
                        && extra_bytes + k.len() + v.len() <= MAX_RESP_HEADER_BYTES
                        && !k.contains(['\r', '\n'])
                        && !v.contains(['\r', '\n'])
                    {
                        extra_bytes += k.len() + v.len();
                        extras.push((k.as_bytes().to_vec(), v.as_bytes().to_vec()));
                    }
                }
                if !key.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, key);
                }
                if !vs.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, vs);
                }
                qjs::sofuu_js_free_value(ctx, val);
                qjs::JS_FreeAtom(ctx, atom);
            }
            qjs::js_free(ctx, ptab as *mut c_void);
        }
        if let Some(mut ct) = ctype {
            /* P2-38 (AUDIT-2026-09-01): the Content-Type is spliced into
             * the response header block by format! — CR/LF in it injects
             * arbitrary response headers (and response splitting with the
             * body under handler control). Same guard class the REQUEST
             * side and the fetch path already enforce. JS here is user
             * script (same trust as the process), so this is parity, not
             * a remote-input fix. */
            ct.retain(|b| *b != b'\r' && *b != b'\n');
            (*c).ctype.borrow_mut().clear();
            (*c).ctype.borrow_mut().extend_from_slice(&ct);
        }
        (*c).out_headers = extras;
    }
    (*c).headers_set = 1;
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_res_write(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let c = qjs::JS_GetOpaque(this_val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() || (*c).finished != 0 {
        return qjs::sofuu_js_new_bool(ctx, 1);
    }
    if (*c).streaming == 0 {
        res_start_chunked(c);
    }
    if argc >= 1 && !qjs::is_undefined(*argv) && !qjs::is_null(*argv) {
        let mut blen: usize = 0;
        let body = qjs::JS_ToCStringLen2(ctx, &mut blen, *argv, 0);
        if !body.is_null() && blen > 0 {
            let prefix = format!("{:x}\r\n", blen);
            let pl = prefix.len();
            let total = pl + blen + 2;
            let buf = libc::malloc(total) as *mut c_char;
            std::ptr::copy_nonoverlapping(prefix.as_ptr(), buf as *mut u8, pl);
            std::ptr::copy_nonoverlapping(body as *const u8, buf.add(pl) as *mut u8, blen);
            *buf.add(pl + blen) = b'\r' as c_char;
            *buf.add(pl + blen + 1) = b'\n' as c_char;
            queue_write(c, buf, total, 0);
        }
        if !body.is_null() {
            qjs::sofuu_js_free_cstring(ctx, body);
        }
    }
    qjs::sofuu_js_new_bool(ctx, 1)
}

/// R2 (T2.4): has the peer gone away? Read EOF/error, or the server already
/// tore the connection down. Lets a streaming handler stop writing into a
/// dead socket instead of buffering forever.
unsafe extern "C" fn js_res_closed(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let c = qjs::JS_GetOpaque(this_val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    let closed = !c.is_null() && ((*c).peer_closed != 0 || (*c).closing != 0);
    qjs::sofuu_js_new_bool(ctx, if closed { 1 } else { 0 })
}

unsafe extern "C" fn js_res_end(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let c = qjs::JS_GetOpaque(this_val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() || (*c).finished != 0 {
        return qjs::sofuu_js_undefined();
    }

    /* Streaming path: flush any trailing body as a chunk, then terminate. */
    if (*c).streaming != 0 {
        if argc >= 1 && !qjs::is_undefined(*argv) && !qjs::is_null(*argv) {
            js_res_write(ctx, this_val, argc, argv);
        }
        (*c).finished = 1;
        let term = libc::malloc(5) as *mut c_char;
        std::ptr::copy_nonoverlapping(b"0\r\n\r\n".as_ptr(), term as *mut u8, 5);
        queue_write(c, term, 5, 1);
        /* AUDIT-2026-10-31 P1-11: the response ended — fire req.on('close').
         * A response ended INSIDE the handler frame dispatches from
         * cb_message_complete instead, once that frame returns. */
        if (*c).handler_active == 0 {
            req_dispatch_pending(c);
        }
        return qjs::sofuu_js_undefined();
    }

    let mut body: &[u8] = &[];
    let mut tmp: *const c_char = ptr::null();
    if argc >= 1 && !qjs::is_undefined(*argv) && !qjs::is_null(*argv) {
        let mut blen: usize = 0;
        tmp = qjs::JS_ToCStringLen2(ctx, &mut blen, *argv, 0);
        if !tmp.is_null() {
            body = std::slice::from_raw_parts(tmp as *const u8, blen);
        }
    }
    let st = if (*c).status > 0 { (*c).status } else { 200 };
    let ct = if !(*c).ctype.borrow().is_empty() {
        String::from_utf8_lossy(&(*c).ctype.borrow()).into_owned()
    } else {
        "text/plain".to_string()
    };
    (*c).finished = 1;
    send_response(c, st, &ct, body);
    if !tmp.is_null() {
        qjs::sofuu_js_free_cstring(ctx, tmp);
    }
    /* AUDIT-2026-10-31 P1-11: the response ended — fire req.on('close') (see
     * the streaming path above for why the handler frame defers its own). */
    if (*c).handler_active == 0 {
        req_dispatch_pending(c);
    }
    qjs::sofuu_js_undefined()
}

unsafe extern "C" fn js_res_send(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let c = qjs::JS_GetOpaque(this_val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() {
        return qjs::sofuu_js_undefined();
    }
    if (*c).status == 0 {
        (*c).status = 200;
    }
    js_res_end(ctx, this_val, argc, argv)
}

// ── AUDIT-2026-10-31 P1-11: async-handler rejections + req.on() ─────

/// One log-safe line for a JS value: ToString with newlines collapsed. A
/// throwing `toString` must not leave a pending exception behind — it would
/// fire on the next, unrelated JS call.
unsafe fn value_one_line(ctx: *mut JSContext, v: JSValueConst) -> String {
    let s = qjs::JS_ToString(ctx, v);
    if qjs::is_exception(s) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        return "(unprintable)".to_string();
    }
    let p = qjs::sofuu_js_to_cstring(ctx, s);
    qjs::sofuu_js_free_value(ctx, s);
    if p.is_null() {
        return "(unprintable)".to_string();
    }
    // SAFETY: p is a NUL-terminated string owned by ctx, freed below.
    let out = CStr::from_ptr(p).to_string_lossy().replace(['\r', '\n'], " ");
    qjs::sofuu_js_free_cstring(ctx, p);
    out
}

/// AUDIT-2026-10-31 P1-11 (a): the rejection handler for ONE request.
///
/// JS_NewCFunctionData holds a strong reference to that request's response
/// object, so the client is resolved from the response's opaque at call time.
/// The binding is therefore per-request — a global "current response" would
/// answer the WRONG client under concurrent connections — and it cannot
/// outlive the client, because the res object itself holds a client reference.
unsafe extern "C" fn js_handler_rejected(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    if data.is_null() {
        return qjs::sofuu_js_undefined();
    }
    let res = *data;
    let c = qjs::JS_GetOpaque(res, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() {
        return qjs::sofuu_js_undefined();
    }
    let reason = if argc >= 1 {
        *argv
    } else {
        qjs::sofuu_js_undefined()
    };
    let msg = value_one_line(ctx, reason);
    let route = route_for_log(c);
    eprintln!("[sofuu/http] error: handler rejected route={route}: {msg}");
    /* Answer the client — but only while the response is still open, and never
     * with the error text (that is script-internal). res_end is the single
     * path that puts response bytes on the wire and it latches `finished`, so
     * a second rejection (or a late res.end) cannot send a second response. */
    fail_client(ctx, c, res);
    qjs::sofuu_js_undefined()
}

/// AUDIT-2026-10-31 P1-11: answer a handler failure with 500 — but only while
/// the response is still open, and never with the error text (that is
/// script-internal). Shared by the rejected-promise path and the synchronous
/// throw path so a handler that fails either way produces the same answer
/// instead of leaving the client to hit its own deadline.
unsafe fn fail_client(ctx: *mut JSContext, c: *mut Client, res: JSValueConst) {
    if (*c).finished != 0 || (*c).closing != 0 || (*c).peer_closed != 0 {
        return;
    }
    if (*c).streaming != 0 {
        /* The status line is already on the wire — all that is left is to
         * terminate the chunk stream instead of leaving the reader hung. */
        js_res_end(ctx, res, 0, ptr::null());
    } else {
        (*c).status = 500;
        (*c).ctype.borrow_mut().clear();
        (*c).ctype.borrow_mut().extend_from_slice(b"text/plain");
        let body = qjs::sofuu_js_new_string(ctx, c"Internal Server Error".as_ptr());
        js_res_end(ctx, res, 1, &body);
        qjs::sofuu_js_free_value(ctx, body);
    }
}

/// AUDIT-2026-10-31 P1-11 (a): a thenable returned by a handler gets a
/// rejection handler. Pre-fix the returned promise was freed untouched, so a
/// throw inside an `async` handler surfaced as a process-wide
/// "[sofuu] UnhandledPromiseRejection" and the client sat until its own
/// deadline. Attaching the handler also retracts that pending rejection (the
/// tracker reports is_handled), so nothing is printed twice.
unsafe fn attach_handler_rejection(ctx: *mut JSContext, c: *mut Client, ret: JSValueConst) {
    let then_fn = qjs::sofuu_js_get_property_str(ctx, ret, c"then".as_ptr());
    if qjs::is_exception(then_fn) {
        /* A throwing `then` getter: nothing attachable. Consume it. */
        qjs::js_std_dump_error(ctx);
        return;
    }
    if qjs::JS_IsFunction(ctx, then_fn) == 0 {
        qjs::sofuu_js_free_value(ctx, then_fn);
        return; /* plain (non-thenable) return value — nothing to wait on */
    }
    let bound: [JSValue; 1] = [(*c).res_val.get()];
    let handler = qjs::JS_NewCFunctionData(ctx, js_handler_rejected, 1, 0, 1, bound.as_ptr());
    if qjs::is_exception(handler) {
        qjs::js_std_dump_error(ctx);
        qjs::sofuu_js_free_value(ctx, then_fn);
        return;
    }
    let mut args: [JSValueConst; 2] = [qjs::sofuu_js_undefined(), handler];
    let chained = qjs::JS_Call(ctx, then_fn, ret, 2, args.as_mut_ptr());
    if qjs::is_exception(chained) {
        qjs::js_std_dump_error(ctx);
    }
    qjs::sofuu_js_free_value(ctx, chained);
    qjs::sofuu_js_free_value(ctx, handler);
    qjs::sofuu_js_free_value(ctx, then_fn);
}

/// Release every listener reference an event list still owns.
unsafe fn free_listeners(ctx: *mut JSContext, list: &mut Vec<JSValue>) {
    for v in list.drain(..) {
        qjs::sofuu_js_free_value(ctx, v);
    }
}

/// AUDIT-2026-10-31 P1-11 (b): drop req.on() listeners a dying connection will
/// never dispatch (peer gone, deadline, malformed request, failed write).
/// Loop-thread paths only: client_release can run from the res finalizer, and
/// freeing JSValues there (inside GC) is not allowed.
unsafe fn client_free_listeners(c: *mut Client) {
    if c.is_null() || (*c).srv.is_null() {
        return;
    }
    if (*c).on_data.is_empty() && (*c).on_end.is_empty() && (*c).on_close.is_empty() {
        return;
    }
    let ctx = (*(*c).srv).ctx;
    if ctx.is_null() {
        return;
    }
    free_listeners(ctx, &mut (*c).on_data);
    free_listeners(ctx, &mut (*c).on_end);
    free_listeners(ctx, &mut (*c).on_close);
}

/// AUDIT-2026-10-31 P1-11 (b): fire this connection's req.on() events —
/// 'data' once with the buffered body, then 'end'; 'close' once the response
/// ended. Called right after the handler's synchronous frame (so every
/// listener the handler registers is in place) and again when the response
/// ends. `ev_dispatching` keeps a listener that ends the response from
/// re-entering the dispatcher.
unsafe fn req_dispatch_pending(c: *mut Client) {
    if (*c).ev_dispatching != 0 || (*c).srv.is_null() {
        return;
    }
    let ctx = (*(*c).srv).ctx;
    if ctx.is_null() {
        return;
    }
    (*c).ev_dispatching = 1;
    if (*c).ev_data_sent == 0 {
        (*c).ev_data_sent = 1;
        (*c).ev_end_sent = 1;
        /* Node emits 'data' only when there IS a body. req.body carries the
         * raw bytes (net-1: length-based, NUL-safe); it is undefined when the
         * request had no body, so read it back off the live request object. */
        let body = qjs::sofuu_js_get_property_str(ctx, (*c).req_val.get(), c"body".as_ptr());
        let has_body = !qjs::is_undefined(body);
        for f in std::mem::take(&mut (*c).on_data) {
            if has_body {
                let r = qjs::JS_Call(ctx, f, qjs::sofuu_js_undefined(), 1, &body);
                if qjs::is_exception(r) {
                    qjs::js_std_dump_error(ctx);
                }
                qjs::sofuu_js_free_value(ctx, r);
            }
            qjs::sofuu_js_free_value(ctx, f);
        }
        qjs::sofuu_js_free_value(ctx, body);
        for f in std::mem::take(&mut (*c).on_end) {
            let r = qjs::JS_Call(ctx, f, qjs::sofuu_js_undefined(), 0, ptr::null());
            if qjs::is_exception(r) {
                qjs::js_std_dump_error(ctx);
            }
            qjs::sofuu_js_free_value(ctx, r);
            qjs::sofuu_js_free_value(ctx, f);
        }
    }
    if (*c).ev_close_sent == 0 && (*c).finished != 0 {
        (*c).ev_close_sent = 1;
        for f in std::mem::take(&mut (*c).on_close) {
            let r = qjs::JS_Call(ctx, f, qjs::sofuu_js_undefined(), 0, ptr::null());
            if qjs::is_exception(r) {
                qjs::js_std_dump_error(ctx);
            }
            qjs::sofuu_js_free_value(ctx, r);
            qjs::sofuu_js_free_value(ctx, f);
        }
    }
    (*c).ev_dispatching = 0;
}

/// AUDIT-2026-10-31 P1-11 (b): `req.on(type, fn)` — Node's echo pattern
/// (`req.on('data', …)` / `req.on('end', …)`) used to be a TypeError here,
/// because the request object exposed url/method/headers/body only. Minimal,
/// over the already-buffered body: 'data' fires once with req.body, then
/// 'end'; 'close' fires when the response ends. Returns the request
/// (chainable), like Node. Bound to this request's response object, so the
/// binding is per-request.
unsafe extern "C" fn js_req_on(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
    _magic: c_int,
    data: *mut JSValue,
) -> JSValue {
    if data.is_null() || argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"req.on(type, fn): two arguments required".as_ptr());
    }
    let c = qjs::JS_GetOpaque(*data, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    if c.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"req.on: the request is no longer live".as_ptr());
    }
    if argc < 2 || qjs::JS_IsFunction(ctx, *argv.add(1)) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"req.on: listener must be a function".as_ptr());
    }
    let name_p = qjs::sofuu_js_to_cstring(ctx, *argv);
    if name_p.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"req.on: event name must be a string".as_ptr());
    }
    // SAFETY: NUL-terminated string owned by ctx, freed below.
    let name = CStr::from_ptr(name_p).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, name_p);
    let slot = match name.as_str() {
        "data" => &mut (*c).on_data,
        "end" => &mut (*c).on_end,
        "close" => &mut (*c).on_close,
        _ => {
            /* Loud, not silently ignored: a typo'd event name must not look
             * like a working subscription for the rest of the process's life. */
            return qjs::JS_ThrowTypeError(
                ctx,
                c"req.on: only 'data', 'end' and 'close' are supported".as_ptr(),
            );
        }
    };
    /* 'data'/'end' fire once, as soon as the handler's synchronous frame
     * returns (the body is already buffered). A listener registered LATER —
     * after an `await`, say — would never be called, so say so now instead of
     * leaving the script waiting on an event that already happened. */
    if name != "close" && (*c).ev_data_sent != 0 {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"req.on: 'data'/'end' already fired (the body is buffered in req.body)".as_ptr(),
        );
    }
    if slot.len() >= MAX_REQ_LISTENERS {
        return qjs::JS_ThrowRangeError(ctx, c"req.on: too many listeners".as_ptr());
    }
    slot.push(qjs::sofuu_js_dup_value(ctx, *argv.add(1)));
    /* Node returns the emitter, so `req.on(...).on(...)` chains. */
    qjs::sofuu_js_dup_value(ctx, this_val)
}

// ── http_parser callbacks ───────────────────────────────────────────

/// Commits the pending (field, value) pair if one is complete.
unsafe fn client_commit_header(c: *mut Client) -> c_int {
    if (*c).hdr_field.is_empty() {
        return 0;
    }
    if (*c).headers.len() + 1 > MAX_REQ_HEADER_COUNT {
        return 1; /* → parser error → close */
    }
    let f = std::mem::take(&mut (*c).hdr_field);
    let v = std::mem::take(&mut (*c).hdr_value);
    (*c).headers.push((f, v));
    0
}

unsafe extern "C" fn cb_header_field(p: *mut HttpParser, at: *const c_char, len: usize) -> c_int {
    let c = (*p).data as *mut Client;
    /* A new field name starting means the previous pair is complete. */
    if !(*c).hdr_value.is_empty() {
        let rc = client_commit_header(c);
        if rc != 0 {
            return rc;
        }
    }
    if (*c).hdr_bytes + len > MAX_REQ_HEADERS {
        return 1; /* cap, no OOM */
    }
    (*c).hdr_bytes += len;
    (*c).hdr_field.extend_from_slice(std::slice::from_raw_parts(at as *const u8, len));
    0
}

unsafe extern "C" fn cb_header_value(p: *mut HttpParser, at: *const c_char, len: usize) -> c_int {
    let c = (*p).data as *mut Client;
    if (*c).hdr_bytes + len > MAX_REQ_HEADERS {
        return 1; /* cap, no OOM */
    }
    (*c).hdr_bytes += len;
    (*c).hdr_value.extend_from_slice(std::slice::from_raw_parts(at as *const u8, len));
    0
}

unsafe extern "C" fn cb_headers_complete(p: *mut HttpParser) -> c_int {
    let c = (*p).data as *mut Client;
    /* Commit the final pending pair. Return 0 = keep parsing the body. */
    let rc = client_commit_header(c);
    if rc == 0 {
        /* net-3: headers arrived — the slowloris window closes; switch to
         * the idle deadline (a dribbling/stalled body now trips it). */
        client_arm_timer(c, IDLE_TIMEOUT_MS.load(Ordering::Relaxed));
    }
    rc
}

unsafe extern "C" fn cb_url(p: *mut HttpParser, at: *const c_char, len: usize) -> c_int {
    let c = (*p).data as *mut Client;
    if (*c).url.len() + len + 1 > MAX_REQ_URL {
        return 1; /* → parser error → close */
    }
    (*c).url.extend_from_slice(std::slice::from_raw_parts(at as *const u8, len));
    0
}

unsafe extern "C" fn cb_body(p: *mut HttpParser, at: *const c_char, len: usize) -> c_int {
    let c = (*p).data as *mut Client;
    /* net-3: body progress resets the idle deadline. */
    client_arm_timer(c, IDLE_TIMEOUT_MS.load(Ordering::Relaxed));
    if (*c).body.len() + len + 1 > MAX_REQ_BODY {
        return 1; /* cap, no OOM */
    }
    (*c).body.extend_from_slice(std::slice::from_raw_parts(at as *const u8, len));
    0
}

unsafe extern "C" fn cb_message_complete(p: *mut HttpParser) -> c_int {
    let c = (*p).data as *mut Client;
    let ctx = (*(*c).srv).ctx;

    /* R-32: keep the route for the disconnect log — `url` is reset at the
     * end of this call, so a later EOF would have nothing to name. */
    (*c).route = (*c).url.clone();

    (*c).req_val.set(qjs::sofuu_js_new_object(ctx));
    let method_c = CString::new(
        CStr::from_ptr(http_parser::http_method_str((*p).method())).to_bytes(),
    )
    .unwrap_or_default();
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).req_val.get(),
        c"method".as_ptr(),
        qjs::sofuu_js_new_string(ctx, method_c.as_ptr()),
    );
    let url_c = if (*c).url.is_empty() {
        CString::new("/").unwrap()
    } else {
        CString::new((*c).url.clone()).unwrap_or_default()
    };
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).req_val.get(),
        c"url".as_ptr(),
        qjs::sofuu_js_new_string(ctx, url_c.as_ptr()),
    );
    if !(*c).body.is_empty() {
        /* net-1: length-based API — pass raw bytes; a CString detour
         * truncated bodies at their first NUL byte. */
        qjs::sofuu_js_set_property_str(
            ctx,
            (*c).req_val.get(),
            c"body".as_ptr(),
            qjs::JS_NewStringLen(
                ctx,
                (*c).body.as_ptr() as *const c_char,
                (*c).body.len(),
            ),
        );
    }
    if !(*c).headers.is_empty() {
        /* req.headers: lowercased names; duplicates comma-joined (HTTP list
         * semantics). The brain server's bearer auth reads
         * req.headers.authorization. */
        let hdr_obj = qjs::sofuu_js_new_object(ctx);
        let mut merged: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();
        for (name, value) in (*c).headers.iter() {
            let mut lower = name.clone();
            lower.make_ascii_lowercase();
            if let Some(entry) = merged.iter_mut().find(|(n, _)| *n == lower) {
                entry.1.push(value.clone());
            } else {
                merged.push((lower, vec![value.clone()]));
            }
        }
        for (name, values) in merged {
            let joined = values.join(&b", "[..]);
            let name_c = CString::new(name).unwrap_or_default();
            let val_c = CString::new(joined).unwrap_or_default();
            let val_v = qjs::JS_NewStringLen(ctx, val_c.as_ptr(), val_c.as_bytes().len());
            qjs::sofuu_js_set_property_str(ctx, hdr_obj, name_c.as_ptr(), val_v);
        }
        qjs::sofuu_js_set_property_str(ctx, (*c).req_val.get(), c"headers".as_ptr(), hdr_obj);
    }
    /* Connections are Connection: close, but reset anyway in case that
     * ever changes (keep-alive would otherwise append to stale state). */
    (*c).url.clear();
    (*c).body.clear();
    (*c).headers.clear();
    (*c).hdr_bytes = 0;

    (*c).res_val.set(qjs::JS_NewObjectClass(ctx, RES_CLASS_ID.load(Ordering::Relaxed) as c_int));
    qjs::JS_SetOpaque((*c).res_val.get(), c as *mut c_void);
    (*c).refcount += 1; /* res JS object now holds a reference to the client */
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).res_val.get(),
        c"writeHead".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_res_write_head, c"writeHead".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).res_val.get(),
        c"write".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_res_write, c"write".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).res_val.get(),
        c"end".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_res_end, c"end".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).res_val.get(),
        c"send".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_res_send, c"send".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        (*c).res_val.get(),
        c"closed".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_res_closed, c"closed".as_ptr(), 0),
    );

    /* AUDIT-2026-10-31 P1-11 (b): req.on(type, fn) for Node's echo pattern.
     * Bound (JS_NewCFunctionData) to THIS request's response object,
     * so the binding is per-request — never a global "current request". */
    let bound: [JSValue; 1] = [(*c).res_val.get()];
    let on_fn = qjs::JS_NewCFunctionData(ctx, js_req_on, 2, 0, 1, bound.as_ptr());
    if qjs::is_exception(on_fn) {
        qjs::js_std_dump_error(ctx);
    } else {
        qjs::sofuu_js_set_property_str(ctx, (*c).req_val.get(), c"on".as_ptr(), on_fn);
    }

    (*c).handler_active = 1;
    let mut args: [JSValueConst; 2] = [(*c).req_val.get(), (*c).res_val.get()];
    let ret = qjs::JS_Call(ctx, (*(*c).srv).handler, qjs::sofuu_js_undefined(), 2, args.as_mut_ptr());
    (*c).handler_active = 0;
    if qjs::is_exception(ret) {
        qjs::js_std_dump_error(ctx);
        /* AUDIT-2026-10-31 P1-11: a handler that throws synchronously only
         * dumped the error and let the client time out. Same 500 as a
         * rejected promise. */
        fail_client(ctx, c, (*c).res_val.get());
    } else if qjs::is_object(ret) {
        /* AUDIT-2026-10-31 P1-11 (a): an `async` handler returns a promise.
         * Pre-fix it was freed without a rejection handler, so a throw inside
         * became a process-wide UnhandledPromiseRejection and this client hung
         * until its own deadline. */
        attach_handler_rejection(ctx, c, ret);
    }
    qjs::sofuu_js_free_value(ctx, ret);
    /* AUDIT-2026-10-31 P1-11 (b): every listener the handler registers is in
     * place once its synchronous frame returns — fire data → end here, while
     * req_val (and the body it carries) is still alive. */
    req_dispatch_pending(c);
    qjs::sofuu_js_free_value(ctx, (*c).req_val.replace(qjs::sofuu_js_undefined()));
    qjs::sofuu_js_free_value(ctx, (*c).res_val.replace(qjs::sofuu_js_undefined()));
    0
}

thread_local! {
    // Port of the C static `http_parser_settings g_settings`.
    static SETTINGS: HttpParserSettings = HttpParserSettings {
        on_message_begin: None,
        on_url: Some(cb_url),
        on_status: None,
        on_header_field: Some(cb_header_field),
        on_header_value: Some(cb_header_value),
        on_headers_complete: Some(cb_headers_complete),
        on_body: Some(cb_body),
        on_message_complete: Some(cb_message_complete),
        on_chunk_header: None,
        on_chunk_complete: None,
    };
}

// ── libuv callbacks ─────────────────────────────────────────────────

unsafe extern "C" fn on_alloc(_h: *mut UvHandle, sz: usize, b: *mut uv::UvBuf) {
    // SAFETY: libuv-provided buffer slot.
    (*b).base = libc::malloc(sz) as *mut c_char;
    (*b).len = (*b).base.is_null().then_some(0).unwrap_or(sz);
}

unsafe extern "C" fn on_read(s: *mut UvStream, n: isize, b: *const uv::UvBuf) {
    // SAFETY: the client tcp handle's data points at the client.
    let c = *(s as *mut *mut Client);
    if n > 0 {
        let parsed = http_parser::http_parser_execute(
            &mut (*c).parser,
            SETTINGS.with(|st| st as *const HttpParserSettings),
            (*b).base,
            n as usize,
        );
        if parsed < n as usize || (*c).parser.http_errno() != http_parser::HPE_OK {
            /* Malformed, oversized, or half-parsed request: close instead
             * of holding the fd + buffers forever. */
            if (*c).closing == 0 {
                /* AUDIT-2026-10-31 P1-11: malformed/oversized request — no
                 * response will ever dispatch this connection's listeners. */
                client_free_listeners(c);
                (*c).closing = 1;
                uv::uv_close(s as *mut UvHandle, Some(client_free));
            }
        }
    } else if n < 0 {
        /* R2 (T2.4): read EOF/error = the peer went away. Recorded so the
         * handler can observe the disconnect instead of writing into a
         * dead socket forever. */
        (*c).peer_closed = 1;
        /* R-32: say so at info level — route, elapsed ms, and whether a
         * streaming generation (SSE/chunked) is being cancelled. */
        let kind = if (*c).streaming != 0 && (*c).finished == 0 {
            "stream aborted"
        } else {
            "disconnected"
        };
        log_client_gone(c, kind);
        if (*c).closing == 0 {
            /* AUDIT-2026-10-31 P1-11: the peer is gone — release any
             * req.on() listener references still held for this connection. */
            client_free_listeners(c);
            (*c).closing = 1;
            uv::uv_close(s as *mut UvHandle, Some(client_free));
        }
    }
    if !(*b).base.is_null() {
        libc::free((*b).base as *mut c_void);
    }
}

unsafe extern "C" fn on_connection(s: *mut UvStream, status: c_int) {
    if status < 0 {
        return;
    }
    // SAFETY: the listen handle's data points at the server.
    let srv = *(s as *mut *mut Server);

    let c = Box::into_raw(Box::new(Client::new(srv)));
    (*c).refcount = 1; /* connection reference */
    let tcp = libc::malloc(uv::sofuu_uv_tcp_size()) as *mut UvTcp;
    (*c).tcp = tcp;
    uv::uv_tcp_init(sofuu_loop_get(), tcp);
    http_parser::http_parser_init(&mut (*c).parser, http_parser::HTTP_REQUEST);
    (*c).parser.data = c as *mut c_void;
    *(tcp as *mut *mut Client) = c; /* for on_read / client_free */
    /* net-3: arm the absolute headers deadline right at accept. The timer
     * carries the client in its data slot like the tcp handle does. */
    let timer = libc::malloc(uv::sofuu_uv_timer_size()) as *mut UvTimer;
    (*c).hdr_timer = timer;
    if !timer.is_null() {
        uv::uv_timer_init(sofuu_loop_get(), timer);
        *(timer as *mut *mut Client) = c; /* for on_deadline_timeout */
        client_arm_timer(c, HEADERS_TIMEOUT_MS.load(Ordering::Relaxed));
    }
    /* F-2: both handles are armed per-ctx (the connection's engine) — a
     * multi-engine teardown must close them. The deadline timer is tracked
     * with a None close cb: its own lifecycle closes it via timer_free_cb. */
    unsafe {
        crate::rt::event_loop::track_handle((*srv).ctx, tcp as *mut UvHandle, Some(client_free));
        if !timer.is_null() {
            crate::rt::event_loop::track_handle((*srv).ctx, timer as *mut UvHandle, None);
        }
    }
    if uv::uv_accept(s, tcp as *mut UvStream) == 0 {
        uv::uv_read_start(tcp as *mut UvStream, Some(on_alloc), Some(on_read));
    } else {
        uv::uv_close(tcp as *mut UvHandle, Some(client_free));
    }
}

// ── Start listening ─────────────────────────────────────────────────

unsafe fn do_listen(srv: *mut Server, host: *const c_char, port: c_int) -> c_int {
    let tcp = libc::malloc(uv::sofuu_uv_tcp_size()) as *mut UvTcp;
    (*srv).tcp = tcp;
    uv::uv_tcp_init(sofuu_loop_get(), tcp);
    (*srv).listening = 1; /* tcp handle is now initialized (finalizer may close it) */
    *(tcp as *mut *mut Server) = srv; /* for on_connection */
    /* F-2: the listening socket is armed per-ctx — a multi-engine teardown
     * must close it. Every close path routes through srv_close_cb. */
    unsafe {
        crate::rt::event_loop::track_handle((*srv).ctx, tcp as *mut UvHandle, Some(srv_close_cb))
    };

    /* P1 (audit 2026-10-14): resolve the bind host BEFORE touching libuv.
     * The old code ignored uv_ip4_addr's return — and a failed parse
     * ZEROES the sockaddr, so any host that is not a numeric IPv4 address
     * ("localhost", a typo, a DNS name) silently bound 0.0.0.0 while the
     * CLI treated "localhost" as local-only. Only the two documented
     * shapes are accepted; everything else fails CLOSED (never listen, so
     * nothing is exposed), and the caller reports the libuv error. */
    let host_str = if host.is_null() || *host == 0 {
        "0.0.0.0".to_string()
    } else {
        std::ffi::CStr::from_ptr(host).to_string_lossy().into_owned()
    };
    let bind_ip: std::borrow::Cow<'_, str> = match host_str.as_str() {
        "localhost" => std::borrow::Cow::Borrowed("127.0.0.1"),
        "" => std::borrow::Cow::Borrowed("0.0.0.0"),
        other => std::borrow::Cow::Borrowed(other),
    };
    let cbind = match std::ffi::CString::new(bind_ip.as_bytes()) {
        Ok(c) => c,
        Err(_) => return libc::EINVAL as c_int,
    };
    let mut a: uv::SockaddrIn = std::mem::zeroed();
    let parsed = uv::uv_ip4_addr(cbind.as_ptr(), port, &mut a);
    if parsed != 0 {
        eprintln!(
            "[sofuu/http] cannot bind host '{host_str}' (only numeric IPv4 addresses and 'localhost' are supported) — refusing to listen on every interface"
        );
        return parsed;
    }
    if uv::uv_tcp_bind(tcp, &a as *const _ as *const libc::sockaddr, 0) != 0 {
        return -1;
    }
    uv::uv_listen(tcp as *mut UvStream, 128, Some(on_connection))
}

// ── server.listen(port [, host] [, cb]) ─────────────────────────────

unsafe extern "C" fn js_server_listen(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let srv = qjs::JS_GetOpaque(this_val, SRV_CLASS_ID.load(Ordering::Relaxed)) as *mut Server;
    if srv.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"server.listen: bad object".as_ptr());
    }
    let mut port: i32 = 3000;
    if argc >= 1 {
        qjs::JS_ToInt32(ctx, &mut port, *argv);
    }

    /* Node-style overloads: listen(port, cb), listen(port, host),
       listen(port, host, cb). A string arg is the bind host. */
    let mut host: *const c_char = c"0.0.0.0".as_ptr();
    let mut host_tmp: *const c_char = ptr::null();
    let mut cb = qjs::sofuu_js_undefined();
    for i in 1..argc {
        let a = *argv.add(i as usize);
        if qjs::sofuu_js_is_string(a) != 0 {
            let h = qjs::sofuu_js_to_cstring(ctx, a);
            if !h.is_null() {
                host_tmp = h;
                host = h;
            }
        } else if qjs::JS_IsFunction(ctx, a) != 0 {
            cb = a;
        }
    }

    let r = do_listen(srv, host, port as c_int);
    if !host_tmp.is_null() {
        qjs::sofuu_js_free_cstring(ctx, host_tmp);
    }
    if r < 0 {
        return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), uv::uv_strerror(r));
    }
    /* Keep the server object alive for as long as it is listening; see the
     * anchor field. Idempotent, so a second listen() does not double-hold. */
    if qjs::is_undefined((*srv).anchor) {
        (*srv).anchor = qjs::sofuu_js_dup_value(ctx, this_val);
    }
    /* Node semantics: a listening server keeps the process alive, so
     * `createServer(h).listen(3000)` at the end of a script actually serves.
     * (AUDIT-2026-10-31 P0: this used to uv_unref() for everyone, which made
     * the README's headline example exit before it could accept a connection,
     * and silently killed any server written in Sofuu JS — including this
     * repo's own MCP HTTP fixture.)
     *
     * An EMBEDDED host drives the loop itself and must not be pinned by a
     * listener it did not ask for, so it keeps the old unref'd behaviour.
     * server.unref() is the opt-out for everyone else. */
    if (*srv).listening != 0 {
        /* A harness that drives servers as fixtures (start it, finish the
         * script, expect the process to end) declares that with
         * SOFUU_SERVERS_UNREF=1 rather than losing the Node default for
         * everyone. */
        let unref_only = crate::embed_config::is_embedded()
            || std::env::var_os("SOFUU_SERVERS_UNREF").is_some();
        if unref_only {
            uv::uv_unref((*srv).tcp as *mut UvHandle);
        } else {
            uv::uv_ref((*srv).tcp as *mut UvHandle);
        }
    }
    if qjs::JS_IsFunction(ctx, cb) != 0 {
        let ret = qjs::JS_Call(ctx, cb, qjs::sofuu_js_undefined(), 0, ptr::null());
        qjs::sofuu_js_free_value(ctx, ret);
    }
    // SAFETY: this_val is a dup'd ref (JS_Call does not consume).
    qjs::sofuu_js_dup_value(ctx, this_val)
}

// ── createServer(handler) ───────────────────────────────────────────

/// net-3: an embedder (e.g. `serve --infer`, whose local-inference prefill
/// can legitimately stay silent for many minutes on large models) may widen
/// the connection deadlines once, before the first server is created, via
/// SOFUU_HTTP_HEADERS_MS / SOFUU_HTTP_IDLE_MS (milliseconds, "0" = never).
/// Defaults stay 30000/75000 for every other embedding, so the slowloris
/// posture is unchanged.
fn apply_deadline_env_once() {
    static DONE: std::sync::Once = std::sync::Once::new();
    DONE.call_once(|| {
        if let Ok(v) = std::env::var("SOFUU_HTTP_HEADERS_MS") {
            if let Ok(ms) = v.trim().parse::<u64>() {
                HEADERS_TIMEOUT_MS.store(ms, Ordering::Relaxed);
            }
        }
        if let Ok(v) = std::env::var("SOFUU_HTTP_IDLE_MS") {
            if let Ok(ms) = v.trim().parse::<u64>() {
                IDLE_TIMEOUT_MS.store(ms, Ordering::Relaxed);
            }
        }
    });
}

unsafe extern "C" fn js_create_server(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    apply_deadline_env_once();
    if argc < 1 || qjs::JS_IsFunction(ctx, *argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"createServer: function required".as_ptr());
    }
    let srv = Box::into_raw(Box::new(Server {
        tcp: ptr::null_mut(),
        ctx,
        handler: qjs::sofuu_js_dup_value(ctx, *argv),
        listening: 0,
        anchor: qjs::sofuu_js_undefined(),
    }));
    let obj = qjs::JS_NewObjectClass(ctx, SRV_CLASS_ID.load(Ordering::Relaxed) as c_int);
    qjs::JS_SetOpaque(obj, srv as *mut c_void);
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"listen".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_server_listen, c"listen".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"ref".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_server_ref, c"ref".as_ptr(), 0),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"unref".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_server_unref, c"unref".as_ptr(), 0),
    );
    obj
}

/* server.ref() / server.unref(): whether the listening socket alone keeps the
 * event loop alive. listen() refs by default (Node semantics) outside an
 * embedded host, so a server keeps serving after the script's top level
 * finishes; unref() lets a script exit while the socket stays bound. Both
 * return the server (chainable). */
unsafe extern "C" fn js_server_ref(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let srv = qjs::JS_GetOpaque(this_val, SRV_CLASS_ID.load(Ordering::Relaxed)) as *mut Server;
    if srv.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"server.ref: bad object".as_ptr());
    }
    if (*srv).listening != 0 {
        uv::uv_ref((*srv).tcp as *mut UvHandle);
    }
    qjs::sofuu_js_dup_value(ctx, this_val)
}

unsafe extern "C" fn js_server_unref(
    ctx: *mut JSContext,
    this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let srv = qjs::JS_GetOpaque(this_val, SRV_CLASS_ID.load(Ordering::Relaxed)) as *mut Server;
    if srv.is_null() {
        return qjs::JS_ThrowTypeError(ctx, c"server.unref: bad object".as_ptr());
    }
    if (*srv).listening != 0 {
        uv::uv_unref((*srv).tcp as *mut UvHandle);
    }
    qjs::sofuu_js_dup_value(ctx, this_val)
}

// ── Legacy sofuu.serve(port, handler) ───────────────────────────────

unsafe extern "C" fn js_serve(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    apply_deadline_env_once();
    if argc < 2 || qjs::JS_IsFunction(ctx, *argv.add(1)) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"serve(port, fn)".as_ptr());
    }
    let mut port: i32 = 0;
    qjs::JS_ToInt32(ctx, &mut port, *argv);
    let srv = Box::into_raw(Box::new(Server {
        tcp: ptr::null_mut(),
        ctx,
        handler: qjs::sofuu_js_dup_value(ctx, *argv.add(1)),
        listening: 0,
        anchor: qjs::sofuu_js_undefined(),
    }));
    let r = do_listen(srv, c"0.0.0.0".as_ptr(), port as c_int);
    if r < 0 {
        /* P3 (AUDIT-2026-09-07): the old error path just dropped the boxed
         * Server and returned — leaking the dup'd handler JSValue (Box drop
         * knows nothing about JSValues; srv_finalizer frees it properly on
         * the success path), the malloc'd tcp storage, and the initialized-
         * but-never-closed uv handle do_listen left behind. */
        qjs::sofuu_js_free_value((*srv).ctx, (*srv).handler);
        (*srv).handler = qjs::sofuu_js_undefined();
        if !(*srv).tcp.is_null() {
            uv::uv_close((*srv).tcp as *mut UvHandle, Some(srv_close_cb));
            (*srv).tcp = ptr::null_mut();
        }
        drop(Box::from_raw(srv));
        return qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), uv::uv_strerror(r));
    }
    qjs::sofuu_js_undefined()
}

// ── Class defs ──────────────────────────────────────────────────────

unsafe extern "C" fn res_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    // SAFETY: the opaque is a client pointer (or null).
    let c = qjs::JS_GetOpaque(val, RES_CLASS_ID.load(Ordering::Relaxed)) as *mut Client;
    client_release(c);
}

unsafe extern "C" fn srv_close_cb(handle: *mut UvHandle) {
    crate::rt::event_loop::untrack_handle(handle);
    // SAFETY: handle->data holds the Server* (set at do_listen via tcp slot).
    // Free the listening tcp storage that do_listen malloc'd.
    libc::free(handle as *mut c_void);
}

unsafe extern "C" fn srv_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let srv = qjs::JS_GetOpaque(val, SRV_CLASS_ID.load(Ordering::Relaxed)) as *mut Server;
    if srv.is_null() {
        return;
    }
    if (*srv).listening != 0
        && !(*srv).tcp.is_null()
        && uv::uv_is_closing((*srv).tcp as *const UvHandle) == 0
    {
        uv::uv_close((*srv).tcp as *mut UvHandle, Some(srv_close_cb));
        // Ownership of tcp moves to the close callback — avoid double-close.
        (*srv).tcp = ptr::null_mut();
        (*srv).listening = 0;
    } else if !(*srv).tcp.is_null() && (*srv).listening == 0 {
        // Never listened (createServer without listen): plain free.
        libc::free((*srv).tcp as *mut c_void);
        (*srv).tcp = ptr::null_mut();
    }
    qjs::sofuu_js_free_value((*srv).ctx, (*srv).handler);
    (*srv).handler = qjs::sofuu_js_undefined();
    drop(Box::from_raw(srv));
}

// ── registration (C symbol replacement for mod_http_server_register) ─

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_http_server_register(ctx: *mut JSContext) {
    let mut class_id: qjs::JSClassID = 0;
    qjs::JS_NewClassID(&mut class_id);
    RES_CLASS_ID.store(class_id, Ordering::Relaxed);
    let res_def = qjs::JSClassDef {
        class_name: c"SofuuResponse".as_ptr(),
        finalizer: Some(res_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    let _rc_res = qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &res_def);

    class_id = 0; /* JS_NewClassID allocates only when *pclass_id == 0 */
    qjs::JS_NewClassID(&mut class_id);
    SRV_CLASS_ID.store(class_id, Ordering::Relaxed);
    let srv_def = qjs::JSClassDef {
        class_name: c"SofuuServer".as_ptr(),
        finalizer: Some(srv_finalizer),
        gc_mark: ptr::null_mut(),
        call: ptr::null_mut(),
        exotic: ptr::null_mut(),
    };
    let _rc_srv = qjs::JS_NewClass(qjs::JS_GetRuntime(ctx), class_id, &srv_def);

    let global = qjs::sofuu_js_get_global_object(ctx);

    /* Global createServer (also copied to Sofuu by engine) */
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"createServer".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_create_server, c"createServer".as_ptr(), 1),
    );

    /* sofuu.serve (legacy) */
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu) || qjs::is_null(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu,
        c"serve".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_serve, c"serve".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        sofuu,
        c"createServer".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_create_server, c"createServer".as_ptr(), 1),
    );

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::CtxPtr;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::mpsc;

    /// Boot a runtime + loop, register the server surface, eval the script
    /// (which must call createServer + listen), then pump the loop while
    /// `driver` runs concurrently (it should hit the server over TCP).
    /// `done` polls a JS global to detect completion; its value is read
    /// into `out` while the runtime is still alive. Bounded at ~2s of pumps.
    /// Restores the global deadline statics when dropped — with_server swaps
    /// the test's deadlines in and the guard puts the previous values back,
    /// both while TEST_LOOP_LOCK is held (guards drop in reverse declaration
    /// order). The old outside-the-lock store/restore raced: a parallel
    /// server test's restore reset the deadlines to the 30s/75s defaults
    /// while another test still needed 250ms, so its probe never saw EOF.
    struct DeadlineGuard(Option<(u64, u64)>);
    impl Drop for DeadlineGuard {
        fn drop(&mut self) {
            if let Some((h, i)) = self.0 {
                HEADERS_TIMEOUT_MS.store(h, Ordering::Relaxed);
                IDLE_TIMEOUT_MS.store(i, Ordering::Relaxed);
            }
        }
    }

    unsafe fn with_server(
        script: &str,
        driver: mpsc::Sender<()>,
        deadlines: Option<(u64, u64)>,
        done: &dyn Fn(*mut qjs::JSContext) -> bool,
        out: &mut String,
    ) -> Result<(), String> {
        let _loop_guard = crate::rt::test_loop_lock();
        let _dl_guard = DeadlineGuard(deadlines.map(|(h, i)| {
            (
                HEADERS_TIMEOUT_MS.swap(h, Ordering::Relaxed),
                IDLE_TIMEOUT_MS.swap(i, Ordering::Relaxed),
            )
        }));
        let rt = qjs::JS_NewRuntime();
        let ctx = qjs::JS_NewContext(rt);
        let _ctx_guard = CtxPtr::new(ctx);
        crate::rt::event_loop::sofuu_loop_init();
        mod_http_server_register(ctx);

        let mut eval_err: Option<String> = None;
        let script_c = CString::new(script).unwrap();
        let r = qjs::JS_Eval(
            ctx,
            script_c.as_ptr(),
            script_c.as_bytes().len(),
            c"<server-test>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
        if qjs::is_exception(r) {
            qjs::js_std_dump_error(ctx);
            eval_err = Some("script eval threw".into());
        } else {
            qjs::sofuu_js_free_value(ctx, r);
            driver.send(()).ok(); /* server is listening — let the client go */
        }

        if eval_err.is_none() {
            for _ in 0..2000 {
                crate::rt::promise::sofuu_flush_jobs(ctx);
                if uv::uv_run(sofuu_loop_get(), uv::UV_RUN_ONCE) == 0 && !done(ctx) {
                    /* Idle loop but test not done — brief park to avoid a
                     * hot spin on the (rare) timer-less waits. */
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                crate::rt::promise::sofuu_flush_jobs(ctx);
                if done(ctx) {
                    *out = read_global(ctx, c"got");
                    break;
                }
            }
        }

        crate::rt::event_loop::sofuu_loop_close();
        qjs::JS_FreeContext(ctx);
        qjs::JS_FreeRuntime(rt);
        eval_err.map_or(Ok(()), Err)
    }

    unsafe fn read_global(ctx: *mut qjs::JSContext, name: &CStr) -> String {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let v = qjs::sofuu_js_get_property_str(ctx, global, name.as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let p = qjs::sofuu_js_to_cstring(ctx, v);
        qjs::sofuu_js_free_value(ctx, v);
        if p.is_null() {
            String::new()
        } else {
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
            s
        }
    }

    /// Raw client: NUL-safe full-response read (up to first close).
    fn raw_exchange(port: u16, req: &[u8]) -> Vec<u8> {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        s.write_all(req).unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match s.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
        buf
    }

    fn pick_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    // ── net-1: req.body survives interior NUL bytes ──────────────────
    // Negative control: against the pre-fix CString detour the handler saw
    // codes "65,68" — the body truncated at the NUL — and this failed.

    #[test]
    fn server_req_body_nul_round_trips() {
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               const codes = [];
               const s = String(req.body);
               for (let i = 0; i < s.length; i++) codes.push(s.charCodeAt(i));
               globalThis.got = codes.join(',');
               res.writeHead(200, {{ 'Content-Type': 'text/plain' }});
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let mut got = String::new();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            let body = b"A\x00CD";
            let head = format!(
                "POST /x HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let mut wire = head.into_bytes();
            wire.extend_from_slice(body);
            raw_exchange(port, &wire)
        });
        unsafe {
            with_server(&script, tx, None, &|ctx| !read_global(ctx, c"got").is_empty(), &mut got)
                .unwrap();
        }
        let resp = t.join().unwrap();
        let text = String::from_utf8_lossy(&resp);
        assert!(text.contains("200"), "expected 200 on the wire, got: {text}");
        assert_eq!(got, "65,0,67,68", "handler must see the NUL byte, got: {got}");
    }

    // ── net-2: writeHead headers beyond Content-Type reach the wire ────
    // Pre-fix, only Content-Type was extracted and every other header
    // set programmatically was silently dropped.

    #[test]
    fn server_write_head_extra_headers_buffered() {
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.got = '1';
               res.writeHead(200, {{ 'Content-Type': 'text/x-net2', 'X-Custom-A': 'v1', 'X-Second': 'v2' }});
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            raw_exchange(port, b"GET /x HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
        });
        unsafe {
            with_server(&script, tx, None, &|ctx| !read_global(ctx, c"got").is_empty(), &mut String::new())
                .unwrap();
        }
        let wire = t.join().unwrap();
        let text = String::from_utf8_lossy(&wire);
        let (head, _body) = text
            .split_once("\r\n\r\n")
            .expect("response must have a header/body separator");
        /* split_once ate the blank-line terminator — restore it so the
         * LAST header line still matches with its trailing CRLF. */
        let block = format!("{head}\r\n");
        assert!(
            block.contains("X-Custom-A: v1\r\n"),
            "buffered response must carry X-Custom-A, got: {text}"
        );
        assert!(
            block.contains("X-Second: v2\r\n"),
            "buffered response must carry X-Second, got: {text}"
        );
        assert!(
            block.contains("Content-Type: text/x-net2\r\n"),
            "Content-Type must survive the rework, got: {text}"
        );
    }

    #[test]
    fn server_write_head_extra_headers_chunked() {
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.got = '1';
               res.writeHead(200, {{ 'Content-Type': 'text/x-net2', 'X-Chunk-A': 'c1' }});
               res.write('p1');
               res.end('p2');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            raw_exchange(port, b"GET /x HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
        });
        unsafe {
            with_server(&script, tx, None, &|ctx| !read_global(ctx, c"got").is_empty(), &mut String::new())
                .unwrap();
        }
        let wire = t.join().unwrap();
        let text = String::from_utf8_lossy(&wire);
        let (head, _body) = text
            .split_once("\r\n\r\n")
            .expect("response must have a header/body separator");
        let block = format!("{head}\r\n");
        assert!(
            block.contains("X-Chunk-A: c1\r\n"),
            "chunked response must carry X-Chunk-A, got: {text}"
        );
        assert!(
            block.contains("Transfer-Encoding: chunked\r\n"),
            "chunked framing must be intact, got: {text}"
        );
    }

    #[test]
    fn server_write_head_crlf_header_skipped() {
        // A CR/LF inside a name or value would inject extra headers /
        // split the response — the header is skipped entirely (fetch-path
        // rule), while benign neighbors still ship.
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.got = '1';
               res.writeHead(200, {{ 'Content-Type': 'text/plain', 'X-Evil': 'a\\r\\nX-Injected: yes', 'X-Good': 'g' }});
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            raw_exchange(port, b"GET /x HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
        });
        unsafe {
            with_server(&script, tx, None, &|ctx| !read_global(ctx, c"got").is_empty(), &mut String::new())
                .unwrap();
        }
        let wire = t.join().unwrap();
        let text = String::from_utf8_lossy(&wire);
        assert!(
            !text.contains("X-Injected"),
            "CR/LF header must not reach the wire, got: {text}"
        );
        let (head, _body) = text
            .split_once("\r\n\r\n")
            .expect("response must have a header/body separator");
        let block = format!("{head}\r\n");
        assert!(
            block.contains("X-Good: g\r\n"),
            "benign neighbor header must still ship, got: {text}"
        );
    }

    // ── net-3: deadlines reap silent/slowloris connections ─────────────
    // Pre-fix, an accepted connection that never finished its request (idle
    // hold-open or header dribble) was held — fd, buffers, parser — forever.

    #[test]
    fn server_deadline_reaps_idle_connection() {
        let port = pick_port();
        let script = format!(
            "const srv = createServer(function (req, res) {{ res.end('x'); }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let eof = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let eof_flag = eof.clone();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = stop.clone();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            /* send NOTHING — the server must reap the silent connection */
            s.set_read_timeout(Some(std::time::Duration::from_secs(4))).unwrap();
            let start = std::time::Instant::now();
            let mut buf = [0u8; 64];
            let eof = matches!(s.read(&mut buf), Ok(0));
            let ms = start.elapsed().as_millis() as u64;
            drop(s);
            eof_flag.store(eof, Ordering::SeqCst);
            /* Wake the harness pump on a cadence: with the probe closed (or
             * a connection that never dies) the loop holds no short-term
             * events and uv_run(ONCE) would block indefinitely. */
            while !stop_flag.load(Ordering::SeqCst)
                && start.elapsed() < std::time::Duration::from_secs(8)
            {
                let _wake = TcpStream::connect(("127.0.0.1", port));
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
            (eof, ms)
        });
        let started = std::time::Instant::now();
        unsafe {
            with_server(
                &script,
                tx,
                Some((250, 250)),
                &|_| {
                    eof.load(Ordering::SeqCst)
                        || started.elapsed() > std::time::Duration::from_secs(3)
                },
                &mut String::new(),
            )
            .unwrap()
        }
        stop.store(true, Ordering::SeqCst);
        let (eof, ms) = t.join().unwrap();
        assert!(eof, "server never closed the silent connection");
        assert!(
            ms < 1500,
            "idle reaper took {ms}ms — absolute deadline not honored"
        );
    }

    #[test]
    fn server_deadline_reaps_header_dribble_slowloris() {
        let port = pick_port();
        let script = format!(
            "const srv = createServer(function (req, res) {{ res.end('x'); }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let eof = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let eof_flag = eof.clone();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = stop.clone();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            /* Dribble one request byte per ~110ms step — each step is far
             * below the 250ms deadline. If the deadline reset on traffic
             * (idle-style) the connection would live indefinitely; the
             * ABSOLUTE deadline must close it mid-dribble. */
            s.set_read_timeout(Some(std::time::Duration::from_millis(60))).unwrap();
            let start = std::time::Instant::now();
            let mut buf = [0u8; 64];
            let mut eof = false;
            for b in b"GET / HTTP/1.1\r\nHo" {
                if s.write_all(std::slice::from_ref(b)).is_err() {
                    eof = true; /* EPIPE — the server already dropped us */
                    break;
                }
                match s.read(&mut buf) {
                    Ok(0) => {
                        eof = true;
                        break;
                    }
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => {
                        eof = true; /* reset — server dropped us */
                        break;
                    }
                    Ok(_) => {}
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            while !eof && start.elapsed() < std::time::Duration::from_millis(1200) {
                match s.read(&mut buf) {
                    Ok(0) => eof = true,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => eof = true,
                    Ok(_) => {}
                }
            }
            let ms = start.elapsed().as_millis() as u64;
            drop(s);
            eof_flag.store(eof, Ordering::SeqCst);
            /* Wake the harness pump on a cadence (see idle test) so the
             * negative-control run fails loudly instead of hanging. */
            while !stop_flag.load(Ordering::SeqCst)
                && start.elapsed() < std::time::Duration::from_secs(8)
            {
                let _wake = TcpStream::connect(("127.0.0.1", port));
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
            (eof, ms)
        });
        let started = std::time::Instant::now();
        unsafe {
            with_server(
                &script,
                tx,
                Some((250, 250)),
                &|_| {
                    eof.load(Ordering::SeqCst)
                        || started.elapsed() > std::time::Duration::from_secs(3)
                },
                &mut String::new(),
            )
            .unwrap()
        }
        stop.store(true, Ordering::SeqCst);
        let (eof, ms) = t.join().unwrap();
        assert!(eof, "server never closed the slowloris dribbler");
        assert!(
            ms < 1200,
            "dribbler survived {ms}ms — deadline was reset by traffic?"
        );
    }

    #[test]
    fn server_deadline_spares_healthy_traffic() {
        // The reaper must never kill a request that completes promptly —
        // with 250ms deadlines the whole round trip runs in ~1ms.
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.got = '1';
               res.writeHead(200, {{ 'Content-Type': 'text/plain' }});
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (tx, rx) = mpsc::channel::<()>();
        let t = std::thread::spawn(move || {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
            raw_exchange(port, b"GET /x HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
        });
        unsafe {
            with_server(
                &script,
                tx,
                Some((250, 250)),
                &|ctx| !read_global(ctx, c"got").is_empty(),
                &mut String::new(),
            )
            .unwrap()
        }
        let wire = t.join().unwrap();
        let text = String::from_utf8_lossy(&wire);
        assert!(
            text.contains("200") && text.contains("ok"),
            "healthy request must complete normally under short deadlines, got: {text}"
        );
    }
    /// Connect with a bounded retry: a test's client thread must never race the
    /// server's `listen()` (or another test holding the loop lock) into a bare
    /// `unwrap` panic — that used to leave the pump with no event and hang the
    /// whole run.
    fn connect_retry(port: u16) -> TcpStream {
        for _ in 0..600 {
            if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
                return s;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("server on 127.0.0.1:{port} never accepted a connection");
    }

    /// `raw_exchange` over a retrying connect (same read-to-EOF contract).
    fn raw_exchange_retry(port: u16, req: &[u8]) -> Vec<u8> {
        let mut s = connect_retry(port);
        s.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        s.write_all(req).unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match s.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
        buf
    }

    /// AUDIT-2026-10-31 P1-11: drive `reqs` (concurrently) against a fresh
    /// server and return (raw wire bytes per request, the `got` global).
    ///
    /// Two harness details this test class needs:
    ///  * a WAKER thread keeps connecting on a cadence, because
    ///    `uv_run(UV_RUN_ONCE)` blocks forever on an event-less loop (the same
    ///    trick the deadline tests use);
    ///  * `done` bails at a hard deadline, so a broken fix fails loudly instead
    ///    of hanging the suite.
    ///
    /// The pump also asserts the run never NEWLY records an uncaught error: a
    /// leaked rejection sets exactly that status flag (print_rejection →
    /// note_uncaught_error), so this is the in-process proof that the handler's
    /// rejection was handled rather than surfacing as an
    /// UnhandledPromiseRejection. The baseline is taken on the first poll
    /// because RUN_STATUS is keyed by the raw context pointer, which a freed
    /// context can hand to the next test in the same process.
    fn serve_requests(script: &str, port: u16, reqs: &[&[u8]]) -> (Vec<Vec<u8>>, String) {
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut clients = Vec::new();
        for req in reqs {
            let body = req.to_vec();
            let fin = finished.clone();
            clients.push(std::thread::spawn(move || {
                let r = raw_exchange_retry(port, &body);
                fin.fetch_add(1, Ordering::SeqCst);
                r
            }));
        }
        let stop_w = stop.clone();
        let waker = std::thread::spawn(move || {
            while !stop_w.load(Ordering::SeqCst) {
                let _wake = TcpStream::connect(("127.0.0.1", port));
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        let started = std::time::Instant::now();
        let fin = finished.clone();
        let baseline: std::cell::Cell<Option<bool>> = std::cell::Cell::new(None);
        let (tx, _rx) = mpsc::channel::<()>();
        let mut out = String::new();
        unsafe {
            with_server(
                script,
                tx,
                Some((2000, 2000)),
                &|ctx| {
                    let now = crate::rt::engine::run_status(ctx).has_uncaught_error;
                    let base = match baseline.get() {
                        Some(b) => b,
                        None => {
                            baseline.set(Some(now));
                            now
                        }
                    };
                    assert!(
                        !(now && !base),
                        "the handler's rejection escaped as an uncaught error \
                         (UnhandledPromiseRejection) instead of being handled"
                    );
                    fin.load(Ordering::SeqCst) == reqs.len()
                        || started.elapsed() > std::time::Duration::from_secs(12)
                },
                &mut out,
            )
            .unwrap();
        }
        stop.store(true, Ordering::SeqCst);
        let mut wires = Vec::new();
        for c in clients {
            wires.push(c.join().unwrap());
        }
        waker.join().ok();
        (wires, out)
    }

    const GET_BOOM: &[u8] = b"GET /boom HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n";
    const POST_ECHO: &[u8] =
        b"POST /echo HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello";

    // ── P1-11 (a): async handler rejections answer 500, exactly once ────
    // Negative control: pre-fix the handler's promise was freed without a
    // rejection handler, so the client got NOTHING back (curl timed out after
    // 5s) and the process printed "[sofuu] UnhandledPromiseRejection: Error: …".

    #[test]
    fn server_async_handler_rejection_answers_500_once() {
        let port = pick_port();
        let script = format!(
            "const srv = createServer(async function (req, res) {{
               throw new Error('boom-immediate');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, _) = serve_requests(&script, port, &[GET_BOOM]);
        let text = String::from_utf8_lossy(&wires[0]);
        assert!(
            text.contains("HTTP/1.1 500"),
            "a rejected async handler must answer 500, got: {text:?}"
        );
        assert!(
            text.contains("Internal Server Error"),
            "the 500 must carry a body, got: {text:?}"
        );
        assert!(
            !text.contains("boom-immediate"),
            "the error text is script-internal and must not reach the client, got: {text:?}"
        );
        assert_eq!(
            text.matches("HTTP/1.1 500").count(),
            1,
            "the rejection must produce EXACTLY one response, got: {text:?}"
        );
    }

    #[test]
    fn server_async_handler_rejection_after_await_answers_500() {
        // This rejection lands a microtask AFTER the handler returns, so it also
        // proves the returned promise (and its reaction) survives the server
        // releasing its reference.
        let port = pick_port();
        let script = format!(
            "const srv = createServer(async function (req, res) {{
               await null;
               throw new Error('boom-after-await');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, _) = serve_requests(&script, port, &[GET_BOOM]);
        let text = String::from_utf8_lossy(&wires[0]);
        assert!(
            text.contains("HTTP/1.1 500"),
            "a rejection AFTER an await must still answer 500, got: {text:?}"
        );
        assert_eq!(
            text.matches("HTTP/1.1 500").count(),
            1,
            "the rejection must produce EXACTLY one response, got: {text:?}"
        );
    }

    #[test]
    fn server_async_handler_resolve_still_answers_normally() {
        // The rejection handler must not disturb a handler that fulfils.
        let port = pick_port();
        let script = format!(
            "const srv = createServer(async function (req, res) {{
               await null;
               res.writeHead(200, {{ 'Content-Type': 'text/plain' }});
               res.end('async-ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, _) = serve_requests(
            &script,
            port,
            &[b"GET /ok HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n"],
        );
        let text = String::from_utf8_lossy(&wires[0]);
        assert!(
            text.contains("HTTP/1.1 200") && text.contains("async-ok"),
            "a fulfilled async handler must answer normally, got: {text:?}"
        );
        assert_eq!(
            text.matches("HTTP/1.1 ").count(),
            1,
            "one response only, got: {text:?}"
        );
    }

    #[test]
    fn server_overlapping_rejections_answer_their_own_client() {
        // P1-11 trap: a GLOBAL "current response" would answer the WRONG
        // connection. Two overlapping rejecting requests must each get exactly
        // one 500 on their own socket and never another request's response.
        let port = pick_port();
        let script = format!(
            "const srv = createServer(async function (req, res) {{
               await null; await null; await null;
               throw new Error('boom ' + req.url);
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, _) = serve_requests(
            &script,
            port,
            &[
                b"GET /a HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n",
                b"GET /b HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n",
            ],
        );
        for wire in &wires {
            let text = String::from_utf8_lossy(wire);
            assert!(
                text.contains("HTTP/1.1 500"),
                "each rejected request must get its own 500, got: {text:?}"
            );
            assert!(
                !text.contains("HTTP/1.1 200"),
                "a connection must not receive another request's response, got: {text:?}"
            );
            assert_eq!(
                text.matches("HTTP/1.1 500").count(),
                1,
                "exactly one response per connection, got: {text:?}"
            );
        }
    }

    // ── P1-11 (b): req.on('data'|'end'|'close') ─────────────────────────
    // Negative control: pre-fix the request object exposed url/method/headers/
    // body only, so the echo pattern died with "req.on is not a function" and
    // the client hung until its deadline.

    #[test]
    fn server_req_on_data_end_echo_round_trips() {
        let port = pick_port();
        let script = format!(
            "const srv = createServer(function (req, res) {{
               let body = '';
               req.on('data', function (c) {{ body += c; }});
               req.on('end', function () {{
                 res.writeHead(200, {{ 'Content-Type': 'text/plain' }});
                 res.end('echo:' + body);
               }});
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, _) = serve_requests(&script, port, &[POST_ECHO]);
        let text = String::from_utf8_lossy(&wires[0]);
        assert!(
            text.contains("HTTP/1.1 200"),
            "the echo handler must answer 200, got: {text:?}"
        );
        assert!(
            text.contains("echo:hello"),
            "req.on('data')/'end' must deliver the request body, got: {text:?}"
        );
    }

    #[test]
    fn server_req_on_chains_and_close_fires_after_response() {
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.chain =
                 (req.on('data', function () {{}}) === req) ? '1' : '0';
               req.on('close', function () {{
                 globalThis.got = 'closed:' + globalThis.chain;
               }});
               res.writeHead(200, {{ 'Content-Type': 'text/plain' }});
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (wires, got) = serve_requests(
            &script,
            port,
            &[b"GET /chain HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n"],
        );
        let text = String::from_utf8_lossy(&wires[0]);
        assert!(
            text.contains("HTTP/1.1 200") && text.contains("ok"),
            "the plain handler must still answer, got: {text:?}"
        );
        assert_eq!(
            got, "closed:1",
            "req.on must return the request (chainable) and 'close' must fire \
             once the response ended"
        );
    }

    #[test]
    fn server_req_on_unknown_event_is_loud() {
        // This request object is not an EventEmitter and can never emit
        // 'error'/'timeout'/…, so subscribing to one is refused rather than
        // silently accepted — a silent accept looks like a working subscription
        // for the rest of the process's life.
        let port = pick_port();
        let script = format!(
            "globalThis.got = '';
             const srv = createServer(function (req, res) {{
               globalThis.got = 'silent';
               try {{
                 req.on('timeout', function () {{}});
               }} catch (e) {{
                 globalThis.got = 'threw:' + (e instanceof TypeError);
               }}
               res.end('ok');
             }});
             srv.listen({port}, '127.0.0.1');",
            port = port
        );
        let (_, got) = serve_requests(
            &script,
            port,
            &[b"GET /u HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n"],
        );
        assert_eq!(
            got, "threw:true",
            "an unsupported event name must throw a TypeError, not be ignored"
        );
    }

}