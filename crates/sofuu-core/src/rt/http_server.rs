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
use std::sync::atomic::{AtomicU32, Ordering};

use sofuu_ffi::http_parser::{self, HttpParser, HttpParserSettings};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvHandle, UvStream, UvTcp, UvWriteReq};

use crate::rt::event_loop::sofuu_loop_get;

const MAX_REQ_URL: usize = 8 * 1024;
const MAX_REQ_BODY: usize = 16 * 1024 * 1024;

static RES_CLASS_ID: AtomicU32 = AtomicU32::new(0);
static SRV_CLASS_ID: AtomicU32 = AtomicU32::new(0);

// ── structs ─────────────────────────────────────────────────────────

struct Server {
    tcp: *mut UvTcp,
    ctx: *mut JSContext,
    handler: JSValue,
    listening: c_int, /* tcp handle was uv_tcp_init'd */
}

struct Client {
    tcp: *mut UvTcp,
    srv: *mut Server,
    parser: HttpParser,
    /* request accumulation */
    url: Vec<u8>,
    body: Vec<u8>,
    /* response state */
    status: c_int,
    ctype: RefCell<Vec<u8>>,
    headers_set: c_int,
    streaming: c_int, /* chunked response started */
    finished: c_int,  /* response ended — further res calls are no-ops */
    refcount: c_int,  /* connection (1) + live res JS object (1) */
    closing: c_int,   /* uv_close already requested — no double close */
    /* JS objects */
    req_val: Cell<JSValue>,
    res_val: Cell<JSValue>,
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
            status: 0,
            ctype: RefCell::new(Vec::new()),
            headers_set: 0,
            streaming: 0,
            finished: 0,
            refcount: 1,
            closing: 0,
            req_val: Cell::new(zero_jsvalue()),
            res_val: Cell::new(zero_jsvalue()),
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
        libc::free((*c).tcp as *mut c_void);
        drop(Box::from_raw(c));
    }
}

unsafe extern "C" fn client_free(h: *mut UvHandle) {
    // SAFETY: the tcp handle's data points at the client (set at accept).
    let c = *(h as *mut *mut Client);
    client_release(c);
}

fn status_text(status: c_int) -> &'static str {
    match status {
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
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

unsafe extern "C" fn on_write_done(req: *mut UvWriteReq, _status: c_int) {
    // SAFETY: req->data (first field of uv_req_t) was set at queue_write;
    // libuv reserves it for the user.
    let wr = *(req as *mut *mut WriteReq);
    let c = (*wr).client;
    let close_after = (*wr).close_after;
    libc::free((*wr).data as *mut c_void);
    libc::free(req as *mut c_void);
    drop(Box::from_raw(wr));
    if close_after != 0 && (*c).closing == 0 {
        /* Only ONE path may own uv_close: a racing on_read(EOF/abort)
         * would otherwise double-close the handle and double-free. */
        (*c).closing = 1;
        uv::uv_close((*c).tcp as *mut UvHandle, Some(client_free));
    }
}

/// Queue a heap buffer for writing; takes ownership of `buf`.
unsafe fn queue_write(c: *mut Client, buf: *mut c_char, len: usize, close_after: c_int) {
    let wr = Box::into_raw(Box::new(WriteReq {
        data: buf,
        client: c,
        close_after,
    }));
    let wreq = libc::malloc(uv::sofuu_uv_write_size()) as *mut UvWriteReq;
    *(wreq as *mut *mut WriteReq) = wr;
    let b = uv::UvBuf { base: buf, len };
    uv::uv_write(wreq, (*c).tcp as *mut UvStream, &b, 1, Some(on_write_done));
}

/// Buffered single-shot response with Content-Length, then close.
unsafe fn send_response(c: *mut Client, status: c_int, ct: &str, body: &[u8]) {
    let hdr = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        status_text(status),
        ct,
        body.len()
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
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        st,
        status_text(st),
        ct
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
        for key in [c"Content-Type".as_ptr(), c"content-type".as_ptr()] {
            let v = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), key);
            if !qjs::is_undefined(v) {
                let s = qjs::sofuu_js_to_cstring(ctx, v);
                if !s.is_null() {
                    let bytes = CStr::from_ptr(s).to_bytes();
                    ctype = Some(bytes[..bytes.len().min(255)].to_vec());
                    qjs::sofuu_js_free_cstring(ctx, s);
                }
                qjs::sofuu_js_free_value(ctx, v);
                break;
            }
            qjs::sofuu_js_free_value(ctx, v);
        }
        if let Some(ct) = ctype {
            (*c).ctype.borrow_mut().clear();
            (*c).ctype.borrow_mut().extend_from_slice(&ct);
        }
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

// ── http_parser callbacks ───────────────────────────────────────────

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
    if (*c).body.len() + len + 1 > MAX_REQ_BODY {
        return 1; /* cap, no OOM */
    }
    (*c).body.extend_from_slice(std::slice::from_raw_parts(at as *const u8, len));
    0
}

unsafe extern "C" fn cb_message_complete(p: *mut HttpParser) -> c_int {
    let c = (*p).data as *mut Client;
    let ctx = (*(*c).srv).ctx;

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
        let body_c = CString::new((*c).body.clone()).unwrap_or_default();
        qjs::sofuu_js_set_property_str(
            ctx,
            (*c).req_val.get(),
            c"body".as_ptr(),
            qjs::JS_NewStringLen(ctx, body_c.as_ptr(), (*c).body.len()),
        );
    }

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

    let mut args: [JSValueConst; 2] = [(*c).req_val.get(), (*c).res_val.get()];
    let ret = qjs::JS_Call(ctx, (*(*c).srv).handler, qjs::sofuu_js_undefined(), 2, args.as_mut_ptr());
    if qjs::is_exception(ret) {
        qjs::js_std_dump_error(ctx);
    }
    qjs::sofuu_js_free_value(ctx, ret);
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
        on_header_field: None,
        on_header_value: None,
        on_headers_complete: None,
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
                (*c).closing = 1;
                uv::uv_close(s as *mut UvHandle, Some(client_free));
            }
        }
    } else if n < 0 {
        if (*c).closing == 0 {
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

    let mut a: libc::sockaddr_in = std::mem::zeroed();
    let ip = if host.is_null() || *host == 0 { c"0.0.0.0".as_ptr() } else { host };
    uv::uv_ip4_addr(ip, port, &mut a);
    uv::uv_tcp_bind(tcp, &a as *const _ as *const libc::sockaddr, 0);
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
    if qjs::JS_IsFunction(ctx, cb) != 0 {
        let ret = qjs::JS_Call(ctx, cb, qjs::sofuu_js_undefined(), 0, ptr::null());
        qjs::sofuu_js_free_value(ctx, ret);
    }
    // SAFETY: this_val is a dup'd ref (JS_Call does not consume).
    qjs::sofuu_js_dup_value(ctx, this_val)
}

// ── createServer(handler) ───────────────────────────────────────────

unsafe extern "C" fn js_create_server(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 || qjs::JS_IsFunction(ctx, *argv) == 0 {
        return qjs::JS_ThrowTypeError(ctx, c"createServer: function required".as_ptr());
    }
    let srv = Box::into_raw(Box::new(Server {
        tcp: ptr::null_mut(),
        ctx,
        handler: qjs::sofuu_js_dup_value(ctx, *argv),
        listening: 0,
    }));
    let obj = qjs::JS_NewObjectClass(ctx, SRV_CLASS_ID.load(Ordering::Relaxed) as c_int);
    qjs::JS_SetOpaque(obj, srv as *mut c_void);
    qjs::sofuu_js_set_property_str(
        ctx,
        obj,
        c"listen".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_server_listen, c"listen".as_ptr(), 2),
    );
    obj
}

// ── Legacy sofuu.serve(port, handler) ───────────────────────────────

unsafe extern "C" fn js_serve(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
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
    }));
    let r = do_listen(srv, c"0.0.0.0".as_ptr(), port as c_int);
    if r < 0 {
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

unsafe extern "C" fn srv_finalizer(_rt: *mut qjs::JSRuntime, val: JSValue) {
    let srv = qjs::JS_GetOpaque(val, SRV_CLASS_ID.load(Ordering::Relaxed)) as *mut Server;
    if srv.is_null() {
        return;
    }
    if (*srv).listening != 0 && uv::uv_is_closing((*srv).tcp as *const UvHandle) == 0 {
        uv::uv_close((*srv).tcp as *mut UvHandle, None);
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
