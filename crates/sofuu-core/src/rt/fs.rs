// rt/fs.rs — async file I/O on libuv (PLAN-RUST-MIGRATION M2).
//
// Port of the deleted `src/io/fs.c`, semantics verbatim: sofuu.fs.readFile /
// writeFile / appendFile / exists / readdir / mkdir{recursive} / rm /
// readFileBytes. Request structs live in Boxes (indexed by the malloc'd
// uv_fs_t's leading `data` field); the uv_fs_t storages are freed by the
// request's final callback, exactly like the C calloc/free pairing.
//
// C symbol replaced: `mod_fs_register` (engine.c calls it unchanged).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use sofuu_ffi::uv::{self, UvBuf, UvFs};

use crate::rt::event_loop::sofuu_loop_get;
use crate::rt::promise::{
    sofuu_flush_jobs, sofuu_promise_new, sofuu_promise_reject_str, sofuu_promise_resolve,
    PromiseHandle,
};

// libuv requests are allocated at their true size via the C shim.
fn fs_req_alloc() -> *mut UvFs {
    unsafe { libc::malloc(uv::sofuu_uv_fs_size()) as *mut UvFs }
}

/// C error formatting: uv_strerror(neg_errno) — the helper mirrors fs.c.
fn uv_err_str(err: c_int) -> CString {
    // SAFETY: uv_strerror returns a static string.
    let p = unsafe { uv::uv_strerror(err) };
    if p.is_null() {
        CString::new("unknown error").unwrap_or_default()
    } else {
        CString::new(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()).unwrap_or_default()
    }
}

// ── read_file: open → fstat → read → close → resolve ─────────────────

struct ReadReq {
    promise: *mut PromiseHandle, /* NULL after resolve/reject */
    fd: c_int,
    buf: Vec<u8>,
    path: CString,
}

unsafe extern "C" fn rf_close_cb(req_fs: *mut UvFs) {
    // SAFETY: req_fs is a live uv_fs_t whose data points at the Box.
    uv::uv_fs_req_cleanup(req_fs);
    let r = *(req_fs as *mut *mut ReadReq);
    libc::free(req_fs as *mut c_void);
    drop(Box::from_raw(r)); /* frees path + buf */
}

unsafe extern "C" fn rf_read_cb(req_fs: *mut UvFs) {
    // SAFETY: live request; result read via shim BEFORE cleanup.
    let r = *(req_fs as *mut *mut ReadReq);
    let n = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    let ctx = (*(*r).promise).ctx;

    if n < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(n as c_int).as_ptr());
    } else {
        let r_ref = &mut *r;
        r_ref.buf[n as usize] = 0;
        let str_v = qjs::JS_NewStringLen(ctx, (*r).buf.as_ptr() as *const c_char, n as usize);
        sofuu_promise_resolve((*r).promise, str_v);
        qjs::sofuu_js_free_value(ctx, str_v);
        sofuu_flush_jobs(ctx);
    }
    (*r).promise = ptr::null_mut();

    /* Close the fd */
    let cr = fs_req_alloc();
    *(cr as *mut *mut ReadReq) = r;
    uv::uv_fs_close(sofuu_loop_get(), cr, (*r).fd as u64, Some(rf_close_cb));
}

unsafe extern "C" fn rf_open_cb(req_fs: *mut UvFs) {
    // SAFETY: live request; result saved before cleanup.
    let r = *(req_fs as *mut *mut ReadReq);
    let result = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    if result < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(result as c_int).as_ptr());
        drop(Box::from_raw(r)); /* frees path + buf */
        return;
    }
    (*r).fd = result as c_int;

    /* Synchronous fstat to get file size */
    let st = fs_req_alloc();
    uv::uv_fs_fstat(sofuu_loop_get(), st, (*r).fd as u64, None);
    let sz = uv::sofuu_uv_fs_stat_size(st);
    uv::uv_fs_req_cleanup(st);
    libc::free(st as *mut c_void);

    (*r).buf = vec![0u8; sz as usize + 1];

    let rr = fs_req_alloc();
    *(rr as *mut *mut ReadReq) = r;
    let buf = UvBuf {
        base: (*r).buf.as_mut_ptr() as *mut c_char,
        len: sz as usize,
    };
    uv::uv_fs_read(sofuu_loop_get(), rr, (*r).fd as u64, &buf, 1, 0, Some(rf_read_cb));
}

unsafe extern "C" fn js_read_file(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"readFile: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let path = CStr::from_ptr(path_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, path_ptr);
    let path = match CString::new(path) {
        Ok(p) => p,
        Err(_) => return qjs::sofuu_js_exception(), // embedded NUL — parity: strdup stops there; keep simple
    };

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    let r = Box::into_raw(Box::new(ReadReq {
        promise: out,
        fd: -1,
        buf: Vec::new(),
        path,
    }));
    let req = fs_req_alloc();
    *(req as *mut *mut ReadReq) = r;
    uv::uv_fs_open(sofuu_loop_get(), req, (*r).path.as_ptr(), libc::O_RDONLY, 0, Some(rf_open_cb));

    promise
}

// ── write_file / append_file: open → write → close → resolve ─────────

struct WriteReq {
    promise: *mut PromiseHandle,
    fd: c_int,
    buf: Vec<u8>,
    path: CString,
}

unsafe extern "C" fn wf_close_cb(req_fs: *mut UvFs) {
    uv::uv_fs_req_cleanup(req_fs);
    let r = *(req_fs as *mut *mut WriteReq);
    libc::free(req_fs as *mut c_void);
    drop(Box::from_raw(r));
}

unsafe extern "C" fn wf_write_cb(req_fs: *mut UvFs) {
    let r = *(req_fs as *mut *mut WriteReq);
    let ctx = (*(*r).promise).ctx;
    let result = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    if result < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(result as c_int).as_ptr());
    } else {
        sofuu_promise_resolve((*r).promise, qjs::sofuu_js_undefined());
    }
    sofuu_flush_jobs(ctx);
    (*r).promise = ptr::null_mut();

    let cr = fs_req_alloc();
    *(cr as *mut *mut WriteReq) = r;
    uv::uv_fs_close(sofuu_loop_get(), cr, (*r).fd as u64, Some(wf_close_cb));
}

unsafe extern "C" fn wf_open_cb(req_fs: *mut UvFs) {
    let r = *(req_fs as *mut *mut WriteReq);
    let result = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    if result < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(result as c_int).as_ptr());
        drop(Box::from_raw(r));
        return;
    }
    (*r).fd = result as c_int;

    let wr = fs_req_alloc();
    *(wr as *mut *mut WriteReq) = r;
    let buf = UvBuf {
        base: (*r).buf.as_mut_ptr() as *mut c_char,
        len: (*r).buf.len(),
    };
    uv::uv_fs_write(sofuu_loop_get(), wr, (*r).fd as u64, &buf, 1, -1, Some(wf_write_cb));
}

unsafe fn write_impl(
    ctx: *mut JSContext,
    path_val: JSValueConst,
    data_val: JSValueConst,
    flags: c_int,
) -> JSValue {
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, path_val);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let path = CStr::from_ptr(path_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, path_ptr);

    let data_ptr = qjs::sofuu_js_to_cstring(ctx, data_val);
    if data_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let data = CStr::from_ptr(data_ptr).to_bytes().to_vec();
    qjs::sofuu_js_free_cstring(ctx, data_ptr);

    let path = match CString::new(path) {
        Ok(p) => p,
        Err(_) => return qjs::sofuu_js_exception(),
    };

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    let r = Box::into_raw(Box::new(WriteReq {
        promise: out,
        fd: -1,
        buf: data,
        path,
    }));
    let req = fs_req_alloc();
    *(req as *mut *mut WriteReq) = r;
    uv::uv_fs_open(
        sofuu_loop_get(),
        req,
        (*r).path.as_ptr(),
        flags,
        0o644,
        Some(wf_open_cb),
    );
    promise
}

unsafe extern "C" fn js_write_file(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"writeFile: 2 args required".as_ptr());
    }
    write_impl(ctx, *argv, *argv.add(1), libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC)
}

unsafe extern "C" fn js_append_file(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"appendFile: 2 args required".as_ptr());
    }
    write_impl(ctx, *argv, *argv.add(1), libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND)
}

// ── exists — sync stat wrapped in a resolved promise ─────────────────

unsafe extern "C" fn js_exists(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_new_bool(ctx, 0);
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }

    let req = fs_req_alloc();
    let r = uv::uv_fs_stat(sofuu_loop_get(), req, path_ptr, None);
    uv::uv_fs_req_cleanup(req);
    libc::free(req as *mut c_void);
    qjs::sofuu_js_free_cstring(ctx, path_ptr);

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    sofuu_promise_resolve(out, qjs::sofuu_js_new_bool(ctx, if r == 0 { 1 } else { 0 }));
    promise
}

// ── readdir — sync scandir wrapped in a resolved promise ─────────────

unsafe extern "C" fn js_readdir(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"readdir: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }

    let req = fs_req_alloc();
    let r = uv::uv_fs_scandir(sofuu_loop_get(), req, path_ptr, 0, None);
    qjs::sofuu_js_free_cstring(ctx, path_ptr);

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    if r < 0 {
        uv::uv_fs_req_cleanup(req);
        libc::free(req as *mut c_void);
        sofuu_promise_reject_str(out, uv_err_str(r).as_ptr());
        return promise;
    }

    let arr = qjs::JS_NewArray(ctx);
    let mut idx: u32 = 0;
    loop {
        let mut ent = uv::UvDirent {
            name: ptr::null(),
            d_type: 0,
        };
        if uv::uv_fs_scandir_next(req, &mut ent) == uv::UV_EOF {
            break;
        }
        // SAFETY: ent.name is NUL-terminated during the scandir iteration.
        let name = CStr::from_ptr(ent.name).to_string_lossy();
        let sv = qjs::sofuu_js_new_string(ctx, ent.name);
        qjs::JS_SetPropertyUint32(ctx, arr, idx, sv);
        let _ = name;
        idx += 1;
    }
    uv::uv_fs_req_cleanup(req);
    libc::free(req as *mut c_void);

    sofuu_promise_resolve(out, arr);
    qjs::sofuu_js_free_value(ctx, arr); /* resolve dups it */
    promise
}

// ── mkdir — sync uv calls wrapped in a pre-resolved promise ──────────
// Recursive: creates missing parents when opts.recursive is truthy.

unsafe extern "C" fn js_mkdir(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"mkdir: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let path_bytes = CStr::from_ptr(path_ptr).to_bytes();
    let path_cstring = CString::new(path_bytes).unwrap_or_default();

    let mut recursive = 0;
    if argc > 1 && qjs::is_object(*argv.add(1)) {
        let rv = qjs::sofuu_js_get_property_str(ctx, *argv.add(1), c"recursive".as_ptr());
        recursive = qjs::JS_ToBool(ctx, rv);
        qjs::sofuu_js_free_value(ctx, rv);
    }

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);
    let r: c_int;

    if recursive == 0 {
        let req = fs_req_alloc();
        r = uv::uv_fs_mkdir(sofuu_loop_get(), req, path_ptr, 0o755, None);
        uv::uv_fs_req_cleanup(req);
        libc::free(req as *mut c_void);
    } else {
        /* Recursive: mkdir every path component (left to right), ignoring
         * EEXIST — the C port also scanned for an existing ancestor, but its
         * mkdir loop always started from the path root anyway, so the scan
         * had no observable effect and is dropped here. */
        let mut bytes = path_bytes.to_vec();
        // strip trailing slashes (C: while (n > 0 && copy[n-1] == '/') ...)
        let mut n = bytes.len();
        while n > 0 && bytes[n - 1] == b'/' {
            bytes.truncate(n - 1);
            n -= 1;
        }
        let result: c_int;
        let mut s = 0usize;
        loop {
            if s == bytes.len() || bytes[s] == b'/' {
                let prefix = if s == 0 && bytes.first() == Some(&b'/') {
                    CString::new("/").unwrap()
                } else {
                    // SAFETY: no interior NUL (bytes came from a C string).
                    CString::new(&bytes[..s]).unwrap_or_default()
                };
                let p = if prefix.as_bytes().is_empty() { c"/".as_ptr() } else { prefix.as_ptr() };
                let req = fs_req_alloc();
                let er = uv::uv_fs_mkdir(sofuu_loop_get(), req, p, 0o755, None);
                uv::uv_fs_req_cleanup(req);
                libc::free(req as *mut c_void);
                if er < 0 && er != uv::UV_EEXIST {
                    result = er;
                    break;
                }
                if s == bytes.len() {
                    result = 0;
                    break;
                }
            }
            s += 1;
        }
        r = result;
    }

    if r == 0 {
        sofuu_promise_resolve(out, qjs::sofuu_js_undefined());
    } else {
        sofuu_promise_reject_str(out, uv_err_str(r).as_ptr());
    }

    qjs::sofuu_js_free_cstring(ctx, path_ptr);
    let _ = path_cstring;
    promise
}

// ── rm — sync unlink with rmdir fallback, in a pre-resolved promise ──
// Does not recurse into non-empty dirs. UV_ENOENT is success (no-op).

unsafe extern "C" fn js_rm(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"rm: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    let req = fs_req_alloc();
    let mut r = uv::uv_fs_unlink(sofuu_loop_get(), req, path_ptr, None);
    uv::uv_fs_req_cleanup(req);
    libc::free(req as *mut c_void);
    if r == uv::UV_EISDIR || r == uv::UV_EPERM {
        /* Likely a directory (or a dir where unlink is denied) — try rmdir. */
        let req2 = fs_req_alloc();
        r = uv::uv_fs_rmdir(sofuu_loop_get(), req2, path_ptr, None);
        uv::uv_fs_req_cleanup(req2);
        libc::free(req2 as *mut c_void);
    }
    /* UV_ENOENT is treated as success: rm of a missing path is a no-op. */
    if r == 0 || r == uv::UV_ENOENT {
        sofuu_promise_resolve(out, qjs::sofuu_js_undefined());
    } else {
        sofuu_promise_reject_str(out, uv_err_str(r).as_ptr());
    }

    qjs::sofuu_js_free_cstring(ctx, path_ptr);
    promise
}

// ── readFileBytes — binary read returning a Uint8Array ───────────────

struct RfbReq {
    promise: *mut PromiseHandle,
    fd: c_int,
    buf: Vec<u8>,
    path: CString,
}

unsafe extern "C" fn rfb_close_cb(req_fs: *mut UvFs) {
    uv::uv_fs_req_cleanup(req_fs);
    let r = *(req_fs as *mut *mut RfbReq);
    libc::free(req_fs as *mut c_void);
    drop(Box::from_raw(r));
}

unsafe extern "C" fn rfb_read_cb(req_fs: *mut UvFs) {
    let r = *(req_fs as *mut *mut RfbReq);
    let n = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    let ctx = (*(*r).promise).ctx;

    if n < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(n as c_int).as_ptr());
        (*r).promise = ptr::null_mut();
    } else {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let ctor = qjs::sofuu_js_get_property_str(ctx, global, c"Uint8Array".as_ptr());
        qjs::sofuu_js_free_value(ctx, global);
        let lenv = qjs::sofuu_js_new_int32(ctx, n as i32);
        let arr = qjs::JS_CallConstructor(ctx, ctor, 1, &lenv);
        qjs::sofuu_js_free_value(ctx, lenv);
        qjs::sofuu_js_free_value(ctx, ctor);
        if qjs::is_exception(arr) {
            sofuu_promise_reject_str((*r).promise, c"failed to allocate Uint8Array".as_ptr());
            (*r).promise = ptr::null_mut();
        } else {
            /* Copy bytes into the array's backing buffer. */
            let mut off: usize = 0;
            let mut len: usize = 0;
            let ab = qjs::JS_GetTypedArrayBuffer(ctx, arr, &mut off, &mut len, ptr::null_mut());
            if !qjs::is_exception(ab) {
                let mut ab_size: usize = 0;
                let dst = qjs::JS_GetArrayBuffer(ctx, &mut ab_size, ab);
                if !dst.is_null() && off + (n as usize) <= ab_size {
                    std::ptr::copy_nonoverlapping((*r).buf.as_ptr(), dst.add(off), n as usize);
                }
                qjs::sofuu_js_free_value(ctx, ab);
            } else {
                qjs::sofuu_js_free_value(ctx, ab);
            }
            sofuu_promise_resolve((*r).promise, arr);
            qjs::sofuu_js_free_value(ctx, arr);
        }
        sofuu_flush_jobs(ctx);
    }

    let cr = fs_req_alloc();
    *(cr as *mut *mut RfbReq) = r;
    uv::uv_fs_close(sofuu_loop_get(), cr, (*r).fd as u64, Some(rfb_close_cb));
}

unsafe extern "C" fn rfb_open_cb(req_fs: *mut UvFs) {
    let r = *(req_fs as *mut *mut RfbReq);
    let result = uv::sofuu_uv_fs_result(req_fs);
    uv::uv_fs_req_cleanup(req_fs);
    libc::free(req_fs as *mut c_void);

    if result < 0 {
        sofuu_promise_reject_str((*r).promise, uv_err_str(result as c_int).as_ptr());
        drop(Box::from_raw(r));
        return;
    }
    (*r).fd = result as c_int;

    let st = fs_req_alloc();
    uv::uv_fs_fstat(sofuu_loop_get(), st, (*r).fd as u64, None);
    let sz = uv::sofuu_uv_fs_stat_size(st);
    uv::uv_fs_req_cleanup(st);
    libc::free(st as *mut c_void);

    (*r).buf = vec![0u8; if sz > 0 { sz as usize } else { 1 }];

    let rr = fs_req_alloc();
    *(rr as *mut *mut RfbReq) = r;
    let buf = UvBuf {
        base: (*r).buf.as_mut_ptr() as *mut c_char,
        len: sz as usize,
    };
    uv::uv_fs_read(sofuu_loop_get(), rr, (*r).fd as u64, &buf, 1, 0, Some(rfb_read_cb));
}

unsafe extern "C" fn js_read_file_bytes(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"readFileBytes: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let path = CStr::from_ptr(path_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, path_ptr);
    let path = match CString::new(path) {
        Ok(p) => p,
        Err(_) => return qjs::sofuu_js_exception(),
    };

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    let r = Box::into_raw(Box::new(RfbReq {
        promise: out,
        fd: -1,
        buf: Vec::new(),
        path,
    }));
    let req = fs_req_alloc();
    *(req as *mut *mut RfbReq) = r;
    uv::uv_fs_open(sofuu_loop_get(), req, (*r).path.as_ptr(), libc::O_RDONLY, 0, Some(rfb_open_cb));

    promise
}

// ── Registration (C symbol replacement for mod_fs_register) ──────────

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_fs_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);

    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
    }

    let fs = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"readFile".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_read_file, c"readFile".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"writeFile".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_write_file, c"writeFile".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"appendFile".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_append_file, c"appendFile".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"exists".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_exists, c"exists".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"readdir".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_readdir, c"readdir".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"mkdir".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_mkdir, c"mkdir".as_ptr(), 2),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"rm".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_rm, c"rm".as_ptr(), 1),
    );
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"readFileBytes".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_read_file_bytes, c"readFileBytes".as_ptr(), 1),
    );

    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"fs".as_ptr(), fs);
    qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}
