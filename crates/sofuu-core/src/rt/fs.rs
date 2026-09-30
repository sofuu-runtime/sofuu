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

/// proc-4: readFile refuses to buffer more than 256MB (the JS string for it
/// would need the same again; the 2^30−1 char cap would reject far above
/// this anyway). Oversize stat claims reject up front.
const READ_FILE_MAX: u64 = 1 << 28;
/// proc-4: when fstat's size is 0 or a lie (FIFOs, procfs-style nodes), the
/// read grows the buffer in chunks of this size until EOF instead of
/// trusting the stat.
const READ_CHUNK: usize = 64 * 1024;

struct ReadReq {
    promise: *mut PromiseHandle, /* NULL after resolve/reject */
    fd: c_int,
    buf: Vec<u8>,
    path: CString,
    /* proc-4: bytes read so far + the stat's size claim (0 when unknown) —
     * the chunk loop reads until EOF, never trusting st_size alone. */
    offset: usize,
    fsize: u64,
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
        (*r).promise = ptr::null_mut();
        /* Close the fd */
        let cr = fs_req_alloc();
        *(cr as *mut *mut ReadReq) = r;
        uv::uv_fs_close(sofuu_loop_get(), cr, (*r).fd as u64, Some(rf_close_cb));
        return;
    }

    /* proc-4: chunk loop — accumulate until EOF (n == 0) or the stat size
     * is consumed. The buffer already has room for one more chunk; grow it
     * only when the next read would overflow it. */
    let r_ref = &mut *r;
    r_ref.offset += n as usize;
    let more_needed = if r_ref.fsize > 0 {
        (r_ref.offset as u64) < r_ref.fsize && n > 0
    } else {
        n > 0
    };

    if more_needed {
        if r_ref.offset + READ_CHUNK > r_ref.buf.len() {
            r_ref.buf.resize(r_ref.offset + READ_CHUNK, 0);
        }
        let rr = fs_req_alloc();
        *(rr as *mut *mut ReadReq) = r;
        let buf = UvBuf {
            base: r_ref.buf.as_mut_ptr().add(r_ref.offset) as *mut c_char,
            len: r_ref.buf.len() - r_ref.offset,
        };
        /* offset −1 = "use the fd's file position" (pread would fail with
         * ESPIPE on FIFOs); uv advances the position per read. */
        uv::uv_fs_read(sofuu_loop_get(), rr, (*r).fd as u64, &buf, 1, -1, Some(rf_read_cb));
        return;
    }

    let len = r_ref.offset;
    let str_v = qjs::JS_NewStringLen(ctx, (*r).buf.as_ptr() as *const c_char, len);
    if qjs::is_exception(str_v) {
        /* P1-11: a failed build (OOM, or a file beyond the 2^30−1 char
         * cap; invalid UTF-8 does NOT take this path — QuickJS maps it
         * to U+FFFD) leaves an exception-tagged value + a pending
         * exception. Resolving the promise with the tag is UB — reject
         * the readFile promise instead. */
        let exc = qjs::sofuu_js_get_exception(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        sofuu_promise_reject_str(
            (*r).promise,
            c"file could not be converted to a string (out of memory)".as_ptr(),
        );
    } else {
        sofuu_promise_resolve((*r).promise, str_v);
    }
    qjs::sofuu_js_free_value(ctx, str_v);
    sofuu_flush_jobs(ctx);
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

    /* proc-4: refuse to buffer a stat claim beyond the read cap. */
    if sz > READ_FILE_MAX {
        sofuu_promise_reject_str(
            (*r).promise,
            c"readFile: file is over the 256MB read cap".as_ptr(),
        );
        (*r).promise = ptr::null_mut();
        let cr = fs_req_alloc();
        *(cr as *mut *mut ReadReq) = r;
        uv::uv_fs_close(sofuu_loop_get(), cr, (*r).fd as u64, Some(rf_close_cb));
        return;
    }

    /* proc-4: zero/unknown stat size (FIFOs, procfs nodes) gets one chunk
     * and the EOF loop decides the real length; honest sizes get their
     * exact buffer. Either way the first read asks for a full chunk's
     * window so short reads loop instead of truncating. */
    let first_len = if sz > 0 { sz as usize } else { READ_CHUNK };
    (*r).fsize = sz;
    (*r).offset = 0;
    (*r).buf = vec![0u8; first_len + 1];

    let rr = fs_req_alloc();
    *(rr as *mut *mut ReadReq) = r;
    let buf = UvBuf {
        base: (*r).buf.as_mut_ptr() as *mut c_char,
        len: first_len,
    };
    /* offset −1 = streaming (fd position), not pread — FIFOs fail pread. */
    uv::uv_fs_read(sofuu_loop_get(), rr, (*r).fd as u64, &buf, 1, -1, Some(rf_read_cb));
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
        offset: 0,
        fsize: 0,
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

    /* proc-5: length-preserving extraction — the C-string round trip
     * truncated content at the first NUL (writeFile('a\0b') wrote "a").
     * JS_ToCStringLen2 reports the real UTF-8 byte length; embedded NULs
     * inside the string survive the copy. */
    let mut data_len: usize = 0;
    let data_ptr = qjs::JS_ToCStringLen2(ctx, &mut data_len, data_val, 0);
    if data_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    // SAFETY: data_ptr points at data_len bytes from QuickJS's allocator.
    let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) }.to_vec();
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
    /* P1-12 (AUDIT-2026-09-07): the write jail checks the realpath in
     * tools.js (mustBeInRoot), then hands the LEXICAL path here — an
     * attacker-writable directory could swap the checked path for a symlink
     * between check and open, and this open followed it, writing outside
     * the project. O_NOFOLLOW closes the window for the final component:
     * a symlink there fails the open with ELOOP instead of being followed.
     * In-project symlink targets stay writable because mustBeInRoot now
     * hands writeFile the canonical path it already vetted. Windows has no
     * O_NOFOLLOW equivalent through these CRT flags. */
    #[cfg(unix)]
    let flags = flags | libc::O_NOFOLLOW;
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
        // SAFETY: uv_fs_read wrote exactly n bytes into the spare capacity
        // reserved in rfb_open_cb; the Vec's length now covers only those
        // bytes, so no uninitialized byte is ever observable.
        unsafe { (*r).buf.set_len(n as usize) };
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

    // P3 (AUDIT-2026-09-07) pf-2: the buffer is reserved WITHOUT zeroing —
    // the old `vec![0u8; sz]` memset the whole file before uv_fs_read
    // overwrote it (2× memory traffic). The Vec stays len 0 (clippy::uninit_vec
    // clean) until the read lands: rfb_read_cb set_len's exactly the n bytes
    // uv wrote, and an error path drops the Vec without ever observing the
    // capacity. SAFETY: uv writes only into the spare capacity reserved here.
    let cap = if sz > 0 { sz as usize } else { 1 };
    let mut b: Vec<u8> = Vec::with_capacity(cap);
    (*r).buf = b;

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

// ── realpath — canonicalize via std::fs (SYMLINK-AWARE jail primitive) ─
// P1-4 (AUDIT-2026-09-01): the write jail used to be purely lexical, so a
// symlink inside the project pointing outside it passed the prefix check
// and write_file/edit_file could write through the link. Callers
// (tools.js mustBeInRoot) now resolve the realpath of BOTH the target and
// the project root before the prefix comparison — the same check
// std::fs::canonicalize makes resistant to symlinks, `..`, and case
// tricks. Resolved promise: string canonical path, or reject on missing
// path / failure (a jail check on a nonexistent target falls back to the
// lexical result in the JS caller).

unsafe extern "C" fn js_realpath(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"realpath: path required".as_ptr());
    }
    let path_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if path_ptr.is_null() {
        return qjs::sofuu_js_exception();
    }
    let path_str = CStr::from_ptr(path_ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, path_ptr);

    let mut out: *mut PromiseHandle = ptr::null_mut();
    let promise = sofuu_promise_new(ctx, &mut out);

    match std::fs::canonicalize(&path_str) {
        Ok(canonical) => {
            let s = canonical.to_string_lossy().into_owned();
            let c = CString::new(s).unwrap_or_default();
            sofuu_promise_resolve(out, qjs::sofuu_js_new_string(ctx, c.as_ptr()));
        }
        Err(e) => {
            let msg = format!("realpath: {e}");
            let c = CString::new(msg).unwrap_or_default();
            sofuu_promise_reject_str(out, c.as_ptr());
        }
    }
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
    qjs::sofuu_js_set_property_str(
        ctx,
        fs,
        c"realpath".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_realpath, c"realpath".as_ptr(), 1),
    );

    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"fs".as_ptr(), fs);
    qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ffi::qjs::{self, CtxPtr};
    use std::sync::atomic::{AtomicU32, Ordering};

    fn read_str_global(ctx: *mut JSContext, global: JSValue, name: &CStr) -> String {
        // SAFETY: global is a live object; the returned value is freed here.
        let v = unsafe { qjs::sofuu_js_get_property_str(ctx, global, name.as_ptr()) };
        let s = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
        let out = if s.is_null() {
            String::new()
        } else {
            // SAFETY: s is a live C string from QuickJS, freed below.
            unsafe { CStr::from_ptr(s).to_string_lossy().into_owned() }
        };
        unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        unsafe { qjs::sofuu_js_free_value(ctx, v) };
        out
    }

    /// Audit P1-12 regression: a symlink swapped into the FINAL component of
    /// a write path must not be followed. The old open used O_WRONLY|O_CREAT
    /// (no O_NOFOLLOW), so `writeFile('<proj>/link', …)` — with link →
    /// `<outside>/victim.txt` — happily wrote through the link and escaped
    /// the project root. With the fix, the open fails with ELOOP and the
    /// promise rejects; the victim file is untouched. The positive control
    /// proves ordinary in-project writes still work under O_NOFOLLOW.
    /// (The tools.js half of P1-12 — mustBeInRoot returning the canonical
    /// path — is exercised by tests/agent_test.js via ./sofuu.)
    #[cfg(unix)]
    #[test]
    fn write_rejects_symlinked_final_component() {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir()
            .join(format!("sofuu_p112_{}_{}", std::process::id(), seq));
        let proj = tmp.join("proj");
        let victim = tmp.join("victim.txt");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(&victim, b"safe").unwrap();
        std::os::unix::fs::symlink(&victim, proj.join("link")).unwrap();

        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let script = CString::new(format!(
            "var rejected = 'no-reject', resolved = false, posOk = false, posErr = ''; \
sofuu.fs.writeFile('{}/link', 'pwned').then(function () {{ resolved = true; }}, \
function (e) {{ rejected = (e && e.message) ? e.message : String(e); }}); \
sofuu.fs.writeFile('{}/real.txt', 'hello').then(function () {{ posOk = true; }}, \
function (e) {{ posErr = (e && e.message) ? e.message : String(e); }});",
            proj.to_string_lossy(),
            proj.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-p112>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "fs setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: both opens resolve/reject and their handlers run.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let rejected = read_str_global(ctx, global, c"rejected");
        let pos_ok = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"posOk".as_ptr()) };
        let pos_ok_b = unsafe { qjs::JS_ToBool(ctx, pos_ok) };
        unsafe { qjs::sofuu_js_free_value(ctx, pos_ok) };
        let pos_err = read_str_global(ctx, global, c"posErr");

        assert_ne!(rejected, "no-reject", "symlink write must be rejected");
        assert!(
            rejected.contains("symbolic"),
            "ELOOP rejection must name the open failure, got: {rejected}"
        );
        assert_eq!(
            std::fs::read_to_string(&victim).unwrap(),
            "safe",
            "victim file outside the project must be untouched"
        );
        assert!(pos_ok_b == 1, "ordinary in-project write must still succeed");
        assert_eq!(pos_err, "", "positive control must not reject: {pos_err}");
        assert_eq!(
            std::fs::read_to_string(proj.join("real.txt")).unwrap(),
            "hello",
            "positive-control payload must be written verbatim"
        );
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // SAFETY: teardown AFTER the loop is closed (M0 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// P3 (AUDIT-2026-09-07) pf-2 companion: readFileBytes reserves its
    /// buffer WITHOUT zeroing (capacity only, `set_len(n)` after the read —
    /// the clippy::uninit_vec-clean discipline). This pins the observable
    /// contract end to end: every byte of a 0..=255 payload (NULs and high
    /// bytes included) must come back exactly, and an empty file must give
    /// a zero-length Uint8Array, never garbage or a wrong length.
    #[test]
    fn read_file_bytes_delivers_exact_bytes_from_uninit_capacity() {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir()
            .join(format!("sofuu_rfb_{}_{}", std::process::id(), seq));
        std::fs::create_dir_all(&tmp).unwrap();
        let payload: Vec<u8> = (0..=255u8).collect();
        std::fs::write(tmp.join("bin.dat"), &payload).unwrap();
        std::fs::write(tmp.join("empty.bin"), b"").unwrap();

        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let script = CString::new(format!(
            "var got = 'unset', gotEmpty = 'unset'; \
sofuu.fs.readFileBytes('{}/bin.dat').then(function (b) {{ \
var s = ''; for (var i = 0; i < b.length; i++) s += b[i] + ','; \
got = s + 'len=' + b.length; }}, \
function (e) {{ got = 'ERR:' + ((e && e.message) ? e.message : String(e)); }}); \
sofuu.fs.readFileBytes('{}/empty.bin').then(function (b) {{ \
gotEmpty = 'len=' + b.length; }}, \
function (e) {{ gotEmpty = 'ERR:' + ((e && e.message) ? e.message : String(e)); }});",
            tmp.to_string_lossy(),
            tmp.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-rfb>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(r) }, "readFileBytes eval must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: both reads resolve and their handlers run.
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let got = read_str_global(ctx, global, c"got");
        let got_empty = read_str_global(ctx, global, c"gotEmpty");
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        let mut expected = String::new();
        for i in 0..=255u8 {
            expected.push_str(&i.to_string());
            expected.push(',');
        }
        expected.push_str("len=256");
        assert_eq!(got, expected, "256-byte payload must round-trip byte-exact");
        assert_eq!(got_empty, "len=0", "empty file must give a zero-length view");
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Audit P1-11 regression: when JS_NewStringLen fails in rf_read_cb the
    /// returned JSValue is exception-tagged; resolving the readFile promise
    /// with it is UB. Deterministic trigger: JS_SetMemoryLimit caps the JS
    /// heap (8 MB) AFTER the readFile calls are issued but BEFORE the read
    /// completes, so the 32 MB ASCII file's string build OOMs while the
    /// tiny handlers still run. (The audit's original trigger claim — any
    /// invalid-UTF-8/binary file — is wrong: QuickJS maps invalid bytes to
    /// U+FFFD; only OOM and the 2^30−1 char cap throw.) With the fix the
    /// big read rejects cleanly and the runtime stays usable; the positive
    /// control proves ordinary reads under the same limit still resolve.
    #[test]
    fn read_rejects_when_string_build_ooms() {
        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let tmp = std::env::temp_dir().join(format!("sofuu_p111_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let big = tmp.join("big.txt");
        let small = tmp.join("small.txt");
        std::fs::write(&big, vec![b'a'; 32 * 1024 * 1024]).unwrap();
        std::fs::write(&small, b"ok").unwrap();

        let script = CString::new(format!(
            "var rejected = 'no-reject', resolved = false, posVal = '', posErr = ''; \
sofuu.fs.readFile('{}').then(function (v) {{ resolved = true; }}, \
function (e) {{ rejected = (e && e.message) ? e.message : String(e); }}); \
sofuu.fs.readFile('{}').then(function (v) {{ posVal = v; }}, \
function (e) {{ posErr = (e && e.message) ? e.message : String(e); }});",
            big.to_string_lossy(),
            small.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-p111>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "fs setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // P1-11 fault injection: cap the JS heap between issuing the reads
        // and draining them. The 32 MB string build now fails; libc/libuv
        // buffers (the Rust Vec holding file bytes) are not JS allocations
        // and are unaffected.
        // SAFETY: rt is live on this thread.
        unsafe { qjs::JS_SetMemoryLimit(rt, 8 * 1024 * 1024) };

        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let rejected = read_str_global(ctx, global, c"rejected");
        let resolved = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"resolved".as_ptr()) };
        let resolved_b = unsafe { qjs::JS_ToBool(ctx, resolved) };
        unsafe { qjs::sofuu_js_free_value(ctx, resolved) };
        let pos_val = read_str_global(ctx, global, c"posVal");
        let pos_err = read_str_global(ctx, global, c"posErr");

        assert!(
            rejected.contains("out of memory"),
            "OOM string build must reject with the out-of-memory error, got: {rejected}"
        );
        assert!(resolved_b == 0, "the failed read must NOT resolve");
        assert_eq!(pos_val, "ok", "small read under the same limit must still resolve");
        assert_eq!(pos_err, "", "positive control must not reject: {pos_err}");
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // SAFETY: teardown AFTER the loop is closed (M0 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Audit proc-4 regression: the read was sized from fstat and ran ONCE —
    /// any file whose stat size is 0 (FIFOs, procfs-style nodes) resolved as
    /// EMPTY because st_size said 0 bytes even though content arrives. The
    /// read is now a chunk loop until EOF. A FIFO with a writer proves the
    /// content path; the positive control proves ordinary files still read.
    #[cfg(unix)]
    #[test]
    fn read_file_reads_fifo_to_eof() {
        use std::io::Write;
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("sofuu_proc4a_{}_{}", std::process::id(), seq));
        std::fs::create_dir_all(&tmp).unwrap();
        let fifo = tmp.join("pipe.fifo");
        let cpath = CString::new(fifo.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) }, 0, "mkfifo must succeed");

        // The writer blocks in open() until the reader side opens — start
        // it now so the readFile below can't deadlock alone.
        let wpath = fifo.clone();
        let writer = std::thread::spawn(move || {
            let mut f = std::fs::File::create(&wpath).unwrap();
            let _ = f.write_all(b"hello-fifo");
        });

        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let script = CString::new(format!(
            "var got = 'unset', err = ''; \
sofuu.fs.readFile('{}').then(function (v) {{ got = v; }}, \
function (e) {{ err = (e && e.message) ? e.message : String(e); }});",
            fifo.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-proc4a>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "fs setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // Drain: open unblocks, the chunk loop reads to EOF, promise
        // resolves (pre-fix: one 0-sized read → resolves "").
        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };
        writer.join().unwrap();

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let got = read_str_global(ctx, global, c"got");
        let err = read_str_global(ctx, global, c"err");
        assert_eq!(err, "", "the FIFO read must not reject: {err}");
        assert_eq!(got, "hello-fifo", "zero-stat files must still deliver content");

        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        // SAFETY: teardown AFTER the loop is closed (M0 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Audit proc-4 regression: the allocation was stat_size+1 bytes with no
    /// cap — a stat size beyond the string cap was a straight OOM abort
    /// path. A sparse file larger than the 256MB read cap must reject
    /// cleanly instead.
    #[cfg(unix)]
    #[test]
    fn read_file_rejects_oversize_stat() {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("sofuu_proc4b_{}_{}", std::process::id(), seq));
        std::fs::create_dir_all(&tmp).unwrap();
        let big = tmp.join("sparse.bin");
        // Sparse: set_len costs no real disk on APFS/ext4.
        let f = std::fs::File::create(&big).unwrap();
        f.set_len((1u64 << 28) + 1).unwrap(); // 256MB + 1 — over the cap
        drop(f);

        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let script = CString::new(format!(
            "var rejected = 'no-reject', resolved = false; \
sofuu.fs.readFile('{}').then(function (v) {{ resolved = true; }}, \
function (e) {{ rejected = (e && e.message) ? e.message : String(e); }});",
            big.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-proc4b>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "fs setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let rejected = read_str_global(ctx, global, c"rejected");
        let resolved = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"resolved".as_ptr()) };
        let resolved_b = unsafe { qjs::JS_ToBool(ctx, resolved) };
        unsafe { qjs::sofuu_js_free_value(ctx, resolved) };
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        assert!(
            rejected.contains("cap"),
            "an over-cap stat size must reject naming the cap, got: {rejected}"
        );
        assert!(resolved_b == 0, "the over-cap read must NOT resolve");

        // SAFETY: teardown AFTER the loop is closed (M0 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Audit proc-5 regression: write_impl extracted content through a C
    /// string (CStr::to_bytes) — writeFile('a\u0000b') truncated at the NUL
    /// and wrote "a". Content must be extracted length-exact and land on
    /// disk byte-exact.
    #[cfg(unix)]
    #[test]
    fn write_file_writes_embedded_nul() {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("sofuu_proc5_{}_{}", std::process::id(), seq));
        std::fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("nul.bin");

        let _loop_guard = crate::rt::test_loop_lock();
        // SAFETY: standalone runtime + context (M0 pattern).
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let _ctp = unsafe { CtxPtr::new(ctx) };
        unsafe { crate::rt::event_loop::sofuu_loop_init() };
        // SAFETY: registers sofuu.fs.
        unsafe { mod_fs_register(ctx) };

        let script = CString::new(format!(
            "var err = ''; \
sofuu.fs.writeFile('{}', 'a\\u0000b').then(function () {{}}, \
function (e) {{ err = (e && e.message) ? e.message : String(e); }});",
            target.to_string_lossy(),
        ))
        .unwrap();
        let r = unsafe {
            qjs::JS_Eval(
                ctx,
                script.as_ptr(),
                script.to_bytes().len(),
                c"<fs-proc5>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        if unsafe { qjs::is_exception(r) } {
            unsafe { qjs::js_std_dump_error(ctx) };
        }
        assert!(!unsafe { qjs::is_exception(r) }, "fs setup must not throw");
        unsafe { qjs::sofuu_js_free_value(ctx, r) };

        // SAFETY: ctx live on this thread.
        unsafe { crate::rt::event_loop::sofuu_loop_run_bounded(ctx, Some(std::time::Duration::from_secs(60))) };

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
        let err = read_str_global(ctx, global, c"err");
        unsafe { qjs::sofuu_js_free_value(ctx, global) };
        assert_eq!(err, "", "the NUL write must not reject: {err}");
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"a\0b",
            "embedded NUL must reach disk byte-exact (pre-fix: 1 byte \"a\")"
        );

        // SAFETY: teardown AFTER the loop is closed (M0 discipline).
        unsafe { crate::rt::event_loop::sofuu_loop_close() };
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
