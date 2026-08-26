// sofuu-ffi — libuv bindings (M0 keystone, PLAN-RUST-MIGRATION.md).
//
// Rust calls uv_* over FFI the same way the C glue did. Mirrors for the
// handle structs we touch (libuv structs are public; we only access the
// `data` field, which is first in every handle). All uv callbacks run on
// the loop thread — the `LoopPtr` wrapper is `!Send` to make that
// compile-time-enforced, matching `qjs::CtxPtr`.

use std::os::raw::{c_char, c_int, c_void};

pub const UV_RUN_DEFAULT: c_int = 0;
pub const UV_RUN_ONCE: c_int = 1;
pub const UV_RUN_NOWAIT: c_int = 2;

// ── uv error codes (uv/errno.h — POSIX mapping: -errno) ──────────────
pub const UV_EOF: c_int = -4095;
pub const UV_EPERM: c_int = -1;
pub const UV_ENOENT: c_int = -2;
pub const UV_EEXIST: c_int = -17;
pub const UV_EISDIR: c_int = -21;

/// `uv_dirent_t { const char *name; uv_dirent_type_t type; }`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UvDirent {
    pub name: *const c_char,
    pub d_type: c_int,
}

/// `uv_walk_cb` — visits every live handle in the loop (`uv_walk`).
pub type UvWalkCb = unsafe extern "C" fn(*mut UvHandle, *mut c_void);

pub type UvLoop = c_void;
pub type UvHandle = c_void;
pub type UvStream = c_void;
pub type UvTimer = c_void;
pub type UvCheck = c_void;
pub type UvAsync = c_void;
pub type UvFs = c_void;
pub type UvProcess = c_void;
pub type UvSignal = c_void;
pub type UvTty = c_void;
pub type UvPoll = c_void;
pub type UvTcp = c_void;

/// uv_connection_cb — (server_stream, status).
pub type UvConnectionCb = unsafe extern "C" fn(*mut UvStream, c_int);

/// uv_poll_cb — (handle, status, events) (curl socket bridge, M4).
pub type UvPollCb = unsafe extern "C" fn(*mut UvPoll, c_int, c_int);

pub const UV_DISCONNECT: c_int = 0;
pub const UV_READABLE: c_int = 1;
pub const UV_WRITABLE: c_int = 2;

/// Loop handle — `!Send`: libuv loops must run on their creating thread.
pub struct LoopPtr {
    ptr: *mut UvLoop,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl LoopPtr {
    /// # Safety
    /// `p` must be a valid, initialized uv_loop_t owned by this thread.
    pub unsafe fn new(p: *mut UvLoop) -> Self {
        Self { ptr: p, _not_send: std::marker::PhantomData }
    }

    pub fn as_ptr(&self) -> *mut UvLoop {
        self.ptr
    }
}

// ── extern decls (only the surface the runtime uses) ─────────────

extern "C" {
    pub fn uv_loop_init(loop_: *mut UvLoop) -> c_int;
    pub fn uv_loop_close(loop_: *mut UvLoop) -> c_int;
    pub fn uv_loop_alive(loop_: *const UvLoop) -> c_int;
    pub fn uv_run(loop_: *mut UvLoop, mode: c_int) -> c_int;
    pub fn uv_close(handle: *mut UvHandle, close_cb: Option<unsafe extern "C" fn(*mut UvHandle)>);
    pub fn uv_is_closing(handle: *const UvHandle) -> c_int;
    pub fn uv_ref(handle: *mut UvHandle);
    pub fn uv_unref(handle: *mut UvHandle);
    pub fn uv_walk(loop_: *mut UvLoop, walk_cb: Option<UvWalkCb>, arg: *mut c_void);

    // ── M10: real libuv accessors that replace the src/ffi_shim.c size and
    // field shims (the sofuu_uv_* functions below are plain Rust now) ──
    pub fn uv_loop_size() -> usize;
    pub fn uv_handle_size(handle_type: c_int) -> usize;
    pub fn uv_req_size(req_type: c_int) -> usize;
    pub fn uv_fs_get_result(req: *const UvFs) -> isize;
    pub fn uv_fs_get_statbuf(req: *mut UvFs) -> *mut UvStat;

    pub fn uv_timer_init(loop_: *mut UvLoop, timer: *mut UvTimer) -> c_int;
    pub fn uv_timer_start(
        timer: *mut UvTimer,
        cb: Option<unsafe extern "C" fn(*mut UvTimer)>,
        timeout: u64,
        repeat: u64,
    ) -> c_int;
    pub fn uv_timer_stop(timer: *mut UvTimer) -> c_int;

    pub fn uv_check_init(loop_: *mut UvLoop, check: *mut UvCheck) -> c_int;
    pub fn uv_check_start(check: *mut UvCheck, cb: Option<unsafe extern "C" fn(*mut UvCheck)>) -> c_int;
    pub fn uv_check_stop(check: *mut UvCheck) -> c_int;

    // ── host poke (PLAN-DESKTOP A): cross-thread wakeup of a blocked loop ──
    pub fn uv_async_init(
        loop_: *mut UvLoop,
        async_: *mut UvAsync,
        cb: Option<unsafe extern "C" fn(*mut UvAsync)>,
    ) -> c_int;
    pub fn uv_async_send(async_: *mut UvAsync) -> c_int;

    pub fn uv_fs_open(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        flags: c_int,
        mode: c_int,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_close(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        file: u64,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_read(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        file: u64,
        bufs: *const UvBuf,
        nbufs: usize,
        offset: i64,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_write(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        file: u64,
        bufs: *const UvBuf,
        nbufs: usize,
        offset: i64,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_unlink(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_mkdir(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        mode: c_int,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_rmdir(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_req_cleanup(req: *mut UvFs);
    pub fn uv_fs_stat(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_fstat(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        file: u64,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_scandir(
        loop_: *mut UvLoop,
        req: *mut UvFs,
        path: *const c_char,
        flags: c_int,
        cb: Option<unsafe extern "C" fn(*mut UvFs)>,
    ) -> c_int;
    pub fn uv_fs_scandir_next(req: *mut UvFs, ent: *mut UvDirent) -> c_int;

    pub fn uv_spawn(
        loop_: *mut UvLoop,
        process: *mut UvProcess,
        options: *const UvProcessOptions,
    ) -> c_int;

    pub fn uv_write(
        req: *mut UvWriteReq,
        stream: *mut UvStream,
        bufs: *const UvBuf,
        nbufs: usize,
        cb: Option<unsafe extern "C" fn(*mut UvWriteReq, c_int)>,
    ) -> c_int;
    pub fn uv_read_start(
        stream: *mut UvStream,
        alloc_cb: Option<unsafe extern "C" fn(*mut UvHandle, usize, *mut UvBuf)>,
        read_cb: Option<unsafe extern "C" fn(*mut UvStream, isize, *const UvBuf)>,
    ) -> c_int;
    pub fn uv_read_stop(stream: *mut UvStream) -> c_int;
    pub fn uv_pipe_init(loop_: *mut UvLoop, pipe: *mut UvPipe, ipc: c_int) -> c_int;
    pub fn uv_pipe_open(pipe: *mut UvPipe, fd: u64) -> c_int;

    pub fn uv_process_kill(process: *mut UvProcess, signum: c_int) -> c_int;
    pub fn uv_signal_init(loop_: *mut UvLoop, signal: *mut UvSignal) -> c_int;
    pub fn uv_signal_start(
        signal: *mut UvSignal,
        cb: Option<unsafe extern "C" fn(*mut UvSignal, c_int)>,
        signum: c_int,
    ) -> c_int;

    // ── M3: process module (TTY readline + cwd) ──────────────────
    pub fn uv_tty_init(loop_: *mut UvLoop, tty: *mut UvTty, fd: c_int, readable: c_int) -> c_int;
    pub fn uv_tty_set_mode(tty: *mut UvTty, mode: c_int) -> c_int;
    pub fn uv_tty_reset_mode() -> c_int;
    pub fn uv_cwd(buf: *mut c_char, size: *mut usize) -> c_int;
    pub fn uv_chdir(dir: *const c_char) -> c_int;

    // ── M4: HTTP client (curl ↔ libuv socket bridge) ─────────────
    pub fn uv_poll_init_socket(loop_: *mut UvLoop, poll: *mut UvPoll, socket: c_int) -> c_int;
    pub fn uv_poll_start(
        poll: *mut UvPoll,
        events: c_int,
        cb: Option<UvPollCb>,
    ) -> c_int;
    pub fn uv_poll_stop(poll: *mut UvPoll) -> c_int;

    // ── M5: HTTP server (uv_tcp) ────────────────────────────────
    pub fn uv_tcp_init(loop_: *mut UvLoop, tcp: *mut UvTcp) -> c_int;
    pub fn uv_tcp_bind(tcp: *mut UvTcp, addr: *const libc::sockaddr, flags: u32) -> c_int;
    pub fn uv_ip4_addr(ip: *const c_char, port: c_int, addr: *mut libc::sockaddr_in) -> c_int;
    pub fn uv_listen(stream: *mut UvStream, backlog: c_int, cb: Option<UvConnectionCb>) -> c_int;
    pub fn uv_accept(server: *mut UvStream, client: *mut UvStream) -> c_int;

    pub fn uv_strerror(err: c_int) -> *const c_char;
}

// ── M10: the deleted src/ffi_shim.c size/field shims, now plain Rust ──
// The opaque-to-Rust structs (uv_loop_t, uv_timer_t, uv_fs_t, …) are
// allocated at their true size through libuv's own public accessors
// (uv_loop_size / uv_handle_size / uv_req_size) and read through
// uv_fs_get_result / uv_fs_get_statbuf — no layout guessing, no C.

/// `uv_handle_type` values — enum order from UV_HANDLE_TYPE_MAP (uv.h).
const UV_HANDLE_ASYNC: c_int = 1;
const UV_HANDLE_CHECK: c_int = 2;
const UV_HANDLE_NAMED_PIPE: c_int = 7;
const UV_HANDLE_POLL: c_int = 8;
const UV_HANDLE_PROCESS: c_int = 10;
const UV_HANDLE_TCP: c_int = 12;
const UV_HANDLE_TIMER: c_int = 13;
const UV_HANDLE_TTY: c_int = 14;
const UV_HANDLE_SIGNAL: c_int = 16;

/// `UV_WRITE` / `UV_FS` (uv_req_type) — enum order from UV_REQ_TYPE_MAP.
const UV_REQ_WRITE: c_int = 3;
const UV_REQ_FS: c_int = 6;

/// `uv_stat_t` — the portable libuv stat struct (uv.h:376-401). st_size is
/// the 8th u64 field (offset 56) on every platform libuv normalizes into
/// this shape, so reading `.st_size` is portable.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UvStat {
    pub st_dev: u64,
    pub st_mode: u64,
    pub st_nlink: u64,
    pub st_uid: u64,
    pub st_gid: u64,
    pub st_rdev: u64,
    pub st_ino: u64,
    pub st_size: u64,
    pub st_blksize: u64,
    pub st_blocks: u64,
    pub st_flags: u64,
    pub st_gen: u64,
    pub st_atim: [i64; 2],
    pub st_mtim: [i64; 2],
    pub st_ctim: [i64; 2],
    pub st_birthtim: [i64; 2],
}

/// `sizeof(uv_loop_t)` — global loop storage (rt/loop.rs).
#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_loop_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_loop_size() }
}

/// `sizeof(uv_timer_t)` — M1 timer handles.
#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_timer_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_TIMER) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_fs_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_req_size(UV_REQ_FS) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_write_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_req_size(UV_REQ_WRITE) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_pipe_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_NAMED_PIPE) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_process_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_PROCESS) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_tty_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_TTY) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_signal_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_SIGNAL) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_poll_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_POLL) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_check_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_CHECK) }
}

/// `sizeof(uv_async_t)` — host poke handle (PLAN-DESKTOP A).
#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_async_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_ASYNC) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_tcp_size() -> usize {
    // SAFETY: trivial getter.
    unsafe { uv_handle_size(UV_HANDLE_TCP) }
}

/// `req->result` (ssize_t) — via libuv's public accessor.
#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_fs_result(req: *mut UvFs) -> isize {
    // SAFETY: req must be a live uv_fs_t.
    unsafe { uv_fs_get_result(req) }
}

/// `req->statbuf.st_size` — via uv_fs_get_statbuf + the portable uv_stat_t.
#[no_mangle]
pub unsafe extern "C" fn sofuu_uv_fs_stat_size(req: *mut UvFs) -> u64 {
    // SAFETY: req must be a live uv_fs_t (stat/fstat already completed).
    let st = unsafe { uv_fs_get_statbuf(req) };
    if st.is_null() {
        return 0;
    }
    // SAFETY: st is the live statbuf of req.
    unsafe { (*st).st_size }
}

pub const UV_TTY_MODE_NORMAL: c_int = 0;
pub const UV_TTY_MODE_RAW: c_int = 1;
pub const UV_TTY_MODE_IO: c_int = 2;

// ── struct mirrors (fields we touch) ──────────────────────────────

/// `uv_buf_t` — { base, len }.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UvBuf {
    pub base: *mut c_char,
    pub len: usize,
}

impl UvBuf {
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            base: data.as_ptr() as *mut c_char,
            len: data.len(),
        }
    }
}

/// `uv_write_t` — libuv write request; `data` is the first field.
#[repr(C)]
pub struct UvWriteReq {
    pub data: *mut c_void,
    // (the rest of uv_write_t is internal — we never touch it)
}

/// `uv_pipe_t` — the first field is `data` (uv_handle_t starts with data).
#[repr(C)]
pub struct UvPipe {
    pub data: *mut c_void,
}

/// `uv_process_options_t` — stdio_count + stdio array + exit_cb.
/// We mirror the fields the spawn path sets.
#[repr(C)]
pub struct UvProcessOptions {
    pub exit_cb: Option<unsafe extern "C" fn(*mut UvProcess, i64, c_int)>,
    pub file: *const c_char,
    pub args: *mut *mut c_char,
    pub env: *mut *mut c_char,
    pub cwd: *const c_char,
    pub flags: u32,
    pub stdio_count: c_int,
    pub stdio: *mut UvStdioContainer,
    pub uid: u32,
    pub gid: u32,
}

/// `uv_stdio_container_t.data` — the union { uv_stream_t *stream; int fd; }.
#[repr(C)]
#[derive(Clone, Copy)]
pub union UvStdioData {
    pub stream: *mut c_void,
    pub fd: c_int,
}

/// `uv_stdio_container_t` — { int flags; union { stream | fd } data; }.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UvStdioContainer {
    pub flags: c_int,
    pub data: UvStdioData,
}

pub const UV_IGNORE: c_int = 0x00;
pub const UV_CREATE_PIPE: c_int = 0x01;
pub const UV_WRITABLE_PIPE: c_int = 0x20; // uv.h: WRITABLE=0x20, READABLE=0x10
pub const UV_READABLE_PIPE: c_int = 0x10;

// `uv_poll_t`/`uv_tcp_t`/`uv_tty_t` — same leading `data` field pattern;
// the runtime mostly uses pipe/timer/fs/process/signal/check, so those are
// the ones mirrored above. (tcp/tty/poll can be added when M5 needs them.)


#[cfg(test)]
mod spawn_tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::ptr;

    #[repr(C)]
    struct Mini {
        out: *mut UvPipe,
        proc: *mut UvProcess,
        got: [u8; 64],
        got_len: usize,
        done: std::sync::atomic::AtomicI32,
    }

    unsafe extern "C" fn mini_alloc(_h: *mut UvHandle, sz: usize, b: *mut UvBuf) {
        (*b).base = libc::malloc(sz) as *mut c_char;
        (*b).len = sz;
    }

    unsafe extern "C" fn mini_read(s: *mut UvStream, nread: isize, b: *const UvBuf) {
        let m = *(s as *mut *mut Mini);
        if nread > 0 {
            let c = std::slice::from_raw_parts((*b).base as *const u8, nread as usize);
            (&mut (*m).got)[(*m).got_len..(*m).got_len + c.len()].copy_from_slice(c);
            (*m).got_len += c.len();
        }
        if !(*b).base.is_null() {
            libc::free((*b).base as *mut c_void);
        }
        if nread < 0 {
            (*m).done.store(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn mini_exit(p: *mut UvProcess, _s: i64, _t: c_int) {
        let m = *(p as *mut *mut Mini);
        (*m).done.store(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn m6_mini_spawn_reads_child_stdout() {
        // SAFETY: standalone loop (test-scoped, like the M1/M2 tests).
        unsafe {
        let loop_storage = libc::malloc(sofuu_uv_loop_size()) as *mut UvLoop;
        uv_loop_init(loop_storage);
        let lp = unsafe { LoopPtr::new(loop_storage) };

        let out = libc::malloc(sofuu_uv_pipe_size()) as *mut UvPipe;
        let proc = libc::malloc(sofuu_uv_process_size()) as *mut UvProcess;
        let init_rc = unsafe { uv_pipe_init(lp.as_ptr(), out, 0) };
        let t0 = unsafe { *(out.add(16 / 8) as *mut c_int) };
        eprintln!("[mini] pipe_init_rc={} type_after_init={}", init_rc, t0);

        let mut m = Box::new(Mini {
            out,
            proc,
            got: [0u8; 64],
            got_len: 0,
            done: std::sync::atomic::AtomicI32::new(0),
        });
        unsafe { *(out as *mut *mut Mini) = &mut *m };

        let file = CString::new("sh").unwrap();
        let arg1 = CString::new("-c").unwrap();
        let arg2 = CString::new("echo MINI-OK").unwrap();
        let mut args = [file.as_ptr() as *mut c_char, arg1.as_ptr() as *mut c_char, arg2.as_ptr() as *mut c_char, ptr::null_mut()];

        let mut stdio: [UvStdioContainer; 3] = [
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
            UvStdioContainer { flags: UV_CREATE_PIPE | UV_WRITABLE_PIPE, data: UvStdioData { stream: out as *mut c_void } },
            UvStdioContainer { flags: 0, data: UvStdioData { stream: ptr::null_mut() } },
        ];
        let opts = UvProcessOptions {
            exit_cb: Some(mini_exit),
            file: file.as_ptr(),
            args: args.as_mut_ptr(),
            env: ptr::null_mut(),
            cwd: ptr::null(),
            flags: 0,
            stdio_count: 3,
            stdio: stdio.as_mut_ptr(),
            uid: 0,
            gid: 0,
        };
        let m_ptr = Box::into_raw(m);
        unsafe { *(out as *mut *mut Mini) = m_ptr };
        let spawn_rc = unsafe { uv_spawn(lp.as_ptr(), proc, &opts) };
        // PROBE: uv_handle_t { data(0), loop_(8), type(16), flags(20) }
        let typev = unsafe { *(out.add(16 / 8) as *mut c_int) };
        let flags = unsafe { *(out.add(20 / 8) as *mut c_int) };
        eprintln!("[mini] spawn={} type={} flags=0x{:x}", spawn_rc, typev, flags);
        assert_eq!(spawn_rc, 0, "spawn");
        unsafe { *(proc as *mut *mut Mini) = m_ptr };
        let rs = unsafe { uv_read_start(out as *mut UvStream, Some(mini_alloc), Some(mini_read)) };
        eprintln!("[mini] spawn={} read_start={}", spawn_rc, rs);
        assert_eq!(rs, 0, "read_start must not be ENOTCONN");

        while unsafe { (*m_ptr).done.load(std::sync::atomic::Ordering::SeqCst) } == 0 {
            unsafe { uv_run(lp.as_ptr(), UV_RUN_ONCE) };
        }
        unsafe {
            uv_read_stop(out as *mut UvStream);
            uv_close(out as *mut UvHandle, None);
            uv_loop_close(lp.as_ptr());
        }
        let got = CStr::from_bytes_with_nul_unchecked(&(&(*m_ptr).got)[..(*m_ptr).got_len + 1])
            .to_string_lossy()
            .into_owned();
        eprintln!("[mini] got={:?}", got);
        assert!(got.contains("MINI-OK"), "child stdout must arrive: {got}");
        drop(Box::from_raw(m_ptr));
        libc::free(lp.as_ptr() as *mut c_void);
        }
    }
}
